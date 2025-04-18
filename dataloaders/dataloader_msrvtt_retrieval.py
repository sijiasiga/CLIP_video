from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import random
from dataloaders.rawvideo_util import RawVideoExtractor
import torch

class MSRVTT_DataLoader(Dataset):
    """MSRVTT dataset loader."""
    
    def scene_change_based_sampling(self, raw_video_slice, max_frames):
        """Scene-Change Based Sampling - selects frames at scene transitions
        
        1. Calculates frame differences to detect scene changes
        2. Samples frames after major scene transitions
        3. Fills remaining frames with uniform sampling for coverage
        """
        # Convert to numpy for easier manipulation if it's a torch tensor
        if isinstance(raw_video_slice, torch.Tensor):
            raw_video_np = raw_video_slice.cpu().numpy()
        else:
            raw_video_np = raw_video_slice
            
        # Calculate frame differences to detect scene changes
        if len(raw_video_np) <= 1:
            return raw_video_slice
            
        frame_diffs = []
        for i in range(1, len(raw_video_np)):
            # Calculate mean absolute difference between consecutive frames
            diff = np.mean(np.abs(
                raw_video_np[i, 0, 0].astype(float) - raw_video_np[i-1, 0, 0].astype(float)
            ))
            frame_diffs.append(diff)
            
        # Detect scene changes (frames with difference higher than threshold)
        frame_diffs = np.array(frame_diffs)
        mean_diff = np.mean(frame_diffs)
        std_diff = np.std(frame_diffs)
        # Define threshold as mean + 1.5*std (adaptive to video content)
        threshold = mean_diff + 1.5 * std_diff
        
        # Find frames that exceed the threshold (potential scene changes)
        # Add 1 because frame_diffs indices are offset by 1
        scene_change_indices = np.where(frame_diffs > threshold)[0] + 1
        
        # Always include first frame
        if len(scene_change_indices) == 0 or scene_change_indices[0] != 0:
            scene_indices = np.concatenate(([0], scene_change_indices))
        else:
            scene_indices = scene_change_indices
            
        # If we have more scene changes than max_frames, select the most significant ones
        if len(scene_indices) > max_frames:
            # Sort scene changes by magnitude of difference
            sorted_indices = np.argsort(frame_diffs[scene_indices-1])[::-1]
            scene_indices = scene_indices[sorted_indices[:max_frames]]
            scene_indices = np.sort(scene_indices)
        
        # If we have fewer scene changes than max_frames, fill with uniform sampling
        if len(scene_indices) < max_frames:
            remaining_frames = max_frames - len(scene_indices)
            # Add uniformly sampled frames for coverage
            uniform_indices = np.linspace(0, len(raw_video_np) - 1, num=remaining_frames + 2, dtype=int)
            # Remove first and last frame which might duplicate scene change frames
            uniform_indices = uniform_indices[1:-1]
            
            # Combine scene change frames with uniform frames
            all_indices = np.concatenate([scene_indices, uniform_indices])
            all_indices = np.sort(np.unique(all_indices))[:max_frames]
        else:
            all_indices = scene_indices[:max_frames]
            
        return raw_video_slice[all_indices]
    
    def motion_guided_adaptive_sampling(self, raw_video_slice, max_frames):
        """Motion-Guided Adaptive Sampling - novel frame selection method
        
        Allocates frames based on motion intensity:
        1. Calculates frame-to-frame differences to detect motion
        2. Ensures global context with minimum uniform sampling
        3. Allocates remaining frames to high-motion segments
        """
        # Convert to numpy for easier manipulation if it's a torch tensor
        if isinstance(raw_video_slice, torch.Tensor):
            raw_video_np = raw_video_slice.cpu().numpy()
        else:
            raw_video_np = raw_video_slice
            
        # 1. Calculate inter-frame motion (simplified for implementation)
        frame_diffs = []
        for i in range(1, len(raw_video_np)):
            # Calculate mean squared difference between consecutive frames
            # Focus on visual content (channel 0) for efficiency
            diff = np.mean(np.square(
                raw_video_np[i, 0, 0].astype(float) - raw_video_np[i-1, 0, 0].astype(float)
            ))
            frame_diffs.append(diff)
        
        # Normalize motion scores
        if len(frame_diffs) > 0:
            motion_scores = np.array(frame_diffs) / (np.max(frame_diffs) + 1e-5)
        else:
            # Fallback for very short videos
            return raw_video_slice
        
        # 2. Ensure global context with minimum sampling
        global_context_frames = max(max_frames // 4, 1)  # 25% for global context
        global_indices = np.linspace(0, len(raw_video_np) - 1, num=global_context_frames, dtype=int)
        
        # 3. Allocate remaining frames to high-motion areas
        remaining_frames = max_frames - global_context_frames
        
        if remaining_frames <= 0 or len(motion_scores) == 0:
            # If max_frames is very small or no motion data, just use global sampling
            all_indices = global_indices
        else:
            # Create probability distribution favoring high-motion frames
            # Add small constant to avoid zero probabilities
            probs = motion_scores + 0.1
            probs = probs / np.sum(probs)
            
            # Sample frame indices based on motion probability
            # +1 because motion_scores starts from second frame
            motion_indices = np.random.choice(
                np.arange(1, len(raw_video_np)),
                size=min(remaining_frames, len(raw_video_np)-1),
                replace=False,
                p=probs
            )
            
            # 4. Combine and sort frames by their original temporal position
            all_indices = np.concatenate([global_indices, motion_indices])
            all_indices = np.sort(np.unique(all_indices))[:max_frames]  # Ensure no duplicates
        
        return raw_video_slice[all_indices]
    
    def diversity_based_keyframe_selection(self, raw_video_slice, max_frames):
        """Diversity-Based Keyframe Selection - selects the most diverse frames
        
        1. Starts with the first frame
        2. Iteratively selects frames that are most different from already selected ones
        3. This ensures maximum visual diversity in the selected frames
        """
        # Convert to numpy for easier manipulation if it's a torch tensor
        if isinstance(raw_video_slice, torch.Tensor):
            raw_video_np = raw_video_slice.cpu().numpy()
        else:
            raw_video_np = raw_video_slice
            
        if len(raw_video_np) <= max_frames:
            return raw_video_slice
            
        # Flatten frames for easier distance calculation (using first channel)
        flattened_frames = raw_video_np[:, 0, 0].reshape(len(raw_video_np), -1)
        
        # Initialize with the first frame
        selected_indices = [0]
        remaining_indices = list(range(1, len(raw_video_np)))
        
        # Iteratively select the most diverse frames
        while len(selected_indices) < max_frames and remaining_indices:
            max_min_distance = -1
            next_frame_idx = -1
            
            # For each candidate frame
            for idx in remaining_indices:
                candidate = flattened_frames[idx]
                
                # Find minimum distance to any already selected frame
                min_distance = float('inf')
                for selected_idx in selected_indices:
                    selected = flattened_frames[selected_idx]
                    distance = np.mean(np.square(candidate - selected))  # MSE as distance
                    min_distance = min(min_distance, distance)
                
                # Select the frame with maximum minimum distance (most different)
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    next_frame_idx = idx
            
            if next_frame_idx != -1:
                selected_indices.append(next_frame_idx)
                remaining_indices.remove(next_frame_idx)
            else:
                break
                
        # Sort indices to maintain temporal order
        selected_indices.sort()
        
        return raw_video_slice[selected_indices]
    
    def __init__(
            self,
            csv_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        self.data = pd.read_csv(csv_path)
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly; 
        # 3: sparse+dense selection; 4: motion-guided adaptive sampling; 5: scene-change based sampling;
        # 6: diversity-based keyframe selection
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2, 3, 4, 5, 6]

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return len(self.data)

    def _get_text(self, video_id, sentence):
        choice_video_ids = [video_id]
        n_caption = len(choice_video_ids)

        k = n_caption
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(sentence)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        # video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
        #                   self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float) --DELETE--
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=float)

        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to it video format
            video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
            if os.path.exists(video_path) is False:
                video_path = video_path.replace(".mp4", ".webm")

            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']
            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    elif self.slice_framepos == 2:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                    elif self.slice_framepos == 3:
                        # X-CLIP style: long-range sparse + short-range dense frames
                        n_sparse = self.max_frames // 2  # Half frames for sparse coverage
                        n_dense = self.max_frames - n_sparse  # Half frames for dense coverage
                        
                        # Get sparse frames evenly distributed across entire video
                        sparse_indices = np.linspace(0, raw_video_slice.shape[0] - 1, num=n_sparse, dtype=int)
                        sparse_frames = raw_video_slice[sparse_indices]
                        
                        # Get dense frames from middle segment (or another interesting segment)
                        mid_point = raw_video_slice.shape[0] // 2
                        dense_start = max(0, mid_point - n_dense)
                        dense_end = min(raw_video_slice.shape[0], mid_point + n_dense)
                        
                        if dense_end - dense_start >= n_dense:
                            # If segment is large enough, sample uniformly within it
                            dense_indices = np.linspace(dense_start, dense_end - 1, num=n_dense, dtype=int)
                            dense_frames = raw_video_slice[dense_indices]
                        else:
                            # If segment is too small, take all frames and pad if needed
                            dense_frames = raw_video_slice[dense_start:dense_end]
                            # Pad if necessary by repeating last frame
                            if len(dense_frames) < n_dense:
                                padding = n_dense - len(dense_frames)
                                # Use last frame for padding
                                padding_frames = np.repeat(dense_frames[-1:], padding, axis=0)
                                dense_frames = np.concatenate([dense_frames, padding_frames], axis=0)
                        
                        # Combine sparse and dense frames
                        video_slice = np.concatenate([sparse_frames, dense_frames], axis=0)
                    elif self.slice_framepos == 4:
                        # Motion-Guided Adaptive Sampling
                        video_slice = self.motion_guided_adaptive_sampling(raw_video_slice, self.max_frames)
                    elif self.slice_framepos == 5:
                        # Scene-Change Based Sampling
                        video_slice = self.scene_change_based_sampling(raw_video_slice, self.max_frames)
                    elif self.slice_framepos == 6:
                        # Diversity-Based Keyframe Selection
                        video_slice = self.diversity_based_keyframe_selection(raw_video_slice, self.max_frames)
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, idx):
        video_id = self.data['video_id'].values[idx]
        sentence = self.data['sentence'].values[idx]

        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, sentence)
        video, video_mask = self._get_rawvideo(choice_video_ids)
        return pairs_text, pairs_mask, pairs_segment, video, video_mask

class MSRVTT_TrainDataLoader(Dataset):
    """MSRVTT train dataset loader."""
    
    def motion_guided_adaptive_sampling(self, raw_video_slice, max_frames):
        """Motion-Guided Adaptive Sampling - novel frame selection method
        
        Allocates frames based on motion intensity:
        1. Calculates frame-to-frame differences to detect motion
        2. Ensures global context with minimum uniform sampling
        3. Allocates remaining frames to high-motion segments
        """
        # Convert to numpy for easier manipulation if it's a torch tensor
        if isinstance(raw_video_slice, torch.Tensor):
            raw_video_np = raw_video_slice.cpu().numpy()
        else:
            raw_video_np = raw_video_slice
            
        # 1. Calculate inter-frame motion (simplified for implementation)
        frame_diffs = []
        for i in range(1, len(raw_video_np)):
            # Calculate mean squared difference between consecutive frames
            # Focus on visual content (channel 0) for efficiency
            diff = np.mean(np.square(
                raw_video_np[i, 0, 0].astype(float) - raw_video_np[i-1, 0, 0].astype(float)
            ))
            frame_diffs.append(diff)
        
        # Normalize motion scores
        if len(frame_diffs) > 0:
            motion_scores = np.array(frame_diffs) / (np.max(frame_diffs) + 1e-5)
        else:
            # Fallback for very short videos
            return raw_video_slice
        
        # 2. Ensure global context with minimum sampling
        global_context_frames = max(max_frames // 4, 1)  # 25% for global context
        global_indices = np.linspace(0, len(raw_video_np) - 1, num=global_context_frames, dtype=int)
        
        # 3. Allocate remaining frames to high-motion areas
        remaining_frames = max_frames - global_context_frames
        
        if remaining_frames <= 0 or len(motion_scores) == 0:
            # If max_frames is very small or no motion data, just use global sampling
            all_indices = global_indices
        else:
            # Create probability distribution favoring high-motion frames
            # Add small constant to avoid zero probabilities
            probs = motion_scores + 0.1
            probs = probs / np.sum(probs)
            
            # Sample frame indices based on motion probability
            # +1 because motion_scores starts from second frame
            motion_indices = np.random.choice(
                np.arange(1, len(raw_video_np)),
                size=min(remaining_frames, len(raw_video_np)-1),
                replace=False,
                p=probs
            )
            
            # 4. Combine and sort frames by their original temporal position
            all_indices = np.concatenate([global_indices, motion_indices])
            all_indices = np.sort(np.unique(all_indices))[:max_frames]  # Ensure no duplicates
        
        return raw_video_slice[all_indices]
    
    def diversity_based_keyframe_selection(self, raw_video_slice, max_frames):
        """Diversity-Based Keyframe Selection - selects the most diverse frames
        
        1. Starts with the first frame
        2. Iteratively selects frames that are most different from already selected ones
        3. This ensures maximum visual diversity in the selected frames
        """
        # Convert to numpy for easier manipulation if it's a torch tensor
        if isinstance(raw_video_slice, torch.Tensor):
            raw_video_np = raw_video_slice.cpu().numpy()
        else:
            raw_video_np = raw_video_slice
            
        if len(raw_video_np) <= max_frames:
            return raw_video_slice
            
        # Flatten frames for easier distance calculation (using first channel)
        flattened_frames = raw_video_np[:, 0, 0].reshape(len(raw_video_np), -1)
        
        # Initialize with the first frame
        selected_indices = [0]
        remaining_indices = list(range(1, len(raw_video_np)))
        
        # Iteratively select the most diverse frames
        while len(selected_indices) < max_frames and remaining_indices:
            max_min_distance = -1
            next_frame_idx = -1
            
            # For each candidate frame
            for idx in remaining_indices:
                candidate = flattened_frames[idx]
                
                # Find minimum distance to any already selected frame
                min_distance = float('inf')
                for selected_idx in selected_indices:
                    selected = flattened_frames[selected_idx]
                    distance = np.mean(np.square(candidate - selected))  # MSE as distance
                    min_distance = min(min_distance, distance)
                
                # Select the frame with maximum minimum distance (most different)
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    next_frame_idx = idx
            
            if next_frame_idx != -1:
                selected_indices.append(next_frame_idx)
                remaining_indices.remove(next_frame_idx)
            else:
                break
                
        # Sort indices to maintain temporal order
        selected_indices.sort()
        
        return raw_video_slice[selected_indices]
    
    def scene_change_based_sampling(self, raw_video_slice, max_frames):
        """Scene-Change Based Sampling - selects frames at scene transitions
        
        1. Calculates frame differences to detect scene changes
        2. Samples frames after major scene transitions
        3. Fills remaining frames with uniform sampling for coverage
        """
        # Convert to numpy for easier manipulation if it's a torch tensor
        if isinstance(raw_video_slice, torch.Tensor):
            raw_video_np = raw_video_slice.cpu().numpy()
        else:
            raw_video_np = raw_video_slice
            
        # Calculate frame differences to detect scene changes
        if len(raw_video_np) <= 1:
            return raw_video_slice
            
        frame_diffs = []
        for i in range(1, len(raw_video_np)):
            # Calculate mean absolute difference between consecutive frames
            diff = np.mean(np.abs(
                raw_video_np[i, 0, 0].astype(float) - raw_video_np[i-1, 0, 0].astype(float)
            ))
            frame_diffs.append(diff)
            
        # Detect scene changes (frames with difference higher than threshold)
        frame_diffs = np.array(frame_diffs)
        mean_diff = np.mean(frame_diffs)
        std_diff = np.std(frame_diffs)
        # Define threshold as mean + 1.5*std (adaptive to video content)
        threshold = mean_diff + 1.5 * std_diff
        
        # Find frames that exceed the threshold (potential scene changes)
        # Add 1 because frame_diffs indices are offset by 1
        scene_change_indices = np.where(frame_diffs > threshold)[0] + 1
        
        # Always include first frame
        if len(scene_change_indices) == 0 or scene_change_indices[0] != 0:
            scene_indices = np.concatenate(([0], scene_change_indices))
        else:
            scene_indices = scene_change_indices
            
        # If we have more scene changes than max_frames, select the most significant ones
        if len(scene_indices) > max_frames:
            # Sort scene changes by magnitude of difference
            sorted_indices = np.argsort(frame_diffs[scene_indices-1])[::-1]
            scene_indices = scene_indices[sorted_indices[:max_frames]]
            scene_indices = np.sort(scene_indices)
        
        # If we have fewer scene changes than max_frames, fill with uniform sampling
        if len(scene_indices) < max_frames:
            remaining_frames = max_frames - len(scene_indices)
            # Add uniformly sampled frames for coverage
            uniform_indices = np.linspace(0, len(raw_video_np) - 1, num=remaining_frames + 2, dtype=int)
            # Remove first and last frame which might duplicate scene change frames
            uniform_indices = uniform_indices[1:-1]
            
            # Combine scene change frames with uniform frames
            all_indices = np.concatenate([scene_indices, uniform_indices])
            all_indices = np.sort(np.unique(all_indices))[:max_frames]
        else:
            all_indices = scene_indices[:max_frames]
            
        return raw_video_slice[all_indices]
    
    def __init__(
            self,
            csv_path,
            json_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            unfold_sentences=False,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        self.csv = pd.read_csv(csv_path)
        self.data = json.load(open(json_path, 'r'))
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly; 
        # 3: sparse+dense selection; 4: motion-guided adaptive sampling; 5: scene-change based sampling;
        # 6: diversity-based keyframe selection
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2, 3, 4, 5, 6]

        self.unfold_sentences = unfold_sentences
        self.sample_len = 0
        if self.unfold_sentences:
            train_video_ids = list(self.csv['video_id'].values)
            self.sentences_dict = {}
            for itm in self.data['sentences']:
                if itm['video_id'] in train_video_ids:
                    self.sentences_dict[len(self.sentences_dict)] = (itm['video_id'], itm['caption'])
            self.sample_len = len(self.sentences_dict)
        else:
            num_sentences = 0
            self.sentences = defaultdict(list)
            s_video_id_set = set()
            for itm in self.data['sentences']:
                self.sentences[itm['video_id']].append(itm['caption'])
                num_sentences += 1
                s_video_id_set.add(itm['video_id'])

            # Use to find the clips in the same video
            self.parent_ids = {}
            self.children_video_ids = defaultdict(list)
            for itm in self.data['videos']:
                vid = itm["video_id"]
                url_posfix = itm["url"].split("?v=")[-1]
                self.parent_ids[vid] = url_posfix
                self.children_video_ids[url_posfix].append(vid)
            self.sample_len = len(self.csv)

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption=None):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            if caption is not None:
                words = self.tokenizer.tokenize(caption)
            else:
                words = self._get_single_text(video_id)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = self.sentences[video_id][rind]
        words = self.tokenizer.tokenize(caption)
        return words

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        # video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
        #                   self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)  --DELETE--
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=float)

        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to it video format
            video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
            if os.path.exists(video_path) is False:
                video_path = video_path.replace(".mp4", ".webm")

            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']
            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    elif self.slice_framepos == 2:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                    elif self.slice_framepos == 3:
                        # X-CLIP style: long-range sparse + short-range dense frames
                        n_sparse = self.max_frames // 2  # Half frames for sparse coverage
                        n_dense = self.max_frames - n_sparse  # Half frames for dense coverage
                        
                        # Get sparse frames evenly distributed across entire video
                        sparse_indices = np.linspace(0, raw_video_slice.shape[0] - 1, num=n_sparse, dtype=int)
                        sparse_frames = raw_video_slice[sparse_indices]
                        
                        # Get dense frames from middle segment (or another interesting segment)
                        mid_point = raw_video_slice.shape[0] // 2
                        dense_start = max(0, mid_point - n_dense)
                        dense_end = min(raw_video_slice.shape[0], mid_point + n_dense)
                        
                        if dense_end - dense_start >= n_dense:
                            # If segment is large enough, sample uniformly within it
                            dense_indices = np.linspace(dense_start, dense_end - 1, num=n_dense, dtype=int)
                            dense_frames = raw_video_slice[dense_indices]
                        else:
                            # If segment is too small, take all frames and pad if needed
                            dense_frames = raw_video_slice[dense_start:dense_end]
                            # Pad if necessary by repeating last frame
                            if len(dense_frames) < n_dense:
                                padding = n_dense - len(dense_frames)
                                # Use last frame for padding
                                padding_frames = np.repeat(dense_frames[-1:], padding, axis=0)
                                dense_frames = np.concatenate([dense_frames, padding_frames], axis=0)
                        
                        # Combine sparse and dense frames
                        video_slice = np.concatenate([sparse_frames, dense_frames], axis=0)
                    elif self.slice_framepos == 4:
                        # Motion-Guided Adaptive Sampling
                        video_slice = self.motion_guided_adaptive_sampling(raw_video_slice, self.max_frames)
                    elif self.slice_framepos == 5:
                        # Scene-Change Based Sampling
                        video_slice = self.scene_change_based_sampling(raw_video_slice, self.max_frames)
                    elif self.slice_framepos == 6:
                        # Diversity-Based Keyframe Selection
                        video_slice = self.diversity_based_keyframe_selection(raw_video_slice, self.max_frames)
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, idx):
        if self.unfold_sentences:
            video_id, caption = self.sentences_dict[idx]
        else:
            video_id, caption = self.csv['video_id'].values[idx], None
        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
        video, video_mask = self._get_rawvideo(choice_video_ids)
        return pairs_text, pairs_mask, pairs_segment, video, video_mask
