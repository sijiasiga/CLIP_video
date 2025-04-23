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
from scipy import signal

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
    
    def enhanced_uniform_sampling(self, raw_video_slice, max_frames):
        """Enhanced Uniform Sampling - combines uniform sampling with key frames
        
        1. Samples most frames uniformly for global coverage
        2. Dedicates some frames to the beginning/end (important context)
        3. Ensures no frames are duplicated
        """
        if len(raw_video_slice) <= max_frames:
            return raw_video_slice
            
        # Calculate how many frames to allocate for each approach
        uniform_frames = int(max_frames * 0.8)  # 80% uniform
        special_frames = max_frames - uniform_frames  # 20% special
        
        # Get uniform frames
        uniform_indices = np.linspace(0, raw_video_slice.shape[0] - 1, num=uniform_frames, dtype=int)
        
        # Get special frames from beginning and end (these are often important for context)
        begin_frames = special_frames // 2
        end_frames = special_frames - begin_frames
        
        special_indices = np.concatenate([
            np.arange(min(begin_frames, raw_video_slice.shape[0] // 10)),  # First 10% or fewer
            np.arange(max(0, raw_video_slice.shape[0] - end_frames), raw_video_slice.shape[0])  # Last few frames
        ])
        
        # Combine indices and remove duplicates
        all_indices = np.concatenate([uniform_indices, special_indices])
        all_indices = np.sort(np.unique(all_indices))[:max_frames]
        
        return raw_video_slice[all_indices]
    
    def improved_sparse_dense_sampling(self, raw_video_slice, max_frames):
        """Improved Sparse+Dense Sampling - better implementation of X-CLIP style sampling
        
        1. Uses sparse sampling across entire video for global context
        2. Uses multiple dense sampling clusters focused on high-activity regions
        3. Balances coverage and detail based on video length
        """
        if len(raw_video_slice) <= max_frames:
            return raw_video_slice
            
        video_length = raw_video_slice.shape[0]
        
        # Adaptive allocation of frames based on video length
        if video_length > max_frames * 3:
            # For long videos: 40% sparse, 60% dense
            n_sparse = max(max_frames * 4 // 10, 1)
        else:
            # For shorter videos: 60% sparse, 40% dense
            n_sparse = max(max_frames * 6 // 10, 1)
            
        n_dense = max_frames - n_sparse
        
        # Get sparse frames evenly distributed for global context
        sparse_indices = np.linspace(0, video_length - 1, num=n_sparse, dtype=int)
        sparse_frames = raw_video_slice[sparse_indices]
        
        # If the video is very short, fall back to uniform sampling
        if video_length <= n_sparse * 2:
            return raw_video_slice[sparse_indices]
            
        # Calculate frame differences to identify high-activity regions
        frame_diffs = []
        for i in range(1, video_length):
            # Calculate mean absolute difference between consecutive frames
            if isinstance(raw_video_slice, torch.Tensor):
                diff = torch.mean(torch.abs(
                    raw_video_slice[i, 0, 0].float() - raw_video_slice[i-1, 0, 0].float()
                )).item()
            else:
                diff = np.mean(np.abs(
                    raw_video_slice[i, 0, 0].astype(float) - raw_video_slice[i-1, 0, 0].astype(float)
                ))
            frame_diffs.append(diff)
            
        frame_diffs = np.array(frame_diffs)
        
        # Find peaks in frame differences (high activity regions)
        if len(frame_diffs) > 5:  # Need enough frames to find peaks
            # Identify peaks (high motion areas)
            peaks, _ = signal.find_peaks(frame_diffs, distance=max(3, video_length // 20))
            
            # If no clear peaks found, use middle section
            if len(peaks) == 0:
                # Fallback to middle-focused sampling
                mid_point = video_length // 2
                start_idx = max(0, mid_point - n_dense // 2)
                end_idx = min(video_length, start_idx + n_dense)
                dense_indices = np.linspace(start_idx, end_idx - 1, num=n_dense, dtype=int)
            else:
                # Rank peaks by magnitude
                peak_values = frame_diffs[peaks]
                sorted_peak_indices = np.argsort(-peak_values)  # Descending order
                
                # Select top clusters
                n_clusters = min(3, len(peaks))
                selected_peaks = [peaks[sorted_peak_indices[i]] for i in range(n_clusters)]
                
                # Get dense samples around these peaks
                dense_indices = []
                frames_per_cluster = n_dense // n_clusters
                
                for peak in selected_peaks:
                    # Define a window around the peak
                    half_window = max(2, video_length // 20)
                    start = max(0, peak - half_window)
                    end = min(video_length - 1, peak + half_window)
                    
                    # Sample frames within this window
                    if end - start > frames_per_cluster:
                        cluster_indices = np.linspace(start, end, num=frames_per_cluster, dtype=int)
                    else:
                        # If window is too small, take all frames
                        cluster_indices = np.arange(start, end + 1)
                    
                    dense_indices.extend(cluster_indices)
                    
                # If we didn't get enough dense frames, add some from highest activity regions
                while len(dense_indices) < n_dense and len(frame_diffs) > 0:
                    # Find the highest activity frame that's not already included
                    candidate_indices = np.argsort(-frame_diffs)
                    for idx in candidate_indices:
                        if idx not in dense_indices and idx + 1 not in dense_indices:
                            dense_indices.append(idx + 1)  # +1 because diffs are between frames i-1 and i
                            break
                    if len(dense_indices) < n_dense:
                        # If we still need frames, break the loop to avoid infinite looping
                        break
                
                # Ensure we don't exceed n_dense
                dense_indices = dense_indices[:n_dense]
        else:
            # For very short videos with few frame differences
            mid_point = video_length // 2
            start_idx = max(0, mid_point - n_dense // 2)
            end_idx = min(video_length, start_idx + n_dense)
            dense_indices = np.linspace(start_idx, end_idx - 1, num=n_dense, dtype=int)
        
        # Combine sparse and dense frames
        dense_indices = np.array(dense_indices, dtype=int)
        all_indices = np.concatenate([sparse_indices, dense_indices])
        all_indices = np.sort(np.unique(all_indices))[:max_frames]
        
        return raw_video_slice[all_indices]
    
    def advanced_hybrid_sampling(self, raw_video_slice, max_frames):
        """Advanced Hybrid Sampling - combines intelligent sparse and focused dense sampling
        
        1. Separates sparse global frames and dense local frames
        2. Applies strategic weighting to prioritize informative frames
        3. Uses activity-based analysis to identify important segments
        """
        if len(raw_video_slice) <= max_frames:
            return raw_video_slice
            
        video_length = raw_video_slice.shape[0]
        
        # Use roughly half for sparse, half for dense
        n_sparse = max_frames // 2
        n_dense = max_frames - n_sparse
        
        # --- PART 1: Intelligent sparse frame selection ---
        # Calculate importance of each frame using temporal gradients
        if isinstance(raw_video_slice, torch.Tensor):
            raw_video_np = raw_video_slice.cpu().numpy()
        else:
            raw_video_np = raw_video_slice
        
        # Compute frame importance scores based on temporal differences
        frame_diffs = np.zeros(video_length)
        
        # Forward differences
        for i in range(1, video_length):
            diff = np.mean(np.abs(
                raw_video_np[i, 0, 0].astype(float) - raw_video_np[i-1, 0, 0].astype(float)
            ))
            frame_diffs[i-1] += diff
            frame_diffs[i] += diff
        
        # Normalize importance scores
        if np.max(frame_diffs) > 0:
            importance = frame_diffs / np.max(frame_diffs)
        else:
            importance = np.ones(video_length) / video_length
        
        # Combine uniform sampling and importance-based sampling
        # 70% uniform, 30% importance-weighted for sparse frames
        uniform_sparse = int(n_sparse * 0.7)
        weighted_sparse = n_sparse - uniform_sparse
        
        # Get uniformly sampled frames for coverage
        uniform_indices = np.linspace(0, video_length - 1, num=uniform_sparse, dtype=int)
        
        # Get importance-weighted frames
        # Create a probability distribution based on importance
        if weighted_sparse > 0:
            # Avoid selecting frames too close to uniform indices
            mask = np.ones(video_length, dtype=bool)
            for idx in uniform_indices:
                window = 3  # Avoid frames within this window
                low = max(0, idx - window)
                high = min(video_length, idx + window + 1)
                mask[low:high] = False
            
            # If we've masked too much, fall back to uniform
            if np.sum(mask) < weighted_sparse:
                remaining_indices = np.linspace(0, video_length - 1, num=weighted_sparse, dtype=int)
            else:
                # Select based on importance, but avoid already selected frames
                masked_importance = importance.copy()
                masked_importance[~mask] = 0
                
                # Normalize to create a probability distribution
                if np.sum(masked_importance) > 0:
                    probs = masked_importance / np.sum(masked_importance)
                    remaining_indices = np.random.choice(
                        np.arange(video_length),
                        size=weighted_sparse,
                        replace=False,
                        p=probs
                    )
                else:
                    # Fallback if all important frames were masked
                    remaining_indices = np.linspace(0, video_length - 1, num=weighted_sparse, dtype=int)
        else:
            remaining_indices = np.array([], dtype=int)
        
        # Combine uniform and importance-weighted sparse indices
        sparse_indices = np.concatenate([uniform_indices, remaining_indices])
        sparse_indices = np.sort(np.unique(sparse_indices))
        
        # --- PART 2: Multi-scale dense frame selection ---
        # Focus dense frames on regions with high activity
        
        # Find high-activity regions
        activity_scores = np.zeros(video_length)
        window_size = max(5, video_length // 20)  # Adaptive window
        
        # Compute activity over windows
        for i in range(video_length):
            start = max(0, i - window_size // 2)
            end = min(video_length, i + window_size // 2 + 1)
            activity_scores[i] = np.mean(frame_diffs[start:end])
        
        # Find the highest activity region center
        if np.sum(activity_scores) > 0:
            center_idx = np.argmax(activity_scores)
        else:
            # Fallback to middle if no clear activity peak
            center_idx = video_length // 2
        
        # Calculate start and end of dense region with adaptive sizing
        # Longer videos get relatively smaller dense regions
        region_size_factor = max(0.3, min(0.6, 30 / video_length))
        region_radius = int(video_length * region_size_factor / 2)
        
        dense_start = max(0, center_idx - region_radius)
        dense_end = min(video_length, center_idx + region_radius)
        
        # Get dense frames with slightly higher sampling rate near the center
        if dense_end - dense_start > n_dense:
            # Create non-uniform sampling that's denser near the center
            center_weight = np.ones(dense_end - dense_start)
            center_pos = center_idx - dense_start
            
            # Apply Gaussian-like weighting centered at the activity peak
            for i in range(len(center_weight)):
                dist = abs(i - center_pos) / (dense_end - dense_start)
                center_weight[i] = np.exp(-5 * dist)
            
            # Normalize to create a probability distribution
            center_weight = center_weight / np.sum(center_weight)
            
            # Sample dense frames with this non-uniform distribution
            dense_local_indices = np.random.choice(
                np.arange(dense_start, dense_end),
                size=n_dense,
                replace=False,
                p=center_weight
            )
            dense_indices = np.sort(dense_local_indices)
        else:
            # If region is smaller than needed frames, take all and pad uniformly
            dense_indices = np.arange(dense_start, dense_end)
            
            # Add uniformly sampled frames outside the dense region if needed
            if len(dense_indices) < n_dense:
                remaining = n_dense - len(dense_indices)
                
                # Create mask to exclude already selected frames
                mask = np.ones(video_length, dtype=bool)
                mask[dense_start:dense_end] = False
                for idx in sparse_indices:
                    if idx < video_length:
                        mask[idx] = False
                
                # Get remaining indices
                valid_indices = np.where(mask)[0]
                if len(valid_indices) >= remaining:
                    extra_indices = np.random.choice(valid_indices, size=remaining, replace=False)
                    dense_indices = np.concatenate([dense_indices, extra_indices])
                else:
                    # If not enough unique frames left, allow some duplicates
                    # (though this should rarely happen)
                    extra_indices = np.linspace(0, video_length - 1, num=remaining, dtype=int)
                    dense_indices = np.concatenate([dense_indices, extra_indices])
                
                dense_indices = np.sort(np.unique(dense_indices))[:n_dense]
        
        # --- PART 3: Combine sparse and dense frames ---
        all_indices = np.concatenate([sparse_indices, dense_indices])
        all_indices = np.sort(np.unique(all_indices))[:max_frames]
        
        # Return the selected frames
        return raw_video_slice[all_indices]
    
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
        
        # Frame selection methods:
        # slice_framepos = 0: Head frames (cut from beginning of video)
        # slice_framepos = 1: Tail frames (cut from end of video)
        # slice_framepos = 2: Uniform frames (evenly distributed)
        # slice_framepos = 3: Sparse+dense frames (half uniform, half from middle)
        # slice_framepos = 4: Motion-guided adaptive sampling (based on movement)
        # slice_framepos = 5: Scene-change based sampling (based on visual transitions)
        # slice_framepos = 6: Diversity-based keyframe selection (maximal visual diversity)
        # slice_framepos = 7: Improved sparse+dense sampling (activity-aware)
        # slice_framepos = 8: Enhanced uniform sampling (uniform + key frames)
        # slice_framepos = 9: Advanced hybrid sampling (intelligent sparse + focused dense)
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

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
                    elif self.slice_framepos == 7:
                        # Improved Sparse+Dense Sampling
                        video_slice = self.improved_sparse_dense_sampling(raw_video_slice, self.max_frames)
                    elif self.slice_framepos == 8:
                        # Enhanced Uniform Sampling
                        video_slice = self.enhanced_uniform_sampling(raw_video_slice, self.max_frames)
                    elif self.slice_framepos == 9:
                        # Advanced Hybrid Sampling
                        video_slice = self.advanced_hybrid_sampling(raw_video_slice, self.max_frames)
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
    
    def enhanced_uniform_sampling(self, raw_video_slice, max_frames):
        """Enhanced Uniform Sampling - combines uniform sampling with key frames
        
        1. Samples most frames uniformly for global coverage
        2. Dedicates some frames to the beginning/end (important context)
        3. Ensures no frames are duplicated
        """
        if len(raw_video_slice) <= max_frames:
            return raw_video_slice
            
        # Calculate how many frames to allocate for each approach
        uniform_frames = int(max_frames * 0.8)  # 80% uniform
        special_frames = max_frames - uniform_frames  # 20% special
        
        # Get uniform frames
        uniform_indices = np.linspace(0, raw_video_slice.shape[0] - 1, num=uniform_frames, dtype=int)
        
        # Get special frames from beginning and end (these are often important for context)
        begin_frames = special_frames // 2
        end_frames = special_frames - begin_frames
        
        special_indices = np.concatenate([
            np.arange(min(begin_frames, raw_video_slice.shape[0] // 10)),  # First 10% or fewer
            np.arange(max(0, raw_video_slice.shape[0] - end_frames), raw_video_slice.shape[0])  # Last few frames
        ])
        
        # Combine indices and remove duplicates
        all_indices = np.concatenate([uniform_indices, special_indices])
        all_indices = np.sort(np.unique(all_indices))[:max_frames]
        
        return raw_video_slice[all_indices]
    
    def improved_sparse_dense_sampling(self, raw_video_slice, max_frames):
        """Improved Sparse+Dense Sampling - better implementation of X-CLIP style sampling
        
        1. Uses sparse sampling across entire video for global context
        2. Uses multiple dense sampling clusters focused on high-activity regions
        3. Balances coverage and detail based on video length
        """
        if len(raw_video_slice) <= max_frames:
            return raw_video_slice
            
        video_length = raw_video_slice.shape[0]
        
        # Adaptive allocation of frames based on video length
        if video_length > max_frames * 3:
            # For long videos: 40% sparse, 60% dense
            n_sparse = max(max_frames * 4 // 10, 1)
        else:
            # For shorter videos: 60% sparse, 40% dense
            n_sparse = max(max_frames * 6 // 10, 1)
            
        n_dense = max_frames - n_sparse
        
        # Get sparse frames evenly distributed for global context
        sparse_indices = np.linspace(0, video_length - 1, num=n_sparse, dtype=int)
        sparse_frames = raw_video_slice[sparse_indices]
        
        # If the video is very short, fall back to uniform sampling
        if video_length <= n_sparse * 2:
            return raw_video_slice[sparse_indices]
            
        # Calculate frame differences to identify high-activity regions
        frame_diffs = []
        for i in range(1, video_length):
            # Calculate mean absolute difference between consecutive frames
            if isinstance(raw_video_slice, torch.Tensor):
                diff = torch.mean(torch.abs(
                    raw_video_slice[i, 0, 0].float() - raw_video_slice[i-1, 0, 0].float()
                )).item()
            else:
                diff = np.mean(np.abs(
                    raw_video_slice[i, 0, 0].astype(float) - raw_video_slice[i-1, 0, 0].astype(float)
                ))
            frame_diffs.append(diff)
            
        frame_diffs = np.array(frame_diffs)
        
        # Find peaks in frame differences (high activity regions)
        if len(frame_diffs) > 5:  # Need enough frames to find peaks
            # Identify peaks (high motion areas)
            peaks, _ = signal.find_peaks(frame_diffs, distance=max(3, video_length // 20))
            
            # If no clear peaks found, use middle section
            if len(peaks) == 0:
                # Fallback to middle-focused sampling
                mid_point = video_length // 2
                start_idx = max(0, mid_point - n_dense // 2)
                end_idx = min(video_length, start_idx + n_dense)
                dense_indices = np.linspace(start_idx, end_idx - 1, num=n_dense, dtype=int)
            else:
                # Rank peaks by magnitude
                peak_values = frame_diffs[peaks]
                sorted_peak_indices = np.argsort(-peak_values)  # Descending order
                
                # Select top clusters
                n_clusters = min(3, len(peaks))
                selected_peaks = [peaks[sorted_peak_indices[i]] for i in range(n_clusters)]
                
                # Get dense samples around these peaks
                dense_indices = []
                frames_per_cluster = n_dense // n_clusters
                
                for peak in selected_peaks:
                    # Define a window around the peak
                    half_window = max(2, video_length // 20)
                    start = max(0, peak - half_window)
                    end = min(video_length - 1, peak + half_window)
                    
                    # Sample frames within this window
                    if end - start > frames_per_cluster:
                        cluster_indices = np.linspace(start, end, num=frames_per_cluster, dtype=int)
                    else:
                        # If window is too small, take all frames
                        cluster_indices = np.arange(start, end + 1)
                    
                    dense_indices.extend(cluster_indices)
                    
                # If we didn't get enough dense frames, add some from highest activity regions
                while len(dense_indices) < n_dense and len(frame_diffs) > 0:
                    # Find the highest activity frame that's not already included
                    candidate_indices = np.argsort(-frame_diffs)
                    for idx in candidate_indices:
                        if idx not in dense_indices and idx + 1 not in dense_indices:
                            dense_indices.append(idx + 1)  # +1 because diffs are between frames i-1 and i
                            break
                    if len(dense_indices) < n_dense:
                        # If we still need frames, break the loop to avoid infinite looping
                        break
                
                # Ensure we don't exceed n_dense
                dense_indices = dense_indices[:n_dense]
        else:
            # For very short videos with few frame differences
            mid_point = video_length // 2
            start_idx = max(0, mid_point - n_dense // 2)
            end_idx = min(video_length, start_idx + n_dense)
            dense_indices = np.linspace(start_idx, end_idx - 1, num=n_dense, dtype=int)
        
        # Combine sparse and dense frames
        dense_indices = np.array(dense_indices, dtype=int)
        all_indices = np.concatenate([sparse_indices, dense_indices])
        all_indices = np.sort(np.unique(all_indices))[:max_frames]
        
        return raw_video_slice[all_indices]
    
    def advanced_hybrid_sampling(self, raw_video_slice, max_frames):
        """Advanced Hybrid Sampling - combines intelligent sparse and focused dense sampling
        
        1. Separates sparse global frames and dense local frames
        2. Applies strategic weighting to prioritize informative frames
        3. Uses activity-based analysis to identify important segments
        """
        if len(raw_video_slice) <= max_frames:
            return raw_video_slice
            
        video_length = raw_video_slice.shape[0]
        
        # Use roughly half for sparse, half for dense
        n_sparse = max_frames // 2
        n_dense = max_frames - n_sparse
        
        # --- PART 1: Intelligent sparse frame selection ---
        # Calculate importance of each frame using temporal gradients
        if isinstance(raw_video_slice, torch.Tensor):
            raw_video_np = raw_video_slice.cpu().numpy()
        else:
            raw_video_np = raw_video_slice
        
        # Compute frame importance scores based on temporal differences
        frame_diffs = np.zeros(video_length)
        
        # Forward differences
        for i in range(1, video_length):
            diff = np.mean(np.abs(
                raw_video_np[i, 0, 0].astype(float) - raw_video_np[i-1, 0, 0].astype(float)
            ))
            frame_diffs[i-1] += diff
            frame_diffs[i] += diff
        
        # Normalize importance scores
        if np.max(frame_diffs) > 0:
            importance = frame_diffs / np.max(frame_diffs)
        else:
            importance = np.ones(video_length) / video_length
        
        # Combine uniform sampling and importance-based sampling
        # 70% uniform, 30% importance-weighted for sparse frames
        uniform_sparse = int(n_sparse * 0.7)
        weighted_sparse = n_sparse - uniform_sparse
        
        # Get uniformly sampled frames for coverage
        uniform_indices = np.linspace(0, video_length - 1, num=uniform_sparse, dtype=int)
        
        # Get importance-weighted frames
        # Create a probability distribution based on importance
        if weighted_sparse > 0:
            # Avoid selecting frames too close to uniform indices
            mask = np.ones(video_length, dtype=bool)
            for idx in uniform_indices:
                window = 3  # Avoid frames within this window
                low = max(0, idx - window)
                high = min(video_length, idx + window + 1)
                mask[low:high] = False
            
            # If we've masked too much, fall back to uniform
            if np.sum(mask) < weighted_sparse:
                remaining_indices = np.linspace(0, video_length - 1, num=weighted_sparse, dtype=int)
            else:
                # Select based on importance, but avoid already selected frames
                masked_importance = importance.copy()
                masked_importance[~mask] = 0
                
                # Normalize to create a probability distribution
                if np.sum(masked_importance) > 0:
                    probs = masked_importance / np.sum(masked_importance)
                    remaining_indices = np.random.choice(
                        np.arange(video_length),
                        size=weighted_sparse,
                        replace=False,
                        p=probs
                    )
                else:
                    # Fallback if all important frames were masked
                    remaining_indices = np.linspace(0, video_length - 1, num=weighted_sparse, dtype=int)
        else:
            remaining_indices = np.array([], dtype=int)
        
        # Combine uniform and importance-weighted sparse indices
        sparse_indices = np.concatenate([uniform_indices, remaining_indices])
        sparse_indices = np.sort(np.unique(sparse_indices))
        
        # --- PART 2: Multi-scale dense frame selection ---
        # Focus dense frames on regions with high activity
        
        # Find high-activity regions
        activity_scores = np.zeros(video_length)
        window_size = max(5, video_length // 20)  # Adaptive window
        
        # Compute activity over windows
        for i in range(video_length):
            start = max(0, i - window_size // 2)
            end = min(video_length, i + window_size // 2 + 1)
            activity_scores[i] = np.mean(frame_diffs[start:end])
        
        # Find the highest activity region center
        if np.sum(activity_scores) > 0:
            center_idx = np.argmax(activity_scores)
        else:
            # Fallback to middle if no clear activity peak
            center_idx = video_length // 2
        
        # Calculate start and end of dense region with adaptive sizing
        # Longer videos get relatively smaller dense regions
        region_size_factor = max(0.3, min(0.6, 30 / video_length))
        region_radius = int(video_length * region_size_factor / 2)
        
        dense_start = max(0, center_idx - region_radius)
        dense_end = min(video_length, center_idx + region_radius)
        
        # Get dense frames with slightly higher sampling rate near the center
        if dense_end - dense_start > n_dense:
            # Create non-uniform sampling that's denser near the center
            center_weight = np.ones(dense_end - dense_start)
            center_pos = center_idx - dense_start
            
            # Apply Gaussian-like weighting centered at the activity peak
            for i in range(len(center_weight)):
                dist = abs(i - center_pos) / (dense_end - dense_start)
                center_weight[i] = np.exp(-5 * dist)
            
            # Normalize to create a probability distribution
            center_weight = center_weight / np.sum(center_weight)
            
            # Sample dense frames with this non-uniform distribution
            dense_local_indices = np.random.choice(
                np.arange(dense_start, dense_end),
                size=n_dense,
                replace=False,
                p=center_weight
            )
            dense_indices = np.sort(dense_local_indices)
        else:
            # If region is smaller than needed frames, take all and pad uniformly
            dense_indices = np.arange(dense_start, dense_end)
            
            # Add uniformly sampled frames outside the dense region if needed
            if len(dense_indices) < n_dense:
                remaining = n_dense - len(dense_indices)
                
                # Create mask to exclude already selected frames
                mask = np.ones(video_length, dtype=bool)
                mask[dense_start:dense_end] = False
                for idx in sparse_indices:
                    if idx < video_length:
                        mask[idx] = False
                
                # Get remaining indices
                valid_indices = np.where(mask)[0]
                if len(valid_indices) >= remaining:
                    extra_indices = np.random.choice(valid_indices, size=remaining, replace=False)
                    dense_indices = np.concatenate([dense_indices, extra_indices])
                else:
                    # If not enough unique frames left, allow some duplicates
                    # (though this should rarely happen)
                    extra_indices = np.linspace(0, video_length - 1, num=remaining, dtype=int)
                    dense_indices = np.concatenate([dense_indices, extra_indices])
                
                dense_indices = np.sort(np.unique(dense_indices))[:n_dense]
        
        # --- PART 3: Combine sparse and dense frames ---
        all_indices = np.concatenate([sparse_indices, dense_indices])
        all_indices = np.sort(np.unique(all_indices))[:max_frames]
        
        # Return the selected frames
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
        
        # Frame selection methods:
        # slice_framepos = 0: Head frames (cut from beginning of video)
        # slice_framepos = 1: Tail frames (cut from end of video)
        # slice_framepos = 2: Uniform frames (evenly distributed)
        # slice_framepos = 3: Sparse+dense frames (half uniform, half from middle)
        # slice_framepos = 4: Motion-guided adaptive sampling (based on movement)
        # slice_framepos = 5: Scene-change based sampling (based on visual transitions)
        # slice_framepos = 6: Diversity-based keyframe selection (maximal visual diversity)
        # slice_framepos = 7: Improved sparse+dense sampling (activity-aware)
        # slice_framepos = 8: Enhanced uniform sampling (uniform + key frames)
        # slice_framepos = 9: Advanced hybrid sampling (intelligent sparse + focused dense)
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

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
                    elif self.slice_framepos == 7:
                        # Improved Sparse+Dense Sampling
                        video_slice = self.improved_sparse_dense_sampling(raw_video_slice, self.max_frames)
                    elif self.slice_framepos == 8:
                        # Enhanced Uniform Sampling
                        video_slice = self.enhanced_uniform_sampling(raw_video_slice, self.max_frames)
                    elif self.slice_framepos == 9:
                        # Advanced Hybrid Sampling
                        video_slice = self.advanced_hybrid_sampling(raw_video_slice, self.max_frames)
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
