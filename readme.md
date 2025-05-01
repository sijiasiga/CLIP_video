# CLIP_video 
This project is based on CLIP4Clip(https://github.com/ArrowLuo/CLIP4Clip), a framework for video-text retrieval using CLIP.
We adapted the original version to support single-GPU training and evaluation for more accessible development and experimentation

## Key Changes:
1. Removed all dependencies on distributed training (torch.distributed)
2. Eliminated local_rank, world_size, and related parameters
3. Integrated tqdm progress bars for clearer training feedback in notebooks
4. Retain only MSRVTT dataloader
5. Easy to run in Colab or on local machines with only one GPU
6. Add max pooling simulation calculator
7. Add Sparse+dense frames sampling (half uniform, half from middle) method
8.  Motion-guided adaptive sampling (based on movement) method
9.  Add scene-change based sampling (based on visual transitions) method
10.  Add diversity-based keyframe selection (maximal visual diversity) method
11. Add improved sparse+dense sampling (activity-aware) method
12. Add enhanced uniform sampling (uniform + key frames) method
13. Add advanced hybrid sampling (intelligent sparse + focused dense) method
   
## Demo Notebook:
We provide an interactive notebook:
**CLIP_video_training.ipynb** <br>
This notebook demonstrates the full workflow — from loading MSRVTT features to training and evaluation — using a subset of 9k video-text pairs for quick testing.

## Config(example)
num_thread_reader: 8 
epochs: 20 
batch_size: 64 
lr: 1e-4 
max_words: 32 
max_frames: 8 
batch_size_val: 8 
feature_framerate: 1 
coef_lr: 1e-3 
freeze_layer_num: 6 
slice_framepos: 2  
loose_type 
linear_patch: 2d 
sim_header: seqLSTM 
pretrained_clip_name: ViT-B/32 

