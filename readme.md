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

## Configuration Example

| Parameter             | Value       | Description (optional)                        |
|-----------------------|-------------|-----------------------------------------------|
| `num_thread_reader`   | 8           | Number of threads for data loading            |
| `epochs`              | 20          | Total number of training epochs               |
| `batch_size`          | 64          | Training batch size                           |
| `lr`                  | 1e-4        | Learning rate for main model parameters       |
| `max_words`           | 32          | Maximum number of words in a caption          |
| `max_frames`          | 8           | Maximum number of video frames                |
| `batch_size_val`      | 8           | Batch size for validation                     |
| `feature_framerate`   | 1           | Frame sampling rate                           |
| `coef_lr`             | 1e-3        | Learning rate coefficient for the text encoder|
| `freeze_layer_num`    | 6           | Number of frozen layers in CLIP backbone      |
| `slice_framepos`      | 2           | Frame sampling position strategy              |
| `loose_type`          | (enabled)   | Indicates use of loose similarity matching    |
| `linear_patch`        | 2d          | Patch type for visual transformer             |
| `sim_header`          | seqLSTM     | Type of similarity module used                |
| `pretrained_clip_name`| ViT-B/32    | Pretrained CLIP model backbone                |

