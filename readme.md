# CLIP_video 
This project is based on CLIP4Clip, a framework for video-text retrieval using CLIP.
We adapted the original version to support single-GPU training and evaluation for more accessible development and experimentation

## Key Changes:
1. Removed all dependencies on distributed training (torch.distributed)
2. Eliminated local_rank, world_size, and related parameters
3. Integrated tqdm progress bars for clearer training feedback in notebooks
4. Easy to run in Colab or on local machines with only one GPU
   
## Demo Notebook:
We provide an interactive notebook:
**CLIP_video.ipynb** <br>
This notebook demonstrates the full workflow — from loading MSRVTT features to training and evaluation — using a subset of 9k video-text pairs for quick testing.
