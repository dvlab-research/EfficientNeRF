## The official code for "[EfficientNeRF: Efficient Neural Radiance Fields](https://arxiv.org/abs/2206.00878)" in CVPR2022.

### Environment (Tested)
- Ubuntu 18.04
- Python 3.7
- CUDA 11.x
- Pytorch 1.9.1
- Pytorch-Lightning 1.6.4

### Install via Anaconda
```
$ conda create -n EfficientNeRF python=3.8
$ conda activate EfficientNeRF
$ pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install -r requirements.txt
```

### Training
```
$ DATA_DIR=/path/to/lego
$ python train.py \
   --dataset_name blender \
   --root_dir $DATA_DIR \
   --N_samples 128 \
   --N_importance 5 --img_wh 800 800 \
   --num_epochs 16 --batch_size 4096 \
   --optimizer radam --lr 2e-3 \
   --lr_scheduler poly \
   --coord_scope 3.0 \
   --warmup_step 5000\
   --sigma_init 30.0 \
   --weight_threashold 1e-5 \
   --exp_name lego_coarse128_fine5_V384
```

### Visualization
```
$ tensorboard --logdir=./logs
```

### Question
- Q1. Different hyperparameters from the original paper
* A1. There are many combinations between these hyperparameters. You are free to balance the training speed and accuracy by modify them. 
- Q2. When will NeRF-Tree released?
* A2. Hard to say a specific date. The data structure NeRF-Tree is closed to Octree.

### Progress
More scenes and applications will be suported soon. Stay tune!

### Acknowledgement
Our initial code was borrowed from 
- [nerf-pl:https://github.com/kwea123/nerf_pl](https://github.com/kwea123/nerf_pl)

### Citation
If you find our code or paper helps, please cite our paper:
```
@InProceedings{Hu_2022_CVPR,
    author    = {Hu, Tao and Liu, Shu and Chen, Yilun and Shen, Tiancheng and Jia, Jiaya},
    title     = {EfficientNeRF  Efficient Neural Radiance Fields},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {12902-12911}
}
```

