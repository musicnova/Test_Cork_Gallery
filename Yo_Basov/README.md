Мой дневник по стажировке https://trello.com/b/18rcLAl6/design


Разминка для мозгов https://www.youtube.com/watch?v=hEerhyybTXs


Мануал по Stable Diffusion: https://pyimagesearch.com/2024/03/11/understanding-tasks-in-diffusers-part-2/


![photo1](https://github.com/NeuronsUII/Cork_Gallery_g1/assets/29410375/09da5a4c-e2d9-45f2-8ba6-b8857004dd1e)


https://github.com/automl/auto-sklearn/issues/1424


https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-mmcv


upernet_beit-large_fp16_8x1_640x640_160k_ade20k.py
```
_base_ = [
    '../_base_/models/upernet_beit.py', '../_base_/datasets/ade20k_640x640.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_320k.py'
]

model = dict(
    pretrained='pretrain/beit_large_patch16_224_pt22k_ft22k.pth',
    backbone=dict(
        type='BEiT',
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        mlp_ratio=4,
        qv_bias=True,
        init_values=1e-6,
        drop_path_rate=0.2,
        out_indices=[7, 11, 15, 23]),
    neck=dict(embed_dim=1024, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        in_channels=[1024, 1024, 1024, 1024], num_classes=150, channels=1024),
    auxiliary_head=dict(in_channels=1024, num_classes=150),
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(426, 426)))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=2e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.95))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=3000,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

data = dict(samples_per_gpu=1)
optimizer_config = dict(
    type='GradientCumulativeFp16OptimizerHook', cumulative_iters=2)

fp16 = dict()
```


install.txt
```
Numpy version 2.0.0
MMCV Version 1.6.0
MMEngine version: 0.10.4
MMSegmentation version: 0.30.0

conda create --name probki python=3.9
GPU: conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
CPU: conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install mmcv==1.6.0
pip install numpy==2.0.0
pip install mmengine==0.10.4
pip install mmsegmentation=0.30.0
```
