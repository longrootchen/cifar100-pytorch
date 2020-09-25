# Re-implementation of ConvNets on CIFAR-100 with PyTorch

Contact email: imdchan@yahoo.com

## Introduction

Here are some re-implementations of Convolutional Networks on CIFAR-100 dataset.

## Requirements

- Python 3.7+

- PyTorch 1.0+

## Usage

1. Clone this repository

        git clone https://github.com/longrootchen/cifar100-pytorch.git

2. Train a model, taking resnext29_16x64d as an example

        python -u train.py --work-dir ./experiments/resnext29_16x64d
        
3. Evaluate a model, taking resnext29_16x64d as an example

        python -u eval.py --arch resnext29_16x64d --num-classes 100 --ckpt-path ./experiments/resnet20/checkpoints/last_checkpoint.pth --gpu cuda:0
        
        
## Results

| Error Rate (%)  | original paper | re-implementation |
| ----- | ----- | ----- |
| ResNeXt-29, 8x64d | 17.77 [1] |  |
| ResNeXt-29, 16x64d | 17.31 [1] |  |
| DenseNet-40, k=12 | 24.42 [2] |  |
| DenseNet-100, k=12 | 20.20 [2] |  |
| DenseNet-100, k=24 | 19.25 [2] |  |
| DenseNet-100-BC, k=12 | 22.27 [2] |  |
| DenseNet-250-BC, k=24 | 17.60 [2] |  |
| DenseNet-190-BC, k=40 | 17.18 [2] |  |


## References

[1] Saining Xie, Ross Girshick, Piotr Dollár, Zhouwen Tu, Kaiming He. Aggregated Residual Transformations for Deep Neural Networks. In CVPR, 2017.

[2] Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger. Densely Connected Convolutional Networks. In CVPR, 2017.
