# Re-implementation of ConvNets on CIFAR-100 with PyTorch

Contact email: imdchan@yahoo.com

## Introduction

Here are some re-implementations of Convolutional Networks on CIFAR-100 dataset.

Note that the training set that consists of 50k training images was divided into 45k/5k train/val split. So I first made stratefied 10-fold split, resulting in the 'train_folds.csv'.

## Requirements

- A single TITAN RTX (24G memory) is used.

- Python 3.7+

- PyTorch 1.0+

## Usage

1. Clone this repository

        git clone https://github.com/longrootchen/cifar100-pytorch.git

2. Train a model, taking resnext29_16x64d as an example

        python -u train.py --work-dir ./experiments/resnext29_16x64d --resume ./experiments/resnext29_16x64d/checkpoints/last_checkpoint.pth
        
3. Evaluate a model, taking resnext29_16x64d as an example

        python -u eval.py --work-dir ./experiments/resnext29_16x64d --ckpt-name last_checkpoint.pth
        
        
## Results

| Error Rate (%)  | original paper | re-implementation |
| ----- | ----- | ----- |
| ResNeXt-29, 8x64d | 17.77 [1] |  |
| ResNeXt-29, 16x64d | 17.31 [1] | 18.75 |
| DenseNet-100-BC, k=12 | 22.27 [2] |  |
| DenseNet-250-BC, k=24 | 17.60 [2] |  |
| DenseNet-190-BC, k=40 | 17.18 [2] |  |
| SE-ResNet-101 | 23.85 [3] |  |
| SE-ResNet-164 | 21.31 [3] |  |

## References

[1] Saining Xie, Ross Girshick, Piotr Dollár, Zhouwen Tu, Kaiming He. Aggregated Residual Transformations for Deep Neural Networks. In CVPR, 2017.

[2] Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger. Densely Connected Convolutional Networks. In CVPR, 2017.

[3] Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu. Squeeze-and-Excitation Networks. In CVPR, 2018.
