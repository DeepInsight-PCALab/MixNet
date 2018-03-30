# Mixed Link Networks
MixNet: [[Arxiv](https://arxiv.org/abs/1802.01808)]

by Wenhai Wang, Xiang Li, Jian Yang, Tong Lu

IMAGINE Lab@National Key Laboratory for Novel Software Technology, Nanjing University.  
DeepInsight@PCALab, Nanjing University of Science and Technology.

## Requirements
* Install [PyTorch v0.2.0](http://pytorch.org/)
* Clone recursively
```
git clone --recursive https://github.com/DeepInsight-PCALab/MixNet.git
```
* Download the ImageNet dataset and move validation images to labeled subfolders
    * To do this, you can use the following script: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

## Training
### CIFAR-10
```
CUDA_VISIBLE_DEVICES=0 python cifar.py --dataset cifar10 --depth 100 --k1 12 --k2 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint checkpoints/cifar10/mixnet-100/
```

### ImageNet
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet.py -d ../imagenet/ -j 4 --arch mixnet105 --train-batch 200 --checkpoint checkpoints/imagenet/mixnet-105/
```

## Testing on ImageNet
```
CUDA_VISIBLE_DEVICES=0 python imagenet.py -d ../imagenet/ -j 4 --arch mixnet105 --test-batch 20 --pretrained pretrained/mixnet105.pth.tar --evaluate
```

## Results on CIFAR
| Model | Parameters | CIFAR-10 | CIFAR-100 | 
| - | - | - | - |
| MixNet-100 (k1 = 12, k2 = 12) | 1.5M | 4.19 | 21.12 |
| MixNet-250 (k1 = 24, k2 = 24) | 29.0M | 3.32 | 17.06 |
| MixNet-190 (k1 = 40, k2 = 40) | 48.5M | 3.13 | 16.96 |

## Results on ImageNet and Pretrained Models

| Method | Parameters | Top-1 error | Pretrained model |
| - | - | - | - |
| MixNet-105 (k1 = 32, k2 = 32) | 11.16M | 23.3 | [Download(43.2M)](https://pan.baidu.com/s/1q-LjwofEu2nM7feZClTA7w) |
| MixNet-121 (k1 = 40, k2 = 40) | 21.86M | 21.9 | [Download(84.3M)](https://pan.baidu.com/s/1wIzkO0UVIXd_BPx_lmT7_w) |
| MixNet-141 (k1 = 48, k2 = 48) | 41.07M | 20.4 | [Download(158.1M)](https://pan.baidu.com/s/1lYczUcAczhkQqpEwjZT66Q) |

## Citation
```
@article{wang2018mixed,  
  title={Mixed Link Networks},  
  author={Wang, Wenhai and Li, Xiang and Yang, Jian and Lu, Tong},  
  journal={arXiv preprint arXiv:1802.01808},  
  year={2018}  
}
```