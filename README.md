# Mixed Link Networks
MixNet: [[Arxiv](https://arxiv.org/abs/1802.01808)]

by Wenhai Wang, Xiang Li, Jian Yang, Tong Lu

IMAGINE Lab@National Key Laboratory for Novel Software Technology, Nanjing University.  
DeepInsight@PCALab, Nanjing University of Science and Technology.

## Install
* Install [PyTorch v0.2.0](http://pytorch.org/)
* Clone recursively
```
git clone --recursive https://github.com/DeepInsight-PCALab/MixNet.git
```

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

## Pretrained models

| Method | Top-1 error | Pretrained model |
| - | - | - | 
| MixNet-105 | 23.3 | [Download(https://pan.baidu.com/s/1q-LjwofEu2nM7feZClTA7w)] |
| MixNet-121 | 21.9 | [Download(https://pan.baidu.com/s/1wIzkO0UVIXd_BPx_lmT7_w)] |
| MixNet-141 | 20.4 | [Download(https://pan.baidu.com/s/1lYczUcAczhkQqpEwjZT66Q)] |

## Citation
```
@article{wang2018mixed,  
  title={Mixed Link Networks},  
  author={Wang, Wenhai and Li, Xiang and Yang, Jian and Lu, Tong},  
  journal={arXiv preprint arXiv:1802.01808},  
  year={2018}  
}
```