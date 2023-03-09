Dive into Deep Learning is the book that helps me dive into this domain. As the objective of this book is to really play the code, I make this repo to record some personal experiments while reading and practising it.

This repo is to benchmark image classification algorithms.

* * * 
Benchmark
| Network | #Parameter | Dataset | Epoch | Device | Time cost(sec) | Batch size | Lr | Best test acc | 
|  ----  | ----  | ----  | ----  | ----  | ----  | ----  | ----  | ----  |
| Lenet | 120382 | FashionMNIST | 10 | Tesla T4 | 26.7 | 256 | 0.1 | 0.8808 |
| Lenet | 120382 | FashionMNIST | 10 | Quadro RTX 5000 | 19.4 | 256 | 0.1 | 0.8623 |
| Alexnet | 46764746 | FashionMNIST | 10 | Tesla T4 | 444.4 | 256 | 0.1 | 0.896 |
| Alexnet | 46764746 | FashionMNIST | 10 | Quadro RTX 5000 | 264.4 | 256 | 0.1 | 0.8898 |
| VGG11 | 128806154 | FashionMNIST | 10 | Tesla T4 | 4087.4 | 128 | 0.1 | 0.9284 |
| VGG11 | 128806154 | FashionMNIST | 10 | Quadro RTX 5000 | 3106.5 | 256 | 0.1 | 0.9186 |
| VGG13 | 128990666 | FashionMNIST | 10 | Tesla T4 | 6914.1 | 128 | 0.1 | 0.9266 |
| VGG13 | 128990666 | FashionMNIST | 10 | Quadro RTX 5000 | 4044.5 | 128 | 0.1 | 0.9255 |
| VGG16 | 134300362 | FashionMNIST | 10 | Quadro RTX 5000 | 7263.1 | 128 | 0.1 | 0.9277 |
| VGG19 | 139610058 | FashionMNIST | 10 | Quadro RTX 5000 | 5607.1 | 128 | 0.1 | 0.9276 |
| GoogleNet | 10327758 | FashionMNIST | 10 | Quadro RTX 5000 | 1168.9 | 128 | 0.1| 0.8831 |
| resnet18 | 11,175,370 | FashionMNIST | 10 | Quadro RTX 5000 | 972.4 | 128 | 0.1 | 0.9176 | 
| resnet34 | 21,283,530 | FashionMNIST | 10 | Quadro RTX 5000 | 1587.0 | 128 | 0.1 | 0.9198
| resnet50 | 23,522,250 | FashionMNIST | 10 | Quadro RTX 5000 | 3138.0 | 128 | 0.1 | 0.9065 | 
| resnet101 | 42,514,378 | FashionMNIST | 10 | Quadro RTX 5000 | 5169.2 | 64 | 0.1 | 0.9201 | 
| resnet152 | 120,337,610 | FashionMNIST | 10 | Quadro RTX 5000 | 18862.0 | 32 | 0.1 | 0.9199 | 
| resnext50_2x40d | 25,418,728 | FashionMNIST | 10 | Quadro RTX 5000 | 3889.7 | 32 | | 0.1 | 0.9194 | 
| resnext50_4x24d | 25,286,696 | FashionMNIST | 10 | Quadro RTX 5000 | 4092.9 | 32 | | 0.1 | 0.9248 |
| resnext50_8x14d | 25,596,744 | FashionMNIST | 10 | Quadro RTX 5000 | 4551.3 | 32 | | 0.1 | 0.922 |
| resnext50_32x4d | 22,994,122 | FashionMNIST | 10 | Quadro RTX 5000 | 4521.1 | 32 | | 0.1 | 0.9231 |
| resnext101_2x40d | 44,450,024 | FashionMNIST | 10 | Quadro RTX 5000 | 6334.2 | 32 | | 0.1 | 0.9209 |
| resnext101_4x24d | 44,357,160 | FashionMNIST | 10 | Quadro RTX 5000 | 6708.1 | 32 | | 0.1 | 0.9298 |
| resnext101_8x14d | 45,098,056 | FashionMNIST | 10 | Quadro RTX 5000 | 7477.0 | 32 | | 0.1 | 0.9237 |
| resnext101_32x8d | 86,756,554 | FashionMNIST | 10 | Quadro RTX 5000 | 13108.5 | 32 | | 0.1 | 0.9167 |
| MobileNetV1 | 3,227,562 | FashionMNIST | 10 | Quadro RTX 5000 | 1144.9 | 128 | 0.1 | 0.9252 |
| MobileNetV2 | 2,236,106 | FashionMNIST | 10 | Quadro RTX 5000 | 1293.3 | 128 | 0.1 | 0.9208 |
| MobileNetV3_L | 4,214,554 | FashionMNIST | 10 | Quadro RTX 5000 | 1902.3 | 128 | 0.1 | 0.9169 |
| MobileNetV3_S | 1,527,818 | FashionMNIST | 10 | Quadro RTX 5000 | 730.5 | 128 | 0.1 | 0.9121 |
| shufflenet_v2_x0_5 | 351,610 | FashionMNIST | 10 | Quadro RTX 5000 | 362.8 | 128 | 0.1 | 0.9097 |
| shufflenet_v2_x1_0 | 351,610 | FashionMNIST | 10 | Quadro RTX 5000 | 746.2 | 128 | 0.1 | 0.9167 |
| shufflenet_v2_x1_5 | 351,610 | FashionMNIST | 10 | Quadro RTX 5000 | 1145.4 | 128 | 0.1 | 0.9204 |
| shufflenet_v2_x2_0 | 351,610 | FashionMNIST | 10 | Quadro RTX 5000 | 1565.6 | 128 | 0.1 | 0.9219 |
| efficientnet_b0 | 4,019,782 | FashionMNIST | 10 | Quadro RTX 5000 | 2990.4 | 64 | 0.1 | 0.9314 |
| efficientnet_b1 | 6,525,418 | FashionMNIST | 10 | Quadro RTX 5000 | 4181.8 | 64 | 0.1 | 0.9335 |
| efficientnet_b2 | 7,714,508 | FashionMNIST | 10 | Quadro RTX 5000 | 4482.5 | 64 | 0.1 | 0.9343 |
| efficientnet_b3 | 10,710,882 | FashionMNIST | 10 | Quadro RTX 5000 | 6685.9 | 64 | 0.1 | 0.9314 |
| efficientnet_b4 | 17,565,682 | FashionMNIST | 10 | Quadro RTX 5000 | 8606.5 | 32 | 0.1 | 0.9395 |
| efficientnet_b5 | 28,360,410 | FashionMNIST | 10 | Quadro RTX 5000 | 13037.7 | 32 | 0.1 | 0.9378 |
| efficientnet_b6 | 40,757,746 | FashionMNIST | 10 | Quadro RTX 5000 | 18381.4 | 16 | 0.1 | 0.9362 |
| efficientnet_b7 | 63,811,418 | FashionMNIST | 10 | Quadro RTX 5000 | 24998.1 | 16 | 0.1 | 0.9314 |


* * * 
Credit:
1. https://zh.d2l.ai/index.html
2. https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification
