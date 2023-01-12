"Dive into Deep Learning" is the book that helps me dive into this domain. As the objective of this book is to really play the code, I make this repo to record some personal experiments while reading and practising it.

* * *
This repo is to benchmark image classification algorithms.

* * * 
Benchmark
Network | Parameter | Dataset | Epoch | Device | Time cost(sec) | Batch size | Lr | Best test acc
Lenet | "120 | 382" | FashionMNIST | 10 | Tesla T4 | 26.7 | 256 | 0.1 | 0.8808
Lenet | "120 | 382" | FashionMNIST | 10 | Quadro RTX 5000 | 19.4 | 256 | 0.1 | 0.8623
Alexnet | "46 | 764 | 746" | FashionMNIST | 10 | Tesla T4 | 444.4 | 256 | 0.1 | 0.896
Alexnet | "46 | 764 | 746" | FashionMNIST | 10 | Quadro RTX 5000 | 264.4 | 256 | 0.1 | 0.8898
VGG11 | "128 | 806 | 154" | FashionMNIST | 10 | Tesla T4 | 4087.4 | 128 | 0.1 | 0.9284
VGG11 | "128 | 806 | 154" | FashionMNIST | 10 | Quadro RTX 5000 | 3106.5 | 256 | 0.1 | 0.9186
VGG13 | "128 | 990 | 666" | FashionMNIST | 10 | Tesla T4 | 6914.1 | 128 | 0.1 | 0.9266
VGG13 | "128 | 990 | 666" | FashionMNIST | 10 | Quadro RTX 5000 | 4044.5 | 128 | 0.1 | 0.9255
VGG16 | "134 | 300 | 362" | FashionMNIST | 10 | Quadro RTX 5000 | 7263.1 | 128 | 0.1 | 0.1
VGG19 | "139 | 610 | 058" | FashionMNIST | 10 | Quadro RTX 5000 | 5607.1 | 128 | 0.1 | 0.1

* * * 
Credit:
1. https://zh.d2l.ai/index.html
2. https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification
