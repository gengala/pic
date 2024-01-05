# Probabilistic Integral Circuits - PICs

This repository is the official implementation of [Probabilistic Integral Circuits](https://arxiv.org/abs/2310.16986).

## Datasets

- **MNIST-famility datasets**: all MNIST-famility datasets are available in [torchvision](https://pytorch.org/vision/stable/datasets.html), and automatically downloaded if needed.

- **PTB288**: download it [here](https://github.com/UCLA-StarAI/SparsePC/tree/main/datasets/ptbchar_288), and place it in `data/ptbchar_288`

- **Binary datasets (DEBD)**: download them [here](https://github.com/UCLA-StarAI/Density-Estimation-Datasets), and place them in `data/debd`

- **UCI datasets**: download non pre-processed datasets [here](https://zenodo.org/record/1161203#.Wmtf_XVl8eN) or pre-processed datasets [here](https://drive.google.com/file/d/1tUGEc1Dk2Cny1kG-Du3QYRuH5sDHLtys/view?usp=share_link), and place them in `data/UCI`

## Training PICs

```shell
python train_pic.py -ds mnist                   -bs 256 -nip 128 -int trapezoidal 
python train_pic.py -ds fashion_mnist           -bs 256 -nip 128 -int trapezoidal 
python train_pic.py -ds emnist -split mnist     -bs 256 -nip 128 -int trapezoidal 
python train_pic.py -ds emnist -split letters	-bs 256 -nip 128 -int trapezoidal 
python train_pic.py -ds emnist -split balanced	-bs 256 -nip 128 -int trapezoidal 
python train_pic.py -ds emnist -split byclass 	-bs 256 -nip 128 -int trapezoidal
```

## Training HCLTs

```shell
python train_hclt.py -ds mnist                  -bs 256 -hd 128
python train_hclt.py -ds fashion_mnist          -bs 256 -hd 128
python train_hclt.py -ds emnist -split mnist    -bs 256 -hd 128
python train_hclt.py -ds emnist -split letters	-bs 256 -hd 128
python train_hclt.py -ds emnist -split balanced	-bs 256 -hd 128
python train_hclt.py -ds emnist -split byclass 	-bs 256 -hd 128
```
