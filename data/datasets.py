from torchvision.datasets import MNIST, FashionMNIST, EMNIST
from typing import Optional
import numpy as np
import torch
import csv


DEBD_DATASETS = ['nltcs', 'msnbc', 'kdd', 'plants', 'baudio', 'jester', 'bnetflix', 'accidents', 'tretail',
                 'pumsb_star', 'dna', 'kosarek', 'msweb', 'book', 'tmovie', 'binarized_mnist', 'cwebkb', 'cr52',
                 'c20ng', 'bbc', 'ad']

MNIST_DATASETS = ['mnist', 'fashion_mnist', 'emnist']

UCI_DATASETS = ['bsds300', 'gas', 'miniboone', 'power', 'hepmass']


def load_dataset(
    ds_name: str,
    return_torch: Optional[bool] = True,
    split: Optional[str] = 'mnist',
    flatten: Optional[bool] = True,
    transform=None
):
    if ds_name in DEBD_DATASETS:
        return load_debd(ds_name, return_torch)
    elif ds_name in UCI_DATASETS:
        return load_uci(ds_name, return_torch)
    elif ds_name in MNIST_DATASETS:
        return load_mnist(ds_name, split, flatten, transform)
    elif ds_name == 'ptb288':
        return load_ptb288()
    else:
        raise Exception('Dataset %s not found' % ds_name)


def load_debd(
    ds_name: str,
    return_torch: Optional[bool] = True,
    dtype: str = 'int32'
):
    assert ds_name in DEBD_DATASETS
    train_path = './data/debd/%s/%s.train.data' % (ds_name, ds_name)
    valid_path = './data/debd/%s/%s.valid.data' % (ds_name, ds_name)
    test_path = './data/debd/%s/%s.test.data' % (ds_name, ds_name)
    reader = csv.reader(open(train_path, 'r'), delimiter=',' if ds_name != 'binarized_mnist' else ' ')
    train = np.array(list(reader)).astype(dtype)
    reader = csv.reader(open(test_path, 'r'), delimiter=',' if ds_name != 'binarized_mnist' else ' ')
    test = np.array(list(reader)).astype(dtype)
    reader = csv.reader(open(valid_path, 'r'), delimiter=',' if ds_name != 'binarized_mnist' else ' ')
    valid = np.array(list(reader)).astype(dtype)
    if return_torch:
        train, valid, test = torch.Tensor(train), torch.Tensor(valid), torch.Tensor(test)
    return train, valid, test


def load_uci(
    ds_name: str,
    return_torch: Optional[bool] = True
):
    # https://github.com/conormdurkan/autoregressive-energy-machines/blob/master/pytorch/utils/uciutils.py
    assert ds_name in UCI_DATASETS
    train = np.load('./data/UCI/%s/train.npy' % ds_name)
    valid = np.load('./data/UCI/%s/valid.npy' % ds_name)
    test = np.load('./data/UCI/%s/test.npy' % ds_name)
    if return_torch:
        train, valid, test = torch.Tensor(train), torch.Tensor(valid), torch.Tensor(test)
    return train, valid, test


def load_mnist(
    ds_name: Optional[str] = 'mnist',
    split: Optional[str] = 'mnist',
    flatten: Optional[bool] = True,
    transform=None,
    dtype: torch.dtype = torch.int64
):
    assert ds_name in MNIST_DATASETS
    if ds_name == 'mnist':
        train = MNIST(root="./data/", train=True, download=True, transform=transform).data.to(dtype=dtype)
        test = MNIST(root="./data/", train=False, download=True, transform=transform).data.to(dtype=dtype)
    elif ds_name == 'fashion_mnist':
        train = FashionMNIST(root="./data/", train=True, download=True, transform=transform).data.to(dtype=dtype)
        test = FashionMNIST(root="./data/", train=False, download=True, transform=transform).data.to(dtype=dtype)
    else:
        assert split in ['mnist', 'letters', 'balanced', 'byclass']
        train = EMNIST(root="./data/", split=split, train=True, download=True, transform=transform).data.to(dtype=dtype)
        test = EMNIST(root="./data/", split=split, train=False, download=True, transform=transform).data.to(dtype=dtype)
    if flatten:
        train = train.flatten(1)
        test = test.flatten(1)
    return train, test


def load_ptb288(
    dtype: str = 'int32',
    return_torch: Optional[bool] = True
):
    train = np.array(list(csv.reader(open('./data/ptbchar_288/ptbchar_288.train.data', 'r')))).astype(dtype)
    test = np.array(list(csv.reader(open('./data/ptbchar_288/ptbchar_288.test.data', 'r')))).astype(dtype)
    valid = np.array(list(csv.reader(open('./data/ptbchar_288/ptbchar_288.valid.data', 'r')))).astype(dtype)
    if return_torch:
        train, valid, test = torch.Tensor(train), torch.Tensor(valid), torch.Tensor(test)
    return train, valid, test


def show_picture():
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    train, test = load_mnist(ds_name='emnist', split='byclass')
    print(train.shape, test.shape)
    plt.imshow(train[np.random.randint(len(train))].view(28, 28).numpy(), cmap='gray')
