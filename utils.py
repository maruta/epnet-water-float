import numpy as np
import torch

def pHankel(data, origin, hp, sample_size):
    mat = np.zeros((sample_size, data[1].size*hp))
    n = data[1].size
    for m in range(hp):
        mat[:, n*m:n*(m+1)] = data[origin-hp+m:origin-hp+m+sample_size, :]
    return mat


def fHankel(data, origin, hf, sample_size):
    mat = np.zeros((sample_size, data[1].size*hf))
    n = data[1].size
    for m in range(hf):
        mat[:, n*m:n*(m+1)] = data[origin+m:origin+m+sample_size, :]
    return mat

class DumbDataLoader(object):
    """
    Simple data loader for when GPU memory is sufficient to store Hankel matrices
    """

    def __init__(self, Up, Yp, Uf, Yf, batch_size=512, shuffle=True):
        self.Up = Up
        self.Yp = Yp
        self.Uf = Uf
        self.Yf = Yf
        self.batch_size = batch_size
        self.i = 0
        self.len = len(Up)
        self.shuffle = shuffle
        if self.shuffle:
            self.shuffle_data()

    def __iter__(self):
        return self

    def shuffle_data(self):
        idx = torch.randperm(self.len)
        self.Up = self.Up[idx]
        self.Yp = self.Yp[idx]
        self.Uf = self.Uf[idx]
        self.Yf = self.Yf[idx]

    def __next__(self):
        if self.i+self.batch_size > self.len:
            self.i = 0
            self.shuffle_data()
            raise StopIteration()
        ret = (
            self.Up[self.i:self.i+self.batch_size],
            self.Yp[self.i:self.i+self.batch_size],
            self.Uf[self.i:self.i+self.batch_size],
            self.Yf[self.i:self.i+self.batch_size],
        )
        self.i += self.batch_size
        return ret
