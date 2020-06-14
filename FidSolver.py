# Jack12

import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

from PIL import Image

import pickle
from inception import InceptionV3  
from fid_score import calculate_frechet_distance

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x
    


class FidSolver():

    def __init__(self, dims, batch, mu2_path='./ffhq_mu512.npy', sigma2_path='ffhq_sigme512.py' ):
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx])
        self.dims = dims
        self.batch = batch

        self.m2 = np.load(mu2_path)
        self.s2 = np.load(sigma2_path)

    def set_arr(self, length):
        self.pred_arr = np.array((length, self.dims))

    def cal_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    def cal_get_pred(self, idx, input):
        pred = self.model(input)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        start = idx
        end = idx + self.batch
        self.pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)


    def get_frechet(self):

        m1 = np.mean(self.pred_arr, axis=0)
        s1 = np.cov(self.pred_arr, rowvar=False)

        fid_value = calculate_frechet_distance(m1, s1, self.m2, self.s2)

        return fid_value