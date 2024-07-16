# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 12:55:05 2024

@author: vk8889
"""
import numpy as np

def bin2D(array, factor, dtype=np.float64):
    """
    Bin a 2D ndarray by binfactor.

    Args:
        array (2D numpy array):
        factor (int): the binning factor
        dtype (numpy dtype): datatype for binned array. default is numpy default for
            np.zeros()

    Returns:
        the binned array
    """
    x, y = array.shape
    binx, biny = x // factor, y // factor
    xx, yy = binx * factor, biny * factor

    # Make a binned array on the device
    binned_ar = np.zeros((binx, biny), dtype=dtype)
    array = array.astype(dtype)

    # Collect pixel sums into new bins
    for ix in range(factor):
        for iy in range(factor):
            binned_ar += array[0 + ix : xx + ix : factor, 0 + iy : yy + iy : factor]
    return binned_ar


def Threshold_defect_pixels (data, threshold):
    position = np.argwhere(data[0]>threshold)
    data[:,position[:,0],position[:,1]]=0 
    return data

d = np.zeros((data.shape[0],data.shape[1]//2,data.shape[2]//2))
for i in range(data.shape[0]):
    d[i] = bin2D(data[i],2, dtype=data.dtype)