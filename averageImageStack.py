"""
Average a stack of images:
inputs:
  1. imStack = 2-d matrix (images to average are column vectors)
Output:
  1. averageImage: column vector
"""

import numpy as np


def averageImageStack(imStack):

    aveIm = np.zeros(imStack.shape[0])
    for i in range(imStack.shape[1]):
        aveIm = np.add(aveIm, imStack[:, i])

    # normalize
    averageIm = aveIm/imStack.shape[1]

    return averageIm





