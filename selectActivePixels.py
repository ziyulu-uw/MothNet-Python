"""
Select the most active pixels, considering all class average images, to use as features.
Inputs:
   1. featureArray: 3-D array nC x nF x nS, where nC = number of classes, nF = # of features, nS = # samples per class.
       As created by generateDwnsampledMnistSet.py
   2. numFeatures: The number of active pixels to use (these form the receptive field).
   3. showImages:  1 means show average class images, 0 = don't show.
Output:
  1. activePixelInds: 1 x nF vector of indices to use as features.
      Indices are relative to the vectorized thumbnails (so between 1 and 144).
"""
import numpy as np
from averageImageStack import averageImageStack
from showFeatureArrayThumbnails import showFeatureArrayThumbnails


def selectActivePixels(featureArray, numFeatures, showImages):
    # make a classAves matrix, each col a class ave 0 to 9, and add a col for the overallAve
    numPerClass = featureArray.shape[2]
    cA = np.zeros((featureArray.shape[1], featureArray.shape[0] + 1))

    for i in range(featureArray.shape[0]):
        temp = np.zeros((featureArray.shape[1], featureArray.shape[2]))
        temp[:,:] = featureArray[i, :, :]
        cA[:, i] = averageImageStack(temp)

    # last col = overall average image:
    cA[:,-1] = np.sum(cA[:, 0:cA.shape[1]-1], 1)/(cA.shape[1]-1)

    # normed version. Do not rescale the overall average:
    z = np.amax(cA, axis=0)
    z_ = np.append(z[0:-1], [1])
    caNormed = np.divide(cA, np.tile(z_, [cA.shape[0], 1]))

    # select most active 'numFeatures' pixels:
    this = cA[:, 0:-1]
    thisLogical = np.zeros(this.shape)
    vals = np.sort(this, axis=None)[::-1]  # all the pixel values from all the class averages, in descending order

    # start selecting the highest-valued pixels:
    stop = 0
    while not stop:
        thresh = np.amax(vals)
        thisLogical[this >= thresh] = 1
        activePixels = np.sum(thisLogical, axis=1)  # sum the rows
        if np.count_nonzero(activePixels) >= numFeatures:
            stop = 1  # we have enough pixels
        vals = vals[vals < thresh]  # peel off the value(s) just used

    activePixelInds = np.nonzero(activePixels > 0)[0]

    if showImages:
        # plot the normalized classAves pre-ablation:
        normalize = 0
        titleStr = 'class aves, all pixels'
        showFeatureArrayThumbnails(caNormed, caNormed.shape[1], normalize, titleStr)

        # look at active pixels of the classAves, ie post-ablation:
        normalize = 0
        caActiveOnly = np.zeros(caNormed.shape)
        caActiveOnly[activePixelInds, :] = caNormed[activePixelInds, :]
        titleStr = 'class aves, active pixels only'
        showFeatureArrayThumbnails(caActiveOnly, caActiveOnly.shape[1], normalize, titleStr)

    return activePixelInds




