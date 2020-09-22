"""
Extract a subset of the samples from each class, convert the images to doubles on [0 1], and return a 4-D array:
  dim 1: class
  dim 2: indexes images within a class
  dim 3, 4 = image

Inputs:
  classesToUse = vector of the classes (digits) you want to extract
  maxInd = number of images you want from each class.
  trainOrTest = 'train' or 'test'. Determines which images you draw from
     (since we only need a small subset, one or the other is fine).

Outputs:
  imageArray = numberClasses x numberImages x h x w  4-D array
  labels = numberClasses x numberImages matrix. Each row is a class label.
"""
import numpy as np


def extractMnistFeatureArray(classesToUse, maxInd, trainOrTest):

    trainDict = np.load('classfiedTrainMnist.npy', allow_pickle=True)
    testDict = np.load('classfiedTestMnist.npy', allow_pickle=True)

    # get some dimensions:
    im = trainDict.item().get(0)[0,:,:]
    h, w = im.shape

    # initialize outputs:
    imageArray = np.zeros((len(classesToUse), maxInd, h, w))
    labels = np.zeros((len(classesToUse), maxInd))

    # process each class in turn:
    for c in classesToUse:

        if trainOrTest == 'train':
            if c == 0:
                t = trainDict.item().get(0)
            if c == 1:
                t = trainDict.item().get(1)
            if c == 2:
                t = trainDict.item().get(2)
            if c == 3:
                t = trainDict.item().get(3)
            if c == 4:
                t = trainDict.item().get(4)
            if c == 5:
                t = trainDict.item().get(5)
            if c == 6:
                t = trainDict.item().get(6)
            if c == 7:
                t = trainDict.item().get(7)
            if c == 8:
                t = trainDict.item().get(8)
            if c == 9:
                t = trainDict.item().get(9)

        if trainOrTest == 'test':
            if c == 0:
                t = testDict.item().get(0)
            if c == 1:
                t = testDict.item().get(1)
            if c == 2:
                t = testDict.item().get(2)
            if c == 3:
                t = testDict.item().get(3)
            if c == 4:
                t = testDict.item().get(4)
            if c == 5:
                t = testDict.item().get(5)
            if c == 6:
                t = testDict.item().get(6)
            if c == 7:
                t = testDict.item().get(7)
            if c == 8:
                t = testDict.item().get(8)
            if c == 9:
                t = testDict.item().get(9)
        # now we have the correct image stack:
        # convert to double:
        t = t.astype('float64')
        t = t/256
        imageArray[c, 0:maxInd, :, :] = t[0:maxInd, :, :]

    return imageArray
