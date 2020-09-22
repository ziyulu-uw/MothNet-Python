"""
For each image in a stack of images: Crop, then downsample, then make into a col vector.
Inputs:
  1. imStack = numImages x height x width array
  2. cropVal = number of pixels to shave off each side. can be a scalar or a
      1 x 4 vector: top, bottom, left, right.
  3. downsampleVal = amount to downsample
  4. downsampleMethod: if 0, do downsampling by summing square patches. If 1, use bicubic interpolation.
Output:
  1. imColArray = a x numImages matrix, where a = number of pixels in the cropped and downsampled images

"""
import numpy as np
import cv2


def cropDownsampleVectorizeImageStack(imStack, cropVal, downsampleVal, downsampleMethod):

    if isinstance(cropVal, int):
        cropVal = cropVal*np.ones(4)

    if len(imStack.shape) == 3:
        z, h, w = imStack.shape

    else:
        h, w = imStack.shape
        z = 1

    width = np.arange(cropVal[2], w-cropVal[3])
    height = np.arange(cropVal[0], h-cropVal[1])

    pixelNum = (len(width)/downsampleVal)*(len(height)/downsampleVal)
    imColArray = np.zeros((int(pixelNum), z))

    # crop, downsample, vectorize the thumbnails one-by-one:
    for s in range(z):
        t = imStack[s, :, :]
        t = t[int(min(height)):int(max(height))+1, int(min(width)):int(max(width))+1]
        d = downsampleVal
        # to downsample, do bicubic interp or sum the blocks:
        if downsampleMethod == 1: # bicubic
            t2 = cv2.resize(t, (0,0), fx=1/downsampleVal, fy=1/downsampleVal, interpolation=cv2.INTER_CUBIC)
        else:  # downsampleMethod == 0: sum 2 x 2 blocks
            t2 = np.zeros((int(len(height)/d), int(len(width)/d)))
            for i in range(int(len(height)/d)):
                for j in range(int(len(width)/d)):
                    b = t[int(i * d):int((i + 1) * d), int(j * d):int((j + 1) * d)]
                    t2[i, j] = np.sum(b)

        t2 = t2/np.amax(t2)  # normalize to [0,1]
        t2 = t2.flatten('F')
        imColArray[:, s] = t2

    return imColArray

