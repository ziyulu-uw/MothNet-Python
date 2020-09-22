"""
Loads the mnist dataset (the version in PMTK3 by Kevin Murphy, Machine Learning 2012),
then applies various preprocessing steps to reduce the number of pixels (each pixel will be a feature).
The 'receptive field' step destroys spatial relationships, so to reconstruct a
12 x 12 thumbnail (eg for viewing, or for CNN use) the active pixel indices can be embedded in a
144 x 1 col vector of zeros, then reshaped into a 12 x 12 image.
Modify the path for the mnist data file as needed.

Inputs:
  1. preprocessingParams = a class with attributes corresponding to relevant variables

Outputs:
  1. featureArray = n x m x 10 array. n = #active pixels, m = #digits from each class that will be used.
     The 3rd dimension gives the class, 0:9.
  2. activePixelInds: list of pixel indices to allow re-embedding into empty thumbnail for viewing.
  3. lengthOfSide: allows reconstruction of thumbnails given from the  feature vectors.
"""

"""
Preprocessing includes:
%   1. Load MNIST set.  
%   2. cropping and downsampling 
%   3. mean-subtract, make non-negative, normalize pixel sums
%   4. select active pixels (receptive field)
"""
import numpy as np

# from runMothLearnerOnReducedMnist import preprocessingParams
from extractMnistFeatureArray import extractMnistFeatureArray
from cropDownsampleVectorizeImageStack import cropDownsampleVectorizeImageStack
from averageImageStack import averageImageStack
from selectActivePixels import selectActivePixels


def generateDownsampledMnistSet(preprocessingParams):

    # extract the required images and classes:

    imageArray = extractMnistFeatureArray(preprocessingParams.classLabels, preprocessingParams.maxInd, 'train')
    # imageArray = numberClasses x numberImages x h x w (10 x 1000 x 28 x 28) 4-D array. the classes are ordered 0 to 9

    # crop, downsample, and vectorize the average images and the image stacks \
    # process each class in turn:
    for c in range(imageArray.shape[0]):
        thisStack = imageArray[c, :, :, :]
        thisFeatureMatrix = cropDownsampleVectorizeImageStack(thisStack,
                                                              preprocessingParams.crop, preprocessingParams.downsampleRate,
                                                              preprocessingParams.downsampleMethod)
        if c == 0: # get the dimensions of feature matrices to initialize featureArray
            featureArray = np.zeros((imageArray.shape[0], thisFeatureMatrix.shape[0], thisFeatureMatrix.shape[1]))

        featureArray[c, :, :] = thisFeatureMatrix

    # Subtract a mean image from all feature vectors, then make values non-negative:

    # a. Make an overall average feature vector, using the samples specified in 'indsToAverage'
    overallAve = np.zeros((featureArray.shape[1], 1))  # initialize col vector
    classAvesRaw = np.zeros((featureArray.shape[1], featureArray.shape[0]))

    for c in range(featureArray.shape[0]):
        thisClass = featureArray[c]
        # classAvesRaw[:, c] = averageImageStack(featureArray[c, :, preprocessingParams.indsToAverageGeneral])
        classAvesRaw[:, c] = averageImageStack(thisClass[:, preprocessingParams.indsToAverageGeneral])
        thisclassAvesRaw = np.atleast_2d(classAvesRaw[:, c]).transpose()  # reshape classAvesRaw[:, c] into a column vector
        overallAve = np.add(overallAve, thisclassAvesRaw)

    overallAve = overallAve / featureArray.shape[0]

    # b. Subtract this overallAve image from all images:
    featureArray = featureArray - np.tile(overallAve, (featureArray.shape[0], 1, featureArray.shape[2]))
    featureArray[featureArray < 0] = 0  # kill negative pixel values

    # c. Normalize each image so the pixels sum to the same amount:
    fSums = np.sum(featureArray, axis=1)
    fSumsMatrix = np.zeros(featureArray.shape)
    for i in range(featureArray.shape[0]):
        fSumsMatrix[i, :, :] = np.tile(fSums[i], (featureArray.shape[1],1))
    fNorm = np.divide(preprocessingParams.pixelSum*featureArray, fSumsMatrix)
    featureArray = fNorm
    # featureArray now consists of mean-subtracted, non-negative,
    # normalized (by sum of pixels) columns, each column a vectorized thumbnail. size = 10 x 144 x numDigitsPerClass

    lengthOfSide = featureArray.shape[1]  # save to allow sde_EM_evolution to print thumbnails.

    # d. Define a Receptive Field, ie the active pixels:
    # Reduce the number of features by getting rid of less-active pixels.
    # If we are using an existing moth then activePixelInds is already defined, so
    # we need to load the modelParams to get the number of features (since this is defined by the AL architecture):
    if preprocessingParams.useExistingConnectionMatrices:
        pass

    activePixelInds = selectActivePixels(featureArray[:, :, preprocessingParams.indsToCalculateReceptiveField],
                                         preprocessingParams.numFeatures, preprocessingParams.showAverageImages)
    featureArray = featureArray[:, activePixelInds, :]  # Project onto the active pixels
    # print(featureArray.shape)

    return featureArray, activePixelInds, lengthOfSide











