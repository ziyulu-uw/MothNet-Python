"""
Show thumbnails of inputs used in the experiment.
Inputs:
  1. featureArray = either 3-D (dim 1 = class, dim 2 = cols of features, dim 3 = within class samples)
                        or 2-D (dim 1 = cols of features, dim 2 = within class samples, no dim 3)
  2. numPerClass = how many of the thumbnails from each class to show.
  3. normalize = 1 if you want to rescale thumbs to [0 1], 0 if you don't
  4. titleString = string
"""
import numpy as np
from matplotlib import pyplot as plt


def showFeatureArrayThumbnails(featureArray, numPerClass, normalize, titleString):
    # bookkeeping: change dim if needed:
    if len(featureArray.shape) == 2:
        f = np.zeros((1, featureArray.shape[0], featureArray.shape[1]))
        f[0, :, :] = featureArray
        featureArray = f

    nC = featureArray.shape[0]
    total = nC*numPerClass
    numRows = np.ceil(np.sqrt(total/2))
    numCols = np.ceil(np.sqrt(total * 2))

    fig = plt.figure()
    for c in range(nC):
        for i in range(numPerClass):
            col = numPerClass*c + i
            thisInput = featureArray[c, :, i]
            # show the thumbnail of the input:
            if normalize:
                thisInput = thisInput / np.amax(thisInput)

            ax = fig.add_subplot(numRows, numCols, col+1)
            ax.axis('off')
            plt.imshow(np.reshape(thisInput,(int(np.sqrt(len(thisInput))), int(np.sqrt(len(thisInput))))).transpose())  # Assumes square thumbnails

    plt.title(titleString)
    plt.show()

