"""
Loads the mnist dataset and divides both training and test images into 10 classes.
The classfied training images are organized in the dictionary trainDict and are saved in the file classfiedTrainMnist.npy,
and the classfied test images are organized in the dictionary testDict are saved in the file classfiedTestMnist.npy.
"""
import numpy as np
from tensorflow.keras.datasets import mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

trainDict = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}

for i in range(len(training_labels)):

    if training_labels[i] == 0:
        trainDict[0].append(training_images[i, :, :])

    elif training_labels[i] == 1:
        trainDict[1].append(training_images[i, :, :])

    elif training_labels[i] == 2:
        trainDict[2].append(training_images[i, :, :])

    elif training_labels[i] == 3:
        trainDict[3].append(training_images[i, :, :])

    elif training_labels[i] == 4:
        trainDict[4].append(training_images[i, :, :])

    elif training_labels[i] == 5:
        trainDict[5].append(training_images[i, :, :])

    elif training_labels[i] == 6:
        trainDict[6].append(training_images[i, :, :])

    elif training_labels[i] == 7:
        trainDict[7].append(training_images[i, :, :])

    elif training_labels[i] == 8:
        trainDict[8].append(training_images[i, :, :])

    elif training_labels[i] == 9:
        trainDict[9].append(training_images[i, :, :])

    else:
        print('Something is wrong')

for i in range(len(trainDict)):
    trainDict[i] = np.array(trainDict[i])

testDict = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}

for i in range(len(test_labels)):

    if test_labels[i] == 0:
        testDict[0].append(test_images[i, :, :])

    elif test_labels[i] == 1:
        testDict[1].append(test_images[i, :, :])

    elif test_labels[i] == 2:
        testDict[2].append(test_images[i, :, :])

    elif test_labels[i] == 3:
        testDict[3].append(test_images[i, :, :])

    elif test_labels[i] == 4:
        testDict[4].append(test_images[i, :, :])

    elif test_labels[i] == 5:
        testDict[5].append(test_images[i, :, :])

    elif test_labels[i] == 6:
        testDict[6].append(test_images[i, :, :])

    elif test_labels[i] == 7:
        testDict[7].append(test_images[i, :, :])

    elif test_labels[i] == 8:
        testDict[8].append(test_images[i, :, :])

    elif test_labels[i] == 9:
        testDict[9].append(test_images[i, :, :])

    else:
        print('Something is wrong')

for i in range(len(testDict)):
    testDict[i] = np.array(testDict[i])

np.save('classfiedTrainMnist.npy',trainDict, allow_pickle=True)
np.save('classfiedTestMnist.npy',testDict, allow_pickle=True)

# from matplotlib import pyplot as plt
# trainDict = np.load('classfiedTrainMnist.npy', allow_pickle=True)
# plotData = trainDict.item().get(2)[0,:,:]
# plt.imshow(plotData)
# plt.show()
