"""
runMothLearnerOnReducedMnist:

Main script to train a moth brain model on a crude (downsampled) MNIST set.
The moth can be generated from template or loaded complete from file.

Preparation:
  0. Run loadMnist.py to generate training and testing images!!
  1. Modify specifyModelParamsMnist.py with the desired parameters for
     generating a moth (ie neural network), or specify a pre-existing 'modelParams' file to load.
  2. Edit USER ENTRIES
Order of events:
  1. Load and pre-process dataset
  Within the loop over number of simulations:
      2. Select a subset of the dataset for this simulation (only a few samples are used).
      3. Create a moth (neural net). Either select an existing moth file, or generate a new moth in 2 steps:
          a) run 'specifyModelParamsMnist_fn' and
              incorporate user entry edits such as 'goal'.
          b) create connection matrices via 'initializeConnectionMatrices_fn'
      4. Load the experiment parameters.
      5. Run the simulation with 'sdeWrapper_fn'
      6. Plot results, print results to console
"""
import time
# print('start_time', time.time())
import numpy as np
from generateDownsampledMnistSet import generateDownsampledMnistSet
from showFeatureArrayThumbnails import showFeatureArrayThumbnails
from specifyModelParamsMnist import specifyModelParamsMnist
from initializeConnectionMatrices import initializeConnectionMatrices
from setMnistExperimentParams import setMnistExperimentParams
from sdeWrapper import sdeWrapper
from viewENresponses import viewENresponses
from classifyDigitsViaLogLikelihood import classifyDigitsViaLogLikelihood
from classifyDigitsViaThresholding import classifyDigitsViaThresholding
import copy
import scipy.io as sio

# -- USER ENTRIES --
DEBUG = False  # debug this Python code by comparing with the original Matlab code
useExistingConnectionMatrices = 0  # if = 1, load 'matrixParamsFilename', which includes filled-in connection matrices\
                                   # if = 0, generate new moth from template in specifyModelParamsMnist_fn.m
matrixParamsFilename = 'sampleMothModelParams'  # struct with all info, including connection matrices, of a particular moth.
numRuns = 1  # how many runs you wish to do with this moth or moth template, each run using random draws from the mnist set.
goal = 15  # defines the moth's learning rates, in terms of how many training samples per class give max accuracy. So "goal = 1" gives a very fast learner.\
           # if goal == 0, the rate parameters defined the template will be used as-is. if goal > 1, the rate parameters will be updated, even in a pre-set moth.
trPerClass = 3  # the number of training samples per class
numSniffs = 2  # number of exposures each training sample

# Flags to show various images:
showAverageImages = False  # to show thumbnails in 'examineClassAveragesAndCorrelations'
showThumbnailsUsed = 0  # N means show N experiment inputs from each class. 0 means don't show any.
showENPlots = [0, 0]  # 1 to plot, 0 to ignore
# arg1 above refers to statistical plots of EN response changes. One image (with 8 subplots) per EN.
# arg2 above refers to EN timecourses. Three scaled ENs timecourses on each of 4 images (only one EN on the 4th image).

# To save results if wished:
saveAllNeuralTimecourses = 0  # 0 -> save only EN (ie readout) timecourses.  Caution: 1 -> very high memory demands, hinders longer runs.
resultsFilename = 'result_run'  # will get the run number appended to it.
saveResultsDataFolder = ''   # String. If non-empty, 'resultsFilename' will be saved here.
                             # Need to create the folder manully before saving data to it.
saveResultsImageFolder = ''  # String. If non-empty, images will be saved here (if showENPlots also non-zero).
                             # Need to create the folder manully before saving images to it.

#  -- Misc book-keeping --
classLabels = np.arange(10)  # For MNIST.
valPerClass = 5  # number of digits used in validation sets and in baseline sets # use 5 when debugging
# make a vector of the classes of the training samples, randomly mixed:
trClasses = np.tile(classLabels, trPerClass)
np.random.shuffle(trClasses)
# repeat these inputs if taking multiple sniffs of each training sample:
trClasses = np.tile(trClasses, numSniffs)


# -- Load and preprocess the dataset--
"""
The dataset:
Because the moth brain architecture, as evolved, only handles ~60 features, we need to
create a new, MNIST-like task but with many fewer than 28x 28 pixels-as-features.
We do this by cropping and downsampling the mnist thumbnails, then selecting a subset of the 
remaining pixels.
This results in a cruder dataset (set various view flags to see thumbnails).
However, it is sufficient for testing the moth brain's learning ability. Other ML methods need  
to be tested on this same cruder dataset to make useful comparisons.

Define train and control pools for the experiment, and determine the receptive field.
This is done first because the receptive field determines the number of AL units, which
     must be updated in modelParams before 'initializeMatrixParams' runs.
This dataset will be used for each simulation in numRuns. Each
     simulation draws a new set of samples from this set.
"""

# Parameters:
# Parameters required for the dataset generation function are attached to a class preP.
# 1. The images used. This includes pools for mean-subtraction, baseline, train, and val.
#     This is NOT the number of training samples per class. That is trPerClass, defined above.

# specify pools of indices from which to draw baseline, train, val sets.
indPoolForBaseline = np.arange(100)
indPoolForTrain = np.arange(100, 300)
indPoolForPostTrain = np.arange(300, 400)


class preP:
    def __init__(self):
        # Population preprocessing pools of indices:
        self.indsToAverageGeneral = np.arange(550, 1000)
        self.indsToCalculateReceptiveField = np.arange(550, 1000)
        self.maxInd = max(max(self.indsToCalculateReceptiveField), max(indPoolForTrain))+1
        # Pre-processing parameters for the thumbnails:
        self.downsampleRate = 2
        self.crop = 2
        self.numFeatures = 85  # number of pixels in the receptive field
        self.pixelSum = 6
        self.showAverageImages = showAverageImages
        self.downsampleMethod = 1  # 0 means sum square patches of pixels. 1 means use bicubic interpolation.
        self.classLabels = classLabels
        self.useExistingConnectionMatrices = useExistingConnectionMatrices
        self.matrixParamsFilename = matrixParamsFilename


preprocessingParams = preP()

fA, activePixelInds, lengthOfSide = generateDownsampledMnistSet(preprocessingParams)  # average images are shown in this step

# The dataset fA is a feature array ready for running experiments. Each experiment uses a random draw from this dataset.
# fA = 10 x n x m array where 10 = #classes, n = #active pixels = 85, m = #digits from each class that will be used = 1000.
# -----------------------------

# Loop through the number of simulations specified:
print('starting sim(s) for goal =', goal, ', trPerClass = ', trPerClass, ', numSniffsPerSample = ', numSniffs, ':')

for run in range(numRuns):
    # -- Subsample the dataset for this simulation --
    # Line up the images for the experiment (in 10 parallel queues)

    if not DEBUG:
        digitQueues = np.zeros(fA.shape)
        for i in classLabels:
            # 1. Baseline (pre-train) images:
            # choose some images from the baselineIndPool:
            theseInds = np.random.randint(min(indPoolForBaseline), max(indPoolForBaseline) + 1, valPerClass)
            digitQueues[i, :, np.arange(valPerClass)] = fA[i, :, theseInds]

            # 2. Training images:
            # choose some images from the trainingIndPool:
            theseInds = np.random.randint(min(indPoolForTrain), max(indPoolForTrain) + 1, trPerClass)
            # repeat these inputs if taking multiple sniffs of each training sample:
            theseInds = np.repeat(theseInds, numSniffs)
            digitQueues[i, :, np.arange(valPerClass, valPerClass + trPerClass*numSniffs)] = fA[i, :, theseInds]

            # 3. Post-training (val) images:
            # choose some images from the postTrainIndPool:
            theseInds = np.random.randint(min(indPoolForPostTrain), max(indPoolForPostTrain) + 1, valPerClass)
            digitQueues[i, :, np.arange(valPerClass + trPerClass*numSniffs, valPerClass + trPerClass*numSniffs + valPerClass)] = fA[i, :, theseInds]

    if DEBUG:
        # ---- DEBUG ----
        mat_dq = sio.loadmat('DQ.mat')
        digitQueues = np.array(mat_dq['digitQueues'])
        digitQueues = digitQueues.swapaxes(0, 2)
        digitQueues = digitQueues.swapaxes(1, 2)

        mat_ap = sio.loadmat('AP.mat')
        activePixelInds = np.array(mat_ap['activePixelInds'])
        activePixelInds = activePixelInds.flatten()

        lengthOfSide = 144
        # -------------------

    # show the final versions of thumbnails to be used, if wished:
    if showThumbnailsUsed:
        tempArray = np.zeros((digitQueues.shape[0], lengthOfSide, digitQueues.shape[2]))
        tempArray[:, activePixelInds, :] = digitQueues  # fill in the non-zero pixels
        titleString = 'Input thumbnails'
        normalize = 1
        showFeatureArrayThumbnails(tempArray, showThumbnailsUsed, normalize, titleString)

    # -----------------------------

    # Create a moth. Either load an existing moth, or create a new moth:
    if useExistingConnectionMatrices:
        pass  # DIY if you need
    else:
        # load template params:
        modelParams = specifyModelParamsMnist(len(activePixelInds), goal)
        # Now populate the moth's connection matrices using the modelParams:
        modelParams = initializeConnectionMatrices(modelParams, DEBUG)

    modelParams.trueClassLabels = classLabels  # misc parameter tagging along
    modelParams.saveAllNeuralTimecourses = saveAllNeuralTimecourses

    # Define the experiment parameters, including book-keeping for time-stepped evolutions, eg
    # when octopamine occurs, time regions to poll for digit responses, windowing of Firing rates, etc
    experimentParams = setMnistExperimentParams(trClasses, classLabels, valPerClass, DEBUG)

    # -----------------------------

    # Run this experiment as sde time-step evolution:
    simResults = sdeWrapper(modelParams, experimentParams, digitQueues, DEBUG)

    # -----------------------------

    # Experiment Results: EN behavior, classifier calculations:
    # Process the sim results to group EN responses by class and time:
    r = viewENresponses(simResults, modelParams, experimentParams, showENPlots, classLabels, resultsFilename, saveResultsImageFolder)

    # Calculate the classification accuracy:
    # for baseline accuracy function argin, substitute pre- for post-values in r:
    rNaive = copy.deepcopy(r)
    for i in range(len(r)):
        rNaive[i].postMeanResp = r[i].preMeanResp
        rNaive[i].postStdResp = r[i].preStdResp
        rNaive[i].postTrainOdorResp = r[i].preTrainOdorResp

    # 1. Using Log-likelihoods over all ENs:
    # Baseline accuracy:
    outputNaiveLogL = classifyDigitsViaLogLikelihood(rNaive)
    print('Naive accuracy:', np.round(outputNaiveLogL.totalAccuracy), '%, by class:', np.round(outputNaiveLogL.accuracyPercentages), '%')

    # Post-training accuracy using log-likelihood over all ENs:
    outputTrainedLogL = classifyDigitsViaLogLikelihood(r)
    print('Trained accuracy:', np.round(outputTrainedLogL.totalAccuracy), '%, by class:', np.round(outputTrainedLogL.accuracyPercentages), '%')

    # 2. Using single EN thresholding: # doesn't work well. Not appeared in the paper
    '''
    outputNaiveThresholding = classifyDigitsViaThresholding(rNaive, 1e9, -1, 10)
    print('Naive accuracy:', np.round(outputNaiveThresholding.totalAccuracy), '%, by class:',
          np.round(outputNaiveThresholding.accuracyPercentages), '%')

    outputTrainedThresholding = classifyDigitsViaThresholding(r, 1e9, -1, 10)
    print('Trained accuracy:', np.round(outputTrainedThresholding.totalAccuracy), '%, by class:',
          np.round(outputTrainedThresholding.accuracyPercentages), '%')
    '''

    # save the accuracy results, and other run data:
    if len(saveResultsDataFolder) > 0:
        # Temporarily unavailable; get errors when trying to pickle class objects like r or modelParams.
        fname = saveResultsDataFolder + '/' + resultsFilename + '_{}'.format(run) + '.pkl'











