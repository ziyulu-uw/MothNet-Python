"""
This function defines parameters of a time-evolution experiment: overall timing, stim timing and strength,
octo timing and strength, lowpass window parameter,  etc.
It does book-keeping to allow analysis of the SDE time-stepped evolution of the neural firing rates.
Inputs:
      1. trClasses: vector of indices giving the classes of the training digits in order.
                   The first entry must be nonzero. Unused entries can be filled with -1s if wished.
      2. classLabels: a list of labels, eg 0:9 for mnist
      3. valPerClass: how many digits of each class to use for baseline and post-train
Output:
  1. expParams: struct with experiment info.
"""
import numpy as np


def setMnistExperimentParams(trClasses, classLabels, valPerClass, DEBUG=False):
    # Order of time periods:
    # 1. no event period: allow system to settle to a steady state spontaneous FR baseline
    # 2. baseline period: deliver a group of digits for each class
    # 3. no event buffer
    # 5. training period:  deliver digits + octopamine + allow hebbian updates
    # 6. no event buffer
    # 7. post-training period: deliver a group of digits for each class
    stimMag = 20  # (stim magnitudes as passed into AL. See original version in smartAsABug codebase)
    stimLength = 0.22
    nC = len(classLabels)  # number of classes in this experiment

    # Define the time span and events:
    step = 3  # the time between digits (3 seconds)
    trStep = step + 2  # allow more time between training digits
    simStart = -30  # use negative start-time for convenience (artifact)

    # Baseline period:
    # do a loop, to allow gaps between class groups:
    baselineTimes = []
    startTime = 30
    gap = 10
    for i in range(nC):
        baselineTimesThisClass = [i for i in range(startTime, startTime + valPerClass*step, step)]  # vector of timepoints, one digit applied every 'step' seconds
        baselineTimes += baselineTimesThisClass
        startTime = baselineTimes[-1] + gap

    endOfBaseline = baselineTimes[-1] + 25  # include extra buffer before training

    # Training period:
    trainTimes = [i for i in range(endOfBaseline, endOfBaseline + len(trClasses)*trStep, trStep)]  # vector of timepoints, one digit every 'trStep' seconds
    endOfTrain = trainTimes[-1] + 25  # includes buffer before Validation

    # Val period:
    # do a loop, to allow gaps between class groups:
    valTimes = []
    startTime = endOfTrain
    for i in range(nC):
        valTimesThisClass = [i for i in range(startTime, startTime + valPerClass * step, step)]
        valTimes += valTimesThisClass
        startTime = valTimes[-1] + gap

    endOfVal = valTimes[-1] + 4

    # assemble vectors of stimulus data for export:
    # Assign the classes of each stim. Assign the baseline and val in blocks, and the training stims in the order passed in:
    whichClass = np.zeros(len(baselineTimes)+len(trainTimes)+len(valTimes))
    numBaseline = valPerClass * nC
    numTrain = len(trClasses)
    for c in range(nC):
        whichClass[c * valPerClass: (c+1) * valPerClass] = classLabels[c]  # the baseline groups
        whichClass[numBaseline + numTrain + c * valPerClass: numBaseline + numTrain + (c+1) * valPerClass] = classLabels[c]  # the val groups
    whichClass[numBaseline: numBaseline + numTrain] = trClasses

    stimStarts = np.array(baselineTimes + trainTimes + valTimes)
    durations = stimLength * np.ones(len(stimStarts))
    classMags = stimMag * np.ones(len(stimStarts))

    if DEBUG:
        # ---- DEBUG ----
        import scipy.io as sio
        mat_dq = sio.loadmat('WC.mat')
        whichClass = np.array(mat_dq['whichClass'])
        whichClass = whichClass.flatten() - 1  # class 0 <-> digit 1 images, ... class 9 <-> digit 0 images
        # whichClass = np.where(whichClass < 10, whichClass, 0)
        # -------------------

    class ExpParams:
        def __init__(self):
            self.simStart = simStart
            self.whichClass = whichClass
            self.stimStarts = stimStarts
            self.durations = durations
            self.classMags = classMags
            # octopamine input timing:
            self.octoMag = 1
            self.octoStart = np.array(trainTimes)
            self.durationOcto = 1
            # heb timing: Hebbian updates are enabled 25% of the way into the stimulus, and
            # last until 75% of the way through (ie active during the peak response period)
            self.hebStarts = np.array(trainTimes) + 0.25*stimLength
            self.hebDurations = 0.5*stimLength*np.ones(len(trainTimes))
            self.startTrain = min(self.hebStarts)
            self.endTrain = max(self.hebStarts) + max(self.hebDurations)
            # Other time parameters required for time evolution book-keeping:
            # the numbers 1,2,3 refer to time periods where spont responses are allowed to settle before recalibration.
            self.startPreNoiseSpontMean1 = -25
            self.stopPreNoiseSpontMean1 = -15
            # Currently no change is made in start/stopSpontMean2. So spontaneous behavior may be stable in this range.
            self.startSpontMean2 = -10
            self.stopSpontMean2 = -5
            # currently, spontaneous behavior is steady-state by startSpontMean3.
            self.startSpontMean3 = 0
            self.stopSpontMean3 = 28
            self.preHebPollTime = min(trainTimes) - 5
            self.postHebPollTime = max(trainTimes) + 5
            # timePoints for plotting EN responses:
            # spontaneous response periods, before and after training, to view effect of training on spontaneous FRs:
            self.preHebSpontStart = self.startSpontMean3
            self.preHebSpontStop = self.stopSpontMean3
            self.postHebSpontStart = max(trainTimes) + 5
            self.postHebSpontStop = min(valTimes) - 3
            # hamming filter window parameter (= width of transition zone in seconds). The lp filter is applied to odors and to octo
            self.lpParam = 0.12
            self.simStop = max(stimStarts) + 10  # 1405

    expParams = ExpParams()

    return expParams











