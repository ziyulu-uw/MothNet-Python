"""
Prepares for and runs the SDE time-stepped evolution of neural firing rates.
Inputs:
  1. modelParams: struct with connection matrices etc
  2. expParams: struct with timing info about experiment, eg when stimuli are given.
  3. featureArray: numClassesnum x Features x numStimsPerClass array of stimuli
Output:
  1. simResults: EN timecourses and final P2K and K2E connection matrices. Note that
      other neurons' timecourses (outputted from sdeEvolutionMnist) are not retained in simResults.
"""

import numpy as np
import time
"""
4 sections:
1. load various params needed for pre-evolution prep
2. specify stim and octo courses
3. interaction equations and step through simulation
4. unpack evolution output and export
"""
from sdeEvolutionMnist import sdeEvolutionMnist


def sdeWrapper(modelParams, expParams, featureArray, DEBUG=False):

    # 1. initialize states of various components:
    # unpack a few variables that are needed before the evolution stage:
    nP = modelParams.nP  # = nG
    nG = modelParams.nG
    nPI = modelParams.nPI
    nK = modelParams.nK
    nR = modelParams.nR  # = nG
    nE = modelParams.nE
    F2R = modelParams.F2R

    # 2. Define Stimuli and Octopamine time courses:
    # set time span and events:
    simStart = expParams.simStart
    simStop = expParams.simStop
    timeStep = 2 * 0.01
    Time = np.arange(simStart, simStop, timeStep)

    # create stimMags, a matrix n x m where n = # of odors and m = # timesteps.
    stimStarts = expParams.stimStarts
    durations = expParams.durations
    whichClass = expParams.whichClass
    classList = np.arange(10)
    classMags = expParams.classMags
    # create a classMagMatrix, each row giving the stimulus magnitudes of a different class:
    classMagMatrix = np.zeros((len(classList), len(Time)))
    small = 1e-6
    for i in range(len(classList)):
        # extract the relevant odor puffs. All vectors should be same size, in same order
        theseClassStarts = stimStarts[whichClass == classList[i]]
        theseDurations = durations[whichClass == classList[i]]
        theseMags = classMags[whichClass == classList[i]]
        for j in range(len(theseClassStarts)):
            classMagMatrix[i, (Time > theseClassStarts[j]) & (Time < theseClassStarts[j] + theseDurations[j] - small)] = theseMags[j]

    # Apply a lowpass to round off the sharp start-stop edges of stimuli and octopamine:
    lpParam = expParams.lpParam
    L = round(lpParam / timeStep)  # define the slope of low pass transitions here
    lpWindow = np.hamming(L)  # window of width L
    lpWindow = lpWindow / np.sum(lpWindow)

    # window the stimulus time courses:
    zeroArr = np.array([0.0])
    for i in range(len(classList)):
        shifted = np.convolve(classMagMatrix[i,:], lpWindow, 'same')[1:]
        classMagMatrix[i,:] = np.concatenate((shifted, zeroArr))

    # window the octopamine:
    octoMag = expParams.octoMag
    octoHits = np.zeros(len(Time))
    octoStart = expParams.octoStart
    durationOcto = expParams.durationOcto
    octoStop = octoStart + durationOcto
    small = 1e-6
    for i in range(len(octoStart)):
        octoHits[(Time > octoStart[i] - small) & (Time < octoStop[i] - small)] = octoMag
    shifted = np.convolve(octoHits, lpWindow, 'same')[1:]
    octoHits = np.concatenate((shifted, zeroArr))

    # 3. do SDE time-step evolution:
    # Use euler-maruyama SDE method, milstein's version.
    # Y (the vector of all neural firing rates) is structured as a row vector as follows: [ P, PI, L, K, E ]
    Po = 1 * np.ones(nP)  # P are the normalized FRs of the excitatory PNs
    PIo = 1 * np.ones(nPI)  # PI are the normed FRs of the inhib PNs
    Lo = 1 * np.ones(nG)
    Ro = modelParams.Rspont.transpose()[0]
    Ko = 1 * np.ones(modelParams.nK)  # K are the normalized firing rates of the kenyon cells
    Eo = 0 * np.ones(modelParams.nE)
    initCond = np.concatenate((Po, PIo, Lo, Ro, Ko, Eo))  # initial conditions for Y
    tspan = [simStart, simStop]
    seedValue = None  # to free up or fix randn. If None, a random seed value will be chosen. Otherwise, the seed will be defined.

    # Run the SDE evolution:
    thisRun = sdeEvolutionMnist(tspan, initCond, Time, classMagMatrix, featureArray, octoHits, modelParams, expParams, seedValue, DEBUG)

    # 4. Unpack Y and save results:
    class SimResults:
        def __init__(self):
            self.T = None
            self.E = None
            self.P = None
            self.K = None
            self.octoHits = None
            self.P2Kfinal = None
            self.K2Efinal = None

    simResults = SimResults()

    Y = thisRun.Y  # Y is a matrix numTimePoints x nG. Each col is a PN, each row holds values for a single timestep
    if modelParams.saveAllNeuralTimecourses:
        simResults.P = Y[:, 0: nP]
        simResults.K = Y[:, nP + nPI + nG + nR: nP + nPI + nG + nR + nK]
    simResults.T = thisRun.T  # row vector
    simResults.E = thisRun.E
    simResults.octoHits = octoHits
    simResults.K2Efinal = thisRun.K2Efinal
    simResults.P2Kfinal = thisRun.P2Kfinal

    return simResults




















