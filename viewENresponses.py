"""
View readout neurons (EN):
Color-code them dots by class and by concurrent octopamine.
Collect stats: median, mean, and std of FR for each digit, pre- and post-training.
Throughout, digits may be referred to as odors, or as odor puffs.
'Pre' = naive. 'Post' = post-training

Inputs:
  1. simResults: output of sdeWrapper.py
  2. modelParams: struct of this moth's parameters
  3. experimentParams: struct with timing and digit class info from the experiment.
  4. showPlots: 1 x 2 vector. First entry: show changes in accuracy. 2nd entry: show EN timecourses.
  5. classLabels: 0 to 9
  6. resultsFilename:  to generate image filenames if saving. Optional argin
  7. saveImageFolder:  where to save images. If this = [], images will not be saved (ie its also a flag). Optional argin.

Outputs (as fields of resultsStruct):
  1. preMeanResp = numENs x numOdors matrix = mean of EN pre-training
  2. preStdResp = numENs x numOdors matrix = std of EN responses pre-training
  3. ditto for post etc
  4. percentChangeInMeanResp = 1 x numOdors vector
  5. trained = list of indices corresponding to the odor(s) that were trained
  6. preSpontMean = mean(preSpont);
  7. preSpontStd = std(preSpont);
  8. postSpontMean = mean(postSpont);
  9. postSpontStd = std(postSpont);
"""
import numpy as np
import matplotlib.pyplot as plt


def viewENresponses(simResults, modelParams, experimentParams,showPlots, classLabels, resultsFilename, saveImageFolder):

    nE = modelParams.nE

    # pre- and post-heb spont stats:
    preHebSpontStart = experimentParams.preHebSpontStart
    preHebSpontStop = experimentParams.preHebSpontStop
    postHebSpontStart = experimentParams.postHebSpontStart
    postHebSpontStop = experimentParams.postHebSpontStop

    E = simResults.E  # (#timesteps, #ENs)
    T = simResults.T  # (#timesteps,)
    octoHits = simResults.octoHits
    if max(octoHits) > 0:
        octoTimes = T[octoHits > 0]
    else:
        octoTimes = []

    # calc spont stats:
    preSpont = E[(T > preHebSpontStart) & (T < preHebSpontStop)]
    postSpont = E[(T > postHebSpontStart) & (T < postHebSpontStop)]
    preHebMean = np.mean(preSpont, axis=0)
    preHebStd = np.std(preSpont, axis=0)
    postHebMean = np.mean(postSpont, axis=0)
    postHebStd = np.std(postSpont, axis=0)

    # Set regions to examine:
    # 1. data from experimentParams:
    simStart = experimentParams.simStart
    classMags = experimentParams.classMags
    stimStarts = experimentParams.stimStarts  # to get timeSteps from very start of sim
    classMagsMask = (classMags > 0)
    stimStarts = np.multiply(stimStarts, classMagsMask)  # ie only use non-zero puffs
    whichClass = experimentParams.whichClass
    whichClass = np.multiply(whichClass, classMagsMask)
    startTrain = experimentParams.startTrain
    endTrain = experimentParams.endTrain
    classList = np.unique(whichClass)
    numClasses = len(classList)

    class Result:
        def __init__(self):
            self.preTrainOdorResp = None
            self.postTrainOdorResp = None
            self.preRespSniffsAved = None
            self.postRespSniffsAved = None
            self.odorClass = None
            self.percentChangeInMeanResp = None
            self.percentChangeInNoiseSubtractedMeanResp = None
            self.relativeChangeInNoiseSubtractedMeanResp = None
            self.percentChangeInMedianResp = None
            self.percentChangeInNoiseSubtractedMedianResp = None
            self.relativeChangeInNoiseSubtractedMedianResp = None
            self.trained = None
            # EN odor responses, pre and post training:
            self.preMeanResp = None
            self.preStdResp = None
            self.postMeanResp = None
            self.postStdResp = None
            # spont responses, pre and post training:
            self.preSpontMean = None
            self.preSpontStd = None
            self.postSpontMean = None
            self.postSpontStd = None

    results = []  # each element is an instance of the class Result

    # Make one stats plot per EN. Loop through ENs:
    for enInd in range(nE):
        thisEnResponse = E[:, enInd]  # a row vector
        # Calculate pre- and post-train odor response stats:
        # Assumes that there is at least 1 sec on either side of an odor without octo
        preTrainOdorResp = []
        postTrainOdorResp = []

        for i in range(len(stimStarts)):
            t = stimStarts[i]
            # Note: to find no-octo stimStarts, there is a certain amount of machinery in order to mesh with the timing data from the experiment.
            # For some reason octoTimes are not recorded exactly as listed in format short mode. So we need to
            # use abs difference > small thresh
            small = 1e-8
            # assign no-octo, PRE-train response val (or -1):
            if len(octoTimes) == 0:
                preTrainOdorResp.append(max(thisEnResponse[(T > t - 1) & (T < t + 1)]))
            elif (t < startTrain) & (min(abs(octoTimes - t)) > small):
                preTrainOdorResp.append(max(thisEnResponse[(T > t - 1) & (T < t + 1)]))
            else:
                preTrainOdorResp.append(-1)
            # assign no-octo, POST-train response val (or -1):
            if len(octoTimes) != 0:
                if (t > endTrain) & (min(abs(octoTimes - t)) > small):
                    postTrainOdorResp.append(max(thisEnResponse[(T > t - 1) & (T < t + 1)]))
                else:
                    postTrainOdorResp.append(-1)
            else:
                postTrainOdorResp.append(-1)

        preTrainOdorResp = np.array(preTrainOdorResp)
        postTrainOdorResp = np.array(postTrainOdorResp)
        # calc no-octo stats for each odor, pre and post train:
        preMeanResp = np.zeros(numClasses)
        preMedianResp = np.zeros(numClasses)
        preStdResp = np.zeros(numClasses)
        preNumPuffs = np.zeros(numClasses)
        postMeanResp = np.zeros(numClasses)
        postMedianResp = np.zeros(numClasses)
        postStdResp = np.zeros(numClasses)
        postNumPuffs = np.zeros(numClasses)
        for k in range(numClasses):
            # calculate the averaged sniffs of each sample: SA means 'sniffsAveraged'
            preSA = preTrainOdorResp[(preTrainOdorResp >= 0) & (whichClass == classList[k])]
            postSA = postTrainOdorResp[(postTrainOdorResp >= 0) & (whichClass == classList[k])]

            if len(preSA) == 0:
                preMeanResp[k] = -1
                preMedianResp[k] = -1
                preStdResp[k] = -1
                preNumPuffs[k] = 0
            else:
                preMeanResp[k] = np.mean(preSA)
                preMedianResp[k] = np.median(preSA)
                preStdResp[k] = np.std(preSA, ddof=1)
                preNumPuffs[k] = len(preSA)

            if len(postSA) == 0:
                postMeanResp[k] = -1
                postMedianResp[k] = -1
                postStdResp[k] = -1
                postNumPuffs[k] = 0
            else:
                postMeanResp[k] = np.mean(postSA)
                postMedianResp[k] = np.median(postSA)
                postStdResp[k] = np.std(postSA, ddof=1)
                postNumPuffs[k] = len(postSA)

        # print(enInd, 'pre:', preMeanResp)
        # print(enInd, 'post:', postMeanResp)

        preSA = np.argwhere(preNumPuffs > 0).flatten()
        postSA = np.argwhere(postNumPuffs > 0).flatten()
        postOffset = postSA + 0.25

        # a key output
        percentChangeInMeanResp = 100*np.divide((postMeanResp[preSA] - preMeanResp[preSA]), preMeanResp[preSA])
        percentChangeInNoiseSubtractedMeanResp = 100*np.divide((postMeanResp[preSA] - preMeanResp[preSA] - postHebMean),
                                                               preMeanResp[preSA])
        percentChangeInMedianResp = 100*np.divide((postMedianResp[preSA] - preMedianResp[preSA]), preMedianResp[preSA])
        percentChangeInNoiseSubtractedMedianResp = 100*np.divide((postMedianResp[preSA] - preMedianResp[preSA] - postHebMean),
                                                                 preMedianResp[preSA])

        # plot stats if wished
        if showPlots[0]:
            fig, axs = plt.subplots(nrows=2, ncols=3)
            # ---- medians ----
            # raw medians, pre and post:
            axs[0, 0].set_title('EN {} median response change'.format(enInd))
            axs[0, 0].grid()
            axs[0, 0].plot(preSA, preMedianResp[preSA], '*b')
            axs[0, 0].plot(postOffset, postMedianResp[postSA], 'ob')

            # make the home EN of this plot red:
            axs[0, 0].plot(enInd, preMedianResp[enInd], '*r')
            axs[0, 0].plot(enInd + 0.25, postMedianResp[enInd], 'or')

            # connect pre to post with lines for clarity:
            for j in range(len(preSA)):
                if j == enInd:
                    axs[0, 0].plot([preSA[j], postOffset[j]], [preMedianResp[preSA[j]], postMedianResp[preSA[j]]], '-r')
                else:
                    axs[0, 0].plot([preSA[j], postOffset[j]], [preMedianResp[preSA[j]], postMedianResp[preSA[j]]], '-b')

            # percent change in medians:
            axs[0, 1].set_title('median percent change')
            axs[0, 1].grid()
            axs[0, 1].plot(preSA, percentChangeInMedianResp, 'ob')
            axs[0, 1].plot(enInd, percentChangeInMedianResp[enInd], 'or')  # mark the trained odors in red

            # relative changes in median, ie control/trained:
            axs[0, 2].set_title('relative percent change')
            axs[0, 2].grid()
            pn = np.sign(postMedianResp[enInd] - preMedianResp[enInd])
            axs[0, 2].plot(preSA, pn*(percentChangeInMedianResp/percentChangeInMedianResp[enInd]), 'ob')
            axs[0, 2].plot(enInd, pn, 'or')

            # ---- means ----
            # raw means, pre and post:
            axs[1, 0].set_title('EN {} mean response change'.format(enInd))
            axs[1, 0].grid()
            axs[1, 0].errorbar(x=preSA, y=preMeanResp[preSA], yerr=preStdResp[preSA], fmt='ob')
            axs[1, 0].errorbar(x=postOffset, y=postMeanResp[postSA], yerr=postStdResp[postSA], fmt='ob')
            axs[1, 0].errorbar(x=enInd, y=preMeanResp[enInd], yerr=preStdResp[enInd], fmt='or')
            axs[1, 0].errorbar(x=enInd + 0.25, y=postMeanResp[enInd], yerr=postStdResp[enInd], fmt='or')

            # percent change in means:
            axs[1, 1].set_title('mean percent change')
            axs[1, 1].grid()
            axs[1, 1].plot(preSA, percentChangeInMeanResp, 'ob')
            axs[1, 1].plot(enInd, percentChangeInMeanResp[enInd], 'or')

            # relative percent changes in mean:
            axs[1, 2].set_title('relative percent change')
            axs[1, 2].grid()
            pn = np.sign(postMeanResp[enInd] - preMeanResp[enInd])
            axs[1, 2].plot(preSA, pn*(percentChangeInMeanResp/percentChangeInMeanResp[enInd]), 'ob')
            axs[1, 2].plot(enInd, pn, 'or')

            if len(saveImageFolder) > 0:
                name = saveImageFolder + '/' + 'en{}Responses'.format(enInd) + '.png'
                plt.savefig(name)

            plt.show()

        r = Result()
        r.preTrainOdorResp = preTrainOdorResp
        r.postTrainOdorResp = postTrainOdorResp
        r.preRespSniffsAved = preSA
        r.postRespSniffsAved = postSA
        r.odorClass = whichClass
        r.percentChangeInMeanResp = percentChangeInMeanResp
        r.percentChangeInNoiseSubtractedMeanResp = percentChangeInNoiseSubtractedMeanResp
        r.relativeChangeInNoiseSubtractedMeanResp = percentChangeInNoiseSubtractedMeanResp/percentChangeInNoiseSubtractedMeanResp[enInd]
        r.percentChangeInMedianResp = percentChangeInMedianResp
        r.percentChangeInNoiseSubtractedMedianResp = percentChangeInNoiseSubtractedMedianResp
        r.relativeChangeInNoiseSubtractedMedianResp = percentChangeInNoiseSubtractedMedianResp/percentChangeInNoiseSubtractedMedianResp[enInd]
        r.trained = enInd
        r.preMeanResp = preMeanResp
        r.preStdResp = preStdResp
        r.postMeanResp = postMeanResp
        r.postStdResp = postStdResp
        r.preSpontMean = preHebMean
        r.preSpontStd = preHebStd
        r.postSpontMean = postHebMean
        r.postSpontStd = postHebStd

        results.append(r)

    # Plot EN timecourses normalized by mean digit response:
    if showPlots[1]:
        colors = [(0, 0, 1), (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 0, 1), (0, 1, 1),(1, 0.3, 0.8), (0.8, 0.3, 1), (0.8, 1, 0.3), (0.5, 0.5, 0.5)]
        for enInd in range(nE):  # recall EN1 targets digit class 1, EN2 targets digit class 2, etc
            if np.remainder(enInd, 3) == 0:
                fig, axs = plt.subplots(3)
            # plot mean pre and post training of trained digit:
            preMean = results[enInd].preMeanResp
            preMeanControl = np.concatenate((preMean[:enInd], preMean[enInd+1:]))
            preMeanControl = np.mean(preMeanControl)
            postMean = results[enInd].postMeanResp
            postMeanControl = np.concatenate((postMean[:enInd], postMean[enInd+1:]))
            postMeanControl = np.mean(postMeanControl)
            preTime = T[T < startTrain]
            preTimeInds = np.argwhere(T < startTrain).flatten()
            postTime = T[T > endTrain]
            postTimeInds = np.argwhere(T > endTrain).flatten()
            midTime = T[(T > startTrain) & (T < endTrain)]
            midTimeInds = np.argwhere((T > startTrain) & (T < endTrain)).flatten()

            imInd = np.remainder(enInd, 3)
            # plot ENs:
            axs[imInd].plot(preTime, E[preTimeInds, enInd] / preMeanControl, '-b')
            axs[imInd].plot(postTime, E[postTimeInds, enInd] / postMeanControl, '-b')
            axs[imInd].plot(midTime, E[midTimeInds, enInd], '-b')
            # plot stims by color
            for i in range(numClasses):
                axs[imInd].plot(stimStarts[whichClass == classList[i]], np.zeros(len(stimStarts[whichClass == classList[i]])), color=colors[i], marker='.')
                if i == enInd:
                    axs[imInd].plot(stimStarts[whichClass == classList[i]], 0.001*np.ones(len(stimStarts[whichClass == classList[i]])), color=colors[i], marker='.')
            axs[imInd].set_ylim([0, 1.2*np.amax(E[postTimeInds, enInd]/postMeanControl)])
            axs[imInd].set_xlim([-30, max(T)])
            axs[imInd].set_title('EN {} for class {}'.format(enInd, enInd))
            if imInd == 2:

                if len(saveImageFolder) > 0:
                    name = saveImageFolder + '/' + 'en{}Timecourses'.format(enInd) + '.png'
                    plt.savefig(name)

                plt.show()

    return results



































































