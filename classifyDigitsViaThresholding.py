"""
Classify the test digits in a run using log likelihoods from the various EN responses, with the added option of
rewarding high scores relative to an ENs home-class expected response distribution.
One use of this function is to apply de-facto thresholding on discrete ENs, so that the predicted class
corresponds to the EN that spiked most strongly (relative to its usual home-class response).
Inputs:
 1. results = 1 x 10 struct produced by viewENresponses. i'th entry gives results for all classes, in the i'th EN.
  Important fields:
    a. postMeanResp, postStdResp (to calculate post-training, ie val, digit response distributions).
    b. postTrainOdorResponse (gives the actual responses for each val digit, for that EN)
        Note that non-post-train odors have response = -1 as a flag.
    c. odorClass: gives the true labels of each digit, 1 to 10 (10 = '0'). this is the same in each EN.
 2.  'homeAdvantage' is the emphasis given to the home EN. It
          multiplies the off-diagonal of dist. 1 -> no advantage
          (default). Very high means that a test digit will be classified
          according to the home EN it does best in, ie each EN acts on
          it's own.
 3.  'homeThresholdSigmas' = the number of stds below an EN's home-class mean that we set a threshold, such that
          if a digit scores above this threshold in an EN, that EN will
          be rewarded by 'aboveHomeThreshReward' (below)
 4.  'aboveHomeThreshReward': if a digit's response scores above the EN's mean home-class  value, reward it by
          dividing by aboveHomeThreshReward. This reduces the log likelihood score for that EN.
Output:
   A struct with the following fields:
   1. likelihoods = n x 10 matrix, each row a postTraining digit. The entries are summed log likelihoods.
   2. trueClasses = shortened version of whichOdor (with only postTrain, ie validation, entries)
   3. predClasses = predicted classes
   4. confusionMatrix = raw counts, rows = ground truth, cols = predicted
   5. classAccuracies = 1 x 10 vector, with class accuracies as percentages
   6. totalAccuracy = overall accuracy as percentage
---------------------------------------------
plan:
1. for each test digit (ignore non-postTrain digits), for each EN, calculate the # stds from the test
   digit is from each class distribution. This makes a 10 x 10 matrix
   where each row corresponds to an EN, and each column corresponds to a class.
2. Square this matrix by entry. Sum the columns. Select the col with the lowest value as the predicted
   class. Return the vector of sums in 'likelihoods'.
3. The rest is simple calculation

the following values of argin2,3,4 correspond to the log likelihood
classifier in 'classifyDigitsViaLogLikelihood.m':
    homeAdvantage = 1;
    homeThresholdSigmas = any number;
    aboveHomeThreshReward = 1;
The following value enables pure home-class thresholding:
    homeAdvantage = 1e12;        % to effectively  eliminate off-diagonals
"""
import numpy as np
import copy


def classifyDigitsViaThresholding(results, homeAdvantage, homeThresholdSigmas, aboveHomeThreshReward):
    from sklearn.metrics import confusion_matrix
    nEn = len(results)  # number of ENs, same as number of classes
    ptInds = np.argwhere(results[0].postTrainOdorResp >= 0).flatten()  # indices of post-train (ie validation) digits
    nP = len(ptInds)  # number of post-train digits

    # extract true classes:
    temp = results[0].odorClass  # throughout, digits may be referred to as odors or 'odor puffs'
    trueClasses = temp[ptInds]

    # extract the relevant odor puffs: Each row is an EN, each col is an odor puff
    ptResp = np.zeros((nEn, nP))
    for i in range(nEn):
        temp = results[i].postTrainOdorResp
        ptResp[i] = temp[ptInds]

    # make a matrix of mean Class Resps and stds. Each row is an EN, each col is a class.
    # For example, the i'th row, j'th col entry of 'mu' is the mean of the i'th
    # EN in response to digits from the j'th class; the diagonal contains the
    # responses to the home-class.
    mu = np.zeros((nEn, nEn))
    sig = np.zeros((nEn, nEn))
    for i in range(nEn):
        mu[i] = results[i].postMeanResp
        sig[i] = results[i].postStdResp

    # for each EN:
    # get the likelihood of each puff (ie each col of ptResp)
    likelihoods = np.zeros((nP, nEn))
    for i in range(nP):
        dist = np.divide((np.tile(ptResp[:, i].reshape((nEn, 1)), (1, nEn)) - mu), sig)
        # 10 x 10 matrix. The ith row, jth col entry is the mahalanobis distance of
        # this test digit's response from the i'th ENs response to the j'th class. For example, the diagonal contains
        # the mahalanobis distance of this digit's response to each EN's home-class response.

        # 1. Apply rewards for above-threshold responses:
        offDiag = dist - np.diag(np.diag(dist))
        onDiag = np.diag(dist)
        # Reward any onDiags that are above some threshold (mu - n*sigma) of an EN.
        # CAUTION: This reward-by-shrinking only works when off-diagonals are
        # demolished by very high value of 'homeAdvantage'.
        homeThreshs = homeThresholdSigmas * np.diag(sig)
        aboveThreshInds = np.argwhere(onDiag > homeThreshs).flatten()
        onDiag1 = copy.deepcopy(onDiag)
        onDiag1[aboveThreshInds] = onDiag[aboveThreshInds] / aboveHomeThreshReward
        onDiag = np.diag(onDiag1)  # turn back into a matrix

        # 2. Emphasize the home-class results by shrinking off-diagonal values.  This makes the off-diagonals less important
        # in the final likelihood sum. This shrinkage is for a different purpose than in the lines above.
        dist = offDiag / homeAdvantage + onDiag
        likelihoods[i] = np.sum(np.power(dist, 4), axis=0)

    # make predictions
    predClasses = np.zeros(nP)
    for i in range(nP):
        predClasses[i] = np.argmin(likelihoods[i])

    # calc accuracy percentages:
    classAccuracies = np.zeros(nEn)
    for i in range(nEn):
        classAccuracies[i] = 100 * np.sum((predClasses == i) & (trueClasses == i)) / np.sum(trueClasses == i)
    totalAccuracy = 100 * np.sum(predClasses == trueClasses) / len(trueClasses)

    # confusion matrix:
    # i,j'th entry is number of test digits with true label i that were predicted to be j.
    confusion = confusion_matrix(trueClasses, predClasses)

    class Output:
        def __init__(self):
            self.trueClasses = trueClasses
            self.predClasses = predClasses
            self.likelihoods = likelihoods
            self.accuracyPercentages = classAccuracies
            self.totalAccuracy = totalAccuracy
            self.confusionMatrix = confusion

    output = Output()

    return output





