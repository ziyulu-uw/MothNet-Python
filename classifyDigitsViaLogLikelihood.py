"""
Classify the test digits in a run using log likelihoods from the various EN responses:
Inputs:
 results = 1 x 10 struct produced by viewENresponses. i'th entry gives results for all classes, in the i'th EN.
  Important fields:
    1. postMeanResp, postStdResp (to calculate post-training, ie val, digit response distributions).
    2. postTrainOdorResponse (gives the actual responses for each val digit, for that EN)
        Note that non-post-train odors have response = -1 as a flag.
    3. odorClass: gives the true labels of each digit, 0 to 9. this is the same in each EN.

output:
   output = struct with the following fields:
   1. likelihoods = n x 10 matrix, each row a postTraining digit. The entries are summed log likelihoods.
   2. trueClasses = shortened version of whichOdor (with only postTrain, ie validation, entries)
   3. predClasses = predicted classes
   4. confusionMatrix = raw counts, rows = ground truth, cols = predicted
   5. classAccuracies = 1 x 10 vector, with class accuracies as percentages
   6. totalAccuracy = overall accuracy as percentage
"""
"""
plan:
1. for each test digit (ignore non-postTrain digits), for each EN, calculate the # stds from the test
   digit is from each class distribution. This makes a 10 x 10 matrix
   where each row corresponds to an EN, and each column corresponds to a class.
2. Square this matrix by entry. Sum the columns. Select the col with the lowest value as the predicted
   class. Return the vector of sums in 'likelihoods'.
3. The rest is simple calculation
"""
import numpy as np


def classifyDigitsViaLogLikelihood(results):
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

    # make a matrix of mean Class Resps and stds. Each row is an EN, each col is a class:
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
        likelihoods[i] = np.sum(np.power(dist, 4), axis=0)

    # make predictions
    predClasses = np.zeros(nP)
    for i in range(nP):
        predClasses[i] = np.argmin(likelihoods[i])

    # calc accuracy percentages:
    classAccuracies = np.zeros(nEn)
    for i in range(nEn):
        classAccuracies[i] = 100*np.sum((predClasses == i) & (trueClasses == i))/np.sum(trueClasses == i)
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











