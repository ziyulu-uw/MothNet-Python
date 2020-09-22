"""
To include neural noise, evolve the differential equations using euler-maruyama, milstein version
(see Higham's Algorithmic introduction to numerical simulation of SDE)
Called by sdeWrapper. For use with mnist experiments.
Inputs:
  1. tspan: 1 x 2 vector = start and stop timepoints (sec)
  2. initCond: col vector with all starting values for P, L, etc
  3. T: vector of time points for stepping (we assume that noise and FRs have the same step size based on Milstein's method)
  4. classMagMatrix: nC x length(T) matrix where nC = # different classes (for digits, up to 10).
     Each row  contains mags of digits from a given class.
  5. featureArray: numClasses x numFeatures x numStimsPerClass array
  6. octoHits: 1 x length(T) vector with octopamine strengths at each timepoint
  7. params: modelParams, including connection matrices, learning rates, etc
  8. expParams: experiment parameters with some timing info
  9. seedValue: starting seed value for reproducibility. optional arg
Output:
  thisRun: an instance of the class ThisRun with attributes
  T = m x 1 vector, timepoints used in evolution;
  Y = m x K matrix, where K contains all FRs for P, L, PI, KC, etc; and each row is the FR at a given timepoint;
  final P2K and K2E connection matrices.
"""
"""
Comment: 1.for mnist, the book-keeping differs from the odor experiment set-up.
         Let nC = number of classes (1 - 10 for mnist).
         The class may change with each new digit, so there is
         be a counter that increments when stimMag changes from nonzero
         to zero. there are nC counters.
         2.The function uses the noise params to create a Wiener process, then evolves the FR equations with the added noise
         3.Inside the difference equations we use a piecewise linear pseudo sigmoid, rather than a true sigmoid, for speed.
Note re calculating added noise:
  We want noise to be proportional to the mean spontFR of each neuron. So
  we need to get an estimate of this mean spont FR first. Noise is not added while neurons settle to initial SpontFR
  values. Then noise is added, proportional to spontFR. After this  noise begins, meanSpontFRs converge to new values.
 So there is a 'stepped' system, as follows: 
      1. no noise, neurons converge to initial meanSpontFRs = ms1
      2. noise proportional to ms1. neurons converge to new meanSpontFRs = ms2
      3. noise is proportional to ms2. neurons may converge to new
         meanSpontFRs = ms3, but noise is not changed. stdSpontFRs are calculated from ms3 time period.
  This has the following effects on simResults:
      1. In the heat maps and time-courses this will give a period of uniform FRs.
      2. The meanSpontFRs and stdSpontFRs are not 'settled' until after the stopSpontMean3 timepoint.
"""
import numpy as np
import matplotlib.pyplot as plt
from piecewiseLinearPseudoSigmoid import piecewiseLinearPseudoSigmoid


def sdeEvolutionMnist(tspan, initCond, T, classMagMatrix,featureArray, octoHits, params, expParams, seedValue, DEBUG=False):
    np.random.seed(seedValue)

    hebStarts = expParams.hebStarts
    hebDurations = expParams.hebDurations

    hebTauPK = params.hebTauPK
    hebMaxPK = params.hebMaxPK    # max connection weights
    hebTauPIK = params.hebTauPIK  # no PIs for mnist
    hebMaxPIK = params.hebMaxPIK  # no PIs for mnist
    hebTauKE = params.hebTauKE
    hebMaxKE = params.hebMaxKE

    dieBackTauKE = params.dieBackTauKE
    dieBackTauPK = params.dieBackTauPK
    dieBackTauPIK = params.dieBackTauPIK  # no PIs for mnist


    sparsityTarget =  params.sparsityTarget
    octoSparsityTarget = params.octoSparsityTarget

    # unpack connection matrices from params:
    F2R = params.F2R  # note S (stimuli) for odor case is replaced by F (features) for MNIST version
    R2P = params.R2P
    R2PI = params.R2PI  # no PIs for mnist
    R2L = params.R2L
    octo2R = params.octo2R
    octo2P = params.octo2P
    octo2PI = params.octo2PI  # no PIs for mnist
    octo2L = params.octo2L
    octo2E = params.octo2E
    octoNegDiscount = params.octoNegDiscount
    L2P = params.L2P
    L2L = params.L2L
    L2PI = params.L2PI  # no PIs for mnist
    L2R = params.L2R
    G2PI = params.G2PI  # no PIs for mnist
    K2E = params.K2E
    P2K = params.P2K
    PI2K = params.PI2K  # no PIs for mnist
    octo2K = params.octo2K

    # decay constants:
    tauR = params.tauR
    tauP = params.tauP
    tauPI = params.tauPI  # no PIs for mnist
    tauL = params.tauL
    tauK = params.tauK
    tauE = params.tauE

    # coefficients for sigmoids:
    cR = params.cR
    cP = params.cP
    cL = params.cL
    cPI = params.cPI  # no PIs for mnist
    cK = params.cK

    # numbers of objects
    nC = classMagMatrix.shape[0]
    nF = params.nF
    nG = params.nG
    nPI = params.nPI
    nK = params.nK
    nE = params.nE
    nP = nG
    nL = nG
    nR = nG

    # noise in individual neuron FRs. These are vectors, one vector for each type:
    wRsig = params.noiseRvec
    wPsig = params.noisePvec
    wPIsig = params.noisePIvec  # no PIs for mnist
    wLsig = params.noiseLvec
    wKsig = params.noiseKvec
    wEsig = params.noiseEvec

    kGlobalDampVec = params.kGlobalDampVec  # uniform 1's currently, ie LH inhibition hits all KCs equally

    # steady-state RN FR, base + noise:
    Rspont = params.Rspont
    RspontRatios = Rspont / np.mean(Rspont)  # used to scale stim inputs

    # param for sigmoid that squashes inputs to neurons:
    slopeParam = params.slopeParam  # slope of sigmoid at 0 = slopeParam*c/4, where c = cR, cP, cL, etc
    kSlope = slopeParam * cK / 4
    pSlope = slopeParam * cP / 4
    piSlope = slopeParam * cPI / 4  # no PIs for mnist
    rSlope = slopeParam * cR / 4
    lSlope = slopeParam * cL / 4

    # end timepoints for the section used to define mean spontaneous firing rates, in order to calibrate noise.
    # To let the system settle, we recalibrate noise levels to current spontaneous FRs in stages.
    # This ensures that in steady state, noise levels are correct in relation to mean FRs.
    startPreNoiseSpontMean1 = expParams.startPreNoiseSpontMean1
    stopPreNoiseSpontMean1 = expParams.stopPreNoiseSpontMean1
    startSpontMean2 = expParams.startSpontMean2
    stopSpontMean2 = expParams.stopSpontMean2
    startSpontMean3 = expParams.startSpontMean3
    stopSpontMean3 = expParams.stopSpontMean3

    # -----------------------------------------------------------------------------------------------------------

    dt = T[1] - T[0]
    N = len(T)

    def wiener(w_sig, meanSpont_, old_, tau_, inputs_):
        d_ = dt * (-old_ * tau_ + inputs_)

        # Wiener noise:
        noiseStrength = np.multiply(w_sig, meanSpont_)
        dW_ = np.sqrt(dt) * np.multiply(noiseStrength, np.random.normal(size=d_.shape))
        if DEBUG:
            # ---- DEBUG ----
            dW_ = np.zeros(d_.shape)
            # -------------------

        # combine them:
        new_ = old_ + d_ + dW_
        return new_

    P = np.zeros((nP, N))
    PI = np.zeros((nPI, N))  # no PIs for mnist
    L = np.zeros((nL, N))
    R = np.zeros((nR, N))
    K = np.zeros((nK, N))
    E = np.zeros((nE, N))

    # initialize the FR matrices with initial conditions:
    P[:, 0] = initCond[0: nP]
    PI[:, 0] = initCond[nP: nP + nPI]  # no PIs for mnist
    L[:, 0] = initCond[nP + nPI: nP + nPI + nL]
    R[:, 0] = initCond[nP + nPI + nL: nP + nPI + nL + nR]
    K[:, 0] = initCond[nP + nPI + nL + nR: nP + nPI + nL + nR + nK]
    E[:, 0] = initCond[- nE:]
    P2Kmask = P2K > 0
    PI2Kmask = PI2K > 0  # no PIs for mnist
    K2Emask = K2E > 0
    newP2K = P2K  # P2K: nK x nP
    newPI2K = PI2K  # P2K: nK x nPI # no PIs for mnist
    newK2E = K2E

    # initialize the counters for the various classes:
    classCounter = np.zeros((classMagMatrix.shape[0], 1))

    # make a list of time points for which heb is active:
    hebRegion = np.zeros(len(T))
    for i in range(len(hebStarts)):
        hebRegion[(T >= hebStarts[i]) & (T <= hebStarts[i] + hebDurations[i])] = 1

    # debug step
    # plt.plot(T, hebRegion)
    # plt.show()

    meanCalc1Done = False  # flag to prevent redundant calcs of mean spont FRs
    meanCalc2Done = False
    meanCalc3Done = False
    meanSpontR = np.zeros((nR, 1))
    meanSpontP = np.zeros((nP, 1))
    meanSpontPI = np.zeros((nPI, 1))  # no PIs for mnist
    meanSpontL = np.zeros((nL, 1))
    meanSpontK = np.zeros((nK, 1))
    meanSpontE = np.zeros((nE, 1))
    ssMeanSpontP = np.zeros((nP, 1))  # ss: steady state
    ssStdSpontP = np.ones((nP, 1))

    maxSpontP2KtimesPval = 10  # placeholder until we have an estimate based on spontaneous PN firing rates

    # --------------------------------------------------------------------
    # The main evolution loop:
    # iterate through time steps to get the full evolution:
    for i in range(N-1):

        if T[i] < stopSpontMean3 + 5 or params.saveAllNeuralTimecourses:
            oldR = np.atleast_2d(R[:, i]).transpose()
            oldP = np.atleast_2d(P[:, i]).transpose()
            oldPI = np.atleast_2d(PI[:, i]).transpose()  # no PIs for mnist
            oldL = np.atleast_2d(L[:, i]).transpose()
            oldK = np.atleast_2d(K[:, i]).transpose()
        else:
            oldR = np.atleast_2d(R[:, -1]).transpose()
            oldP = np.atleast_2d(P[:, -1]).transpose()
            oldPI = np.atleast_2d(PI[:, -1]).transpose()  # no PIs for mnist
            oldL = np.atleast_2d(L[:, -1]).transpose()
            oldK = np.atleast_2d(K[:, -1]).transpose()

        oldE = np.atleast_2d(E[:, i]).transpose()
        oldT = T[i]

        oldP2K = newP2K  # these are inherited from the previous iteration
        oldPI2K = newPI2K  # no PIs for mnist
        oldK2E = newK2E
        # ---------------------------------------------------------
        # set flags to say:
        #    1. whether we are past the window where meanSpontFR is
        #       calculated, so noise should be weighted according to a first estimate of meanSpontFR (meanSpont1);
        #    2. whether we are past the window where meanSpontFR is recalculated to meanSpont2; and
        #    3. whether we are past the window where final stdSpontFR can be calculated.
        adjustNoiseFlag1 = oldT > stopPreNoiseSpontMean1
        adjustNoiseFlag2 = oldT > stopSpontMean2
        adjustNoiseFlag3 = oldT > stopSpontMean3

        if adjustNoiseFlag1 and (not meanCalc1Done):  # ie we have not yet calc'ed the noise weight vectors:
            inds = np.argwhere((T > startPreNoiseSpontMean1) & (T < stopPreNoiseSpontMean1))
            meanSpontP = np.mean(P[:, inds], axis=1).reshape((nP, 1))
            meanSpontR = np.mean(R[:, inds], axis=1).reshape((nR, 1))
            meanSpontPI = np.mean(PI[:, inds], axis=1).reshape((nPI, 1))  # no PIs for mnist
            meanSpontL = np.mean(L[:, inds], axis=1).reshape((nL, 1))
            meanSpontK = np.mean(K[:, inds], axis=1).reshape((nK, 1))
            meanSpontE = np.mean(E[:, inds], axis=1).reshape((nE, 1))
            meanCalc1Done = 1  # meanSpont_ here are mean spontaneous firing rates of neurons without Wiener noise

        if adjustNoiseFlag2 and (not meanCalc2Done):  # ie we want to calc new noise weight vectors. This stage is surplus.
            inds = np.argwhere((T > startSpontMean2) & (T < stopSpontMean2))
            meanSpontP = np.mean(P[:, inds], axis=1).reshape((nP, 1))
            meanSpontR = np.mean(R[:, inds], axis=1).reshape((nR, 1))
            meanSpontPI = np.mean(PI[:, inds], axis=1).reshape((nPI, 1))  # no PIs for mnist
            meanSpontL = np.mean(L[:, inds], axis=1).reshape((nL, 1))
            meanSpontK = np.mean(K[:, inds], axis=1).reshape((nK, 1))
            meanSpontE = np.mean(E[:, inds], axis=1).reshape((nE, 1))
            stdSpontP = np.std(P[:, inds], axis=1, ddof=1).reshape((nP, 1))
            meanCalc2Done = 1  # meanSpont_ here are mean spontaneous firing rates of neurons with Wiener noise added

        if adjustNoiseFlag3 and (not meanCalc3Done):  # we want to calc stdSpontP for use with LH channel and maybe for use in heb:
            inds = np.argwhere((T > startSpontMean3) & (T < stopSpontMean3))
            ssMeanSpontP = np.mean(P[:, inds], axis=1).reshape((nP, 1))  # 'ss' means steady state
            ssStdSpontP = np.std(P[:, inds], axis=1, ddof=1).reshape((nP, 1))
            ssMeanSpontPI = np.mean(PI[:, inds], axis=1).reshape((nPI, 1))  # no PIs for mnist
            ssStdSpontPI = np.std(PI[:, inds], axis=1, ddof=1).reshape((nPI, 1))  # no PIs for mnist
            # set a minimum damping on KCs based on spontaneous PN activity,
            # sufficient to silence the MB silent absent odor:
            temp = np.matmul(P2K, ssMeanSpontP)
            temp = np.sort(temp, axis=0)  # sorted array in ascending order
            ignoreTopN = 1  # ignore this many of the highest vals
            temp = temp[:len(temp)-ignoreTopN]  # ignore the top few outlier K inputs.
            maxSpontP2KtimesPval = max(temp)  # The minimum global damping on the MB.
            meanCalc3Done = 1

        # update classCounter:
        if i > 0:
            for j in range(nC):
                if classMagMatrix[j,i-1] == 0 and classMagMatrix[j,i] > 0:
                    classCounter[j] += 1

        # get values of feature inputs at time index i, as a col vector.
        # This allows for simultaneous inputs by different classes, but current
        # experiments apply only one class at a time.
        thisInput = np.zeros((nF, 1))
        thisStimClassInd = []
        for j in range(nC):
            if classMagMatrix[j, i] > 0:
                classMags = classMagMatrix[j, i] * featureArray[j, :, int(classCounter[j]-1)]
                classMags = np.atleast_2d(classMags).transpose()
                thisInput += classMags
                thisStimClassInd.append(j)

        # ------------------------------------
        # get value at t for octopamine:
        thisOctoHit = octoHits[i]  # octoHits is a vector with an octopamine magnitude for each time point.
        # ---------------------------------------------------------------
        # Note: quote from the paper: pre-MB connections (L2R, L2P,..) were essentially fixed due to slow learning rates
        # ---------------------------------------------------------------
        # dR:
        # inputs: S = stim,  L = lateral neurons, Rspont = spontaneous FR
        # NOTE: octo does not affect Rspont. It affects R's response to input odors.
        Rinputs = np.clip(a=1 - thisOctoHit * octo2R * octoNegDiscount, a_min=0, a_max=None)  # octo2R: nG x 1
        Rinputs = -np.multiply(np.matmul(L2R, oldL), Rinputs)  # L2R: nG x nG, oldL: nL(=nG) x 1
        neur_act = np.multiply(np.matmul(F2R, thisInput), RspontRatios)  # F2R: nR x nF, thisInput: nF x 1, RspontRatios: nG x 1
        # before simStarts, thisInput is simply a vector of zeros
        neur_act = np.multiply(neur_act, (1 + thisOctoHit * octo2R))
        Rinputs = Rinputs + neur_act + Rspont
        Rinputs = piecewiseLinearPseudoSigmoid(Rinputs, cR, rSlope)

        # Adding Wiener noise
        newR = wiener(wRsig, meanSpontR, oldR, tauR, Rinputs)  # wRsig: nR x 1, meanSpontR: nR x 1
        # ---------------------------------------------------------------
        # dP:
        Pinputs = np.clip(a=1 - thisOctoHit * octo2P * octoNegDiscount, a_min=0, a_max=None)  # octo2P: nG x 1
        Pinputs = -np.multiply(np.matmul(L2P, oldL), Pinputs)
        Pinputs = Pinputs + np.multiply(np.multiply(R2P, oldR), (1 + thisOctoHit * octo2P))
        Pinputs = piecewiseLinearPseudoSigmoid(Pinputs, cP, pSlope)

        # Adding Wiener noise
        newP = wiener(wPsig, meanSpontP, oldP, tauP, Pinputs)  # wPsig: nP x 1, meanSpontP: nP x 1
        # ---------------------------------------------------------------
        # dPI:  # no PIs for mnist
        PIinputs = np.clip(a=1 - thisOctoHit * octo2PI * octoNegDiscount, a_min=0, a_max=None)  # octo2PI: nPI x 1
        PIinputs = -np.multiply(np.matmul(L2PI, oldL), PIinputs)  # L2PI: nPI x nG, oldL: nL(=nG) x 1
        PIinputs = PIinputs + np.multiply(np.matmul(R2PI, oldR), (1 + thisOctoHit * octo2PI))
        PIinputs = piecewiseLinearPseudoSigmoid(PIinputs, cPI, piSlope)

        # Adding Wiener noise
        newPI = wiener(wPIsig, meanSpontPI, oldPI, tauPI, PIinputs)  # wPIsig: nPI x 1, meanSpontR: nPI x 1
        # ---------------------------------------------------------------
        # dL:
        Linputs = np.clip(a=1 - thisOctoHit * octo2L * octoNegDiscount, a_min=0, a_max=None)  # octo2L: nG x 1
        Linputs = -np.multiply(np.matmul(L2L, oldL), Linputs)  # L2L: nG x nG, oldL: nL(=nG) x 1
        Linputs = Linputs + np.multiply(np.multiply(R2L, oldR), (1 + thisOctoHit * octo2L))  # R2L: nG x 1
        Linputs = piecewiseLinearPseudoSigmoid(Linputs, cL, lSlope)

        # Adding Wiener noise
        newL = wiener(wLsig, meanSpontL, oldL, tauL, Linputs)
        # ---------------------------------------------------------------
        # Enforce sparsity on the KCs:
        # Global damping on KCs is controlled by sparsityTarget (during
        # octopamine, by octSparsityTarget). Assume that inputs to KCs form a
        # gaussian, and use a threshold calculated via std devs to enforce the correct sparsity.

        # Delays from AL -> MB and AL -> LH -> MB (~30 mSec) are ignored.

        # the # st devs to give the correct sparsity on MB
        from scipy.special import erfinv
        numNoOctoStds = np.sqrt(2) * erfinv(1 - 2 * sparsityTarget)
        numOctoStds = np.sqrt(2) * erfinv(1 - 2 * octoSparsityTarget)
        # select either octo or no-octo
        numStds = (1 - thisOctoHit) * numNoOctoStds + thisOctoHit * numOctoStds
        # a minimum damping based on spontaneous PN activity, so that the MB is silent absent odor
        minDamperVal = 1.2 * maxSpontP2KtimesPval
        thisKinput = np.matmul(oldP2K, oldP) - np.matmul(oldPI2K, oldPI)
        damper = np.mean(thisKinput) + numStds * np.std(thisKinput, ddof=1)
        damper = max(damper, minDamperVal)

        Kinputs = np.multiply(np.matmul(oldP2K, oldP), (1 + octo2K * thisOctoHit))  # but note that octo2K == 0
        dampening = damper * kGlobalDampVec + np.matmul(oldPI2K, oldPI)
        pos_octo = np.clip(a=1 - octo2K * thisOctoHit, a_min=0, a_max=None)  # octo2K: nK x 1
        Kinputs = Kinputs - np.multiply(dampening, pos_octo)
        Kinputs = piecewiseLinearPseudoSigmoid(Kinputs, cK, kSlope)

        # Adding Wiener noise
        newK = wiener(wKsig, meanSpontK, oldK, tauK, Kinputs)
        # ---------------------------------------------------------------
        # readout neurons E (EN = 'extrinsic neurons'):
        # These are readouts, so there is no sigmoid.
        # octo2E == 0, since we are not stimulating ENs with octo.
        # dWE == 0 since we assume no noise in ENs.

        Einputs = np.matmul(oldK2E, oldK)  # oldK2E: nE x nK
        dE = dt * (-oldE * tauE + Einputs)

        # Adding Wiener noise
        dWE = 0
        # combine them
        newE = oldE + dE + dWE

        # -------------------------------------------------------------------------------------
        # HEBBIAN UPDATES (training):
        # Apply Hebbian learning to P2K, K2E:
        # For ease, use 'newK' and 'oldP', 'newE' and 'oldK', ie 1 timestep of delay.
        # We restrict hebbian growth in K2E to connections into the EN of the training stimulus
        if hebRegion[i]:  # Hebbian updates are active for about half the duration of each stimulus
            # the PN contribution to hebbian is based on raw FR:
            tempP = oldP
            tempPI = oldPI  # no PIs for mnist
            nonNegNewK = np.clip(a=newK, a_min=0, a_max=None)  # since newK has not yet been made non-neg
            # ---------------------------------------------------------------
            # dP2K:
            dp2k = (1 / hebTauPK) * np.matmul(nonNegNewK, tempP.transpose())  # nonNegNewK: nK x 1, tempP: nP x 1, dp2k: nK x nP
            dp2k = np.multiply(dp2k, P2Kmask)  # if original synapse does not exist, it will never grow.

            # decay some P2K connections if wished: (not used for mnist experiments)
            if dieBackTauPK > 0:
                oldP2K = oldP2K - oldP2K * (1 / dieBackTauPK) * dt

            newP2K = oldP2K + dp2k
            newP2K = np.clip(a=newP2K, a_min=0, a_max=hebMaxPK)
            # ---------------------------------------------------------------
            # dPI2K: # no PIs for mnist
            dpi2k = (1 / hebTauPIK) * np.matmul(nonNegNewK, tempPI.transpose())
            dpi2k = np.multiply(dpi2k, PI2Kmask)  # if original synapse does not exist, it will never grow.
            # kill small increases:
            temp = oldPI2K  # this detour prevents dividing by zero
            temp[temp == 0] = 1
            keepMask = np.divide(dpi2k, temp)
            dpi2k = np.multiply(dpi2k, keepMask)
            if dieBackTauPIK > 0:
                oldPI2K = oldPI2K - oldPI2K * (1 / dieBackTauPIK) * dt

            newPI2K = oldPI2K + dpi2k
            newPI2K = np.clip(a=newPI2K, a_min=0, a_max=hebMaxPIK)
            # ---------------------------------------------------------------
            # dK2E:
            tempK = oldK
            dk2e = (1 / hebTauKE) * np.matmul(newE, tempK.transpose())  # oldK is already nonNeg
            dk2e = np.multiply(dk2e, K2Emask)
            # restrict changes to just the i'th row of K2E, where i = ind of training stim
            restrictK2Emask = np.zeros(K2E.shape)
            restrictK2Emask[thisStimClassInd, :] = 1
            dk2e = np.multiply(dk2e, restrictK2Emask)
            # ----------------------------------------
            # inactive connections for this EN die back:
            if dieBackTauKE > 0:
                # restrict dieBacks to only the trained EN:
                targetMask = np.zeros(dk2e.flatten().shape)  # row vector
                targetMask[dk2e.flatten() == 0] = 1
                targetMask = np.reshape(targetMask, dk2e.shape)
                targetMask = np.multiply(targetMask, restrictK2Emask)
                dieBack = (oldK2E + 2) * (1 / dieBackTauKE) * dt
                oldK2E = oldK2E - np.multiply(targetMask, dieBack)

            newK2E = oldK2E + dk2e
            newK2E = np.clip(a=newK2E, a_min=0, a_max=hebMaxKE)

        else:  # case: no heb or no octo
            newP2K = oldP2K
            newPI2K = oldPI2K  # no PIs for mnist
            newK2E = oldK2E

        # --------------------------------------------------------------------
        # update the evolution matrices, disallowing negative FRs.
        if T[i] < stopSpontMean3 + 5 or params.saveAllNeuralTimecourses:
            newR[newR < 0] = 0
            R[:, i + 1] = newR.flatten()
            newP[newP < 0] = 0
            P[:, i + 1] = newP.flatten()
            newPI[newPI < 0] = 0  # no PIs for mnist
            PI[:, i + 1] = newPI.flatten()
            newL[newL < 0] = 0
            L[:, i + 1] = newL.flatten()
            newK[newK < 0] = 0
            K[:, i + 1] = newK.flatten()
        # case: do not save AL and MB neural timecourses after the noise calibration is done, to save on memory
        else:
            R = np.clip(a=newR, a_min=0, a_max=None)
            P = np.clip(a=newP, a_min=0, a_max=None)
            PI = np.clip(a=newPI, a_min=0, a_max=None)
            L = np.clip(a=newL, a_min=0, a_max=None)
            K = np.clip(a=newK, a_min=0, a_max=None)

        E[:, i + 1] = newE.flatten()  # always save full EN timecourses
        # Time-step simulation is now over.

    class ThisRun:
        def __init__(self):
            self.Y = None
            self.T = None
            self.E = None
            self.P2Kfinal = None
            self.K2Efinal = None

    thisRun = ThisRun()

    # combine so that each row of fn output Y is a col of [P; PI; L; R; K]:
    if params.saveAllNeuralTimecourses:
        Y = np.vstack((P, PI, L, R, K, E))
        Y = Y.transpose()
        thisRun.Y = Y.astype(float)  # convert to singles to save memory
    else:
        thisRun.Y = []

    thisRun.T = T.astype(float)  # T is stored as a row vector
    thisRun.E = E.transpose().astype(float)  # length(T) x nE matrix
    thisRun.P2Kfinal = oldP2K.astype(float)
    thisRun.K2Efinal = oldK2E.astype(float)

    return thisRun




























































    

