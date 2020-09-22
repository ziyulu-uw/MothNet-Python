"""
Generates the various connection matrices, given modelParams, and appends them to modelParams.
Input: 'modelParams', an instance of the class ModelParams
Output: 'params', an instance of the class Params that includes connection matrices and other model info necessary to FR evolution and plotting.
--------------------------------------------------------------------
step 1: unpack input modelParams
step 2: build the matrices
step 3: pack the matrices into an instance of the class Params for output
These steps are kept separate for clarity of step 2.
"""
import numpy as np


def initializeConnectionMatrices(modelParams, DEBUG=False):

    if not DEBUG:
        # step 1: unpack modelParams (no editing necessary in this section):
        nG = modelParams.nG
        nP = modelParams.nP
        nR = modelParams.nR
        nPI = modelParams.nPI
        nK = modelParams.nK
        nF = modelParams.nF
        nE = modelParams.nE

        tauR = modelParams.tauR
        tauP = modelParams.tauP
        tauPI = modelParams.tauPI
        tauL = modelParams.tauL
        tauK = modelParams.tauK

        cR = modelParams.cR
        cP = modelParams.cP
        cPI = modelParams.cPI
        cL = modelParams.cL
        cK = modelParams.cK

        spontRdistFlag = modelParams.spontRdistFlag
        spontRmu = modelParams.spontRmu
        spontRstd = modelParams.spontRstd
        spontRbase = modelParams.spontRbase

        RperFFrMu = modelParams.RperFFrMu
        RperFRawNum = modelParams.RperFRawNum
        F2Rmu = modelParams.F2Rmu
        F2Rstd = modelParams.F2Rstd
        R2Gmu = modelParams.R2Gmu
        R2Gstd = modelParams.R2Gstd
        R2Pmult = modelParams.R2Pmult
        R2Pstd = modelParams.R2Pstd
        R2PImult = modelParams.R2PImult
        R2PIstd = modelParams.R2PIstd
        R2Lmult = modelParams.R2Lmult
        R2Lstd = modelParams.R2Lstd

        L2Gfr = modelParams.L2Gfr
        L2Gmu = modelParams.L2Gmu
        L2Gstd = modelParams.L2Gstd

        L2Rmult = modelParams.L2Rmult
        L2Rstd = modelParams.L2Rstd

        L2Pmult = modelParams.L2Pmult
        L2Pstd = modelParams.L2Pstd
        L2PImult = modelParams.L2PImult
        L2PIstd = modelParams.L2PIstd
        L2Lmult = modelParams.L2Lmult
        L2Lstd = modelParams.L2Lstd

        GsensMu = modelParams.GsensMu
        GsensStd = modelParams.GsensStd

        G2PImu = modelParams.G2PImu
        G2PIstd = modelParams.G2PIstd

        KperEfrMu = modelParams.KperEfrMu
        K2Emu = modelParams.K2Emu
        K2Estd = modelParams.K2Estd

        octo2Gmu = modelParams.octo2Gmu
        octo2Gstd = modelParams.octo2Gstd
        octo2Pmult = modelParams.octo2Pmult
        octo2Pstd = modelParams.octo2Pstd
        octo2PImult = modelParams.octo2PImult
        octo2PIstd = modelParams.octo2PIstd
        octo2Lmult = modelParams.octo2Lmult
        octo2Lstd = modelParams.octo2Lstd
        octo2Rmult = modelParams.octo2Rmult
        octo2Rstd = modelParams.octo2Rstd
        octo2Kmu = modelParams.octo2Kmu
        octo2Kstd = modelParams.octo2Kstd
        octo2Emu = modelParams.octo2Emu
        octo2Estd = modelParams.octo2Estd

        noiseR = modelParams.noiseR
        RnoiseStd = modelParams.RnoiseStd
        noiseP = modelParams.noiseP
        PnoiseStd = modelParams.PnoiseStd
        noisePI = modelParams.noisePI
        PInoiseStd = modelParams.PInoiseStd
        noiseL = modelParams.noiseL
        LnoiseStd = modelParams.LnoiseStd
        noiseK = modelParams.noiseK
        KnoiseStd = modelParams.KnoiseStd
        noiseE = modelParams.noiseE
        EnoiseStd = modelParams.EnoiseStd

        KperPfrMu = modelParams.KperPfrMu
        KperPIfrMu = modelParams.KperPIfrMu
        GperPIfrMu = modelParams.GperPIfrMu
        P2Kmu = modelParams.P2Kmu
        P2Kstd = modelParams.P2Kstd
        PI2Kmu = modelParams.PI2Kmu
        PI2Kstd = modelParams.PI2Kstd
        kGlobalDampFactor = modelParams.kGlobalDampFactor
        kGlobalDampStd = modelParams.kGlobalDampStd
        hebMaxPK = modelParams.hebMaxPK
        hebMaxPIK = modelParams.hebMaxPIK
        hebMaxKE = modelParams.hebMaxKE

        # ------------------------------------------------------

        # Step 2: Generate connection matrices
        # Comment: Since there are many zero connections (ie matrices are usually
        # not all-to-all) we often need to apply masks to preserve the zero connections.

        # first make a binary mask S2Rbinary:
        # here only implement the case where RperFFrMu = 0
        F2Rbinary = np.zeros((nR, nF))
        counts = np.zeros((nR, 1))  # to track how many S are hitting each R
        # calc max # of S per any given glom:
        maxFperR = np.ceil(nF * RperFRawNum / nR)
        # connect one R to each S, then go through again to connect a 2nd R to each S, etc
        for i in range(int(maxFperR)):
            for j in range(nF):
                inds = np.nonzero(counts < maxFperR)[0]
                a = np.random.randint(len(inds))
                F2Rbinary[inds[a], j] = 1
                counts[inds[a]] += 1

        # now mask a matrix of gaussian weights:
        F2R = np.multiply((F2Rmu*F2Rbinary + F2Rstd*np.random.normal(size=F2Rbinary.shape)), F2Rbinary)
        F2R[F2R < 0] = 0  # to prevent any negative weights

        # spontaneous FRs for Rs:
        if spontRdistFlag == 1:  # gaussian distribution:
            Rspont = spontRmu * np.ones((nG, 1)) + spontRstd * np.random.normal(size=(nG, 1))
            Rspont[Rspont < 0] = 0
        else:  # == 2 gamma distribution:
            a = spontRmu / spontRstd  # shape parameter
            b = spontRstd  # scale parameter
            Rspont = spontRbase + np.random.gamma(a, b, size=(nG, 1))

        # R2G connection vector. nG x 1 col vector:
        R2G = R2Gmu * np.ones((nG, 1)) + R2Gstd * np.random.normal(size=(nG, 1))  # col vector, each entry is strength of an R in its G
        R2G[R2G < 0] = 0  # prevents negative R2G effects

        # now make R2P, etc, all are cols nG x 1:
        R2P = np.multiply(R2Pmult + R2Pstd * np.random.normal(size=(nG, 1)), R2G)
        R2L = np.multiply(R2Lmult + R2Lstd * np.random.normal(size=(nG, 1)), R2G)
        R2PIcol = np.multiply(R2PImult + R2PIstd * np.random.normal(size=(nG, 1)), R2G)
        # this interim nG x 1 col vector gives the effect of each R on any PI in the R's glom.
        # It will be used below with G2PI to get full effect of Rs on PIs

        # Construct L2G = nG x nG matrix of lateral neurons. This is a precursor to L2P etc
        L2G = L2Gmu + L2Gstd * np.random.normal(size=(nG, nG))
        L2G[L2G < 0] = 0
        # set diagonal = 0:
        L2G = L2G - np.diag(L2G.diagonal())

        # are enough of these values 0?
        numZero = nG*nG - np.count_nonzero(L2G) - nG # ignore the diagonal zeroes
        numToKill = np.floor((1 - L2Gfr) * (nG*nG - nG) - numZero)
        if numToKill > 0:  # case: we need to set more vals to 0 to satisfy frLN constraint:
            L2G = L2G.flatten()
            randList = np.random.rand(len(L2G)) < numToKill / (nG*nG - nG - numZero)
            for i in range(len(L2G)):
                if L2G[i] > 0 & randList[i] == 1:
                    L2G[i] = 0
        L2G = np.reshape(L2G, (nG, nG))
        # Structure of L2G:
        # L2G(i,j) = the synaptic LN weight going to G(i) from G(j),
        # ie the row gives the 'destination glom', the col gives the 'source glom'

        # gloms vary widely in their sensitivity to gaba (Hong, Wilson 2014).
        # multiply the L2* vectors by Gsens + GsensStd:
        gabaSens = GsensMu + GsensStd * np.random.normal(size=(nG, 1))
        L2GgabaSens = np.multiply(L2G, np.tile(gabaSens, (1,nG)))
        # ie each row is multiplied by a different value, since each row represents a destination glom

        # this version of L2G does not encode variable sens to gaba, but is scaled by GsensMu:
        L2G = L2G * GsensMu

        # now generate all the L2etc matrices:
        L2R = np.multiply(L2Rmult + L2Rstd * np.random.normal(size=(nG, nG)), L2GgabaSens)
        L2R[L2R < 0] = 0

        L2P = np.multiply(L2Pmult + L2Pstd * np.random.normal(size=(nG, nG)), L2GgabaSens)
        L2P[L2P < 0] = 0

        L2L = np.multiply(L2Lmult + L2Lstd * np.random.normal(size=(nG, nG)), L2GgabaSens)
        L2L[L2L < 0] = 0

        L2PI = np.multiply(L2PImult + L2PIstd * np.random.normal(size=(nG, nG)), L2GgabaSens) # Masked by G2PI later
        L2PI[L2PI < 0] = 0

        # Ps (excitatory):
        P2KconnMatrix = np.random.rand(nK, nP) < KperPfrMu  # each col is a P, and a fraction of the entries will = 1.
                                                            # different cols (PNs) will have different numbers of 1's (~binomial dist).
        P2K = P2Kmu + P2Kstd*np.random.normal(size=(nK, nP))
        P2K[P2K < 0] = 0
        P2K = np.multiply(P2K, P2KconnMatrix)
        # cap P2K values at hebMaxPK, so that hebbian training never decreases weights:
        P2K = np.clip(P2K, a_min=None, a_max=hebMaxPK)

        # PIs (inhibitory): (not used in mnist)
        # 0. These are more complicated, since each PI is fed by several Gs
        # 1. a) We map from Gs to PIs (binary, one G can feed multiple PI) with G2PIconn
        # 1. b) We give weights to the G-> PI connections. these will be used to calc PI firing rates.
        # 2. a) We map from PIs to Ks (binary), then
        # 2. b) multiply the binary map by a random matrix to get the synapse weights.

        # In the moth, each PI is fed by many gloms
        G2PIconn = np.random.rand(nPI, nG) < GperPIfrMu  # step 1a
        G2PI = G2PIstd*np.random.normal(size=(nPI, nG)) + G2PImu
        G2PI[G2PI < 0] = 0  # step 1b
        G2PI = np.multiply(G2PIconn, G2PI)  # mask with double values, step 1b (cont)
        G2PI = np.divide(G2PI, np.tile(np.atleast_2d(np.sum(G2PI,axis=1)).transpose(), (1, G2PI.shape[1])))
        # mask PI matrices:
        L2PI = np.matmul(G2PI, L2G)  # nPI x nG
        R2PI = np.multiply(G2PI, R2PIcol.transpose())
        # nPI x nG matrices, (i,j)th entry = effect from j'th object to i'th object.
        # eg, the rows with non-zero entries in the j'th col of L2PI are those PIs affected by the LN from the j'th G.
        # eg, the cols with non-zero entries in the i'th row of R2PI are those Rs feeding gloms that feed the i'th PI.

        if nPI > 0:
            PI2Kconn = np.random.rand(nK, nPI) < KperPIfrMu  # step 2a
            PI2K = PI2Kmu + PI2Kstd*np.random.normal(size=(nK, nPI))
            PI2K[PI2K < 0] = 0  # step 2b
            PI2K = np.multiply(PI2K, PI2Kconn)  # mask
            PI2K = np.clip(PI2K, a_min=None, a_max=hebMaxPIK)
            # 1. G2PI maps the Gs to the PIs. It is nPI x nG, doubles.
            #    The weights are used to find the net PI firing rate
            # 2. PI2K maps the PIs to the Ks. It is nK x nPI with entries >= 0.
            #    G2K = PI2K*G2PI; % binary map from G to K via PIs. not used

        # K2E (excit):
        K2EconnMatrix = np.random.rand(nE, nK) < KperEfrMu  # each col is a K, and a fraction of the entries will = 1.
                                                            # different cols (KCs) will have different numbers of 1's (~binomial dist).
        K2E = K2Emu + K2Estd*np.random.normal(size=(nE, nK))
        K2E[K2E < 0] = 0
        K2E = np.multiply(K2E, K2EconnMatrix)
        K2E = np.clip(K2E, a_min=None, a_max=hebMaxKE)
        # K2E maps from the KCs to the ENs. Given firing rates KC, K2E gives the effect on the various ENs.
        # It is nE x nK with entries >= 0.

        # octopamine to Gs and to Ks:
        octo2G = octo2Gmu + octo2Gstd*np.random.normal(size=(nG, 1))
        octo2G[octo2G < 0] = 0  # intermediate step
        octo2K = octo2Kmu + octo2Kstd*np.random.normal(size=(nK, 1))
        octo2K[octo2K < 0] = 0
        # each of these is a col vector with entries >= 0
        octo2P = octo2Pmult*octo2G + octo2Pstd*np.random.normal(size=(nG, 1))
        octo2P[octo2P < 0] = 0  # effect of octo on P, includes gaussian variation from P to P

        octo2L = octo2Lmult*octo2G + octo2Lstd*np.random.normal(size=(nG, 1))
        octo2L[octo2L < 0] = 0

        octo2R = octo2Rmult*octo2G + octo2Rstd*np.random.normal(size=(nG, 1))
        octo2R[octo2R < 0] = 0
        # mask and weight octo2PI:
        octo2PIwts = np.multiply(G2PI, octo2PImult*octo2G.transpose())
        # normalize this by taking average:
        octo2PI = np.divide(np.sum(octo2PIwts,axis=1), np.sum(G2PIconn,axis=1))
        octo2PI = octo2PI.reshape((nPI, 1))
        # net, averaged effect of octo on PI. Includes varying effects of octo on Gs & varying contributions of Gs to PIs.
        # the 1st term = summed weights (col), 2nd term = # Gs contributing to each PI (col)

        octo2E = octo2Emu + octo2Estd*np.random.normal(size=(nE, 1))
        octo2E[octo2E < 0] = 0

        # each neuron has slightly different noise levels for sde use. Define noise vectors for each type:
        # Gaussian versions:
        noisePIvec = noisePI + PInoiseStd * np.random.normal(size=(nPI, 1))
        noisePIvec[noisePIvec < 0] = 0

        noiseKvec = noiseK + KnoiseStd * np.random.normal(size=(nK, 1))
        noiseKvec[noiseKvec < 0] = 0

        noiseEvec = noiseE + EnoiseStd * np.random.normal(size=(nE, 1))
        noiseEvec[noiseEvec < 0] = 0
        # Gamma versions:
        a = noiseR / RnoiseStd
        b = RnoiseStd
        noiseRvec = np.random.gamma(a, b, size=(nR, 1))
        noiseRvec[noiseRvec > 15] = 0

        a = noiseP / PnoiseStd
        b = PnoiseStd
        noisePvec = np.random.gamma(a, b, size=(nP, 1))
        noisePvec[noisePvec > 15] = 0

        a = noiseL / LnoiseStd
        b = LnoiseStd
        noiseLvec = np.random.gamma(a, b, size=(nG, 1))

        kGlobalDampVec = kGlobalDampFactor + kGlobalDampStd * np.random.normal(size=(nK, 1))  # each KC may be affected a bit differently by LH inhibition

        # ------------------------------------------------------
        # append these matrices to 'modelParams' struct (no editing necessary in this section):
        modelParams.F2R = F2R
        modelParams.R2P = R2P
        modelParams.R2PI = R2PI
        modelParams.R2L = R2L
        modelParams.octo2R = octo2R
        modelParams.octo2P = octo2P
        modelParams.octo2PI = octo2PI
        modelParams.octo2L = octo2L
        modelParams.octo2K = octo2K
        modelParams.octo2E = octo2E
        modelParams.L2P = L2P
        modelParams.L2L = L2L
        modelParams.L2PI = L2PI
        modelParams.L2R = L2R
        modelParams.G2PI = G2PI
        modelParams.P2K = P2K
        modelParams.PI2K = PI2K
        modelParams.K2E = K2E
        modelParams.Rspont = Rspont  # col vector

        modelParams.noiseRvec = noiseRvec
        modelParams.noisePvec = noisePvec
        modelParams.noisePIvec = noisePIvec
        modelParams.noiseLvec = noiseLvec
        modelParams.noiseKvec = noiseKvec
        modelParams.noiseEvec = noiseEvec
        modelParams.kGlobalDampVec = kGlobalDampVec

    if DEBUG:

        # ---- DEBUG ----
        import scipy.io as sio
        mat_mp = sio.loadmat('MP.mat')
        mp_struct = mat_mp['modelParams']
        modelParams.F2R = np.asarray(mp_struct['F2R'])[0][0]
        modelParams.R2P = np.asarray(mp_struct['R2P'])[0][0]
        modelParams.R2PI = np.asarray(mp_struct['R2PI'])[0][0]
        modelParams.R2L = np.asarray(mp_struct['R2L'])[0][0]
        modelParams.octo2R = np.asarray(mp_struct['octo2R'])[0][0]
        modelParams.octo2P = np.asarray(mp_struct['octo2P'])[0][0]
        modelParams.octo2PI = np.asarray(mp_struct['octo2PI'])[0][0]
        modelParams.octo2L = np.asarray(mp_struct['octo2L'])[0][0]
        modelParams.octo2K = np.asarray(mp_struct['octo2K'])[0][0]
        modelParams.octo2E = np.asarray(mp_struct['octo2E'])[0][0]
        modelParams.L2P = np.asarray(mp_struct['L2P'])[0][0]
        modelParams.L2L = np.asarray(mp_struct['L2L'])[0][0]
        modelParams.L2PI = np.asarray(mp_struct['L2PI'])[0][0]
        modelParams.L2R = np.asarray(mp_struct['L2R'])[0][0]
        modelParams.G2PI = np.asarray(mp_struct['G2PI'])[0][0]
        modelParams.P2K = np.asarray(mp_struct['P2K'])[0][0]
        modelParams.PI2K = np.asarray(mp_struct['PI2K'])[0][0]
        modelParams.K2E = np.asarray(mp_struct['K2E'])[0][0]
        modelParams.Rspont = np.asarray(mp_struct['Rspont'])[0][0]  # col vector

        modelParams.noiseRvec = np.asarray(mp_struct['noiseRvec'])[0][0]
        modelParams.noisePvec = np.asarray(mp_struct['noisePvec'])[0][0]
        modelParams.noisePIvec = np.asarray(mp_struct['noisePIvec'])[0][0]
        modelParams.noiseLvec = np.asarray(mp_struct['noiseLvec'])[0][0]
        modelParams.noiseKvec = np.asarray(mp_struct['noiseKvec'])[0][0]
        modelParams.noiseEvec = np.asarray(mp_struct['noiseEvec'])[0][0]
        modelParams.kGlobalDampVec = np.asarray(mp_struct['kGlobalDampVec'])[0][0]
        # -------------------

    return modelParams






























