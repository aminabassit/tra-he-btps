
import pickle
import gc

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from datetime import datetime

import os



import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

from utils import *


dataset = 'VGGFace2' # 'LFW'
dataDir = f'./embeddings/{dataset}/'# path to the embedding of the dataset


dimF = 512
precision = 400  


quantApproach = 'Precision'


thrList = ['FMR_1Percent', 'FMR_01Percent', 'FMR_001Percent']
thrPercList = ['@0.1%FMR', '@0.01%FMR', '@0.001%FMR']


if dataset == 'VGGFace2':
    thrIPList = [0.2391948024240369, 0.30819380242219074, 0.4966788024171478]
elif dataset == 'LFW':
    thrIPList = [0.24106651582972638, 0.30361251582805293, 0.3716005158262339]
else:
    print(f'Add the thresholds @0.1%FMR, @0.01%FMR, and @0.001%FMR for the IP comparator tested on your dataset')
thrPrecList = [round(thrIP*(precision**2)) for thrIP in thrIPList]







subjectIDs = os.listdir(dataDir)
subjectIDs.sort()
nSubj = 200

iterPerSubj = 10
kStart = 50
kEnd = 2000
nK = 10


kList = list(range(kStart, kEnd+1, nK))
indxK = kList.index(kEnd)

medianKPerSubj = {thr:None for thr in thrPercList}

for t, thr, thrPrec in zip(thrList, thrPercList, thrPrecList):
    
    resDir = f'./results/UpperBoundAttack/{dataset}/UpperBoundAttack_{quantApproach}_Threshold_{t}_Iterations_{iterPerSubj}'

    print(f'====================\nExperiment threshold at {thr} on {dataset} dataset')
    subScores = {subj:None for subj in subjectIDs}
    subParsedK = {subj:None for subj in subjectIDs}

    for subject in subjectIDs[:nSubj]:
        subjDir = f'{resDir}/subjects/{subject}'
        os.makedirs(subjDir, exist_ok=True)   
        print(subject)
        scoresBest = []
        kParsedBest = []
        for i in range(iterPerSubj):
            scores, kParsed = upperBoundAttackPrec(dataDir, subject, precision, thrPrec, kStart, kEnd, nK)
            itScPth = f'{subjDir}/iter_{i}_subScores_{subject}.txt'
            itKPth = f'{subjDir}/iter_{i}_subParsedK_{subject}.txt'
            np.savetxt(itScPth, scores, delimiter=',', fmt='%d')
            np.savetxt(itKPth, kParsed, delimiter=',', fmt='%d')
            scoresBest.append(scores[-1])
            kParsedBest.append(kParsed[-1])
        subScores[subject] = scoresBest
        subParsedK[subject] = kParsedBest
        
    resultsFile = f'{resDir}/All_subScores_and_kParsed.pkl'
    delEmpty(subjectIDs, subScores, subParsedK)
    pickle.dump((subScores, subParsedK), open(resultsFile, 'wb'))

    medianKPerSubj[thr] = getMedianPerSubj(subParsedK)

plotTitle = f'Estimated k for bypassing systems with thresholds\n {thrPercList}\n for a success rate fixed at 100% (quantization approach {quantApproach})'
plotBoxPlots([medianKPerSubj[thr] for thr in thrPercList], [f'{quantApproach}\n{thr}' for thr in thrPercList], plotTitle=plotTitle, savename=f'./results/UpperBoundAttack/{dataset}/medianBoxPlots_{dataset}_QuantApp_{quantApproach}.pdf')
