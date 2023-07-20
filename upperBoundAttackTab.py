
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

nB = 3
dQ = 0.001


quantApproach = 'MFIP'

tableFile = f'./MFIP/MFIP_nB_{nB}_dimF_{dimF}.pkl'         
bordersFile = f'./MFIP/Borders_nB_{nB}_dimF_{dimF}.pkl'      

tabMFIP = pickle.load(open(tableFile, "rb"))
borders = pickle.load(open(bordersFile, "rb"))        
tabMFIP = np.round(tabMFIP/dQ).astype(int) 
print('Table and borders loaded...')


thrList = ['FMR_1Percent', 'FMR_01Percent', 'FMR_001Percent']
thrPercList = ['@0.1%FMR', '@0.01%FMR', '@0.001%FMR']


if dataset == 'VGGFace2':
    thrTabList = [224, 287, 468]
elif dataset == 'LFW':
    thrTabList = [226, 284, 348]
else:
    print(f'Add the thresholds @0.1%FMR, @0.01%FMR, and @0.001%FMR for the IP comparator tested on your dataset')








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

for t, thr, thrTab in zip(thrList, thrPercList, thrTabList):
    
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
            scores, kParsed = upperBoundAttackTab(dataDir, subject, borders, tabMFIP, thrTab, kStart, kEnd, nK)
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
