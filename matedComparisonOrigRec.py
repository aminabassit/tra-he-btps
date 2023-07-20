import pickle
import gc

import numpy as np
from matplotlib import pyplot as plt
from itertools import repeat
import multiprocessing as mp

from scipy import stats

from datetime import datetime

import os



import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')
from utils import *








def main(): 
    

    now = datetime.now()
    timeTag = now.strftime("%d%m%Y_%H%M%S") 

    dataset = 'VGGFace2' # 'LFW
    dataDir = f'./embeddings/{dataset}/'# path to the embedding of the dataset
    
    dimF = 512
    nB = 3
    dQ = 0.001
    precision = 400 
    
    kFakeTemp = 512 # [400, 512, 600, 700, 1000, 3000, 5000]
    
    resDir = f'./results/Intermediate/{dataset}/matedScores_nB_{nB}_dQ_{dQ}_Precision_{precision}_k_{kFakeTemp}'
    plotDir = f'{resDir}/plots'
    os.makedirs(resDir, exist_ok=True)
    os.makedirs(plotDir, exist_ok=True)



    tableFile = f'./MFIP/MFIP_nB_{nB}_dimF_{dimF}.pkl'         
    bordersFile = f'./MFIP/Borders_nB_{nB}_dimF_{dimF}.pkl'      

    tabMFIP = pickle.load(open(tableFile, "rb"))
    borders = pickle.load(open(bordersFile, "rb"))        
    tabMFIP = np.round(tabMFIP/dQ).astype(int) 
    print('Table and borders are loaded...')
    

    

    subjectIDs = os.listdir(dataDir)
    subjectIDs.sort()
    nSave = 10
    nSubj = len(subjectIDs)
    


    print(f'Mated comparison NonQuantized vs MFIP vs Precision Vs Recovered')  

    print(f'mastedScores_{dataset}_k_{kFakeTemp}_nB_{nB}_dQ_{dQ}_Precision_{precision}')
    
    
    matedScIPAll = np.array([])
    matedScIPPreAll = np.array([])
    matedScIPTabAll = np.array([])
    matedScIPCrossAll = np.array([])
    matedScIPPreRecCrossAll = np.array([])
    matedScIPTabRecCrossAll = np.array([])


    for s in range(0, nSubj, nSave):
        print(f'Save range [{s}:{s+nSave}] from {now.strftime("%d%m%Y_%H%M%S")}')

        pool = mp.Pool(64)         

        gen_args = zip(repeat(dataDir), subjectIDs[s:s+nSave], repeat(precision), repeat(borders), repeat(tabMFIP), repeat(kFakeTemp))    
        matedSc = pool.starmap(matedNonQvsMFIPvsPrecisionVsRecovered, gen_args)    
        gc.collect()


        matedScIP, matedScIPPre, matedScIPTab, matedScIPCross, matedScIPPreRecCross, matedScIPTabRecCross = zip(*matedSc)

        matedScIP = flattenList(matedScIP)
        matedScIPPre = flattenList(matedScIPPre) 
        matedScIPTab = flattenList(matedScIPTab)  

        matedScIPCross = flattenList(matedScIPCross) 
        matedScIPPreRecCross = flattenList(matedScIPPreRecCross) 
        matedScIPTabRecCross = flattenList(matedScIPTabRecCross) 
 

        matedScIP = np.array(matedScIP, dtype = float)  
        matedScIPPre = np.array(matedScIPPre, dtype = int)         
        matedScIPTab = np.array(matedScIPTab, dtype = int)

        matedScIPCross = np.array(matedScIPCross, dtype = float)
        matedScIPPreRecCross = np.array(matedScIPPreRecCross, dtype = int)
        matedScIPTabRecCross = np.array(matedScIPTabRecCross, dtype = int)   


        matedScIPAll = np.concatenate([matedScIPAll, matedScIP]) 
        matedScIPPreAll = np.concatenate([matedScIPPreAll, matedScIPPre])
        matedScIPTabAll = np.concatenate([matedScIPTabAll, matedScIPTab]) 

        matedScIPCrossAll = np.concatenate([matedScIPCrossAll, matedScIPCross]) 
        matedScIPPreRecCrossAll = np.concatenate([matedScIPPreRecCrossAll, matedScIPPreRecCross])
        matedScIPTabRecCrossAll = np.concatenate([matedScIPTabRecCrossAll, matedScIPTabRecCross])  

        

        gc.collect()
        
        pool.close()
        pool.join()

        print('len(matedScIP) = ', len(matedScIP), ' len(matedScIPPre) = ', len(matedScIPPre), ' len(matedScIPTab) = ', len(matedScIPTab), ' len(matedScIPCross) = ', len(matedScIPCross), ' len(matedScIPPreRecCross) = ', len(matedScIPPreRecCross), ' len(matedScIPTabRecCross) = ', len(matedScIPTabRecCross))


        
        
        resultsFile = f'{resDir}/Range_{s}_{s+nSave}_matedScores_{dataset}_k_{kFakeTemp}_nB_{nB}_dQ_{dQ}_Precision_{precision}_{timeTag}.pkl'  
        pickle.dump((matedScIP, matedScIPPre, matedScIPTab, matedScIPCross, matedScIPPreRecCross, matedScIPTabRecCross), open(resultsFile, 'wb'))
    

    resultsFile = f'{resDir}/All_matedScores_{dataset}_k_{kFakeTemp}_nB_{nB}_dQ_{dQ}_Precision_{precision}_{timeTag}.pkl'  
    pickle.dump((matedScIPAll, matedScIPPreAll, matedScIPTabAll, matedScIPCrossAll, matedScIPPreRecCrossAll, matedScIPTabRecCrossAll), open(resultsFile, 'wb'))

    if dataset == 'VGGFace2':
        thrIPList = [0.2391948024240369, 0.30819380242219074, 0.4966788024171478]
        thrTabList = [224, 287, 468]
    elif dataset == 'LFW':
        thrIPList = [0.24106651582972638, 0.30361251582805293, 0.3716005158262339]
        thrTabList = [226, 284, 348]
    else:
        print('Add the thresholds @0.1%FMR, @0.01%FMR, and @0.001%FMR for the IP comparator tested on your dataset')
    thrPrecList = [round(thrIP*(precision**2)) for thrIP in thrIPList]



    print(f'Success rate for k = {kFakeTemp}')           
    print(f'    - IP without quantization: {[successRate(matedScIPAll, matedScIPCrossAll, thrIP) for thrIP in thrIPList]} ')
    print(f'    - IP Table-based quantization: {[successRate(matedScIPTabAll, matedScIPTabRecCrossAll, thrTab) for thrTab in thrTabList]} ')
    print(f'    - IP Precision-based quantization: {[successRate(matedScIPPreAll, matedScIPPreRecCrossAll, thrPrec) for thrPrec in thrPrecList]} ')



    # select the thresholds @0.1%FMR
    thrIP = thrIPList[0]
    thrTab = thrTabList[0]
    thrPrec = thrPrecList[0]

    orgPairsOrgRecPairsHist(matedScIPAll, matedScIPCrossAll, thrIP, 'Original\nIP(Org,Org)', 'Recovered\nIP(Rec,Org)', normalise=True, plotTitle=f'IP Without Quantization $(k= {kFakeTemp})$', savename=f'{plotDir}/matedScores_IP_k_{kFakeTemp}.pdf')

    orgPairsOrgRecPairsHist(scaleDownTab(matedScIPTabAll,dQ), scaleDownTab(matedScIPTabRecCrossAll, dQ), scaleDownTab(thrTab, dQ),'Original\nIP(Org,Org)', 'Recovered\nIP(Rec,Org)', normalise=True, plotTitle=f'IP Table-based Quantization $(k= {kFakeTemp})$', savename=f'{plotDir}/matedScores_nB_{nB}_dQ_{dQ}_IPTab_k_{kFakeTemp}.pdf')

    orgPairsOrgRecPairsHist(scaleDownPrec(matedScIPPreAll,precision), scaleDownPrec(matedScIPPreRecCrossAll,precision), scaleDownPrec(thrPrec, precision), 'Original\nIP(Org,Org)', 'Recovered\nIP(Rec,Org)', normalise=True, plotTitle=f'IP Precision-based Quantization $(k= {kFakeTemp})$', savename=f'{plotDir}/matedScores_Precision_{precision}_IPPre_k_{kFakeTemp}.pdf')
    



if __name__ == '__main__':
    main()

