import numpy as np
import glob
from scipy.optimize import fsolve
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import glob



def normalizeSample(sample):  
    return sample/np.linalg.norm(sample)

def diffFromOne0(lbda,p,s):
    n = p.shape[0]
    nEye = np.identity(n)
    pp = np.matmul(p, p.T)
    ps = np.matmul(p, s)
    r = np.linalg.lstsq(pp + lbda * nEye, ps,rcond=None)[0]
    d = np.matmul(r.T,r)-1
    return d


# ======== MFIP Quantization ============

def getIndexQF(x, t):
    return len(np.where(t<x)[0])

def computeScore(x, y, t, table):
    nFeat = len(x)
    score = 0
    for i in range(nFeat):
        ix = getIndexQF(x[i], t)
        iy = getIndexQF(y[i], t)
        score  += table[ix,iy]
    return score

def flattenList(lists):
    flatList = []
    for l in lists:
        flatList += l    
    return flatList



# ======== Precision Quantization vs. MFIP Quantization vs. Non-quantized ============





def getRecoveredTemp0(r, k):
    n = r.size
    
    # Create probes
    p = np.random.normal(0, 1, (n,k))
    pNorm = np.linalg.norm(p, axis=0)
    p = p/pNorm
    
    # Inner product vector
    s = np.matmul(p.T, r)
    s = np.reshape(s, (s.size, ))
    
    def diffFromOne(x): 
        return diffFromOne0(x,p,s)
    
    lbda0 = fsolve(diffFromOne, 0.1)[0]
    pp = np.matmul(p, p.T)
    ps = np.matmul(p, s)
    nEye = np.identity(n)
    
    r0 = np.linalg.lstsq(pp + lbda0 * nEye, ps, rcond=None)[0]
    
    return normalizeSample(r0)


def getRecoveredTempPrecision(r, k, precision):
    n = r.size
    
    # Create fake templates
    p = np.random.normal(0, 1, (n,k))
    pNorm = np.linalg.norm(p, axis=0)
    p = p/pNorm

    pPre = np.round(precision*p).astype(int)   
    rPre = np.round(precision*r).astype(int)   

    # Inner product vector quantized with precision 
    s = np.matmul(pPre.T, rPre)
    s = np.reshape(s, (s.size, ))
    
    def diffFromOne(x): 
        return diffFromOne0(x,p,s)
    
    lbda0 = fsolve(diffFromOne, 0.1)[0]
    pp = np.matmul(p, p.T)
    ps = np.matmul(p, s)
    nEye = np.identity(n)
    
    r0 = np.linalg.lstsq(pp + lbda0 * nEye, ps, rcond=None)[0]

    return normalizeSample(r0)

def getRecoveredTempMFIP(r, k, borders, tabMFIP):
    n = r.size
    
    # Create fake templates
    p = np.random.normal(0, 1, (n,k))
    pNorm = np.linalg.norm(p, axis=0)
    p = p/pNorm
   

    # Inner product vector quantized with MFIP 
    s = [computeScore(p[:,i], r, borders, tabMFIP) for i in range(k)]
    s = np.reshape(s, (k, ))
    
    def diffFromOne(x): 
        return diffFromOne0(x,p,s)
    
    lbda0 = fsolve(diffFromOne, 0.1)[0]
    pp = np.matmul(p, p.T)
    ps = np.matmul(p, s)
    nEye = np.identity(n)
    
    r0 = np.linalg.lstsq(pp + lbda0 * nEye, ps, rcond=None)[0]

    return normalizeSample(r0)





def matedNonQvsMFIPvsPrecisionVsRecovered(dataDir, subject, precision, borders, tabMFIP, kFakeTemp, nFix = 5):
    matedScIP = []
    matedScIPPre = []    
    matedScIPTab = []

    matedScIPCross = []
    matedScIPPreRecCross = []    
    matedScIPTabRecCross = [] 
   
    if 'LFW' in dataDir:
        subjectImgs = glob.glob(f'{dataDir}/{subject}_*')
    elif 'VGGFace2' in dataDir:
        subjectImgs = glob.glob(f'{dataDir}/{subject}/*.npy')
    else:
        print(f'Adjust how your dataset should be read')
    subjectImgs.sort() 


    n = len(subjectImgs)
    subjectImgs = subjectImgs[:n] if (n < nFix) else subjectImgs[:nFix]  
    
    indexes = [(i,j) for i in range(len(subjectImgs)) for j in range(len(subjectImgs)) if i<j]
    for (i,j) in indexes:        
        x = np.load(subjectImgs[i])
        x = normalizeSample(x)
        xPre = np.round(precision*x).astype(int)  

        try:
            xRec = getRecoveredTemp0(x, kFakeTemp)  
            xPreRec = getRecoveredTempPrecision(x, kFakeTemp, precision) 
            xPreRecQ = np.round(precision*xPreRec).astype(int)
            xTabRec = getRecoveredTempMFIP(x, kFakeTemp, borders, tabMFIP)
        except np.linalg.LinAlgError:
            print(f'k = {kFakeTemp} skipped {subject}')
            continue

        y = np.load(subjectImgs[j])
        y = normalizeSample(y)
        yPre = np.round(precision*y).astype(int) 


        matedScIP.append(np.sum(x*y))
        matedScIPPre.append(np.sum(xPre*yPre))
        matedScIPTab.append(computeScore(x, y, borders, tabMFIP))


        matedScIPCross.append(np.sum(xRec*y))
        matedScIPPreRecCross.append(np.sum(xPreRecQ*yPre))    
        matedScIPTabRecCross.append(computeScore(xTabRec, y, borders, tabMFIP))
        

    return matedScIP, matedScIPPre, matedScIPTab, matedScIPCross, matedScIPPreRecCross, matedScIPTabRecCross


def scaleDownTab(scores, nQ):
    return scores*nQ

def scaleDownPrec(scores, precision):
    return scores/precision**2


def successRate(orgScores, recScores, threshold):
    orgScores = np.array(orgScores)
    recScores = np.array(recScores)
    nRecSuc = len(np.where(threshold <= recScores)[0])
    nOrgSuc = len(np.where(threshold <= orgScores)[0])
    rate = nRecSuc/nOrgSuc
    return round(rate, 4)

def upperBoundAttackIP(dataDir, subject, thrIP, kStart, kEnd, nK):  
    if 'LFW' in dataDir:
        subjectImgs = glob.glob(f'{dataDir}/{subject}_*')
    else:
        subjectImgs = glob.glob(f'{dataDir}/{subject}/*.npy')
    subjectImgs.sort() 
    x = np.load(subjectImgs[0])
    x = normalizeSample(x)

    y = np.load(subjectImgs[1])
    y = normalizeSample(y)

    kList = list(range(kStart, kEnd, nK))
    kParsed = []
    scores = []
    for k in kList:
        try:
            xRec = getRecoveredTemp0(x, k)             
        except np.linalg.LinAlgError:
            print(f'k = {k} skipped {subject}')
            continue
        score = np.sum(xRec*y)
        scores.append(score)
        kParsed.append(k)
        # print(f'{thrIP} <=? {score} for k = {k}')
        if (thrIP <= score):
            print(f'{subject}: {thrIP:.3e} <= {score:.3e} for k = {k}')
            break   
        
    return scores, kParsed


def upperBoundAttackTab(dataDir, subject, borders, tabMFIP, thrTab, kStart, kEnd, nK):  
    if 'LFW' in dataDir:
        subjectImgs = glob.glob(f'{dataDir}/{subject}_*')
    else:
        subjectImgs = glob.glob(f'{dataDir}/{subject}/*.npy')
    subjectImgs.sort() 
    x = np.load(subjectImgs[0])
    x = normalizeSample(x)

    y = np.load(subjectImgs[1])
    y = normalizeSample(y)

    kList = list(range(kStart, kEnd, nK))
    kParsed = []
    scores = []
    for k in kList:
        try:
            xTabRec = getRecoveredTempMFIP(x, k, borders, tabMFIP)
        except np.linalg.LinAlgError:
            print(f'k = {k} skipped {subject}')
            continue
        score = computeScore(xTabRec, y, borders, tabMFIP)
        scores.append(score)
        kParsed.append(k)
        # print(f'{thrTab} <=? {score} for k = {k}')
        if thrTab <= score:
            print(f'{subject}: {thrTab} <= {score} for k = {k}')
            break 
    return scores, kParsed


def upperBoundAttackPrec(dataDir, subject, precision, thrPrec, kStart, kEnd, nK):  
    if 'LFW' in dataDir:
        subjectImgs = glob.glob(f'{dataDir}/{subject}_*')
    else:
        subjectImgs = glob.glob(f'{dataDir}/{subject}/*.npy')
    subjectImgs.sort() 
    x = np.load(subjectImgs[0])
    x = normalizeSample(x)

    y = np.load(subjectImgs[1])
    y = normalizeSample(y)
    yPre = np.round(precision*y).astype(int) 

    kList = list(range(kStart, kEnd, nK))
    kParsed = []
    scores = []
    for k in kList:
        try:
            xPreRec = getRecoveredTempPrecision(x, k, precision) 
        except np.linalg.LinAlgError:
            print(f'k = {k} skipped {subject}')
            continue
        xPreRecQ = np.round(precision*xPreRec).astype(int)
        score = np.sum(xPreRecQ*yPre)
        scores.append(score)
        kParsed.append(k)
        # print(f'{thrPrec} <=? {score} for k = {k}')
        if thrPrec <= score:
            print(f'{subject}: {thrPrec} <= {score} for k = {k}')
            break
    return scores, kParsed





# ======== Plots ============



def delEmpty(subjectIDs, subScores, subParsedK):
    for i in subjectIDs:
        if subParsedK[i] == None:
            subScores.pop(i)
            subParsedK.pop(i)

def getMedianPerSubj(subParsedK):
    medianKPerSubj = []
    for subj in subParsedK.keys():
        medianKPerSubj.append(np.median(subParsedK[subj]))
    return medianKPerSubj

def plotBoxPlots(data, dataLables, plotTitle=None, savename=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = ['#0072B2','#E69F00', '#009E73', '#CC79A7',"#1ECBE1","#77AC30","#4DBEEE"]
    

    sns.set_style('whitegrid')
    sns.boxplot(data=data, palette=palette, showmeans=True, meanprops={"marker":"o", "markerfacecolor":"None", "markeredgecolor":"red", "markersize":12, "markeredgewidth":2}, medianprops=dict(color='red', linewidth=2), ax=ax, boxprops=dict(linewidth=2, edgecolor='black'), whiskerprops=dict(linestyle='--', linewidth=1.5, color='black'), capprops=dict(linewidth=2, color='black'), showfliers=False, notch=True)
    
    for i in range(len(data)):
        median_val = round(np.median(data[i]))
        mean_val = np.mean(data[i])
        ax.text(i, median_val, f'{median_val}', ha='center', va='bottom', fontsize=14, color = 'black', fontweight='bold')
        ax.text(i+0.05, round(mean_val), f'{mean_val:.2f}', ha='left', va='bottom', fontsize=14, color = 'red', fontweight='bold')

    
    ax.set_xticklabels(dataLables, fontsize=14)
    ax.set_ylabel('#FakeTemp k', fontsize=14)
    if plotTitle is not None:
        ax.set_title(plotTitle, fontsize=14, fontweight='bold')

    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='Mean', markerfacecolor='none', markersize=10, markeredgecolor='red'),
                    plt.Line2D([0], [0], color='red', label='Median', linewidth=2)]
    
    ax.legend(handles=legend_elements, ncol=1, loc='lower center', bbox_to_anchor=(0.05, -0.25), fontsize=14)
    
    sns.despine(ax=ax, left=True)
    for i, box in enumerate(ax.artists):        
        box.set_edgecolor('black')
        box.set_alpha(0.5)
        if i % 2 == 0:
            box.set_facecolor('#F0F0F0')
        for j in range(i * 6, i * 6 + 6):
            ax.lines[j].set_color('black')
            ax.lines[j].set_linewidth(1.5)
            if j in range(i * 6, i * 6 + 4):
                ax.lines[j].set_linestyle('-')
    
    if savename is not None:
        plt.savefig(savename, bbox_inches="tight")
        plt.cla()
        plt.clf()
        plt.close()
    else:
        plt.show()




def plot_Scatters(kFakeTempList, maxScOrgIP, minScOrgIP, avgScOrgIP, maxScRecIP, minScRecIP, avgScRecIP, threshold, plotTitle=None, savename=None):
    sns.set(style="white", palette="muted", color_codes=True)
    plt.rc("axes", axisbelow=True) 
    plt.figure(figsize=(12,6))
    plt.rcParams['font.size'] = 16
    plt.rcParams['font.weight'] = 'bold'

    plt.scatter(kFakeTempList, maxScOrgIP, label='Max Org', marker='o', s=200, linewidths=1, edgecolors='#1212A6',color='#4BC1EC')
    plt.scatter(kFakeTempList, maxScRecIP, label='Max Rec', marker='*', s=100, linewidths=2, color='#1212A6') 
    
    plt.scatter(kFakeTempList, avgScOrgIP, label='Avg Org', marker='o', s=200, linewidths=1, edgecolors='green',color='#42D02B')
    plt.scatter(kFakeTempList, avgScRecIP, label='Avg Rec', marker='*', s=100, linewidths=2, color='#226A16')

    plt.scatter(kFakeTempList, minScOrgIP, label='Min Org', marker='o', s=200, linewidths=1, edgecolors='red', color='#F59F9F')   
    plt.scatter(kFakeTempList, minScRecIP, label='Min Rec', marker='*', s=100, linewidths=2, color='#E61919')

    plt.axhline(y = threshold, color='black', linewidth=2)
    plt.text(2000, threshold + 0.04, 'Threshold at 0.1% FMR', rotation=0, horizontalalignment='center', verticalalignment='center', fontsize=16, fontweight='bold')
    plt.xlabel("#FakeTemp $(k)$", fontdict={'fontsize': 16, 'fontweight': 'bold'})
    plt.ylabel("Comparison Score", fontdict={'fontsize': 16, 'fontweight': 'bold'})
    plt.grid(True)
    plt.ylim(-0.2, 1.2)
    plt.legend(ncol=6, loc='lower center', bbox_to_anchor=(0.5, -0.21), fontsize=12)
    plt.text(0.1, -0.35, 'Org(org,org) and Rec(rec,org)', fontsize=12, fontweight='bold')

    if plotTitle is not None:
        plt.title(plotTitle, fontsize=16, fontweight='bold')
        # plt.xticks(fontsize=16, fontweight='bold')
        # plt.yticks(fontsize=16, fontweight='bold')
    
    if savename is not None:
        plt.savefig(savename, bbox_inches="tight")
        plt.cla()
        plt.clf()
        plt.close()
    else:
        plt.show()

def orgPairsOrgRecPairsHist(orgOrgScores, orgRecScores, threshold, orgOrgLabel, orgRecLabel, normalise=True, plotTitle=None, savename=None):
    def normalise_scores(distribution):
      return np.ones_like(distribution) / len(distribution)
    plt.figure(figsize=(12,6))
    plt.rcParams['font.size'] = 16
    plt.rcParams['font.weight'] = 'bold'
    _, bins = np.histogram(orgOrgScores, bins=50)
    if normalise:
        plt.hist(orgOrgScores, bins=bins, weights=normalise_scores(orgOrgScores), color='green', alpha=0.5, label=orgOrgLabel)
        plt.hist(orgRecScores, bins=bins, weights=normalise_scores(orgRecScores), color='red', alpha=0.5, label=orgRecLabel)
        xlabel = "Probability Density"
    else:
        plt.hist(orgOrgScores, bins=bins, color='green', alpha=0.5, label=orgOrgLabel)
        plt.hist(orgRecScores, bins=bins, color='red', alpha=0.5, label=orgRecLabel)
        xlabel = "Count"
    plt.axvline(x = threshold, color='black', linewidth=2)
    plt.text(threshold-0.011, 0.05, 'Threshold at 0.1% FMR', rotation=90, horizontalalignment='center', verticalalignment='center', fontsize=14)
    plt.xlabel("Comparison Score", size=16, fontdict={'fontsize': 16, 'fontweight': 'bold'})
    plt.ylabel(xlabel, size=16, fontdict={'fontsize': 16, 'fontweight': 'bold'})
    plt.grid(True)
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.legend(loc="upper left")

    if plotTitle is not None:
        plt.title(plotTitle, fontsize=20, fontweight='bold')
    
    if savename is not None:
        plt.savefig(savename, bbox_inches="tight")
        plt.cla()
        plt.clf()
        plt.close()
    else:
        plt.show()
