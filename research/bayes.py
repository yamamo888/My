# -*- coding: utf-8 -*-

import os
import sys
import pickle
import time
import glob
import shutil
import pdb

import numpy as np
import matplotlib.pylab as plt
from bayes_opt import BayesianOptimization

import DC as myData
import makingDataPF as myDataPF
import PlotPF as myPlot

import warnings
warnings.filterwarnings('ignore')

# mode ------------------------------------------------------------------------
#　Num. of iter
itrNum = int(sys.argv[1])
# Num. of traial
trID = int(sys.argv[2])
# -----------------------------------------------------------------------------

# path ------------------------------------------------------------------------
dirPath = "bayes"
#featuresPath = "features"
featuresPath = "nankairirekifeature"
logsPath = "logs"
# for paramter & targer
savedirPath = f"BO_{trID}"
# for logs
savedlogPath = f'savedPD_{trID}'
paramCSV = "bayesParam.csv"
batFile = "PyToCBayes.bat"
filePath = "*txt"
# -----------------------------------------------------------------------------

# paramters -------------------------------------------------------------------

slip = 1
aYear = 1400
ntI,tntI,ttI = 0,1,2
nCell = 3
# num.of epoch
nEpoch = 10
# range b
mt = 1000000
# default
# range of under & over in parameter
nkmin,nkmax = 0.014449,0.015499
tnkmin,tnkmax = 0.012,0.014949
#tkmin,tkmax = 0.012,0.0135
tkmin,tkmax = 0.013450,0.013450

defaultpdB = [[nkmin,nkmax],[tnkmin,tnkmax],[tkmin,tkmax]]
# -----------------------------------------------------------------------------

# Prior distribution ----------------------------------------------------------
def setPriorDistribution(pdB):
    
    nkmin,nkmax = pdB[ntI][0],pdB[ntI][1]
    tnkmin,tnkmax = pdB[tntI][0],pdB[tntI][1]
    tkmin,tkmax = pdB[ttI][0],pdB[ttI][1]
    
    # 連続値の場合は、事前分布指定可（default:連続一様分布、対数一様分布も指定可）
    pbounds = {"b1":(nkmin,nkmax),"b2":(tnkmin,tnkmax),"b3":(tkmin,tkmax)}
    
    return pbounds
# -----------------------------------------------------------------------------

# reading files ---------------------------------------------------------------
def readFiles():
    
    # reading predict logs ----------------------------------------------------
    #fileName = f"{cnt}_*"
    fileName = f"*txt"
    filePath = os.path.join(logsPath,dirPath,fileName)
    files = glob.glob(filePath)
    # -------------------------------------------------------------------------
    
    return files
# -----------------------------------------------------------------------------

# moving files ----------------------------------------------------------------
def moveFiles(cpath,newpath):
    '''
    cpath: current directory path
    newpath: distination directory path
    '''
    if os.path.exists(os.path.join(newpath,os.path.basename(cpath))):
        os.remove(cpath)
    else:
        shutil.move(cpath,newpath)
# -----------------------------------------------------------------------------

# making logs -----------------------------------------------------------------
def makeLog(b1,b2,b3):
    #pdb.set_trace()
    # save param b ------------------------------------------------------------
    params = np.concatenate((b1[np.newaxis],b2[np.newaxis],b3[np.newaxis]),0)[:,np.newaxis]
    np.savetxt(paramCSV,params.T*mt,delimiter=",",fmt="%.0f")
    # -------------------------------------------------------------------------
    
    # call bat ----------------------------------------------------------------
    lockPath = "Lock.txt"
    lock = str(1)
    with open(lockPath,"w") as fp:
        fp.write(lock)
    
    os.system(batFile)
    
    sleepTime = 3
    # lockファイル作成時は停止
    while True:
        time.sleep(sleepTime)
        if os.path.exists(lockPath)==False:
            break
    # -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# function --------------------------------------------------------------------
def func(b1,b2,b3):
    
    # simulation
    makeLog(b1,b2,b3)
    
    # reading gt & logs
    logfile = readFiles()[0]
    
    print(logfile)
    # U:[None,10], B:[3,]
    U,B = myData.loadABLV(logfile)
    deltaU = myData.convV2YearlyData(U)
    # each mse var.
    maxSim = myData.MinErrorNankai(deltaU,mode=3)
    maxSim = 1/maxSim
    
    # Move readed logfile
    oldpath = os.path.join(logfile)
    newpath = os.path.join(logsPath,savedlogPath)
    moveFiles(oldpath,newpath)
  
    return maxSim
# -----------------------------------------------------------------------------
"""
# -----------------------------------------------------------------------------
for epoch in np.arange(nEpoch):
    
    # Set 
    if epoch == 0:
        pdB = defaultpdB
    else:
        pdBs = np.loadtxt(os.path.join(savedirPath,f"BO_paramb_{epoch-1}_{itrNum}_{trID}.txt"))
        
        minpdBs = pdBs[-1]/mt
        maxpdBs = pdBs[-2]/mt
        
        print('range min:',minpdBs)
        print('range max:',maxpdBs)
        
        pdB = [[minpdBs[ntI],maxpdBs[ntI]],[minpdBs[tntI],maxpdBs[tntI]],[minpdBs[ttI],maxpdBs[ttI]]]
    
    
    # prior distribution parameter b    
    pbounds = setPriorDistribution(pdB)

    # Start Bayes -------------------------------------------------------------
    # verbose: 学習過程表示 0:無し, 1:すべて, 2:最大値更新時
    opt = BayesianOptimization(f=func,pbounds=pbounds,verbose=2)
    # init_points:最初に取得するf(x)の数、ランダムに選択される
    # n_iter:試行回数(default:25パターンのパラメータで学習)
    opt.maximize(init_points=5,n_iter=itrNum)
    # -------------------------------------------------------------------------
    #pdb.set_trace()
    # Result ------------------------------------------------------------------
    res = opt.res # all
    best_res = opt.max # max optimize
    # sort based on 'target'(maxSim)
    sort_res = sorted(res, key=lambda x: x['target'])
    # -------------------------------------------------------------------------
    #pdb.set_trace()
    # Save params -------------------------------------------------------------
    flag = False
    for line in sort_res:
        
        # directory -> numpy [1,] [3,]
        target = np.array([line['target']])
        param = np.concatenate((np.array([line['params']['b1']]),np.array([line['params']['b2']]),np.array([line['params']['b3']])),0)
        
        if not flag:
            targets = target
            params = param * mt
            flag = True
        else:
            targets = np.vstack([targets,target])
            params = np.vstack([params,param * mt])
    # optimized rate
    np.savetxt(os.path.join(savedirPath,f"BO_target_{epoch}_{itrNum}_{trID}.txt"),targets)
    # parameter b
    np.savetxt(os.path.join(savedirPath,f"BO_paramb_{epoch}_{itrNum}_{trID}.txt"),params,fmt=f"%d")
    # -------------------------------------------------------------------------
# -----------------------------------------------------------------------------
"""
"""
# Make best csv ---------------------------------------------------------------
# Get best paramter b
targetfullPath = os.path.join(savedirPath,'BO_target_*')
targetfiles = glob.glob(targetfullPath)

bfullPath = os.path.join(savedirPath,'BO_paramb_*')
bfiles = glob.glob(bfullPath)

flag = False
for targetfile,bfile in zip(targetfiles,bfiles):
    target = np.loadtxt(targetfile)
    paramb = np.loadtxt(bfile)
    
    if not flag:
        targets = target
        bs = paramb
        flag = True
    else:
        #pdb.set_trace()
        targets = np.hstack([targets,target])
        bs = np.vstack([bs,paramb])

# del multiple parameter b
parambs = [bs[0]]
index = [0]
for ind,line in enumerate(bs):
    if not all(line == parambs[-1]):
        parambs.append(line)
        index.append(ind)

# del multiple targets
maxsims = targets[index]

# list -> numpy
parambs = np.array(parambs)

# min mse index
best100ind = np.argsort(maxsims)[::-1][:100]
best100target = targets[best100ind]
best100b = bs[best100ind.tolist()]

np.savetxt(os.path.join(savedirPath,f'best100_target.txt'),best100target)
# for bat
np.savetxt(os.path.join(savedirPath,f'best100_b.csv'),best100b,delimiter=',',fmt='%d')
# -----------------------------------------------------------------------------

# after featureV.bat

"""
# Make rireki -----------------------------------------------------------------
logsfullPath = os.path.join(logsPath,f'sortbayes_{trID}',filePath)
logsfile = glob.glob(logsfullPath)

with open(os.path.join(featuresPath,"nankairireki.pkl"), "rb") as fp:
    nkfiles = pickle.load(fp)
gt = nkfiles[190,:,:]
  
flag = False
index = []
for iS in np.arange(len(logsfile)):
    
    file = os.path.basename(logsfile[iS])
    print(file)
    # Num. of file for sort index
    fID = int(file.split("_")[-1].split(".")[0])
   
    U, th, V, B = myDataPF.loadABLV(logsPath,f'sortbayes_{trID}',file)
    B = np.concatenate([B[2,np.newaxis],B[4,np.newaxis],B[5,np.newaxis]],0)
    deltaU, yth, yV, pJ_all = myDataPF.convV2YearlyData(U,th,V,nYear=10000,cnt=0,isLast=True)
    deltaU = np.concatenate((deltaU[:,2,np.newaxis],deltaU[:,4,np.newaxis],deltaU[:,5,np.newaxis]),1)
    maxSim, pred = myData.MinErrorNankai(deltaU,mode=3,isPlot=True)
    
    pJ_all = [np.where(pred[:,ntI]>1)[0],np.where(pred[:,tntI]>1)[0],np.where(pred[:,ttI]>1)[0]]
    predV = [pJ_all[ntI]-int(U[0,1]),pJ_all[tntI]-int(U[0,1]),pJ_all[ttI]-int(U[0,1])]
    gtV = [np.where(gt[:,ntI]>0)[0],np.where(gt[:,tntI]>0)[0],np.where(gt[:,ttI]>0)[0]]
    
    # plot & mae eq. of predict & gt
    myPlot.Rireki(gtV,predV,path=f"bayes_{trID}",label=f"{iS}_{np.round(B[ntI],6)}_{np.round(B[tntI],6)}_{np.round(B[ttI],6)}",title=f'{int(maxSim)}')

    if not flag:
        maxSims = maxSim
        flag = True
    else:
        maxSims = np.hstack([maxSims,maxSim])
    
sort_maxSims = np.sort(maxSims)
index = np.argsort(maxSims)

np.savetxt(os.path.join(f"bayes_{trID}",'maxsim.txt'),sort_maxSims,fmt='%d')
np.savetxt(os.path.join(f"bayes_{trID}",'index.txt'),index,fmt='%d')
# -----------------------------------------------------------------------------

