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

# default
# range of under & over in parameter
nkmin,nkmax = 0.014449,0.015499
tnkmin,tnkmax = 0.012,0.014949
tkmin,tkmax = 0.012,0.0135

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

# making logs -----------------------------------------------------------------
def makeLog(b1,b2,b3):
    #pdb.set_trace()
    # save param b ------------------------------------------------------------
    params = np.concatenate((b1[np.newaxis],b2[np.newaxis],b3[np.newaxis]),0)[:,np.newaxis]
    np.savetxt(paramCSV,params.T*1000000,delimiter=",",fmt="%.0f")
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
    shutil.move(oldpath,newpath)

    return maxSim
# -----------------------------------------------------------------------------


for epoch in np.arange(nEpoch):
    
    # Set 
    if epoch == 0:
        pdB = defaultpdB
    else:
        pdBs = np.loadtxt(os.path.join(savedirPath,f"BO_paramb_{epoch-1}_{itrNum}_{trID}.txt"))
        
        best1B = pdBs[-1]
        best2B = pdBs[-2]
        
        minpdBs,maxpdBs = np.min(best1B,0),np.max(best2B,0)
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
    
    # Save params -------------------------------------------------------------
    flag = False
    for line in sort_res:
        
        # directory -> numpy [1,] [3,]
        target = np.array([line['target']])
        param = np.concatenate((np.array([line['params']['b1']]),np.array([line['params']['b2']]),np.array([line['params']['b3']])),0)
        
        if not flag:
            targets = target
            params = param
            flag = True
        else:
            targets = np.vstack([targets,target])
            params = np.vstack([params,param])
    
    # optimized rate
    np.savetxt(os.path.join(savedirPath,f"BO_target_{epoch}_{itrNum}_{trID}.txt"),targets)
    # parameter b
    np.savetxt(os.path.join(savedirPath,f"BO_paramb_{epoch}_{itrNum}_{trID}.txt"),params,fmt=f"%8f")
    # -------------------------------------------------------------------------
    
    
"""
# Make rireki -----------------------------------------------------------------
logsfullPath = os.path.join(logsPath,savedlogsPath,filePath)
logsfile = glob.glob(logsfullPath)

flag = False
index = []
for iS in np.arange(len(logsfile)):
    
    file = os.path.basename(logsfile[iS])
    print(file)
    # Num. of file for sort index
    fID = int(file.split("_")[-1].split(".")[0])
   
    U, th, V, B = myDataPF.loadABLV(logsPath,savedlogsPath,file)
    B = np.concatenate([B[2,np.newaxis],B[4,np.newaxis],B[5,np.newaxis]],0)
    yU, yth, yV, pJ_all = myDataPF.convV2YearlyData(U,th,V,nYear=10000,cnt=0)
    yU, yth, yV, pJ_all, maxSim, sYear = myDataPF.MSErrorNankai(gt,yU,yth,yV,pJ_all,nCell=nCell)
    
    predV = [pJ_all[ntI]-int(U[0,1]),pJ_all[tntI]-int(U[0,1]),pJ_all[ttI]-int(U[0,1])]
    gtV = [np.where(gt[:,ntI]>0)[0],np.where(gt[:,tntI]>0)[0],np.where(gt[:,ttI]>0)[0]]
    
    # plot & mae eq. of predict & gt
    maxSim = myPlot.Rireki(gtV,predV,path="bayes",label=f"{iS}_{np.round(B[ntI],6)}_{np.round(B[tntI],6)}_{np.round(B[ttI],6)}",isResearch=True)
# -----------------------------------------------------------------------------
"""
