# -*- coding: utf-8 -*-

import os
import pickle
import time
import glob
import pdb

import numpy as np
import matplotlib.pylab as plt
from bayes_opt import BayesianOptimization

import DC as myData

import warnings
warnings.filterwarnings('ignore')

# path ------------------------------------------------------------------------
featuresPath = "features"
paramCSV = "bayesParam.csv"
batFile = "featureV.bat"
dirPath = "bayesB"

# paramters -------------------------------------------------------------------
# for nankai rireki
tfID = 190
cnt = 0
slip = 1
aYear = 1400

# Num. of iteration for bayes
itrNum = 10
# under & over limit
nkmin,nkmax = 0.0110,0.0170
tnkmin,tnkmax = 0.0110,0.0170
tkmin,tkmax = 0.0110,0.0170

# パラメータの下限・上限
# 連続値の場合は、事前分布指定可（default:連続一様分布、対数一様分布も指定可）
pbounds = {"b1":(nkmin,nkmax),"b2":(tnkmin,tnkmax),"b3":(tkmin,tkmax)}

# -----------------------------------------------------------------------------

def MinErrorNankai(gt,pred):
    
    gYear_nk = np.where(gt[:,0] > slip)[0]
    gYear_tnk = np.where(gt[:,1] > slip)[0]
    gYear_tk = np.where(gt[:,2] > slip)[0]
    
    flag = False
    for sYear in np.arange(8000-aYear): 
        eYear = sYear + aYear

        pYear_nk = np.where(pred[sYear:eYear,0] > slip)[0]
        pYear_tnk = np.where(pred[sYear:eYear,1] > slip)[0]
        pYear_tk = np.where(pred[sYear:eYear,2] > slip)[0]
        
        ndist_nk = gauss(gYear_nk,pYear_nk.T)
        ndist_tnk = gauss(gYear_tnk,pYear_tnk.T)
        ndist_tk = gauss(gYear_tk,pYear_tk.T)

        yearError_nk = sum(ndist_nk.max(1)/pYear_nk.shape[0])
        yearError_tnk = sum(ndist_tnk.max(1)/pYear_tnk.shape[0])
        yearError_tk = sum(ndist_tk.max(1)/pYear_tk.shape[0])
        
        yearError = yearError_nk + yearError_tnk + yearError_tk
        
        if not flag:
            yearErrors = yearError
            flag = True
        else:
            yearErrors = np.hstack([yearErrors,yearError])

    sInd = np.argmax(yearErrors)
    eInd = sInd + aYear

    maxSim = yearErrors[sInd]
          
    return maxSim
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def gauss(gtY,predY,sigma=100):
    
    predYs = predY.repeat(gtY.shape[0],0).reshape(-1,gtY.shape[0])
    gtYs = gtY.repeat(predY.shape[0],0).reshape(-1,predY.shape[0])

    gauss = np.exp(-(gtYs - predYs.T)**2/(2*sigma**2))

    return gauss
# -----------------------------------------------------------------------------

# reading files ---------------------------------------------------------------
def readFiles():
    
    # reading nankai rireki ---------------------------------------------------
    with open(os.path.join(featuresPath,"nankairireki.pkl"), "rb") as fp:
            nkfiles = pickle.load(fp)
    gtV = nkfiles[tfID,:,:]
    # -------------------------------------------------------------------------
    
    # reading predict logs ----------------------------------------------------
    fileName = f"{cnt}_*"
    filePath = os.path.join(dirPath,fileName)
    files = glob.glob(filePath)
    # -------------------------------------------------------------------------
    cnt += 1
    return gtV, files
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
    gtV, files = readFiles()
    
    for file in files:
        U,B = myData.loadABLV(os.path.join(dirPath,file))
        deltaU = myData.convV2YearlyData(U)
        maxSim = MinErrorNankai(gtV,deltaU)
    
    return maxSim
# -----------------------------------------------------------------------------

# Start Bayes -----------------------------------------------------------------
opt = BayesianOptimization(f=func,pbounds=pbounds)
opt.maximize(init_points=3,n_iter=itrNum)

# result
res = opt.res
best_res = opt.max
# -----------------------------------------------------------------------------
