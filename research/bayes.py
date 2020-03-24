# -*- coding: utf-8 -*-

import os
import sys
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

# mode ------------------------------------------------------------------------
# 0: gauss誤差 1:二乗誤差
mode = int(sys.argv[1])
#　Num. of iter
itrNum = int(sys.argv[2])
# -----------------------------------------------------------------------------

# path ------------------------------------------------------------------------
logsPath = "logs"
#featuresPath = "features"
featuresPath = "nankairirekifeature"
paramCSV = "bayesParam.csv"
batFile = "PyToCBayes.bat"
dirPath = "deltaU_bayes"
# -----------------------------------------------------------------------------

# paramters -------------------------------------------------------------------
# for nankai rireki
tfID = 190
slip = 1
aYear = 1400

# range of under & over in parameter
nkmin,nkmax = 0.0110,0.0170
tnkmin,tnkmax = 0.0110,0.0170
tkmin,tkmax = 0.0110,0.0170

# 連続値の場合は、事前分布指定可（default:連続一様分布、対数一様分布も指定可）
pbounds = {"b1":(nkmin,nkmax),"b2":(tnkmin,tnkmax),"b3":(tkmin,tkmax)}
# -----------------------------------------------------------------------------

 # reading nankai rireki ------------------------------------------------------
with open(os.path.join(featuresPath,"nankairireki.pkl"), "rb") as fp:
        nkfiles = pickle.load(fp)
gt = nkfiles[tfID,:,:]
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def GaussErrorNankai(pred):
    
    gYear_nk = np.where(gt[:,0] > slip)[0]
    gYear_tnk = np.where(gt[:,1] > slip)[0]
    gYear_tk = np.where(gt[:,2] > slip)[0]
    
    flag = False
    for sYear in np.arange(8000-aYear): 
        eYear = sYear + aYear

        pYear_nk = np.where(pred[sYear:eYear,0] > slip)[0]
        pYear_tnk = np.where(pred[sYear:eYear,1] > slip)[0]
        pYear_tk = np.where(pred[sYear:eYear,2] > slip)[0]
        # gauss error
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
    logfiles = readFiles()
    #pdb.set_trace()
    for file in logfiles:
        # U:[None,10], B:[3,]
        U,B = myData.loadABLV(file)
        deltaU = myData.convV2YearlyData(U)
        
        if mode == 0:
            # degree of similatery
            maxSim = GaussErrorNankai(deltaU)
        #elif mode == 1:
            #maxSim = MSErrorNanaki(deltaU)
        
    return maxSim
# -----------------------------------------------------------------------------

# Start Bayes -----------------------------------------------------------------
# verbose: 学習過程表示 0:無し, 1:すべて, 2:最大値更新時
opt = BayesianOptimization(f=func,pbounds=pbounds,verbose=2)
# init_points:最初に取得するf(x)の数、ランダムに選択される
# n_iter:試行回数(default:25パターンのパラメータで学習)
opt.maximize(init_points=3,n_iter=itrNum)
# -----------------------------------------------------------------------------
#pdb.set_trace()
# Result ----------------------------------------------------------------------
res = opt.res # all
best_res = opt.max # max optimize
# sort based on 'target'(maxSim)
sort_res = sorted(res, key=lambda x: x['target'])
# -----------------------------------------------------------------------------

# Save params -----------------------------------------------------------------
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
    
np.savetxt(f"BO_target_{mode}_{itrNum}.txt",targets,fmt="%6f")
np.savetxt(f"BO_paramb_{mode}_{itrNum}.txt",params,fmt=f"%6f")
# -----------------------------------------------------------------------------
