# -*- coding: utf-8 -*-

import os
import sys
import glob
import pickle
import pdb
import time

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pylab as plt

import numpy as np
from scipy import stats

import csv

# ---- command ---- #
cell = int(sys.argv[1])
# ----------------- #

# ---- path ---- #
# dataPath
logsPath = 'logs'
# simulated path
#dataPath = 'b2b3b4b5b60-100'
dataPath = "tmp"
#dataPath = "190"
featurePath = "nankairirekifeature"
fname = '*.txt' 
# -------------- #

# ---- params ---- #

nCell = 8

# gt cell index
if cell == 2:
    gtcell = 0
elif cell == 4:
    gtcell = 1
elif cell == 5:
    gtcell = 2
    
# eq. year in logs     
yrInd = 1
yInd = 1
vInds = [2,3,4,5,6,7,8,9]
simlateCell = 8
# いいんかな？
slip = 0

# 安定した年
stateYear = 2000
# assimilation period
aYear = 1400

nYear = 10000
# ---------------- #

# -----------------------------------------------------------------------------
def loadABLV(logFullPath):
    
    data = open(logFullPath).readlines()
    
    B = np.zeros(nCell)
    
    for i in np.arange(1,nCell+1):
        tmp = np.array(data[i].strip().split(",")).astype(np.float32)
        B[i-1] = tmp[1]
        
    # Vの開始行取得
    isRTOL = [True if data[i].count('value of RTOL')==1 else False for i in np.arange(len(data))]
    vInd = np.where(isRTOL)[0][0]+1
    
    # Vの値の取得（vInd行から最終行まで）
    flag = False
    for i in np.arange(vInd,len(data)):
        tmp = np.array(data[i].strip().split(",")).astype(np.float32)
        
        if not flag:
            V = tmp
            flag = True
        else:
            V = np.vstack([V,tmp])
            
    return V, B

# -----------------------------------------------------------------------------
def convV2YearlyData(V):
        
    # 初めの観測した年
    sYear = np.floor(V[0,yInd])
    yV = np.zeros([nYear,nCell])
    # 観測データがない年には観測データの１つ前のデータを入れる(累積)
    for year in np.arange(sYear,nYear):
        # 観測データがある場合
        if np.sum(np.floor(V[:,yInd])==year):
            # 観測データがあるときはそのまま代入
            yV[int(year)] = V[np.floor(V[:,yInd])==year,vInds[0]:]
        
        # 観測データがない場合
        else:
            # その1つ前の観測データを入れる
            yV[int(year)] = yV[int(year)-1,:]
    # 累積速度から、速度データにする
    deltaV = yV[yInd:]-yV[:-yInd]
    # 一番最初のデータをappendして、10000年にする
    yV = np.concatenate((yV[np.newaxis,0],deltaV),0)
    # shape=[8000,3]
    yV = np.concatenate((yV[stateYear:,2,np.newaxis],yV[stateYear:,4,np.newaxis],yV[stateYear:,5,np.newaxis]),1)
    
    return yV
    
# -----------------------------------------------------------------------------
def MinErrorNankai(gt,pred):
    """
    pred: [8000,3]
    """
    
    if cell == 2 or cell == 4 or cell ==5:
        # ----
        # 真値の地震年数
        gYear = np.where(gt[:,gtcell] > slip)[0]
        # ----

        flag = False
        # Slide each one year 
        for sYear in np.arange(8000-aYear): 
            # 予測した地震の年数 + 1400
            eYear = sYear + aYear

            # 予測した地震年数 only one-cell
            pYear = np.where(pred[sYear:eYear,gtcell] > slip)[0]

            # gaussian distance for year of gt - year of pred (gYears.shape, pred.shape)
            ndist = gauss(gYear,pYear.T)

            # 予測誤差の合計, 回数で割ると当てずっぽうが小さくなる
            yearError = sum(ndist.max(1)/pYear.shape[0])

            if not flag:
                yearErrors = yearError
                flag = True
            else:
                yearErrors = np.hstack([yearErrors,yearError])
         
    # for 3 cell
    elif cell == 245:
        # ----
        # 真値の地震年数
        gYear_nk = np.where(gt[:,0] > slip)[0]
        gYear_tnk = np.where(gt[:,1] > slip)[0]
        gYear_tk = np.where(gt[:,2] > slip)[0]
        # ----

        flag = False
        # Slide each one year 
        for sYear in np.arange(8000-aYear): 
            # 予測した地震の年数 + 1400
            eYear = sYear + aYear

            # 予測した地震年数 only one-cell
            pYear_nk = np.where(pred[sYear:eYear,0] > slip)[0]
            pYear_tnk = np.where(pred[sYear:eYear,1] > slip)[0]
            pYear_tk = np.where(pred[sYear:eYear,2] > slip)[0]


            # gaussian distance for year of gt - year of pred (gYears.shape, pred.shape)
            # for each cell
            ndist_nk = gauss(gYear_nk,pYear_nk.T)
            ndist_tnk = gauss(gYear_tnk,pYear_tnk.T)
            ndist_tk = gauss(gYear_tk,pYear_tk.T)

            # 予測誤差の合計, 回数で割ると当てずっぽうが小さくなる
            # for each cell
            yearError_nk = sum(ndist_nk.max(1)/pYear_nk.shape[0])
            yearError_tnk = sum(ndist_tnk.max(1)/pYear_tnk.shape[0])
            yearError_tk = sum(ndist_tk.max(1)/pYear_tk.shape[0])
            
            # for all cell
            yearError = yearError_nk + yearError_tnk + yearError_tk

            if not flag:
                yearErrors = yearError
                flag = True
            else:
                yearErrors = np.hstack([yearErrors,yearError])
    # 最小誤差開始修了年数(1400年)取得
    sInd = np.argmax(yearErrors)
    eInd = sInd + aYear

    # 最小誤差確率　
    maxSim = yearErrors[sInd]
    
    return pred[sInd:eInd,:], maxSim

# -----------------------------------------------------------------------------
def gauss(gtY,predY,sigma=100):

    # predict matrix for matching times of gt eq.
    predYs = predY.repeat(gtY.shape[0],0).reshape(-1,gtY.shape[0])
    # gt var.
    gtYs = gtY.repeat(predY.shape[0],0).reshape(-1,predY.shape[0])

    gauss = np.exp(-(gtYs - predYs.T)**2/(2*sigma**2))

    return gauss    

# -----------------------------------------------------------------------------
def loadFile(filespath):    

    batFile = "featureV.bat"
    
    flag = False
    for filepath in filespath:
        V,B = loadABLV(filepath)
    
        if not flag:
            param = B[np.newaxis] * 1000000
            flag = True
        else:
            param = np.vstack([param,B[np.newaxis]*1000000])
    
    param = np.concatenate((param[:,2,np.newaxis],param[:,4,np.newaxis],param[:,5,np.newaxis]),1)
    param = np.round(param,1).astype(int)
    
    # save csv ----------------------------------------------------------------
    with open("allneighver_190.csv","w") as f:
        line = csv.writer(f,lineterminator="\n")
        line.writerows(param)
    # -------------------------------------------------------------------------
    pdb.set_trace()
    # bat ---------------------------------------------------------------------
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


if __name__ == "__main__":
    
   
    """
    # --------------------------------------------------------------
    logfiles = glob.glob(os.path.join(logsPath,dataPath,fname))
    # loading logs & output first pf files
    loadFile(logfiles)
    # --------------------------------------------------------------
    """
    # --------------------------------------------------------------
    # simulated rireki
    filePath = os.path.join(logsPath,dataPath,fname)
    files = glob.glob(filePath)
    # --------------------------------------------------------------
    
    # --------------------------------------------------------------
    # loading nankai trough (ground truth)
    with open(os.path.join(logsPath,"nankairireki.pkl"),'rb') as fp:
        gtyV = pickle.load(fp)
    # --------------------------------------------------------------
    
    for tfID in [190]:
        
        # ---- reading gt ---- # 
        # 全領域と確実領域の南海トラフ巨大地震履歴
        gtU = gtyV[tfID,:,:]
        # deltaU -> slip velocity 
        gtUV = np.vstack([np.zeros(3)[np.newaxis], gtU[1:,:] - gtU[:-1,:]])
        # -------------------- #
        
        flag = False
        for fID in np.arange(len(files)):
            
            # file path
            logFullPath = files[fID]
            print(logFullPath)
            
            # loading logs 
            V,B = loadABLV(logFullPath)
            
            # V -> yV 
            yV = convV2YearlyData(V)
            
            # maxSim : Degree of Similatery
            # minimum error yV ,shape=[1400,3]
            minyV, maxSim = MinErrorNankai(gtUV,yV)
            
            if not flag:
                maxSims = maxSim
                paths = logFullPath
                flag = True
            else:
                maxSims = np.hstack([maxSims,maxSim])
                paths = np.hstack([paths,logFullPath])
        
        maxSimInds = np.argsort(maxSims)[::-1][:100]
        # path for reading
        maxpaths = paths[maxSimInds]
        # sort degree of similatery 
        maxSim = maxSims[maxSimInds]
        
        # ----------------------------------------------------------
        for line in maxpaths:
            # output csv for path
            with open(f"path_{tfID}_{dataPath}_{cell}.txt","a") as f:
                f.write(line + "\n")
        for ms in maxSim:
            # output csv for path
            with open(f"DS_{tfID}_{dataPath}_{cell}.txt","a") as f:
                f.write(str(ms) + "\n")
        # ----------------------------------------------------------
        
        
        
        
        
        
    
    
    
    
    
    



