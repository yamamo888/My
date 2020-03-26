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
import seaborn as sns

import numpy as np
from scipy import stats
import pandas as pd
import csv

import PlotPF as myPlot

# ---- path ---- #
logsPath = 'logs'
imgPath = "images"
# simulated path
# for windows var.
dataPath = "tmp"
featurePath = "nankairirekifeature"
# for linux var.
#dataPath = 'b2b3b4b5b60-100'
#featurePath = "features"
dsdirPath = "MSE"
fname = '*.txt' 
# --------------- #

# ---- params ---- #
# eq. year in logs     
yInd = 1
vInds = [2,3,4,5,6,7,8,9]
simlateCell = 8
nCell = 8
# 安定した年
stateYear = 2000
# assimilation period
aYear = 1400
nYear = 10000
slip = 0
# threshold or deltaU
th = 1
ntI,tntI,ttI = 0,1,2
# ---------------- #

# -----------------------------------------------------------------------------
# makingData.pyとはちょっと違う
def loadABLV(logFullPath):
    
    data = open(logFullPath).readlines()
    
    B = np.zeros(nCell)
    
    for i in np.arange(1,nCell+1):
        tmp = np.array(data[i].strip().split(",")).astype(np.float32)
        B[i-1] = tmp[1]
    
    B = np.concatenate((B[2,np.newaxis],B[4,np.newaxis],B[5,np.newaxis]),0)
    
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

# -----------------------------------------------------------------------------
def convV2YearlyData(V,isPlot=False):
    
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
    
    # plot --------------------------------------------------------------
    if isPlot:
        fig, figInds = plt.subplots(nrows=3, sharex=True)
        for figInd in np.arange(len(figInds)):
            figInds[figInd].plot(np.arange(yV.shape[0]), yV[:,figInd])
        
        plt.savefig(os.path.join("deltaU.png"))
        plt.close()
    # -------------------------------------------------------------------
    
    return yV
# -----------------------------------------------------------------------------

# 間隔　------------------------------------------------------------------------
def convV2IntervalData(V):
    
    sYear = np.floor(V[0,yInd])
    yV = np.zeros([nYear,nCell])
    
    for year in np.arange(sYear,nYear):
        if np.sum(np.floor(V[:,yInd])==year):
            yV[int(year)] = V[np.floor(V[:,yInd])==year,vInds[0]:]
        else:
            yV[int(year)] = yV[int(year)-1,:] # shape=[100000,8]
            
    yV = np.concatenate((yV[stateYear:,2,np.newaxis],yV[stateYear:,4,np.newaxis],yV[stateYear:,5,np.newaxis]),1)
    yV = yV[1:] - yV[:-1]
    
    # eq. year
    nkY = np.where(yV[:,0]>slip)[0]
    tnkY = np.where(yV[:,1]>slip)[0]
    tkY = np.where(yV[:,2]>slip)[0]
    
    # interval for each cell, [?]
    intervalnkY = nkY[1:] - nkY[:-1]
    intervaltnkY = tnkY[1:] - tnkY[:-1]
    intervaltkY = tkY[1:] - tkY[:-1]
    
    return intervalnkY, intervaltnkY, intervaltkY
# -----------------------------------------------------------------------------
    
# 全間隔 ----------------------------------------------------------------------
def MinErrorNankai(gt,pred):
    
    # ----
    # 真値の地震発生年数
    gYear_nk = np.where(gt[:,0] > slip)[0]
    gYear_tnk = np.where(gt[:,1] > slip)[0]
    gYear_tk = np.where(gt[:,2] > slip)[0]
    # ----
    
    # 閾値 & 二乗誤差 順番
    if mode == 3:
        flag = False
        # Slide each one year 
        for sYear in np.arange(8000-aYear): 
            eYear = sYear + aYear
            
            # Num. of gt eq
            gNum_nk = gYear_nk.shape[0]
            gNum_tnk = gYear_tnk.shape[0]
            gNum_tk = gYear_tk.shape[0]
            
            # 閾値以上の予測した地震年数
            pYear_nk = np.where(pred[sYear:eYear,0] > th)[0][:gNum_nk]
            pYear_tnk = np.where(pred[sYear:eYear,1] > th)[0][:gNum_tnk]
            pYear_tk = np.where(pred[sYear:eYear,2] > th)[0][:gNum_tk]
            
            # gtよりpredの地震回数が少ない場合
            if pYear_nk.shape[0] < gNum_nk:
                pYear_nk = np.hstack([pYear_nk, np.tile(pYear_nk[-1], gNum_nk-pYear_nk.shape[0])])
            if pYear_tnk.shape[0] < gNum_tnk:
                pYear_tnk = np.hstack([pYear_tnk, np.tile(pYear_tnk[-1], gNum_tnk-pYear_tnk.shape[0])])
            if pYear_tk.shape[0] < gNum_tk:
                pYear_tk = np.hstack([pYear_tk, np.tile(pYear_tk[-1], gNum_tk-pYear_tk.shape[0])])
            # [9,]
            ndist_nk = gauss(gYear_nk,pYear_nk)
            ndist_tnk = gauss(gYear_tnk,pYear_tnk)
            ndist_tk = gauss(gYear_tk,pYear_tk)
            
            # 真値に合わせて二乗誤差
            yearError_nk = np.sum(ndist_nk)
            yearError_tnk = np.sum(ndist_tnk)
            yearError_tk = np.sum(ndist_tk)
            
            yearError = yearError_nk + yearError_tnk + yearError_tk
              
            if not flag:
                yearErrors = yearError
                flag = True
            else:
                yearErrors = np.hstack([yearErrors,yearError])
               
        # 最小誤差開始修了年数(1400年)取得
        sInd = np.argmin(yearErrors)
        

    # 閾値 & 二乗誤差
    elif mode == 2:
       
        flag = False
        # Slide each one year 
        for sYear in np.arange(8000-aYear): 
            eYear = sYear + aYear

            # 閾値以上の予測した地震年数
            pYear_nk = np.where(pred[sYear:eYear,0] > th)[0]
            pYear_tnk = np.where(pred[sYear:eYear,1] > th)[0]
            pYear_tk = np.where(pred[sYear:eYear,2] > th)[0]
            
            ndist_nk = gauss(gYear_nk,pYear_nk.T)
            ndist_tnk = gauss(gYear_tnk,pYear_tnk.T)
            ndist_tk = gauss(gYear_tk,pYear_tk.T)
            
            # 真値に合わせて二乗誤差
            yearError_nk = np.sum(np.min(ndist_nk,1))
            yearError_tnk = np.sum(np.min(ndist_tnk,1))
            yearError_tk = np.sum(np.min(ndist_tk,1))
            
            yearError = yearError_nk + yearError_tnk + yearError_tk
              
            if not flag:
                yearErrors = yearError
                flag = True
            else:
                yearErrors = np.hstack([yearErrors,yearError])
               
        # 最小誤差開始修了年数(1400年)取得
        sInd = np.argmin(yearErrors)
        
    elif mode == 0:
   
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

    # 最小誤差確率　
    maxSim = yearErrors[sInd]
    return maxSim, pred[sInd:sInd+aYear]
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def gauss(gtY,predY,sigma=100):
    
    # gauss
    if mode == 0:
        # predict matrix for matching times of gt eq.
        predYs = predY.repeat(gtY.shape[0],0).reshape(-1,gtY.shape[0])
        # gt var.
        gtYs = gtY.repeat(predY.shape[0],0).reshape(-1,predY.shape[0])
    
        gauss = np.exp(-(gtYs - predYs.T)**2/(2*sigma**2))
     
    # mse
    elif mode == 2:
       
        predYs = predY.repeat(gtY.shape[0],0).reshape(-1,gtY.shape[0])
        gtYs = gtY.repeat(predY.shape[0],0).reshape(-1,predY.shape[0])
    
        gauss = (gtYs - predYs.T)**2
    
    # mse
    elif mode == 3:
        
        gauss = (gtY - predY)**2
    
    return gauss    
# -----------------------------------------------------------------------------
    
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    
    # ---- command ---- #
    mode = int(sys.argv[1]) # 0:全探索 gauss var 2:全探索 vecter var 3:全探索 順番 var
    # ----------------- #

    # ---- bool ---- #
    # 3.
    isplotbestPath = False
    # 2. best 100 txt
    ismakingbestPath = False
    # 1. all research
    ismakingminPath = True
    isPlot = False
    # -------------- #
    
    # -------------------------------------------------------------------------
    # loading nankai trough (ground truth)
    with open(os.path.join(featurePath,"nankairireki.pkl"),'rb') as fp:
        gts = pickle.load(fp)
    # -------------------------------------------------------------------------
  
    # -------------------------------------------------------------------------
    if isplotbestPath:    
        # ---------------------------------------------------------------------
        logfilePath = os.path.join(dsdirPath,"best100_2","*txt")
        
        logfiles = glob.glob(logfilePath)
        # ---------------------------------------------------------------------
        flag = False
        for logfile in logfiles:
            
            print(logfile)
            
            V,B = loadABLV(logfile)
            yV = convV2YearlyData(V)
            maxSim, pred = MinErrorNankai(gts[190,:,:],yV)
            # eq. of pred
            predeq_nk = np.where(pred[:,ntI]>th)[0]
            predeq_tnk = np.where(pred[:,tntI]>th)[0]
            predeq_tk = np.where(pred[:,ttI]>th)[0]
            
            # plot gt & pred rireki
            myPlot.Rireki(gts[190,:,:],pred,title=f"{predeq_nk}\n{predeq_tnk}\n{predeq_tk}",label=f"{maxSim}_{np.round(B[ntI],6)}_{np.round(B[tntI],6)}_{np.round(B[ttI],6)}")
            
            if not flag:
                Bs = B
                flag = True
            else:
                Bs = np.vstack([Bs,B])
        myPlot.scatter3D(Bs[:,ntI],Bs[:,tntI],Bs[:,ttI],rangeP=[np.min(Bs,0),np.max(Bs,0)],title="top 100",label="best100_2")
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    if ismakingbestPath:
        # ---------------------------------------------------------------------
        dsfilePath = os.path.join(dsdirPath,"DS*")
        pathfilePath = os.path.join(dsdirPath,"path*")
        
        dsfiles = glob.glob(dsfilePath)
        pathfiles = glob.glob(pathfilePath)
        # ---------------------------------------------------------------------
    
        # reading DS & path files ---------------------------------------------
        flag = False
        for dfp,pfp in zip(dsfiles,pathfiles):
            ds = np.loadtxt(dfp)
            with open(pfp) as f:
                path = [line.strip() for line in f.readlines()]
            #pdb.set_trace()
            if not flag:
                DS = ds
                Path = np.array(path)
                flag = True
            else:
                DS = np.hstack([DS,ds])
                Path = np.hstack([Path,np.array(path)])
        # get best 100 in ds
        #bestInds = np.argsort(DS)[::-1][:100]
        bestInds = np.argsort(DS)[:100]
        bestPath = Path[bestInds]
        bestDS = DS[bestInds]
        # ---------------------------------------------------------------------
        pdb.set_trace()
        # save best path & ds -------------------------------------------------
        with open(os.path.join(dsdirPath,"bestPath100.txt"),"w") as f:
            f.write("\n".join(bestPath))
        np.savetxt(os.path.join(dsdirPath,"bestDS100.txt"),bestDS,fmt=f"%0d")
        # ---------------------------------------------------------------------
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    if ismakingminPath:    
        # --------------------------------------------------------------
        # simulated rireki mode == 0 
        filePath = os.path.join(logsPath,dataPath,fname)
        files = glob.glob(filePath)
        # --------------------------------------------------------------
        #pdb.set_trace()
        # 全間隔 ---------------------------------------------------------------
        if mode == 0 or mode == 2 or mode == 3:
            cnt = 0
            for tfID in [190]:
                
                # 全領域と確実領域の南海トラフ巨大地震履歴
                gtyV = gts[tfID,:,:]
            
                flag = False
                for file in files:
                    # simulation ------
                    print(file)
                    print(len(files)-cnt)
                    # loading logs B:[3,]
                    V,B = loadABLV(file)    
                    # V -> yV 
                    yV = convV2YearlyData(V)
                    # -----------------

                    # maxSim : Degree of Similatery
                    maxSim, _ = MinErrorNankai(gtyV,yV)
                    
                    if not flag:
                        maxSims = maxSim
                        paths = file
                        flag = True
                    else:
                        maxSims = np.hstack([maxSims,maxSim])
                        paths = np.hstack([paths,file])
                    
                    cnt += 1
                
                # min error top 100
                maxSimInds = np.argsort(maxSims)[:100]
                # path for reading
                maxpaths = paths[maxSimInds]
                # sort degree of similatery 
                maxSims = maxSims[maxSimInds]
                        
            for line in maxpaths:
                # output csv for path
                with open(f"path_{dataPath}_{mode}.txt","a") as f:  
                    f.writelines(line)
                    f.write("\n")
            # output csv for degree of similatery
            np.savetxt(f"DS_{dataPath}_{mode}.txt",maxSims,fmt="%d")
    # -------------------------------------------------------------------------