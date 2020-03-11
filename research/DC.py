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
#dataPath = 'b2b3b4b5b60-100'
dataPath = "deltaU"
#dataPath = "190"
featurePath = "nankairirekifeature"
dsdirPath = "DS_path"
fname = '*.txt' 
# --------------- #

# ---- params ---- #
# eq. year in logs     
yInd = 1
vInds = [2,3,4,5,6,7,8,9]
simlateCell = 8
slip = 0
nCell = 8
# 安定した年
stateYear = 2000
# assimilation period
aYear = 1400
nYear = 10000
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
    nkY = np.where(yV[:,0]>0)[0]
    tnkY = np.where(yV[:,1]>0)[0]
    tkY = np.where(yV[:,2]>0)[0]
    
    # interval for each cell, [?]
    intervalnkY = nkY[1:] - nkY[:-1]
    intervaltnkY = tnkY[1:] - tnkY[:-1]
    intervaltkY = tkY[1:] - tkY[:-1]
    
    return intervalnkY, intervaltnkY, intervaltkY
# -----------------------------------------------------------------------------
    
# 全間隔 ----------------------------------------------------------------------
def MinErrorNankai(gt,pred,label="none",isPlot=False):
    
    # 閾値 & 二乗誤差
    if mode == 2:
        # ----
        gYear_nk = np.where(gt[:,0] > slip)[0]
        gYear_tnk = np.where(gt[:,1] > slip)[0]
        gYear_tk = np.where(gt[:,2] > slip)[0]
        # ----
        
        plot_pred1 = np.zeros([aYear,3])
        plot_pred2 = np.zeros([aYear,3])
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
            
            ind_nk = np.argmin(ndist_nk,1).tolist()
            ind_tnk = np.argmin(ndist_tnk,1).tolist()
            ind_tk = np.argmin(ndist_tk,1).tolist()
            # 各真値に対して1番合ってる予測年数
            matchYear_nk = pYear_nk[ind_nk]
            matchYear_tnk = pYear_tnk[ind_tnk]
            matchYear_tk = pYear_tk[ind_tk]
            
            if not flag:
                yearErrors = yearError
                matchYears_nk = matchYear_nk
                matchYears_tnk = matchYear_tnk
                matchYears_tk = matchYear_tk
                flag = True
            else:
                yearErrors = np.hstack([yearErrors,yearError])
                matchYears_nk = np.vstack([matchYears_nk,matchYear_nk])
                matchYears_tnk = np.vstack([matchYears_tnk,matchYear_tnk])
                matchYears_tk = np.vstack([matchYears_tk,matchYear_tk])
                
        # 最小誤差開始修了年数(1400年)取得
        sInd = np.argmax(yearErrors)
        eInd = sInd + aYear
    
        # 最小誤差確率　
        maxSim = yearErrors[sInd]
        
        # predict [1400,]
        tmp1 = pred[sInd:eInd,ntI]
        tmp2 = pred[sInd:eInd,tntI]
        tmp3 = pred[sInd:eInd,ttI]
        # predict (threshold)
        plot_pred1[np.where(tmp1>th)[0].tolist(),ntI] = 30
        plot_pred1[np.where(tmp2>th)[0].tolist(),tntI] = 30
        plot_pred1[np.where(tmp3>th)[0].tolist(),ttI] = 30
        # best match index
        myear_nk = matchYears_nk[sInd]
        myear_tnk = matchYears_tnk[sInd]
        myear_tk = matchYears_tk[sInd]
        # best predict
        plot_pred2[myear_nk,ntI] = 30
        plot_pred2[myear_tnk,tntI] = 30
        plot_pred2[myear_tk,ttI] = 30
        # gt eq - best predict eq. year
        dyear_nk = np.abs(gYear_nk - myear_nk)
        dyear_tnk = np.abs(gYear_tnk - myear_tnk)
        dyear_tk = np.abs(gYear_tk - myear_tk)
        
        # all
        #myPlot.rireki(gt,plot_pred1,dirPath="deltaU",title=f"{myear_nk}\n{myear_tnk}\n{myear_tk}",label=f"{np.round(maxSim,5)}_{label}")
        # best
        #myPlot.rireki(gt,plot_pred2,dirPath="best_deltaU",title=f"nk:{dyear_nk}\ntnk:{dyear_tnk}\ntk{dyear_tk}",label=f"{np.round(maxSim,5)}_{label}")
      
        return maxSim
    
    elif mode == 0 or mode == 1:
    
        if cell == 2 or cell == 4 or cell == 5:
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
            #pdb.set_trace()
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
                #pdb.set_trace()
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
                  
        return maxSim

# 部分間隔 --------------------------------------------------------------------
def PartMinErrorNankai(pred):
    
    # eq. year for each cell
    nkYear = np.max(np.where(pred[:,0]>0)[0][:-2])
    tnkYear = np.max(np.where(pred[:,1]>0)[0][:-2])
    tkYear = np.max(np.where(pred[:,2]>0)[0][:-2])
    # last eq. year for all cell
    lastYear = np.min([nkYear,tnkYear,tkYear])
    # eq. year for all cell
    allYear = np.unique(np.where(pred>0)[0])
    # nextindがあるように(次の地震があるように)
    gpYear = allYear[:np.where(allYear==lastYear)[0][0]+1]
    
    modes = np.zeros([gpYear.shape[0],6])
    cnt = 0
    # Slide each eq. year
    for sYear in gpYear:
        
        # [eq(t),eq(t+1)]
        pYears_nk = np.where(pred[sYear:,0] > slip)[0][:2] + sYear
        pYears_tnk = np.where(pred[sYear:,1] > slip)[0][:2] + sYear
        pYears_tk = np.where(pred[sYear:,2] > slip)[0][:2] + sYear
        
        # eq(t+1)
        nextind = np.unique(np.hstack([pYears_nk,pYears_tnk,pYears_tk]))[1]
        
        # [eq(t+1),eq(t+2)]
        next_nk = np.where(pred[nextind:,0] > slip)[0][:1] + nextind
        next_tnk = np.where(pred[nextind:,1] > slip)[0][:1] + nextind
        next_tk = np.where(pred[nextind:,2] > slip)[0][:1] + nextind
        
        # interval eq.
        interval_nk = next_nk - sYear
        interval_tnk = next_tnk - sYear
        interval_tk = next_tk - sYear
        
        # onehot
        onehot_nk = (next_nk==nextind).astype(int)
        onehot_tnk = (next_tnk==nextind).astype(int)
        onehot_tk = (next_tk==nextind).astype(int)
        
        # eq(t+1)
        pYear_nk = (interval_nk * onehot_nk)
        pYear_tnk = (interval_tnk * onehot_tnk)
        pYear_tk = (interval_tk * onehot_tk)
        
        # 1
        if pred[sYear,0] == 0 and pred[sYear,1] > 0 and pred[sYear,2] == 0:
            
            gYear_nk = np.array([260])
            gYear_tnk = np.array([260])
            gYear_tk = np.array([0])
        
            # gaussian distance for year of gt - year of pred (gYears.shape, pred.shape)
            # for each cell
            ndist_nk = gauss(gYear_nk,pYear_nk)
            ndist_tnk = gauss(gYear_tnk,pYear_tnk)
            ndist_tk = gauss(gYear_tk,pYear_tk)
            
            max_ndist = ndist_nk + ndist_tnk + ndist_tk
            
            # degree of similatery
            modes[cnt,0] = max_ndist
            
            cnt += 1
        # 2, 4
        elif pred[sYear,0] > 0 and pred[sYear,1] > 0 and pred[sYear,2] == 0:
            
            gYear1 = [0,210,0]
            gYear2 = [210,210,210]
            
            flag2 = False
            for gYear in [gYear1,gYear2]:
                gYear_nk = np.array([gYear[0]])
                gYear_tnk = np.array([gYear[1]])
                gYear_tk = np.array([gYear[2]])
            
                ndist_nk = gauss(gYear_nk,pYear_nk)
                ndist_tnk = gauss(gYear_tnk,pYear_tnk)
                ndist_tk = gauss(gYear_tk,pYear_tk)
                    
                ndist = ndist_nk + ndist_tnk + ndist_tk 
                
                if not flag2:
                    ndists = ndist
                    flag2 = True
                else:
                    ndists = np.hstack([ndists,ndist])
                
            maxind = np.argmax(ndists)
            max_ndist = ndists[maxind]
    
            if maxind == 0:
                modes[cnt,1] = max_ndist
            elif maxind == 1:
                modes[cnt,3] = max_ndist
            
            cnt += 1
        
        # 3, 5, 6
        elif pred[sYear,0] > 0 and pred[sYear,1] > 0 and pred[sYear,2] > 0:
            
            gYear1 = [260,260,0]
            gYear2 = [150,150,150]
            gYear3 = [90,90,0]
             
            flag3 = False
            for gYear in [gYear1,gYear2,gYear3]:
                gYear_nk = np.array([gYear[0]])
                gYear_tnk = np.array([gYear[1]])
                gYear_tk = np.array([gYear[2]])
                
                ndist_nk = gauss(gYear_nk,pYear_nk)
                ndist_tnk = gauss(gYear_tnk,pYear_tnk)
                ndist_tk = gauss(gYear_tk,pYear_tk)
                
                ndist = ndist_nk + ndist_tnk + ndist_tk 
                
                if not flag3:
                    ndists = ndist
                    flag3= True
                else:
                    ndists = np.hstack([ndists,ndist])
    
            maxind = np.argmax(ndists)
            max_ndist = ndists[maxind]
            
            if maxind == 0:
                modes[cnt,2] = max_ndist
            
            elif maxind == 1:    
                modes[cnt,4] = max_ndist
            
            elif maxind == 2:
                modes[cnt,5] = max_ndist
    
            cnt += 1
        # ignore
        else:
            cnt += 1
            pass
    
    # max degree of similatery for all cell, [6,]
    maxSim = np.max(modes,0)

    return maxSim

# -----------------------------------------------------------------------------
def gauss(gtY,predY,sigma=100):
    
    if mode == 0:
        #pdb.set_trace()
        # predict matrix for matching times of gt eq.
        predYs = predY.repeat(gtY.shape[0],0).reshape(-1,gtY.shape[0])
        # gt var.
        gtYs = gtY.repeat(predY.shape[0],0).reshape(-1,predY.shape[0])
    
        gauss = np.exp(-(gtYs - predYs.T)**2/(2*sigma**2))

    elif mode == 1:
        
        gauss = np.exp(-(gtY - predY)**2/(2*sigma**2))
        
    elif mode == 2:
       
        predYs = predY.repeat(gtY.shape[0],0).reshape(-1,gtY.shape[0])
        gtYs = gtY.repeat(predY.shape[0],0).reshape(-1,predY.shape[0])
    
        gauss = (gtYs - predYs.T)**2
        
    return gauss    
# -----------------------------------------------------------------------------
    
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    
    # ---- command ---- #
    cell = int(sys.argv[1])
    mode = int(sys.argv[2])
    # ----------------- #
    
    if mode == 0:
        # gt cell index
        if cell == 2:
            gtcell = 0
        elif cell == 4:
            gtcell = 1
        elif cell == 5:
            gtcell = 2
        
    # ---- bool ---- #
    ismakingbestPath = False
    ismakingbestpartPath = False
    issavebestPath = False
    ismakingminPath = True
    # -------------- #
    
    # -------------------------------------------------------------------------
    with open(os.path.join(featurePath,"nankairireki.pkl"),'rb') as fp:
            gtyV = pickle.load(fp) # [256,1400,3]
    # -------------------------------------------------------------------------
    #pdb.set_trace()
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
        bestInds = np.argsort(DS)[::-1][:100]
        bestPath = Path[bestInds]
        bestDS = DS[bestInds]
        # ---------------------------------------------------------------------
        
        # save best path & ds -------------------------------------------------
        with open(os.path.join(dsdirPath,"bestPath100.txt"),"w") as f:
            f.write("\n".join(bestPath))
        np.savetxt(os.path.join(dsdirPath,"bestDS100.txt"),bestDS)
        # ---------------------------------------------------------------------
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    if issavebestPath:
        with open(os.path.join(dsdirPath,"bestPath100.txt")) as f:
            bestPath = [line.strip() for line in f.readlines()] 
        bestDS = np.loadtxt(os.path.join(dsdirPath,"bestDS100.txt"))
        
        flag = False
        for bp in bestPath:
            fileName = os.path.basename(bp)
            print(fileName)
            # get b     
            V, tmpb = loadABLV(os.path.join(dsdirPath,"DS_190",fileName))
            yV = convV2YearlyData(V)
            # ※エラーはくかも
            b = np.concatenate((tmpb[2,np.newaxis],tmpb[4,np.newaxis],tmpb[5,np.newaxis]),0)
            
            # plot yV ---------------------------------------------------------
            #bestyV, _ = MinErrorNankai(gtyV[190,:,:],yV,label=f"{np.round(b[0],6)}_{np.round(b[1],6)}_{np.round(b[2],6)}",isPlot=True)
            # -----------------------------------------------------------------
            
            if not flag:
                B = b
                flag = True
            else:
                B = np.vstack([B,b])
        
        # save best B ---------------------------------------------------------
        #np.savetxt(os.path.join(dsdirPath,"bestB100.txt"),B,fmt="%5f")
        # ---------------------------------------------------------------------
        
        # plot parameter b ----------------------------------------------------
        myPlot.scatter3D(B[:,0],B[:,1],B[:,2],title="all searth",label="allresearchB")
        myPlot.HistLikelihood(bestDS,label="allsearchDS")
        # ---------------------------------------------------------------------
        
    # -------------------------------------------------------------------------
    if ismakingbestpartPath:
        pathPath = glob.glob(os.path.join(dsdirPath,"part","path*"))
        dsPath = glob.glob(os.path.join(dsdirPath,"part","DS*"))
        
        flag = False
        for i in np.arange(5):
            with open(pathPath[i]) as f:
                # reading path
                path = [line.strip() for line in f.readlines()] 
            # reading ds
            file_pd = pd.read_csv(dsPath[i],header=None)
            file = file_pd.values
            # pandas -> array
            FLAG = False
            for line in file:
                lines = np.array(line[0].split("]")[0][2:].split()).astype(float)
                if not FLAG:
                    ds = lines
                    FLAG = True
                else:
                    ds = np.vstack([ds,lines])
            # concate all files path & ds
            if not flag:
                paths = path
                dses = ds
                flag = True
            else:
                # [5,files]
                paths = np.vstack([paths,path])
                dses = np.vstack([dses,ds])
        pdb.set_trace()
        # index, [100,6]
        bestInd = np.argsort(dses,0)[::-1][:100]
        
        for ind in np.arange(bestInd.shape[1]):
            bestPath = paths[ind,bestInd[:,ind]]
        
            with open(os.path.join(dsdirPath,"part",f"bestpartPath100_{ind}.txt"),"w") as f:
                f.write("\n".join(bestPath))
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    if ismakingminPath:    
        # --------------------------------------------------------------
        # simulated rireki mode == 0 or mode ==1
        filePath = os.path.join(logsPath,dataPath,fname)
        files = glob.glob(filePath)
        # --------------------------------------------------------------
        #pdb.set_trace()
        # 全間隔 ---------------------------------------------------------------
        if mode == 0 or mode == 2:
            # --------------------------------------------------------------
            # loading nankai trough (ground truth)
            #with open(os.path.join(logsPath,"nankairireki.pkl"),'rb') as fp:
            with open(os.path.join(featurePath,"nankairireki.pkl"),'rb') as fp:
                gts = pickle.load(fp)
            # --------------------------------------------------------------
            
            for tfID in [190]:
                
                # 全領域と確実領域の南海トラフ巨大地震履歴
                gtyV = gts[tfID,:,:]
            
                flag = False
                for fID in np.arange(len(files)):
                    # simulation -----
                    # file path
                    logFullPath = files[fID]
                    print(logFullPath)    
                    # loading logs B:[3,]
                    V,B = loadABLV(logFullPath)    
                    # V -> yV 
                    yV = convV2YearlyData(V)
                    # -----------------

                    # maxSim : Degree of Similatery
                    maxSim = MinErrorNankai(gtyV,yV,label=f"{np.round(B[ntI],6)}_{np.round(B[tntI],6)}_{np.round(B[ttI],6)}")
                    #pdb.set_trace()
                    if not flag:
                        maxSims = maxSim
                        paths = logFullPath
                        flag = True
                    else:
                        maxSims = np.hstack([maxSims,maxSim])
                        paths = np.hstack([paths,logFullPath])
                #pdb.set_trace()
                # min error top 100
                maxSimInds = np.argsort(maxSims)[:100]
                # path for reading
                maxpaths = paths[maxSimInds]
                # sort degree of similatery 
                maxSim = maxSims[maxSimInds]
                        
                
        # 部分間隔 -------------------------------------------------------------   
        elif mode == 1:
            
            flag = False
            for fID in np.arange(len(files)):
                # simulation -----
                # file path
                logFullPath = files[fID]
                print(logFullPath)    
                # loading logs 
                V,B = loadABLV(logFullPath)    
                # V -> yV 
                yV = convV2YearlyData(V)
                # -----------------
    
                maxSim = PartMinErrorNankai(yV)
            
                if not flag:
                    maxSims = maxSim
                    paths = logFullPath
                    flag = True
                else:
                    maxSims = np.vstack([maxSims,maxSim])
                    paths = np.vstack([paths,logFullPath])
            # [100,6]
            maxSimInds = np.argsort(maxSims,0)[::-1][:100]
            maxSims = np.sort(maxSims,0)[::-1][:100]
            
        # 部分間隔のとき、コメントアウト外す
        #for ind in np.arange(maxSimInds.shape[1]):
            #maxpath = np.reshape(paths[maxSimInds[:,ind]],[-1,])
    
        for ind in np.arange(maxSimInds.shape[0]):
            maxpath = np.reshape(paths[maxSimInds[ind]],[-1,])
            # 部分間隔のとき、コメントアウト外す
            #maxSim = maxSims[ind]
            # -------------------------------------------------------------
            #pdb.set_trace()
            for line in maxpath:
                # output csv for path
                #with open(f"path_{tfID}_{dataPath}_{cell}.txt","a") as f:
                with open(f"path_{dataPath}_{mode}.txt","a") as f:  
                    f.writelines(line)
                    f.write("\n")
            
            # 部分間隔のとき、コメントアウト外す
            #for ms in maxSim:
                # output csv for path
                #with open(f"DS_{tfID}_{dataPath}_{cell}.txt","a") as f:
                #with open(f"DS_{dataPath}_{mode}_{ind}.txt","a") as f:
                    #f.write(str(ms) + "\n")
            # -------------------------------------------------------------
        np.savetxt(f"DS_{dataPath}_{mode}.txt",maxSims,fmt="%d")  
    # -------------------------------------------------------------------------
    