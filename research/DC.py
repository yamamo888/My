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
import seaborn as sns
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

# reading nankai rireki -------------------------------------------------------
with open(os.path.join(featurePath,"nankairireki.pkl"), "rb") as fp:
        nkfiles = pickle.load(fp)
gt = nkfiles[190,:,:]
# -----------------------------------------------------------------------------
   
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
def convV2YearlyData(V):
    # 初めの観測した年
    sYear = np.floor(V[0,yInd])
    yV = np.zeros([nYear,nCell])
    # 観測データがない年には観測データの１つ前のデータを入れる(累積)
    for year in np.arange(sYear,nYear):
        # 観測データがある場合
        if np.sum(np.floor(V[:,yInd])==year):
            # 観測データがあるときはそのまま代入 (U,theta,V出力された用)
            yV[int(year)] = V[np.floor(V[:,yInd])==year,vInds[0]:][0,:]
        
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
def MinErrorNankai(gt, pred, mode=3, label='test', isPlot=False):

    # ----
    # 真値の地震発生年数
    gYear_nk = np.where(gt[:,0] > slip)[0]
    gYear_tnk = np.where(gt[:,1] > slip)[0]
    gYear_tk = np.where(gt[:,2] > slip)[0]
    # ----
    
    # reverse var. [84,]
    gYear_nk_re = np.abs(gYear_nk-1400)
    gYear_tnk_re = np.abs(gYear_tnk-1400)
    gYear_tk_re = np.abs(gYear_tk-1400)
    
    # Num. of gt eq
    gNum_nk = gYear_nk.shape[0]
    gNum_tnk = gYear_tnk.shape[0]
    gNum_tk = gYear_tk.shape[0]

    # ----
    # part of gt eq. (after 761 year, start 761 = 0 year) [761,898,1005,...]->[0,137,24,...]
    part_nk = gYear_nk[4:] - 761
    part_tnk = gYear_tnk[4:] - 761
    part_tk = gYear_tk[4:] - 761
    # ----
    
    # reverse var.
    part_nk_re = gYear_nk_re[4:]
    part_tnk_re = gYear_tnk_re[4:]
    part_tk_re = gYear_tk_re[4:]
    
    partNum_nk = part_nk.shape[0] # 5
    partNum_tnk = part_tnk.shape[0] # 5
    partNum_tk = part_tk.shape[0] # 3
    
    # 1 対 1 (part & reverse) ----
    if mode == 7:
        
        flag = False
        
        # pred
        nk = np.where(pred[:,0]>1)[0]
        tnk = np.where(pred[:,1]>1)[0]
        tk = np.where(pred[:,2]>1)[0]
        
        lastyear = np.max([np.max(nk),np.max(tnk),np.max(tk)])
       
        for sYear in np.arange(8000-lastyear):
             
            # 閾値以上の予測した地震年数
            pYear_nk = (np.abs(nk[(nk<8000-sYear) & (nk>8000-761-sYear)] - 8000) -sYear)[-partNum_nk:]
            pYear_tnk = (np.abs(tnk[(tnk<8000-sYear) & (tnk>8000-761-sYear)] - 8000) -sYear)[-partNum_tnk:]
            pYear_tk =  (np.abs(tk[(tk<8000-sYear) & (tk>8000-761-sYear)] - 8000) -sYear)[-partNum_tk:]
            
            #pdb.set_trace()
            # pred < gt
            if pYear_nk.shape[0] < partNum_nk:
                # 真値より少ない場合は、600年(+600)以降で合わせる、同じ数になるように
                pYear_nk = (np.abs(nk[nk<8000-sYear] - 8000) -sYear)[-partNum_nk:]
                
                if pYear_nk.shape[0] < partNum_nk:
                     
                     pYear_nk = np.insert(pYear_nk, 0, np.tile(10000, partNum_nk-pYear_nk.shape[0]))
                 
            if pYear_tnk.shape[0] < partNum_tnk:
               
                pYear_tnk = (np.abs(tnk[tnk<8000-sYear] - 8000) -sYear)[-partNum_tnk:]
            
                if pYear_tnk.shape[0] < partNum_tnk:
            
                    pYear_tnk = np.insert(pYear_tnk, 0, np.tile(10000, partNum_tnk-pYear_tnk.shape[0]))
            
            if pYear_tk.shape[0] < partNum_tk:
                
                pYear_tk = (np.abs(tk[tk<8000-sYear] - 8000) -sYear)[-partNum_tk:]
                
                if pYear_tk.shape[0] < partNum_tk:
           
                    pYear_tk = np.insert(pYear_tk, 0, np.tile(10000, partNum_tk-pYear_tk.shape[0]))
            #pdb.set_trace()
            ndist_nk = gauss(part_nk_re,pYear_nk,mode=3)
            ndist_tnk = gauss(part_tnk_re,pYear_tnk,mode=3)
            ndist_tk = gauss(part_tk_re,pYear_tk,mode=3)
            
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
        
        sInd = np.argmin(yearErrors) 
    # ----
   
    # 1 対 1 (part & normal) ----
    if mode == 6:
            
        flag = False
        
        for sYear in np.arange(8000-600):
            #pdb.set_trace()
            eYear = sYear + 600
                
            # 閾値以上の予測した地震年数
            pYear_nk = np.where(pred[sYear:eYear,0] > th)[0][:partNum_nk]
            pYear_tnk = np.where(pred[sYear:eYear,1] > th)[0][:partNum_tnk]
            pYear_tk = np.where(pred[sYear:eYear,2] > th)[0][:partNum_tk]
            
             # pred < gt
            if pYear_nk.shape[0] < partNum_nk:
                # 真値より少ない場合は、1400年以降で合わせる、同じ数になるように
                pYear_nk = np.where(pred[sYear:,0] > th)[0][:partNum_nk]
                 
                if pYear_nk.shape[0] < partNum_nk:
                     
                     pYear_nk = np.hstack([pYear_nk, np.tile(10000, partNum_nk-pYear_nk.shape[0])])
                 
            if pYear_tnk.shape[0] < partNum_tnk:
               
                pYear_tnk = np.where(pred[sYear:,1] > th)[0][:partNum_tnk]
            
                if pYear_tnk.shape[0] < partNum_tnk:
            
                    pYear_tnk = np.hstack([pYear_tnk, np.tile(10000, partNum_tnk-pYear_tnk.shape[0])])
            
            if pYear_tk.shape[0] < partNum_tk:
                
                pYear_tk = np.where(pred[sYear:,2] > th)[0][:partNum_tk]
          
                if pYear_tk.shape[0] < partNum_tk:
           
                    pYear_tk = np.hstack([pYear_tk, np.tile(10000, partNum_tk-pYear_tk.shape[0])])
        
            # [9,]
            ndist_nk = gauss(part_nk,pYear_nk,mode=3)
            ndist_tnk = gauss(part_tnk,pYear_tnk,mode=3)
            ndist_tk = gauss(part_tk,pYear_tk,mode=3)
            
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
         
        sInd = np.argmin(yearErrors) 
    # ----
   
    # 1　対 1 (reverse) ----
    if mode == 5:
    
        flag = False
        
        # pred
        nk = np.where(pred[:,0]>1)[0]
        tnk = np.where(pred[:,1]>1)[0]
        tk = np.where(pred[:,2]>1)[0]
        
        lastyear = np.max([np.max(nk),np.max(tnk),np.max(tk)])
        
        for sYear in np.arange(8000-lastyear):
            
            # 8000年から手間に1400年分シフト(-sYear)していく
            # -8000することで小さい年数が大きい年数になり逆順になる
            pYear_nk = (np.abs(nk[(nk<8000-sYear) & (nk>8000-aYear-sYear)] - 8000) -sYear)[-gNum_nk:]
            pYear_tnk = (np.abs(tnk[(tnk<8000-sYear) & (tnk>8000-aYear-sYear)] - 8000) -sYear)[-gNum_tnk:]
            pYear_tk = (np.abs(tk[(tk<8000-sYear) & (tk>8000-aYear-sYear)] - 8000) -sYear)[-gNum_tk:]
             
            # pred < gt
            if pYear_nk.shape[0] < gNum_nk:
                # 真値より少ない場合は、1400年以降で合わせる、同じ数になるように
                pYear_nk = (np.abs(nk[nk<8000-sYear] - 8000) -sYear)[-gNum_nk:]
                 
                if pYear_nk.shape[0] < gNum_nk:
                     # 先頭から10000足す
                     pYear_nk = np.insert(pYear_nk, 0, np.tile(10000, gNum_nk-pYear_nk.shape[0]))
                 
            if pYear_tnk.shape[0] < gNum_tnk:
               
                pYear_tnk = (np.abs(tnk[tnk<8000-sYear] - 8000) -sYear)[-gNum_tnk:]
            
                if pYear_tnk.shape[0] < gNum_tnk:
            
                    pYear_tnk = np.insert(pYear_tnk, 0, np.tile(10000, gNum_tnk-pYear_tnk.shape[0]))
            
            if pYear_tk.shape[0] < gNum_tk:
              
                pYear_tk = (np.abs(tk[tk<8000-sYear] - 8000) -sYear)[-gNum_tk:]
          
                if pYear_tk.shape[0] < gNum_tk:
           
                    pYear_tk = np.insert(pYear_tk, 0, np.tile(10000, gNum_tk-pYear_tk.shape[0]))
            #pdb.set_trace()
            
            ndist_nk = gauss(gYear_nk_re,pYear_nk,mode=3)
            ndist_tnk = gauss(gYear_tnk_re,pYear_tnk,mode=3)
            ndist_tk = gauss(gYear_tk_re,pYear_tk,mode=3)
            
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
        
        sInd = np.argmin(yearErrors)
    # ----
    
    # 1 対 1 (normal) ----
    if mode == 4:
        
        flag = False
        # Slide each one year 
        for sYear in np.arange(8000-aYear): 
            eYear = sYear + aYear
                
            # 閾値以上の予測した地震年数
            pYear_nk = np.where(pred[sYear:eYear,0] > th)[0][:gNum_nk]
            pYear_tnk = np.where(pred[sYear:eYear,1] > th)[0][:gNum_tnk]
            pYear_tk = np.where(pred[sYear:eYear,2] > th)[0][:gNum_tk]
        
            # pred < gt
            if pYear_nk.shape[0] < gNum_nk:
                # 真値より少ない場合は、1400年以降で合わせる、同じ数になるように
                pYear_nk = np.where(pred[sYear:,0] > th)[0][:gNum_nk]
                 
                if pYear_nk.shape[0] < gNum_nk:
                     
                     pYear_nk = np.hstack([pYear_nk, np.tile(10000, gNum_nk-pYear_nk.shape[0])])
                 
            if pYear_tnk.shape[0] < gNum_tnk:
               
                pYear_tnk = np.where(pred[sYear:,1] > th)[0][:gNum_tnk]
            
                if pYear_tnk.shape[0] < gNum_tnk:
            
                    pYear_tnk = np.hstack([pYear_tnk, np.tile(10000, gNum_tnk-pYear_tnk.shape[0])])
            
            if pYear_tk.shape[0] < gNum_tk:
               
                pYear_tk = np.where(pred[sYear:,2] > th)[0][:gNum_tk]
          
                if pYear_tk.shape[0] < gNum_tk:
           
                    pYear_tk = np.hstack([pYear_tk, np.tile(10000, gNum_tk-pYear_tk.shape[0])])
        
            # [9,]
            ndist_nk = gauss(gYear_nk,pYear_nk,mode=3)
            ndist_tnk = gauss(gYear_tnk,pYear_tnk,mode=3)
            ndist_tk = gauss(gYear_tk,pYear_tk,mode=3)
            
            # 真値に合わせて二乗誤差
            yearError_nk = np.sum(ndist_nk)
            yearError_tnk = np.sum(ndist_tnk)
            yearError_tk = np.sum(ndist_tk)
            
            yearError = yearError_nk + yearError_tnk + yearError_tk
            #pdb.set_trace()
            if not flag:
                yearErrors = yearError
                flag = True
            else:
                yearErrors = np.hstack([yearErrors,yearError])  
         
        sInd = np.argmin(yearErrors)    
    # ----
           
    # 閾値 & 二乗誤差 順番 ----
    elif mode == 3:
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
                try:
                    pYear_nk = np.hstack([pYear_nk, np.tile(pYear_nk[-1], gNum_nk-pYear_nk.shape[0])])
                except IndexError:
                    pYear_nk = np.hstack([pYear_nk, np.tile(10000,gNum_nk)])
            
            if pYear_tnk.shape[0] < gNum_tnk:
                try:
                    pYear_tnk = np.hstack([pYear_tnk, np.tile(pYear_tnk[-1], gNum_tnk-pYear_tnk.shape[0])])
                except IndexError:
                    pYear_tnk = np.hstack([pYear_tnk, np.tile(10000,gNum_tnk)])
            
            if pYear_tk.shape[0] < gNum_tk:
                try:
                    pYear_tk = np.hstack([pYear_tk, np.tile(pYear_tk[-1], gNum_tk-pYear_tk.shape[0])])
                except IndexError:
                    pYear_tk = np.hstack([pYear_tk, np.tile(10000,gNum_tk)])
                
            # [9,]
            ndist_nk = gauss(gYear_nk,pYear_nk,mode=3)
            ndist_tnk = gauss(gYear_tnk,pYear_tnk,mode=3)
            ndist_tk = gauss(gYear_tk,pYear_tk,mode=3)
            
            # 真値に合わせて二乗誤差
            yearError_nk = np.sum(ndist_nk)
            yearError_tnk = np.sum(ndist_tnk)
            yearError_tk = np.sum(ndist_tk)
            
            yearError = yearError_nk + yearError_tnk + yearError_tk
            #pdb.set_trace()
            if not flag:
                yearErrors = yearError
                flag = True
            else:
                yearErrors = np.hstack([yearErrors,yearError])
               
        # 最小誤差開始修了年数(1400年)取得
        sInd = np.argmin(yearErrors)
        #pdb.set_trace()
        # ----
        if isPlot:
            # if slip velocity plot
            sns.set_style("dark")
      
            nk = pred[sInd:sInd+1400,0]
            tnk = pred[sInd:sInd+1400,1]
            tk = pred[sInd:sInd+1400,2]
            
            predVnk,predVtnk,predVtk = np.zeros(1400),np.zeros(1400),np.zeros(1400)
            
            # V > 1
            predVnk[np.where(nk>1)[0].tolist()] = nk[nk>1]
            predVtnk[np.where(tnk>1)[0].tolist()] = tnk[tnk>1]
            predVtk[np.where(tk>1)[0].tolist()] = tk[tk>1]
            
            colors = ["coral","skyblue","coral","skyblue","coral","skyblue"]
            
            # scalling var.
            gtV = np.zeros([1400,3])
            gtV[gYear_nk.tolist(),ntI] = 5
            gtV[gYear_tnk.tolist(),tntI] = 5
            gtV[gYear_tk,ttI] = 5
           
            plot_data = [gtV[:,ntI],predVnk,gtV[:,tntI],predVtnk,gtV[:,ttI],predVtk]
            
            fig = plt.figure()
            fig, axes = plt.subplots(nrows=6,sharex="col")
            for row,(color,data) in enumerate(zip(colors,plot_data)):
                axes[row].plot(np.arange(1400), data, color=color)
            
            plt.suptitle(f'{int(np.min(yearErrors))}')
            plt.savefig(os.path.join('images','bayes',f'{label}.png'))
            plt.close()
        # ----
        
            
    # 最小誤差確率　
    maxSim = yearErrors[sInd]
    print(maxSim)
    
    return maxSim
        
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def gauss(gtY,predY,sigma=100,mode=3):
    
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
        
    # mse or mae
    elif mode == 3:
        
        #gauss = (gtY - predY)**2
        gauss = np.abs(gtY - predY)
    
    return gauss    
# -----------------------------------------------------------------------------
    
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    
    # ---- command ---- #
    mode = int(sys.argv[1]) # 3: 全探索 順番 var, 4:
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
    if isplotbestPath:    
        # ---------------------------------------------------------------------
        logfilePath = os.path.join(dsdirPath,"bestMSE","*txt")
        
        logfiles = glob.glob(logfilePath)
        # ---------------------------------------------------------------------
        flag = False
        for logfile in logfiles:
            
            print(logfile)
            
            V,B = loadABLV(logfile)
            yV = convV2YearlyData(V)
            maxSim, pred = MinErrorNankai(gt,yV)
            # eq. of pred
            predeq_nk = np.where(pred[:,ntI]>th)[0]
            predeq_tnk = np.where(pred[:,tntI]>th)[0]
            predeq_tk = np.where(pred[:,ttI]>th)[0]
            
            # plot gt & pred rireki
            myPlot.Rireki(gt,pred,title=f"{predeq_nk}\n{predeq_tnk}\n{predeq_tk}",label=f"{maxSim}_{np.round(B[ntI],6)}_{np.round(B[tntI],6)}_{np.round(B[ttI],6)}")
            
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
        #pdb.set_trace()
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
        if mode == 3 or mode == 4 or mode == 5 or mode == 6 or mode == 7:
            
            cnt = 0
            
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
                maxSim = MinErrorNankai(gt,yV,mode=mode)
                
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