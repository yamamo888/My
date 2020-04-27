# -*- coding: utf-8 -*-

import os
import glob
import pickle
import pdb
import time
import shutil
import csv

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pylab as plt

import numpy as np
from scipy import stats
from natsort import natsorted

import PlotPF as myPlot

# ---- params ---- #
# eq. year in logs     
yrInd = 1
yInd = 0
vInds = [2,3,4,5,6,7,8,9]
ntI,tntI,ttI = 0,1,2
simlateCell = 8
# いいんかな？
slip = 1
# theresholds of eq. for MSError
th = 1

# 安定した年
stateYear = 2000
# assimilation period
aYear = 1400
# ---------------- #

# ファイル存在確認 ----------------------------------------------------------------
def isDirectory(fpath):
    #pdb.set_trace()
    # 'path' exist -> True
    isdir = os.path.exists(fpath)
    # -> False
    if not isdir:
        os.makedirs(fpath)
#------------------------------------------------------------------------------

# 発生年数誤差出力 (相互誤差) --------------------------------------------------------------      
def eachMAEyear(gt,pred):
    try:
        dists = []
        
        gt_tk = np.array([s for s in gt[ttI] if s != 0])
        
        gNum_nk = gt[ntI].shape[0]
        gNum_tnk = gt[tntI].shape[0]
        gNum_tk = gt[ttI].shape[0]
      
        # del 0 year
        pred_nk = np.array([s for s in pred[ntI].tolist() if s != 0])[:gNum_nk]
        pred_tnk = np.array([s for s in pred[tntI].tolist() if s != 0])[:gNum_tnk]
        pred_tk = np.array([s for s in pred[ttI].tolist() if s != 0])[:gNum_tk]
        
        if pred_nk.shape[0] < gNum_nk:
            pred_nk = np.hstack([pred_nk, np.tile(pred_nk[-1], gNum_nk-pred_nk.shape[0])])
        if pred_tnk.shape[0] < gNum_tnk:
            pred_tnk = np.hstack([pred_tnk, np.tile(pred_tnk[-1], gNum_tnk-pred_tnk.shape[0])])
        if pred_tk.shape[0] < gNum_tk:
            pred_tk = np.hstack([pred_tk, np.tile(pred_tk[-1], gNum_tk-pred_tk.shape[0])])
       
        for gy,py in zip([gt[ntI],gt[tntI],gt_tk],[pred_nk,pred_tnk,pred_tk]):
            # predict year & gt year
            pys = py.repeat(gy.shape[0],0).reshape(-1,gy.shape[0])
            gys = gy.repeat(py.shape[0],0).reshape(-1,py.shape[0])
            # abs error
            dist = np.min(np.abs(gys-pys.T),1)
            dists = np.append(dists,np.sum(dist))
        
        return int(sum(dists))
    
    except ValueError:
        return -1
#------------------------------------------------------------------------------

# 発生年数誤差出力 --------------------------------------------------------------      
def MAEyear(gt,pred):
    """
    Args
        gt: ground truth eq.year (tk in 0)
        pred: predicted eq.year
    """
    try:
        dists = []
        # del 0 year
        pred_nk = np.array([s for s in pred[ntI].tolist() if s != 0])
        pred_tnk = np.array([s for s in pred[tntI].tolist() if s != 0])
        pred_tk = np.array([s for s in pred[ttI].tolist() if s != 0])
        
        gt_tk = np.array([s for s in gt[ttI] if s != 0])
        
        for gy,py in zip([gt[ntI],gt[tntI],gt_tk],[pred_nk,pred_tnk,pred_tk]):
            # predict year & gt year
            pys = py.repeat(gy.shape[0],0).reshape(-1,gy.shape[0])
            gys = gy.repeat(py.shape[0],0).reshape(-1,py.shape[0])
            # abs error
            dist = np.min(np.abs(gys-pys.T),1)
            dists = np.append(dists,np.sum(dist))
            
        return int(sum(dists))
    except ValueError:
        return -1
#------------------------------------------------------------------------------

# データの読み込み ----------------------------------------------------------------
def loadABLV(dirPath,logPath,fName):
    """
    1.初期アンサンブル作成(iS==0のとき)
    安定状態(2000年以降)を初期状態とする
    シミュレーション(M)を用いて時間発展(Xt-1->Xt)
    """
    # logファイル読み込み
    logFullPath = os.path.join(dirPath,logPath,fName)
    data = open(logFullPath).readlines()
    
    # A, B, Lの取得
    #A = np.zeros(nCell)
    B = np.zeros(simlateCell)
    #L = np.zeros(nCell)
    for i in np.arange(1,simlateCell+1):
        # cell番号に合わせてdata読み取り
        tmp = np.array(data[i].strip().split(",")).astype(np.float32)
        B[i-1] = tmp[1]
       
    # Vの開始行取得
    isRTOL = [True if data[i].count('value of RTOL')==1 else False for i in np.arange(len(data))]
    vInd = np.where(isRTOL)[0][0]+1
    
#---------------------------------------------------------------------
                    # PF Ensambleの時 #
#---------------------------------------------------------------------

    # U,th,Vの値と発生年数t取得（vInd行から最終行まで）
    uI,thI,vI,uthvI = 0,1,2,3
    flag = False
    for uI in np.arange(vInd,len(data),uthvI):
        # U->th->Vの順
        tmpU = np.array(data[uI].strip().split(",")[yInd:]).astype(np.float32)
        tmpth = np.array(data[uI+thI].strip().split(",")[yInd:]).astype(np.float32)
        tmpV = np.array(data[uI+vI].strip().split(",")[yInd:]).astype(np.float32)
        if not flag:
            U,th,V = tmpU,tmpth,tmpV
            flag = True
        else:
            U,th,V = np.vstack([U,tmpU]),np.vstack([th,tmpth]),np.vstack([V,tmpV])

    return U,th,V,B
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def convV2YearlyData(U,th,V,nYear=10000,cnt=0,stYear=0,isLast=False):
    """
    Args:
        nYear: number of year (by simulation) -> 10000
        cell: only one cell for ensamble kalman
    """
    
    # シミュレータのU,th,V 発生時データ -> 年数データ 用
    yU, yth, yV = np.zeros([nYear,simlateCell]),np.zeros([nYear,simlateCell]),np.zeros([nYear,simlateCell])
    
    # 初めて観測した年
    sYear = np.floor(V[0,yrInd])
    for year in np.arange(sYear,nYear):
        # 観測値がある場合
        if np.sum(np.floor(V[:,yrInd])==year):
            # 観測値をそのまま代入
            yU[int(year)] = np.reshape(U[np.floor(U.T[yrInd,:])==year,vInds[0]:],[-1,])
            yth[int(year)] = np.reshape(th[np.floor(th.T[yrInd,:])==year,vInds[0]:],[-1,])
            yV[int(year)] = np.reshape(V[np.floor(V.T[yrInd,:])==year,vInds[0]:],[-1,])
    
        # 観測値がない場合
        else:
            # th(状態変数):地震時t-1の観測値代入,V(速度):0.0
            yU[int(year)] = yU[int(year)-1,:] 
            yth[int(year)] = yth[int(year)-1,:] 
            yV[int(year)] = float(0) 
    
    deltaU = np.vstack([yU[0,:], yU[1:] - yU[:-1]])
    #pdb.set_trace()
    # シミュレーションが安定した2000年以降を用いる
    if cnt == 0:
        # ※ Uの発生年数
        nkYear = np.where(deltaU[stateYear:,2]>slip)[0]
        tnkYear = np.where(deltaU[stateYear:,4]>slip)[0]
        tkYear = np.where(deltaU[stateYear:,5]>slip)[0]
        yYear = [nkYear,tnkYear,tkYear]
        
        return yU[stateYear:,:], yth[stateYear:,:], yV[stateYear:,:], yYear
    
    # 途中から始める仕様になってるので、
    elif cnt > 0:
        
        nkYear = np.where(deltaU[:,2]>slip)[0]
        tnkYear = np.where(deltaU[:,4]>slip)[0]
        tkYear = np.where(deltaU[:,5]>slip)[0]
        yYear = [nkYear,tnkYear,tkYear]
        
        if isLast:
            # rireki plot    
            return deltaU[stYear+stateYear:stYear+stateYear+aYear,:], yth[stYear+stateYear:stYear+stateYear+aYear,:], yV[stYear+stateYear:stYear+stateYear+aYear,:], yYear
        else:
            return yU[stYear+stateYear:stYear+stateYear+aYear,:], yth[stYear+stateYear:stYear+stateYear+aYear,:], yV[stYear+stateYear:stYear+stateYear+aYear,:], yYear
#------------------------------------------------------------------------------        

#------------------------------------------------------------------------------
def GaussErrorNankai(gt,yU,yth,yV,pY,nCell=0):
    """
    シミュレーションされたデータの真値との誤差が最小の1400年間を抽出
    Args:
        gt: ground truth V
        yth,yV: pred UV/theta/V, shape=[8000,8]
        pY: eq. year
        isPlot: you want to plot rireki -> isPlot=True
    """
    # シミュレーションの値を格納する
    pred = np.zeros((8000,nCell))
    
    # ----
    # 真値の地震年数
    gYear_nk = np.where(gt[:,ntI] > slip)[0]
    gYear_tnk = np.where(gt[:,tntI] > slip)[0]
    gYear_tk = np.where(gt[:,ttI] > slip)[0]
    # ----
    
    pred[pY[ntI],ntI] = 30
    pred[pY[tntI],tntI] = 30
    pred[pY[ttI],ttI] = 30
    
    flag = False
    # Slide each one year 
    for sYear in np.arange(8000-aYear): 
        # 予測した地震の年数 + 1400
        eYear = sYear + aYear

        # 予測した地震年数 only one-cell
        pYear_nk = np.where(pred[sYear:eYear,ntI] > slip)[0]
        pYear_tnk = np.where(pred[sYear:eYear,tntI] > slip)[0]
        pYear_tk = np.where(pred[sYear:eYear,ttI] > slip)[0]
        
        # gaussian distance for year of gt - year of pred (gYears.shape, pred.shape)
        # for each cell
        ndist_nk = calcDist(gYear_nk,pYear_nk.T)
        ndist_tnk = calcDist(gYear_tnk,pYear_tnk.T)
        ndist_tk = calcDist(gYear_tk,pYear_tk.T)

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
    
    print(">>>>>>>>\n")                
    print(f"最大類似度:{np.round(maxSim,6)}\n")
    print(">>>>>>>>\n")

    nkYear = pY[ntI][(pY[ntI]>sInd)&(pY[ntI]<eInd)]-sInd
    tnkYear = pY[tntI][(pY[tntI]>sInd)&(pY[tntI]<eInd)]-sInd
    tkYear = pY[ttI][(pY[ttI]>sInd)&(pY[ttI]<eInd)]-sInd
    predYear = [nkYear,tnkYear,tkYear]
                    
    return yU[sInd:eInd,:], yth[sInd:eInd,:], yV[sInd:eInd,:], predYear, np.round(maxSim,6), sInd
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def eachMSErrorNankai(gt,yU,yth,yV,pY,nCell=0):
    #pdb.set_trace()
    # ground truth eq.
    gYear_nk = np.where(gt[:,0] > slip)[0]
    gYear_tnk = np.where(gt[:,1] > slip)[0]
    gYear_tk = np.where(gt[:,2] > slip)[0]
    
    # predicted eq.
    pred = np.zeros((8000,nCell))
    pred[pY[ntI],ntI] = 30
    pred[pY[tntI],tntI] = 30
    pred[pY[ttI],ttI] = 30
    
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
        ndist_nk = calcDist(gYear_nk,pYear_nk.T,error=1)
        ndist_tnk = calcDist(gYear_tnk,pYear_tnk.T,error=1)
        ndist_tk = calcDist(gYear_tk,pYear_tk.T,error=1)
        
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
    eInd = sInd + aYear
    
    nkYear = pY[ntI][(pY[ntI]>sInd)&(pY[ntI]<eInd)]-sInd
    tnkYear = pY[tntI][(pY[tntI]>sInd)&(pY[tntI]<eInd)]-sInd
    tkYear = pY[ttI][(pY[ttI]>sInd)&(pY[ttI]<eInd)]-sInd
    predYear = [nkYear,tnkYear,tkYear]
    
    maxSim = yearErrors[sInd]
    
    return yU[sInd:eInd,:], yth[sInd:eInd,:], yV[sInd:eInd,:], predYear, maxSim, sInd
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def MSErrorNankai(gt,yU,yth,yV,pY,nCell=0):
    #pdb.set_trace()
    # ground truth eq.
    gYear_nk = np.where(gt[:,0] > slip)[0]
    gYear_tnk = np.where(gt[:,1] > slip)[0]
    gYear_tk = np.where(gt[:,2] > slip)[0]
    
    # predicted eq.
    pred = np.zeros((8000,nCell))
    pred[pY[ntI],ntI] = 30
    pred[pY[tntI],tntI] = 30
    pred[pY[ttI],ttI] = 30
    
    flag = False
    # Slide each one year 
    for sYear in np.arange(8000-aYear): 
        eYear = sYear + aYear

        # 閾値以上の予測した地震年数
        pYear_nk = np.where(pred[sYear:eYear,ntI] > th)[0]
        pYear_tnk = np.where(pred[sYear:eYear,tntI] > th)[0]
        pYear_tk = np.where(pred[sYear:eYear,ttI] > th)[0]
        
        ndist_nk = calcDist(gYear_nk,pYear_nk.T,error=1)
        ndist_tnk = calcDist(gYear_tnk,pYear_tnk.T,error=1)
        ndist_tk = calcDist(gYear_tk,pYear_tk.T,error=1)
        
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
    eInd = sInd + aYear
    
    nkYear = pY[ntI][(pY[ntI]>sInd)&(pY[ntI]<eInd)]-sInd
    tnkYear = pY[tntI][(pY[tntI]>sInd)&(pY[tntI]<eInd)]-sInd
    tkYear = pY[ttI][(pY[ttI]>sInd)&(pY[ttI]<eInd)]-sInd
    predYear = [nkYear,tnkYear,tkYear]
    
    maxSim = yearErrors[sInd]
    
    return yU[sInd:eInd,:], yth[sInd:eInd,:], yV[sInd:eInd,:], predYear, maxSim, sInd
# -----------------------------------------------------------------------------
    
#------------------------------------------------------------------------------
def calcDist(gtY,predY,sigma=100,error=0):
    if error == 0: # gauss    
        # predict matrix for matching times of gt eq.
        predYs = predY.repeat(gtY.shape[0],0).reshape(-1,gtY.shape[0])
        # gt var.
        gtYs = gtY.repeat(predY.shape[0],0).reshape(-1,predY.shape[0])
    
        dist = np.exp(-(gtYs - predYs.T)**2/(2*sigma**2))

    elif error == 1:
        predYs = predY.repeat(gtY.shape[0],0).reshape(-1,gtY.shape[0])
        gtYs = gtY.repeat(predY.shape[0],0).reshape(-1,predY.shape[0])
    
        dist = np.abs(gtYs - predYs.T)
    
    elif error == 2:
        predYs = predY.repeat(gtY.shape[0],0).reshape(-1,gtY.shape[0])
        gtYs = gtY.repeat(predY.shape[0],0).reshape(-1,predY.shape[0])
    
        dist = np.abs(gtYs - predYs.T)
        
    return dist
#------------------------------------------------------------------------------
