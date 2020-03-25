# -*- coding: utf-8 -*-

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
slip = 0

# 安定した年
stateYear = 2000
# assimilation period
aYear = 1400
# ---------------- #

# ---- path ---- #
copyPath = "logscopy"
imgPath = "images"
savetxtPath = "savetxt"
# -------------- #


#データの読み込み
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
def convV2YearlyData(U,th,V,nYear,cnt=0,stYear=0):
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
        #pdb.set_trace()
        return yU[stYear+stateYear:stYear+stateYear+aYear,:], yth[stYear+stateYear:stYear+stateYear+aYear,:], yV[stYear+stateYear:stYear+stateYear+aYear,:], yYear
   
#---------------------------------------------------------------------

def MinErrorNankai(gt,yU,yth,yV,pY,nCell=0,label="none",isPlot=False):
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
    
    print(">>>>>>>>\n")                
    print(f"最大類似度:{np.round(maxSim,6)}\n")
    print(">>>>>>>>\n")

    nkYear = pY[ntI][(pY[ntI]>sInd)&(pY[ntI]<eInd)]-sInd
    tnkYear = pY[tntI][(pY[tntI]>sInd)&(pY[tntI]<eInd)]-sInd
    tkYear = pY[ttI][(pY[ttI]>sInd)&(pY[ttI]<eInd)]-sInd
    predYear = [nkYear,tnkYear,tkYear]
    
    if isPlot:
        # plot deltaU in pred & gt --------------------------------------------
        fig, figInds = plt.subplots(nrows=3, sharex=True)
        for figInd in np.arange(len(figInds)):
            figInds[figInd].plot(np.arange(1400), pred[sInd:eInd,figInd],color="skyblue")
            figInds[figInd].plot(np.arange(1400), gt[:,figInd],color="coral")
        
        #plt.suptitle(f"nk:{nkYear}\n{tnkYear}\n{tkYear}\n startindex:{sInd}",fontsize="8")
        plt.suptitle(f"start index:{sInd}")
        plt.savefig(os.path.join(imgPath,"deltaU",f"{np.round(maxSim,4)}_{label}.png"))
        plt.close()
        # ---------------------------------------------------------------------
        
        # save eq. ------------------------------------------------------------
        with open(os.path.join(savetxtPath,"eq",f"{np.round(maxSim,4)}_{label}.txt"),"w") as fp:
            writer = csv.writer(fp)
            writer.writerows(predYear)
        # ---------------------------------------------------------------------
                   
    return yU[sInd:eInd,:], yth[sInd:eInd,:], yV[sInd:eInd,:], predYear, np.round(maxSim,6), sInd

#--------------------------
def gauss(gtY,predY,sigma=100):

    # predict matrix for matching times of gt eq.
    predYs = predY.repeat(gtY.shape[0],0).reshape(-1,gtY.shape[0])
    # gt var.
    gtYs = gtY.repeat(predY.shape[0],0).reshape(-1,predY.shape[0])

    gauss = np.exp(-(gtYs - predYs.T)**2/(2*sigma**2))

    return gauss
#--------------------------