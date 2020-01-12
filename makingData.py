# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 16:40:21 2019

@author: yu
"""

import os
import glob
import pickle
import pdb
import time
import shutil

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pylab as plt

import numpy as np
from scipy import stats
from natsort import natsorted


# ---- params ---- #

# eq. year in logs     
yrInd = 1
yInd = 0
vInds = [2,3,4,5,6,7,8,9]
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
# -------------- #

#---------------------------------------------------------------------

#　数値計算できないファイルをコピー
def Empty(logFullPath):
     
    # copy先のディレクトリ
     newdirPath = os.path.join(copyPath,os.path.dirname(logFullPath).split("\\")[1])
     # copy
     shutil.copy2(logFullPath,newdirPath)
     print("Death!")
     #pdb.set_trace()
     pass
#---------------------------------------------------------------------

def Negative(V,logFullPath,cnt,isWindows=False):
    """
    マイナスのすべり速度が出た場合は計算できない
    つぎのディレクトリに送る
    """
    
    # 同化開始のV
    startV = V[0,:]
    if any(startV < 0):
        # copy先のディレクトリ
        newdirPath = os.path.join(copyPath,os.path.dirname(logFullPath).split("\\")[1])
        # copy
        shutil.copy2(logFullPath,newdirPath)
            
        print("Negative Death!")
#---------------------------------------------------------------------

#データの読み込み
def loadABLV(dirPath,logPath,fName,isLAST=False):
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
        #A[i-1] = tmp[0]
        B[i-1] = tmp[1]
        #L[i-1] = tmp[4]

    # Vの開始行取得
    isRTOL = [True if data[i].count('value of RTOL')==1 else False for i in np.arange(len(data))]
    vInd = np.where(isRTOL)[0][0]+1
    
    if isLAST:
        # LastEnsambleの時
        return B,vInd
#---------------------------------------------------------------------
                        # Ensambleの時 #
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
#---------------------------------------------------------------------

def convV2YearlyData(U,th,V,nYear,cell=0,cnt=0):
    """
    Args:
        nYear: number of year (by simulation) -> 10000
        cell: only one cell (※)
    """
    # シミュレータのU,th,V 発生時データ - > 年数データ 用
    yU, yth, yV = np.zeros([nYear,simlateCell]),np.zeros([nYear,simlateCell]),np.zeros([nYear,simlateCell])
    #pdb.set_trace()
    try:
        # 初め手観測した年
        sYear = np.floor(V[0,yrInd])
        for year in np.arange(sYear,nYear):
            # 観測値がある場合
            if np.sum(np.floor(U[:,yrInd])==year):
                # 観測値をそのまま代入
                # self.U[地震,Xyr+nCell]
                yU[int(year)] = np.reshape(U[np.floor(U[:,yrInd])==year,vInds[0]:],[-1,])
                yth[int(year)] = np.reshape(th[np.floor(th.T[yrInd,:])==year,vInds[0]:],[-1,])
                yV[int(year)] = np.reshape(V[np.floor(V.T[yrInd,:])==year,vInds[0]:],[-1,])
        
            # 観測値がない場合
            else:
                
                # U(累積変位),th(状態変数):地震時t-1の観測値代入,V(速度):0.0
                yU[int(year)] = yU[int(year)-1,:] # shape=[100000,8]
                yth[int(year)] = yth[int(year)-1,:] # shape=[100000,8]
                yV[int(year)] = float(0) # shape=[100000,8]
    
    except IndexError:
        print("IndexError: out of range")
    
    # シミュレーションが安定した2000年以降を用いる, 地震発生年 (どこかのセルで発生した場合)
    # 0年目の時のために、yUexを出力
    #pdb.set_trace()
    if cnt == 0:
        return yU[stateYear:,:], yU[np.where(yU[:stateYear,cell]<yU[stateYear,cell])[0][-1],:], yth[stateYear:,:], yV[stateYear:,:], U[:,yrInd], yU[np.where(yU[:,yrInd]-yU[stateYear,yrInd]<0)[0][-1],:]
    # 一番始め以外は、2000年以降&yUexが不明(最小年数に合わせる必要あり)のため
    elif cnt > 0:    
        return yU, yth, yV, U[:,yrInd]

    #---------------------------------------------------------------------
    # 累積変位データ作成(真)を作成するときはコメントアウトをはずす
    # シミュレーションデータのdeltaTとdeltaUの関係を求める
    # 指定した再来間隔のdeltaUを求める、TP&SP(未実装)
    #---------------------------------------------------------------------
    """
    #deltaUを求める再来間隔
    years = [90,100,150,200,260,300]
    uInds = 2
    #for dy in np.arange(sYear,eYear):
    for dy in years:
        jt1 = (np.arange(int(sYear),int(eYear),dy)).tolist()
        jt2 = (np.arange(int(sYear),int(eYear),dy)).tolist()[1:]
        flag = False
        for t1,t2 in zip(jt1,jt2):
            if np.any(np.floor(self.U[:,1]) == t1 ) and np.any(np.floor(self.U[:,1]) == t2 ):
                du,dt = self.U[np.where(self.U[:,self.yrInd].astype(int)==t2)]-self.U[np.where(self.U[:,self.yrInd].astype(int)==t1)],t2-t1
                if not flag:
                    deltaU,deltaT = du,dt
                    flag = True
                else:
                    deltaU,deltaT = np.vstack([deltaU,du]),np.vstack([deltaT,dt])

        meanU = np.mean(deltaU[:,uInds:],axis=0)
        # parFile番号格納
        data = np.c_[meanU]
        np.savetxt("deltaTANDdeltaU\\RegNankai{}{}.csv".format(fI,dy),data,delimiter=",",fmt="%.2f")
    """
#---------------------------------------------------------------------
def MinErrorNankai(gt,yU,yUex,yth,yV,cell=0,mimMode=0):
    """
    シミュレーションされたデータの真値との誤差が最小の1400年間を抽出
    Args:
        gt: ground truth V
        yU,yth,yV: pred UV/theta/V, shape=[8000,8]
        minMode: 0. after 2000 year, 1. degree of similatery
    """
    #pdb.set_trace()
    # ---- 1 ---- #
    if mimMode == 0:
        return yU[:1400,:], yUex, yth[:1400,:], yV[:1400,:]
    # ----------- #
    
    # ---- 2 ---- #
    elif mimMode == 1:        
        if cell == 2 or cell == 4 or cell == 5:
            pred = yV[:,cell]
            #pdb.set_trace()
            # ----
            # 真値の地震年数
            gYear = np.where(gt[:,0] > slip)[0]
            # ----

            flag = False
            # Slide each one year 
            for sYear in np.arange(8000-aYear): 
                # 予測した地震の年数 + 1400
                eYear = sYear + aYear

                # 予測した地震年数 
                pYear = np.where(pred[sYear:eYear] > slip)[0]

                # gaussian distance for year of gt - year of pred (gYears.shape, pred.shape)
                ndist = gauss(gYear,pYear.T)

                # 予測誤差の合計, 回数で割ると当てずっぽうが小さくなる
                yearError = sum(ndist.max(1)/pYear.shape[0])

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
        
        return yU[sInd:eInd,:], yUex, yth[sInd:eInd,:], yV[sInd:eInd,:]
                #--------------------------


#--------------------------
def gauss(gtY,predY,sigma=100):

    # predict matrix for matching times of gt eq.
    predYs = predY.repeat(gtY.shape[0],0).reshape(-1,gtY.shape[0])
    # gt var.
    gtYs = gtY.repeat(predY.shape[0],0).reshape(-1,predY.shape[0])

    gauss = np.exp(-(gtYs - predYs.T)**2/(2*sigma**2))

    return gauss
#--------------------------

    
    
    
    
    
    
