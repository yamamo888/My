# -*- coding: utf-8 -*-

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
import seaborn as sns
from natsort import natsorted

import PlotPF as myPlot

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

def Negative(V,logFullPath,cnt):
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
#def loadABLV(dirPath,logPath,fName):
def loadABLV(logPath,fName):
    
    """
    1.初期アンサンブル作成(iS==0のとき)
    安定状態(2000年以降)を初期状態とする
    シミュレーション(M)を用いて時間発展(Xt-1->Xt)
    """
    # logファイル読み込み
    #logFullPath = os.path.join(dirPath,logPath,fName)
    logFullPath = os.path.join(logPath,fName)
    
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
                # only U or V#
#---------------------------------------------------------------------    
    
    flag = False
    for uI in np.arange(vInd,len(data)): # 2行飛ばしで読み込み
        
        tmpU = np.array(data[uI].strip().split(",")[yInd:]).astype(np.float32)
        
        if not flag:
            U = tmpU
            flag = True
        else:
            U = np.vstack([U,tmpU])
    
    return U,B
    
    """
#---------------------------------------------------------------------
                        # PFの時 #
#---------------------------------------------------------------------
    # ※ Ensambleと違ってUが必要ない
    # th(0番目),V(1番目)と発生年数
    thI,vI,uthvI = 0,1,2
    flag = False
    for uI in np.arange(vInd,len(data),uthvI): # 2行飛ばしで読み込み
        # th V
        tmpth = np.array(data[uI+thI].strip().split(",")[yInd:]).astype(np.float32)
        tmpV = np.array(data[uI+vI].strip().split(",")[yInd:]).astype(np.float32)
        if not flag:
            th,V = tmpth,tmpV
            flag = True
        else:
            th,V = np.vstack([th,tmpth]),np.vstack([V,tmpV])
    
    return th,V,B
    """
#---------------------------------------------------------------------
                        # Ensambleの時 #
#---------------------------------------------------------------------
    """ 
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
    """
#---------------------------------------------------------------------
# ※ Ensambleの時は上の方
#def convV2YearlyData(U,th,V,nYear,cell=0,cnt=0):
# ※ PF
#def convV2YearlyData(th,V,nYear,cnt=0):
def convV2YearlyData(U,B,nYear,cnt=0):

    """
    Args:
        nYear: number of year (by simulation) -> 10000
        cell: only one cell for ensamble kalman
    """

    
#---------------------------------------------------------------------
                # only U or interval of eq. #
#---------------------------------------------------------------------    
    yU = np.zeros([nYear,8])
    
    # 初め手観測した年
    sYear = np.floor(U[0,yrInd])
    for year in np.arange(sYear,nYear):
        # 観測値がある場合
        if np.sum(np.floor(U[:,yrInd])==year):
            # 観測値をそのまま代入
            yU[int(year)] = np.reshape(U[np.floor(U.T[yrInd,:])==year,vInds[0]:],[-1,])
    
        # 観測値がない場合
        else:
            # th(状態変数):地震時t-1の観測値代入,V(速度):0.0
            yU[int(year)] = yU[int(year)-1,:] # shape=[100000,8]
            
    yU = np.concatenate((yU[stateYear:,2,np.newaxis],yU[stateYear:,4,np.newaxis],yU[stateYear:,5,np.newaxis]),1)
    yU = yU[1:] - yU[:-1]
    # [3,]
    B = np.concatenate((B[2,np.newaxis],B[4,np.newaxis],B[5,np.newaxis]),0)
    
    # eq. year
    nkY = np.where(yU[:,0]>0)[0]
    tnkY = np.where(yU[:,1]>0)[0]
    tkY = np.where(yU[:,2]>0)[0]
    
    # interval for each cell, [?]
    intervalnkY = nkY[1:] - nkY[:-1]
    intervaltnkY = tnkY[1:] - tnkY[:-1]
    intervaltkY = tkY[1:] - tkY[:-1]
    #return yU
    return intervalnkY,intervaltnkY,intervaltkY,B
    
    """
    sns.set_style("dark")
    fig, figInds = plt.subplots(nrows=3, sharex=True)
    for figInd in np.arange(len(figInds)):
        figInds[figInd].plot(np.arange(8000), yU[:,figInd])
    
    plt.suptitle(f"nkb:{B[2]} tnkb:{B[4]} tkb:{B[5]}")
    
    plt.savefig(os.path.join("images","Umodel",f"U_{B[2]}_{B[4]}_{B[5]}.png"))
    plt.close()
    """
#---------------------------------------------------------------------
                # PFの時 #
#---------------------------------------------------------------------    
    """
    # シミュレータのU,th,V 発生時データ - > 年数データ 用
    yth, yV = np.zeros([nYear,simlateCell]),np.zeros([nYear,simlateCell])
    
    # 初め手観測した年
    sYear = np.floor(V[0,yrInd])
    for year in np.arange(sYear,nYear):
        # 観測値がある場合
        if np.sum(np.floor(V[:,yrInd])==year):
            # 観測値をそのまま代入
            yth[int(year)] = np.reshape(th[np.floor(th.T[yrInd,:])==year,vInds[0]:],[-1,])
            yV[int(year)] = np.reshape(V[np.floor(V.T[yrInd,:])==year,vInds[0]:],[-1,])
    
        # 観測値がない場合
        else:
            
            # th(状態変数):地震時t-1の観測値代入,V(速度):0.0
            yth[int(year)] = yth[int(year)-1,:] # shape=[100000,8]
            yV[int(year)] = float(0) # shape=[100000,8]
    #pdb.set_trace()
    # シミュレーションが安定した2000年以降を用いる
    if cnt == 0:
        # [year.shape,]
        yYear = np.reshape(np.floor(V[np.where(V[:,yrInd]>stateYear),yrInd]).astype(int),[-1])
        
        #　発生年数の確認 -------------------------------------------------------
        #myPlot.NumberLine(yYear[:5]) # どれぐらい確認したいか指定
        # ---------------------------------------------------------------------
        
        return yth[stateYear:,:], yV[stateYear:,:], yYear
   
    # 途中から始める仕様になってるので、
    elif cnt > 0:
        return yth, yV, np.reshape(np.floor(V[:,yrInd]).astype(int),[-1])
    """
#---------------------------------------------------------------------
                    # Ensambleの時 #
#---------------------------------------------------------------------
    """ 
    # シミュレータのU,th,V 発生時データ - > 年数データ 用
    yU, yth, yV = np.zeros([nYear,simlateCell]),np.zeros([nYear,simlateCell]),np.zeros([nYear,simlateCell])
    
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
    if cnt == 0:
         # 2000年より前に地震が起きていなければ
        if all(yU[:stateYear,cell]<yU[stateYear,cell]) == False:
            return yU[stateYear:,:], yU[stateYear-1], yth[stateYear:,:], yV[stateYear:,:], U[:,yrInd]
        else:
            return yU[stateYear:,:], yU[np.where(yU[:stateYear,cell]<yU[stateYear,cell])[0][-1],:], yth[stateYear:,:], yV[stateYear:,:], U[:,yrInd]
    # 一番始め以外は、2000年以降 & yUexは更新されるyUtを使い続ける
    elif cnt > 0:    
        return yU, yth, yV, U[:,yrInd]
    

    #---------------------------------------------------------------------
    # 累積変位データ作成(真)を作成するときはコメントアウトをはずす
    # シミュレーションデータのdeltaTとdeltaUの関係を求める
    # 指定した再来間隔のdeltaUを求める、TP&SP(未実装)
    #---------------------------------------------------------------------
    
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
#def MinErrorNankai(gt,yU,yUex,yth,yV,cell=0,mimMode=0):    
def MinErrorNankai(gt,yth,yV,pY,cell=0,gtcell=0):
    """
    シミュレーションされたデータの真値との誤差が最小の1400年間を抽出
    Args:
        gt: ground truth V
        yth,yV: pred UV/theta/V, shape=[8000,8]
        minMode: 0. after 2000 year, 1. degree of similatery
        pY: eq. year
    """
    
#---------------------------------------------------------------------
                # PFの時 #
#---------------------------------------------------------------------        
    if cell == 2 or cell == 4 or cell == 5:
        
        # ----
        # 真値の地震年数
        gYear = np.where(gt[:,gtcell] > slip)[0]
        # ----
        
        # ----
        # シミュレーションの値を格納する
        pred = np.zeros((8000,cell))
        # 地震が起きた年数だけに値を代入
        pred[pY,:] = 10
        # ----
        
        flag = False
        # Slide each one year 
        for sYear in np.arange(8000-aYear): 
            # 予測した地震の年数 + 1400
            eYear = sYear + aYear

            # 予測した地震年数 
            pYear = np.where(pred[sYear:eYear] > slip)[0]
            
            # 類似度 -----------------------------------------------------------
            # gaussian distance for year of gt - year of pred (gYears.shape, pred.shape)
            ndist = gauss(gYear,pYear.T)
            # 予測誤差の合計, 回数で割ると当てずっぽうが小さくなる
            yearError = sum(ndist.max(1)/pYear.shape[0])
            # -----------------------------------------------------------------

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
    #pdb.set_trace()
    return yth[sInd:eInd,:], yV[sInd:eInd,:], pY[(pY>sInd)&(pY<eInd)], sInd, np.round(maxSim,6)

#---------------------------------------------------------------------
                    # Ensambleの時 #
#---------------------------------------------------------------------
    """    
    # ---- 1 ---- #
    if mimMode == 0:
        return yU[:1400,:], yUex, yth[:1400,:], yV[:1400,:]
    # ----------- #
    
    # ---- 2 ---- #
    elif mimMode == 1:        
        if cell == 2 or cell == 4 or cell == 5:
            pred = yV[:,cell]
            
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
        
        if sInd != 0 and any(yU[:sInd,cell]<yU[sInd,cell]):
            yUex = yU[np.where(yU[:sInd,cell]<yU[sInd,cell])[0][-1],:]
        # 開始インデックスが 0 or sIndの前に滑ってる場合、そのままyUex出力
        return yU[sInd:eInd,:], yUex, yth[sInd:eInd,:], yV[sInd:eInd,:]
        """
#--------------------------
def gauss(gtY,predY,sigma=100):

    # predict matrix for matching times of gt eq.
    predYs = predY.repeat(gtY.shape[0],0).reshape(-1,gtY.shape[0])
    # gt var.
    gtYs = gtY.repeat(predY.shape[0],0).reshape(-1,predY.shape[0])

    gauss = np.exp(-(gtYs - predYs.T)**2/(2*sigma**2))

    return gauss
#--------------------------
