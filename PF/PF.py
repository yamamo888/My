# -*- coding: utf-8 -*-

import os
import sys
import concurrent.futures
import subprocess

import glob
import shutil
import pickle
import pdb
import time

import matplotlib.pylab as plt
import numpy as np

from natsort import natsorted

import makingData as myData
import PlotPF as myPlot


# -------------------------- command argument --------------------------- #
# gt & obs name of cell, 2 or 4 or 5 or 245
cell = int(sys.argv[1])
# ----------------------------------------------------------------------- #

# ----------------------------- Path ------------------------------------ #
# In first ensamble file & logs file
dirPath = "logs"
# In paramHM* file
paramPath = "parFile"
# gt V
featuresPath = "nankairirekifeature"
firstEnName = "first*"
fileName = "*.txt"
K8_AV2File = "K8_AV2.txt"
paramCSV = "ParamFilePF.csv"
batFile = "PyToCPF.bat"
# ----------------------------------------------------------------------- #
        
# --------------------------- parameter --------------------------------- #

# 南海トラフ巨大地震履歴期間
gt_Year = 1400
# シミュレーションの安定した年
state_Year = 2000
# シミュレータの年数
nYear = 10000

# only one cell ---------------------------  
# select gt & obs cell, nankai(2), tonankai(4), tokai(5)
if cell == 2 or cell == 4 or cell == 5:
    # number of cell
    nCell = 1
    # gt number of cell
    gt_nCell = 1

# gt cell index
if cell == 2:
    gtcell = 0
elif cell == 4:
    gtcell = 1
elif cell == 5:
    gtcell = 2
    
# 3 cell ----------------------------------
elif cell == 245:
    nCell = 3
    # indec of each cell (gt)
    ntI,tntI,ttI = 0,1,2
    # index of each cell (simulation var)
    nI,tnI,tI = 2,4,5

# number of all param Th,V,b
nParam = 3
# slip velocity?
slip = 0
# reading file start & end line
Sfl = 4
Efl = 12
# theta,v,b index 
thInd = 0 
vInd = 1
bInd = 2
# limit decimal
limitNum = 6

# 粒子数
nP = 507
# --------------------------------------------------------------------------- #

# =============================================================================
#         Start Particle Filter
# =============================================================================

# 尤度 ------------------------------------------------------------------------
def norm_likelihood(y,x,s2=100,standY=0,time=0):
    
    gauss,years = np.zeros(nCell),np.zeros(nCell)
    if cell == 2 or cell == 4 or cell == 5:
        # 1番近い尤度 [1,], plot用, 尤度最大のindex(V,Thの年数を指定するため)
        gauss = (np.sqrt(2*np.pi*s2))**(-1) * np.exp(-(y-x)**2/(2*s2))
    
        return np.max(gauss), x[np.argsort(gauss)[::-1][:3]], x[np.array([np.argmax(gauss)])]
    
    elif cell == 245:
        y_nk = y[ntI][y[ntI]==standY]
        y_tnk = y[tntI][y[tntI]==standY]
        y_tk = y[ttI][y[ttI]==standY]
        ys = [y_nk,y_tnk,y_tk]
        # 起こってなかったら、0年ヲ入れる
        for i in np.arange(len(ys)):
            if ys[i] == []:
                ys[i] = np.array([0])
        
        if not y_nk.tolist() == []: # 地震がそのセルで起きてないとき
            # degree of similatery for each cell
            gauss_nk = (np.sqrt(2*np.pi*s2))**(-1) * np.exp(-((y_nk-x[ntI])/gt_Year)**2/(2*s2))
            # max of ds year
            year_nk = x[ntI][np.array([np.argmax(gauss_nk)])]
            # max of gauss & years for each cell
            gauss[ntI] = np.max(gauss_nk)
            years[ntI] = year_nk
        if not y_tnk.tolist() == []:
            gauss_tnk = (np.sqrt(2*np.pi*s2))**(-1) * np.exp(-((y_tnk-x[tntI])/gt_Year)**2/(2*s2))
            year_tnk = x[tntI][np.array([np.argmax(gauss_tnk)])]
            gauss[tntI] = np.max(gauss_tnk)
            years[tntI] = year_tnk
        if not y_tk.tolist() == []:
            gauss_tk = (np.sqrt(2*np.pi*s2))**(-1) * np.exp(-((y_tk-x[ttI])/gt_Year)**2/(2*s2))
            year_tk = x[ttI][np.array([np.argmax(gauss_tk)])]
            gauss[ttI] = np.max(gauss_tk)
            years[ttI] = year_tk
        #pdb.set_trace()
        # get biggest year
        indY = np.argmax(gauss)
        
        return gauss, np.array([int(years[indY])]),years
# -----------------------------------------------------------------------------

# 逆関数 -----------------------------------------------------------------------
def InvF(WC,idex,u):
    """
    Args
        WC: ex) array([1,2,3]) -> array([1,3,6]) 
    """
    if np.any(WC<u) == False:
        return 0
    k = np.max(idex[WC<u])
    #pdb.set_trace()
    return k+1
# -----------------------------------------------------------------------------

# 層化サンプリング -----------------------------------------------------------------
def resampling(weights):
    # weights of index
    idx = np.asanyarray(range(nP))
    initU = np.random.uniform(0,1/nP)
    thres = [1/nP*i+initU for i in range(nP)]
    wc = np.cumsum(weights)
    k = np.asanyarray([InvF(wc,idx,val) for val in thres])
    #pdb.set_trace()
    return k
# -----------------------------------------------------------------------------

# 重み付き平均 ------------------------------------------------------------------
def FilterValue(x,wNorm):
    #pdb.set_trace()
    return np.mean(wNorm * x)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def simulate(features,y,pred,t=0,fID=0,standY=0,pTime=0):
    """
    [Args]
        features: システムモデル値xt, th[1400,粒子], V[1400,粒子]
        y: 観測モデル値yt [地震発生年数,]
        pred: 地震年数(1400年) [(地震発生年数zero padding済み),粒子数]
    """
    #pdb.set_trace()
    # 1. 初期化 ---------------------------------------------------------------
    if cell == 2 or cell == 4 or cell == 5:
        # 時系列データ数
        pf_time = y.shape[0] 
    elif cell == 245:
        pf_time = pTime
        
    # 地震発生年数保存　[すべての時系列,100,粒子数,cell数]
    x = np.zeros((pf_time,pred.shape[0],nP,nCell))
    # 状態ベクトル　※1セルの時おかしいかも
    ThVec = np.zeros((pf_time,nP,nCell))
    VVec = np.zeros((pf_time,nP,nCell))
    # 同化年数
    yearsVec = np.zeros((pf_time,nP,nCell))
    # リサンプリング後の特徴量ベクトル
    xResampled = np.zeros((pf_time,nParam,nP,nCell))
    # 重み
    w = np.zeros((pf_time,nP,nCell))
    wNorm = np.zeros((pf_time,nP,nCell))
    #pdb.set_trace()
    # -------------------------------------------------------------------------
    if cell == 2 or cell == 4 or cell == 5:
        # ※ 地震発生年数 [地震発生年数,粒子数]
        x[t] = pred
    elif cell == 245:
        # nankai,tonakai,tokai
        x[t,:,:,ntI] = pred[:,:,ntI]
        x[t,:,:,tntI] = pred[:,:,tntI]
        x[t,:,:,ttI] = pred[:,:,ttI]
    # -------------------------------------------------------------------------
    #pdb.set_trace()
    flag = False
    for i in np.arange(nP): # アンサンブル分まわす
        # =====================================================================
        #         尤度計算
        # =====================================================================
        if cell == 2 or cell == 4 or cell == 5:    
            # predの大きさをそろえてるのを予測した年数だけにする [地震発生年数(可変),]
            yhat = (x[t,x[t,:,i]>0,i]).astype(int)
            
            # 尤度は地震発生年数、重みとかけるのは状態ベクトル
            # 2.c & 2.d 各粒子の尤度と重み ---------------------------------------
            # sortW: 地震発生年数(短い順) kInd: 尤度の高かった年数
            weight, sortW, kInd = norm_likelihood(y[t],yhat)
            w[t,i] = weight
            # -----------------------------------------------------------------
            
            # 2.a & 2.b システムノイズ ---------------------------------------------
            Thnoise = np.random.normal(0,0.01*np.mean(features[0][kInd]))
            Vnoise = np.random.normal(0,0.01*np.mean(features[1][kInd]))
            # -----------------------------------------------------------------
            
            # 尤度の一番高かった年数に合わせる 1400 -> 1, 状態ベクトル + システムノイズ
            # ※ not noise of b
            ThVec[t,i] = features[0][kInd,i] + np.abs(Thnoise)
            VVec[t,i] = features[1][kInd,i] + np.abs(Vnoise)
            
            if not flag:
                sortWs = sortW[:,np.newaxis]
                kInds = kInd[:,np.newaxis]
                flag = True
            else:
                sortWs = np.hstack([sortWs,sortW[:,np.newaxis]])
                kInds = np.hstack([kInds,kInd[:,np.newaxis]])
        # ---------------------------------------------------------------------
        elif cell == 245:
            yhat_nk = (x[t,x[t,:,i,ntI]>0,i,ntI]).astype(int)
            yhat_tnk = (x[t,x[t,:,i,tntI]>0,i,tntI]).astype(int)
            yhat_tk = (x[t,x[t,:,i,ttI]>0,i,ttI]).astype(int)
            yhat = [yhat_nk,yhat_tnk,yhat_tk]
            
            weight, kInd, years = norm_likelihood(y,yhat,standY=standY,time=t)
            weight_nk,weight_tnk,weight_tk = weight[ntI],weight[tntI],weight[ttI]
            
            w[t,i,ntI] = weight_nk
            w[t,i,tntI] = weight_tnk
            w[t,i,ttI] = weight_tk
            yearsVec[t,i,:] = years
            #pdb.set_trace()
            # 2.a & 2.b システムノイズ --------------------------------------------  
            # [3,]
            Thnoise = np.random.normal(0,0.01*np.mean(features[0][kInd],1))
            Vnoise = np.random.normal(0,0.01*np.mean(features[1][kInd],1))
            # -----------------------------------------------------------------
            
            for indC in [ntI,tntI,ttI]:
                # 尤度の一番高かった年数に合わせる 1400 -> 1, 状態ベクトル + システムノイズ
                ThVec[t,i,indC] = features[0][kInd,i,indC] + np.abs(Thnoise[0][indC])
                VVec[t,i,indC] = features[1][kInd,i,indC] + np.abs(Vnoise[0][indC])
            #pdb.set_trace()
            if not flag:
                kInds = kInd
                flag = True
            else:
                # [perticle,]
                kInds = np.hstack([kInds,kInd])
    #pdb.set_trace()
    # 規格化 -------------------------------------------------------------------
    if cell == 2 or cell == 4 or cell == 5:        
        wNorm[t] = w[t]/np.sum(w[t])
    elif cell == 245:
        wNorm[t] = w[t]/np.sum(w[t],0)
    # -------------------------------------------------------------------------
    if cell == 2 or cell == 4 or cell == 5:        
        # =====================================================================
        #         リサンプリング
        # =====================================================================
        # ※ 状態ベクトルを resampling
        k = resampling(wNorm[t])
        #xResample[t+1] = x[t+1,k]
        # theta, v, b
        xResampled[t,thInd,:] = ThVec[t,k]
        xResampled[t,vInd,:] = VVec[t,k]
        # ---------------------------------------------------------------------
    elif cell == 245:
        #pdb.set_trace()
        k_nk = resampling(wNorm[t,:,ntI])
        k_tnk = resampling(wNorm[t,:,tntI])
        k_tk = resampling(wNorm[t,:,ttI])
        
        xResampled[t,thInd,:,ntI] = ThVec[t,k_nk,ntI]
        xResampled[t,thInd,:,tntI] = ThVec[t,k_tnk,tntI]
        xResampled[t,thInd,:,ttI] = ThVec[t,k_tk,ttI]
        
        xResampled[t,vInd,:,ntI] = VVec[t,k_nk,ntI]
        xResampled[t,vInd,:,tntI] = VVec[t,k_nk,tntI]
        xResampled[t,vInd,:,ttI] = VVec[t,k_nk,ttI]
        
        xResampled[t,bInd,:,ntI] = features[bInd][k_nk,ntI]
        xResampled[t,bInd,:,tntI] = features[bInd][k_nk,tntI]
        xResampled[t,bInd,:,ttI] = features[bInd][k_nk,ttI]
        
        #pdb.set_trace()
    print(f"---- 【{t}】 times ----\n")
    #print(f"重み:",wNorm[t])
    print("before xVec | resampling xVec\n")
    print(f"{np.min(ThVec[t])} {np.mean(ThVec[t])} {np.max(ThVec[t])} | {np.min(xResampled[t,thInd])} {np.mean(xResampled[t,thInd])} {np.max(xResampled[t,thInd])}")
    print(f"{np.min(VVec[t])} {np.mean(VVec[t])} {np.max(VVec[t])} | {np.min(xResampled[t,vInd])} {np.mean(xResampled[t,vInd])} {np.max(xResampled[t,vInd])}")
    print("-------------------------\n")
    #pdb.set_trace()
    # 発生年数 plot ------------------------------------------------------------
    myPlot.NumberLine(np.array([standY]),yearsVec[t],time=t,label=f"years_{t}")
    # -------------------------------------------------------------------------
    # 尤度 plot ------------------------------------------------------------
    myPlot.HistLikelihood(w[t],time=t,label=f"likelihood_{t}")
    # -------------------------------------------------------------------------
    #pdb.set_trace()
    if cell == 2 or cell == 4 or cell == 5:        
        return xResampled[t], np.reshape(kInds,[-1])
    elif cell == 245:
        return xResampled[t], kInds
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
if __name__ == "__main__":
   
    # 1 確実に起きた地震
    # 190 全て起きた地震
    for tfID in [190]:
        
        print("-----------------------------------")
        print("------ {} historical eq data ------".format(tfID))
        
        # ------------------------ path ------------------------------------- #
        # dirpath for each logs
        logsPath = "{}".format(tfID)
        # ------------------------------------------------------------------- #
        
        # ----------------- 真の南海トラフ巨大地震履歴 V------------------------- #
        with open(os.path.join(featuresPath,"nankairireki.pkl"), "rb") as fp:
            nkfiles = pickle.load(fp)
        
        # 発生年数取得 & slip velocity (all 30)
        gtV = nkfiles[tfID,:,:]
        filePath = os.path.join(dirPath,logsPath,fileName)
        
        if cell == 2 or cell == 4 or cell == 5:
            # gt eq. in cell
            gtJ = np.where(gtV[:,gtcell]>0)[0]
        elif cell == 245:
            gtJ = np.unique(np.where(gtV>0)[0])
            gtJ_nk = np.where(gtV[:,ntI]>0)[0]
            gtJ_tnk = np.where(gtV[:,tntI]>0)[0]
            gtJ_tk = np.where(gtV[:,ttI]>0)[0]
            gtJs = [gtJ_nk,gtJ_tnk,gtJ_tk]
        # ------------------------------------------------------------------- #
        #pdb.set_trace()
        # 真の地震回数に合わせて
        for iS in np.arange(gtJ.shape[0]):
            
            # ------ file 読み込み, 最初は初期アンサンブル読み取り (logs\\*) ------- # 
            files = glob.glob(filePath)
            if iS > 0: # not first ensemble
                files = [s for s in files if "log_{}_".format(iS-1) in s]
            # --------------------------------------------------------------- #
            
            # ======================== 粒子 作成 ============================= #
            flag,flag1,flag2 = False,False,False
            for fID in np.arange(len(files)):
                
                # file 読み込み ------------------------------------------------
                print('reading',files[fID])
                file = os.path.basename(files[fID])
                logFullPath = os.path.join(dirPath,logsPath,file)
                data = open(logFullPath).readlines()
                # -------------------------------------------------------------
                
                # 特徴量読み込み -----------------------------------------------
                # loading U,theta,V,B [number of data,10]
                U,th,V,B = myData.loadABLV(dirPath,logsPath,file)
                # ------------------------- Error --------------------------- #
                myData.Negative(V,logFullPath,fID) # すべり速度マイナス判定
                # ----------------------------------------------------------- #
                # pJ: 地震が起きた年(2000年=0), [8000,8]
                yU, yth, yV, pJ_all = myData.convV2YearlyData(U,th,V,nYear,cell=cell,cnt=iS) 
                # -------------------------------------------------------------
                #pdb.set_trace()
                # 1回目 -------------------------------------------------------
                if iS == 0:
                    if cell == 2 or cell == 4 or cell == 5:
                        # 類似度比較 最小誤差年取得 ----------------------------- 
                        # th,V [1400,8] これは1番初めだけ, pJ:発生年数(2000年+sInd=0)
                        yth, yV, pJ_all, maxSim = myData.MinErrorNankai(gtV,yth,yV,pJ_all,cell=cell,gtcell=gtcell,nCell=nCell)
                    elif cell == 245:
                        yU, yth, yV, pJ_all, maxSim = myData.MinErrorNankai(gtV,yU,yth,yV,pJ_all,cell=cell,nCell=nCell)
                    
                if cell == 2 or cell == 4 or cell == 5:
                    # concatするために長さそろえる
                    pJ = np.pad(pJ_all,(0,200-pJ_all.shape[0]),"constant",constant_values=0)
                elif cell == 245:
                    nkJ = np.pad(pJ_all[0],(0,200-pJ_all[0].shape[0]),"constant",constant_values=0)
                    tnkJ = np.pad(pJ_all[1],(0,200-pJ_all[1].shape[0]),"constant",constant_values=0)
                    tkJ = np.pad(pJ_all[2],(0,200-pJ_all[2].shape[0]),"constant",constant_values=0)
                    # [100,3(cell)]
                    pJ = np.concatenate((nkJ[:,np.newaxis],tnkJ[:,np.newaxis],tkJ[:,np.newaxis]),1)
                # -------------------------------------------------------------
                #pdb.set_trace()
                # 状態ベクトル ---------------------------------------------------
                if cell == 2 or cell == 4 or cell == 5:
                    if not flag1:
                        # [1400,8,粒子]
                        yth_all = yth[:,:,np.newaxis]
                        yV_all = yV[:,:,np.newaxis]
                        B_all = B
                        # [1400,粒子]
                        yths = yth[:,cell,np.newaxis]
                        yVs = yV[:,cell,np.newaxis]
                        # 年数
                        pJs = pJ[:,np.newaxis] 
                        
                        flag1 = True
                    else:
                        yth_all = np.concatenate([yth_all,yth[:,:,np.newaxis]],2)
                        yV_all = np.concatenate([yV_all,yV[:,:,np.newaxis]],2)
                        B_all = np.vstack([B_all,B])
                        
                        yths = np.hstack([yths,yth[:,cell,np.newaxis]])
                        yVs = np.hstack([yVs,yV[:,cell,np.newaxis]])
                        
                        pJs = np.hstack([pJs,pJ[:,np.newaxis]])
                elif cell == 245:
                    if not flag1:
                        # [1400,8,粒子]
                        yth_all = yth[:,:,np.newaxis]
                        yV_all = yV[:,:,np.newaxis]
                        yU_all = yU[:,:,np.newaxis]
                        B_all = B
                        # 年数
                        pJs = pJ[:,np.newaxis] 
                        
                        flag1 = True
                    else:
                        yth_all = np.concatenate([yth_all,yth[:,:,np.newaxis]],2)
                        yV_all = np.concatenate([yV_all,yV[:,:,np.newaxis]],2)
                        yU_all = np.concatenate([yU_all,yU[:,:,np.newaxis]],2)
                        B_all = np.vstack([B_all,B])
                    
                        pJs = np.hstack([pJs,pJ[:,np.newaxis]])
                # -------------------------------------------------------------
            if cell == 245:
                # [1400,perticle,3(cell)]
                Bs = np.concatenate((B_all[:,nI,np.newaxis],B_all[:,tnI,np.newaxis],B_all[:,tI,np.newaxis]),1)
                yths = np.concatenate((yth_all[:,nI,:,np.newaxis],yth_all[:,tnI,:,np.newaxis],yth_all[:,tI,:,np.newaxis]),2)
                yVs = np.concatenate((yV_all[:,nI,:,np.newaxis],yV_all[:,tnI,:,np.newaxis],yV_all[:,tI,:,np.newaxis]),2)
            #pdb.set_trace()
            Xt = [yths,yVs,Bs]
            B_all = B_all.T
            # =============================================================== #
            #pdb.set_trace()
            # -------------------------- Call PF ---------------------------- #
            print("---- Start PF !! ----\n")
            if iS >= 0:
                if cell == 2 or cell == 4 or cell == 5:
                    # resampled: [theta,V,B,perticle] kInds: index, [perticle,]
                    resampled, kInds = simulate(Xt,gtJ,pJs,t=iS,fID=tfID)    
                elif cell == 245:
                   # resampled [Th/V,perticles,3(cell)]
                   resampled, kInds = simulate(Xt,gtJs,pJs,t=iS,fID=tfID,standY=gtJ[iS],pTime=gtJ.shape[0])        
            # --------------------------------------------------------------- # 
            #pdb.set_trace()
            # リサンプリングした値を代入 ---------------------------------------------
            if cell == 2 or cell == 4 or cell == 5:
                # 同化年数だけの Th,V,B
                for i in np.arange(kInds.shape[0]):
                    tmp1 = yth_all[kInds[i],:,i]
                    tmp2 = yV_all[kInds[i],:,i]
                    if not flag2:
                        yth_rYear = tmp1[:,np.newaxis]
                        yV_rYear = tmp2[:,np.newaxis]
                        flag2 = True
                    else:
                        yth_rYear = np.concatenate([yth_rYear,tmp1[:,np.newaxis]],1)
                        yV_rYear = np.concatenate([yV_rYear,tmp2[:,np.newaxis]],1)
            
                # [8cell,perticle]
                yth_rYear[cell] = resampled[thInd]
                yV_rYear[cell] = resampled[vInd]
                
            elif cell == 245:
                # 8セル分のth,vにresampleした値を代入(次の初期値の準備)
                for i in np.arange(kInds.shape[0]):
                    # U,theta,V yth_all [1400,8(cell),perticle] -> [8,] -> [8,perticle]
                    tmp0 = yU_all[kInds[i],:,i]
                    tmp1 = yth_all[kInds[i],:,i]
                    tmp2 = yV_all[kInds[i],:,i]
                    
                    if not flag2:
                        yU_rYear = tmp0
                        yth_rYear = tmp1
                        yV_rYear = tmp2
                        flag2 = True
                    else:
                        yU_rYear = np.vstack([yU_rYear,tmp0])
                        yth_rYear = np.vstack([yth_rYear,tmp1])
                        yV_rYear = np.vstack([yV_rYear,tmp2])
                
                # for nankai,tonankai,tokai [perticles,8(cell)]
                yth_rYear.T[1] = resampled[thInd,:,ntI]
                yth_rYear.T[nI] = resampled[thInd,:,ntI]
                yth_rYear.T[3] = resampled[thInd,:,tntI]
                yth_rYear.T[tnI] = resampled[thInd,:,tntI]
                yth_rYear.T[tI] = resampled[thInd,:,ttI]
                
                yV_rYear.T[1] = resampled[vInd,:,ntI]
                yV_rYear.T[nI] = resampled[vInd,:,ntI]
                yV_rYear.T[3] = resampled[vInd,:,tntI]
                yV_rYear.T[tnI] = resampled[vInd,:,tntI]
                yV_rYear.T[tI] = resampled[vInd,:,ttI]
                
                B_all[1] = resampled[bInd].T[ntI]
                B_all[nI] = resampled[bInd].T[ntI]
                B_all[3] = resampled[bInd].T[tntI]
                B_all[tnI] = resampled[bInd].T[tntI]
                B_all[tI] = resampled[bInd].T[ttI]
                
            # --------------------------------------------------------------- # 
            #pdb.set_trace()       
            # --------------------------- Xt-1 作成手順 ---------------------- #
                # 1 parfileをアンサンブル分作成
                # 2 batchファイルでファイル番号(Label etc...)受け渡し
                # 3 受け渡された番号のparfileを読み取りsimulation実行
            # --------------------------------------------------------------- #
            FLAG = False
            for lNum in np.arange(resampled.shape[1]): # perticleの分
                
                # ========================= 1 =============================== #
                # defaultparfileファイルを読み込む
                with open("parfileHM031def.txt","r") as fp:
                    alllines = fp.readlines()
                # parfileHM031の改行コード削除
                alllines = [alllines[i].strip().split(",") for i in np.arange(len(alllines))]
                # ※ gtの発生年数に合わせる
                # 計算ステップ指定 (各データで異なる)
                alllines[0][0] = str(gtJ[iS] + 1)
                alllines[0][1] = str(1400)
                #pdb.set_trace()
                # パラメータ設定行抽出
                lines = alllines[Sfl:Efl]
                for nl in np.arange(len(lines)): # 8 cell times
                    # B, U, theta, V
                    inlines = lines[nl]
                    inlines[1] = str(np.round(B_all[nl][lNum],limitNum))
                    inlines[-3] = str(yU_rYear[lNum][nl])
                    inlines[-2] = str(yth_rYear[lNum][nl])
                    inlines[-1] = str(yV_rYear[lNum][nl])
                #pdb.set_trace()
                # Save parfileHM031 -> parfileHM0*
                parFilePath = os.path.join(paramPath,f"{tfID}",f"parfileHM{iS}_{lNum}.txt")
                # 書式を元に戻す
                alllines = [','.join(alllines[i]) + '\n' for i in np.arange(len(alllines))]
                with open(parFilePath,"w") as fp:
                    for line in alllines:
                        fp.write(line)
                
                if not FLAG:
                    # iS: 同化回数, tfID: 0-256(historical of eq.(=directory))
                    fileLabel = np.hstack([iS,lNum,tfID])
                    FLAG=True
                else:
                    fileLabel = np.vstack([fileLabel,np.hstack([iS,lNum,tfID])])                    
                # =========================================================== #
        
            # ========================== 2 ================================== #
            # parFile番号格納
            data = np.c_[fileLabel]
            np.savetxt(paramCSV,data,delimiter=",",fmt="%.0f")
            # =============================================================== #
            
            # ========================== 3 ================================== #
            # all parHM* files
            parallfiles = glob.glob(os.path.join(paramPath,str(tfID),f"*HM{iS}_*.txt"))  
            # =============================================================== #

            # ========================== 4 ================================== #
            # ---- Making Lock.txt 
            lockPath = "Lock.txt"
            lock = str(1)
            with open(lockPath,"w") as fp:
                fp.write(lock)
            # --------------------
            
            os.system(batFile)
            
            sleepTime = 3
            # lockファイル作成時は停止
            while True:
                time.sleep(sleepTime)
                if os.path.exists(lockPath)==False:
                    break
            # =============================================================== #