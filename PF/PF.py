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
# gt & obs name of cell, 2 or 4 or 5 or 123
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

# only one cell
if cell == 2 or cell == 4 or cell == 5:
    paramCSV = "ParamFilePF.csv"
    batFile = "PyToCPF.bat"
    
# multi cells
elif cell == 245:
    paramCSV= "Param3File.csv"
    batFile = "PyToC3.bat"
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
    # number of all param Th,V,b
    nParam = 2
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
elif cell == 123:
    nParam = 3
    nCell = 3
    gt_nCell = 3
    # indec of each cell (gt)
    ntI,tntI,ttI = 0,1,2
    # index of each cell (simulation var)
    nI,tnI,tI = 2,4,5

# slip velocity?
slip = 0

# reading file start & end line
Sfl = 4
Efl = 12

# theta index 
thInd = 0
# V index 
vInd = 1

# limit decimal
limitNum = 6

# 粒子数
nP = 4
   
# --------------------------------------------------------------------------- #

# =============================================================================
#         Start Particle Filter
# =============================================================================

# 尤度 ------------------------------------------------------------------------
def norm_likelihood(y,x,s2=100):
    
    # 1番近い尤度 [1,], plot用, 尤度最大のindex(V,Thの年数を指定するため)
    gauss = (np.sqrt(2*np.pi*s2))**(-1) * np.exp(-(y-x)**2/(2*s2))
    #pdb.set_trace()
    return np.max(gauss), x[np.argsort(gauss)[::-1][:3]], x[np.array([np.argmax(gauss)])]
# -----------------------------------------------------------------------------

# 逆関数 -----------------------------------------------------------------------
def InvF(WC,idex,u):
    if np.any(WC<u) == False:
        return 0
    k = np.max(idex[WC<u])
    return k+1
# -----------------------------------------------------------------------------

# 層化サンプリング -----------------------------------------------------------------
def resampling(weights):
    idx = np.asanyarray(range(nP))
    initU = np.random.uniform(0,1/nP)
    u = [1/nP*1+initU for i in range(nP)]
    wc = np.cumsum(weights)
    k = np.asanyarray([InvF(wc,idx,val) for val in u])
    return k
# -----------------------------------------------------------------------------

# 重み付き平均 ------------------------------------------------------------------
def FilterValue(x,wNorm):
    return np.mean(wNorm * x)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def simulate(features,y,pred,t=0,fID=0):
    """
    [Args]
        features: システムモデル値xt, th[1400,粒子], V[1400,粒子]
        y: 観測モデル値yt [地震発生年数,]
        pred: 地震年数(1400年) [(地震発生年数zero padding済み),粒子数]
    """
    #pdb.set_trace()
    # 時系列データ数
    pf_time = y.shape[0]
    # -------------------------------------------------------------------------
    # 状態ベクトル
    ThVec = np.zeros((pf_time,nP))
    VVec = np.zeros((pf_time,nP))
    
    # リサンプリング後の特徴量ベクトル
    xResampled = np.zeros((pf_time,nParam,nP))
    # 重み
    w = np.zeros((pf_time,nP))
    wNorm = np.zeros((pf_time,nP))
    #　時刻毎の尤度
    #lh = np.zeros(pf_time)
    # -------------------------------------------------------------------------
    
    # 1. 初期化 ----------------------------------------------------------------
    # 地震発生年数保存　[すべての時系列,450,粒子数]
    x = np.zeros((y.shape[0],pred.shape[0],nP))
    # ※ 地震発生年数 [地震発生年数,粒子数]
    x[t] = pred
    # -------------------------------------------------------------------------
    
    flag = False
    for i in np.arange(nP): # アンサンブル分まわす
        #pdb.set_trace()
        # =====================================================================
        #         尤度計算
        # =====================================================================
        # predの大きさをそろえてるのを予測した年数だけにする [地震発生年数(可変),]
        yhat = (x[t,x[t,:,i]>0,i]).astype(int)
        
        # 尤度は地震発生年数、重みとかけるのは状態ベクトル
        # 2.c & 2.d 各粒子の尤度と重み -------------------------------------------
        # kInd: 尤度の高かった年数
        weight, sortW, kInd = norm_likelihood(y[t],yhat)
        w[t,i] = weight
        # 対数尤度
        #lh[t] = np.log(np.sum(w[t]))
        # ---------------------------------------------------------------------
        
        # 2.a & 2.b システムノイズ ------------------------------------------------  
        Thnoise = np.random.normal(0,0.01*np.mean(features[0][kInd]),nCell)
        Vnoise = np.random.normal(0,0.01*np.mean(features[1][kInd]),nCell)
        # ---------------------------------------------------------------------
        #pdb.set_trace()
        # 尤度の一番高かった年数に合わせる 1400 -> 1, 状態ベクトル + システムノイズ
        ThVec[t,i] = features[0][kInd,i] + np.abs(Thnoise)
        VVec[t,i] = features[1][kInd,i] + np.abs(Vnoise)
        
        if not flag:
            sortWs = sortW[:,np.newaxis]
            kInds = kInd[:,np.newaxis]
            flag = True
        else:
            sortWs = np.hstack([sortWs,sortW[:,np.newaxis]])
            kInds = np.hstack([kInds,kInd[:,np.newaxis]])
    #pdb.set_trace()        
    wNorm[t] = w[t]/np.sum(w[t]) # 規格化
    # -------------------------------------------------------------------------
   
    # フィルタリングされた状態ベクトル --------------------------------------------------
    ThFilter = FilterValue(ThVec[t],wNorm[t])
    VFilter = FilterValue(VVec[t],wNorm[t])
    # -------------------------------------------------------------------------
    #pdb.set_trace()
    # =========================================================================
    #         リサンプリング
    # =========================================================================
    # ※ 状態ベクトルを resamplig
    k = resampling(wNorm[t])
    #xResample[t+1] = x[t+1,k]
    # theta, v, b
    xResampled[t,thInd,:] = ThVec[t,k]
    xResampled[t,vInd,:] = VVec[t,k]
    # -------------------------------------------------------------------------
    
    print(f"---- 【{t}】 times ----\n")
    #print(f"重み:",wNorm[t])
    print("before xVec | resampling xVec\n")
    print(f"{np.min(ThVec[t])} {np.mean(ThVec[t])} {np.max(ThVec[t])} | {np.min(xResampled[t,thInd,:])} {np.mean(xResampled[t,thInd,:])} {np.max(xResampled[t,thInd,:])}")
    print(f"{np.min(VVec[t])} {np.mean(VVec[t])} {np.max(VVec[t])} | {np.min(xResampled[t,vInd,:])} {np.mean(xResampled[t,vInd,:])} {np.max(xResampled[t,vInd,:])}")
    print(f"加重平均 Predict Theta:{ThFilter}, V:{VFilter}")
    print("-------------------------\n")
    
    # 発生年数 plot ------------------------------------------------------------
    #myPlot.NumberLine(y[t],sortWs,label=f"tfID{fID}_times{t}")
    # -------------------------------------------------------------------------
    # 尤度 plot ---------------------------------------------------------------
    #myPlot.histPlot(wNorm[t],label=f"tfID{fID}_times{t}")
    # -------------------------------------------------------------------------

    return xResampled[t], np.reshape(kInds,[-1])

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
if __name__ == "__main__":
   
    # 1 確実に起きた地震
    # 190 全て起きた地震
    cnt = 0
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
            # gt eq. in all cell
            gtJ = np.unique(np.where(gtV[:,gtcell]>0)[0])    
        # ------------------------------------------------------------------- #
        #pdb.set_trace()
        # 地震が完全に起きなくなった時まで
        #while True:
        for iS in np.arange(gtJ.shape[0]):
            
            # ------ file 読み込み, 最初は初期アンサンブル読み取り (logs\\*) ------- # 
            files = glob.glob(filePath)
            if iS > 0: # not first ensemble
                files = [s for s in files if "log_{}_".format(iS-1) in s]
            # --------------------------------------------------------------- #
            
            # ======================== 粒子 作成 ============================= #
            sims = np.zeros((len(files))) # 類似度保存
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
                th,V,B = myData.loadABLV(dirPath,logsPath,file)
                # ------------------------- Error --------------------------- #
                myData.Negative(V,logFullPath,fID) # すべり速度マイナス判定
                # ----------------------------------------------------------- #
                # pJ: 地震が起きた年(2000年=0)
                yth,yV,pJ_all = myData.convV2YearlyData(th,V,nYear,cell=cell,cnt=iS) 
                # -------------------------------------------------------------
                #pdb.set_trace()
                # 1回目 -------------------------------------------------------
                if iS == 0:
                    # 類似度比較 最小誤差年取得 --------------------------------- 
                    # th,V [1400,8] これは1番初めだけ, pJ:発生年数(2000年+sInd=0)
                    yth, yV, pJ_all, maxSim = myData.MinErrorNankai(gtV,yth,yV,pJ_all,cell=cell,gtcell=gtcell,nCell=nCell)
                
                # concatするために長さそろえる
                pJ = np.pad(pJ_all,(0,450-pJ_all.shape[0]),"constant",constant_values=0)
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
                        
                # -------------------------------------------------------------
            Xt = [yths,yVs]
            B_all = B_all.T
            # =============================================================== #
            #pdb.set_trace()
            # -------------------------- Call PF ---------------------------- #
            print("---- Start PF !! ----\n")
            if iS >= 0:
                print(iS)
                # resampled: [theta,V,B,perticle] kInds: index, [perticle,]
                resampled, kInds = simulate(Xt,gtJ,pJs,t=iS,fID=tfID)
                
            # --------------------------------------------------------------- # 
            #pdb.set_trace()
            # --------------------------------------------------------------- # 
            # 同化年数だけの Th,V
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
            
            # リサンプリングした値を代入 ---------------------------------------------
            if cell == 2 or cell == 4 or cell == 5:
                # [8cell,perticle]
                yth_rYear[cell] = resampled[thInd]
                yV_rYear[cell] = resampled[vInd]
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
                    # B, theta, V
                    inlines = lines[nl]
                    inlines[1] = str(np.round(B_all[nl][lNum],limitNum))
                    inlines[-2] = str(np.round(yth_rYear[:,lNum][nl],limitNum))
                    inlines[-1] = str(yV_rYear[:,lNum][nl])
                #pdb.set_trace()
                # Save parfileHM031 -> parfileHM0*
                parFilePath = os.path.join(paramPath,f"{tfID}",f"parfileHM{iS}_{lNum}.txt")
                # 書式を元に戻す
                alllines = [','.join(alllines[i]) + '\n' for i in np.arange(len(alllines))]
                with open(parFilePath,"w") as fp:
                    for line in alllines:
                        fp.write(line)
                
                if not FLAG:
                    FLAG=True
                    # iS: 同化回数
                    # tfID: 0-256(historical of eq.(=directory))
                    parNum = np.hstack([iS,lNum,tfID])
                else:
                    parNum = np.vstack([parNum,np.hstack([iS,lNum,tfID])])                    
                # =========================================================== #
        
            # ========================== 2 ================================== #
            if cell == 2 or cell == 4 or cell == 5:
                fileLabel = parNum
            
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