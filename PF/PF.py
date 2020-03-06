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
# save txt path
savetxtPath = "savetxt"
# gt V
featuresPath = "nankairirekifeature"
firstEnName = "first*"
fileName = "*.txt"
K8_AV2File = "K8_AV2.txt"
paramCSV = "ParamFilePF.csv"
batFile = "PyToCPF.bat"
# ----------------------------------------------------------------------- #
        
# --------------------------- parameter --------------------------------- #

isPlot = True

# 南海トラフ巨大地震履歴期間
gt_Year = 1400
# シミュレーションの安定した年
state_Year = 2000
# シミュレータの年数
nYear = 10000

# ※8cellにするとき同化タイミング年数
timings = [84,287,496,761,898,1005]

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

# 同化回数
ptime = 6

# 粒子数
#nP = 3
nP = 499

# --------------------------------------------------------------------------- #

# =============================================================================
#         Start Particle Filter
# =============================================================================

# 尤度 ------------------------------------------------------------------------
def norm_likelihood(y,x,s2=100,standYs=0,time=0):
    """
    standYs: gt eq. year ex) 1times -> [84,84,84]
    """
    #pdb.set_trace()
    # [3(cell)]
    gauss,years = np.zeros(nCell),np.zeros(nCell)
    if cell == 245:
        y_nk = y[ntI][y[ntI]==standYs[ntI]]
        y_tnk = y[tntI][y[tntI]==standYs[tntI]]
        y_tk = y[ttI][y[ttI]==standYs[ttI]]
        ys = [y_nk,y_tnk,y_tk]
        # 起こってなかったら、0年ヲ入れる
        for i in np.arange(len(ys)):
            if ys[i].tolist() == []:
                ys[i] = np.array([0])
        #pdb.set_trace()
        if not y_nk.tolist() == []: # 地震がそのセルで起きてないとき
            # degree of similatery for each cell
            gauss_nk = 1/(np.sqrt(2*np.pi*s2)) * np.exp(-((y_nk-x[ntI])/10)**2/(2*s2))
            # max of ds year
            year_nk = x[ntI][np.array([np.argmax(gauss_nk)])]
            # max of gauss & years for each cell
            gauss[ntI] = np.max(gauss_nk)
            years[ntI] = year_nk
        if not y_tnk.tolist() == []:
            gauss_tnk = 1/(np.sqrt(2*np.pi*s2)) * np.exp(-((y_tnk-x[tntI])/10)**2/(2*s2))
            year_tnk = x[tntI][np.array([np.argmax(gauss_tnk)])]
            gauss[tntI] = np.max(gauss_tnk)
            years[tntI] = year_tnk
        if not y_tk.tolist() == []:
            gauss_tk = 1/(np.sqrt(2*np.pi*s2)) * np.exp(-((y_tk-x[ttI])/10)**2/(2*s2))
            year_tk = x[ttI][np.array([np.argmax(gauss_tk)])]
            gauss[ttI] = np.max(gauss_tk)
            years[ttI] = year_tk
        #pdb.set_trace()
        # sum of gauss, [1,]
        sumgauss = np.cumsum(gauss)[-1]
        return gauss, sumgauss, years
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
def resampling(initU,weights):
    # weights of index
    idx = np.asanyarray(range(nP))
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
def simulate(features,y,pred,t=0,pTime=0):
    """
    [Args]
        features: システムモデル値xt, th[1400,粒子], V[1400,粒子]
        y: 観測モデル値yt [地震発生年数,]
        pred: 地震年数(1400年) [(地震発生年数zero padding済み),粒子数]
    """
    #pdb.set_trace()
    # 1. 初期化 ---------------------------------------------------------------
    if cell == 245:
        # 時系列データ数
        pf_time = pTime
    
    # 地震発生年数保存　[すべての時系列,100,粒子数,cell数]
    x = np.zeros((pf_time,pred.shape[0],nP,nCell))
    # 状態ベクトル theta,v,year　※1セルの時おかしいかも
    ThVec = np.zeros((pf_time,nP,nCell))
    VVec = np.zeros((pf_time,nP,nCell))
    yearVec = np.zeros((pf_time,nP,nCell))
    # リサンプリング後の特徴量ベクトル
    xResampled = np.zeros((pf_time,nParam,nP,nCell))
    # 全部の重み
    w = np.zeros((pf_time,nP,nCell))
    # 重み
    maxW = np.zeros((pf_time,nP))
    wNorm = np.zeros((pf_time,nP,nCell))
    # -------------------------------------------------------------------------
    #pdb.set_trace()
    # -------------------------------------------------------------------------
    if cell == 245:
        # ※ 地震発生年数 [地震発生年数,粒子数]
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
        if cell == 245:
            # predの大きさをそろえてるのを予測した年数だけにする [地震発生年数(可変),]
            yhat_nk = (x[t,x[t,:,i,ntI]>0,i,ntI]).astype(int)
            yhat_tnk = (x[t,x[t,:,i,tntI]>0,i,tntI]).astype(int)
            yhat_tk = (x[t,x[t,:,i,ttI]>0,i,ttI]).astype(int)
            
            if t > 0:
                try:
                    # 2000年 + 同化した年数
                    standInds = ssYears[i] + state_Year
                    # 1400年のスケールに合わせる
                    yhat_nk = yhat_nk - standInds
                    yhat_tnk = yhat_tnk - standInds
                    yhat_tk = yhat_tk - standInds
                    print(yhat_tk)
                except IndexError:
                    pdb.set_trace()
                #pdb.set_trace()
            yhat = [yhat_nk,yhat_tnk,yhat_tk]
            #pdb.set_trace()
            # 尤度は地震発生年数、重みとかけるのは状態ベクトル
            # 2.c & 2.d 各粒子の尤度と重み ---------------------------------------
            standYs = [y[ntI][t],y[tntI][t],y[ttI][t]]
            # kInd: 尤度の高かった年数
            weight, maxweight, years = norm_likelihood(y,yhat,standYs=standYs,time=t)
            weight_nk,weight_tnk,weight_tk = weight[ntI],weight[tntI],weight[ttI]
            # -----------------------------------------------------------------
            
            w[t,i,ntI] = weight_nk
            w[t,i,tntI] = weight_tnk
            w[t,i,ttI] = weight_tk
            maxW[t,i] = maxweight
            for indY,indC in zip(years,[ntI,tntI,ttI]):
                # 各セルで尤度の一番高かった年数に合わせる 1400 -> 1
                # ※ 別々の同化タイミングになる
                try:
                    ThVec[t,i,indC] = features[0][int(years[indC]),i,indC]
                    VVec[t,i,indC] = features[1][int(years[indC]),i,indC]
                except IndexError:
                    pdb.set_trace()
            if not flag:
                yearInds = years
                flag = True
            else:
                # [perticle,3]
                yearInds = np.vstack([yearInds,years])
    
    yearVec[t] = yearInds
    # 規格化 -------------------------------------------------------------------
    if cell == 245:
        # [perticles,3]
        wNorm[t] = w[t]/np.sum(w[t],0)
    # -------------------------------------------------------------------------
    
    # save likelihood txt -----------------------------------------------------
    np.savetxt(os.path.join(savetxtPath,"lh",f"sum_lh_{t}.txt"),maxW[t],fmt="%4f")
    np.savetxt(os.path.join(savetxtPath,"lh",f"lh_{t}.txt"),w[t])
    # -------------------------------------------------------------------------
    #pdb.set_trace()
    # =========================================================================
    #         リサンプリング
    # =========================================================================
    if cell == 245:
        #pdb.set_trace()
        initU = np.random.uniform(0,1/nP)
        # ※ 状態ベクトルを resampling 全て同じinitU       
        k_nk = resampling(initU,wNorm[t,:,ntI])
        k_tnk = resampling(initU,wNorm[t,:,tntI])
        k_tk = resampling(initU,wNorm[t,:,ttI])
        # theta, v, b
        xResampled[t,thInd,:,ntI] = ThVec[t,k_nk,ntI]
        xResampled[t,thInd,:,tntI] = ThVec[t,k_tnk,tntI]
        xResampled[t,thInd,:,ttI] = ThVec[t,k_tk,ttI]
        
        xResampled[t,vInd,:,ntI] = VVec[t,k_nk,ntI]
        xResampled[t,vInd,:,tntI] = VVec[t,k_tnk,tntI]
        xResampled[t,vInd,:,ttI] = VVec[t,k_tk,ttI]
        
        xResampled[t,bInd,:,ntI] = features[bInd][k_nk,ntI]
        xResampled[t,bInd,:,tntI] = features[bInd][k_tnk,tntI]
        xResampled[t,bInd,:,ttI] = features[bInd][k_tk,ttI]
        
    print(f"---- 【{t}】 times ----\n")
    #print(f"重み:",wNorm[t])
    #print("before xVec | resampling xVec\n")
    #print(f"{np.min(ThVec[t])} {np.mean(ThVec[t])} {np.max(ThVec[t])} | {np.min(xResampled[t,thInd])} {np.mean(xResampled[t,thInd])} {np.max(xResampled[t,thInd])}")
    #print(f"{np.min(VVec[t])} {np.mean(VVec[t])} {np.max(VVec[t])} | {np.min(xResampled[t,vInd])} {np.mean(xResampled[t,vInd])} {np.max(xResampled[t,vInd])}")
    #print("-------------------------\n")
    #pdb.set_trace()
    # 発生年数 plot ------------------------------------------------------------
    myPlot.NumberLine(standYs,yearVec[t],label=f"best_years_{t}")
    # -------------------------------------------------------------------------
    # 尤度 plot ---------------------------------------------------------------
    #myPlot.HistLikelihood(maxW[t],label=f"best_likelihood_{t}")
    # -------------------------------------------------------------------------
    if cell == 245:
        return xResampled[t], yearVec[t].astype(int)
    
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
            # no.4まで学習(津波地震ややこしいから含まない) [11,] [84,287,496or499,761,898,|1005,1107,1254,1344,1346]
            gtJ = np.unique(np.where(gtV>0)[0])
            gtJ_nk = np.where(gtV[:,ntI]>0)[0] # [84,287,499,761,898,1005,1107,1254,1346]
            gtJ_tnk = np.where(gtV[:,tntI]>0)[0] # [84,287,496,761,898,1005,1107,1254,1344]
            gtJ_tk = np.where(gtV[:,ttI]>0)[0] # [84,287,496,761,898,1107,1254]
            gtJs = [gtJ_nk,gtJ_tnk,gtJ_tk]
        # ------------------------------------------------------------------- #
        #pdb.set_trace()
        ssYears = np.zeros(nP) # 2000年始まり
        # 真の地震回数に合わせて 学習期間
        for iS in np.arange(ptime-1):
            print(f"*** gt eq.: {gtJs[ntI][iS]} {gtJs[tntI][iS]} {gtJs[ttI][iS]} ***")
            # ------ file 読み込み, 最初は初期アンサンブル読み取り (logs\\*) ------- # 
            allfiles = glob.glob(filePath)
            files = []
            if iS == 0:
                for lf in natsorted(allfiles):
                    files.append(lf)
            else: # not first ensemble
                logfiles = [s for s in allfiles if "log_{}_".format(iS-1) in s]
                for lf in natsorted(logfiles):
                    files.append(lf)
            # --------------------------------------------------------------- #
            
            # ======================== 粒子 作成 ============================= #
            flag,flag1,flag2 = False,False,False
            for fID in np.arange(nP):
                
                # file 読み込み ------------------------------------------------
                print('reading',files[fID])
                file = os.path.basename(files[fID])
                #logFullPath = os.path.join(dirPath,logsPath,file)
                #data = open(logFullPath).readlines()
                # -------------------------------------------------------------
                
                # 特徴量読み込み -----------------------------------------------
                # loading U,theta,V,B [number of data,10]
                U,th,V,B = myData.loadABLV(dirPath,logsPath,file)
                #pdb.set_trace()
                # 1回目 -------------------------------------------------------
                if iS == 0:
                    # 類似度比較 最小誤差年取得 --------------------------------- 
                    # th,V [1400,8] これは1番初めだけ, sYear: 2000年以降(次のファイルの開始年数)
                    if cell == 245:
                        # pJ: 地震が起きた年(2000年=0), [8000,8]
                        yU, yth, yV, pJ_all = myData.convV2YearlyData(U,th,V,nYear,cell=cell,cnt=iS) 
                                
                        yU, yth, yV, pJ_all, maxSim, sYear = myData.MinErrorNankai(gtV,yU,yth,yV,pJ_all,cell=cell,nCell=nCell,label=f"{np.round(B[nI],limitNum)}_{np.round(B[tnI],limitNum)}_{np.round(B[tI],limitNum)}",isPlot=isPlot)
                
                if iS > 0:
                        yU, yth, yV, pJ_all = myData.convV2YearlyData(U,th,V,nYear,cell=cell,cnt=iS,stYear=int(ssYears[fID])) 
                
                if cell == 245:
                    # concatするために長さそろえる
                    nkJ = np.pad(pJ_all[0],(0,300-pJ_all[0].shape[0]),"constant",constant_values=0)
                    tnkJ = np.pad(pJ_all[1],(0,300-pJ_all[1].shape[0]),"constant",constant_values=0)
                    tkJ = np.pad(pJ_all[2],(0,300-pJ_all[2].shape[0]),"constant",constant_values=0)
                    # [100,3(cell)]
                    pJ = np.concatenate((nkJ[:,np.newaxis],tnkJ[:,np.newaxis],tkJ[:,np.newaxis]),1)
                # -------------------------------------------------------------
                #pdb.set_trace()
                # 状態ベクトル ---------------------------------------------------
                if cell == 245:
                    if not flag1:
                        # [1400,8,粒子]
                        yth_all = yth[:,:,np.newaxis]
                        yV_all = yV[:,:,np.newaxis]
                        yU_all = yU[:,:,np.newaxis]
                        B_all = B
                        # 年数
                        pJs = pJ[:,np.newaxis] 
                        sYears = sYear
                        flag1 = True
                    else:
                        yth_all = np.concatenate([yth_all,yth[:,:,np.newaxis]],2)
                        yV_all = np.concatenate([yV_all,yV[:,:,np.newaxis]],2)
                        yU_all = np.concatenate([yU_all,yU[:,:,np.newaxis]],2)
                        B_all = np.vstack([B_all,B])
                    
                        pJs = np.hstack([pJs,pJ[:,np.newaxis]])
                        sYears = np.vstack([sYears,sYear])
                # -------------------------------------------------------------
            if cell == 245:
                # [1400,perticle,3(cell)]
                Bs = np.concatenate((B_all[:,nI,np.newaxis],B_all[:,tnI,np.newaxis],B_all[:,tI,np.newaxis]),1)
                yths = np.concatenate((yth_all[:,nI,:,np.newaxis],yth_all[:,tnI,:,np.newaxis],yth_all[:,tI,:,np.newaxis]),2)
                yVs = np.concatenate((yV_all[:,nI,:,np.newaxis],yV_all[:,tnI,:,np.newaxis],yV_all[:,tI,:,np.newaxis]),2)
            #pdb.set_trace()
            Xt = [yths,yVs,Bs]
            B_all = B_all.T
            if iS == 0:
                ssYears = sYears
            # =============================================================== #
            #pdb.set_trace()
            # -------------------------- Call PF ---------------------------- #
            print("---- Start PF !! ----\n")
            if cell == 245:
               # resampled [Th/V,perticles,3(cell)]
               resampled, kInds = simulate(Xt,gtJs,pJs,t=iS,pTime=ptime)        
            # --------------------------------------------------------------- # 
            #pdb.set_trace()
            # リサンプリングした値を代入 ---------------------------------------------
            if cell == 245:
                # 8セル分のth,vにresampleした値を代入(次の初期値の準備)
                for i in np.arange(nP): # perticle分
                    # U,theta,V,yth_all [1400,8(cell),perticle] -> [8,] -> [perticles,8]
                    tmp0 = yU_all[timings[iS],:,i]
                    tmp1 = yth_all[timings[iS],:,i]
                    tmp2 = yV_all[timings[iS],:,i]
                    
                    if not flag2:
                        yU_rYear = tmp0
                        yth_rYear = tmp1
                        yV_rYear = tmp2
                        flag2 = True
                    else:
                        yU_rYear = np.vstack([yU_rYear,tmp0])
                        yth_rYear = np.vstack([yth_rYear,tmp1])
                        yV_rYear = np.vstack([yV_rYear,tmp2])
                #pdb.set_trace()
                # [perticles,8(cell)] <- [th/v/b,perticles,3(cell)]
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
            for lNum in np.arange(nP): # perticleの分
                #pdb.set_trace()
                # ========================= 1 =============================== #
                # defaultparfileファイルを読み込む
                with open("parfileHM031def.txt","r") as fp:
                    alllines = fp.readlines()
                # parfileHM031の改行コード削除
                alllines = [alllines[i].strip().split(",") for i in np.arange(len(alllines))]
                # ※ gtの発生年数に合わせる
                # 計算ステップ指定 (各データで異なる)
                alllines[0][0] = str(ssYears[lNum][0] + state_Year)
                alllines[0][1] = str(ssYears[lNum][0] + state_Year + gt_Year)
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