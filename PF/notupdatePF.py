# -*- coding: utf-8 -*-

import os
import sys

import glob
import shutil
import pickle
import pdb
import time

import matplotlib.pylab as plt
import numpy as np

from natsort import natsorted

import makingDataPF as myData
import PlotPF as myPlot



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
fileName = "*txt"
K8_AV2File = "K8_AV2.txt"
paramCSV = "ParamFilePF.csv"
batFile = "PyToCPF.bat"
# ----------------------------------------------------------------------- #

# --------------------------- parameter --------------------------------- #

isPlot = True
isSavetxt = True

# 南海トラフ巨大地震履歴期間
gt_Year = 1400
# シミュレーションの安定した年
state_Year = 2000
# シミュレータの年数
nYear = 10000

# ※8cellにするときの同化タイミング年数 (parHM*用)
timings = [84,287,496,761,898,1005,1107,1254,1344]

nCell = 3
# indec of each cell (gt)
ntI,tntI,ttI = 0,1,2
# index of each cell (simulation var)
nI,tnI,tI = 2,4,5

# number of all param Th,V,b
nParam = 3
# reading file start & end line
Sfl = 4
Efl = 12
# theta,v,b index
thInd = 0
vInd = 1
bInd = 2
# limit decimal
limitNum = 6
# penalty year
penaltyNum = 100
# safety year
safetyNum = 100

# Num. of assimilation
ptime = len(timings)

# --------------------------------------------------------------------------- #

# =============================================================================
#         Start Particle Filter
# =============================================================================


#　尤度 + safety & penalty -----------------------------------------------------
def norm_likelihood_safetypenalty(y,x,s2=100,standYs=0,time=0):
    gauss,years = np.zeros(nCell+1),np.zeros(nCell) # for penalty

    y_nk = np.array([standYs[ntI]])
    y_tnk = np.array([standYs[tntI]])
    y_tk = np.array([standYs[ttI]])
    
    # not eq. in tonakai
    if y_tk[0] == 0:
        # ※同化年数±100年に地震があった場合はpenalty
        penaltyInd = np.where((x[ttI]>y_tnk-safetyNum)&(x[ttI]<y_tnk+safetyNum))[0].tolist()

        # not penalty
        if penaltyInd == []:
            pass
        else:
            xpenalty = x[ttI][penaltyInd]
            #pdb.set_trace()
            # ※加算方式
            gauss_pl = np.max(1/(np.sqrt(2*np.pi*s2)) * np.exp(-((y_tnk-xpenalty)/10)**2/(2*s2)))
            # ペナルティ分引くため
            gauss[-1] = -gauss_pl

    # any eq.
    if not y_nk[0] == 0:
        # nearist index of gt year [1,]
        bestInd = np.abs(np.asarray(x[ntI]) - y_nk).argmin()
        bestX = x[ntI][bestInd]
        
        gauss_nk = 1/(np.sqrt(2*np.pi*s2)) * np.exp(-((y_nk-bestX)/10)**2/(2*s2))
        
        # in 100 year -> safety
        if np.abs(bestX-y_nk) < safetyNum:
            gauss[ntI] = gauss_nk
        elif penaltyNum <= np.abs(bestX-y_nk):
            gauss[ntI] = -gauss_nk
        #pdb.set_trace()    
        years[ntI] = bestX

    if not y_tnk[0] == 0:
        
        bestInd = np.abs(np.asarray(x[tntI]) - y_tnk).argmin()
        bestX = x[tntI][bestInd]
        
        gauss_tnk = 1/(np.sqrt(2*np.pi*s2)) * np.exp(-((y_tnk-bestX)/10)**2/(2*s2))
        
        if np.abs(bestX-y_tnk) < safetyNum:
            gauss[tntI] = gauss_tnk
        elif penaltyNum <= np.abs(bestX-y_tnk):
            gauss[tntI] = -gauss_tnk
        
        years[tntI] = bestX

    if not y_tk[0] == 0:
        
        bestInd = np.abs(np.asarray(x[ttI]) - y_tk).argmin()
        bestX = x[ttI][bestInd]
        
        gauss_tk = 1/(np.sqrt(2*np.pi*s2)) * np.exp(-((y_tk-bestX)/10)**2/(2*s2))
        
        if np.abs(bestX-y_tk) < safetyNum:
            gauss[ttI] = gauss_tk
        elif penaltyNum <= np.abs(bestX-y_tk):
            gauss[ttI] = -gauss_tk
        
        years[ttI] = bestX

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
    return np.mean(wNorm * x)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def simulate(y,x,t=0,pTime=0,ssYears=0,nP=0):
    """
    [Args]
        features: システムモデル値xt. th[1400,perticles,3], V[1400,perticles,3], b[perticles,3]
        y: 観測モデル値yt. [地震発生年数,]
        x: 地震年数(1400年). [(地震発生年数zero padding済み),perticles]
        sy: start of assimilation for perticles
    """
    #pdb.set_trace()
    # 1. 初期化 ----------------------------------------------------------------
    # 状態ベクトル theta,v,year　※1セルの時おかしいかも
    # リサンプリング後の特徴量ベクトル
    # all weight
    maxW = np.zeros((nP))
    # weight in each cell + penalty
    w = np.zeros((nP,nCell+1))
    wNorm = np.zeros((nP))

    # -------------------------------------------------------------------------
    pdb.set_trace()
    # -------------------------------------------------------------------------
    flag = False
    for i in np.arange(nP): # アンサンブル分まわす
        # =====================================================================
        #         尤度計算
        # =====================================================================
        # zero-paddingしたpredを予測した年数だけにする [地震発生年数(可変),]
        yhat_nk = (x[x[:,i,ntI]>0,i,ntI]).astype(int)
        yhat_tnk = (x[x[:,i,tntI]>0,i,tntI]).astype(int)
        yhat_tk = (x[x[:,i,ttI]>0,i,ttI]).astype(int)
        
        # 1400年のスケールに合わせる
        yhat_nk = yhat_nk - ssYears[i]
        yhat_tnk = yhat_tnk - ssYears[i]
        yhat_tk = yhat_tk - ssYears[i]
        yhat = [yhat_nk,yhat_tnk,yhat_tk]
        
        # 尤度は地震発生年数、重みとかけるのは状態ベクトル
        # 2.c & 2.d 各粒子の尤度と重み -------------------------------------------
        standYs = [y[ntI][t],y[tntI][t],y[ttI][t]]
    
        weight, maxweight, years = norm_likelihood_safetypenalty(y,yhat,standYs=standYs,time=t)
        
        w[i] = weight
        maxW[i] = maxweight
    
        if not flag:
            yearInds = years
            flag = True
        else:
            # [perticle,3]
            yearInds = np.vstack([yearInds,years])
    
    # 規格化 -------------------------------------------------------------------
    # scalling maximum(M),minimum(m)
    # ※　0割してもいい？
    #M = 1/(np.sqrt(2*np.pi*10)) * np.exp(-((standYs[tntI]-standYs[tntI])/10)**2/(2*10))
    xmax = np.max(maxW)
    xmin = np.min(maxW)
    
    m = 1/(np.sqrt(2*np.pi*100)) * np.exp(-((standYs[tntI]-gt_Year)/10)**2/(2*100))
    M = xmax + m
    
    # normalization
    scaledW =  ((maxW - xmin)*(M - m) / (xmax - xmin)) + m    
    wNorm = scaledW/np.sum(scaledW)
    # -------------------------------------------------------------------------

    # save year & likelihood txt ----------------------------------------------
    if isSavetxt:
        np.savetxt(os.path.join(savetxtPath,"lh",f"sum_lh_{t}.txt"),scaledW,fmt="%4f")
        np.savetxt(os.path.join(savetxtPath,"lh",f"lh_{t}.txt"),w)
        
    # -------------------------------------------------------------------------
    #pdb.set_trace()
    # =========================================================================
    #         リサンプリング
    # =========================================================================
    initU = np.random.uniform(0,1/nP)

    k = resampling(initU,wNorm)
    
    print(f"---- 【{t}】 times ----\n")
    # 発生年数 plot ------------------------------------------------------------
    if isPlot:
        myPlot.NumberLine(standYs,yearInds,label=f"best_years_{t}")
    # -------------------------------------------------------------------------
    
    return k

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
if __name__ == "__main__":

    # 1 確実に起きた地震
    # 190 全て起きた地震
    for tfID in [190]:

        print("-----------------------------------")
        print("------ {} historical eq data ------".format(tfID))
        
        # ----------------- 真の南海トラフ巨大地震履歴 V------------------------- #
        with open(os.path.join(featuresPath,"nankairireki.pkl"), "rb") as fp:
            nkfiles = pickle.load(fp)

        # 発生年数取得 & slip velocity (all 30)
        gtV = nkfiles[tfID,:,:]
        
        # no.4まで学習(津波地震ややこしいから含まない) [11,] [84,287,496or499,761,898,|1005,1107,1254,1344,1346]
        gtJ = np.unique(np.where(gtV>0)[0])
        gtJ_nk = np.where(gtV[:,ntI]>0)[0] # [84,287,499,761,898,1005,1107,1254,1346]
        gtJ_tnk = np.where(gtV[:,tntI]>0)[0] # [84,287,496,761,898,1005,1107,1254,1344]
        gtJ_tk = [84,287,496,761,898,0,1107,1254,0] # 津波と昭和地震 -> 0
        #gtJ_tk = np.where(gtV[:,ttI]>0)[0] # [84,287,496,761,898,1107,1254]
        gtJs = [gtJ_nk,gtJ_tnk,gtJ_tk]
        # ------------------------------------------------------------------- #
        
        for iS in np.arange(ptime): # gt times

            print(f"*** gt eq.: {gtJs[ntI][iS]} {gtJs[tntI][iS]} {gtJs[ttI][iS]} ***")
            
            # ------------------------ path --------------------------------- #
            # dirpath for each logs
            logsPath = f"{tfID}_{iS}"
            # --------------------------------------------------------------- #
        
            filePath = os.path.join(dirPath,logsPath,fileName)

            # ------ file 読み込み, 最初は初期アンサンブル読み取り (logs\\*) ------- #
            allfiles = glob.glob(filePath)
            files = []
            for lf in natsorted(allfiles):
                files.append(lf)
            # --------------------------------------------------------------- #
            # Num. of perticles
            nP = len(files)

            # ======================== 粒子 作成 ============================= #
            flag,flag1,flag2 = False,False,False
            for fID in np.arange(nP):

                # file 読み込み ------------------------------------------------
                print('reading',files[fID])
                file = os.path.basename(files[fID])
                # -------------------------------------------------------------

                # 特徴量読み込み ------------------------------------------------
                # loading U,theta,V,B [number of data,10]
                U,th,V,B = myData.loadABLV(dirPath,logsPath,file)
                # start year
                sYear = int(U[0,1])
                _, _, _, pJ_all = myData.convV2YearlyData(U,th,V,nYear,cnt=1,stYear=sYear)

                # concatするために長さそろえる
                nkJ = np.pad(pJ_all[0],(0,300-pJ_all[0].shape[0]),"constant",constant_values=0)
                tnkJ = np.pad(pJ_all[1],(0,300-pJ_all[1].shape[0]),"constant",constant_values=0)
                tkJ = np.pad(pJ_all[2],(0,300-pJ_all[2].shape[0]),"constant",constant_values=0)
                # [100,3(cell)]
                pJ = np.concatenate((nkJ[:,np.newaxis],tnkJ[:,np.newaxis],tkJ[:,np.newaxis]),1)
                # -------------------------------------------------------------
                #pdb.set_trace()
                # 状態ベクトル ---------------------------------------------------
                if not flag1:
                    # 年数
                    pJs = pJ[:,np.newaxis]
                    #　最適な1400年の開始年数
                    ssYears = sYear
                    flag1 = True
                else:
                    pJs = np.hstack([pJs,pJ[:,np.newaxis]])
                    ssYears = np.hstack([ssYears,sYear])
                # -------------------------------------------------------------
            # -------------------------- Call PF ---------------------------- #
            print("---- Start PF !! ----\n")
            # updateID: resampled index
            updateID = simulate(gtJs,pJs,t=iS,pTime=ptime,ssYears=ssYears,nP=nP)
            # --------------------------------------------------------------- #
            #pdb.set_trace()
            # Save index for Num. of file
            k = np.unique(updateID)
            # Get log files
            updatefiles = [files[ind] for ind in k.tolist()]
            
            for file in updatefiles:
                # Move log file (perticle) to new directory 
                shutil.copy(os.path.join(dirPath,f"{logsPath}",os.path.basename(file)),os.path.join(dirPath,f"190_{iS+1}",os.path.basename(file)))