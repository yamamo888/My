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


# -------------------------- command argument --------------------------- #
# 1:learning & evaluation, 2:penalty likelihood, 3:safety & penalty likelihood, 4: scalling likelihood
mode = int(sys.argv[1])
# 0: gauss, 1: mse
error = int(sys.argv[2])
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

isPlot = False
isSavetxt = False

# 南海トラフ巨大地震履歴期間
gt_Year = 1400
# シミュレーションの安定した年
state_Year = 2000
# シミュレータの年数
nYear = 10000

# ※8cellにするときの同化タイミング年数 (parHM*用)
if mode == 1:
    timings = [84,287,496,761,898,1005]
if mode == 2 or mode == 3 or mode == 4:
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

# Num. of perticles
#nP = 4
nP = 501
# --------------------------------------------------------------------------- #

# =============================================================================
#         Start Particle Filter
# =============================================================================

# 尤度 ------------------------------------------------------------------------
def norm_likelihood(y,x,s2=100,standYs=0,time=0):
    """
    standYs: gt eq. year ex) 1times -> [84,84,84]
    """
    pdb.set_trace()
    # [3(cell)]
    gauss,years = np.zeros(nCell),np.zeros(nCell)
    y_nk = y[ntI][y[ntI]==standYs[ntI]]
    y_tnk = y[tntI][y[tntI]==standYs[tntI]]
    y_tk = y[ttI][y[ttI]==standYs[ttI]]

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
    # sum of gauss, [1,]
    sumgauss = np.cumsum(gauss)[-1]
    return gauss, sumgauss, years
# -----------------------------------------------------------------------------

#　尤度 + penalty --------------------------------------------------------------
def norm_likelihood_penalty(y,x,s2=100,standYs=0,time=0):

    gauss,years = np.zeros(nCell+1),np.zeros(nCell) # for penalty

    y_nk = np.array([standYs[ntI]])
    y_tnk = np.array([standYs[tntI]])
    y_tk = np.array([standYs[ttI]])
    #pdb.set_trace()
    # not eq. in tonakai
    if y_tk[0] == 0:
        #pdb.set_trace()
        # ※同化年数±100年に地震があった場合はpenalty
        penaltyInd = np.where((x[ttI]>y_tnk-safetyNum)&(x[ttI]<y_tnk+safetyNum))[0].tolist()

        # not penalty
        if penaltyInd == []:
            pass
        else:
            xpenalty = x[ttI][penaltyInd]
            #pdb.set_trace()
            # ※加算方式
            gauss_pl = np.max(1/(np.sqrt(2*np.pi*s2)) * np.exp(-((y_tnk-xpenalty)/10)**2 / (2*s2)))
            # ペナルティ分引くため
            gauss[-1] = -gauss_pl

    # any eq.
    if not y_nk[0] == 0:
        gauss_nk = 1/(np.sqrt(2*np.pi*s2)) * np.exp(-((y_nk-x[ntI])/10)**2/(2*s2))
        year_nk = x[ntI][np.array([np.argmax(gauss_nk)])]
        gauss[ntI] = np.max(gauss_nk)
        years[ntI] = year_nk

    if not y_tnk[0] == 0:
        gauss_tnk = 1/(np.sqrt(2*np.pi*s2)) * np.exp(-((y_tnk-x[tntI])/10)**2/(2*s2))
        year_tnk = x[tntI][np.array([np.argmax(gauss_tnk)])]
        gauss[tntI] = np.max(gauss_tnk)
        years[tntI] = year_tnk

    if not y_tk[0] == 0:
        gauss_tk = 1/(np.sqrt(2*np.pi*s2)) * np.exp(-((y_tk-x[ttI])/10)**2/(2*s2))
        year_tk = x[ttI][np.array([np.argmax(gauss_tk)])]
        gauss[ttI] = np.max(gauss_tk)
        years[ttI] = year_tk
    #pdb.set_trace()
    # sum of gauss, [1,]
    sumgauss = np.cumsum(gauss)[-1]
    return gauss, sumgauss, years
# -----------------------------------------------------------------------------

#　尤度 + safety & penalty -----------------------------------------------------
def norm_likelihood_safetypenalty(y,x,s2=100,standYs=0,time=0):
    gauss,years = np.zeros(nCell+1),np.zeros(nCell) # for penalty

    y_nk = np.array([standYs[ntI]])
    y_tnk = np.array([standYs[tntI]])
    y_tk = np.array([standYs[ttI]])
    #pdb.set_trace()
    # not eq. in tonakai
    if y_tk[0] == 0:
        #pdb.set_trace()
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
def simulate(features,y,x,t=0,pTime=0,sy=0):
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
    ThVec = np.zeros((nP,nCell))
    VVec = np.zeros((nP,nCell))
    # リサンプリング後の特徴量ベクトル
    xResampled = np.zeros((nParam,nP,nCell))
    # all weight
    maxW = np.zeros((nP))
    if mode == 2 or mode == 3 or mode == 4:
        # weight in each cell + penalty
        w = np.zeros((nP,nCell+1))
        wNorm = np.zeros((nP))
    else:
        w = np.zeros((nP,nCell))
        wNorm = np.zeros((nP,nCell))
    # -------------------------------------------------------------------------
    #pdb.set_trace()
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

        if t > 0:
            # 2000年 + 同化した年数
            standInds = ssYears[i] + state_Year
            # 1400年のスケールに合わせる
            yhat_nk = yhat_nk - standInds
            yhat_tnk = yhat_tnk - standInds
            yhat_tk = yhat_tk - standInds
            
        yhat = [yhat_nk,yhat_tnk,yhat_tk]
        #pdb.set_trace()
        # 尤度は地震発生年数、重みとかけるのは状態ベクトル
        # 2.c & 2.d 各粒子の尤度と重み -------------------------------------------
        standYs = [y[ntI][t],y[tntI][t],y[ttI][t]]

        if mode == 1:
            # weightを採用、各セルで異なる尤度に基づきリサンプリング
            weight, maxweight, years = norm_likelihood(y,yhat,standYs=standYs,time=t)
        elif mode == 2:
            # maxweightを採用
            weight, maxweight, years = norm_likelihood_penalty(y,yhat,standYs=standYs,time=t)
        elif mode == 3 or mode == 4:
            weight, maxweight, years = norm_likelihood_safetypenalty(y,yhat,standYs=standYs,time=t)
        
        w[i] = weight
        maxW[i] = maxweight
        # ---------------------------------------------------------------------
        #pdb.set_trace()
        for indY,indC in zip(years,[ntI,tntI,ttI]):
            # 各セルで尤度の一番高かった年数に合わせる 1400 -> 1
            # ※ 別々の同化タイミングになる
            # ※地震が発生していないときは、tonankaiの地震発生年数を採用
            # ※違う年数でも同じ値の時あり
            if int(indY) == 0: # for tk
                ThVec[i,indC] = features[0][int(years[tntI]),i,indC]
                VVec[i,indC] = features[1][int(years[tntI]),i,indC]
            else:
                ThVec[i,indC] = features[0][int(years[indC]),i,indC]
                VVec[i,indC] = features[1][int(years[indC]),i,indC]

        if not flag:
            yearInds = years
            flag = True
        else:
            # [perticle,3]
            yearInds = np.vstack([yearInds,years])
    #pdb.set_trace()
    # 規格化 -------------------------------------------------------------------
    if mode == 2 or mode == 3:
        # [perticles,]
        wNorm = maxW/np.sum(maxW)
        #print(maxW)
    elif mode == 4:
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
        
    else:
        # [perticles,cellz(3)]
        wNorm = w/np.sum(w,0)
    # -------------------------------------------------------------------------

    # save year & likelihood txt ----------------------------------------------
    if isSavetxt:
        if mode == 4:
            np.savetxt(os.path.join(savetxtPath,"lh",f"sum_lh_{t}.txt"),scaledW,fmt="%4f")
        else:    
            np.savetxt(os.path.join(savetxtPath,"lh",f"sum_lh_{t}.txt"),maxW,fmt="%4f")
        np.savetxt(os.path.join(savetxtPath,"lh",f"lh_{t}.txt"),w)
        np.savetxt(os.path.join(savetxtPath,"bestyear",f"by_{t}.txt"),yearInds,fmt="%d")
    # -------------------------------------------------------------------------
    #pdb.set_trace()
    # =========================================================================
    #         リサンプリング
    # =========================================================================
    initU = np.random.uniform(0,1/nP)

    if mode == 2 or mode == 3 or mode == 4:
        
        # system noise --------------------------------------------------------
        # ※元の値と足してもマイナスになるかも
        # array[cell,perticles] only V & theta parameterがすべて同じ組み合わせになるのを防ぐため
        Thnoise = np.array([np.random.normal(0,0.01*np.mean(ThVec[:,cell]),nP) for cell in np.arange(nCell)])
        Vnoise = np.array([np.random.normal(0,0.01*np.mean(VVec[:,cell]),nP) for cell in np.arange(nCell)])
        # ---------------------------------------------------------------------
        # ※3セル同じ組み合わせのbが選ばれる
        # index for resampling
        k = resampling(initU,wNorm)
        xResampled[thInd] = ThVec[k] + np.abs(Thnoise).T
        xResampled[vInd] = VVec[k] + np.abs(Vnoise).T
        xResampled[bInd] = features[bInd][k]
        updatesy = sy[k]
        #print(xResampled)
    else:
        # ※ 状態ベクトルを resampling 全て同じinitU
        k_nk = resampling(initU,wNorm[:,ntI])
        k_tnk = resampling(initU,wNorm[:,tntI])
        k_tk = resampling(initU,wNorm[:,ttI])
        # theta, v, b
        xResampled[thInd,:,ntI] = ThVec[k_nk,ntI]
        xResampled[thInd,:,tntI] = ThVec[k_tnk,tntI]
        xResampled[thInd,:,ttI] = ThVec[k_tk,ttI]

        xResampled[vInd,:,ntI] = VVec[k_nk,ntI]
        xResampled[vInd,:,tntI] = VVec[k_tnk,tntI]
        xResampled[vInd,:,ttI] = VVec[k_tk,ttI]

        xResampled[bInd,:,ntI] = features[bInd][k_nk,ntI]
        xResampled[bInd,:,tntI] = features[bInd][k_tnk,tntI]
        xResampled[bInd,:,ttI] = features[bInd][k_tk,ttI]
    #pdb.set_trace()
    #print(xResampled)
    print(f"---- 【{t}】 times ----\n")
    # 発生年数 plot ------------------------------------------------------------
    if isPlot:
        myPlot.NumberLine(standYs,yearInds,label=f"best_years_{t}")
    # -------------------------------------------------------------------------

    return xResampled, yearInds.astype(int), updatesy, k

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

        # no.4まで学習(津波地震ややこしいから含まない) [11,] [84,287,496or499,761,898,|1005,1107,1254,1344,1346]
        gtJ = np.unique(np.where(gtV>0)[0])
        gtJ_nk = np.where(gtV[:,ntI]>0)[0] # [84,287,499,761,898,1005,1107,1254,1346]
        gtJ_tnk = np.where(gtV[:,tntI]>0)[0] # [84,287,496,761,898,1005,1107,1254,1344]
        gtJ_tk = [84,287,496,761,898,0,1107,1254,0] # 津波と昭和地震 -> 0
        #gtJ_tk = np.where(gtV[:,ttI]>0)[0] # [84,287,496,761,898,1107,1254]
        gtJs = [gtJ_nk,gtJ_tnk,gtJ_tk]
        # ------------------------------------------------------------------- #
        #pdb.set_trace()
        ssYears = np.zeros(nP) # 2000年始まりするために

        for iS in np.arange(ptime): # gt times

            print(f"*** gt eq.: {gtJs[ntI][iS]} {gtJs[tntI][iS]} {gtJs[ttI][iS]} ***")

            # ------ file 読み込み, 最初は初期アンサンブル読み取り (logs\\*) ------- #
            allfiles = glob.glob(filePath)
            files = []
            if iS == 0: # first time
                for lf in natsorted(allfiles):
                    files.append(lf)
            else:
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
                # -------------------------------------------------------------

                # 特徴量読み込み ------------------------------------------------
                # loading U,theta,V,B [number of data,10]
                U,th,V,B = myData.loadABLV(dirPath,logsPath,file)

                if iS == 0:
                    # 類似度比較 最小誤差年取得 ---------------------------------
                    # th,V [1400,8] これは1番初めだけ, sYear: 2000年以降(次のファイルの開始年数)
                    # pJ: 地震が起きた年(2000年=0), [8000,8]
                    yU, yth, yV, pJ_all = myData.convV2YearlyData(U,th,V,nYear,cnt=iS)
                    
                    if error == 0: # gauss
                        yU, yth, yV, pJ_all, maxSim, sYear = myData.GaussErrorNankai(gtV,yU,yth,yV,pJ_all,nCell=nCell)
                    elif error == 1: # mse
                        yU, yth, yV, pJ_all, maxSim, sYear = myData.MSErrorNankai(gtV,yU,yth,yV,pJ_all,nCell=nCell)
                        
                if iS > 0:
                    yU, yth, yV, pJ_all = myData.convV2YearlyData(U,th,V,nYear,cnt=iS,stYear=int(ssYears[fID]))

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
            #pdb.set_trace()
            # [1400,perticle,3(cell)]
            Bs = np.concatenate((B_all[:,nI,np.newaxis],B_all[:,tnI,np.newaxis],B_all[:,tI,np.newaxis]),1)
            yths = np.concatenate((yth_all[:,nI,:,np.newaxis],yth_all[:,tnI,:,np.newaxis],yth_all[:,tI,:,np.newaxis]),2)
            yVs = np.concatenate((yV_all[:,nI,:,np.newaxis],yV_all[:,tnI,:,np.newaxis],yV_all[:,tI,:,np.newaxis]),2)

            Xt = [yths,yVs,Bs]
            B_all = B_all.T

            if iS == 0:
                ssYears = sYears
            # =============================================================== #
            #pdb.set_trace()
            # -------------------------- Call PF ---------------------------- #
            print("---- Start PF !! ----\n")
            # resampled: [Th/V,perticles,3(cell)], 
            # kInds: best year
            # ssYears: update start of assimulation years
            # updateID: resampled index
            resampled, bestyears, ssYears, updateID = simulate(Xt,gtJs,pJs,t=iS,pTime=ptime,sy=ssYears)
            # --------------------------------------------------------------- #
            #pdb.set_trace()
            # リサンプリングした値を代入 -----------------------------------------------
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
            print(yth_rYear)
            print(yV_rYear)
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
                # Reading default parfile
                with open("parfileHM031def.txt","r") as fp:
                    alllines = fp.readlines()
                # parfileHM031の改行コード削除
                alllines = [alllines[i].strip().split(",") for i in np.arange(len(alllines))]
                # ※ gtの発生年数に合わせる
                # 計算ステップ指定 (各データで異なる)
                alllines[0][0] = str(ssYears[lNum][0] + state_Year)
                alllines[0][1] = str(ssYears[lNum][0] + state_Year + gt_Year -1)

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