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

import simulatePF
import makingDataPF as myData
import PlotPF as myPlot


"""
動かし方：
logs/190_1 作成
"""

# -------------------------------- Path ------------------------------------- #
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
# --------------------------------------------------------------------------- #

# --------------------------- parameter ------------------------------------- #

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

ssYears = []
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
def simulate(y,x,t=0,pTime=0,ssYears=0,nP=0):

    print(f"---- 【{t}】 times ----\n")
    
    # 1. 初期化 ----------------------------------------------------------------
    # 状態ベクトル theta,v,year　※1セルの時おかしいかも
    # リサンプリング後の特徴量ベクトル
    # all weight
    maxW = np.zeros((nP))
    # weight in each cell + penalty
    gw = np.zeros((nP,nCell+1))
    # weight for eq. times
    pw = np.zeros((nP,nCell+1))
    wNorm = np.zeros((nP))
    # -------------------------------------------------------------------------
    #pdb.set_trace()
    # -------------------------------------------------------------------------
    flag = False
    for i in np.arange(nP): # アンサンブル分まわす
        # zero-paddingしたpredを予測した年数だけにする [地震発生年数(可変),]
        yhat_nk = (x[x[:,i,ntI]>0,i,ntI]).astype(int)
        yhat_tnk = (x[x[:,i,tntI]>0,i,tntI]).astype(int)
        yhat_tk = (x[x[:,i,ttI]>0,i,ttI]).astype(int)
        #pdb.set_trace()
        yhat_nk = np.array([s for s in yhat_nk.tolist() if s != 0])
        yhat_tnk = np.array([s for s in yhat_tnk.tolist() if s != 0])
        yhat_tk = np.array([s for s in yhat_tk.tolist() if s != 0])
        
        # 1400年のスケールに合わせる
        yhat_nk = yhat_nk - ssYears[i]
        yhat_tnk = yhat_tnk - ssYears[i]
        yhat_tk = yhat_tk - ssYears[i]
        yhat = [yhat_nk,yhat_tnk,yhat_tk]
        
        # 尤度は地震発生年数、重みとかけるのは状態ベクトル
        # 2.c & 2.d 各粒子の尤度と重み -------------------------------------------
        standYs = [y[ntI][t],y[tntI][t],y[ttI][t]]
    
        gweight, gmaxweight, years = simulatePF.norm_likelihood.norm_likelihood_safetypenalty(y,yhat,standYs=standYs,time=t)
        pweight = simulatePF.norm_likelihood.norm_likelihood_times(y,yhat,standYs=standYs)
        
        gw[i] = gweight
        pw[i] = pweight
        
        maxW[i] = gmaxweight + pweight
        
        if not flag:
            yearInds = years
            flag = True
        else:
            # [perticle,3]
            yearInds = np.vstack([yearInds,years])
        
        #----------------------------------------------------------------------
        
    # 規格化 -------------------------------------------------------------------
    # scalling maximum(M),minimum(m)
    xmax = np.max(maxW)
    xmin = np.min(maxW)
    
    m = 1/(np.sqrt(2*np.pi*100)) * np.exp(-((standYs[tntI]-gt_Year)/10)**2/(2*100))
    M = xmax + m
    
    # normalization
    scaledW =  ((maxW - xmin)*(M - m) / (xmax - xmin)) + m
    
    wNorm = scaledW/np.sum(scaledW)
    #--------------------------------------------------------------------------
    
    # リサンプリング ---------------------------------------------------------------
    initU = np.random.uniform(0,1/nP)
    k = simulatePF.resampling(initU,wNorm,nP=nP)
    #--------------------------------------------------------------------------
     
    if isSavetxt:   
        np.savetxt(os.path.join(savetxtPath,"lh",f"lh_p_{t}.txt"),pw)
        np.savetxt(os.path.join(savetxtPath,"lh",f"lh_g_{t}.txt"),gw)
        np.savetxt(os.path.join(savetxtPath,"lh",f"sum_lh_{t}.txt"),maxW,fmt="%4f")
    #pdb.set_trace()
    return k
# --------------------------------------------------------------------------- #

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
            #pdb.set_trace()
            filePath = os.path.join(dirPath,logsPath,fileName)

            # ------ file 読み込み, 最初は初期アンサンブル読み取り (logs\\*) ------- #
            allfiles = glob.glob(filePath)
            files = []
            for lf in natsorted(allfiles):
                files.append(lf)
            # --------------------------------------------------------------- #
            # Num. of perticles
            nP = len(files)
            #pdb.set_trace()
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
                    sYears = sYear
                    flag1 = True
                else:
                    pJs = np.hstack([pJs,pJ[:,np.newaxis]])
                    sYears = np.hstack([sYears,sYear])
                # -------------------------------------------------------------
            if iS == 0:
                #pdb.set_trace()
                ssYears = sYears
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
                
                # 次に作成するファイル名
                nextdir = f"190_{iS+1}"
                # 作成したいファイルある -> True
                isdir = os.path.exists(nextdir)
                # ないとき
                if not isdir:
                    os.path.makedir(os.path.join(dirPath,nextdir))
                    
                # Move log file (perticle) to new directory 
                shutil.copy(os.path.join(dirPath,f"{logsPath}",os.path.basename(file)),os.path.join(dirPath,nextdir,os.path.basename(file)))