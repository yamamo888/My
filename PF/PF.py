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
from scipy.stats import poisson

from natsort import natsorted

import setenv
import simulatePF as sp
import updateX
import makingDataPF as myData
import PlotPF as myPlot


# command argument
#args = sys.argv
#option = []
#option = setenv.setenv(args)
#pdb.set_trace()
# -------------------------- command argument ------------------------------- #
mode = int(sys.argv[1])
# --------------------------------------------------------------------------- #

# ----------------------------- Path ---------------------------------------- #
# In first ensamble file & logs file
logsPath = "logs"
imgPath = "images"
# gt V
featuresPath = "nankairirekifeature"
firstEnName = "first*"
fileName = "*.txt"
K8_AV2File = "K8_AV2.txt"
paramCSV = "ParamFilePF.csv"
# --------------------------------------------------------------------------- #

# ------------------------------- parameter --------------------------------- #

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

# theta,v,b index
thInd = 0
vInd = 1
bInd = 2

# penalty year
penaltyNum = 100
# safety year
safetyNum = 100

# Num. of assimilation
ptime = len(timings)

# 1400年間の開始年数保存用
ssYears = []
# --------------------------------------------------------------------------- #

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
if __name__ == "__main__":

    # 1 確実に起きた地震
    # 190 全て起きた地震
    for tfID in [190]:

        print("-----------------------------------")
        print("------ {} historical eq data ------".format(tfID))

        # dirpath for each logs
        dirPath = f"{tfID}"
        # full path for each logs
        filePath = os.path.join(logsPath,dirPath,fileName)
        # path exist or not exist
        myData.isDirectory(os.path.join(logsPath,dirPath))
        myData.isDirectory(os.path.join('parFile',dirPath))
    
        # ----------------- 真の南海トラフ巨大地震履歴 V------------------------- #
        with open(os.path.join(featuresPath,"nankairireki.pkl"), "rb") as fp:
            nkfiles = pickle.load(fp)

        # 発生年数取得 & slip velocity (all 30)
        gtV = nkfiles[tfID,:,:]
      
        # 重複なしの真値地震
        #gtJ = np.unique(np.where(gtV>0)[0])
        gtJ_nk = np.where(gtV[:,ntI]>0)[0] # [84,287,499,761,898,1005,1107,1254,1346]
        gtJ_tnk = np.where(gtV[:,tntI]>0)[0] # [84,287,496,761,898,1005,1107,1254,1344]
        gtJ_tk = np.array([84,287,496,761,898,0,1107,1254,0]) # 津波と昭和地震 -> 0
        #gtJ_tk = np.where(gtV[:,ttI]>0)[0] # [84,287,496,761,898,1107,1254]
        gtJs = [gtJ_nk,gtJ_tnk,gtJ_tk]
        # ------------------------------------------------------------------- #
        
        for iS in np.arange(ptime): # gt times
            print(f"*** gt eq.: {gtJs[ntI][iS]} {gtJs[tntI][iS]} {gtJs[ttI][iS]} ***")

            # reading logs file, 最初は初期アンサンブル読み取り (logs\\*)
            allfiles = glob.glob(filePath)
            files = []
            if iS == 0: # first time
                for lf in natsorted(allfiles):
                    files.append(lf)
            else:
                logfiles = [s for s in allfiles if "log_{}_".format(iS-1) in s]
                for lf in natsorted(logfiles):
                    files.append(lf)
            
            # Num. of perticles
            nP = len(files)            
            # ======================== 粒子 作成 ============================= #
            flag,flag1,flag2 = False,False,False
            for fID in np.arange(nP):
           
                # reading file
                print('reading',files[fID])
                file = os.path.basename(files[fID])
                
                # reading features (X_t) U,theta,V,B [number of data,10]
                U,th,V,B = myData.loadABLV(logsPath,dirPath,file)

                if iS == 0:
                    # 類似度比較 最小誤差年取得
                    # yU, yth, yV [1400,8]
                    # sYear: 2000年以降(次のファイルの開始年数)
                    # pJ: 地震が起きた年(2000年=0), [8000,8]
                    yU, yth, yV, pJ_all = myData.convV2YearlyData(U,th,V,nYear,cnt=iS)
                    yU, yth, yV, pJ_all, _, sYear = myData.MSErrorNankai(gtV,yU,yth,yV,pJ_all,nCell=nCell)
                    
                    
                if iS > 0:
                    # yU, yth, yV [1400,8]
                    yU, yth, yV, pJ_all = myData.convV2YearlyData(U,th,V,nYear,cnt=iS,stYear=int(ssYears[fID]))

                # concatするために長さそろえる
                nkJ = np.pad(pJ_all[0],(0,300-pJ_all[0].shape[0]),"constant",constant_values=0)
                tnkJ = np.pad(pJ_all[1],(0,300-pJ_all[1].shape[0]),"constant",constant_values=0)
                tkJ = np.pad(pJ_all[2],(0,300-pJ_all[2].shape[0]),"constant",constant_values=0)
                # eq. years [100,3(cell)]
                pJ = np.concatenate((nkJ[:,np.newaxis],tnkJ[:,np.newaxis],tkJ[:,np.newaxis]),1)
                # 類似度
                maxSim = myData.MAEyear(gtJs,pJ_all)
                # -------------------------------------------------------------
                #pdb.set_trace()
                if not flag1:
                    # features X_t [1400,8,perticles]
                    yth_all = yth[:,:,np.newaxis]
                    yV_all = yV[:,:,np.newaxis]
                    yU_all = yU[:,:,np.newaxis]
                    B_all = B
                    # 年数
                    pJs = pJ[:,np.newaxis]
                    sYears = sYear
                    # 類似度
                    maxSims = maxSim
                    flag1 = True
                else:
                    yth_all = np.concatenate([yth_all,yth[:,:,np.newaxis]],2)
                    yV_all = np.concatenate([yV_all,yV[:,:,np.newaxis]],2)
                    yU_all = np.concatenate([yU_all,yU[:,:,np.newaxis]],2)
                    B_all = np.vstack([B_all,B])

                    pJs = np.hstack([pJs,pJ[:,np.newaxis]])
                    sYears = np.vstack([sYears,sYear])
                    
                    maxSims = np.vstack([maxSims,maxSim])
                # -------------------------------------------------------------
            #pdb.set_trace()
            # 必要なセルの特徴量抽出 [1400,perticle,3(cell)]
            Bs = np.concatenate((B_all[:,nI,np.newaxis],B_all[:,tnI,np.newaxis],B_all[:,tI,np.newaxis]),1)
            yths = np.concatenate((yth_all[:,nI,:,np.newaxis],yth_all[:,tnI,:,np.newaxis],yth_all[:,tI,:,np.newaxis]),2)
            yVs = np.concatenate((yV_all[:,nI,:,np.newaxis],yV_all[:,tnI,:,np.newaxis],yV_all[:,tI,:,np.newaxis]),2)

            Xt = [yths,yVs,Bs]
            B_all = B_all.T

            if iS == 0:
                ssYears = sYears
            # Plot 一番類似度の高い履歴
            bestind = np.argmin(maxSims)
            # 一番高い類似度
            bestSim = maxSims[bestind]
            bestB = Bs[bestind]
            bestpJs = pJs[:,bestind,:]
            bestpJ = [np.array(bestpJs[:,cell][bestpJs[:,cell]>0]) for cell in np.arange(3)]
            
            if iS > 0:
                bestpJ = bestpJ - ssYears[bestind] - state_Year
            
            # saved rireki path
            rirekipath = os.path.join(imgPath,f'exrireki_{mode}')
            myData.isDirectory(rirekipath)
            myPlot.Rireki(gtJs,bestpJ,path=rirekipath,label=f"{iS}_{bestind}_{np.round(bestB[ntI],6)}_{np.round(bestB[tntI],6)}_{np.round(bestB[ttI],6)}",title=f'{bestSim[0]}')
            
            # =============================================================== #
            #pdb.set_trace()
            print("---- Start PF !! ----\n")
            # resampled: [Th/V,perticles,3(cell)]
            # kInds: best year
            # ssYears: update start of assimulation years
            # updateID: resampled index
            resampled, bestyears, ssYears, updateID = sp.simulate(Xt,gtJs,pJs,ssYears,mode=mode,t=iS,pTime=ptime,sy=ssYears,nP=nP,nCell=nCell,isSavetxt=isSavetxt,isPlot=isPlot)
            #pdb.set_trace()
            # 8セル分のth,vにresampleした値を代入(次の初期値の準備) ------------------
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
            # Call simulation (X_t -> X_t+1)
            updateX.updateParameters(yU_rYear,yth_rYear,yV_rYear,B_all,ssYears,tfID=tfID,iS=iS,nP=nP)