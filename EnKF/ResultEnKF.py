# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 11:21:11 2019

@author: yu
"""

import os
import glob
import shutil
import pickle
import pdb
import time

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pylab as plt

import numpy as np

from natsort import natsorted

import makingData as myData

# ---- path ---- #
imgPath = "image"
paramPath = "params"
# -------------- #

#------------------------------------------------------------------------------
                # パラメータの推移
#------------------------------------------------------------------------------
def MoveParamPlot(params,years):
    """
    [Args]
    params : Kij,U,th,V,B (縦軸) params[1] [3,5]
    years: eq. year (横軸)
    """
    pdb.set_trace()
    # ---- parameters ---- #
    Kij = params[0] # [5,1]
    U = params[1][:,0] # [ensembles]
    Uex = params[1][:,1]
    theta = params[1][:,2]
    V = params[1][:,3]
    B = params[1][:,-1]
    # -------------------- #
    
    # ---- save png ---- #
    # Kij
    plt.plot(Kij,".",color="b")
    plt.title("Kij")
    plt.savefig(os.path.join(imgPath,paramPath,"Kij.png"))
    plt.close()
    
    # U 
    plt.plot(U,".",color="b")
    plt.title("U")
    plt.savefig(os.path.join(imgPath,paramPath,"U.png"))
    plt.close()
    
    # theta 
    plt.plot(theta,".",color="b")
    plt.title("theta")
    plt.savefig(os.path.join(imgPath,paramPath,"theta.png"))
    plt.close()
    
    # V
    plt.plot(V,".",color="b")
    plt.title("V")
    plt.savefig(os.path.join(imgPath,paramPath,"V.png"))
    plt.close()
    
    # B
    plt.plot(B,".",color="b")
    plt.title("B")
    plt.savefig(os.path.join(imgPath,paramPath,"B.png"))
    plt.close()
    
    

#------------------------------------------------------------------------------
                # 予測したbの推移plot
#------------------------------------------------------------------------------
def MoveB(): # ※ 未実装
    # 予測したbが入っているlogファイル
    dirPath = "prologs"
    fileName = "*.txt"
    
    isWindows = True
    nCell = 8
    
    for fInd in np.arange(256):
    
        logsPath = "{}".format(fInd)
        # 今回は毎回125ファイルある、EnKFがしんでない (全ファイル/125)
        FFlag,flagyr = False,False
        for cellInd in np.arange(1,6):
            
            # all logs(with first ensembles) path
            filePath = os.path.join(dirPath,logsPath,fileName)
            
            allfiles = glob.glob(filePath)
            #pdb.set_trace()
                
            # not first ensembles(複数ある場合はここを調整)
            files = [s for s in allfiles if "log_" in s]
            ffiles = [s for s in allfiles if "first" in s]
            
            # predb格納、南海東南海東海
            flag = False
            cnt=0
            # first ensemble と同じファイル数ある前提
            for fID in np.arange(len(files)):
                # fIDがアンサンブルメンバーに対応
                print('reading',files[fID])
                
                if isWindows:
                    file = files[fID].split('\\')[2]
                    ffile = ffiles[fID].split('\\')[2]
                else:
                    file = files[fID].split('/')[4]
                    ffile = ffiles[fID].split('\\')[4]
                
                """
                # 各年数の初めのファイルから年数を取得し、保存 (発生時年数を横軸に使うため)
                if cnt==0:
                    tmpyr = file.split("_")[1]
                    if not flagyr:
                        predyr = np.array([int(tmpyr)])
                        flagyr = True
                    else:
                        predyr = np.append(predyr,int(tmpyr))"""
                
                # データ読み取り 
                tmpb,_,_,_ = myData.loadABLV(dirPath,logsPath,file,nCell,isLAST=True)
                fb,_,_,_ = myData.loadABLV(dirPath,logsPath,ffile,nCell,isLAST=True)
        
                # 東海のbだけ取得
                if not flag:
                    predb = tmpb[cellInd]
                    firstb = fb[cellInd]
                    flag = True
                else:
                    predb = np.hstack([predb,tmpb[cellInd]])
                    firstb = np.hstack([firstb,fb[cellInd]])
            
            
            # allpredb:更新したbの値 predb:firstEnKFのb
            allpredb = np.vstack([predb,firstb])
            # menapredb:更新したbの平均値
            meanpredb = np.vstack([np.mean(predb),np.mean(firstb)])
            
            # --------------------------------------------------------------------------- #
                
            # 予測したbとその平均をPlot 横軸年数・縦軸予測したb
            # ※初めのアンサンブルだけ多い
            # 地震発生時のパラメータb -> 青
            # 地震が発生しなかった時のパラメータb -> 緑
            #plt.plot(predyr,eqpredb,".",color="b")
            #plt.plot(predyr,noneqpredb,".",color="g")
            plt.plot(allpredb,".",color="b")
            plt.plot(meanpredb,"-",color="red",alpha=0.5,linewidth=5.0)
            plt.xlabel("Time[year]")
            plt.ylabel("Predict b")
            #　保存
            plt.savefig(os.path.join("PredB","{}".format(fInd),"{}.png".format(cellInd)))
            plt.close()
            