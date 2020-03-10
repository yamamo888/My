# -*- coding: utf-8 -*-

import os
import glob
import pdb

import numpy as np
import matplotlib.pylab as plt
import statistics as stat
import seaborn as sns

import DC as myData
import heatmapSample as myHeatmap
import PlotPF as myPlot

# bool ----------
isStatus = False
isdeltaU = True
# ---------------


# -----------------------------------------------------------------------------
def Status():
    flag = False
    statusnk = np.zeros([len(files),6])
    statustnk = np.zeros([len(files),6])
    statustk = np.zeros([len(files),6])
    for fID in np.arange(len(files)):
                    
        # file 読み込み ------------------------------------------------
        print('reading',files[fID])
        file = os.path.basename(files[fID])
        logFullPath = os.path.join(logsPath,file)
        data = open(logFullPath).readlines()
        # -------------------------------------------------------------
        
        # 特徴量読み込み -----------------------------------------------
        # loading U,B [number of data,10] ※エラーかも
        U,B = myData.loadABLV(os.path.join(logsPath,file))
        
        intervalnkY,intervaltnkY,intervaltkY = myData.convV2YearlyData(U)
        # -------------------------------------------------------------
        
        # status -----------------------------------------------------
        # mean
        meannkY = np.mean(intervalnkY)
        meantnkY = np.mean(intervaltnkY)
        meantkY = np.mean(intervaltkY)
        
        # median
        mediannkY = np.median(intervalnkY)
        mediantnkY = np.median(intervaltnkY)
        mediantkY = np.median(intervaltkY)
        
        # mode
        modenkY = stat.mode(intervalnkY)
        modetnkY = stat.mode(intervaltnkY)
        modetkY = stat.mode(intervaltkY)
        
        # variance
        varnkY = np.var(intervalnkY)
        vartnkY = np.var(intervaltnkY)
        vartkY = np.var(intervaltkY)
        
        # max
        maxnkY = np.max(intervalnkY)
        maxtnkY = np.max(intervaltnkY)
        maxtkY = np.max(intervaltkY)
        
        # min
        minnkY = np.min(intervalnkY)
        mintnkY = np.min(intervaltnkY)
        mintkY = np.min(intervaltkY)
        
        # status in nk
        statusnk[fID,0] = meannkY
        statusnk[fID,1] = mediannkY
        statusnk[fID,2] = modenkY
        statusnk[fID,3] = varnkY
        statusnk[fID,4] = maxnkY
        statusnk[fID,5] = minnkY
        
        # status in tnk
        statustnk[fID,0] = meantnkY
        statustnk[fID,1] = mediantnkY
        statustnk[fID,2] = modetnkY
        statustnk[fID,3] = vartnkY
        statustnk[fID,4] = maxtnkY
        statustnk[fID,5] = mintnkY
        
        # status in tk
        statustk[fID,0] = meantkY
        statustk[fID,1] = mediantkY
        statustk[fID,2] = modetkY
        statustk[fID,3] = vartkY
        statustk[fID,4] = maxtkY
        statustk[fID,5] = mintkY
        
        if not flag:
            flag = True
            Bs = B
        else:
            Bs = np.vstack([Bs,B])
        # ---------------------------------------------------------------------
    
    # save --------------------------------------------------------------------
    np.savetxt(f"{savePath}_{dataPath}_nk.txt",statusnk,fmt="%5f")
    np.savetxt(f"{savePath}_{dataPath}_tnk.txt",statustnk,fmt="%5f")
    np.savetxt(f"{savePath}_{dataPath}_tk.txt",statustk,fmt="%5f")
    np.savetxt(f"{savePath}_{dataPath}_b.txt",Bs,fmt="%6f")
    # -------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    
if isStatus:
    # path ------
    logsPath = "logs"
    imgsPath = "images"
    dataPath = "b2b3b4b5b60-100"
    fileName = "*txt"
    savePath = "status"
    # -----------
    #pdb.set_trace()
    #filePath = os.path.join(logsPath,dataPath,fileName)
    filePath = os.path.join(savePath,fileName)
    files = glob.glob(filePath)
        
    cell = 1
    flag = False
    #for i in np.arange(0,20,4):
    for i in [[0.012,0.0126],[0.0126,0.0132],[0.0132,0.0138],[0.0138,0.0144],[0.0144,0.015],[0.015,0.0156],[0.0156,0.0162]]:
        #pdb.set_trace()
        # reading status & paramb -------------------------------------------------
        # [files,3]
        paramb = np.loadtxt(files[0])
        # [files,6]
        #print("reading.")
        #status_nk = np.loadtxt(files[i+1])
        print("reading..")
        tmp_tnk = np.loadtxt(files[2])
        #print("reading...")
        #tmp_tk = np.loadtxt(files[1])
        status_tnk = tmp_tnk[(paramb[:,cell]>=i[0])&(paramb[:,cell]<i[1]),:]
        # -------------------------------------------------------------------------
         
        #pathNames = ["mean","median","mode","var","max"]
        modes = ["mean","var"]
        colors = ["skyblue","lightgreen","orange"]
        
        for ind,mode in zip([3],modes[1]):
            #path = f"{mode}_{os.path.basename(files[i])}" # nk
            path = f"{mode}_{i}" # tnk,tk
            myHeatmap.histgram2D(status_tnk[:,ind],color=f"{colors[cell]}",sPath=f"{path}")    
# -----------------------------------------------------------------------------

if isdeltaU:
    
    logsFullPath = os.path.join(logsPath,"deltaU",fileName)
    logfiles = glob.glob(logsFullPath)
    
    flag = False
    for file in logfiles:
        
        print(file)
    
        U,_ = myData.loadABLV(file)
        yU = myData.convV2YearlyData(U)
        
        if not flag:
            yUs = yU
            flag = True
        else:
            yUs = np.vstack([yUs,yU])
        
        myPlot.Histgram(yUs[:,0],label="yU_nk",color="coral")
        myPlot.Histgram(yUs[:,1],label="yU_tnk",color="forestgreen")
        myPlot.Histgram(yUs[:,2],label="yU_tk",color="royalblue")
        
        
        

