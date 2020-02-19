# -*- coding: utf-8 -*-

import os
import glob
import pdb

import numpy as np
import matplotlib.pylab as plt
import statistics as stat
import collections

import makingData as myData

# path ------
logsPath = "logs"
dataPath = "b2b3b4b5b6205-300"
fileName = "*txt"
savePath = "status"
# -----------

filePath = os.path.join(logsPath,dataPath,fileName)
files = glob.glob(filePath)

flag = False
statusnk = np.zeros([len(files),6])
statustnk = np.zeros([len(files),6])
statustk = np.zeros([len(files),6])
for fID in np.arange(len(files)):
                
    # file 読み込み ------------------------------------------------
    print('reading',files[fID])
    file = os.path.basename(files[fID])
    logFullPath = os.path.join(logsPath,dataPath,file)
    data = open(logFullPath).readlines()
    # -------------------------------------------------------------
    
    # 特徴量読み込み -----------------------------------------------
    # loading U,B [number of data,10]
    U,B = myData.loadABLV(logsPath,dataPath,file)
    
    intervalnkY,intervaltnkY,intervaltkY,B = myData.convV2YearlyData(U,B,nYear=10000)
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
    #modenkY = stat.mode(intervalnkY)
    #modetnkY = stat.mode(intervaltnkY)
    #modetkY = stat.mode(intervaltkY)
    modenkY = collections.Counter(intervalnkY).most_common()[0][0]
    modetnkY = collections.Counter(intervaltnkY).most_common()[0][0]
    modetkY = collections.Counter(intervaltkY).most_common()[0][0]

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
    statustnk[fID,4] = maxnkY
    statustnk[fID,5] = minnkY
    
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
    # -------------------------------------------------------------------------

# save ------------------------------------------------------------------------
np.savetxt(f"{savePath}_{dataPath}_nk.txt",statusnk,fmt="%5f")
np.savetxt(f"{savePath}_{dataPath}_tnk.txt",statustnk,fmt="%5f")
np.savetxt(f"{savePath}_{dataPath}_tk.txt",statustk,fmt="%5f")
np.savetxt(f"{savePath}_{dataPath}_b.txt",Bs,fmt="%6f")
# -----------------------------------------------------------------------------
        
