# -*- coding: utf-8 -*-

import os
import glob

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

import PlotPF as myPlot
import DC as myData




# parameter -------------------------------------------------------------------
cellname = ["nk","tnk","tk"]
ntI,tntI,ttI = 0,1,2
# -----------------------------------------------------------------------------

# mode
isLoss = True
isInterval = False

def CountLoss3(gt,pred1,pred2,mode="none"):
    """
    loss of win or lose 3cell var.
    gt,pred1,pred2: [files,3(cell)]
    """
    
    # mean square error
    norm1 = np.square(gt-pred1)
    norm2 = np.square(gt-pred2)
    
    # loss of pred1 bigger than that of pred2 >>> True
    bool12 = (norm1 >= norm2)
    # [True,True,False] >>> True, [False,True,False] >>> False
    mode12 = np.reshape(stats.mode(bool12,1)[0],[-1,])
    
    # pred2 win & lose norm of paramb for histgram, [files,3]
    winNorms = norm2[mode12,:]
    loseNorms = norm2[np.logical_not(mode12),:]
    # vs var.
    winvsNorms = norm1[mode12,:]
    losevsNorms = norm1[np.logical_not(mode12),:]
    # [files,1]
    winNorm = np.sum(winNorms,1)
    loseNorm = np.sum(loseNorms,1) 
    
    # best & worst 1% index
    bestInd = np.argsort(winNorm)[:750]
    worstInd = np.argsort(loseNorm)[::-1][:750]
    # 100 var.
    best100Ind = np.argsort(winNorm)[:100]
    worst100Ind = np.argsort(loseNorm)[::-1][:100]
    
    # stand pred win & lose paramb for scatter
    winB = pred2[bestInd,:]
    loseB = pred2[worstInd,:]
    # gt var.
    gtwinB = gt[bestInd,:]
    gtloseB = gt[worstInd,:]
    # vs var.
    winvsB = pred1[bestInd,:]
    losevsB = pred1[worstInd,:]
    # 100 var.
    win100B = pred2[best100Ind,:]
    worst100B = pred2[worst100Ind,:]
    
    # -------------------------------------------------------------------------
    
    # range of paramb for plot
    minB,maxB = np.min(gt,0),np.max(gt,0)
    # scatter for predict paramb
    myPlot.scatter3D(winB[:,ntI],winB[:,tntI],winB[:,ttI],rangeP=[minB,maxB],title=f"Win {mode}",label=f"winB_{mode}")
    myPlot.scatter3D(loseB[:,ntI],loseB[:,tntI],loseB[:,ttI],rangeP=[minB,maxB],title=f"Lose {mode}",label=f"loseB_{mode}")
    # scatter for gt paramb
    myPlot.scatter3D(gtwinB[:,ntI],gtwinB[:,tntI],gtwinB[:,ttI],rangeP=[minB,maxB],title=f"Win {mode} gt var",label=f"gtwinB_{mode}")
    myPlot.scatter3D(gtloseB[:,ntI],gtloseB[:,tntI],gtloseB[:,ttI],rangeP=[minB,maxB],title=f"Lose {mode} gt var",label=f"gtloseB_{mode}")
    # 100 var.
    myPlot.scatter3D(win100B[:,ntI],win100B[:,tntI],win100B[:,ttI],rangeP=[minB,maxB],title=f"Win {mode} 100 var",label=f"winB_{mode}")
    myPlot.scatter3D(worst100B[:,ntI],worst100B[:,tntI],worst100B[:,ttI],rangeP=[minB,maxB],title=f"Lose {mode} 100 var",label=f"loseB_{mode}")
    
    # histgram for norm of paramb
    for cell,name in enumerate(cellname):
        myPlot.Histgram(winNorms[:,cell],label=f"winNorm_{mode}_{name}",color="royalblue")
        myPlot.Histgram(loseNorms[:,cell],label=f"loseNorm_{mode}_{name}",color="royalblue")
        # vs var.
        myPlot.Histgram(winvsNorms[:,cell],label=f"vswinNorm_{mode}_{name}",color="coral")
        myPlot.Histgram(losevsNorms[:,cell],label=f"vsloseNorm_{mode}_{name}",color="coral")
    
    # save txt for paramb
    np.savetxt(os.path.join("research",f"winB_{mode}.csv"),win100B*1000000,fmt="%.0f",delimiter=",")
    np.savetxt(os.path.join("research",f"loseB_{mode}.csv"),worst100B*1000000,fmt="%.0f",delimiter=",")
    # -------------------------------------------------------------------------

# interval histgram -----------------------------------------------------------
if isInterval:
    # path --------------------------------------------------------------------
    logsPath = os.path.join("research","win","*txt")
    logfiles = glob.glob(logsPath)
    # -------------------------------------------------------------------------
    
    for file in logfiles:
        V,B = myData.loadABLV(file)
        iv_nk,iv_tnk,iv_tk = myData.convV2IntervalData(V)
        
        myPlot.Histgram(iv_nk,label=f"interval_nk",color="coral")
        myPlot.Histgram(iv_tnk,label=f"interval_tnk",color="forestgreen")
        myPlot.Histgram(iv_tk,label=f"interval_tk",color="royalblue")
    
if isLoss:
    # path --------------------------------------------------------------------
    filePath = os.path.join("research","paramB","*txt")
    files = glob.glob(filePath)
    # -------------------------------------------------------------------------
    #pdb.set_trace()
    # reading paramb files ----------------------------------------------------
    atrB = np.loadtxt(files[0])
    orB = np.loadtxt(files[2])
    gtB = np.loadtxt(files[1])
    # -------------------------------------------------------------------------
    CountLoss3(gtB,atrB,orB,mode="or")
    CountLoss3(gtB,orB,atrB,mode="atr500")
    
