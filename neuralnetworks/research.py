# -*- coding: utf-8 -*-

import os
import glob

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

import PlotPF as myPlot


# path ------------------------------------------------------------------------
filePath = os.path.join("research","paramB","*txt")
files = glob.glob(filePath)
# -----------------------------------------------------------------------------
#pdb.set_trace()
# reading paramb files --------------------------------------------------------
atrB = np.loadtxt(files[0])
orB = np.loadtxt(files[2])
gtB = np.loadtxt(files[1])
# -----------------------------------------------------------------------------

# parameter -------------------------------------------------------------------
cellname = ["nk","tnk","tk"]
ntI,tntI,ttI = 0,1,2
# -----------------------------------------------------------------------------

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
    
    # pred2 win & lose norm of paramb for histgram
    winNorm = norm2[mode12,:]
    loseNorm = norm2[np.logical_not(mode12),:]
    
    # best & worst 1% index
    bestInd = np.argsort(winNorm)[:750]
    worstInd = np.argsort(loseNorm)[::-1][:750]
    # 100 var.
    best100Ind = np.argsort(winNorm)[:100]
    worst100Ind = np.argsort(loseNorm)[::-1][:100]
    
    # pred2 win & lose paramb for scatter
    winB = pred2[bestInd,:]
    loseB = pred2[worstInd,:]
    # 100 var.
    win100B = pred2[best100Ind,:]
    worst100B = pred2[worst100Ind,:]
    
    # pred1 win & lose norm of paramb for histgram
    winvsNorm = norm1[mode12,:]
    losevsNorm = norm1[np.logical_not(mode12),:]
  
    
    
    
    # -------------------------------------------------------------------------
    # range of paramb for plot
    minB,maxB = np.min(pred2,0),np.max(pred2,0)
    
    # scatter for paramb
    myPlot.scatter3D(winB[:,ntI],winB[:,tntI],winB[:,ttI],rangeB=[minB,maxB],title=f"Win {mode}",label=f"winB_{mode}")
    myPlot.scatter3D(loseB[:,ntI],loseB[:,tntI],loseB[:,ttI],rangeB=[minB,maxB],title=f"Lose {mode}",label=f"loseB_{mode}")
    
    # histgram for norm of paramb
    for cell,name in enumerate(cellname):
        myPlot.Histgram(winNorm[:,cell],label=f"winNorm_{mode}_{name}",color="coral")
        myPlot.Histgram(loseNorm[:,cell],label=f"loseNorm_{mode}_{name}",color="coral")
        # vs var.
        myPlot.Histgram(winvsNorm[:,cell],label=f"winvsNorm_{mode}_{name}",color="royalblue")
        myPlot.Histgram(losevsNorm[:,cell],label=f"losevsNorm_{mode}_{name}",color="royalblue")
        
    # save txt for paramb
    np.savetxt(os.path.join("research","paramB",f"winB_{mode}.txt"),win100B)
    np.savetxt(os.path.join("research","paramB",f"loseB_{mode}.txt"),worst100B)
    # -------------------------------------------------------------------------
    
# call ------------------------------------------------------------------------
CountLoss3(gtB,orB,atrB,mode="atr500")