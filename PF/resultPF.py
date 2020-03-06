# -*- coding: utf-8 -*-

import os
import sys
import glob
import pickle
import pdb

import seaborn as sns
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import makingData as myData
import PlotPF as myPlot

# bool --------
isLH = True
isBVTh = False
# -------------

# path ----------
logsPath = "logs"
imgPath = "images"
savetxtPath = "savetxt"
paramPath = "parFile"
dirPath = "last"
outputPath = "190"
featuresPath = "nankairirekifeature"
txtPath = "*txt"
saveimgPath = "PF"
# ---------------

#filePath = os.path.join(logsPath,dirPath,txtPath)
filePath = os.path.join(paramPath,outputPath,txtPath)
files = glob.glob(filePath)

# parameter ----
nCell = 8
Sfl = 4
Efl = 12
limitNum = 6
ntI,tntI,ttI = 0,1,2
nYear = 1400
slip = 1
iS = 5
# --------------

# likelihood hist plot --------------------------------------------------------  
if isLH:
    lhfilePath = os.path.join(savetxtPath,"lh","lh_*")
    sumlhfilePath = os.path.join(savetxtPath,"lh","sum_lh_*")
    lhfiles = glob.glob(lhfilePath)
    sumlhfiles = glob.glob(sumlhfilePath)
    
    for t,(lhfile,sumlhfile) in enumerate(zip(lhfiles,sumlhfiles)):
        lh = np.loadtxt(lhfile)
        sum_lh = np.loadtxt(sumlhfile)
         
        myPlot.HistLikelihood(lh[:,ntI],label=f"nk_{t}",color="skyblue")
        myPlot.HistLikelihood(lh[:,tntI],label=f"tnk_{t}",color="forestgreen")
        myPlot.HistLikelihood(lh[:,ttI],label=f"tk_{t}",color="coral")
        # nk + tnk + tk of likelihood
        myPlot.HistLikelihood(sum_lh,label=f"all_{t}")
# -----------------------------------------------------------------------------

# b,V,theta Plot --------------------------------------------------------------
if isBVTh:
    
    # [perticles,cell,times]
    Bs = np.zeros([499,3,iS])
    Thetas = np.zeros([499,3,iS])
    Vs = np.zeros([499,3,iS])
    # only first b file
    ffilePath = os.path.join(logsPath,"bzero",txtPath)
    ffiles = glob.glob(ffilePath)
    
    flag = False
    for fID in np.arange(iS):
        file = os.path.basename(ffiles[fID])
        _,_,_,tmpB = myData.loadABLV(logsPath,"bzero",file)
        
        tmpB = np.concatenate((tmpB[2,np.newaxis],tmpB[4,np.newaxis],tmpB[5,np.newaxis]),0)
        
        if not flag:
            B = tmpB
            flag = True
        else:
            B = np.vstack([B,tmpB])
    
    minB,maxB = np.min(B,0),np.max(B,0)
    # first b
    myPlot.scatter3D(B[:,ntI],B[:,tntI],B[:,ttI],rangeP=[minB,maxB],title="first B",label="first_B")
    # -------------------------------------------------------------------------
    
    flag2 = False
    for iS in np.arange(Bs.shape[0]):
            
        parfiles = [s for s in files if "parfileHM{}_".format(iS) in s]
        
        pi = 0
        for parfile in parfiles:
            with open(parfile,"r") as fp:
                alllines = fp.readlines()
                # parfileHM031の改行コード削除
                alllines = [alllines[i].strip().split(",") for i in np.arange(len(alllines))]
                flag = False
                lines = alllines[Sfl:Efl]
             
                for nl in [2,4,5]:
                    # B, theta, V
                    inlines = np.array([lines[nl]]).astype(float).T
                    tmpB = inlines[1]
                    tmpTheta = inlines[-2]
                    tmpV = inlines[-1]
                    if not flag:
                        B = tmpB
                        Theta = tmpTheta
                        V = tmpV
                        flag = True
                    else:
                        B = np.hstack([B,tmpB])
                        Theta = np.hstack([Theta,tmpTheta])
                        V = np.hstack([V,tmpV])
                
                Bs[pi,:,iS] = B
                Thetas[pi,:,iS] = Theta
                Vs[pi,:,iS] = V
                pi += 1
         
    # min & max [3(cell),7(files)]
    minThetas,maxThetas = np.min(Thetas,0),np.max(Thetas,0)
    minVs,maxVs = np.min(Vs,0),np.max(Vs,0)
    #minBs,maxBs = np.min(Bs,0),np.max(Bs,0)
    
    minTheta,maxTheta = np.min(minThetas,1),np.max(maxThetas,1)
    minV,maxV = np.min(minVs,1),np.max(maxVs,1)
    #minB,maxB = np.min(minBs,1),np.max(maxBs,1)
    
    for iS in np.arange(Bs.shape[-1]):
        # Scatter B, theta, V
        myPlot.scatter3D(Bs[:,ntI,iS],Bs[:,tntI,iS],Bs[:,ttI,iS],rangeP=[minB,maxB],title=f"B {iS}times",label=f"B_{iS}")
        myPlot.scatter3D(Thetas[:,ntI,iS],Thetas[:,tntI,iS],Thetas[:,ttI,iS],rangeP=[minTheta,maxTheta],title=f"Theta {iS}times",label=f"Theta_{iS}")
        myPlot.scatter3D(Vs[:,ntI,iS],Vs[:,tntI,iS],Vs[:,ttI,iS],rangeP=[minV,maxV],title=f"V {iS}times",label=f"V_{iS}")
# -----------------------------------------------------------------------------