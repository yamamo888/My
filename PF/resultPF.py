# -*- coding: utf-8 -*-

import os
import sys
import glob
import pickle
import pdb
import time
import collections

import seaborn as sns
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import makingDataPF as myData
import PlotPF as myPlot

# bool --------
isLH = True
isBVTh = True
isLast = True # for notupdataPF
isAnima = False
# -------------

# path ----------
logsPath = "logs"
imgPath = "images"
savetxtPath = "savetxt"
paramPath = "parFile"
outputPath = "190"
featuresPath = "nankairirekifeature"
txtPath = "*txt"
saveimgPath = "PF"
animaPath = "animaPF"
batFile = "PyToCPF.bat"
# ---------------

filePath = os.path.join(paramPath,outputPath,txtPath)
files = glob.glob(filePath)

with open(os.path.join(featuresPath,"nankairireki.pkl"), "rb") as fp:
    nkfiles = pickle.load(fp)
gtV = nkfiles[190,:,:]
        
# parameter ----
Sfl = 4
Efl = 12

ntI,tntI,ttI = 0,1,2
nYear = 1400
nCell = 3
# Num. of assimulation times
iS = 9
# Num. of perticles
nP = 501
# --------------

# likelihood hist plot --------------------------------------------------------  
if isLH:
    #lhfilePath = os.path.join(savetxtPath,"lh","lh_*")
    #lhfiles = glob.glob(lhfilePath)
    
    glhfilePath = os.path.join(savetxtPath,"lh","lh_g_*")
    plhfilePath = os.path.join(savetxtPath,"lh","lh_p_*")
    glhfiles = glob.glob(glhfilePath)
    plhfiles = glob.glob(plhfilePath)
    
    sumlhfilePath = os.path.join(savetxtPath,"lh","sum_lh_*")
    sumlhfiles = glob.glob(sumlhfilePath)
    
    print("Start plot histgram.")
    """
    for t,(lhfile,sumlhfile) in enumerate(zip(lhfiles,sumlhfiles)):
        lh = np.loadtxt(lhfile)
        myPlot.HistLikelihood(lh[:,ntI],label=f"nk_{t}",color="orange")
        myPlot.HistLikelihood(lh[:,tntI],label=f"tnk_{t}",color="forestgreen")
        myPlot.HistLikelihood(lh[:,ttI],label=f"tk_{t}",color="royalblue")
        myPlot.HistLikelihood(sum_lh,label=f"all_{t}")
    """ 
    
    for t,(glhfile,plhfile,sumlhfile) in enumerate(zip(glhfiles,plhfiles,sumlhfiles)):
        #pdb.set_trace()
        glh = np.loadtxt(glhfile)
        plh = np.loadtxt(plhfile)
        
        sum_lh = np.loadtxt(sumlhfile)
        # for eq. year error of likelihood
        myPlot.HistLikelihood(glh[:,ntI],label=f"g_nk_{t}",color="orange")
        myPlot.HistLikelihood(glh[:,tntI],label=f"g_tnk_{t}",color="forestgreen")
        myPlot.HistLikelihood(glh[:,ttI],label=f"g_tk_{t}",color="royalblue")
        # for eq. times error of likelihood
        myPlot.HistLikelihood(plh[:,ntI],label=f"p_nk_{t}",color="orange")
        myPlot.HistLikelihood(plh[:,tntI],label=f"p_tnk_{t}",color="forestgreen")
        myPlot.HistLikelihood(plh[:,ttI],label=f"p_tk_{t}",color="royalblue")
        
        # nk + tnk + tk of likelihood
        myPlot.HistLikelihood(sum_lh,label=f"all_{t}")
# -----------------------------------------------------------------------------

# b,V,theta Plot --------------------------------------------------------------
if isBVTh:
    
    # [perticles,cell,times]
    Bs = np.zeros([nP,nCell,iS])
    Thetas = np.zeros([nP,nCell,iS])
    Vs = np.zeros([nP,nCell,iS])
    # only first b file
    ffilePath = os.path.join(logsPath,"bzero",txtPath)
    ffiles = glob.glob(ffilePath)
    """
    # first b -----------------------------------------------------------------
    print("Start plot first B..")
    flag = False
    for fID in np.arange(len(ffiles)):
        file = os.path.basename(ffiles[fID])
        _,_,_,tmpB = myData.loadABLV(logsPath,"bzero",file)
        
        tmpB = np.concatenate((tmpB[2,np.newaxis],tmpB[4,np.newaxis],tmpB[5,np.newaxis]),0)
        
        if not flag:
            firstB = tmpB
            flag = True
        else:
            firstB = np.vstack([firstB,tmpB])
    
    minB,maxB = np.min(firstB,0),np.max(firstB,0)
    pdb.set_trace()
    #myPlot.scatter3D(firstB[:,ntI],firstB[:,tntI],firstB[:,ttI],rangeP=[minB,maxB],path="PF",title="first B",label="B")
    # -------------------------------------------------------------------------
    """
    """
    # -------------------------------------------------------------------------
    print("Start plot next B...")
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
    minBs,maxBs = np.min(Bs,0),np.max(Bs,0)
    
    minTheta,maxTheta = np.min(minThetas,1),np.max(maxThetas,1)
    minV,maxV = np.min(minVs,1),np.max(maxVs,1)
    minB,maxB = np.min(minBs,1),np.max(maxBs,1)
    #pdb.set_trace()
    for iS in np.arange(Bs.shape[-1]):
        # Scatter B, theta, V
        myPlot.scatter3D(Bs[:,ntI,iS],Bs[:,tntI,iS],Bs[:,ttI,iS],rangeP=[minB,maxB],path="PF",title=f"B {iS}times",label=f"B_{iS}")
        myPlot.scatter3D(Thetas[:,ntI,iS],Thetas[:,tntI,iS],Thetas[:,ttI,iS],rangeP=[minTheta,maxTheta],path="PF",title=f"Theta {iS}times",label=f"Theta_{iS}")
        myPlot.scatter3D(Vs[:,ntI,iS],Vs[:,tntI,iS],Vs[:,ttI,iS],rangeP=[minV,maxV],path="PF",title=f"V {iS}times",label=f"V_{iS}")
    # -------------------------------------------------------------------------
    """
    # -------------------------------------------------------------------------
    # reading B file
    bfiles = glob.glob(os.path.join(savetxtPath,'B',txtPath))
    
    for iS in np.arange(9):
        # [perticle,cell]
        bs = np.loadtxt(bfiles[iS])
        Bs[:,:,iS] = bs    
    
    minB,maxB = np.min(np.min(Bs,0),1),np.max(np.max(Bs,0),1)
    
    for iS in np.arange(Bs.shape[-1]):    
        seen = []
        # stand of paramb for sampling index (not overlapping)
        standB = [x for x in Bs[:,:,iS].tolist() if x not in seen and not seen.append(x)]
        
        # Get Num. of matching params ----
        ratebs = []
        for ind in np.arange(len(standB)):
            # rate of parameter b
            rateb = np.where(np.all(standB[ind] == Bs[:,:,iS], axis=1))[0].shape[0]    
            ratebs = np.hstack([ratebs,rateb])
        
        #pdb.set_trace()    
        # update paramb 
        updateBs = np.array([sb for sb in standB])
        # plot
        meanB,medianB = np.mean(updateBs,0),np.median(updateBs,0)
        # Num.of perticle for label
        numBs = updateBs.shape[0]
        # index max perticle
        maxBsind = [i for i,x in enumerate(ratebs) if x == max(ratebs)]
        # max perticle
        maxBs = np.array([np.array(updateBs[ind]) for ind in maxBsind])
        # rate of max perticle
        maxrate = int(np.max(ratebs[maxBsind]))
        #pdb.set_trace()
        # save paramters (high rate of perticle)
        np.savetxt(os.path.join(savetxtPath,'maxB',f'maxB_{iS+1}_{maxrate}.txt'),maxBs*1000000,fmt='%d',delimiter=',')
        myPlot.scatter3D_heatmap(updateBs[:,ntI],updateBs[:,tntI],updateBs[:,ttI],ratebs,rangeP=[minB,maxB],path='PF',title=f'mean:{meanB}\n median:{medianB}',label=f'Bheatmap_{iS+1}_{numBs}')
    # -------------------------------------------------------------------------
    
# -----------------------------------------------------------------------------

# last paramter b -------------------------------------------------------------
if isLast:
    """
    1. there are no log file.
    2. ParamFilePF.csv is written for 8 var.
    3. output logs/190
    """
    print("Last research....")
    """
    # Start making log files --------------------------------------------------
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
    # -------------------------------------------------------------------------    
    """
    # logs 8 var.
    lastlogsPath = os.path.join(logsPath,outputPath,txtPath)
    lastlogs = glob.glob(lastlogsPath)
    
    flag = False
    index = []
    for iS in np.arange(nP):
       file = os.path.basename(lastlogs[iS])
       print(file)
       # Num. of file for sort index
       fID = int(file.split("_")[-1].split(".")[0])
       
       U, th, V, B = myData.loadABLV(logsPath,outputPath,file)
       B = np.concatenate([B[2,np.newaxis],B[4,np.newaxis],B[5,np.newaxis]],0)
       stYear = int(U[0,1]) - 2000
       deltaU, _, _, pJ_all = myData.convV2YearlyData(U,th,V,nYear=10000,cnt=1,stYear=stYear)
       #pdb.set_trace()
       # predict eq.
       deltaU = np.concatenate([deltaU[:,2,np.newaxis],deltaU[:,4,np.newaxis],deltaU[:,5,np.newaxis]],1)
       pred = [pJ_all[ntI]-int(U[0,1]),pJ_all[tntI]-int(U[0,1]),pJ_all[ttI]-int(U[0,1])]
       gt = [np.where(gtV[:,ntI]>0)[0],np.where(gtV[:,tntI]>0)[0],np.where(gtV[:,ttI]>0)[0]]
       # plot & mae eq. of predict & gt
       maxSim = myPlot.Rireki(gt,pred,path="rireki",label=f"{iS}_{np.round(B[ntI],6)}_{np.round(B[tntI],6)}_{np.round(B[ttI],6)}",isResearch=True)

       index = np.append(index,fID)
       if not flag:
           maxSims = maxSim
           flag = True
       else:
           maxSims = np.hstack([maxSims,maxSim])
    
    sortInd = np.argsort(maxSims)
    sort_maxSims = maxSims[sortInd]
    sortfID = index[sortInd]
    
    # save sort mae & index
    np.savetxt(os.path.join(savetxtPath,'mae',f"maxSims.txt"),sort_maxSims,fmt=f"%d")
    np.savetxt(os.path.join(savetxtPath,'mae',f"index.txt"),sortInd,fmt=f"%d")       
# -----------------------------------------------------------------------------

# animate image of param b ----------------------------------------------------
if isAnima:
    gifPath = os.path.join(imgPath,animaPath,"*png")
    myPlot.gif2Animation(gifPath,"paramB")
# -----------------------------------------------------------------------------