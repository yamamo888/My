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


mode = int(sys.argv[1])


# bool --------
isLH = True
isBVTh = True
isLast = True
isRirekierror = True
isnotupdateLast = False # for notupdataPF
isAnima = False
# -------------

# path ----------
logsPath = "logs"
imgPath = "images"
savetxtPath = "savetxt"
firstPath = "nk"
outputPath = f"190_{mode}"
featuresPath = "nankairirekifeature"
txtPath = "*txt"
saveimgPath = "PF"
animaPath = "animaPF"
batFile = "PyToCPF.bat"
# ---------------


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
#nP = 100

# --------------

# likelihood hist plot --------------------------------------------------------  
if isLH:
    print("Start plot histgram.")
    
    savedlhpath = os.path.join(imgPath,f"lh_{mode}")
    lhPath = os.path.join(savetxtPath,f"lh_{mode}")
    # weight path
    wfilePath = os.path.join(lhPath,"w_*")
    wfiles = glob.glob(wfilePath)
    
    for t,wfile in enumerate(wfiles):
        weight = np.loadtxt(wfile)
        
        myPlot.HistLikelihood(weight,path=savedlhpath,label=f"w_{t}",color="black")
    
    # only eq.years ----    
    lhfilePath = os.path.join(lhPath,"lh_*")
    lhfiles = glob.glob(lhfilePath)

    sumlhfilePath = os.path.join(lhPath,"sum_lh_*")
    sumlhfiles = glob.glob(sumlhfilePath)
    
    for t,(lhfile,sumlhfile) in enumerate(zip(lhfiles,sumlhfiles)):
        lh = np.loadtxt(lhfile)
        sum_lh = np.loadtxt(sumlhfile)
     
        myPlot.HistLikelihood(lh[:,ntI],path=savedlhpath,label=f"nk_{t}",color="orange")
        myPlot.HistLikelihood(lh[:,tntI],path=savedlhpath,label=f"tnk_{t}",color="forestgreen")
        myPlot.HistLikelihood(lh[:,ttI],path=savedlhpath,label=f"tk_{t}",color="royalblue")
        myPlot.HistLikelihood(sum_lh,path=savedlhpath,label=f"all_{t}")
    """
    # eq.years & eq.times mode=15 ----
    # path
    glhfilePath = os.path.join(lhPath,"lh_g_*")
    plhfilePath = os.path.join(lhPath,"lh_p_*")
    glhfiles = glob.glob(glhfilePath)
    plhfiles = glob.glob(plhfilePath)
    
    sumglhfilePath = os.path.join(lhPath,"sum_lh_g_*")
    sumglhfiles = glob.glob(sumglhfilePath)
    sumplhfilePath = os.path.join(lhPath,"sum_lh_p_*")
    sumplhfiles = glob.glob(sumplhfilePath)
    
    for t,(glhfile,plhfile,sumglhfile,sumplhfile) in enumerate(zip(glhfiles,plhfiles,sumglhfiles,sumplhfiles)):
        #pdb.set_trace()
        glh = np.loadtxt(glhfile)
        plh = np.loadtxt(plhfile)
        
        sum_glh = np.loadtxt(sumglhfile)
        sum_plh = np.loadtxt(sumplhfile)
        
        # for eq. year error of likelihood
        myPlot.HistLikelihood(glh[:,ntI],label=f"g_nk_{t}",color="orange")
        myPlot.HistLikelihood(glh[:,tntI],label=f"g_tnk_{t}",color="forestgreen")
        myPlot.HistLikelihood(glh[:,ttI],label=f"g_tk_{t}",color="royalblue")
        # for eq. times error of likelihood
        myPlot.HistLikelihood(plh[:,ntI],label=f"p_nk_{t}",color="orange")
        myPlot.HistLikelihood(plh[:,tntI],label=f"p_tnk_{t}",color="forestgreen")
        myPlot.HistLikelihood(plh[:,ttI],label=f"p_tk_{t}",color="royalblue")
        
        # nk + tnk + tk of likelihood
        myPlot.HistLikelihood(sum_glh,label=f"all_g_{t}")
        myPlot.HistLikelihood(sum_plh,label=f"all_p_{t}")
    """
# -----------------------------------------------------------------------------

# b,V,theta Plot --------------------------------------------------------------
if isBVTh:
    
    # [perticles,cell,times]
    Bs = np.zeros([nP,nCell,iS])
    Thetas = np.zeros([nP,nCell,iS])
    Vs = np.zeros([nP,nCell,iS])
    # only first b file
    ffilePath = os.path.join(logsPath,firstPath,txtPath)
    ffiles = glob.glob(ffilePath)
    """
    # first b -----------------------------------------------------------------
    print("Start plot first B..")
    flag = False
    for fID in np.arange(len(ffiles)):
        file = os.path.basename(ffiles[fID])
        _,_,_,tmpB = myData.loadABLV(logsPath,firstPath,file)
        
        tmpB = np.concatenate((tmpB[2,np.newaxis],tmpB[4,np.newaxis],tmpB[5,np.newaxis]),0)
        
        if not flag:
            firstB = tmpB
            flag = True
        else:
            firstB = np.vstack([firstB,tmpB])
    
    minB,maxB = np.min(firstB,0),np.max(firstB,0)
    
    myPlot.scatter3D(firstB[:,ntI],firstB[:,tntI],firstB[:,ttI],rangeP=[minB,maxB],path="PF",title="first B",label="B")
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
    bfiles = glob.glob(os.path.join(savetxtPath,f'B_{mode}',txtPath))
    
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
        
        # save paramters (high rate of perticle)
        maxbpath = os.path.join(savetxtPath,f'maxB_{mode}')
        myData.isDirectory(maxbpath)
        np.savetxt(os.path.join(maxbpath,f'maxB_{iS+1}_{maxrate}.txt'),maxBs*1000000,fmt='%d',delimiter=',')
        # plot 3D heatmap scatter
        s3hpath = os.path.join(imgPath,f'PF_{mode}')
        myPlot.scatter3D_heatmap(updateBs[:,ntI],updateBs[:,tntI],updateBs[:,ttI],ratebs,rangeP=[minB,maxB],path=s3hpath,title=f'mean:{meanB}\n median:{medianB}',label=f'Bheatmap_{iS+1}_{numBs}')
    # -------------------------------------------------------------------------
    
# -----------------------------------------------------------------------------

# last paramter b -------------------------------------------------------------
if isLast:
    """
    In logs/190: only log_8*
    """
    print("Last research....")
    
    for ide in np.arange(9):
        
        # logs 8 var.
        lastlogsPath = os.path.join(logsPath,outputPath,f'log_{ide}_*')
        lastlogs = glob.glob(lastlogsPath)
        
        flag = False
        index = []
        for iS in np.arange(len(lastlogs)):
           file = os.path.basename(lastlogs[iS])
           print(file)
           # Num. of file for sort index
           fID = int(file.split("_")[-1].split(".")[0])
           
           U, th, V, B = myData.loadABLV(logsPath,outputPath,file)
           B = np.concatenate([B[2,np.newaxis],B[4,np.newaxis],B[5,np.newaxis]],0)
           stYear = int(U[0,1]) - 2000
           deltaU, _, _, pJ_all = myData.convV2YearlyData(U,th,V,nYear=10000,cnt=1,stYear=stYear,isLast=True)
           
           # predict eq. [1400,3]
           deltaU = np.concatenate([deltaU[:,2,np.newaxis],deltaU[:,4,np.newaxis],deltaU[:,5,np.newaxis]],1)
           pred = [pJ_all[ntI]-int(U[0,1]),pJ_all[tntI]-int(U[0,1]),pJ_all[ttI]-int(U[0,1])]
           gt = [np.where(gtV[:,ntI]>0)[0],np.where(gtV[:,tntI]>0)[0],np.where(gtV[:,ttI]>0)[0]]
           #pdb.set_trace()
           if ide == 8:
               # plot & mae eq. of predict & gt
               rpath = os.path.join(imgPath,f'rireki_{mode}')   
               maxSim = myPlot.Rireki(gt,pred,path=rpath,label=f"{iS}_{np.round(B[ntI],6)}_{np.round(B[tntI],6)}_{np.round(B[ttI],6)}",isResearch=True)
           else:
               maxSim = myData.MAEyear(gt,pred)
               #pdb.set_trace()
            
           index = np.append(index,fID)
           
           if not flag:
               maxSims = maxSim
               flag = True
           else:
               maxSims = np.hstack([maxSims,maxSim])
        
        sortInd = np.argsort(maxSims)
        sort_maxSims = maxSims[sortInd]
        sortfID = index[sortInd]
        statis = []
        # statistics or maxSims
        meanDS = np.mean(maxSims)
        minDS,maxDS = np.min(maxSims),np.max(maxSims)
        statis = np.append(statis,[meanDS,minDS,maxDS])
        # save sort mae & index
        maepath = os.path.join(savetxtPath,f'mae_{mode}')
        myData.isDirectory(maepath)
        np.savetxt(os.path.join(maepath,f"maxSims_{ide}.txt"),sort_maxSims,fmt=f"%d")
        np.savetxt(os.path.join(maepath,f"index_{ide}.txt"),sortInd,fmt=f"%d")       
        np.savetxt(os.path.join(maepath,f'statis_{ide}.txt'),statis,fmt=f"%d")
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
if isRirekierror:
    
    # saved maxSims file path
    boxpath = os.path.join(savetxtPath,f'mae_{mode}','maxSims_*')
    boxfullpath = glob.glob(boxpath)
    
    flag = False
    for path in boxfullpath:
        # loading maxsim
        maxsim = np.loadtxt(path)
        
        if not flag:
            maxsims = maxsim
            flag = True
        else:
            # [id,perticles]
            maxsims = np.vstack([maxsims,maxsim])
    
    # Save
    boxpath = os.path.join(imgPath,f'box_{mode}')
    myData.isDirectory(boxpath)
    myPlot.BoxPlot(maxsims.T,path=boxpath,label=f'box_{mode}')

# -----------------------------------------------------------------------------    
if isnotupdateLast:

    lastlogsPath = os.path.join(logsPath,outputPath,txtPath)
    lastlogs = glob.glob(lastlogsPath)
    
    flag = False
    index = []
    for iS in np.arange(len(lastlogs)):
       file = os.path.basename(lastlogs[iS])
       print(file)
       # Num. of file for sort index
       fID = int(file.split("_")[-1].split(".")[0])
       
       U, th, V, B = myData.loadABLV(logsPath,outputPath,file)
       B = np.concatenate([B[2,np.newaxis],B[4,np.newaxis],B[5,np.newaxis]],0)
       
       yU, yth, yV, pJ_all = myData.convV2YearlyData(U,th,V,nYear=10000,cnt=0)
       yU, yth, yV, pJ_all, maxSim, sYear = myData.MSErrorNankai(gtV,yU,yth,yV,pJ_all,nCell=nCell)
       pred = [pJ_all[ntI],pJ_all[tntI],pJ_all[ttI]]
       gt = [np.where(gtV[:,ntI]>0)[0],np.where(gtV[:,tntI]>0)[0],np.where(gtV[:,ttI]>0)[0]]
       
       # plot & mae eq. of predict & gt
       rpath = os.path.join(imgPath,'rireki')
       maxSim = myPlot.Rireki(gt,pred,path=rpath,label=f"{iS}_{np.round(B[ntI],6)}_{np.round(B[tntI],6)}_{np.round(B[ttI],6)}",isResearch=True)
# -----------------------------------------------------------------------------    
       
# animate image of param b ----------------------------------------------------
if isAnima:
    gifPath = os.path.join(imgPath,animaPath,"*png")
    myPlot.gif2Animation(gifPath,"paramB")
# -----------------------------------------------------------------------------