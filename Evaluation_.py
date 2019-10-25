import numpy as np
import pdb
from scipy import stats
import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
import time
import pandas as pd
import seaborn as sns
import os
import sys
import glob


# ---------------------------- Path ------------------------------

visualPath = "visualization"
scatterPath = "scatter"
resultsPath = "results"
savedPklPath = "nankaipickles"

# -----------------------------------------------------------------

# -----------------------------------------------------------------
def Ttest(pred1,pred2):
    size = int(pred1.shape[0])
    #size = int(pred1.shape[0]/10)
    pnum = 0.03
    for cnt in np.arange(int(pred1.shape[0]/size)):
        sInd = size * cnt
        eInd = sInd + size
        #pdb.set_trace()
        t1,p1 = stats.ttest_rel(pred2[sInd:eInd],pred1[sInd:eInd])

        if p1<pnum:
            print("有意")
            print("p1:%f"%p1)
        else:
            print('有意でない')
            print("p1:%f"%p1)
    
        time.sleep(1)
# -----------------------------------------------------------------
def BasicStatistics(gt,pred,mode="none"):
    # mean evary cell
    loss = np.mean(np.square(gt-pred),1)
    
    print("{}".format(mode))
    print("MSE: {}".format(np.mean(loss)))
    print("Percentiles 95%, 75%, 50%, 25%, 5%: {}".format(np.percentile(loss,[95,75,50,25,5])))

    print("--------------------------------")
# -----------------------------------------------------------------
def BottomScatter(gt,pred1,pred2,mode="none",cellname="none",savefilePath="test.png"):
    
    # 1. pred1 >  pred2 ---> red
    # 2. pred1 <= pred2 ---> blue
    # if stand OR, many red is good
    residual12 = abs(gt-pred1) - abs(gt-pred2)
    
    fig = plt.figure(figsize=(9,6))    
    # 45° line
    line = np.arange(np.min(gt),np.max(gt)+0.001,0.001)

    # many plot
    if  gt.shape[0] > 10:
        
        for num in np.arange(gt.shape[0]):
            plt.plot(gt[num],pred1[num],color="black",marker="${}$".format(str(num)),linestyle="None",ms=10)
            # scatter ancor-based or atr-nets
            plt.plot(gt[num],pred2[num],color="red",marker="${}$".format(str(num)),linestyle="None",ms=10)
        
    else:
        #pdb.set_trace()
        
        colors = []
        for res in residual12:
            if res < 0.000000000:
                colors.append("blue")
            elif res > 0.0000000000:
                colors.append("red")
        
        for num,(color,mark) in enumerate(colors):
            # scatter oridinary
            #pdb.set_trace()
            plt.plot(gt[num],pred1[num],color="black",marker="${}$".format(str(num)),linestyle="None",ms=10)
            # scatter ancor-based or atr-nets
            plt.plot(gt[num],pred2[num],color=color,marker="${}$".format(str(num)),linestyle="None",ms=10)
    # line
    plt.plot(line,line,"-",color="gold",linewidth=2)
            
    plt.xlabel('ground truth',fontsize=18)
    plt.ylabel('predict',fontsize=18)
                
    plt.xlim([np.min(gt),np.max(gt)])
    
    plt.title("{} in {} stand {}".format(mode,cellname,standName)) 
    fig.subplots_adjust(left=0.2,bottom=0.2)
    plt.tick_params(labelsize=18)
    plt.legend()
    
    savePath = os.path.join("{}.png".format(savefilePath))    
    plt.savefig(savePath)
        
    plt.close()
    
    # save txt
    strgtall = str(gtall)
    strall1 = str(all1)
    strall2 = str(all2)
    
    f = open("{}.txt".format(savefilePath),"a")
    f.write("gt:" + strgtall + "\n")
    f.write("OR:" + strall1 + "\n")
    f.write("ANC(ATR):" + strall2 + "\n")
    f.close()


# ------------------------ loss  -----------------------------------
def PlotLoss(data,fileName="test.png"):
    data = data[10:]
    plt.plot(data,linewidth=5.0)
    plt.ylim([min(data),0.0000005])

    plt.savefig(fileName)
    plt.close()
# -----------------------------------------------------------------

# ----------------- Call Bottom Scatter----------------------------

# ----------------------- Path -------------------------------------
# number of bottom data
bottomNum = int(sys.argv[1])
# loading pkl (number of step)
fileNum = int(sys.argv[2])
# 0: stand OR, 1: stand ATR, 2: stand ATR class, 3: stand OR, 4: stand OR class
standMode = int(sys.argv[3])
# -----------------------------------------------------------------

# dir path
pklPath = os.path.join(resultsPath,savedpklPath)
# file path
filePath = "test_{}_*"
# full path
pklfullPath = glob.glob(os.path.join(pklPath,filePath))

pdb.set_trace()
# loading OR pkl data
with open(os.path.join(pklPath,pklfullPath[0]),"rb") as fp:
    teY = pickle.load(fp)
    predY_or = pickle.load(fp)
    teX_or = pickle.load(fp)
    _ = pickle.load(fp)
    _ = pickle.load(fp)
    trloss_or = pickle.load(fp)
    teloss_or = pickle.load(fp)

# loading Anc pkl data
with open(os.path.join(pklPath,pklnames[1]),"rb") as fp:
    teY = pickle.load(fp)
    predY_anc = pickle.load(fp)
    teX_anc = pickle.load(fp)
    predCls_anc = pickle.load(fp)
    predRes_anc = pickle.load(fp)
    _ = pickle.load(fp)
    _ = pickle.load(fp)
    trclsloss_anc = pickle.load(fp)
    teclsloss_anc = pickle.load(fp)
    trresloss_anc = pickle.load(fp)
    teresloss_anc = pickle.load(fp)
    trloss_anc = pickle.load(fp)
    teloss_anc = pickle.load(fp)

# loading ATR pkl data
with open(os.path.join(pklPath,pklnames[2]),"rb") as fp:
    teY = pickle.load(fp)
    predY_atr = pickle.load(fp)
    teX_atr = pickle.load(fp)
    predCls_atr = pickle.load(fp)
    predRes_atr = pickle.load(fp)
    _ = pickle.load(fp)
    _ = pickle.load(fp)
    trclsloss_atr = pickle.load(fp)
    teclsloss_atr = pickle.load(fp)
    trresloss_atr = pickle.load(fp)
    teresloss_atr = pickle.load(fp)
    trloss_atr = pickle.load(fp)
    teloss_atr = pickle.load(fp)

pdb.set_trace()

if standMode == 0:
    # stand atr
    ind_atr = np.argsort(np.mean(np.square(teY-predY_or),1))[::-1][:bottomNum]
    standName = "OR"

elif standMode == 1:
    # stand atr
    ind = np.argsort(np.mean(np.square(teY-predY_atr),1))[::-1][:bottomNum]
    standName = "ATR(cls+reg)"

elif standMode == 2:
    # stand atr
    ind = np.argsort(np.mean(np.square(teY-predCls_atr),1))[::-1][:bottomNum]
    standName = "ATR(cls)"

elif standMode == 3:
    # stand atr
    ind = np.argsort(np.mean(np.square(teY-predY_anc),1))[::-1][:bottomNum]
    standName = "ANC(cls+reg)"

elif standMode == 4:
    # stand atr
    ind = np.argsort(np.mean(np.square(teY_atr-predCls_anc),1))[::-1][:bottomNum]
    standName = "ANC(cls)"

pdb.set_trace()

cellName = ["nk","tnk","tk"]
for i,name in enumerate(cellName):
    BottomScatter(teY[ind,i],predY1[ind,i],predY2[ind,i],mode="OR vs {}".format(standName),cellname=name,savefilePath="ORANC{}_{}_{}".format(name,bottomNum,standName))
    
# ------------ Call Mean & Percentiles -------------------

BasicStatistics(teY,predY_or,mode="OR")
BasicStatistics(teY,predY_anc,mode="Anc(cls+reg)")
BasicStatistics(teY,predY_atr,mode="Atr(cls+reg)")
BasicStatistics(teY,predCls_anc,mode="Anc(cls)")
BasicStatistics(teY,predCls_atr,mode="Atr(cls)")

# ----------------- Call loss ----------------------------
"""
# Ordinary 
PlotLoss(trloss_or,"trloss_or.png")
PlotLoss(teloss_or,"teloss_or.png")
# Anchor-based
PlotLoss(trclsloss_an,"trclsloss_an.png")
PlotLoss(teclsloss_an,"teclsloss_an.png")
PlotLoss(trresloss_an,"trresloss_an.png")
PlotLoss(teresloss_an,"teresloss_an.png")
PlotLoss(trloss_an,"trloss_an.png")
PlotLoss(teloss_an,"teloss_an.png")
# ATR-Nets
PlotLoss(trclsloss_at,"trclsloss_at.png")
PlotLoss(teclsloss_at,"teclsloss_at.png")
PlotLoss(trresloss_at,"trresloss_at.png")
PlotLoss(teresloss_at,"teresloss_at.png")
PlotLoss(trloss_at,"trloss_at.png")
PlotLoss(teloss_at,"teloss_at.png")
PlotLoss(trloss_at,"trloss_at.png")
PlotLoss(teloss_at,"teloss_at.png")

pdb.set_trace()
"""
# -------- Call scatter -------------------------------------------
"""
filepath1 = "*_1.pkl"
resultsPath1 = os.path.join(results,pickles,filepath1)
file1 = glob.glob(resultsPath1)

# loading pkl 
with open(file1[0],"rb") as fp:
    _ = pickle.load(fp)
    _ = pickle.load(fp)
    te_gt1 = pickle.load(fp)
    te_pred1 = pickle.load(fp)
# select data
te_gt1 = te_gt1[:700]
te_pred1 = te_pred1[:700]

for cellInd in range(3):
    #axes.append(fig.add_subplot(1,3,cellInd))

    fig = plt.figure(figsize=(9,6))

    line = np.arange(np.min(te_gt1),np.max(te_gt1)+0.001,0.001)
    # scatter
    plt.plot(te_gt1[:,cellInd],te_pred1[:,cellInd],".",color="black",linestyle="None",ms=10)
    #plt.plot(te_gt2[:,cellInd],te_pred2[:,cellInd],".",color="c",linestyle="None",label="ATR-Nets")
    # line
    plt.plot(line,line,"-",color="red",linewidth=4)
        
    plt.xlabel('ground truth',fontsize=22)
    plt.ylabel('predict',fontsize=22)
        
    plt.ylim([np.min(te_gt1),np.max(te_gt1)])
    plt.xlim([np.min(te_gt1),np.max(te_gt1)])
    #mpl.rcParams["axes.xmargin"]  = 1
    #mpl.rcParams["axes.ymargin"]  = 1
    fig.subplots_adjust(left=0.2,bottom=0.2)
    #plt.legend(loc="best")
    plt.tick_params(labelsize=22)

    savePath = os.path.join("Scatter01_{}.png".format(cellInd))    
    plt.savefig(savePath)

    plt.close()
"""
"""
# ------------ Call t-test -------------------------------------
"""
file1 = file1[0]
files2 = files2[1:]
pdb.set_trace()
for file2 in files2:
    print(file2)
    print(file1)
    
    with open(file2,"rb") as fp:
        _ = pickle.load(fp)
        _ = pickle.load(fp)
        te_gt2 = pickle.load(fp)
        te_pred2 = pickle.load(fp)
    
    with open(file1,"rb") as fp:
        _ = pickle.load(fp)
        _ = pickle.load(fp)
        te_gt1 = pickle.load(fp)
        te_pred1 = pickle.load(fp)

    Ttest(te_pred1,te_pred2)
    print("--------------------------------")
"""
