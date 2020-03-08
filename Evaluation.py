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
from natsort import natsorted
import subprocess

# ---------------------------- Path ------------------------------
visualPath = "visualization"
scatterPath = "scatter"
resultsPath = "results"
savedpklPath = "nankaipicklessoft"
savedpklPath = "nankaipickles"
# -----------------------------------------------------------------

# -----------------------------------------------------------------
def Ttest(pred1,pred2):
    size = int(pred1.shape[0])
    #size = int(pred1.shape[0]/10)
    pnum = 0.03
    for cnt in np.arange(int(pred1.shape[0]/size)):
        sInd = size * cnt
        eInd = sInd + size
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
    
    # under 5% or 25% mean square loss 
    under5loss = np.sort(loss)[::-1][:int(loss.shape[0]*0.05)]
    under25loss = np.sort(loss)[::-1][:int(loss.shape[0]*0.25)]

    # gt mean, shape=[cell(=3),]
    gtmean = np.mean(gt,0)
    w1 = np.mean(np.sum(np.square(gt-pred),0))
    w2 = np.mean(np.sum(np.square(gt-gtmean),0))
     
    print("{}".format(mode))
    print("MSE: {}".format(np.mean(loss)))
    print("under5% MSE: {}".format(np.mean(under5loss)))
    print("under25% MSE: {}".format(np.mean(under25loss)))
    print("R2: {}".format(1-(w1/w2)))
    #print("RMSE: {}".format(np.square(np.mean(loss))))
    #print("Percentiles 95%, 75%, 50%, 25%, 5%: {}".format(np.percentile(loss,[95,75,50,25,5])))
    print("--------------------------------")
# -----------------------------------------------------------------
#def BottomScatter(gt,pred1,pred2,mode="none",cellname="none",savefilePath="none",standName="none",vsName="none",gName="none"):
def BottomScatter(gtY,predY1,predY2,mode="none"):
    
    # 1. pred1 >  pred2 ---> red
    # 2. pred1 <= pred2 ---> blue
    # if stand OR, many red is good
    #residual12 = abs(gt-pred1) - abs(gt-pred2)    

    
    for cell in np.arange(3):
        gt = gtY[:,cell]
        pred1 = predY1[:,cell] 
        pred2 = predY2[:,cell] 
        
        loss = np.square(gt-pred2)
        under5inds = np.argsort(loss)[::-1][:int(loss.shape[0]*0.05)]
        under25inds = np.argsort(loss)[::-1][:int(loss.shape[0]*0.01)]
        
        gt = gt[under5inds]
        pred2 = pred2[under5inds]
        
        #gt = gt[under25inds]
        #pred2 = pred2[under25inds]
        
        # 45° line
        line = np.arange(np.min(gt),np.max(gt)+0.001,0.001)
        
        # many plot
        if gt.shape[0] > 10:
             
            for num in np.arange(gt.shape[0]):
                plt.plot(gt[num],pred2[num],color="black",marker="o",linestyle="None",ms=5)
        else:
            colors = []
            for res in residual12:
                if res < 0.000000000:
                    colors.append("blue")
                elif res > 0.0000000000:
                    colors.append("red")
            
            for num,color in enumerate(colors):
                # scatter oridinary
                plt.plot(gt[num],pred1[num],color="black",marker="${}$".format(str(num)),linestyle="None",ms=20)
                # scatter ancor-based or atr-nets
                plt.plot(gt[num],pred2[num],color=color,marker="${}$".format(str(num)),linestyle="None",ms=20)
        # line
        plt.plot(line,line,"-",color="gold",linewidth=2)
        
        #plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["mathtext.fontset"] = "stix"
           
        plt.xlabel('Ground truth',fontsize=24)
        plt.ylabel('Predict',fontsize=24)
                    
        plt.xlim([np.min(gt),np.max(gt)])
        
        #plt.title(r"{} in {} stand {}".format(mode,cellname,standName)) 
        #plt.title(r"Observed-Predicted of {}".format(standName),fontsize=24) 
        #fig.subplots_adjust(left=0.2,bottom=0.2)
        #plt.tick_params(labelsize=18)
        #plt.legend()
        
        savePath = os.path.join(f"{mode}_{cell}.png")    
        plt.savefig(savePath)
        plt.close()

    #ToGoogle(dataName=savePath,dirName=gName)
    """
    # save txt
    strgtall = str(gt)
    strall1 = str(pred1)
    strall2 = str(pred2)
    
    f = open("{}.txt".format(savefilePath),"a")
    f.write("GT:" + strgtall + "\n")
    f.write("{}:".format(standName) + strall1 + "\n")
    f.write("{}:".format(vsName) + strall2 + "\n")
    f.close()
    """
# -----------------------------------------------------------------
def PlotX(data,mode="none",fileName="test.png",gName="none"):
    #stands = np.tile(data[0][:,np.newaxis],data.shape[0]-1).T
    stands = np.tile(data[0][:,np.newaxis],bottomNum-1).T
    meanVecInds = np.append(np.array([0]),np.argsort(np.mean(np.abs(stands - data[1:]),1)) + 1)
    #meanVecInds = meanVecInds[5:]

    sns.set_palette("cool")
     
    for ind in meanVecInds:
        plt.plot(data[ind],linewidth=2,label="{}".format(ind))
        #print(ind)


    plt.title("{} in X".format(mode)) 
    plt.legend(bbox_to_anchor=(1.1,1.05))

    plt.savefig("{}.png".format(fileName))
    plt.close()

    #ToGoogle(dataName="f{fileName}.png",dirName=gName)

# ------------------------ loss  -----------------------------------
def PlotLoss(data,mode="none",fileName="test.png",gName="none"):
    data = data[1:]
    plt.plot(data,linewidth=5.0)
    
    #plt.ylim([min(data),0.0])
    plt.ylim([min(data)-0.01,max(data)+0.01])
    plt.title("{} Loss".format(mode)) 

    plt.savefig(fileName)
    plt.close()

    #ToGoogle(dataName="",dirName=gName)

# -----------------------------------------------------------------
#def CountLoss(gt,pred1,pred2,mode1="none",mode2="none",cellname="none",dataNum="none"):
def CountLoss(trueY,predY1,predY2,mode1="none",mode2="none"):
    
    flag = False 
    for cell in np.arange(3):
        gt = trueY[:,cell]
        pred1 = predY1[:,cell]
        pred2 = predY2[:,cell]

        # based on under 5% & 25% data
        loss = np.square(gt-pred1)
        under5inds = np.argsort(loss)[::-1][:int(loss.shape[0]*0.05)]
        under25inds = np.argsort(loss)[::-1][:int(loss.shape[0]*0.25)]
        
        #gt = gt[under5inds]
        #pred1 = pred1[under5inds]
        #pred2 = pred2[under5inds]
        #gt = gt[under25inds]
        #pred1 = pred1[under25inds]
        #pred2 = pred2[under25inds]

        cnt1 = 0
        cnt2 = 0
        for num in np.arange(gt.shape[0]):
            # ORの方が精度低い 
            if np.square(gt[num] - pred1[num]) - np.square(gt[num] - pred2[num]) > 0:
                cnt1 += 1
            # ATRの方が精度低い
            elif np.square(gt[num] - pred1[num]) - np.square(gt[num] - pred2[num]) < 0: 
                cnt2 += 1
            elif np.square(gt[num] - pred1[num]) - np.square(gt[num] - pred2[num]) == 0:
                cnt1 += 0
                cnt2 += 0
        
        if not flag:
            cnt1s = cnt1
            cnt2s = cnt2
            flag = True
        else:
            cnt1s = np.hstack([cnt1s,cnt1])
            cnt2s = np.hstack([cnt2s,cnt2])
    
    allcnt1 = sum(cnt1s)
    allcnt2 = sum(cnt2s)
    # same accuracy
    if allcnt1 == allcnt2:
        #print(f"{mode1}:{cnt1} vs {mode2}:{cnt2} Num.of{dataNum}")
        print(f"{mode1}:{allcnt1} vs {mode2}:{allcnt2}")
        print("AIKO")
    # not same accuracy loose or win
    else:
        winInd = np.argmin([allcnt1,allcnt2])
        modes = [mode1,mode2]
        winMode = modes[winInd]

        #print(f"{mode1}:{cnt1} vs {mode2}:{cnt2} Num.of{dataNum}")
        print(f"{mode1}:{allcnt1} vs {mode2}:{allcnt2}")
        print(f"WIN!:{winMode}")
# -----------------------------------------------------------------
def HistLoss(gt,pred1,pred2,mode1="none",mode2="none",savefilePath="none"):
    loss1 = np.abs(np.square(gt - pred1))
    loss2 = np.abs(np.square(gt - pred2))
    loss1 = np.reshape(loss1,[-1,])
    loss2 = np.reshape(loss2,[-1,])
    
    sns.distplot(loss1,kde=False,label=f"{mode1}",color="royalblue")
    sns.distplot(loss2,kde=False,label=f"{mode2}",color="coral")
    
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    
    #plt.title(f"{mode1} vs {mode2}",size=18)
    plt.title(f"{mode1} vs {mode2}",size=24)
    plt.xlabel("Error",fontsize=24)
    plt.ylabel("# of data",fontsize=24)
    plt.legend(fontsize=20)
    savePath = os.path.join("{}.png".format(savefilePath))    
    plt.savefig(savePath)
    plt.close()
# -----------------------------------------------------------------
def HistPercentile(gt,pred,mode="none"):
    loss = np.abs(np.square(gt - pred))
    loss = np.reshape(loss,[-1,])
     
    percentiles = np.percentile(loss,[5,25,50,75,95])
    print(mode)
    print(percentiles[-1])
    print("---")
    for pt in percentiles:
        plt.axvline(pt,color="navy",ls="--",linewidth=0.5)
    
    sns.distplot(loss,kde=False,color="coral")
    #plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    
    plt.title(f"{percentiles}")
    plt.xlabel("Error",fontsize=24)
    plt.ylabel("# of data",fontsize=24) 
    label = mode.split("/")[-1][:-4]

    plt.savefig(f"{label}.png")
    plt.close()

# -----------------------------------------------------------------
"""
# ----------------------- Command ---------------------------------
# number of bottom data
bottomNum = int(sys.argv[1])
# 0:Oridinary Regression(OR), 1: Anchor-based, 2: Anchor-based (cls), 3: ATR-Nets, 4: ATR-Nets (cls)
standMode = int(sys.argv[2])
# ATR-Netsは追加でalphaの大きさを指定 20 or 200 or 500 or 1000
alphaMode = int(sys.argv[3])
# -----------------------------------------------------------------
"""
# ----------------------- loading ---------------------------------
# dir path
pklPath = os.path.join(resultsPath,savedpklPath)
# file path
filePath = "*.pkl"
# full path
paths = glob.glob(os.path.join(pklPath,filePath))

# natural sort
pklfullPath = []
for path in natsorted(paths):
    pklfullPath.append(path)    

pdb.set_trace()
# only Ordinary Regression pkl saved path
orpklpath = glob.glob(os.path.join(resultsPath,orPath,filePath))
# loading OR pkl data
#with open(pklfullPath[3],"rb") as fp:
with open(orpklpath[-1],"rb") as fp:
    teY = pickle.load(fp)
    predY_or = pickle.load(fp)
    teX_or = pickle.load(fp)
    trloss_or = pickle.load(fp)
    teloss_or = pickle.load(fp)
label = orpklpath[-1].split("/")[-1][:-4]
#BasicStatistics(teY,predY_or,mode=f"{label}")
#BottomScatter(teY,predY_or,predY_or,mode=f"{label}")
"""
# loading Anc pkl data
#with open(pklfullPath[4],"rb") as fp:
with open(orpklpath[0],"rb") as fp:
    teY = pickle.load(fp)
    predY_anc = pickle.load(fp)
    teX_anc = pickle.load(fp)
    predCls_anc = pickle.load(fp)
    predRes_anc = pickle.load(fp)
    trclsloss_anc = pickle.load(fp)
    teclsloss_anc = pickle.load(fp)
    trresloss_anc = pickle.load(fp)
    teresloss_anc = pickle.load(fp)
    trloss_anc = pickle.load(fp)
    teloss_anc = pickle.load(fp)

label = orpklpath[0].split("/")[-1][:-4]
#BasicStatistics(teY,predY_anc,mode=f"{label}")
#CountLoss(teY,predY_or,predY_anc,mode1="OR",mode2=f"{label}")
BottomScatter(teY,predY_or,predY_anc,mode=f"{label}")
"""
"""
# select alpha
if alphaMode == 20:
    atrInd = 5
elif alphaMode == 200:
    atrInd = 0
elif alphaMode == 500:
    atrInd = 1
elif alphaMode == 1000:
    atrInd = 2
"""
# loading ATR pkl data
#with open(pklfullPath[atrInd],"rb") as fp:
with open(orpklpath[1],"rb") as fp:
    teY = pickle.load(fp)
    predY_atr20 = pickle.load(fp)
    teX_atr = pickle.load(fp)
    predCls_atr = pickle.load(fp)
    predRes_atr = pickle.load(fp)
    trclsloss_atr = pickle.load(fp)
    teclsloss_atr = pickle.load(fp)
    trresloss_atr = pickle.load(fp)
    teresloss_atr = pickle.load(fp)
    trloss_atr = pickle.load(fp)
    teloss_atr = pickle.load(fp)

pdb.set_trace()
label = orpklpath[1].split("/")[-1][:-4]
#BasicStatistics(teY,predY_atr20,mode=f"{label}")
#CountLoss(teY,predY_or,predY_atr20,mode1="OR",mode2=f"{label}")
#BottomScatter(teY,predY_or,predY_atr20,mode=f"{label}")
"""
with open(orpklpath[2],"rb") as fp:
    teY = pickle.load(fp)
    predY_atr500 = pickle.load(fp)
    teX_atr = pickle.load(fp)
    predCls_atr = pickle.load(fp)
    predRes_atr = pickle.load(fp)
    trclsloss_atr = pickle.load(fp)
    teclsloss_atr = pickle.load(fp)
    trresloss_atr = pickle.load(fp)
    teresloss_atr = pickle.load(fp)
    trloss_atr = pickle.load(fp)
    teloss_atr = pickle.load(fp)

label = orpklpath[2].split("/")[-1][:-4]
#BasicStatistics(teY,predY_atr500,mode=f"{label}")
CountLoss(teY,predY_or,predY_atr500,mode1="OR",mode2=f"{label}")
BottomScatter(teY,predY_or,predY_atr500,mode=f"{label}")
"""
for i in np.arange(len(pklfullPath)): 
    # loading soft ATR pkl data
    with open(pklfullPath[i],"rb") as fp:
        teX = pickle.load(fp)
        teY = pickle.load(fp)
        testPred_sat = pickle.load(fp)
        testClsCenter_sat = pickle.load(fp)
        testSoftResPred_sat = pickle.load(fp)
        testResidual_sat = pickle.load(fp)
        testSoftRes_sat = pickle.load(fp)
        testSoftRResPred_sat = pickle.load(fp)
        trclsloss_sat = pickle.load(fp)
        teclsloss_sat = pickle.load(fp)
        trresloss_sat = pickle.load(fp)
        teresloss_sat = pickle.load(fp)
        trloss_sat = pickle.load(fp)
        teloss_sat = pickle.load(fp)
    
    #pdb.set_trace()
    label = pklfullPath[i].split("/")[-1][:-4]
    BasicStatistics(teY,testPred_sat,mode=f"{label}")
    #HistPercentile(teY,testPred_sat,mode=f"{label}")
    #CountLoss(teY,predY_or,testPred_sat,mode1="OR",mode2=f"{label}")
    #BottomScatter(teY,predY_or,testPred_sat,mode=f"{label}")
    #PlotLoss(trloss_sat,mode="Train SATR(cls+res)",fileName=f"trloss_sat{label}.png",gName="八谷先生共有用_acml")
    #PlotLoss(teloss_sat,mode="Test SATR(cls+res)",fileName=f"teloss_sat{label}.png",gName="八谷先生共有用_acml")

"""
#pdb.set_trace()
# stand for OR
if standMode == 0:
    # stand atr
    ind = np.argsort(np.mean(np.square(teY-predY_or),1))[::-1][:bottomNum]
    standName = "OR"
    vs1Name = "ANC(cls+reg)"
    #vs2Name = "ATR(cls+reg) with alpha={}".format(alphaMode)
    vs2Name = r"ATR($\alpha={}$)".format(alphaMode)
    vs3Name = "ATR(cls) with alpha={}".format(alphaMode)
    predY1 = predY_or
    predY2 = predY_anc
    predY3 = predY_atr
    predY4 = predCls_atr
    teX = teX_or

elif standMode == 1:
    # stand anc
    ind = np.argsort(np.mean(np.square(teY-predY_anc),1))[::-1][:bottomNum]
    standName = "ANC(cls+reg)"
    vs1Name = "ATR(cls+reg) with alpha={}".format(alphaMode)
    vs2Name = "ANC(cls) with alpha={}".format(alphaMode)
    vs3Name = "OR"
    predY1 = predY_anc
    predY2 = predY_atr
    predY3 = predCls_anc
    predY4 = predY_or
    teX = teX_anc

elif standMode == 2:
    # stand anc
    ind = np.argsort(np.mean(np.square(teY_atr-predCls_anc),1))[::-1][:bottomNum]
    standName = "ANC(cls)"
    vs1Name = "ANC(cls+reg)"
    vs2Name = "ATR(cls+reg) with alpha={}".format(alphaMode)
    vs3Name = "OR"
    predY1 = predCls_anc
    predY2 = predY_anc
    predY3 = predY_atr
    predY4 = predY_or
    teX = teX_anc

# stand for ATR-Nets (cls+reg)
elif standMode == 3:
    # stand atr
    ind = np.argsort(np.mean(np.square(teY-predY_atr),1))[::-1][:bottomNum]
    standName = r"ATR($\alpha={}$)".format(alphaMode)
    vs1Name = "ANC(cls+reg)"
    vs2Name = "OR"
    vs3Name = "ATR(cls)"
    predY1 = predY_atr
    predY2 = predY_anc
    predY3 = predY_or
    predY4 = predCls_atr
    teX = teX_atr

# stand for ATR-Nets (cls)
elif standMode == 4:
    # stand atr
    ind = np.argsort(np.mean(np.square(teY-predCls_atr),1))[::-1][:bottomNum]
    standName = "ATR(cls) with alpha={}".format(alphaMode)
    vs1Name = "ANC(cls+reg)"
    vs2Name = "ATR(cls+reg)"
    vs3Name = "OR"
    predY1 = predCls_atr
    predY2 = predY_anc
    predY3 = predY_atr
    predY4 = predY_or
    teX = teX_atr
"""
# -----------------------------------------------------------------

# ------------------- Call Count ---------------------------------------
"""
cellName = ["nk","tnk","tk"]
for i,name in enumerate(cellName):
    #pdb.set_trace()
    CountLoss(teY[ind,i],predY1[ind,i],predY2[ind,i],mode1=standName,mode2=vs1Name,cellname=name,dataNum=f"{bottomNum}")
    CountLoss(teY[:,i],predY1[:,i],predY2[:,i],mode1=standName,mode2=vs1Name,cellname=name,dataNum="all")
    CountLoss(teY[ind,i],predY1[ind,i],predY3[ind,i],mode1=standName,mode2=vs2Name,cellname=name,dataNum=f"{bottomNum}")
    CountLoss(teY[:,i],predY1[:,i],predY3[:,i],mode1=standName,mode2=vs2Name,cellname=name,dataNum="all")
    CountLoss(teY[ind,i],predY1[ind,i],predY4[ind,i],mode1=standName,mode2=vs3Name,cellname=name,dataNum=f"{bottomNum}")
    CountLoss(teY[:,i],predY1[:,i],predY4[:,i],mode1=standName,mode2=vs3Name,cellname=name,dataNum="all")
"""
# ------------------- Call Hist ----------------------------------------
"""
cellName = ["nk","tnk","tk"]
for i,name in enumerate(cellName):
    HistLoss(teY[:,i],pred1=predY1[:,i],pred2=predY2[:,i],mode1=standName,mode2=vs1Name,savefilePath=f"{standName}vs{vs1Name}in{name}")
    HistLoss(teY[:,i],pred1=predY1[:,i],pred2=predY3[:,i],mode1=standName,mode2=vs2Name,savefilePath=f"{standName}vs{vs2Name}in{name}")
    HistLoss(teY[:,i],pred1=predY1[:,i],pred2=predY4[:,i],mode1=standName,mode2=vs3Name,savefilePath=f"{standName}vs{vs3Name}in{name}")
"""
"""
HistLoss(teY,pred1=predY1,pred2=predY3,mode1=standName,mode2=vs2Name,savefilePath=f"{standName}vs{vs2Name}")

# ------------------- Call Scatter -------------------------------------

cellName = ["nk","tnk","tk"]
for i,name in enumerate(cellName):
    BottomScatter(teY[ind,i],predY1[ind,i],predY2[ind,i],mode="{} vs {}".format(standName,vs1Name),cellname=name,savefilePath="{}vs{}in{}_numof{}".format(standName,vs1Name,name,bottomNum),standName=standName,vsName=vs1Name,gName="八谷先生共有用_acml")
    BottomScatter(teY[ind,i],predY1[ind,i],predY3[ind,i],mode="{} vs {}".format(standName,vs2Name),cellname=name,savefilePath="{}vs{}in{}_numof{}".format(standName,vs2Name,name,bottomNum),standName=standName,vsName=vs2Name,gName="八谷先生共有用_acml")
    #BottomScatter(teY[ind,i],predY1[ind,i],predY4[ind,i],mode="{} vs {}".format(standName,vs3Name),cellname=name,savefilePath="{}vs{}in{}_numof{}".format(standName,vs3Name,name,bottomNum),standName=standName,vsName=vs3Name,gName="八谷先生共有用_acml")
"""

# ------------ Call Plot X -------------------

#PlotX(teX[ind],mode=standName,fileName=standName)

# ------------ Call Mean & Percentiles -------------------
"""
BasicStatistics(teY,predY_or,mode="OR")
BasicStatistics(teY,predY_anc,mode="Anc(cls+reg) with alpha={}".format(alphaMode))
BasicStatistics(teY,predCls_anc,mode="Anc(cls) with alpha={}".format(alphaMode))
BasicStatistics(teY,predY_atr,mode="Atr(cls+reg)")
BasicStatistics(teY,predCls_atr,mode="Atr(cls)")
"""

# ----------------- Call loss ----------------------------
"""
# Ordinary 
#PlotLoss(trloss_or,mode="Train OR",fileName="trloss_or.png",gName="八谷先生共有用_acml")
#PlotLoss(teloss_or,mode="Test OR",fileName="teloss_or.png",gName="八谷先生共有用_acml")
# Anchor-based
#PlotLoss(trclsloss_anc,,mode="Train ANC(cls)","trclsloss_an.png",gName="八谷先生共有用_acml")
#PlotLoss(teclsloss_anc,mode="Test ANC(cls)",fileName="teclsloss_anc.png",gName="八谷先生共有用_acml")
#PlotLoss(trresloss_an,"trresloss_an.png")
#PlotLoss(teresloss_anc,mode="Test ANC(res)",fileName="teresloss_anc.png",gName="八谷先生共有用_acml")
PlotLoss(trloss_anc,mode="Train ANC(cls+res)",fileName="trloss_anc.png",gName="八谷先生共有用_acml")
PlotLoss(teloss_anc,mode="Test ANC(cls+res)",fileName="teloss_anc.png",gName="八谷先生共有用_acml")
# ATR-Nets
PlotLoss(trclsloss_atr,mode="Train ATR(cls) with alpha={}".format(alphaMode) ,fileName="trclsloss_atrwith alpha={}.png".format(alphaMode),gName="八谷先生共有用_acml")
PlotLoss(teclsloss_atr,mode="Test ATR(cls)with alpha={}".format(alphaMode),fileName="teclsloss_atrwith alpha={}.png".format(alphaMode),gName="八谷先生共有用_acml")
PlotLoss(trresloss_atr,mode="Train ATR(res)with alpha={}".format(alphaMode),fileName="trresloss_atrwith alpha={}.png".format(alphaMode),gName="八谷先生共有用_acml")
PlotLoss(teresloss_atr,mode="Test ATR(res)with alpha={}".format(alphaMode),fileName="teresloss_atrwith alpha={}.png".format(alphaMode),gName="八谷先生共有用_acml")
PlotLoss(trloss_atr,mode="Train ATR(cls+res)with alpha={}".format(alphaMode),fileName="trloss_atrwith alpha={}.png".format(alphaMode),gName="八谷先生共有用_acml")
PlotLoss(teloss_atr,mode="Test ATR(cls+res)with alpha={}".format(alphaMode),fileName="teloss_atrwith alpha={}.png".format(alphaMode),gName="八谷先生共有用_acml")

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
