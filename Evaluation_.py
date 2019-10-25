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

visualPath = "visualization"
scatterPath = "scatter"

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
def AverageVariable(gt,pred):
    print("Mean",np.mean(np.square(gt-pred),0))
    print("Var",np.var(np.square(gt-pred),0))
    print("--------------------------------")
# -----------------------------------------------------------------
def BottomScatter(gt,pred2,pred1,gtall,all2,all1,color="yellow",mode="none",cellname="none",savefilePath="test.png",titleName="test"):
     
    fig = plt.figure(figsize=(9,6))    
    # 45° line
    line = np.arange(np.min(gt),np.max(gt)+0.001,0.001)
    
    #strPred1 = pred1.astype(str).tolist()
    #strPred2 = pred2.astype(str).tolist()
    
    strgtall = str(gtall)
    strall1 = str(all1)
    strall2 = str(all2)
    
    residual12_ind = np.argsort(abs(gt-pred1)-abs(gt-pred2))[::-1]
    residual12 = abs(gt-pred1) - abs(gt-pred2)
    marks = np.zeros(gt.shape[0])
    sort_mark = np.arange(10,21,1)
    colors = []
    for res in residual12:
        if res < 0.000000000:
            colors.append("blue")
        elif res > 0.0000000000:
            colors.append("red")
    
    #goodcnt
    for ind,size in  zip(residual12_ind,sort_mark):
        marks[ind] = size
    """
    for num in np.arange(gt.shape[0]):
        plt.plot(gt[num],pred1[num],color="black",marker="${}$".format(str(num)),linestyle="None",ms=10)
        # scatter ancor-based or atr-nets
        plt.plot(gt[num],pred2[num],color=color,marker="${}$".format(str(num)),linestyle="None",ms=10)
    """
    
    #pdb.set_trace() 
    for num,(color,mark) in enumerate(zip(colors,marks)):
        # scatter oridinary
        #pdb.set_trace()
        plt.plot(gt[num],pred1[num],color="black",marker="${}$".format(str(num)),linestyle="None",ms=10)
        # scatter ancor-based or atr-nets
        plt.plot(gt[num],pred2[num],color=color,marker="${}$".format(str(num)),linestyle="None",ms=mark)
    # line
    plt.plot(line,line,"-",color="gold",linewidth=2)
            
    plt.xlabel('ground truth',fontsize=18)
    plt.ylabel('predict',fontsize=18)
                
    #plt.ylim([np.min(gt),np.max(gt)])
    #plt.xlim([np.min(gt),np.max(gt)])
    plt.title("{} in {}".format(mode,cellname)) 
    fig.subplots_adjust(left=0.2,bottom=0.2)
    plt.tick_params(labelsize=18)
    plt.legend()
    
    savePath = os.path.join("{}.png".format(savefilePath))    
    plt.savefig(savePath)
        
    plt.close()
    
    # save txt
    f = open("{}.txt".format(savefilePath),"a")
    f.write("gt:" + strgtall + "\n")
    f.write("OR:" + strall1 + "\n")
    f.write("ANC(ATR):" + strall2 + "\n")
    f.close()


# ------------------------ loss  -----------------------------------
def loss(data,fileName="test.png"):
    data = data[10:]
    plt.plot(data,linewidth=5.0)
    plt.ylim([min(data),0.0000005])

    plt.savefig(fileName)
    plt.close()
# -----------------------------------------------------------------

# ----------------- Call bottom scatter----------------------------
bottomNum = int(sys.argv[1])

pklPath = os.path.join("results","nankaipickles")

pklnames = ["test_{}_nankai_1_0_5_1000_0_0_1.pkl".format(149999),"test_{}_nankai_2_1_21_5_1000_0_0_2.pkl".format(149999),"test_{}_nankai_3_2_21_5_1000_[20. 20. 20.]_0_0_3.pkl".format(149999)]
#pklnames = ["test_{}_nankai_1_0_4_1000_0_0_1.pkl".format(10000),"test_{}_nankai_1_1_21_5_1000_0_0_1.pkl".format(10000),"test_{}_nankai_3_2_21_5_1000_[20. 20. 20.]_0_0_3.pkl".format(10000)]
with open(os.path.join(pklPath,pklnames[0]),"rb") as fp:
    _ = pickle.load(fp)
    _ = pickle.load(fp)
    teY1 = pickle.load(fp)
    predY1 = pickle.load(fp)
    _ = pickle.load(fp)
    _ = pickle.load(fp)
    trloss_or = pickle.load(fp)
    teloss_or = pickle.load(fp)

with open(os.path.join(pklPath,pklnames[1]),"rb") as fp:
    _ = pickle.load(fp)
    _ = pickle.load(fp)
    teY2 = pickle.load(fp)
    predY2 = pickle.load(fp)
    _ = pickle.load(fp)
    _ = pickle.load(fp)
    trclsloss_an = pickle.load(fp)
    teclsloss_an = pickle.load(fp)
    trresloss_an = pickle.load(fp)
    teresloss_an = pickle.load(fp)
    trloss_an = pickle.load(fp)
    teloss_an = pickle.load(fp)

with open(os.path.join(pklPath,pklnames[2]),"rb") as fp:
    _ = pickle.load(fp)
    _ = pickle.load(fp)
    teY3 = pickle.load(fp)
    predY3 = pickle.load(fp)
    _ = pickle.load(fp)
    _ = pickle.load(fp)
    trclsloss_at = pickle.load(fp)
    teclsloss_at = pickle.load(fp)
    trresloss_at = pickle.load(fp)
    teresloss_at = pickle.load(fp)
    trloss_at = pickle.load(fp)
    teloss_at = pickle.load(fp)


#print("or msq:\n")
orloss = np.mean(np.square(teY1-predY1),1)
#print("atr msq:\n")
ancloss = np.mean(np.square(teY2-predY2),1)
atloss = np.mean(np.square(teY3-predY3),1)

"""
print("or msq:{}".format(np.mean(orloss)))
print("anc msq:{}".format(np.mean(ancloss)))
print("atr msq:{}".format(np.mean(atloss)))

print("-----")
print("or hakohige:{}".format(np.percentile(orloss,[95,75,50,25,5])))
print("anc hakohige:{}".format(np.percentile(ancloss,[95,75,50,25,5])))
print("at hakohige:{}".format(np.percentile(atloss,[95,75,50,25,5])))
"""


#pdb.set_trace()
#np.sort(orloss)


# ----------------- Call loss ----------------------------
"""
# Ordinary 
loss(trloss_or,"trloss_or.png")
loss(teloss_or,"teloss_or.png")
# Anchor-based
loss(trclsloss_an,"trclsloss_an.png")
loss(teclsloss_an,"teclsloss_an.png")
loss(trresloss_an,"trresloss_an.png")
loss(teresloss_an,"teresloss_an.png")
loss(trloss_an,"trloss_an.png")
loss(teloss_an,"teloss_an.png")
# ATR-Nets
loss(trclsloss_at,"trclsloss_at.png")
loss(teclsloss_at,"teclsloss_at.png")
loss(trresloss_at,"trresloss_at.png")
loss(teresloss_at,"teresloss_at.png")
loss(trloss_at,"trloss_at.png")
loss(teloss_at,"teloss_at.png")
loss(trloss_at,"trloss_at.png")
loss(teloss_at,"teloss_at.png")

pdb.set_trace()
"""
"""
# select data bottom 10 (stand ordinary)
ind_nk1 = np.argsort(np.abs(teY1[:,0]-predY1[:,0]))[::-1][:bottomNum]
ind_tnk1 = np.argsort(np.abs(teY1[:,1]-predY1[:,1]))[::-1][:bottomNum]
ind_tk1 = np.argsort(np.abs(teY1[:,2]-predY1[:,2]))[::-1][:bottomNum]
index = [ind_nk1,ind_tnk1,ind_tk1]
standName = "OR"
"""

# select data bottom 10 (stand atr-nets)
ind_nk3 = np.argsort(np.abs(teY3[:,0]-predY3[:,0]))[::-1][:bottomNum]
ind_tnk3 = np.argsort(np.abs(teY3[:,1]-predY3[:,1]))[::-1][:bottomNum]
ind_tk3 = np.argsort(np.abs(teY3[:,2]-predY3[:,2]))[::-1][:bottomNum]
index = [ind_nk3,ind_tnk3,ind_tk3]
standName = "ATR"
pdb.set_trace()
#ind_all = np.argsort(np.mean(np.square(teY1-predY1),1))[::-1][:bottomNum]
ind_all = np.argsort(np.mean(np.square(teY3-predY3),1))[::-1][:bottomNum]
# select data bottom 10 (stand atr-nets)
index = [ind_all,ind_all,ind_all]
standName = "alldata"

#pdb.set_trace()

cellName = ["nk","tnk","tk"]
for i,(name,ind) in enumerate(zip(cellName,index)):
    BottomScatter(teY1[ind,i],predY1[ind,i],predY2[ind,i],teY1[ind],predY1[ind],predY2[ind],color="red",mode="OR vs ANC ,stand {}".format(standName),cellname=name,savefilePath="ORANC{}_{}_{}".format(name,bottomNum,standName),titleName="{}".format(name))
    BottomScatter(teY1[ind,i],predY1[ind,i],predY3[ind,i],teY1[ind],predY1[ind],predY3[ind],color="red",mode="OR vs ATR ,stand {}".format(standName),cellname=name,savefilePath="ORATR{}_{}_{}".format(name,bottomNum,standName),titleName="{}".format(name))
    #BottomScatter(teY1[ind,i],predY1[ind,i],predY3[ind,i],teY1[ind],predY1[ind],predY3[ind],color="red",mode="OR vs ATR ,stand {}".format(standName),cellname=name,savefilePath="ORATR{}_{}_{}".format(name,bottomNum,standName),titleName="{}".format(name))


# ---------- Call average ----------------------------------------
"""
results = "results"
#pickles = "toypickles"
    BottomScatter(teY1[ind,i],predY1[ind,i],predY2[ind,i],teY1[ind],predY1[ind],predY2[ind],color="red",mode="OR vs ANC ,stand {}".format(standName),cellname=name,savefilePath="ORANC{}_{}_{}".format(name,bottomNum,standName),titleName="{}".format(name))
    BottomScatter(teY1[ind,i],predY1[ind,i],predY3[ind,i],teY1[ind],predY1[ind],predY3[ind],color="red",mode="OR vs ATR ,stand {}".format(standName),cellname=name,savefilePath="ORATR{}_{}_{}".format(name,bottomNum,standName),titleName="{}".format(name))


# ---------- Call average ----------------------------------------
"""
results = "results"
#pickles = "toypickles"
pickles = "nankaipickles"
filepath = "*.pkl"
resultsPath = os.path.join(results,pickles,filepath)

files = glob.glob(resultsPath)

for file in files:
    print(file)

    with open(file,"rb") as fp:
        _ = pickle.load(fp)
        _ = pickle.load(fp)
        te_gt = pickle.load(fp)
        te_pred = pickle.load(fp)
    
    AverageVariable(te_gt,te_pred)
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
# ------------ Call tsne ---------------------------------------

"""
