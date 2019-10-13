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

results = "results"
#pickles = "toypickles"
#pickles = "nkpickles"
pickles = "nankaipickles"
filepath = "*_400.pkl"
resultsPath = os.path.join(results,pickles,filepath)

files = glob.glob(resultsPath)

for file in files:
    print(file)
    #pdb.set_trace() 
    with open(file,"rb") as fp:
        _ = pickle.load(fp)
        _ = pickle.load(fp)
        te_gt = pickle.load(fp)
        te_pred = pickle.load(fp)
        #_ = pickle.load(fp)
        #_ = pickle.load(fp)
        #_ = pickle.load(fp)
        #_ = pickle.load(fp)
        #_ = pickle.load(fp)
        #_ = pickle.load(fp)
    """
    with open(file,"rb") as fp:
        _ = pickle.load(fp)
        _ = pickle.load(fp)
        te_gt = pickle.load(fp)
        te_pred = pickle.load(fp)
        #_ = pickle.load(fp)
        #_ = pickle.load(fp)
    """
    AverageVariable(te_gt,te_pred)

#pdb.set_trace()
"""

fpath1 = "*_3.pkl"
fpath2 = "*_6.pkl"
fpath3 = "*_400.pkl"
#fpath4 = "*_300.pkl"

rPath1 = os.path.join(results,pickles,fpath1)
rPath2 = os.path.join(results,pickles,fpath2)
rPath3 = os.path.join(results,pickles,fpath3)

#pdb.set_trace() 
file1 = glob.glob(rPath1)
file2 = glob.glob(rPath2)
file3 = glob.glob(rPath3)
#pdb.set_trace()
with open(file1[0],"rb") as fp:
    _ = pickle.load(fp)
    _ = pickle.load(fp)
    te_gt1 = pickle.load(fp)
    te_pred1 = pickle.load(fp)

with open(file2[0],"rb") as fp:
    _ = pickle.load(fp)
    _ = pickle.load(fp)
    te_gt2 = pickle.load(fp)
    te_pred2 = pickle.load(fp)

with open(file3[0],"rb") as fp:
    _ = pickle.load(fp)
    _ = pickle.load(fp)
    te_gt3 = pickle.load(fp)
    te_pred3 = pickle.load(fp)

axes = []

for cellInd in range(3):
    #axes.append(fig.add_subplot(1,3,cellInd))

    fig = plt.figure(figsize=(9,6))

    line = np.arange(np.min(te_gt1),np.max(te_gt1)+0.001,0.001)
    # scatter
    plt.plot(te_gt1[:,cellInd],te_pred1[:,cellInd],".",color="black",linestyle="None")
    #plt.plot(te_gt2[:,cellInd],te_pred2[:,cellInd],".",color="c",linestyle="None",label="ATR-Nets")
    # line
    plt.plot(line,line,"-",color="red",linewidth=4)
        
    plt.xlabel('ground truth',fontsize=40)
    plt.ylabel('predict',fontsize=40)
        
    plt.ylim([np.min(te_gt1),np.max(te_gt1)])
    plt.xlim([np.min(te_gt1),np.max(te_gt1)])
    #mpl.rcParams["axes.xmargin"]  = 1
    #mpl.rcParams["axes.ymargin"]  = 1
    fig.subplots_adjust(left=0.2,bottom=0.2)
    #plt.legend(loc="best")
    
    savePath = os.path.join("Scatter01_{}.png".format(cellInd))    
    plt.savefig(savePath)

    plt.close()

axes = []
for cellInd in range(3):
    #axes.append(fig.add_subplot(1,3,cellInd))

    fig = plt.figure(figsize=(9,6))
    line = np.arange(np.min(te_gt2),np.max(te_gt2)+0.001,0.001)
    # scatter
    plt.plot(te_gt2[:,cellInd],te_pred2[:,cellInd],".",color="black",linestyle="None")
    #plt.plot(te_gt2[:,cellInd],te_pred2[:,cellInd],".",color="c",linestyle="None",label="ATR-Nets")
    # line
    plt.plot(line,line,"-",color="red",linewidth=4)
        
    plt.xlabel('ground truth',fontsize=40)
    plt.ylabel('predict',fontsize=40)
        
    plt.ylim([np.min(te_gt2),np.max(te_gt2)])
    plt.xlim([np.min(te_gt2),np.max(te_gt2)])
    #mpl.rcParams["axes.xmargin"]  = 1
    #mpl.rcParams["axes.ymargin"]  = 1
    fig.subplots_adjust(left=0.2,bottom=0.2)
        
    #plt.legend(loc="best")
    
    savePath = os.path.join("Scatter02_{}.png".format(cellInd))    
    plt.savefig(savePath)

    plt.close()


axes = []
for cellInd in range(3):
    #axes.append(fig.add_subplot(1,3,cellInd))

    fig = plt.figure(figsize=(9,6))
    line = np.arange(np.min(te_gt3),np.max(te_gt3)+0.001,0.001)
    # scatter
    plt.plot(te_gt3[:,cellInd],te_pred3[:,cellInd],".",color="black",linestyle="None")
    #plt.plot(te_gt2[:,cellInd],te_pred2[:,cellInd],".",color="c",linestyle="None",label="ATR-Nets")
    # line
    plt.plot(line,line,"-",color="red",linewidth=4)
        
    plt.xlabel('ground truth',fontsize=40)
    plt.ylabel('predict',fontsize=40)
        
    plt.ylim([np.min(te_gt3),np.max(te_gt3)])
    plt.xlim([np.min(te_gt3),np.max(te_gt3)])
    fig.subplots_adjust(left=0.2,bottom=0.2)
    #mpl.rcParams["axes.xmargin"]  = 1
    #mpl.rcParams["axes.ymargin"]  = 1
        
    #plt.legend(loc="best")
    
    savePath = os.path.join("Scatter03_{}.png".format(cellInd))    
    plt.savefig(savePath)

    plt.close()

pdb.set_trace()


file1 = file1[0]
#files2 = files2[1:]
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

"""
def tSNE(self):    
    decomp = TSNE(n_components=2)
    X1_decomp = decomp.fit_transform(self.x11Train[:10,:])
    print("1")
"""        



