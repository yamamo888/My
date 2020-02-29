# -*- coding: utf-8 -*-

import os

import numpy as np

import matplotlib.pylab as plt
import matplotlib.pyplot as plt2
import pylab
import seaborn as sns

import pdb

# path ------------------------------------------------------------------------
images = "images"
numLinePath = "numlines"
lhPath = "lhs"
# -----------------------------------------------------------------------------


# 数直線 -----------------------------------------------------------------------
def NumberLine(x,pred,time=0,label="auto"):
    """
    発生年数がどうなってるかを確認したくって
    [Args]
        x   : 真値t list[nk,tnk,tk]
        pred: 予測値t
    """
    for cell in np.arange(3):
        #pdb.set_trace()
        # predict year [perticles,]
        xhat = pred[:,cell]
        # y = 0
        #y = [0] * x.shape[0]
        y = [0] * 1
        yhat = [0] * xhat.shape[0]
        
        # 数直線 -----------------------------------------------------------
        fig,ax=plt2.subplots(figsize=(10,10)) #画像サイズ
        fig.set_figheight(1) #高さ調整
        ax.tick_params(labelbottom=True, bottom=False) #x軸設定
        ax.tick_params(labelleft=False, left=False) #y軸設定
        # -----------------------------------------------------------------
        
        # 散布図 -----------------------------------------------------------
        plt2.scatter(x,y[0],c='coral') # 真値
        plt2.scatter(xhat,yhat,c='skyblue') # 予測値
        # -----------------------------------------------------------------
       
        # グラフの体裁--------------------------------------------------------
        xMin, xMax = np.min(np.append(x,xhat)), np.max(np.append(x,xhat))  
        plt2.tight_layout() #グラフの自動調整    
        plt2.hlines(y=0,xmin=xMin,xmax=xMax,color="silver") #横軸
        #plt2.xticks(np.arange(xMin,xMax,2)) #目盛り数値
        pylab.box(False) #枠を消す
        # -----------------------------------------------------------------
        
        plt2.savefig(os.path.join(images,numLinePath,f"{label}_{cell}.png"))
        plt2.close()
# -----------------------------------------------------------------------------

# ヒストグラム --------------------------------------------------------------------
def HistLikelihood(weights,time=0,label="auto"):
    
    # mean & var for label
    lhMean = np.mean(weights,0)
    lhVar = np.var(weights,0)
    #pdb.set_trace()
    # plot
    sns.set_style("dark")
    
    sns.distplot(weights[:,0],label="nk",kde=False,rug=False)
    sns.distplot(weights[:,1],label="tnk",kde=False,rug=False)
    sns.distplot(weights[:,2],label="tk",kde=False,rug=False)
    
    plt.legend(loc="upper right")
    plt.suptitle(f"mean:{lhMean}\n var:{lhVar}")
    plt.savefig(os.path.join(images,lhPath,f"{label}.png"))
    plt.close()
# -----------------------------------------------------------------------------

        
        
        
        
        