# -*- coding: utf-8 -*-

import os

import numpy as np

import matplotlib.pylab as plt
import matplotlib.pyplot as plt2
import pylab
import seaborn as sns

import pdb

# path ------------------------------------------------------------------------
imgPath = "images"
numLinePath = "numlines"
lhPath = "lhs"
# -----------------------------------------------------------------------------

# parameter -------------------------------------------------------------------
cellname = ["nk","tnk","tk"]
# -----------------------------------------------------------------------------

# 数直線 -----------------------------------------------------------------------
def NumberLine(gt,pred,label="auto"):
    """
    発生年数がどうなってるかを確認したくって
    [Args]
        gt   : 真値t list[nk,tnk,tk]
        pred: 予測値t [perticles,cells]
    """
    for cell in np.arange(3):
        #pdb.set_trace()
        # predict year [perticles,]
        x = gt[cell]
        xhat = pred[:,cell]
        y = [0] * 1 # y = 0
        yhat = [0] * xhat.shape[0]
        
        # 数直線 -----------------------------------------------------------
        fig,ax=plt2.subplots(figsize=(10,10)) #画像サイズ
        fig.set_figheight(1) #高さ調整
        ax.tick_params(labelbottom=True, bottom=False) #x軸設定
        ax.tick_params(labelleft=False, left=False) #y軸設定
        # -----------------------------------------------------------------
       
        # グラフの体裁--------------------------------------------------------
        xMin, xMax = np.min(np.append(x,xhat)), np.max(np.append(x,xhat))  
        plt2.tight_layout() #グラフの自動調整    
        plt2.hlines(y=0,xmin=xMin,xmax=xMax,color="silver") #横軸
        pylab.box(False) #枠を消す
        # -----------------------------------------------------------------
       
        # 散布図 -----------------------------------------------------------
        plt2.scatter(xhat,yhat,c='skyblue') # 予測値
        plt2.scatter(x,y[0],c='coral') # 真値
        plt2.title(f"min:{int(np.min(xhat))} max:{int(np.max(xhat))}")
        # -----------------------------------------------------------------
        plt2.savefig(os.path.join(imgPath,numLinePath,f"{label}_{cellname[cell]}.png"),bbox_inches="tight")
        #plt2.savefig(os.path.join(imgPath,numLinePath,f"{label}.png"))
        
        plt2.close()
# -----------------------------------------------------------------------------

# ヒストグラム --------------------------------------------------------------------
def HistLikelihood(weights,time=0,label="auto"):
    #pdb.set_trace()
    
    # mean & var for label
    lhMean = np.mean(weights,0)
    lhVar = np.var(weights,0)
    
    sns.set_style("dark")
    sns.distplot(weights,kde=False,rug=False) 
    plt.suptitle(f"mean:{lhMean}\n var:{lhVar}")
    plt.savefig(os.path.join(imgPath,lhPath,f"{label}.png"))
    plt.close()

# -----------------------------------------------------------------------------

        
        
        
        
        