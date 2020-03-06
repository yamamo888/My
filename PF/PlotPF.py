# -*- coding: utf-8 -*-

import os

import numpy as np

import matplotlib.pylab as plt
import matplotlib.pyplot as plt2
from mpl_toolkits.mplot3d import Axes3D
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
ntI,tntI,ttI = 0,1,2
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
def HistLikelihood(weights,label="auto",color="black"):
    #pdb.set_trace()
    
    # mean & var for label
    lhMean = np.mean(weights,0)
    lhVar = np.var(weights,0)
    
    sns.set_style("dark")
    sns.distplot(weights,kde=False,rug=False,color=color) 
    
    plt.xlim([0,0.12])
    plt.ylim([0,175])
    plt.suptitle(f"mean:{lhMean}\n var:{lhVar}")
    plt.savefig(os.path.join(imgPath,lhPath,f"{label}.png"))
    plt.close()

# -----------------------------------------------------------------------------
    
# 散布図 -----------------------------------------------------------------------
def scatter3D(x,y,z,rangeP,title="none",label="none"):
    #pdb.set_trace()
    #print("-----")
    #print(np.min(x),np.min(y),np.min(z))
    #print(np.max(x),np.max(y),np.max(z))
    sns.set_style("dark")

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x,y,z,c="black",marker="o",alpha=0.5,linewidths=0.5)
    
    ax.set_xlabel("nk")
    ax.set_ylabel("tnk")
    ax.set_zlabel("tk")
    
    ax.set_xlim(rangeP[0][ntI],rangeP[1][ntI])
    ax.set_ylim(rangeP[0][tntI],rangeP[1][tntI])
    ax.set_zlim(rangeP[0][ttI],rangeP[1][ttI])

    ax.set_title(f"{title}")
    
    plt.savefig(os.path.join(imgPath,"PF",f"{label}.png"))
    #plt.show()
    plt.close()
# -----------------------------------------------------------------------------


        
        
        
        
        