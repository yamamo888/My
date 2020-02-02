# -*- coding: utf-8 -*-

import os

import numpy as np

import matplotlib.pylab as plt
import matplotlib.pyplot as plt2
import pylab

import pdb

# path ------------------------------------------------------------------------
images = "images"
numLinePath = "numlines"
# -----------------------------------------------------------------------------


# 数直線 ----------------------------------------------------------------------
def NumberLine(x,pred,label="auto"):
    """
    発生年数がどうなってるかを確認したくって
    [Args]
        x: 真値t
        pred: 予測値t(zero padding 済み)
    """
    
    for p in np.arange(pred.shape[1]):
        
        xhat = pred[:,p]
        # y = 0
        #y = [0] * x.shape[0]
        y = [0] * 1
        yhat = [0] * xhat.shape[0]
        
        # 数直線 -------------------------------------------------------------------
        fig,ax=plt2.subplots(figsize=(10,10)) #画像サイズ
        fig.set_figheight(1) #高さ調整
        ax.tick_params(labelbottom=True, bottom=False) #x軸設定
        ax.tick_params(labelleft=False, left=False) #y軸設定
        # -------------------------------------------------------------------------
        
        # 数直線上の数値を表示 -------------------------------------------------------
        # 真値
        ax.annotate(f'{x}',
                         xy=(x,y[0]),
                         xytext=(10, 20),
                         textcoords='offset points',
                         arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=.3")
                        )
        for i in np.arange(xhat.shape[0]):
            # 予測値
            ax.annotate(f'{xhat[i]}',
                         xy=(xhat[i],yhat[i]),
                         xytext=(10, 20),
                         textcoords='offset points',
                         arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=.3")
                        )
        # -------------------------------------------------------------------------
        
        # 散布図 ------------------------------------------------------------------
        plt2.scatter(x,y[0],c='coral') # 真値
        plt2.scatter(xhat,yhat,c='skyblue') # 予測値
        # -------------------------------------------------------------------------
       
        # グラフの体裁----------------------------------------------------------------
        xMin, xMax = np.min(np.append(x,xhat)), np.max(np.append(x,xhat))  
        plt2.tight_layout() #グラフの自動調整    
        plt2.hlines(y=0,xmin=xMin,xmax=xMax) #横軸
        #plt2.xticks(np.arange(xMin,xMax,2)) #目盛り数値
        pylab.box(False) #枠を消す
        # -------------------------------------------------------------------------
        
        #plt2.show()
        plt2.savefig(os.path.join(images,numLinePath,f"{label}_perticles{p}.png"))
        plt2.close()
        
# -----------------------------------------------------------------------------
    

# 散布図 ----------------------------------------------------------------------
def FeaturePlot(x):
    """
    V,Th,Bの推移を見たかった
    """
    
    for cell in np.arange(x[0].shape[1]):
        # V
        plt.scatter(x[0].shape[0],x[0][:,cell],color="blue")
        plt.show()
        plt.savefig(os.path.join(images,f"V_{cell}.png"))
        plt.close()
        # theta
        plt.scatter(x[1].shape[0],x[1][:,cell],color="lightgreen")
        plt.show()
        plt.savefig(os.path.join(images,f"Theta_{cell}.png"))
        plt.close()
        # B
        plt.scatter(x[2].shape[0],x[1][:,cell],color="pink")
        plt.show()
        plt.savefig(os.path.join(images,f"B_{cell}.png"))
        plt.close()
# -----------------------------------------------------------------------------
    