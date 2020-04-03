# -*- coding: utf-8 -*-

import os
import glob

import numpy as np
import statistics

import matplotlib.pylab as plt
import matplotlib.pyplot as plt2
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

import pylab
import seaborn as sns

import pdb

import makingDataPF as myData

# path ------------------------------------------------------------------------
imgPath = "images"
numLinePath = "numlines"
lhPath = "lhs"
# -----------------------------------------------------------------------------

# parameter -------------------------------------------------------------------
cellname = ["nk","tnk","tk"]
ntI,tntI,ttI = 0,1,2
th = 0
# -----------------------------------------------------------------------------

# 数直線 -----------------------------------------------------------------------
def NumberLine(gt,pred,label="auto",isSave=True):
    """
    発生年数がどうなってるかを確認したくって
    [Args]
        gt   : 真値t list[nk,tnk,tk]
        pred: 予測値t [perticles,cells]
    """
    for cell in np.arange(3):
        
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
        plt2.scatter(xhat,yhat,c='skyblue') # predict
        plt2.scatter(x,y[0],c='coral') # ground truth
        plt2.scatter(int(np.mean(xhat)),y[0],c='royalblue') # mean of predict
        plt2.scatter(int(statistics.median(xhat)),y[0],c='forestgreen') # median of predict
        plt2.title(f"min:{int(np.min(xhat))} max:{int(np.max(xhat))} mean:{int(np.mean(xhat))}")
        # -----------------------------------------------------------------
        
        if isSave:
            plt2.savefig(os.path.join(imgPath,numLinePath,f"{label}_{cellname[cell]}.png"),bbox_inches="tight")
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
    
    #plt.xlim([0,0.12])
    #plt.ylim([0,175])
    plt.suptitle(f"mean:{lhMean}\n var:{lhVar}")
    plt.savefig(os.path.join(imgPath,lhPath,f"{label}.png"))
    plt.close()
# -----------------------------------------------------------------------------
    
# 散布図 -----------------------------------------------------------------------
def scatter3D(x,y,z,rangeP,path="none",title="none",label="none",isResults=False):
    #pdb.set_trace()
    
    sns.set_style("dark")

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x,y,z,c="black",marker="o",alpha=0.5,linewidths=0.5)
    
    if isResults:
        xmean,ymean,zmean = np.mean(x),np.mean(y),np.mean(z)
        xmedian,ymedian,zmedian = statistics.median(x),statistics.median(y),statistics.median(z)
        
        ax.scatter(xmean,ymean,zmean,c="red",marker="o",alpha=0.5,linewidths=0.5)
        ax.scatter(xmedian,ymedian,zmedian,c="orange",marker="o",alpha=0.5,linewidths=0.5)
    
    ax.set_xlabel("nk")
    ax.set_ylabel("tnk")
    ax.set_zlabel("tk")
    
    ax.set_xlim(rangeP[0][ntI],rangeP[1][ntI])
    ax.set_ylim(rangeP[0][tntI],rangeP[1][tntI])
    ax.set_zlim(rangeP[0][ttI],rangeP[1][ttI])
    
    if isResults:
        ax.set_title(f"{title}\n{np.round(xmean,3)} {np.round(ymean,3)} {np.round(zmean,3)}")
    else:
        ax.set_title(f"{title}")
    
    
    plt.savefig(os.path.join(imgPath,path,f"{label}.png"))
    #plt.show()
    plt.close()
# -----------------------------------------------------------------------------

# アニメーション -------------------------------------------------------------------
def gif2Animation(gifPath,label="none"):
    #pdb.set_trace()
    # *png path
    gifs = glob.glob(gifPath)
    
    fig = plt.figure()
    
    # animationの体裁を整える
    ax = plt.subplot()
    ax.spines['right'].set_color('None')
    ax.spines['left'].set_color('None')
    ax.spines['top'].set_color('None')
    ax.spines['bottom'].set_color('None')
    ax.tick_params(axis='x',which='both',top='off',bottom='off',labelbottom='off')
    ax.tick_params(axis='y',which='both',left='off',right='off',labelleft='off')

    imgs = []
    for gif in gifs:
        img = Image.open(gif)
        imgs.append([plt.imshow(img)])
    
    # making animation
    myAnima = animation.ArtistAnimation(fig, imgs, repeat_delay=10000)
    #plt.show()
    pw = animation.PillowWriter(fps=20)
    myAnima.save(os.path.join(imgPath,"animaPF",f"{label}.gif"), writer=pw)
# -----------------------------------------------------------------------------

# 地震履歴 ---------------------------------------------------------------------
def Rireki(gt,predict,path="deltaU",title="none",label="none",isResearch=False,isShare=False,isSeparate=True):
    
    sns.set_style("dark")

    pred = np.zeros([1400,3])
    
    if isResearch:
        # gt eq.
        gYear_nk = np.where(gt[:,0] > th)[0]
        gYear_tnk = np.where(gt[:,1] > th)[0]
        gYear_tk = np.where(gt[:,2] > th)[0]
        # mse
        ndist_nk = myData.calcDist(gYear_nk,predict[ntI].T,error=2)
        ndist_tnk = myData.calcDist(gYear_tnk,predict[tntI].T,error=2)
        ndist_tk = myData.calcDist(gYear_tk,predict[ttI].T,error=2)
        
        yearError_nk = np.sum(np.min(ndist_nk,1))
        yearError_tnk = np.sum(np.min(ndist_tnk,1))
        yearError_tk = np.sum(np.min(ndist_tk,1))
        # sum year error
        yearError = yearError_nk + yearError_tnk + yearError_tk
        
        title = yearError
        
        pred[predict[ntI].tolist(),ntI] = 25
        pred[predict[tntI].tolist(),tntI] = 25
        pred[predict[ttI].tolist(),ttI] = 25
        
    
    else:
        pred[np.where(predict[:,ntI]>1)[0],ntI] = 25
        pred[np.where(predict[:,tntI]>1)[0],tntI] = 25
        pred[np.where(predict[:,ttI]>1)[0],ttI] = 25
        
    # share gt & pred
    if isShare:
        fig, figInds = plt.subplots(nrows=3, sharex=True)
        for figInd in np.arange(len(figInds)):
            figInds[figInd].plot(np.arange(pred.shape[0]), pred[:,figInd],color="skyblue")
            figInds[figInd].plot(np.arange(gt.shape[0]), gt[:,figInd],color="coral")
    
    if isSeparate:
        colors = ["coral","skyblue","coral","skyblue","coral","skyblue"]
        plot_data = [gt[:,ntI],pred[:,ntI],gt[:,tntI],pred[:,tntI],gt[:,ttI],pred[:,ttI]]
        
        fig = plt.figure()
        fig, axes = plt.subplots(nrows=6,sharex="col")
        for row,(color,data) in enumerate(zip(colors,plot_data)):
            axes[row].plot(np.arange(1400), data, color=color)
    
    plt.suptitle(f"{title}", fontsize=8)
    plt.savefig(os.path.join(imgPath,path,f"{label}.png"))
    plt.close()
    
    if isResearch:
        return yearError
    
# -----------------------------------------------------------------------------

    
        
        
        
        
        