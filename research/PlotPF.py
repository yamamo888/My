# -*- coding: utf-8 -*-

import os
import glob
import pdb

import numpy as np
import matplotlib.pylab as plt
import matplotlib.pyplot as plt2
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.axes_grid1
from PIL import Image
import pylab
import seaborn as sns

import makingDataPF as myData

# parameter -------------------------------------------------------------------
cellname = ["nk","tnk","tk"]
ntI,tntI,ttI = 0,1,2
# -----------------------------------------------------------------------------

# 数直線 -----------------------------------------------------------------------
def NumberLine(gt,pred,path,label="auto"):
    """
    発生年数がどうなってるかを確認したくって
    Args
        gt: 真値t list[nk,tnk,tk]
        pred: 予測値t [perticles,cells]
    """
    for cell in np.arange(3):
        #pdb.set_trace()
        # predict year [perticles,]
        x = gt[cell]
        xhat = pred[:,cell]
        y = [0] * 1 # y = 0
        yhat = [0] * xhat.shape[0]
        
        # 数直線 ---------------------------------------------------------------
        fig,ax=plt2.subplots(figsize=(10,10)) #画像サイズ
        fig.set_figheight(1) #高さ調整
        ax.tick_params(labelbottom=True, bottom=False) #x軸設定
        ax.tick_params(labelleft=False, left=False) #y軸設定
        # ---------------------------------------------------------------------
       
        # グラフの体裁 -----------------------------------------------------------
        #xMin, xMax = np.min(np.append(x,xhat)), np.max(np.append(x,xhat))  
        xMin, xMax = 0, 1400  
        plt2.tight_layout() #グラフの自動調整    
        plt2.hlines(y=0,xmin=xMin,xmax=xMax,color="silver") #横軸
        pylab.box(False) #枠を消す
        # -----------------------------------------------------------------
       
        # 散布図 -----------------------------------------------------------
        plt2.scatter(xhat,yhat,c='skyblue') # 予測値
        plt2.scatter(x,y[0],c='coral') # 真値
        plt2.title(f"min:{int(np.min(xhat))} max:{int(np.max(xhat))}")
        # -----------------------------------------------------------------
        myData.isDirectory(path)
        plt2.savefig(os.path.join(path,f"{label}_{cellname[cell]}.png"),bbox_inches="tight")
        
        plt2.close()
# -----------------------------------------------------------------------------

# ヒストグラム --------------------------------------------------------------------
def HistLikelihood(weights,path,label="auto",color="black"):
    #pdb.set_trace()
    
    # mean & var for label
    lhMean = np.mean(weights,0)
    lhVar = np.var(weights,0)
    
    sns.set_style("dark")
    sns.distplot(weights,kde=False,rug=False,color=color) 
    
    #plt.xlim([0,0.12])
    #plt.ylim([0,175])
    plt.suptitle(f"mean:{lhMean}\n var:{lhVar}")
    myData.isDirectory(path)
    plt.savefig(os.path.join(path,f"{label}.png"))
    plt.close()

# -----------------------------------------------------------------------------
    
# 散布図 -----------------------------------------------------------------------
def scatter3D(x,y,z,rangeP,path="none",title="none",label="none"):
    
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
    
    myData.isDirectory(path)  
    plt.savefig(os.path.join(path,f"{label}.png"))
    plt.close()
# -----------------------------------------------------------------------------
    
# heatmap + 散布図 -------------------------------------------------------------
def scatter3D_heatmap(x,y,z,var,rangeP,path="none",title="none",label="none"):
    """
    Args
        val: variable for heatmap
    """
    #pdb.set_trace()
    
    sns.set_style("dark")
    
    # normalize variable(var) into 0 to 1
    nlVar = list((var - min(var)) / (max(var) - min(var)))
    
    # blue(<<< 0) -> red
    colors = plt.cm.cool(nlVar)
    # colormap
    colormap = plt.cm.ScalarMappable(cmap=plt.cm.cool)
    colormap.set_array(nlVar)
    
    fig = plt.figure()
    ax = Axes3D(fig)
    fig.colorbar(colormap, shrink=0.75)
    
    #pdb.set_trace()
    ax.scatter(x,y,z,c=colors,marker="o",alpha=0.5)
    
    ax.set_xlim(rangeP[0][ntI],rangeP[1][ntI])
    ax.set_ylim(rangeP[0][tntI],rangeP[1][tntI])
    ax.set_zlim(rangeP[0][ttI],rangeP[1][ttI])

    ax.set_title(f"{title}")
    
    myData.isDirectory(path)  
    plt.savefig(os.path.join(path,f"{label}.png"))
    plt.close()
# -----------------------------------------------------------------------------

# 箱ひげ -----------------------------------------------------------------------
def BoxPlot(x,path='none',title='none',label='none'):
    """
    x: numpy [perticles,id]
    """
    
    sns.set_style("dark")

    # numpy -> list 
    x_tuple = tuple(x[:,i].tolist() for i in np.arange(x.shape[1]))
    
    idx = []
    for i in np.arange(x.shape[-1]): # data [peticles]
        idx = np.append(idx,str(i+1))
    
    fig,ax = plt.subplots()
    ax.boxplot(x_tuple,sym='d',patch_artist=True,boxprops=dict(facecolor='lightblue',color='gray'),medianprops=dict(color='gray'))
    ax.set_xticklabels(idx)
        
    myData.isDirectory(path)  
    plt.savefig(os.path.join(path,f"{label}.png"))
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
def Rireki(gt,pred,path='none',title="none",label="none",isShare=False,isSeparate=True,isResearch=False,iseach=False):
    """
    Args
        gt: gt eq. (best year). [1400,3] 
    """
    
    if isResearch:
        if iseach:
            dists = myData.eachMAEyear(gt,pred)
        else:
            # degree of similatery
            dists = myData.MAEyear(gt,pred)
            
        title = dists
        
    sns.set_style("dark")
    # share gt & pred
    if isShare:
        fig, figInds = plt.subplots(nrows=3, sharex=True)
        for figInd in np.arange(len(figInds)):
            figInds[figInd].plot(np.arange(pred.shape[0]), pred[:,figInd],color="skyblue")
            figInds[figInd].plot(np.arange(gt.shape[0]), gt[:,figInd],color="coral")
        
    #pdb.set_trace()
    if isSeparate:
        colors = ["coral","skyblue","coral","skyblue","coral","skyblue"]
        
        # scalling var.
        predV,gtV = np.zeros([1400,3]),np.zeros([1400,3])
        
        # del first year 
        pred_nk = [s for s in pred[ntI].tolist() if s != 0]
        pred_tnk = [s for s in pred[tntI].tolist() if s != 0]
        pred_tk = [s for s in pred[ttI].tolist() if s != 0]
        
        gt_tk = [s for s in gt[ttI].tolist() if s != 0]
        
        predV[pred_nk,ntI] = 5
        predV[pred_tnk,tntI] = 5
        predV[pred_tk,ttI] = 5
        
        gtV[gt[ntI].tolist(),ntI] = 5
        gtV[gt[tntI].tolist(),tntI] = 5
        gtV[gt_tk,ttI] = 5
        #pdb.set_trace()
        
        # [1400,3]
        plot_data = [gtV[:,ntI],predV[:,ntI],gtV[:,tntI],predV[:,tntI],gtV[:,ttI],predV[:,ttI]]
        # not scalling var. [1400,3]
        #plot_data = [gt[:,ntI],pred[:,ntI],gt[:,tntI],pred[:,tntI],gt[:,ttI],pred[:,ttI]]
        
        fig = plt.figure()
        fig, axes = plt.subplots(nrows=6,sharex="col")
        for row,(color,data) in enumerate(zip(colors,plot_data)):
            axes[row].plot(np.arange(1400), data, color=color)
    
    plt.suptitle(f"{title}", fontsize=8)
    
    myData.isDirectory(path)  
    plt.savefig(os.path.join(path,f"{label}.png"))
    plt.close()
    
    if isResearch:
        return int(np.sum(dists))
# -----------------------------------------------------------------------------