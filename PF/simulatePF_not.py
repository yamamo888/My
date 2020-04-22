# -*- coding: utf-8 -*-

import os
import pdb

import matplotlib.pylab as plt
import numpy as np

import norm_likelihood
import PlotPF as myPlot
import makingDataPF as myData

# Num. of all param Th,V,b
nParam = 1
ntI,tntI,ttI = 0,1,2
thInd,vInd,bInd = 0,1,2
gt_Year = 1400
state_Year = 2000
# -----------------------------------------------------------------------------

# path ------------------------------------------------------------------------
# save txt path
savetxtPath = "savetxt"
# save img path
imgPath = "images"
lhPath = "lhs"
# -----------------------------------------------------------------------------

# 逆関数 -----------------------------------------------------------------------
def InvF(WC,idex,u):
    """
    Args
        WC: ex) array([1,2,3]) -> array([1,3,6])
    """
    if np.any(WC<u) == False:
        return 0
    k = np.max(idex[WC<u])
    
    return k+1
# -----------------------------------------------------------------------------

# 層化サンプリング -----------------------------------------------------------------
def resampling(initU,weights,nP=0):
    # weights of index
    idx = np.asanyarray(range(nP))
    thres = [1/nP*i+initU for i in range(nP)]
    wc = np.cumsum(weights)
    k = np.asanyarray([InvF(wc,idx,val) for val in thres])
    #pdb.set_trace()
    return k
# -----------------------------------------------------------------------------

# 重み付き平均 ------------------------------------------------------------------
def FilterValue(x,wNorm):
    return np.mean(wNorm * x)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def simulate(features,y,x,mode=0,t=0,pTime=0,nP=0,nCell=3,isSavetxt=False,isPlot=False):
    """
    Args
        features: システムモデル値xt. th[1400,perticles,3], V[1400,perticles,3], b[perticles,3]
        y: 観測モデル値yt. [eq.years,]
        x: 地震年数(1400年). [(eq.years zero padding),perticles]
    """
    #pdb.set_trace()
    # 1. 初期化 ----------------------------------------------------------------
    # 状態ベクトル theta,v,year　※1セルの時おかしいかも
    # リサンプリング後の特徴量ベクトル
    xResampled = np.zeros((nP,nCell))
    # all norm-likelihood
    maxgW = np.zeros((nP))
    maxpW = np.zeros((nP))
    # weight for eq. year in each cell + penalty
    gw = np.zeros((nP,nCell+1))
    # weight for eq. times
    pw = np.zeros((nP,nCell+1))
    # -------------------------------------------------------------------------
    #pdb.set_trace()
    # -------------------------------------------------------------------------
    flag = False
    for i in np.arange(nP): # アンサンブル分まわす
        # =====================================================================
        #         尤度計算
        # =====================================================================
        # zero-paddingしたpredを予測した年数だけにする [地震発生年数(可変),]
        # predicted eq.year
        yhat_nk = (x[x[:,i,ntI]>0,i,ntI]).astype(int)
        yhat_tnk = (x[x[:,i,tntI]>0,i,tntI]).astype(int)
        yhat_tk = (x[x[:,i,ttI]>0,i,ttI]).astype(int)
        yhat = [yhat_nk,yhat_tnk,yhat_tk]
        
        #pdb.set_trace()
        # 尤度は地震発生年数、重みとかけるのは状態ベクトル
        # 2.c & 2.d 各粒子の尤度と重み -------------------------------------------
        # ground truth eq.year (time=t)
        standYs = [y[ntI][t],y[tntI][t],y[ttI][t]]
       
        # nearist -----
        if mode == 100:
            weight, maxweight, years = norm_likelihood.norm_likelihood_nearest(y,yhat,standYs=standYs,time=t)
        
            gw[i] = weight
            maxgW[i] = maxweight
        
        if mode == 13:
            weight, maxweight, years = norm_likelihood.norm_likelihood_nearest_penalty(y,yhat,standYs=standYs,time=t)
        
            gw[i] = weight
            maxgW[i] = maxweight
        
        if mode == 101:
            weight, maxweight, years = norm_likelihood.norm_likelihood_nearest_safetypenalty(y,yhat,standYs=standYs,time=t)
                 
            gw[i] = weight
            maxgW[i] = maxweight
        
        if mode == 102:
            gweight, gmaxweight, years = norm_likelihood.norm_likelihood_nearest_safetypenalty(y,yhat,standYs=standYs,time=t)
            pweight = norm_likelihood.norm_likelihood_alltimes(y,yhat)
        
            gw[i] = gweight
            pw[i] = pweight
            
            maxgW[i] = gmaxweight
            maxpW[i] = pweight 
        # ---------------------------------------------------------------------
        
        # for plot ------------------------------------------------------------
        if not flag:
            yearInds = years
            flag = True
        else:
            # [perticle,3]
            yearInds = np.vstack([yearInds,years])
        # ---------------------------------------------------------------------
        
    # 規格化 -------------------------------------------------------------------
    # only eq.years
    if mode == 100 or mode == 101:
        # 全セルがぴったりの時
        if any(maxgW==0):
            zeroind = np.where(maxgW==0)[0].tolist()
            maxgW[zeroind] = -1
        
        tmpgW = 1/-maxgW
        wNorm = tmpgW/np.sum(tmpgW)
        
    # eq.years & eq.times
    elif mode == 102:    
        if any(maxgW==0):
            zeroind = np.where(maxgW==0)[0].tolist()
            maxgW[zeroind] = -1
        
        tmpgW = 1/-maxgW
        
        maxW = tmpgW + maxpW 
        wNorm = maxW/np.sum(maxW)
    
    print(wNorm)
    # -------------------------------------------------------------------------
    #pdb.set_trace()
    # =========================================================================
    #         リサンプリング
    # =========================================================================
    initU = np.random.uniform(0,1/nP)

    # ※3セル同じ組み合わせのbが選ばれる
    # index for resampling
    k = resampling(initU,wNorm,nP=nP)
    
    # not update b var. ----
    if mode == 90:
        xResampled = features[k] 
    
    # simple var. ----
    if mode == 100:
        # system noise --------------------------------------------------------
        # ※元の値と足してもマイナスになるかも
        # array[cell,perticles] V & theta parameterがすべて同じ組み合わせになるのを防ぐため
        bnoise = np.array([np.random.normal(0,0.01*np.mean(features[:,cell]),nP) for cell in np.arange(nCell)])
        # ---------------------------------------------------------------------
    
        # Add noise
        xResampled = features[k] + np.abs(bnoise).T
    
    # 尤度工夫 var. ----
    if mode == 101 or mode == 102:
        #pdb.set_trace()
        # index for mean theta,V,b
        muind = np.argmax(wNorm)
        # highest of norm likelihood (index) for mean & sigma
        muB = features[bInd][muind] * 1000000
        # variable & co-variable matrix (xy,yz,zx)
        Bin = 10
        sigmaB = [[0,Bin,Bin],[Bin,0,Bin],[Bin,Bin,0]]
        
        # 尤度の1番高いところを中心に、次のbの分布決定
        # system noise --------------------------------------------------------
        # [perticle,cell]
        bnoise = np.random.multivariate_normal(muB,sigmaB,nP)
        # ---------------------------------------------------------------------
        # [perticle,cell]
        xResampled = np.abs(bnoise)*0.000001
        xResampled[0] = features[muind]
    
    #pdb.set_trace()
    
    print(f"---- 【{t}】 times ----\n")
    # 発生年数 plot ------------------------------------------------------------
    if isPlot:
        nlpath = os.path.join(imgPath,f'numlines_{mode}')
        myData.isDirectory(nlpath)
        myPlot.NumberLine(standYs,yearInds,path=nlpath,label=f"best_years_{t}")
    # -------------------------------------------------------------------------
    
    # save year & likelihood txt ----------------------------------------------
    if isSavetxt:
     
        # nearist ----
        lhpath = os.path.join(savetxtPath,f"lh_{mode}")
        myData.isDirectory(lhpath)
        if mode == 100 or 101:
            np.savetxt(os.path.join(lhpath,f"lh_{t}.txt"),gw)
            np.savetxt(os.path.join(lhpath,f"sum_lh_{t}.txt"),maxgW)
        else:
            np.savetxt(os.path.join(lhpath,f"lh_g_{t}.txt"),gw)
            np.savetxt(os.path.join(lhpath,f"lh_p_{t}.txt"),pw)
            np.savetxt(os.path.join(lhpath,f"sum_lh_g_{t}.txt"),maxgW)
            np.savetxt(os.path.join(lhpath,f"sum_lh_p_{t}.txt"),maxpW)
        
        np.savetxt(os.path.join(lhpath,f"w_{t}.txt"),wNorm)
        
        # Save param b
        xt = features[k]
        bpath = os.path.join(savetxtPath,f'B_{mode}')
        myData.isDirectory(bpath)
        np.savetxt(os.path.join(bpath,f'{t}.txt'),xt,fmt='%6f')    
    # -------------------------------------------------------------------------
    
    return xResampled, k


