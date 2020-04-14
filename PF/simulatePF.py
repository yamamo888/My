# -*- coding: utf-8 -*-

import os
import pdb

import matplotlib.pylab as plt
import numpy as np

import norm_likelihood
import PlotPF as myPlot


# Num. of all param Th,V,b
nParam = 3
ntI,tntI,ttI = 0,1,2
thInd,vInd,bInd = 0,1,2
gt_Year = 1400
state_Year = 2000
# save txt path
savetxtPath = "savetxt"


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
    pdb.set_trace()
    return k
# -----------------------------------------------------------------------------

# 重み付き平均 ------------------------------------------------------------------
def FilterValue(x,wNorm):
    return np.mean(wNorm * x)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def simulate(features,y,x,ssYears,mode=0,t=0,pTime=0,sy=0,nP=0,nCell=3,isSavetxt=False,isPlot=False):
    """
    Args
        features: システムモデル値xt. th[1400,perticles,3], V[1400,perticles,3], b[perticles,3]
        y: 観測モデル値yt. [eq.years,]
        x: 地震年数(1400年). [(eq.years zero padding),perticles]
        sy: start of assimilation for perticles.
    """
    pdb.set_trace()
    # 1. 初期化 ----------------------------------------------------------------
    # 状態ベクトル theta,v,year　※1セルの時おかしいかも
    ThVec = np.zeros((nP,nCell))
    VVec = np.zeros((nP,nCell))
    # リサンプリング後の特徴量ベクトル
    xResampled = np.zeros((nParam,nP,nCell))
    # all weight
    maxW = np.zeros((nP))
    # weight for eq. year in each cell + penalty
    gw = np.zeros((nP,nCell+1))
    # weight for eq. times
    pw = np.zeros((nP,nCell+1))
    
    wNorm = np.zeros((nP))
    # -------------------------------------------------------------------------
    #pdb.set_trace()
    # -------------------------------------------------------------------------
    flag = False
    for i in np.arange(nP): # アンサンブル分まわす
        # =====================================================================
        #         尤度計算
        # =====================================================================
        # zero-paddingしたpredを予測した年数だけにする [地震発生年数(可変),]
        yhat_nk = (x[x[:,i,ntI]>0,i,ntI]).astype(int)
        yhat_tnk = (x[x[:,i,tntI]>0,i,tntI]).astype(int)
        yhat_tk = (x[x[:,i,ttI]>0,i,ttI]).astype(int)

        if t > 0:
            # 2000年 + 同化した年数
            standInds = ssYears[i] + state_Year
            # 1400年のスケールに合わせる
            yhat_nk = yhat_nk - standInds
            yhat_tnk = yhat_tnk - standInds
            yhat_tk = yhat_tk - standInds
            
        yhat = [yhat_nk,yhat_tnk,yhat_tk]
        #pdb.set_trace()
        # 尤度は地震発生年数、重みとかけるのは状態ベクトル
        # 2.c & 2.d 各粒子の尤度と重み -------------------------------------------
        standYs = [y[ntI][t],y[tntI][t],y[ttI][t]]

        if mode == 3 or mode == 4 or mode == 5:
            weight, maxweight, years = norm_likelihood.norm_likelihood_safetypenalty(y,yhat,standYs=standYs,time=t)
            
            gw[i] = weight
            maxW[i] = maxweight
            
        if mode == 6 or mode == 7 or mode == 9 or mode == 10:
            gweight, gmaxweight, years = norm_likelihood.norm_likelihood_safetypenalty(y,yhat,standYs=standYs,time=t)
            pweight = norm_likelihood.norm_likelihood_times(y,yhat,standYs=standYs)
            
            gw[i] = gweight
            pw[i] = pweight
            
            maxW[i] = gmaxweight + pweight
        
        if mode == 8: # No.9-55
            gweight, gmaxweight, years = norm_likelihood.norm_likelihood_safetypenalty(y,yhat,standYs=standYs,time=t)
            pweight = norm_likelihood.norm_likelihood_alltimes(y,yhat)
        
            gw[i] = gweight
            pw[i] = pweight
            
            maxW[i] = gmaxweight + pweight
            
        if mode == 11 or mode == 12:
            weight = norm_likelihood.norm_likelihood_times(y,yhat,standYs=standYs)
            maxW[i] = weight
            years = np.array(standYs)
            
        # ---------------------------------------------------------------------
        #pdb.set_trace()
        for indY,indC in zip(years,[ntI,tntI,ttI]):
            # 各セルで尤度の一番高かった年数に合わせる 1400 -> 1
            # ※ 別々の同化タイミングになる
            # ※地震が発生していないときは、tonankaiの地震発生年数を採用
            # ※違う年数でも同じ値の時あり
            if int(indY) == 0: # for tk
                ThVec[i,indC] = features[0][int(years[tntI]),i,indC]
                VVec[i,indC] = features[1][int(years[tntI]),i,indC]
            else:
                ThVec[i,indC] = features[0][int(years[indC]),i,indC]
                VVec[i,indC] = features[1][int(years[indC]),i,indC]

        if not flag:
            yearInds = years
            flag = True
        else:
            # [perticle,3]
            yearInds = np.vstack([yearInds,years])
    #pdb.set_trace()
    # 規格化 -------------------------------------------------------------------
    if mode == 3 or mode == 9 or mode == 10 or mode == 11 or mode == 12:
        # [perticles,]
        wNorm = maxW/np.sum(maxW)
        
    elif mode == 4 or mode == 5 or mode == 6 or mode == 7 or mode == 8:
        # scalling maximum(M),minimum(m)
        xmax = np.max(maxW)
        xmin = np.min(maxW)
        
        m = 1/(np.sqrt(2*np.pi*100)) * np.exp(-((standYs[tntI]-gt_Year)/10)**2/(2*100))
        M = xmax + m
        
        # normalization
        scaledW =  ((maxW - xmin)*(M - m) / (xmax - xmin)) + m
        
        wNorm = scaledW/np.sum(scaledW)    
    # -------------------------------------------------------------------------

    # save year & likelihood txt ----------------------------------------------
    if isSavetxt:
        """
        if mode == 4 or mode == 5 or mode == 6 or mode == 7 or mode == 8:
            np.savetxt(os.path.join(savetxtPath,"lh",f"sum_lh_{t}.txt"),scaledW,fmt="%4f")
        else:    
            np.savetxt(os.path.join(savetxtPath,"lh",f"sum_lh_{t}.txt"),maxW,fmt="%4f")
        """
        if mode == 6 or mode == 7 or mode == 8 or mode == 9 or mode == 10:            
            np.savetxt(os.path.join(savetxtPath,"lh",f"lh_p_{t}.txt"),pw)
            np.savetxt(os.path.join(savetxtPath,"lh",f"lh_g_{t}.txt"),gw)
        elif mode == 11 or mode == 12:
            np.savetxt(os.path.join(savetxtPath,"lh",f"sum_lh_{t}.txt"),maxW,fmt="%4f")
        else:
            np.savetxt(os.path.join(savetxtPath,"lh",f"lh_{t}.txt"),gw)
            
        np.savetxt(os.path.join(savetxtPath,"bestyear",f"by_{t}.txt"),yearInds,fmt="%d")
    # -------------------------------------------------------------------------
    pdb.set_trace()
    # =========================================================================
    #         リサンプリング
    # =========================================================================
    initU = np.random.uniform(0,1/nP)

    # ※3セル同じ組み合わせのbが選ばれる
    # index for resampling
    k = resampling(initU,wNorm,nP=nP)
        
    if mode == 3 or mode == 4 or mode == 7 or mode == 10 or mode == 12:
        
        # system noise --------------------------------------------------------
        # ※元の値と足してもマイナスになるかも
        # array[cell,perticles] V & theta parameterがすべて同じ組み合わせになるのを防ぐため
        Thnoise = np.array([np.random.normal(0,0.01*np.mean(ThVec[:,cell]),nP) for cell in np.arange(nCell)])
        Vnoise = np.array([np.random.normal(0,0.01*np.mean(VVec[:,cell]),nP) for cell in np.arange(nCell)])
        #bnoise = np.array([np.random.normal(0,0.01*np.mean(features[bInd][:,cell]),nP) for cell in np.arange(nCell)])
        # ---------------------------------------------------------------------
    
        xResampled[thInd] = ThVec[k] + np.abs(Thnoise).T
        xResampled[vInd] = VVec[k] + np.abs(Vnoise).T
        xResampled[bInd] = features[bInd][k]
        # Add noise
        #xResampled[bInd] = features[bInd][k] + np.abs(bnoise).T
        
        updatesy = sy[k]
    
    if mode == 5 or mode == 6 or mode == 8 or mode == 9 or mode == 11:
        #pdb.set_trace()
        # index for mean theta,V,b
        indmu = np.argmax(wNorm)
        # highest of norm likelihood (index) for mean & sigma
        muB = features[bInd][indmu] * 1000000
        Bin = 10
        sigmaB = [[0,Bin,Bin],[Bin,0,Bin],[Bin,Bin,0]]
        
        # 尤度の高いところにばらまく
        # system noise --------------------------------------------------------
        # [cell,perticle]
        Thnoise = np.array([np.random.normal(0,0.01*np.mean(ThVec[:,cell]),nP) for cell in np.arange(nCell)])
        Vnoise = np.array([np.random.normal(0,0.01*np.mean(VVec[:,cell]),nP) for cell in np.arange(nCell)])
        # [perticle,cell]
        bnoise = np.random.multivariate_normal(muB,sigmaB,nP)
        # ---------------------------------------------------------------------
        # [perticle,cell]
        xResampled[thInd] = ThVec[k] + np.abs(Thnoise).T
        xResampled[vInd] = VVec[k] + np.abs(Vnoise).T
        xResampled[bInd] = np.abs(bnoise)*0.000001
        xResampled[bInd][0] = features[bInd][indmu] # No.12.5
        #xResampled[bInd] = features[bInd][k] + np.abs(bnoise)*0.000001
        # 尤度が一番高いperticleの年数に合わせる
        updatesy = np.array(sy[indmu].tolist()*nP)[:,np.newaxis]
    #pdb.set_trace()
    
    print(f"---- 【{t}】 times ----\n")
    # 発生年数 plot ------------------------------------------------------------
    if isPlot:
        myPlot.NumberLine(standYs,yearInds,label=f"best_years_{t}")
        
        if mode == 4 or mode == 5 or mode == 6 or mode == 7 or mode == 8 or mode == 9 or mode == 10 or mode == 11 or mode == 12:
            xt = features[bInd][k]
            np.savetxt(os.path.join(savetxtPath,'B',f'{t}.txt'),xt,fmt='%6f')
    # -------------------------------------------------------------------------
    
    return xResampled, yearInds.astype(int), updatesy, k


