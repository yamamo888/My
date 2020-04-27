# -*- coding: utf-8 -*-

import os
import pdb

import matplotlib.pylab as plt
import numpy as np

import norm_likelihood
import PlotPF as myPlot
import makingDataPF as myData

# Num. of all param Th,V,b
nParam = 3
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
def simulate(features,y,x,ssYears,mode=0,t=0,pTime=0,sy=0,nP=0,nCell=3,isSavetxt=False,isPlot=False):
    """
    Args
        features: システムモデル値xt. th[1400,perticles,3], V[1400,perticles,3], b[perticles,3]
        y: 観測モデル値yt. [eq.years,]
        x: 地震年数(1400年). [(eq.years zero padding),perticles]
        sy: start of assimilation for perticles.
    """
    #pdb.set_trace()
    # 1. 初期化 ----------------------------------------------------------------
    # 状態ベクトル theta,v,year　※1セルの時おかしいかも
    ThVec = np.zeros((nP,nCell))
    VVec = np.zeros((nP,nCell))
    # リサンプリング後の特徴量ベクトル
    xResampled = np.zeros((nParam,nP,nCell))
    # all norm-likelihood
    maxgW = np.zeros((nP))
    maxpW = np.zeros((nP))
    # weight for eq. year in each cell + penalty
    gw = np.zeros((nP,nCell+1))
    # weight for eq. times
    pw = np.zeros((nP,nCell+1))
    # weight
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
       
        # nearist -----
        # if mode == 'near' or mode == 'simple' or mode == 'b_near'
        if mode == 0:
            weight, maxweight, years = norm_likelihood.norm_likelihood_nearest(y,yhat,standYs=standYs,time=t)
        
            gw[i] = weight
            maxgW[i] = maxweight
         
        # if mode == 'sp_time_near' or mode == 'b_sp_time_near'
        elif mode == 3 or mode == 2:
            gweight, gmaxweight, years = norm_likelihood.norm_likelihood_nearest_safetypenalty(y,yhat,standYs=standYs,time=t)
            pweight = norm_likelihood.norm_likelihood_alltimes(y,yhat)
        
            gw[i] = gweight
            pw[i] = pweight
            
            maxgW[i] = gmaxweight
            maxpW[i] = pweight
        
        elif mode == 4:
            weight, maxweight, years = norm_likelihood.norm_likelihood_eachnearest(y,yhat,standYs=standYs,time=t)
        
            gw[i] = weight
            maxgW[i] = maxweight
        
        elif mode == 5 or mode == 6:
            weight, maxweight, years = norm_likelihood.norm_likelihood_eachnearest_penalty(y,yhat,standYs=standYs,time=t)
        
            gw[i] = weight
            maxgW[i] = maxweight
        
        elif mode == 7:
            pweight = norm_likelihood.norm_likelihood_alltimes(y,yhat)
            gweight, gmaxweight, years = norm_likelihood.norm_likelihood_eachnearest_penalty(y,yhat,standYs=standYs,time=t)
        
            gw[i] = gweight
            pw[i] = pweight
        
            maxgW[i] = gmaxweight
            maxpW[i] = pweight
        
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
    # only eq.years
    if mode == 0 or mode == 4 or mode == 5 or mode == 6:
        # 全セルがぴったりの時
        if any(maxgW==0):
            zeroind = np.where(maxgW==0)[0].tolist()
            maxgW[zeroind] = 1
        
        tmpgW = 1/maxgW
        wNorm = tmpgW/np.sum(tmpgW)
        
    # eq.years & eq.times
    elif mode == 3 or mode == 2 or mode == 7:    
        if any(maxgW==0):
            zeroind = np.where(maxgW==0)[0].tolist()
            maxgW[zeroind] = 1
        
        tmpgW = 1/maxgW
        
        maxW = tmpgW + maxpW 
        wNorm = maxW/np.sum(maxW)
    
    # -------------------------------------------------------------------------
    #pdb.set_trace()
    # =========================================================================
    #         リサンプリング
    # =========================================================================
    initU = np.random.uniform(0,1/nP)

    # ※3セル同じ組み合わせのbが選ばれる
    # index for resampling
    k = resampling(initU,wNorm,nP=nP)
 
    # simple var. ----
    if mode == 0:
        # system noise --------------------------------------------------------
        # ※元の値と足してもマイナスになるかも
        # array[cell,perticles] V & theta parameterがすべて同じ組み合わせになるのを防ぐため
        Thnoise = np.array([np.random.normal(0,0.01*np.mean(ThVec[:,cell]),nP) for cell in np.arange(nCell)])
        Vnoise = np.array([np.random.normal(0,0.01*np.mean(VVec[:,cell]),nP) for cell in np.arange(nCell)])
        bnoise = np.array([np.random.normal(0,0.01*np.mean(features[bInd][:,cell]),nP) for cell in np.arange(nCell)])
        # ---------------------------------------------------------------------
    
        xResampled[thInd] = ThVec[k] + np.abs(Thnoise).T
        xResampled[vInd] = VVec[k] + np.abs(Vnoise).T
        # Add noise
        xResampled[bInd] = features[bInd][k] + np.abs(bnoise).T
        
        updatesy = sy[k]
    
    # 尤度工夫 var. ----
    # if mode == 'sp_alltime' or mode == 'b_sp_nl' or mode == 'b_sp_time_nl':
    elif mode == 3 or mode == 4 or mode == 5 or mode == 7:
        # index for mean theta,V,b
        muind = np.argmax(wNorm)
        # highest of norm likelihood (index) for mean & sigma
        muB = features[bInd][muind] * 1000000
        # variable & co-variable matrix (xy,yz,zx)
        Bin = 10
        sigmaB = [[0,Bin,Bin],[Bin,0,Bin],[Bin,Bin,0]]
        
        # 尤度の1番高いところを中心に、次のbの分布決定
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
        xResampled[bInd][0] = features[bInd][muind]
        # 尤度が一番高いperticleの年数に合わせる
        updatesy = np.array(sy[muind].tolist()*nP)[:,np.newaxis]
    
    # not update b var. ----
    elif mode == 2 or mode == 6:
        # system noise --------------------------------------------------------
        # ※元の値と足してもマイナスになるかも
        # array[cell,perticles] V & theta parameterがすべて同じ組み合わせになるのを防ぐため
        Thnoise = np.array([np.random.normal(0,0.01*np.mean(ThVec[:,cell]),nP) for cell in np.arange(nCell)])
        Vnoise = np.array([np.random.normal(0,0.01*np.mean(VVec[:,cell]),nP) for cell in np.arange(nCell)])
        # ---------------------------------------------------------------------
    
        xResampled[thInd] = ThVec[k] + np.abs(Thnoise).T
        xResampled[vInd] = VVec[k] + np.abs(Vnoise).T
        xResampled[bInd] = features[bInd][k]
        
        updatesy = sy[k]
    
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
        if mode == 0 or mode == 4 or mode == 5 or mode == 6:
            np.savetxt(os.path.join(lhpath,f"lh_{t}.txt"),gw)
            np.savetxt(os.path.join(lhpath,f"sum_lh_{t}.txt"),maxgW)
        else:
            np.savetxt(os.path.join(lhpath,f"lh_g_{t}.txt"),gw)
            np.savetxt(os.path.join(lhpath,f"lh_p_{t}.txt"),pw)
            np.savetxt(os.path.join(lhpath,f"sum_lh_g_{t}.txt"),maxgW)
            np.savetxt(os.path.join(lhpath,f"sum_lh_p_{t}.txt"),maxpW)
        
        np.savetxt(os.path.join(lhpath,f"w_{t}.txt"),wNorm)
        
        # Save param b
        xt = features[bInd][k]
        bpath = os.path.join(savetxtPath,f'B_{mode}')
        myData.isDirectory(bpath)
        np.savetxt(os.path.join(bpath,f'{t}.txt'),xt,fmt='%6f')    
    # -------------------------------------------------------------------------
    
    return xResampled, yearInds.astype(int), updatesy, k


