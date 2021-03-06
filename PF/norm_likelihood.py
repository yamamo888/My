# -*- coding: utf-8 -*-

import pdb

import matplotlib.pylab as plt
import numpy as np
from scipy.stats import poisson

import makingDataPF as myData


# parameter -------------------------------------------------------------------
nCell = 3
ntI,tntI,ttI = 0,1,2
penaltyNum = 100
safetyNum = 100
# num.of eq.times
nnk,ntk = 9,7
# -----------------------------------------------------------------------------

# 尤度 + 最近傍 (相互地震発生年数)  ---------------------------------------------
def norm_likelihood_eachnearest(y,x,s2=100,standYs=0,time=0):
    gauss,years = np.zeros(nCell+1),np.zeros(nCell) # for penalty

    y_nk = np.array([standYs[ntI]])
    y_tnk = np.array([standYs[tntI]])
    y_tk = np.array([standYs[ttI]])
    
    x_nk = x[ntI][:nnk]
    x_tnk = x[tntI][:nnk]
    x_tk = x[ttI][:ntk]
    
    if x_nk.shape[0] < nnk:
        x_nk = np.hstack([x_nk, np.tile(x_nk[-1], nnk-x_nk.shape[0])])
    if x_tnk.shape[0] < nnk:
        x_tnk = np.hstack([x_tnk, np.tile(x_tnk[-1], nnk-x_tnk.shape[0])])
    if x_tk.shape[0] < ntk:
        x_tk = np.hstack([x_tk, np.tile(x_tk[-1], ntk-x_tk.shape[0])])
  
    #pdb.set_trace()
    # any eq.
    if not y_nk[0] == 0:
        bestX = x_nk[time]
        # mse       
        error_nk = (y_nk-bestX)**2
        
        gauss[ntI] = error_nk
        years[ntI] = bestX

    if not y_tnk[0] == 0:
        
        bestX = x_tnk[time]
        
        error_tnk = (y_tnk-bestX)**2
        
        gauss[tntI] = error_tnk
        years[tntI] = bestX

    if not y_tk[0] == 0:
        
        if time > 4:
            time = time - 1
            
        bestX = x_tk[time]
        
        error_tk = (y_tk-bestX)**2
        
        gauss[ttI] = error_tk
        years[ttI] = bestX
    
    sumgauss = np.cumsum(gauss)[-1]
    
    return gauss, sumgauss, years
# -----------------------------------------------------------------------------
    
# 尤度 + 最近傍 (相互地震発生年数)  ---------------------------------------------
def norm_likelihood_eachnearest_penalty(y,x,s2=100,standYs=0,time=0):
    gauss,years = np.zeros(nCell+1),np.zeros(nCell) # for penalty

    y_nk = np.array([standYs[ntI]])
    y_tnk = np.array([standYs[tntI]])
    y_tk = np.array([standYs[ttI]])
    
    x_nk = x[ntI][:nnk]
    x_tnk = x[tntI][:nnk]
    x_tk = x[ttI][:ntk]
    
    if x_nk.shape[0] < nnk:
        x_nk = np.hstack([x_nk, np.tile(x_nk[-1], nnk-x_nk.shape[0])])
    if x_tnk.shape[0] < nnk:
        x_tnk = np.hstack([x_tnk, np.tile(x_tnk[-1], nnk-x_tnk.shape[0])])
    if x_tk.shape[0] < nnk:
        x_tk = np.hstack([x_tk, np.tile(x_tk[-1], ntk-x_tk.shape[0])])
  
    
     # not eq. in tonakai
    if y_tk[0] == 0:
        # ※同化年数±100年に地震があった場合はpenalty
        penaltyInd = np.where((x_tk>y_tnk-safetyNum)&(x_tk<y_tnk+safetyNum))[0].tolist()
        
        if penaltyInd == []:
            pass
        else:
            pind = penaltyInd[0]
            xpenalty = x_tk[pind]
            # nearist penalty year [1,]
            penalty_tnk = (y_tnk-xpenalty)**2
            # ペナルティ分引くため
            gauss[-1] = penalty_tnk
   
    #pdb.set_trace()
    # any eq.
    if not y_nk[0] == 0:
        bestX = x_nk[time]
        # mse       
        error_nk = (y_nk-bestX)**2
        
        gauss[ntI] = error_nk
        years[ntI] = bestX

    if not y_tnk[0] == 0:
        
        bestX = x_tnk[time]
        
        error_tnk = (y_tnk-bestX)**2
        
        gauss[tntI] = error_tnk
        years[tntI] = bestX

    if not y_tk[0] == 0:
        
        if time > 4:
            time = time - 1
            
        bestX = x_tk[time]
        
        error_tk = (y_tk-bestX)**2
        
        gauss[ttI] = error_tk
        years[ttI] = bestX
    
    sumgauss = np.cumsum(gauss)[-1]
    
    return gauss, sumgauss, years
# -----------------------------------------------------------------------------

# 尤度 + 最近傍 (地震発生年数)  -------------------------------------------------
def norm_likelihood_nearest(y,x,s2=100,standYs=0,time=0):
    gauss,years = np.zeros(nCell+1),np.zeros(nCell) # for penalty

    y_nk = np.array([standYs[ntI]])
    y_tnk = np.array([standYs[tntI]])
    y_tk = np.array([standYs[ttI]])
         
    # any eq.
    if not y_nk[0] == 0:
        # nearist index of gt year [1,]
        bestInd = np.abs(np.asarray(x[ntI]) - y_nk).argmin()
        bestX = x[ntI][bestInd]
        # mse       
        error_nk = (y_nk-bestX)**2
        
        gauss[ntI] = error_nk
        years[ntI] = bestX

    if not y_tnk[0] == 0:
        
        bestInd = np.abs(np.asarray(x[tntI]) - y_tnk).argmin()
        bestX = x[tntI][bestInd]
        
        error_tnk = (y_tnk-bestX)**2
        
        gauss[tntI] = error_tnk
        years[tntI] = bestX

    if not y_tk[0] == 0:
        
        bestInd = np.abs(np.asarray(x[ttI]) - y_tk).argmin()
        bestX = x[ttI][bestInd]
        
        error_tk = (y_tk-bestX)**2
        
        gauss[ttI] = error_tk
        years[ttI] = bestX
    
    sumgauss = np.cumsum(gauss)[-1]
    
    return gauss, sumgauss, years
# -----------------------------------------------------------------------------

# 尤度 + penalty 最近傍 (地震発生年数)  -----------------------------------------
def norm_likelihood_nearest_penalty(y,x,s2=100,standYs=0,time=0):
    gauss,years = np.zeros(nCell+1),np.zeros(nCell) # for penalty

    y_nk = np.array([standYs[ntI]])
    y_tnk = np.array([standYs[tntI]])
    y_tk = np.array([standYs[ttI]])
    
    # not eq. in tonakai
    if y_tk[0] == 0:
        # ※同化年数±100年に地震があった場合はpenalty
        penaltyInd = np.where((x[ttI]>y_tnk-safetyNum)&(x[ttI]<y_tnk+safetyNum))[0].tolist()
        
        if penaltyInd == []:
            pass
        else:
            xpenalty = x[ttI][penaltyInd]
            # nearist penalty year [1,]
            penalty_tnk = np.max((y_tnk-xpenalty)**2)
            # ペナルティ分引くため
            gauss[-1] = penalty_tnk
        
    # any eq.
    if not y_nk[0] == 0:
        # nearist index of gt year [1,]
        bestInd = np.abs(np.asarray(x[ntI]) - y_nk).argmin()
        bestX = x[ntI][bestInd]
        # mse 
        error_nk = (y_nk-bestX)**2
       
        gauss[ntI] = error_nk
        years[ntI] = bestX

    if not y_tnk[0] == 0:
        
        bestInd = np.abs(np.asarray(x[tntI]) - y_tnk).argmin()
        bestX = x[tntI][bestInd]
        
        error_tnk = (y_tnk-bestX)**2
        
        gauss[tntI] = error_tnk
        years[tntI] = bestX

    if not y_tk[0] == 0:
        
        bestInd = np.abs(np.asarray(x[ttI]) - y_tk).argmin()
        bestX = x[ttI][bestInd]
        
        error_tk = (y_tk-bestX)**2
        
        gauss[ttI] = error_tk
        years[ttI] = bestX
    
    sumgauss = np.cumsum(gauss)[-1]
    
    return gauss, sumgauss, years

# -----------------------------------------------------------------------------

# 尤度 + safety & penalty 最近傍 (地震発生年数)  --------------------------------
def norm_likelihood_nearest_safetypenalty(y,x,s2=100,standYs=0,time=0):
    gauss,years = np.zeros(nCell+1),np.zeros(nCell) # for penalty

    y_nk = np.array([standYs[ntI]])
    y_tnk = np.array([standYs[tntI]])
    y_tk = np.array([standYs[ttI]])
    
    # not eq. in tonakai
    if y_tk[0] == 0:
        # ※同化年数±100年に地震があった場合はpenalty
        penaltyInd = np.where((x[ttI]>y_tnk-safetyNum)&(x[ttI]<y_tnk+safetyNum))[0].tolist()
        #pdb.set_trace()
        if penaltyInd == []:
            pass
        else:
            xpenalty = x[ttI][penaltyInd]
            # nearist penalty year [1,]
            penalty_tnk = np.max((y_tnk-xpenalty)**2)
            # ペナルティ分引くため
            gauss[-1] = penalty_tnk
        
    # any eq.
    if not y_nk[0] == 0:
        # nearist index of gt year [1,]
        bestInd = np.abs(np.asarray(x[ntI]) - y_nk).argmin()
        # no.1
        bestX = x[ntI][bestInd]
        delbestX = x[ntI][x[ntI] != bestX]
        # no.2
        best2ind = np.abs(np.asanyarray(delbestX) - y_nk).argmin() 
        best2X = delbestX[best2ind]
        
        # mse for no.1   
        error_nk = (y_nk-bestX)**2
        # mae for no.2
        penalty_nk = (y_nk-best2X)**2
    
        if np.abs(best2X-y_nk) <= penaltyNum:
            # error no.1 + no.2
            gauss[ntI] = error_nk + penalty_nk
        else:
            # error only no.1
            gauss[ntI] = error_nk
            
        years[ntI] = bestX
        
    if not y_tnk[0] == 0:
        
        bestInd = np.abs(np.asarray(x[tntI]) - y_tnk).argmin()
        # no.1
        bestX = x[tntI][bestInd]
        delbestX = x[tntI][x[tntI] != bestX]
        # no.2
        best2ind = np.abs(np.asanyarray(delbestX) - y_tnk).argmin() 
        best2X = delbestX[best2ind]
        
        # mse for no.1   
        error_tnk = (y_tnk-bestX)**2
        # mae for no.2
        penalty_tnk = (y_tnk-best2X)**2
    
        if np.abs(best2X-y_tnk) <= penaltyNum:
            # error no.1 + no.2
            gauss[tntI] = error_tnk + penalty_tnk
        else:
            # error only no.1
            gauss[tntI] = error_tnk
        
        years[tntI] = bestX
        
    if not y_tk[0] == 0:
        
        bestInd = np.abs(np.asarray(x[ttI]) - y_tk).argmin()
        # no.1
        bestX = x[ttI][bestInd]
        delbestX = x[ttI][x[ttI] != bestX]
        # no.2
        best2ind = np.abs(np.asanyarray(delbestX) - y_tk).argmin() 
        best2X = delbestX[best2ind]
        
        # mse for no.1   
        error_tk = (y_tk-bestX)**2
        # mae for no.2
        penalty_tk = (y_tk-best2X)**2
    
        if np.abs(best2X-y_tk) <= penaltyNum:
            # error no.1 + no.2
            gauss[ttI] = error_tk + penalty_tk
        else:
            # error only no.1
            gauss[ttI] = error_tk
        
        years[ttI] = bestX

    sumgauss = np.cumsum(gauss)[-1]
    
    return gauss, sumgauss, years
# -----------------------------------------------------------------------------

# 尤度 (すべての地震発生回数) -----------------------------------------------------
def norm_likelihood_alltimes(y,x):
    
    # gt eq. times
    yt_nk = y[ntI].shape[0]
    yt_tnk = y[tntI].shape[0]
    yt_tk = np.array(y[ttI]).shape[0]
    y_times = yt_nk + yt_tnk + yt_tk
    
    # pred eq. times
    xt_nk = x[ntI].shape[0]
    xt_tnk = x[tntI].shape[0]
    xt_tk = x[ttI].shape[0]
    x_times = xt_nk + xt_tnk + xt_tk

    # poisson for eq. times
    poissons = poisson.pmf(x_times,y_times)
    
    #poisson_nk = poisson.pmf(xt_nk,yt_nk)
    #poisson_tnk = poisson.pmf(xt_tnk,yt_tnk)
    #poisson_tk = poisson.pmf(xt_tk,yt_tk)
    
    #poissons = poisson_nk + poisson_tnk + poisson_tk
    
    return poissons
# -----------------------------------------------------------------------------

