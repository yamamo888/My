# -*- coding: utf-8 -*-

import matplotlib.pylab as plt
import numpy as np
from scipy.stats import poisson


# parameter -------------------------------------------------------------------
nCell = 3
ntI,tntI,ttI = 0,1,2
penaltyNum = 100
safetyNum = 100
# -----------------------------------------------------------------------------

#　尤度 + safety & penalty (地震発生年数) ---------------------------------------
def norm_likelihood_safetypenalty(y,x,s2=100,standYs=0,time=0):
    gauss,years = np.zeros(nCell+1),np.zeros(nCell) # for penalty

    y_nk = np.array([standYs[ntI]])
    y_tnk = np.array([standYs[tntI]])
    y_tk = np.array([standYs[ttI]])
    
    # not eq. in tonakai
    if y_tk[0] == 0:
        
        # ※同化年数±100年に地震があった場合はpenalty
        penaltyInd = np.where((x[ttI]>y_tnk-safetyNum)&(x[ttI]<y_tnk+safetyNum))[0].tolist()

        # not penalty
        if penaltyInd == []:
            pass
        else:
            xpenalty = x[ttI][penaltyInd]
            #pdb.set_trace()
            # ※加算方式
            gauss_pl = np.max(1/(np.sqrt(2*np.pi*s2)) * np.exp(-((y_tnk-xpenalty)/10)**2/(2*s2)))
            # ペナルティ分引くため
            gauss[-1] = -gauss_pl

    # any eq.
    if not y_nk[0] == 0:
        # nearist index of gt year [1,]
        bestInd = np.abs(np.asarray(x[ntI]) - y_nk).argmin()
        bestX = x[ntI][bestInd]
        
        gauss_nk = 1/(np.sqrt(2*np.pi*s2)) * np.exp(-((y_nk-bestX)/10)**2/(2*s2))
        
        # in 100 year -> safety
        if np.abs(bestX-y_nk) < safetyNum:
            gauss[ntI] = gauss_nk
        elif penaltyNum <= np.abs(bestX-y_nk):
            gauss[ntI] = -gauss_nk
     
        years[ntI] = bestX

    if not y_tnk[0] == 0:
        
        bestInd = np.abs(np.asarray(x[tntI]) - y_tnk).argmin()
        bestX = x[tntI][bestInd]
        
        gauss_tnk = 1/(np.sqrt(2*np.pi*s2)) * np.exp(-((y_tnk-bestX)/10)**2/(2*s2))
        
        if np.abs(bestX-y_tnk) < safetyNum:
            gauss[tntI] = gauss_tnk
        elif penaltyNum <= np.abs(bestX-y_tnk):
            gauss[tntI] = -gauss_tnk
        
        years[tntI] = bestX

    if not y_tk[0] == 0:
        
        bestInd = np.abs(np.asarray(x[ttI]) - y_tk).argmin()
        bestX = x[ttI][bestInd]
        
        gauss_tk = 1/(np.sqrt(2*np.pi*s2)) * np.exp(-((y_tk-bestX)/10)**2/(2*s2))
        
        if np.abs(bestX-y_tk) < safetyNum:
            gauss[ttI] = gauss_tk
        elif penaltyNum <= np.abs(bestX-y_tk):
            gauss[ttI] = -gauss_tk
        
        years[ttI] = bestX

    # sum of gauss, [1,]
    sumgauss = np.cumsum(gauss)[-1]
    #pdb.set_trace()
    return gauss, sumgauss, years
# -----------------------------------------------------------------------------
    
# 尤度 (penalty区間地震発生回数) ------------------------------------------------
def norm_likelihood_times(y,x,standYs=0):
    """
    Args
        y: gt eq.
        x: pred eq.
    """
    
    y_nk = np.array([standYs[ntI]])
    y_tnk = np.array([standYs[tntI]])
    y_tk = np.array([standYs[ttI]])
    # gt eq. times
    yt_nk = y_nk.shape[0]
    yt_tnk = y_tnk.shape[0]
    yt_tk = y_tk.shape[0]
    
    # gt eq. times in all cell [1,] == 2 or 3?
    y_times = yt_nk + yt_tnk + yt_tk
    
    # pred eq. for |pred eq.| < +-penalty year
    x_nk = np.where((y_nk - penaltyNum < x[ntI]) & (x[ntI] < y_nk + penaltyNum)) 
    x_tnk = np.where((y_tnk - penaltyNum < x[tntI]) & (x[tntI] < y_tnk + penaltyNum)) 
    x_tk = np.where((y_tk - penaltyNum < x[ttI]) & (x[ttI] < y_tk + penaltyNum)) 
    # pred eq. times
    xt_nk = x_nk[0].shape[0]
    xt_tnk = x_tnk[0].shape[0]
    xt_tk = x_tk[0].shape[0]
    # pred eq. times in all cell
    x_times = xt_nk + xt_tnk + xt_tk
    
    # poisson for eq. times
    poissons = poisson.pmf(x_times,y_times)
    
    return poissons
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
    
    return poissons
# -----------------------------------------------------------------------------
