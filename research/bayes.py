# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

import pdb

import numpy as np
import matplotlib.pylab as plt

from bayes_opt import BayesianOptimization

import warnings
warnings.filterwarnings('ignore')

isPlot = False

# paramters -------------------------------------------------------------------
# 下限・上限
xmin = -5.0
xmax = 5.0
ymin = -5.0
ymax = 5.0
zmin = -5.0
zmax = 5.0

# パラメータの下限・上限
# 連続値の場合は、事前分布指定可（default:連続一様分布、対数一様分布も指定可）
#pbounds = {"x":(xmin,xmax)}
pbounds = {"x":(xmin,xmax),"y":(ymin,ymax),"z":(zmin,zmax)}

# -----------------------------------------------------------------------------

# function --------------------------------------------------------------------
#def f(x):
def f(x,y,z):
    #return -(x**4 - 20 * x**2 + 10 * x) + 300
    return -(x**4 - 20 * y**2 + 10 * z) + 300
# -----------------------------------------------------------------------------

# optimizer -------------------------------------------------------------------
opt = BayesianOptimization(f=f,pbounds=pbounds)
opt.maximize(init_points=3,n_iter=10)
# -----------------------------------------------------------------------------

# result ----------------------------------------------------------------------
res = opt.res
best_res = opt.max
#print(res)
#print(best_res)
# -----------------------------------------------------------------------------
#pdb.set_trace()
if isPlot:
    # plot ------------------------------------------------------------------------
    x = np.arange(xmin,xmax)
    plt.plot(x,f(x))
    # ※エラーはくかも
    predX = [p['params']['x'] for p in opt.res]
    predY = [f(p['params']['x']) for p in opt.res]
    
    plt.scatter(predX,predY,color="g",s=20,zorder=10)
    mean,sigma = opt._gp.predict(x.reshape(-1,1),return_std=True)
    plt.plot(x,mean)
    #plt.fill_between(x,mean+sigma,mean-sigma,alpha=0.1)
    pdb.set_trace()
    plt.grid()
    plt.show()
    # -----------------------------------------------------------------------------
