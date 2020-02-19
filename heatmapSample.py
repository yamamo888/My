# -*- coding: utf-8 -*-

import os
import pickle
import pdb

import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot as plt


# path ------------------------------------------------------------------------
dataPath = "sample190" 
filePath = "path_190_atr500_sample10.txt"
# -----------------------------------------------------------------------------

# Reading txt -----------------------------------------------------------------
# path
paths = open(os.path.join(dataPath,filePath)).readlines()
# degree of similateries
degreeSim = open(os.path.join(dataPath,"DS_190_atr500_sample10.txt")).readlines()
# -----------------------------------------------------------------------------

# Making heatmap --------------------------------------------------------------
def heatmap3D(x,y,z,var):

    sns.set_style("dark")

    normalVal =  (var - min(var)) / (max(var) - min(var))
    # gradetion
    colorsHot = plt.cm.hot_r(normalVal)
    
    # color-bar
    colorsMap = plt.cm.ScalarMappable(cmap=plt.cm.hot_r)
    colorsMap.set_array(normalVal)
    
    fig = plt.figure()
    ax = Axes3D(fig)
    
    ax.scatter(x,y,z,c=colorsHot,marker="o",alpha=0.5,linewidths=0.5)
    fig.colorbar(colorsMap,shrink=0.5)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    
    ax.set_xlabel("nkB")
    ax.set_ylabel("tnkB")
    ax.set_zlabel("tkB")
    
    plt.show()
    plt.close()
# -----------------------------------------------------------------------------

# param b ---------------------------------------------------------------------
flag = False
for path,ds in zip(paths,degreeSim):
    # only param b
    tmpB = path.split("\\")[-1].split(".")[0].split("_")
    # degree of similatery
    tmpDS = np.round(float(ds),4)
    
    # for each cell
    b1 = int(tmpB[0])
    b2 = int(tmpB[1])
    b3 = int(tmpB[2])
    
    if not flag:
        nkB = np.array([b1])
        tnkB = np.array([b2])
        tkB = np.array([b3])
        dSim = np.array([tmpDS])
        flag = True
    else:
        # [len(paths),]
        nkB = np.hstack([nkB,b1])
        tnkB = np.hstack([tnkB,b2])
        tkB = np.hstack([tkB,b3])
        dSim = np.hstack([dSim,tmpDS])
# -----------------------------------------------------------------------------

heatmap3D(nkB,tnkB,tnkB,dSim)

    



