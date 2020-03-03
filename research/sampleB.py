# -*- coding: utf-8 -*-

import sys
import os
import time
import pdb

from numpy.random import *
import numpy as np
import seaborn as sns
import itertools
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot as plt

# atr500 16882 16136 15434
# stand b ----------------
standB1 = int(sys.argv[1])
standB2 = int(sys.argv[2])
standB3 = int(sys.argv[3])
# fileの名前
mode = str(sys.argv[4])
# ------------------------

# file -------------------
paramCSV = f"SampledB_{mode}.csv"
# ------------------------

# change variance-covariance matrix to correlation matrix ---------------------
def cov2corr(cov):
    D = np.diag(np.power(np.diag(cov),-0.5))
    corr = np.dot(np.dot(D,cov),D)
    return corr
# -----------------------------------------------------------------------------
"""
# b:big s:strong m:middle w:weake
bsSigma = [[120,100,100],[100,120,100],[100,100,120]]
mwSigma = [[50,10,10],[10,55,10],[10,10,55]]

print(bsSigma)
print(cov2corr(bsSigma))
"""
# multi gauss -----------------------------------------------------------------
def multinormalSample():
    
    sns.set_style("dark")
    
    mu = [standB1,standB2,standB3]
    bsSigma = [[120,100,100],[100,120,100],[100,100,120]]
    
    bs_mgauss = multivariate_normal(mu,bsSigma,500)
    
    # save txt
    np.savetxt(paramCSV,bs_mgauss,delimiter=",",fmt="%.0f")

    # plot sample scatter
    fig = plt.figure()
    ax = Axes3D(fig)
    
    ax.scatter(bs_mgauss[:,0],bs_mgauss[:,1],bs_mgauss[:,2],c="black",marker="o",alpha=0.5,linewidths=0.5)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    
    ax.set_xlabel("nkB")
    ax.set_ylabel("tnkB")
    ax.set_zlabel("tkB")
    
    #plt.show()
    plt.savefig("bs_sampleB.png")
    plt.close()
# -----------------------------------------------------------------------------

multinormalSample()

# 1D gauss --------------------------------------------------------------------
def normalSample():
    # Sample parameter b for each cell ---
    nkB = np.random.normal(standB1,50,9)
    tnkB = np.random.normal(standB2,50,9)
    tkB = np.random.normal(standB3,50,9)
    # ------------------------------------
    
    # concate sampleB + standB
    sample_nkB = np.append(nkB,standB1).astype(int)
    sample_tnkB = np.append(tnkB,standB2).astype(int)
    sample_tkB = np.append(tkB,standB3).astype(int)
    
    # 組合せ -----------------------------------------------------------
    sampleB1 = []
    sampleB2 = []
    sampleB3 = []
    for i,j,k in itertools.product(sample_nkB,sample_tnkB,sample_tkB):
        sampleB1 = np.append(sampleB1,i)
        sampleB2 = np.append(sampleB2,j)
        sampleB3 = np.append(sampleB3,k)
    # -----------------------------------------------------------------
    # concat to all cell
    sampleB = np.concatenate((sampleB1[np.newaxis],sampleB2[np.newaxis],sampleB3[np.newaxis]),0).T
    # Make *csv
    np.savetxt(paramCSV,sampleB,delimiter=",",fmt="%.0f")
# -----------------------------------------------------------------------------
