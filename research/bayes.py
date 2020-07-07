# -*- coding: utf-8 -*-

import os
import pickle
import time
import glob
import shutil
import csv
import argparse
import random
from decimal import Decimal
import pdb

import numpy as np
import matplotlib.pylab as plt
from bayes_opt import BayesianOptimization

import DC as myData
import makingDataPF as myDataPF
import PlotPF as myPlot

import warnings
warnings.filterwarnings('ignore')

# Argument --------------------------------------------------------------------
parser = argparse.ArgumentParser()
#　Num. of iter
parser.add_argument('--itrNum', type=int, default=100)
# Num. of traial
parser.add_argument('--trID', type=int, default=0)
# mode
parser.add_argument('--mode', required=True, choices=['paramb','Kij'])
# 引数展開
args = parser.parse_args()

itrNum = args.itrNum
trID = args.trID
mode = args.mode
# -----------------------------------------------------------------------------

# path ------------------------------------------------------------------------
dirPath = 'bayes'
#featuresPath = "features"
featuresPath = 'nankairirekifeature'
logsPath = 'logs'
# for paramter & targer
savedirPath = f'BO_{trID}'
# for logs
savedlogPath = f'savedPD_{trID}'
paramCSV = 'bayesParam.csv'
batFile = 'PyToCBayes.bat'
filePath = '*txt'
Kijfile = 'K8_AV2.txt'
# -----------------------------------------------------------------------------

# paramters -------------------------------------------------------------------

slip = 1
aYear = 1400
ntI,tntI,ttI = 0,1,2
nCell = 3
# num.of epoch
nEpoch = 5
# range b
mt = 1000000

## for researching Kij (default) ##
# sampling range
K22samples = np.arange(-139157.4154072-10000, -139157.4154072+10000)
K22 = random.sample(K22samples.tolist(),2)
K22min = Decimal(str(np.min(K22))).quantize(Decimal('0.0000001')) # ※ 小数点以下7桁しか受け付けてくれへん
K22max = Decimal(str(np.max(K22))).quantize(Decimal('0.0000001'))
defaultK22= [K22min, K22max]
#pdb.set_trace()
##
print(f'default >> {K22}')
## for researching for param b (default) ##
# range of under & over in parameter
nkmin,nkmax = 0.014449,0.015499
tnkmin,tnkmax = 0.012,0.014949
tkmin,tkmax = 0.012,0.0135
# for all search
nkmin,nkmax = 0.011,0.0165
tnkmin,tnkmax = 0.011,0.0165
tkmin,tkmax = 0.011,0.0170

defaultpdB = [[nkmin,nkmax],[tnkmin,tnkmax],[tkmin,tkmax]]
##

# -----------------------------------------------------------------------------

# ファイル存在確認 ----------------------------------------------------------------
def isDirectory(fpath):
    # 'path' exist -> True
    isdir = os.path.exists(fpath)
    # -> False
    if not isdir:
        os.makedirs(fpath)
#------------------------------------------------------------------------------

# Prior distribution ----------------------------------------------------------
def setPriorDistribution(pd):
    
    if mode == 'paramb':
        nkmin,nkmax = pd[ntI][0],pd[ntI][1]
        tnkmin,tnkmax = pd[tntI][0],pd[tntI][1]
        tkmin,tkmax = pd[ttI][0],pd[ttI][1]
        
        # 連続値の場合は、事前分布指定可（default:連続一様分布、対数一様分布も指定可）
        pbounds = {"b1":(nkmin,nkmax),"b2":(tnkmin,tnkmax),"b3":(tkmin,tkmax)}
    
    elif mode == 'Kij':
        
        pbounds = {"K22":(pd[0],pd[1])}
    
    return pbounds
# -----------------------------------------------------------------------------

# reading files ---------------------------------------------------------------
def readlogsFiles():
    
    # reading predict logs ----------------------------------------------------
    #fileName = f"{cnt}_*"
    fileName = f"*txt"
    filePath = os.path.join(logsPath,dirPath,fileName)
    files = glob.glob(filePath)
    # -------------------------------------------------------------------------
    
    return files
# -----------------------------------------------------------------------------

# making logs -----------------------------------------------------------------
def makeLog(b1=np.array(0.011),b2=np.array(0.011),b3=np.array(0.011)):
    #pdb.set_trace()
    # save param b ------------------------------------------------------------
    params = np.concatenate((b1[np.newaxis],b2[np.newaxis],b3[np.newaxis]),0)[:,np.newaxis]
    np.savetxt(paramCSV,params.T*mt,delimiter=",",fmt="%.0f")
    # -------------------------------------------------------------------------
    
    # call bat ----------------------------------------------------------------
    lockPath = "Lock.txt"
    lock = str(1)
    with open(lockPath,"w") as fp:
        fp.write(lock)
    
    os.system(batFile)
    
    sleepTime = 3
    # lockファイル作成時は停止
    while True:
        time.sleep(sleepTime)
        if os.path.exists(lockPath)==False:
            break
    # -------------------------------------------------------------------------
# -----------------------------------------------------------------------------
 
# function (paramb) -----------------------------------------------------------
def bfunc(b1,b2,b3):
    
    # simulation
    makeLog(b1=b1,b2=b2,b3=b3)
    
    # reading gt & logs
    logfile = readlogsFiles()[0]
    print(logfile)
    # U:[None,10], B:[3,]
    U,B = myData.loadABLV(logfile)
    deltaU = myData.convV2YearlyData(U)
    # each mse var.
    maxSim = myData.MinErrorNankai(deltaU,mode=3)
    maxSim = 1/maxSim
    
    # Delate logfile
    os.remove(logfile)
    
    return maxSim
# -----------------------------------------------------------------------------

# function (Kij) --------------------------------------------------------------
def Kijfunc(K22):
    
    # Updata Kij
    # reading Kij file(K8_AV)
    Kij = open(Kijfile).readlines()
    # 上書きされるKij (別のKijを変えたいときは+以降も変える必要あり)
    K22 = Decimal(str(K22)).quantize(Decimal('0.0000001'))
    Kij[17] = str(K22) + ',3,2\n'
    print(K22)
    # 上書きされた値を含めて保存
    with open(Kijfile,"w") as fp:
        for line in Kij:
            fp.write(line)
    print('make logs')
    
    # simulation (相互誤差のparamb固定)
    makeLog(b1=np.array(0.015499),b2=np.array(0.01205),b3=np.array(0.01205))
    
    # reading gt & logs
    logfile = readlogsFiles()[0]
    
    print(logfile)
    # U:[None,10], B:[3,]
    U,B = myData.loadABLV(logfile)
    deltaU = myData.convV2YearlyData(U)
    
    # each mse var.
    print('start gauss')
    maxSim = myData.MinErrorNankai(deltaU,mode=3)
    maxSim = 1/maxSim
    
    # Delate logfile
    os.remove(logfile)
    
    return maxSim
# -----------------------------------------------------------------------------
    
# -----------------------------------------------------------------------------
for epoch in np.arange(nEpoch):
    
    # paramb ##################################################################
    if mode == 'paramb':
    
        # Set prior
        if epoch == 0:
            pd = defaultpdB
        else:
            pdBs = np.loadtxt(os.path.join(savedirPath,f"BO_paramb_{epoch-1}_{itrNum}_{trID}.txt"))
        
            minpdBs = pdBs[-1]/mt
            maxpdBs = pdBs[-2]/mt
            
            print('range min:',minpdBs)
            print('range max:',maxpdBs)
            
            pd = [[minpdBs[ntI],maxpdBs[ntI]],[minpdBs[tntI],maxpdBs[tntI]],[minpdBs[ttI],maxpdBs[ttI]]]
        
        # prior distribution parameter b    
        pbounds = setPriorDistribution(pd)
    
        # Start Bayes ---------------------------------------------------------
        # verbose: 学習過程表示 0:無し, 1:すべて, 2:最大値更新時
        opt = BayesianOptimization(f=bfunc,pbounds=pbounds,verbose=1)
        
    # Kij #####################################################################
    elif mode == 'Kij':
        if epoch == 0:
            pd = defaultK22
        
        if epoch > 0:
            K22 = random.sample(K22samples.tolist(),2)
            pd = [np.min(K22), np.max(K22)]
        
        # prior distribution parameter b    
        pbounds = setPriorDistribution(pd)
    
        # Start Bayes ---------------------------------------------------------
        # verbose: 学習過程表示 0:無し, 1:すべて, 2:最大値更新時
        opt = BayesianOptimization(f=Kijfunc,pbounds=pbounds,verbose=1)
        
    # init_points:最初に取得するf(x)の数、ランダムに選択される
    # n_iter:試行回数(default:25パターンのパラメータで学習)
    opt.maximize(init_points=5,n_iter=itrNum,acq='ucb',kappa=100)
    # -------------------------------------------------------------------------
    #pdb.set_trace()
    # Result ------------------------------------------------------------------
    res = opt.res # all
    best_res = opt.max # max optimize
    # sort based on 'target'(maxSim)
    sort_res = sorted(res, key=lambda x: x['target'])
    # -------------------------------------------------------------------------
    #pdb.set_trace()
    # Save params -------------------------------------------------------------
    
    if mode == 'paramb':
        
        flag = False
        for line in sort_res:
            
            # directory -> numpy [1,] [3,]
            target = np.array([line['target']])
            param = np.concatenate((np.array([line['params']['b1']]),np.array([line['params']['b2']]),np.array([line['params']['b3']])),0)
            
            if not flag:
                targets = target
                params = param * mt
                flag = True
            else:
                targets = np.vstack([targets,target])
                params = np.vstack([params,param * mt])
        # optimized rate
        np.savetxt(os.path.join(savedirPath,f"BO_target_{epoch}_{itrNum}_{trID}.txt"),targets)
        # parameter b
        np.savetxt(os.path.join(savedirPath,f"BO_{mode}_{epoch}_{itrNum}_{trID}.txt"),params,fmt=f"%d")
    # -------------------------------------------------------------------------
    elif mode == 'Kij':
        
        flag = False
        for line in sort_res:
            
            # directory -> numpy [1,] [3,]
            target = np.array([line['target']])
            param = np.array([line['params']['K22']])
        
            if not flag:
                targets = target
                params = param
                flag = True
            else:
                targets = np.vstack([targets, target])
                params = np.vstack([params, param])
        
        isDirectory(savedirPath)
        # optimized rate
        np.savetxt(os.path.join(savedirPath,f"BO_target_{epoch}_{itrNum}_{trID}.txt"),targets)
        # parameter Kij
        np.savetxt(os.path.join(savedirPath,f"BO_{mode}_{epoch}_{itrNum}_{trID}.txt"),params,fmt=f"%7f")
        
# -----------------------------------------------------------------------------

"""
# Make best csv ---------------------------------------------------------------
# Get best paramter b
targetfullPath = os.path.join(savedirPath,'BO_target_*')
targetfiles = glob.glob(targetfullPath)

bfullPath = os.path.join(savedirPath,'BO_paramb_*')
bfiles = glob.glob(bfullPath)

flag = False
for targetfile,bfile in zip(targetfiles,bfiles):
    target = np.loadtxt(targetfile)
    paramb = np.loadtxt(bfile)
    
    if not flag:
        targets = target
        bs = paramb
        flag = True
    else:
        #pdb.set_trace()
        targets = np.hstack([targets,target])
        bs = np.vstack([bs,paramb])

# del multiple parameter b
parambs = [bs[0]]
index = [0]
for ind,line in enumerate(bs):
    if not all(line == parambs[-1]):
        parambs.append(line)
        index.append(ind)

# del multiple targets
maxsims = targets[index]

# list -> numpy
parambs = np.array(parambs)

# min mse index
best100ind = np.argsort(maxsims)[::-1][:100]
best100target = targets[best100ind]
best100b = bs[best100ind.tolist()]

np.savetxt(os.path.join(savedirPath,f'best100_target.txt'),best100target)
# for bat
np.savetxt(os.path.join(savedirPath,f'best100_b_{trID}.csv'),best100b,delimiter=',',fmt='%d')
# -----------------------------------------------------------------------------
"""
# after featureV.bat
"""
# Plot rireki -----------------------------------------------------------------
logsfullPath = os.path.join(logsPath,f'sortbayes_{trID}',filePath)
logsfile = glob.glob(logsfullPath)

with open(os.path.join(featuresPath,"nankairireki.pkl"), "rb") as fp:
    nkfiles = pickle.load(fp)
gt = nkfiles[190,:,:]
  
flag = False
index = []
for iS in np.arange(len(logsfile)):
    
    file = os.path.basename(logsfile[iS])
    print(file)
    # Num. of file for sort index
    fID = int(file.split("_")[-1].split(".")[0])
   
    U, th, V, B = myDataPF.loadABLV(logsPath,f'sortbayes_{trID}',file)
    B = np.concatenate([B[2,np.newaxis],B[4,np.newaxis],B[5,np.newaxis]],0)
    deltaU, _, _, _ = myDataPF.convV2YearlyData(U,th,V,nYear=10000,cnt=0,isLast=True)
    deltaU = np.concatenate((deltaU[:,2,np.newaxis],deltaU[:,4,np.newaxis],deltaU[:,5,np.newaxis]),1)
    maxSim, pred = myData.MinErrorNankai(deltaU,mode=3,isPlot=True)
    
    pJ_all = [np.where(pred[:,ntI]>1)[0],np.where(pred[:,tntI]>1)[0],np.where(pred[:,ttI]>1)[0]]
    predV = [pJ_all[ntI]-int(U[0,1]),pJ_all[tntI]-int(U[0,1]),pJ_all[ttI]-int(U[0,1])]
    gtV = [np.where(gt[:,ntI]>0)[0],np.where(gt[:,tntI]>0)[0],np.where(gt[:,ttI]>0)[0]]
    #pdb.set_trace()
    # plot & mae eq. of predict & gt
    myPlot.Rireki(gtV,predV,path=f"bayes_{trID}",label=f"{iS}_{np.round(B[ntI],6)}_{np.round(B[tntI],6)}_{np.round(B[ttI],6)}",title=f'{int(maxSim)}\n{predV[0].tolist()}\n{predV[1].tolist()}\n{predV[2].tolist()}',iseach=True)

    if not flag:
        maxSims = maxSim
        flag = True
    else:
        maxSims = np.hstack([maxSims,maxSim])
    
sort_maxSims = np.sort(maxSims)
index = np.argsort(maxSims)

np.savetxt(os.path.join(f"bayes_{trID}",'maxsim.txt'),sort_maxSims,fmt='%d')
np.savetxt(os.path.join(f"bayes_{trID}",'index.txt'),index,fmt='%d')
# -----------------------------------------------------------------------------
"""
"""
# Plot scatter parameter b (heatmap) ----------------------------------------------------

bs = np.loadtxt('ucb_kappa10.csv', delimiter=',')
var = np.loadtxt('ucb_kappa10.txt')

x = bs[:,ntI]
y = bs[:,tntI]
z = bs[:,ttI]


nkmin, nkmax = 0.0152, 0.0154
tnkmin, tnkmax = 0.012, 0.0128
tkmin, tkmax = 0.012, 0.0125

rangeb = [[nkmin*1000000,tnkmin*1000000,tkmin*1000000],[nkmax*1000000,tnkmax*1000000,tkmax*1000000]]

minvar = np.min(1/var)
maxvar = np.max(1/var)
sigmavar = np.var(1/var)
muvar = np.mean(1/var)

myPlot.scatter3D_heatmap(x,y,z,var,rangeP=rangeb,path=os.path.join('images','allsearch','sort_ucb_kappa10'),title=f'top100',label=f"heatmap_sort_ucb_kappa10")
"""
# -----------------------------------------------------------------------------
"""
# Plot scatter parameter b ----------------------------------------------------

bs = np.loadtxt('ucb_kappa10.csv', delimiter=',')

x = bs[:,ntI]
y = bs[:,tntI]
z = bs[:,ttI]

nkmin, nkmax = np.min(x), np.max(x)
tnkmin, tnkmax = np.min(y), np.max(y)
tkmin, tkmax = np.min(z), np.max(z)

nkmin, nkmax = 0.0146, 0.0154
tnkmin, tnkmax = 0.012, 0.0146
tkmin, tkmax = 0.012, 0.0134

rangeb = [[nkmin*1000000,tnkmin*1000000,tkmin*1000000],[nkmax*1000000,tnkmax*1000000,tkmax*1000000]]

myPlot.scatter3D(x,y,z,rangeP=rangeb,path=os.path.join('images','allsearch','sort_ucb_kappa10'),title='top100',label='sort_ucb_kappa10')    
# -----------------------------------------------------------------------------
"""
