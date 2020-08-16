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
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

# ※ not sklearn
from myBayesianOptimization.bayes_opt import BayesianOptimization
#import bayes_opt as BayesianOptimization

import DC as myData
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
parser.add_argument('--mode', required=True, choices=['paramb','Kij','ABL','shift_time','shift_mean'])
# kappa for ucb mean + kappa * sigma
parser.add_argument('--kappa', type=float, default=2.5)
# kappa decay % for ucb (default:None)
# smaller decay % -> more decrease
parser.add_argument('--kDecay', type=float, default=1)
# kappa decay delay for ucb (default:None)
parser.add_argument('--kitr', type=int, default=0)

# 引数展開
args = parser.parse_args()

itrNum = args.itrNum
trID = args.trID
mode = args.mode
kappa = args.kappa
kDecay = args.kDecay
kitr = args.kitr
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
cnt = 0
## for researching Kij (default) ##
#K32min, K32max = -139157.4154072-60000, -139157.4154072+30000
#K23min, K23max = -140041.1031088-50000, -140041.1031088+40000

#defaultK2332 = [[K23min, K23max],[K32min, K32max]]
#pdb.set_trace()
##

## for researching for param a (default) ##
amin,amax = 0.01,0.020

defaultA = [[amin,amax],[amin,amax],[amin,amax]]

## for researching for param b (default) ##
# range of under & over in parameter
bnkmin,bnkmax = 0.014449,0.015499
btnkmin,btnkmax = 0.012,0.014949
btkmin,btkmax = 0.012,0.0135
# for all search
#bnkmin,bnkmax = 0.011,0.0165
#btnkmin,btnkmax = 0.011,0.0165
#btkmin,btkmax = 0.011,0.0170

defaultB = [[bnkmin,bnkmax],[btnkmin,btnkmax],[btkmin,btkmax]]

## for researching for param L (default) ##
Lmin,Lmax = 0.0005,0.01

defaultL = [[Lmin,Lmax],[Lmin,Lmax],[Lmin,Lmax]]


with open(os.path.join(featuresPath,"nankairireki.pkl"), "rb") as fp:
    nkfiles = pickle.load(fp)
gt = nkfiles[190,:,:]

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
def setPriorDistribution(pd, pname='paramb'):
    '''
    pd: searching parameter
    pname: parameter
    '''
    if pname == 'parama':
        
        nkmin,nkmax = pd[ntI][0],pd[ntI][1]
        tnkmin,tnkmax = pd[tntI][0],pd[tntI][1]
        tkmin,tkmax = pd[ttI][0],pd[ttI][1]
        
        # 連続値の場合は、事前分布指定可（default:連続一様分布、対数一様分布も指定可）
        pbounds = {"a1":(nkmin,nkmax),"a2":(tnkmin,tnkmax),"a3":(tkmin,tkmax)}
    
    elif pname == 'paramb':
        
        nkmin,nkmax = pd[ntI][0],pd[ntI][1]
        tnkmin,tnkmax = pd[tntI][0],pd[tntI][1]
        tkmin,tkmax = pd[ttI][0],pd[ttI][1]
        
        # 連続値の場合は、事前分布指定可（default:連続一様分布、対数一様分布も指定可）
        pbounds = {"b1":(nkmin,nkmax),"b2":(tnkmin,tnkmax),"b3":(tkmin,tkmax)}
    
    elif pname == 'paramL':
        
        nkmin,nkmax = pd[ntI][0],pd[ntI][1]
        tnkmin,tnkmax = pd[tntI][0],pd[tntI][1]
        tkmin,tkmax = pd[ttI][0],pd[ttI][1]
        
        # 連続値の場合は、事前分布指定可（default:連続一様分布、対数一様分布も指定可）
        pbounds = {"L1":(nkmin,nkmax),"L2":(tnkmin,tnkmax),"L3":(tkmin,tkmax)}
    
    elif mode == 'Kij':
        
        #pbounds = {"K22":(pd[0],pd[1])}
        pbounds = {"K23":(pd[0][0],pd[0][1]), "K32":(pd[1][0],pd[1][1])}
    
    elif mode == 'ABL':
        
        ankmin,ankmax = pd[0][ntI][0],pd[0][ntI][1]
        atnkmin,atnkmax = pd[0][tntI][0],pd[0][tntI][1]
        atkmin,atkmax = pd[0][ttI][0],pd[0][ttI][1]
        
        bnkmin,bnkmax = pd[1][ntI][0],pd[1][ntI][1]
        btnkmin,btnkmax = pd[1][tntI][0],pd[1][tntI][1]
        btkmin,btkmax = pd[1][ttI][0],pd[1][ttI][1]
        
        Lnkmin,Lnkmax = pd[2][ntI][0],pd[2][ntI][1]
        Ltnkmin,Ltnkmax = pd[2][tntI][0],pd[2][tntI][1]
        Ltkmin,Ltkmax = pd[2][ttI][0],pd[2][ttI][1]
        
        pbounds = {'a1':(ankmin,ankmax),'a2':(atnkmin,atnkmax),'a3':(atkmin,atkmax),
                   'b1':(bnkmin,bnkmax),'b2':(btnkmin,btnkmax),'b3':(btkmin,btkmax),
                   'L1':(Lnkmin,Lnkmax),'L2':(Ltnkmin,Ltnkmax),'L3':(Ltkmin,Ltkmax)}
    
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

# making logs (paramb) --------------------------------------------------------
def makeLog(p1=np.array(0.011),p2=np.array(0.011),p3=np.array(0.011)):
    '''
    one kind of parameter(a,b,L)
    '''
    
    #pdb.set_trace()
    # save param b ------------------------------------------------------------
    params = np.concatenate((p1[np.newaxis],p2[np.newaxis],p3[np.newaxis]),0)[:,np.newaxis]
    np.savetxt(paramCSV,params.T*mt,delimiter=",",fmt="%.0f")
    # -------------------------------------------------------------------------
    
    callSimulation()
# -----------------------------------------------------------------------------

# making logs (parama,paramb,paramL) ------------------------------------------
def makeLogABL(a1=np.array(0.011),a2=np.array(0.011),a3=np.array(0.011),
               b1=np.array(0.011),b2=np.array(0.011),b3=np.array(0.011),
               L1=np.array(0.011),L2=np.array(0.011),L3=np.array(0.011)):
    #pdb.set_trace()
    
    # save param ABL ----------------------------------------------------------
    params = np.concatenate((a1[np.newaxis],a2[np.newaxis],a3[np.newaxis],
                             b1[np.newaxis],b2[np.newaxis],b3[np.newaxis],
                             L1[np.newaxis],L2[np.newaxis],L3[np.newaxis]),0)[:,np.newaxis]
    np.savetxt(paramCSV,params.T*mt,delimiter=",",fmt="%.0f")
    # -------------------------------------------------------------------------
    
    callSimulation()
# -----------------------------------------------------------------------------

# simulation ------------------------------------------------------------------
def callSimulation():    
    '''
    Do bat (making logs)
    '''
    
    if mode == 'ABL' or mode == 'shift_time':
        batFile = 'PyToCBayesABL.bat'
    elif mode == 'paramb':
        batFile = 'PyToCBayes.bat'
    
    #pdb.set_trace()
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

# -----------------------------------------------------------------------------
def objective():
    
    # reading gt & logs
    logfile = readlogsFiles()[0]
    
    tmpA = os.path.basename(logfile).split('_')[:3]
    tmpB = os.path.basename(logfile).split('_')[3:6]
    
    tmpA = np.array(tmpA).astype(int)
    tmpB = np.array(tmpB).astype(int)

    AB = tmpA - tmpB
    print(f'nk:{AB[0]} tnk:{AB[1]} tk:{AB[2]}')
    
    # OK a-b<0
    if all(AB<0):
    
        # U:[None,10], B:[3,]
        U,B = myData.loadABLV(logfile)
        deltaU = myData.convV2YearlyData(U)
        # one : one (※MSE) mode==4(normal) mode==5(reverse)
        maxSim = myData.MinErrorNankai(gt, deltaU, mode=4)
        
    else:
        # out value
        maxSim = 100000000
    
    maxSim = 1/maxSim
    
    # Delate logfile
    os.remove(logfile)
    
    return maxSim
# -----------------------------------------------------------------------------

# function (parama) -----------------------------------------------------------
def afunc(a1,a2,a3):
    
    b, L = paramStock(pname='parama')
    b1,b2,b3 = b[0],b[1],b[2]
    L1, L2, L3 = L[0],L[1],L[2]
    
    # simulation
    makeLogABL(a1=a1,a2=a2,a3=a3,
               b1=b1,b2=b2,b3=b3,
               L1=L1,L2=L2,L3=L3)

    func = objective()
    
    return func
# -----------------------------------------------------------------------------
    
# function (paramb) -----------------------------------------------------------
def bfunc(b1,b2,b3):
    
    a, L = paramStock(pname='paramb')
    
    a1, a2, a3 = a[0],a[1],a[2]
    L1, L2, L3 = L[0],L[1],L[2] 
    
    # simulation
    makeLogABL(a1=a1,a2=a2,a3=a3,
               b1=b1,b2=b2,b3=b3,
               L1=L1,L2=L2,L3=L3)
    
    func = objective()
    
    return func
# -----------------------------------------------------------------------------
        
# function (paramL) -----------------------------------------------------------
def Lfunc(L1,L2,L3):
    
    a, b = paramStock(pname='paramL')
    
    a1, a2, a3 = a[0],a[1],a[2]
    b1,b2,b3 = b[0],b[1],b[2]
    
    # simulation
    makeLogABL(a1=a1,a2=a2,a3=a3,
               b1=b1,b2=b2,b3=b3,
               L1=L1,L2=L2,L3=L3)
    
    func = objective()
    
    return func
# -----------------------------------------------------------------------------
    
# other parameter -------------------------------------------------------------
def paramStock(pname='paramb'):
    '''
    load other two parameter file (not update by bayes)
    if paramb optimize -> loading a & L
    !! first epoch time a=0.01, L=0.01 (manual)
    '''
    
    if pname == 'parama':
        
        otherparams1 = np.loadtxt(os.path.join('params','updateparamb.csv'), delimiter=',')    
        otherparams2 = np.loadtxt(os.path.join('params','updateparamL.csv'), delimiter=',')    
    
    elif pname == 'paramb':
        
        otherparams1 = np.loadtxt(os.path.join('params','updateparama.csv'), delimiter=',')    
        otherparams2 = np.loadtxt(os.path.join('params','updateparamL.csv'), delimiter=',')    
        
    elif pname == 'paramL':
        
        otherparams1 = np.loadtxt(os.path.join('params','updateparama.csv'), delimiter=',')    
        otherparams2 = np.loadtxt(os.path.join('params','updateparamb.csv'), delimiter=',')    
    
    return otherparams1, otherparams2
# -----------------------------------------------------------------------------
    
# function (parama,paramb,paramL) ---------------------------------------------   
def ABLfunc(a1,a2,a3,b1,b2,b3,L1,L2,L3):

    # simulation
    makeLogABL(a1=a1,a2=a2,a3=a3,
               b1=b1,b2=b2,b3=b3,
               L1=L1,L2=L2,L3=L3)
    
    func = objective()
    
    return func
# -----------------------------------------------------------------------------
    
# function (Kij) --------------------------------------------------------------
#def Kijfunc(K22):
def Kijfunc(K23,K32):
    
    # Updata Kij
    # reading Kij file(K8_AV)
    Kij = open(Kijfile).readlines()
    # 上書きされるKij (別のKijを変えたいときは+以降も変える必要あり)
    #K22 = Decimal(str(K22)).quantize(Decimal('0.0000001'))
    K23 = Decimal(str(K23)).quantize(Decimal('0.0000001'))
    K32 = Decimal(str(K32)).quantize(Decimal('0.0000001'))
    
    # K23
    Kij[10] = str(K23) + ',2,3\n'
    # K32
    Kij[17] = str(K32) + ',3,2\n'
    
    #pdb.set_trace()
    #print(K22)
    # 上書きされた値を含めて保存
    with open(Kijfile,"w") as fp:
        for line in Kij:
            fp.write(line)
    
    # simulation (相互誤差のparamb固定)
    makeLog(b1=np.array(0.015499),b2=np.array(0.01205),b3=np.array(0.01205))
    
    func = objective()
    
    return func
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
for epoch in np.arange(nEpoch):
    
    # paramb ##################################################################
    if mode == 'paramb':
    
        # Set prior
        if epoch == 0:
            pd = defaultpd
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
        #opt = myBayesianOptimization(f=bfunc,pbounds=pbounds,verbose=1)
        opt = BayesianOptimization(f=bfunc,pbounds=pbounds,verbose=1)
        
    # Kij #####################################################################
    elif mode == 'Kij':
        
        if epoch == 0:
            #pd = defaultK22
            pd = defaultK2332
        
        elif epoch > 0:
            pds = np.loadtxt(os.path.join(savedirPath,f"BO_Kij_{epoch-1}_{itrNum}_{trID}.txt"))
            
            minpds = np.min(pds,0)
            maxpds = np.max(pds,0)
           
            #K22 = random.sample(Kijsamples.tolist(),2)
            #pd = [np.min(K22), np.max(K22)]
        
            #K32 = random.sample(K32samples.tolist(),2)
            #K23 = random.sample(K23samples.tolist(),2)
            pd = [[minpds[0], maxpds[0]],[minpds[1], maxpds[1]]]
        
        # prior distribution parameter b    
        pbounds = setPriorDistribution(pd)
    
        # Start Bayes ---------------------------------------------------------
        # verbose: 学習過程表示 0:無し, 1:すべて, 2:最大値更新時
        #opt = myBayesianOptimization(f=ABLfunc,pbounds=pbounds,verbose=1)
        opt = BayesianOptimization(f=ABLfunc,pbounds=pbounds,verbose=1)
   
    # ABL #####################################################################
    elif mode == 'ABL':
        
        pd = [defaultA, defaultB, defaultL]
        
        # prior distribution parameter b    
        pbounds = setPriorDistribution(pd)
    
        # Start Bayes ---------------------------------------------------------
        # verbose: 学習過程表示 0:無し, 1:すべて, 2:最大値更新時
        #opt = myBayesianOptimization(f=ABLfunc,pbounds=pbounds,verbose=1)
        opt = BayesianOptimization(f=ABLfunc,pbounds=pbounds,verbose=1)
   
    # b -> a -> L -> K? -> a-b? ###############################################
    elif mode == 'shift_time':
        
        # parama
        if epoch == 1 or epoch == 4 or epoch == 7:
       
            pname = 'parama'
            
            paramb = np.loadtxt(os.path.join('params','updateparamb.csv'), delimiter=',')
            
            # reading parameter a
            tmpa = np.loadtxt(os.path.join('params','pdparama.csv'), delimiter=',')    
            
            #pdb.set_trace()
            
            # ok: max parama < paramb
            if paramb[0] < tmpa[1][0]:
                # min
                tmpa[0][0] = paramb[0] - 0.002
                # max
                tmpa[1][0] = paramb[0] - 0.001
            if paramb[1] < tmpa[1][1]:
                tmpa[0][1] = paramb[1] - 0.002
                tmpa[1][1] = paramb[1] - 0.001
            if paramb[2] < tmpa[1][2]:
                tmpa[0][2] = paramb[2] - 0.002
                tmpa[1][2] = paramb[2] - 0.001
            
            pd = [tmpa[:,0],tmpa[:,1],tmpa[:,2]]
           
            # prior distribution parameter a 
            pbounds = setPriorDistribution(pd, pname=pname)
        
            # Start Bayes -----------------------------------------------------
            #opt = myBayesianOptimization(f=afunc,pbounds=pbounds,verbose=1)
            opt = BayesianOptimization(f=afunc,pbounds=pbounds,verbose=1)
        
        # paramb
        if epoch == 0 or epoch == 3 or epoch == 6:
            
            pname = 'paramb'
            
            # reading parameter
            tmp = np.loadtxt(os.path.join('params','pdparamb.csv'), delimiter=',')    
            pd = [tmp[:,0],tmp[:,1],tmp[:,2]]
            
            # prior distribution parameter b
            pbounds = setPriorDistribution(pd, pname=pname)
            
            #pdb.set_trace()
        
            # Start Bayes -----------------------------------------------------
            #opt = myBayesianOptimization(f=bfunc,pbounds=pbounds,verbose=1)
            opt = BayesianOptimization(f=bfunc,pbounds=pbounds,verbose=2)
        
        # paramL
        if epoch == 2 or epoch == 5 or epoch == 8:
        
            pname = 'paramL'
            
            # reading parameter
            tmp = np.loadtxt(os.path.join('params','pdparamL.csv'), delimiter=',')    
            pd = [tmp[:,0],tmp[:,1],tmp[:,2]]
            
            # prior distribution parameter L
            pbounds = setPriorDistribution(pd, pname=pname)
        
            # Start Bayes -----------------------------------------------------
            #opt = myBayesianOptimization(f=Lfunc,pbounds=pbounds,verbose=1)
            opt = BayesianOptimization(f=Lfunc,pbounds=pbounds,verbose=1)
        
    ###########################################################################
    #pdb.set_trace()
    
    # init_points:最初に取得するf(x)の数、ランダムに選択される
    # n_iter:試行回数(default:5パターンのパラメータで学習)
    opt.maximize(init_points=5, n_iter=itrNum, acq='ucb', kappa=kappa, kappa_decay=kDecay, kappa_decay_delay=kitr)
    #opt.maximize(init_points, n_iter, acq, kappa, kappa_decay, kappa_decay_delay)
    
    # -------------------------------------------------------------------------
         
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
  
    elif mode == 'Kij':
        
        flag = False
        for line in sort_res:
            
            # directory -> numpy [1,] [3,]
            target = np.array([line['target']])
            #param = np.array([line['params']['K22']])
            
            param = np.concatenate((np.array([line['params']['K23']]), np.array([line['params']['K32']])), 0)
        
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
        
    elif mode == 'ABL':
        flag = False
        for line in sort_res:
            
            # directory -> numpy [1,] [3,]
            target = np.array([line['target']])
            
            parama = np.concatenate((np.array([line['params']['a1']]),np.array([line['params']['a2']]),np.array([line['params']['a3']])),0)
            paramb = np.concatenate((np.array([line['params']['b1']]),np.array([line['params']['b2']]),np.array([line['params']['b3']])),0)
            paramL = np.concatenate((np.array([line['params']['L1']]),np.array([line['params']['L2']]),np.array([line['params']['L3']])),0)
            
            if not flag:
                targets = target
                paramas = parama * mt
                parambs = paramb * mt
                paramLs = paramL * mt
                
                flag = True
            else:
                targets = np.vstack([targets,target])
                paramas = np.vstack([paramas,parama * mt])
                parambs = np.vstack([parambs,paramb * mt])
                paramLs = np.vstack([paramLs,paramL * mt])
                
        
        # optimized rate
        np.savetxt(os.path.join(savedirPath,f"BO_target_{epoch}_{itrNum}_{trID}.txt"),targets)
        # parameter 
        np.savetxt(os.path.join(savedirPath,f"BO_{mode}_{epoch}_{itrNum}_{trID}_a.txt"),paramas,fmt=f"%d")
        np.savetxt(os.path.join(savedirPath,f"BO_{mode}_{epoch}_{itrNum}_{trID}_b.txt"),parambs,fmt=f"%d")
        np.savetxt(os.path.join(savedirPath,f"BO_{mode}_{epoch}_{itrNum}_{trID}_L.txt"),paramLs,fmt=f"%d")
    
    elif mode == 'shift_time':
        
        flag = False
        for line in sort_res:
            
            # directory -> numpy [1,] [3,]
            target = np.array([line['target']])
            
            if pname == 'parama':
            
                param = np.concatenate((np.array([line['params']['a1']]),np.array([line['params']['a2']]),np.array([line['params']['a3']])),0)
                
                if not flag:
                    targets = target
                    params = param * mt
                    flag = True
                else:
                    targets = np.vstack([targets,target])
                    params = np.vstack([params,param * mt])
        
            elif pname == 'paramb':
            
                param = np.concatenate((np.array([line['params']['b1']]),np.array([line['params']['b2']]),np.array([line['params']['b3']])),0)
          
                if not flag:
                    targets = target
                    params = param * mt
                    flag = True
                else:
                    targets = np.vstack([targets,target])
                    params = np.vstack([params,param * mt])
        
                
            elif pname == 'paramL':
            
                param = np.concatenate((np.array([line['params']['L1']]),np.array([line['params']['L2']]),np.array([line['params']['L3']])),0)
            
                if not flag:
                    targets = target
                    params = param * mt
                    flag = True
                else:
                    targets = np.vstack([targets,target])
                    params = np.vstack([params,param * mt])
        
        #pdb.set_trace()
        
        # best 1&2 param (sort)
        best1param = params[-1]
        best2param = params[-2]
        
        # fot next pd, [b1min,b2min,b3min| b1max,b2max,b3max]
        best12param = np.sort(np.vstack([best1param,best2param]),0)
        
        # write update best target of param ex) a1, a2, a3
        if pname == 'parama':
            # for other param
            np.savetxt(os.path.join('params','updateparama.csv'), best1param[np.newaxis]/mt, fmt='%5f', delimiter=',')
            # for pd
            np.savetxt(os.path.join('params','pdparama.csv'), best12param/mt, fmt='%5f', delimiter=',')
            
            # other param, update
            b,L = paramStock(pname=pname)
            # other two param
            np.savetxt(os.path.join(savedirPath,f"BO_{mode}_{epoch}_{itrNum}_{pname}_{trID}_b.txt"), b*mt, fmt='%d', delimiter=',')
            np.savetxt(os.path.join(savedirPath,f"BO_{mode}_{epoch}_{itrNum}_{pname}_{trID}_L.txt"), L*mt, fmt='%d', delimiter=',')
                
        elif pname == 'paramb':
            # for other param
            np.savetxt(os.path.join('params','updateparamb.csv'), best1param[np.newaxis]/mt, fmt='%5f', delimiter=',')
            # for pd
            np.savetxt(os.path.join('params','pdparamb.csv'), best12param/mt, fmt='%5f', delimiter=',')
           
            # other param, update
            a,L = paramStock(pname=pname)
            # other two param
            np.savetxt(os.path.join(savedirPath,f"BO_{mode}_{epoch}_{itrNum}_{pname}_{trID}_a.txt"), a*mt, fmt='%d', delimiter=',')
            np.savetxt(os.path.join(savedirPath,f"BO_{mode}_{epoch}_{itrNum}_{pname}_{trID}_L.txt"), L*mt, fmt='%d', delimiter=',')
            
        elif pname == 'paramL':
            # for other param
            np.savetxt(os.path.join('params','updateparamL.csv'), best1param[np.newaxis]/mt, fmt='%5f', delimiter=',')
            # for pd
            np.savetxt(os.path.join('params','pdparamL.csv'), best12param/mt, fmt='%5f', delimiter=',')
         
            # other param, update
            a,b = paramStock(pname=pname)
            # other two param
            np.savetxt(os.path.join(savedirPath,f"BO_{mode}_{epoch}_{itrNum}_{pname}_{trID}_a.txt"), a*mt, fmt='%d', delimiter=',')
            np.savetxt(os.path.join(savedirPath,f"BO_{mode}_{epoch}_{itrNum}_{pname}_{trID}_b.txt"), b*mt, fmt='%d', delimiter=',')
            
        #pdb.set_trace()
        # optimized rate
        np.savetxt(os.path.join(savedirPath,f"BO_target_{epoch}_{itrNum}_{pname}_{trID}.txt"),targets)
        # update param
        np.savetxt(os.path.join(savedirPath,f"BO_{mode}_{epoch}_{itrNum}_{pname}_{trID}.txt"),params,fmt="%d")
    
# -----------------------------------------------------------------------------

"""
# Make best csv ---------------------------------------------------------------
# Get best paramter b
targetfullPath = os.path.join(savedirPath,'BO_target_*')
targetfiles = glob.glob(targetfullPath)

paramfullPath = os.path.join(savedirPath,f'BO_{mode}_*')
paramfiles = glob.glob(paramfullPath)

paramafullPath = os.path.join(savedirPath,f'*_a.csv')
paramafiles = glob.glob(paramafullPath)    
paramBfullPath = os.path.join(savedirPath,f'*_b.csv')
paramBfiles = glob.glob(paramBfullPath)    
paramLfullPath = os.path.join(savedirPath,f'*_L.csv')
paramLfiles = glob.glob(paramLfullPath)

targeta = np.loadtxt(paramafiles[0],delimiter=',')
targetb = np.loadtxt(paramBfiles[0],delimiter=',')
targetl = np.loadtxt(paramLfiles[0],delimiter=',')

targets = np.concatenate([targeta,targetb,targetl],1)

np.savetxt(os.path.join(savedirPath,f'best100_ABL.csv'),targets,delimiter=',',fmt='%d')
 
pdb.set_trace()
   
if mode == 'ABL':
    paramAfullPath = os.path.join(savedirPath,f'*_a.txt')
    paramAfiles = glob.glob(paramAfullPath)    
    paramBfullPath = os.path.join(savedirPath,f'*_b.txt')
    paramBfiles = glob.glob(paramBfullPath)    
    paramLfullPath = os.path.join(savedirPath,f'*_L.txt')
    paramLfiles = glob.glob(paramLfullPath)
    
    flag = False
    for targetfile,paramAfile,paramBfile,paramLfile in zip(targetfiles,paramAfiles,paramBfiles,paramLfiles):
        target = np.loadtxt(targetfile)
        paramA = np.loadtxt(paramAfile)
        paramB = np.loadtxt(paramBfile)
        paramL = np.loadtxt(paramLfile)
       
        if not flag:
            targets = target
            paramsA = paramA
            paramsB = paramB
            paramsL = paramL
            flag = True
        else:
            targets = np.hstack([targets,target])
            paramsA = np.vstack([paramsA,paramA])
            paramsB = np.vstack([paramsB,paramB])
            paramsL = np.vstack([paramsL,paramL])
            
     # select 100 best targets(maxSim) & parameter
    best100ind = np.argsort(targets)[::-1][:100]
    best100target = targets[best100ind]
    best100paramA = paramsA[best100ind.tolist()]
    best100paramB = paramsB[best100ind.tolist()]
    best100paramL = paramsL[best100ind.tolist()]
    
    # save target & param
    np.savetxt(os.path.join(savedirPath,f'best100_target.txt'),best100target)
    # for bat
    np.savetxt(os.path.join(savedirPath,f'best100_{mode}_{trID}_a.csv'),best100paramA,delimiter=',',fmt='%d')
    np.savetxt(os.path.join(savedirPath,f'best100_{mode}_{trID}_b.csv'),best100paramB,delimiter=',',fmt='%d')
    np.savetxt(os.path.join(savedirPath,f'best100_{mode}_{trID}_L.csv'),best100paramL,delimiter=',',fmt='%d')

if mode == 'paramb':

    flag = False
    # ファイル文
    for targetfile,paramfile in zip(targetfiles,paramfiles):
        target = np.loadtxt(targetfile)
        param = np.loadtxt(paramfile)
       
        if not flag:
            targets = target
            params = param
            flag = True
        else:
            targets = np.hstack([targets,target])
            params = np.vstack([params,param])
            
    # del multiple parameter b
    parambs = [params[0]]
    index = [0]
    for ind,line in enumerate(params):
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
    best100param = params[best100ind.tolist()]

    # save target & param
    np.savetxt(os.path.join(savedirPath,f'best100_target.txt'),best100target)
    # for bat
    np.savetxt(os.path.join(savedirPath,f'best100_b_{trID}.csv'),best100param,delimiter=',',fmt='%d')

elif mode == 'Kij':
    flag = False
    # ファイル文
    for targetfile,paramfile in zip(targetfiles,paramfiles):
        target = np.loadtxt(targetfile)
        param = np.loadtxt(paramfile)
       
        if not flag:
            targets = target
            params = param
            flag = True
        else:
            targets = np.hstack([targets,target])
            params = np.vstack([params,param])
    
    # select 100 best targets(maxSim) & parameter
    best100ind = np.argsort(targets)[::-1][:100]
    best100target = targets[best100ind]
    best100param = params[best100ind.tolist()]
    
    # save target & param
    np.savetxt(os.path.join(savedirPath,f'best100_target.txt'),best100target)
    # for bat
    np.savetxt(os.path.join(savedirPath,f'best100_{mode}_{trID}.csv'),best100param,delimiter=',')
# -----------------------------------------------------------------------------
"""
# after featureV.bat
"""
# Plot rireki -----------------------------------------------------------------
logsfullPath = os.path.join(logsPath,f'bayes',filePath)
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
    #fID = int(file.split("_")[-1].split(".")[0])
    fID = file.split("_")[0]
    #pdb.set_trace()
    # B:[3,]
    U,_ = myData.loadABLV(logsfile[iS])
    deltaU = myData.convV2YearlyData(U)
    maxSim = myData.MinErrorNankai(deltaU,mode=3,label=fID,isPlot=True)
    
    if not flag:
        maxSims = maxSim
        flag = True
    else:
        maxSims = np.hstack([maxSims,maxSim])

sort_maxSims = np.sort(maxSims)
index = np.argsort(maxSims)

np.savetxt(os.path.join(savedirPath,'maxsim.txt'),sort_maxSims,fmt='%d')
np.savetxt(os.path.join(savedirPath,'index.txt'),index,fmt='%d')
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

# -----------------------------------------------------------------------------
"""
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
'''
sns.set()

# gt
xnk = [84,287,499,761,898,1005,1107,1254,1346]
# pred (all search)
ynk1 = [79,285,491,702,866,1075,1285,1285,1285]
# pred (bayes)
ynk2 = [84,288,491,697,871,1063,1272,1272,1272]

ynklabel = [0,30,60,180]
subnk1 = np.abs(np.array(xnk)-np.array(ynk1))
subnk2 = np.abs(np.array(xnk)-np.array(ynk2))

xtnk = [84,287,496,761,898,1005,1107,1254,1344]
ytnk1 = [79,285,491,702,925,1004,1099,1263,1285]
ytnk2 = [84,288,491,697,899,1026,1082,1250,1272]

ytnklabel = [0,30,60]
subtnk1 = np.abs(np.array(xtnk)-np.array(ytnk1))
subtnk2 = np.abs(np.array(xtnk)-np.array(ytnk2))

xtk = [84,287,496,761,898,1107,1254]
ytk1 = [79,285,491,702,925,1099,1285]
ytk2 = [84,288,491,697,913,1082,1272]

ytklabel = [0,30,60]
subtk1 = np.abs(np.array(xtk)-np.array(ytk1))
subtk2 = np.abs(np.array(xtk)-np.array(ytk2))

pdb.set_trace()
'''
'''
gts = [xnk,xtnk,xtk]
preds = [ynk,ytnk,ytk]
subs = [subnk,subtnk,subtk]
ylabels = [ynklabel,ytnklabel,ytklabel]

fig = plt.figure()
ax = fig.add_subplot(111)

fig, figInds = plt.subplots(nrows=3)
for figInd,(label,gt,pred,sub) in enumerate(zip(ylabels,gts,preds,subs)):
    figInds[figInd].plot(np.array(gt), sub, marker='o', alpha=0.5, color="black")
    # gt目盛り
    figInds[figInd].set_xticks(gt)
    figInds[figInd].set_yticks(label)
    
plt.show()
plt.close()

sns.set_style('dark')

fig = plt.figure()
ax = fig.add_subplot(111)

subs = np.concatenate([subnk,subtnk,subtk],0)
data = [subnk,subtnk,subtk,subs]
sns.boxplot(data=data, palette='Set3', width=0.5)

ax.set_xticklabels(['nankai','tonankai','tokai','all'])
plt.show()
'''
