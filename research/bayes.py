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
parser.add_argument('--mode', required=True, choices=['paramb','Kij','ABL'])
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
if mode == 'ABL':
    batFile = 'PyToCBayesABL.bat'
else:
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
nEpoch = 10
# range b
mt = 1000000

## for researching Kij (default) ##
#K32min, K32max = -139157.4154072-60000, -139157.4154072+30000
#K23min, K23max = -140041.1031088-50000, -140041.1031088+40000

#defaultK2332 = [[K23min, K23max],[K32min, K32max]]
#pdb.set_trace()
##

## for researching for param a (default) ##
amin,amax = 0.01,0.02

defaultA = [[amin,amax],[amin,amax],[amin,amax]]

## for researching for param b (default) ##
# range of under & over in parameter
#bnkmin,bnkmax = 0.014449,0.015499
#btnkmin,btnkmax = 0.012,0.014949
#btkmin,btkmax = 0.012,0.0135
# for all search
bnkmin,bnkmax = 0.011,0.0165
btnkmin,btnkmax = 0.011,0.0165
btkmin,btkmax = 0.011,0.0170

defaultB = [[bnkmin,bnkmax],[btnkmin,btnkmax],[btkmin,btkmax]]

## for researching for param L (default) ##
Lmin,Lmax = 0.01,0.05

defaultL = [[Lmin,Lmax],[Lmin,Lmax],[Lmin,Lmax]]

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
    '''
    pd: searching parameter
    '''
    if mode == 'paramb':
        nkmin,nkmax = pd[ntI][0],pd[ntI][1]
        tnkmin,tnkmax = pd[tntI][0],pd[tntI][1]
        tkmin,tkmax = pd[ttI][0],pd[ttI][1]
        
        # 連続値の場合は、事前分布指定可（default:連続一様分布、対数一様分布も指定可）
        pbounds = {"b1":(nkmin,nkmax),"b2":(tnkmin,tnkmax),"b3":(tkmin,tkmax)}
    
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
def makeLog(b1=np.array(0.011),b2=np.array(0.011),b3=np.array(0.011)):
    #pdb.set_trace()
    # save param b ------------------------------------------------------------
    params = np.concatenate((b1[np.newaxis],b2[np.newaxis],b3[np.newaxis]),0)[:,np.newaxis]
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

# function (paramb) -----------------------------------------------------------
def bfunc(b1,b2,b3):
    
    # simulation
    makeLog(b1=b1,b2=b2,b3=b3)
    
    func = objective()
    
    return func
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
        opt = BayesianOptimization(f=ABLfunc,pbounds=pbounds,verbose=1)
   
    # ABL #####################################################################
    elif mode == 'ABL':
        
        pd = [defaultA, defaultB, defaultL]
        
        # prior distribution parameter b    
        pbounds = setPriorDistribution(pd)
    
        # Start Bayes ---------------------------------------------------------
        # verbose: 学習過程表示 0:無し, 1:すべて, 2:最大値更新時
        opt = BayesianOptimization(f=ABLfunc,pbounds=pbounds,verbose=1)
       
    
    # init_points:最初に取得するf(x)の数、ランダムに選択される
    # n_iter:試行回数(default:25パターンのパラメータで学習)
    opt.maximize(init_points=5,n_iter=itrNum,acq='ucb',kappa=10)
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
      
# -----------------------------------------------------------------------------
        
"""
# Make best csv ---------------------------------------------------------------
# Get best paramter b
targetfullPath = os.path.join(savedirPath,'BO_target_*')
targetfiles = glob.glob(targetfullPath)

paramfullPath = os.path.join(savedirPath,f'BO_{mode}_*')
paramfiles = glob.glob(paramfullPath)

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
        #pdb.set_trace()
        targets = np.hstack([targets,target])
        # for paramb
        #params = np.vstack([params,param])
        # for Kij
        params = np.hstack([params,param])

if mode == 'paramb':

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

elif mode == 'Kij':
    #pdb.set_trace()
    best100ind = np.argsort(targets)[::-1][:100]
    best100target = targets[best100ind]
    best100param = params[best100ind.tolist()]

# save target & param
np.savetxt(os.path.join(savedirPath,f'best100_target.txt'),best100target)
# for bat
np.savetxt(os.path.join(savedirPath,f'best100_b_{trID}.csv'),best100param,delimiter=',',fmt='%d')
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
