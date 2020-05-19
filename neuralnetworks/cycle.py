# -*- coding: utf-8 -*-

import sys
import os
import glob
import time

import numpy as np

import random
import pickle
import pdb

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt


class Cycle:
    def __init__(self, logpath='logs', parampath='params', trialID=0):
        
        # slip velocity
        self.slip = 0
        # cell index
        self.ntI = 0
        self.tntI = 1
        self.ttI = 2
        self.allCell = 8
        
        self.saveparamPath = os.path.join(parampath, f'{trialID}')
        self.savelogPath = os.path.join(logpath, f'{trialID}')
    
    # ----
    def simulate(self, params, itr=0):
    
        # for loading batfile
        np.savetxt(os.path.join(self.saveparampath, 'logs.csv'), params*100000, fmt='%d')
        # copy
        np.savetxt(os.path.join(self.saveparampath, f'logs{itr}.csv'), params*100000, fmt='%d')
        pdb.set_trace()
        # Make logs ----
        lockPath = "Lock.txt"
        lock = str(1)
        with open(lockPath,"w") as fp:
            fp.write(lock)
        
        batFile = 'makelogs.bat'
        os.system(batFile)
    
        sleepTime = 3
        while True:
            time.sleep(sleepTime)
            if os.path.exists(lockPath)==False:
                break
    # ----
    
    # ----
    def loadBV(self, logFullPath):
         
        yInd = 0        
        b = np.zeros(self.allCell)
        
        data = open(logFullPath).readlines()
        
        # B ----
        for i in np.arange(1,self.allCell+1):
            # cell番号に合わせてdata読み取り
            tmp = np.array(data[i].strip().split(",")).astype(np.float32)
            b[i-1] = tmp[1]
        
        # V ----
        isRTOL = [True if data[i].count('value of RTOL')==1 else False for i in np.arange(len(data))]
        
        try:
            vInd = np.where(isRTOL)[0][0]+1
        except IndexError:
            pdb.set_trace()
    
        flag = False
        for uI in np.arange(vInd,len(data)):
            tmpV = np.array(data[uI].strip().split(",")[yInd:]).astype(np.float32)
            
            if not flag:
                v = tmpV
                flag = True
            else:
                v = np.vstack([v,tmpV])
        self.B = b
        self.V = v
        return self.B, self.V
    # ----
    
    # ----
    def convV2YearlyData(self):
    
        yrInd = 1
        stateYear = 2000        
        vInds = [2,3,4,5,6,7,8,9]
        nYear = 10000
        
        yV = np.zeros([nYear, self.allCell])
        
        sYear = np.floor(self.V[0,yrInd])
        for year in np.arange(sYear,nYear):
            # 観測値がある場合
            if np.sum(np.floor(self.V[:,yrInd])==year):
                yV[int(year)] = np.reshape(self.V[np.floor(self.V.T[yrInd,:])==year,vInds[0]:],[-1,])
        
            # 観測値がない場合
            else:
                # th(状態変数):地震時t-1の観測値代入,V(速度):0.0
                yV[int(year)] = yV[int(year)-1,:] 
        
        deltaU = np.vstack([yV[0,:], yV[1:] - yV[:-1]])
        
        # ※ Uの発生年数
        nkYear = np.where(deltaU[stateYear:,vInds[0]] > self.slip)[0]
        tnkYear = np.where(deltaU[stateYear:,vInds[2]] > self.slip)[0]
        tkYear = np.where(deltaU[stateYear:,vInds[3]] > self.slip)[0]
        self.predyear = [nkYear,tnkYear,tkYear]
    # ----
    
    # ---- 
    def calcYearMSE(self, gt):
        '''
        gt: exact eq.year, list[numpy]
        '''

        aYear = 1400
        nCell = 3
        th = 1
        
        # ground truth eq.
        gYear_nk = gt[self.ntI]
        gYear_tnk = gt[self.ntI]
        gYear_tk = gt[self.ntI]
        
        # predicted eq.
        pred = np.zeros((8000,nCell))
        pred[self.predyear[self.ntI], self.ntI] = 30
        pred[self.predyear[self.tntI], self.tntI] = 30
        pred[self.predyear[self.ttI], self.ttI] = 30
        
        flag = False
        # Slide each one year 
        for sYear in np.arange(8000-aYear): 
            eYear = sYear + aYear
            
            # Num. of gt eq
            gNum_nk = gYear_nk.shape[0]
            gNum_tnk = gYear_tnk.shape[0]
            gNum_tk = gYear_tk.shape[0]
            
            # eq.year > 1
            pYear_nk = np.where(pred[sYear:eYear,0] > th)[0][:gNum_nk]
            pYear_tnk = np.where(pred[sYear:eYear,1] > th)[0][:gNum_tnk]
            pYear_tk = np.where(pred[sYear:eYear,2] > th)[0][:gNum_tk]
            
            # gtよりpredの地震回数が少ない場合
            if pYear_nk.shape[0] < gNum_nk:
                pYear_nk = np.hstack([pYear_nk, np.tile(pYear_nk[-1], gNum_nk-pYear_nk.shape[0])])
            if pYear_tnk.shape[0] < gNum_tnk:
                pYear_tnk = np.hstack([pYear_tnk, np.tile(pYear_tnk[-1], gNum_tnk-pYear_tnk.shape[0])])
            if pYear_tk.shape[0] < gNum_tk:
                pYear_tk = np.hstack([pYear_tk, np.tile(pYear_tk[-1], gNum_tk-pYear_tk.shape[0])])
            
            # sum(abs(exact eq.year - pred eq.year))
            ndist_nk = np.abs(gYear_nk - pYear_nk)
            ndist_tnk = np.abs(gYear_tnk - pYear_tnk)
            ndist_tk = np.abs(gYear_tk - pYear_tk)
            
            yearError_nk = np.sum(ndist_nk)
            yearError_tnk = np.sum(ndist_tnk)
            yearError_tk = np.sum(ndist_tk)
            yearError = yearError_nk + yearError_tnk + yearError_tk
            
            if not flag:
                yearErrors = yearError
                flag = True
            else:
                yearErrors = np.hstack([yearErrors,yearError])
               
        # minimum mse eq.year
        sInd = np.argmin(yearErrors)
        eInd = sInd + aYear
        self.maxSim = yearErrors[sInd]
        
        # minimum eq.year 
        nkYear = self.predyear[self.ntI][(self.predyear[self.ntI] > sInd) & (self.predyear[self.ntI] < eInd)] - sInd
        tnkYear = self.predyear[self.tntI][(self.predyear[self.tntI] > sInd) & (self.predyear[self.tntI] < eInd)] - sInd
        tkYear = self.predyear[self.ttI][(self.predyear[self.ttI] > sInd) & (self.predyear[self.ttI] < eInd)] - sInd
        nkJ = np.pad(nkYear, (0, 500 - nkYear.shape[0]), 'constant', constant_values=0)
        tnkJ = np.pad(tnkYear, (0, 500 - tnkYear.shape[0]), 'constant', constant_values=0)
        tkJ = np.pad(tkYear, (0, 500 - tkYear.shape[0]), 'constant', constant_values=0)
        self.pJ = np.concatenate([nkJ[:,np.newaxis], tnkJ[:,np.newaxis], tkJ[:,np.newaxis]],1) #[150,3]

        return self.pJ, self.maxSim

    # ----
    
    # ----
    def loss(self, predParams, exactCycles, itr=0):
        
        # simulation b1,b2,b3 -> logs
        self.simulate(predParams, itr=itr)
        
        logspath = glob.glob(os.path.join(self.savelogPath, '*.txt'))
        
        pdb.set_trace()
        flag = False
        for logpath in logspath:
            
            print(logpath)
            
            self.loadV(logpath)
            self.convV2YearlyData()
            mse = self.calcYearMSE(exactCycles)
            
            if not flag:
                MSE = mse.maxSim
                flag = True
            else:
                MSE = np.vstack([mse, MSE])
        
        return MSE
    # ----
         
