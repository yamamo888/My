# -*- coding: utf-8 -*-

import os
import glob
import time
import shutil

import numpy as np

import pdb

import matplotlib.pylab as plt


class Cycle:
    def __init__(self, logpath='logs', trialID=0):
        
        # slip velocity
        self.slip = 0
        # cell index
        self.ntI = 0
        self.tntI = 1
        self.ttI = 2
        self.allCell = 8
        self.trialID = trialID
        
        self.logsPath = logpath
        
        
    # ----
    def simulate(self, allparams, itr=0, dirpath='train'):
        
        minB,maxB = 0.011,0.0165
        params = np.zeros(3)
        
        saveparampath = os.path.join('model', 'params', f'{dirpath}', f'{self.trialID}')
        
        if itr % 1000 == 0:
            # copy all paramb
            np.savetxt(os.path.join(saveparampath, f'logs{itr}.csv'), allparams, delimiter=',', fmt='%5f')
            
        # Select paramb for simulater
        flag = False
        for allparam in allparams:
            
            if (minB <= allparam[self.ntI] <= maxB) and (minB <= allparam[self.tntI] <= maxB) and (minB <= allparam[self.ttI] <= maxB):
                #pdb.set_trace()
                pind = np.where((allparam[self.ntI] == allparams[:,self.ntI])&
                                (allparam[self.tntI] == allparams[:,self.tntI])&
                                (allparam[self.ttI] == allparams[:,self.ttI]))[0][0]
                
                if not flag:
                    params = allparam
                    pinds = pind
                    flag = True
                else:
                    params = np.vstack([params,allparam])
                    pinds = np.vstack([pinds,pind])
        
        # no params for simulater
        if np.sum(params) == 0:
            pass
        
        else:
            #pdb.set_trace()
            params = params*1000000
            self.ind_params = np.hstack([params, pinds])
            
            if self.ind_params.ndim == 1:
                self.ind_params = self.ind_params[np.newaxis]
            
            # for loading batfile
            np.savetxt(os.path.join('model', 'logs.csv'), self.ind_params, delimiter=',', fmt='%d')
            
            print(f'>>> {itr}times Start simulation')
            
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
    def convV2YearlyData(self, isLSTM=False):
    
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
        
        if isLSTM:
            # for input (5cell)
            nk1Year = np.where(deltaU[stateYear:,1] > self.slip)[0]
            nk2Year = np.where(deltaU[stateYear:,vInds[0]] > self.slip)[0]
            tnk1Year = np.where(deltaU[stateYear:,vInds[1]] > self.slip)[0]
            tnk2Year = np.where(deltaU[stateYear:,vInds[2]] > self.slip)[0]
            tkYear = np.where(deltaU[stateYear:,vInds[3]] > self.slip)[0]
            self.predyear = [nk1Year,nk2Year,tnk1Year,tnk2Year,tkYear]
    
        else:
            # ※ Uの発生年数
            nkYear = np.where(deltaU[stateYear:,vInds[0]] > self.slip)[0]
            tnkYear = np.where(deltaU[stateYear:,vInds[2]] > self.slip)[0]
            tkYear = np.where(deltaU[stateYear:,vInds[3]] > self.slip)[0]
            self.predyear = [nkYear,tnkYear,tkYear]
    # ----
    
    # ---- 
    def calcYearMSE(self, gt, isLSTM=False):
        '''
        gt: exact eq.year, list[numpy]
        '''
        #pdb.set_trace()
        aYear = 1400
        nCell = 3
        th = 1
        
        # ground truth eq.
        gYear_nk = gt[self.ntI]
        gYear_tnk = gt[self.tntI]
        gYear_tk = gt[self.ttI]
        
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
        
        # minimum mse year
        self.maxSim = yearErrors[sInd]
        
        if isLSTM:
            # minimum eq.year 
            self.nk1Year = self.predyear[0][(self.predyear[0] > sInd) & (self.predyear[0] < eInd)] - sInd
            self.nk2Year = self.predyear[1][(self.predyear[1] > sInd) & (self.predyear[1] < eInd)] - sInd
            self.tnk1Year = self.predyear[2][(self.predyear[2] > sInd) & (self.predyear[2] < eInd)] - sInd
            self.tnk2Year = self.predyear[3][(self.predyear[3] > sInd) & (self.predyear[3] < eInd)] - sInd
            self.tkYear = self.predyear[4][(self.predyear[4] > sInd) & (self.predyear[4] < eInd)] - sInd
            
        else:
            # minimum eq.year 
            nkYear = self.predyear[self.ntI][(self.predyear[self.ntI] > sInd) & (self.predyear[self.ntI] < eInd)] - sInd
            tnkYear = self.predyear[self.tntI][(self.predyear[self.tntI] > sInd) & (self.predyear[self.tntI] < eInd)] - sInd
            tkYear = self.predyear[self.ttI][(self.predyear[self.ttI] > sInd) & (self.predyear[self.ttI] < eInd)] - sInd
            
            #nkJ = np.pad(nkYear, (0, 500 - nkYear.shape[0]), 'constant', constant_values=0)
            #tnkJ = np.pad(tnkYear, (0, 500 - tnkYear.shape[0]), 'constant', constant_values=0)
            #tkJ = np.pad(tkYear, (0, 500 - tkYear.shape[0]), 'constant', constant_values=0)
            #self.pJ = np.concatenate([nkJ[:,np.newaxis], tnkJ[:,np.newaxis], tkJ[:,np.newaxis]],1) #[150,3]
            
            return self.maxSim
            #return self.pJ, self.maxSim
    # ----
    
    # ----
    def calcInterval(self):
        '''
        return
            interval.
            year interval.
        '''
        pdb.set_trace()
        
        # interval
        interval_nk1 = self.nk1Year[1:] - self.nk1Year[:-1]
        interval_nk2 = self.nk2Year[1:] - self.nk2Year[:-1]
        interval_tnk1 = self.tnk1Year[1:] - self.tnk1Year[:-1]
        interval_tnk2 = self.tnk2Year[1:] - self.tnk2Year[:-1]
        interval_tk = self.tkYear[1:] - self.tkYear[:-1]
        
        intervals = [interval_nk1,interval_nk2,interval_tnk1,interval_tnk2,interval_tk]
        
        # length of interval
        seq_nk1 = len(interval_nk1)
        seq_nk2 = len(interval_nk2)
        seq_tnk1 = len(interval_tnk1)
        seq_tnk2 = len(interval_tnk2)
        seq_tk = len(interval_tk)
        
        # maximum length of interval
        max_seq = np.max([seq_nk1,seq_nk2,seq_tnk1,seq_tnk2,seq_tk])
        
        return intervals, max_seq
    # ----

    # ----
    def loss(self, predParams, gtCycles, itr=0, dirpath='train'):
        
        '''
        gtCycles: gt eq.year, numpy [batchsize,500(zero-padding),3]
        '''
        
        # Make logs        
        self.simulate(predParams, itr=itr, dirpath=dirpath)
        
        # Save predb
        savelogspath = os.path.join(self.logsPath, 'pcNN', '*.txt')
        logsallpath = glob.glob(savelogspath)
        
        MSE = np.zeros(predParams.shape[0])
        # all no paramb
        if logsallpath == []:
            # ※
            MSE = np.sum(np.abs(predParams),1)*1000
        
        else:
            
            # if maxB > paramB
            delInd = self.ind_params[:,-1].astype(int).tolist()
            pinds = np.ones(predParams.shape[0], dtype=bool)
            pinds[delInd] = False
                    
            # if minB < paramB < maxB
            for logpath in logsallpath:
                
                #pdb.set_trace()
                ind = int(os.path.basename(logpath).split('_')[0])
                
                # logs -> V,B
                self.loadBV(logpath)
                self.convV2YearlyData()
                
                gtCycles_nk = np.trim_zeros(gtCycles[ind,:,self.ntI])
                gtCycles_tnk = np.trim_zeros(gtCycles[ind,:,self.tntI])
                gtCycles_tk = np.trim_zeros(gtCycles[ind,:,self.ttI])            
                gt = [gtCycles_nk, gtCycles_tnk, gtCycles_tk]
                
                self.calcYearMSE(gt)            
                MSE[ind] = self.maxSim
                
                os.remove(logpath)
            
            # ※ sum(paramB)        
            worstMS = np.max(MSE)
            MSE[pinds] = worstMS * np.sum(np.abs(predParams[pinds]),1)
        
        return MSE
    # ----
         
