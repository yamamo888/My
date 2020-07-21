# -*- coding: utf-8 -*-

import os
import glob
import pdb
import pickle

import numpy as np
import scipy.io

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

import plot

np.random.seed(1234)


class pdeData:
    def __init__(self, pdeMode='test', dataMode='test'):
        '''
        datamode: small(256/100) or middle(256/50) or large(256/10)
                  Num of X.
        '''
        
        
        self.pdeMode = 'burgers'
        self.modelPath = 'model'
        self.dataMode = dataMode
    
    # ---- 
    def burgers(self, nu=0.01):
        '''
        # viscosity (parameter)
        nu = 0.01 (default)
        '''
        
        # datapoint
        # Num.of time
        tNum = 100 
        # range of time
        tMin = 0.0
        tMax = 1.0
        # time
        t = np.linspace(tMin, tMax, tNum) # [100,]
        
        # Num.of space
        xNum = 256
        # range of space
        xMin = 0.0 # xMin > 0
        xMax = 2.0 * np.pi
        # space
        x = np.linspace(xMin, xMax, xNum) # [256,]
        # observation
        obsu = np.zeros([xNum, tNum])

        for j in range (0, tNum):
            for i in range (0, xNum):
                a = ( x[i] - 4.0 * t[j] )
                b = ( x[i] - 4.0 * t[j] - 2.0 * np.pi )
                c = 4.0 * nu * ( t[j] + 1.0 )
                #pdb.set_trace()
                phi = np.exp ( - a * a / c ) + np.exp ( - b * b / c )
                dphi = - 2.0 * a * np.exp ( - a * a / c ) / c \
                       - 2.0 * b * np.exp ( - b * b / c ) / c
                
                obsu[i,j] = 4.0 - 2.0 * nu * dphi / phi
        #pdb.set_trace()
        return x, t, obsu
    # ---- 
    
    # ----
    def saveXTU(self):
        '''
        minnu, maxnu, swnu 変更で、データ量・データ範囲変更可能.
        '''
        # default
        defaultnu = 0.01
        minnu = 0.005 # minnu < 0.0004　は対応できません(理由不明)
        maxnu = 5.001
        swnu = 0.001
        #swnu = 1.0
        # 一様分布
        samplenu = np.arange(minnu, maxnu, swnu)
        
        cnt = 0
        flag = False
        for nu in samplenu:
            cnt += 1
            print(cnt)
            # x:[256,] t:[100,] obsu:[256,100]
            x, t, obsu = self.burgers(nu=nu)
            
            if not flag:
                X = x
                T = t
                U = obsu.T[np.newaxis]
                NU = np.array([nu])
                flag = True
            else:
                X = np.vstack([X, x]) # [data, 256]
                T = np.vstack([T, t]) # [data, 100]
                U = np.vstack([U, obsu.T[np.newaxis]]) # [data, 100, 256]
                NU = np.hstack([NU, np.array([nu])]) # [data,]
        
        with open(os.path.join(self.modelPath, self.pdeMode, 'XTUNU.pkl'), 'wb') as fp:
            pickle.dump(X, fp)
            pickle.dump(T, fp)
            pickle.dump(U, fp)
            pickle.dump(NU, fp)
    # ----
        
    # ----
    def savetraintest(self, size=10, savepklname='test.pkl'):
        '''
        xSize: small -> 1/1000, middle -> 1/100, large -> 1/10
        '''
        
        with open(os.path.join(self.modelPath, self.pdeMode, 'XTUNU.pkl'), 'rb') as fp:
            X = pickle.load(fp)
            T = pickle.load(fp)
            U = pickle.load(fp)
            NU = pickle.load(fp)
            
        # train data
        # Num.of data
        nData = X.shape[0]
        nTrain = int(nData*0.8)
        
        xSize = int(X.shape[1] / size)
        
        # select x
        # [1] static x (とりあえずこっち) [2] random
        idx = np.random.choice(X.shape[1], xSize, replace=False)
        
        # Index of traindata
        trainID = np.random.choice(X.shape[0], nTrain, replace=False)
        # expectiong trainID
        allind = np.arange(X.shape[0])
        ind = np.ones(nData,dtype=bool)
        ind[trainID] = False
        testID = allind[ind]
        
        # train data
        trainX = X[trainID][:,idx] # [traindata,xsize]
        trainT = T[trainID]
        
        flag = False
        for id in idx:
            allu = U[trainID]
            u = allu[:,:,id]
            
            if not flag:
                trainU = u[:,:,np.newaxis]
                flag = True
            else:
                trainU = np.concatenate([trainU, u[:,:,np.newaxis]],2)
                
        trainNU = NU[trainID]
        
        # test data
        testX = X[testID]
        testT = T[testID] 
        testU = U[testID] # [testdata,100,256]
        testNU = NU[testID] # [testdata]
        
        with open(os.path.join(self.modelPath, self.pdeMode, f'trainXTUNU_{savepklname}.pkl'), 'wb') as fp:
            pickle.dump(trainX, fp)
            pickle.dump(trainT, fp)
            pickle.dump(trainU, fp)
            pickle.dump(trainNU, fp)
        
        with open(os.path.join(self.modelPath, self.pdeMode, f'testXTUNU_{savepklname}.pkl'), 'wb') as fp:
            pickle.dump(testX, fp)
            pickle.dump(testT, fp)
            pickle.dump(testU, fp)
            pickle.dump(testNU, fp)
    # ----
        
    # ----    
    def traintest(self):
        
        # train data
        with open(os.path.join(self.modelPath, self.pdeMode, f'trainXTUNU_{self.dataMode}.pkl'), 'rb') as fp:
            self.trainX = pickle.load(fp)
            self.trainT = pickle.load(fp)
            self.trainU = pickle.load(fp)
            self.trainNU = pickle.load(fp)
        
        # test data
        with open(os.path.join(self.modelPath, self.pdeMode, f'testXTUNU_{self.dataMode}.pkl'), 'rb') as fp:
            testX = pickle.load(fp)
            testT = pickle.load(fp)
            testU = pickle.load(fp)
            testNU = pickle.load(fp)
       
        return testX, testT, testU, testNU
    # ----
        
    # ----
    def nextBatch(self, index):
        
        batchU = self.trainU[index]
        batchNU = self.trainNU[index]
        
        return self.trainX, self.trainT, batchU, batchNU
    # ----
    
#myData = pdeData(datamode='small')
#trainXY, testXY, xt = myData.Loadingburgers()
#trainXY, testXY, xt = myData.burgers()
#myData.saveXTU()

#Size = [10, 50, 100]
#Name = ['large','middle','small']

#for size, name in zip(Size,Name):
    #myData.savetraintest(size=size, savepklname=name)

#myData.traintest()
#myPlot = plot.Plot(figurepath='figure', trialID=0)
#myPlot.udata(xt, trainXY, testXY[1], testXY, testXY)