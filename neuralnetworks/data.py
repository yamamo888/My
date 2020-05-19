#-*- coding: utf-8 -*-

import os
import glob
import pdb
import pickle

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import cycle

class NankaiData:
    def __init__(self, logpath='none', nCell=5, nClass=10, nWindow=10):
	
        # number of input cell
        self.nCell = nCell
        # number of class
        self.nClass = nClass
        # number of sliding window
        self.nWindow = nWindow
        # init batch count
        self.batchCnt = 0
        # cell index
        self.nI = 2
        self.tnI = 4
        self.tI = 5
        
        self.logPath = logpath
    
    # ----
    def makeCycleData(self):
        flag = False 
        for ind in np.arange(8):
            self.loadTrainTestData(nameInds=[ind])
            
            x = self.x11Train
            y1 = self.y21Train
            y2 = self.y41Train
            y3 = self.y51Train

            if not flag:
                X = x
                Y1 = y1
                Y2 = y2
                Y3 = y3

                flag = True
            else:
                X = np.vstack([X,x])
                Y1 = np.hstack([Y1,y1])
                Y2 = np.hstack([Y2,y2])
                Y3 = np.hstack([Y3,y3])
        
        #pdb.set_trace()
        yTrain = np.concatenate((Y1[:,np.newaxis],Y2[:,np.newaxis],Y3[:,np.newaxis]),1)
        X = X[:,1:6,:]
        xTrain = np.reshape(X, [-1,self.nCell*self.nWindow]).astype(np.float32)
        
        tr_randind =  np.random.permutation(xTrain.shape[0])[:int(xTrain.shape[0]*0.01)]
        te_randind =  np.random.permutation(self.xTest.shape[0])[:int(self.xTest.shape[0]*0.01)]
        
        cycle_xTrain = xTrain[tr_randind]
        cycle_yTrain = yTrain[tr_randind]
        cycle_xTest = self.xTest[te_randind]
        cycle_yTest = self.yTest[te_randind]

        myCycle = cycle.Cycle()
        self.loadNankaiRireki()
        
        filename = ['0-100','105-200','205-300','tmp300','400-450'] 

        flag1,flag2 = False,False
        for fID in ['tmp300','400-450']:
            pdb.set_trace()
            logspath = glob.glob(os.path.join('logs',f'b2b3b4b5b6{fID}','*.txt'))
            for logpath in logspath:
                B,_ = myCycle.loadBV(logpath)
                B = np.concatenate([B[self.nI,np.newaxis],B[self.tnI,np.newaxis],B[self.tI,np.newaxis]],0)
                print(logpath)
                # test
                for j in np.arange(te_randind.shape[0]):
                    yb = cycle_yTest[j]
                    
                    if all(B == yb):
                        print('test same')
                        print(yb)
                        print(B)

                        myCycle.convV2YearlyData()
                        pJ,_ = myCycle.calcYearMSE(self.xCycleEval)

                        if not flag1:
                            te_yearMSE = pJ[np.newaxis]
                            te_paramB = yb
                            flag1 = True
                        else:
                            te_yearMSE = np.vstack([te_yearMSE, pJ[np.newaxis]]) # [num.data,250,3]
                            te_paramB = np.vstack([paramB, yb]) # [num.data,3]
                            

                # train
                for i in np.arange(tr_randind.shape[0]):
                    yb = cycle_yTrain[i]

                    if all(B == yb):
                        
                        print('train same')
                        print(yb)
                        print(B)

                        myCycle.convV2YearlyData()
                        pJ,_ = myCycle.calcYearMSE(self.xCycleEval)

                        if not flag2:
                            yearMSE = pJ[np.newaxis]
                            paramB = yb
                            flag2 = True
                        else:
                            yearMSE = np.vstack([yearMSE, pJ[np.newaxis]])
                            paramB = np.vstack([paramB, yb])


        with open(os.path.join(self.features,'cycle','train_cycleVXY.pkl'),'wb') as fp:
            pickle.dump(cycle_xTrain, fp)
            pickle.dump(paramB, fp)
            pickle.dump(yearMSE, fp)
            
        with open(os.path.join(self.features,'cycle','test_cycleVXY.pkl'),'wb') as fp:
            pickle.dump(cycle_xTest, fp)
            pickle.dump(te_paramB, fp)
            pickle.dump(te_yearMSE, fp)
    # ----

    # Load train & test dataset ----
    def loadTrainTestData(self, nameInds=[0]):
        
        # name of train & test pickles
        trainNames = ["b2b3b4b5b6_train{}{}".format(num,self.nClass) for num in np.arange(1,9)]
        testNames = ["b2b3b4b5b6_test1{}".format(self.nClass)]
        
        # train
        flag = False
        for di in nameInds:
            # reading train data from pickle
            with open(os.path.join(self.features,self.nankaipkls,trainNames[di]),'rb') as fp:
                self.xTrain = pickle.load(fp)
                _ = pickle.load(fp)
                _ = pickle.load(fp)
                _ = pickle.load(fp)
                _ = pickle.load(fp)
                _ = pickle.load(fp)
                self.y1Train = pickle.load(fp)
                self.y2Train = pickle.load(fp)
                self.y3Train = pickle.load(fp)
                self.y4Train = pickle.load(fp)
                self.y5Train = pickle.load(fp)
            
            if not flag:
                X = self.xTrain
                Y1 = self.y2Train
                Y2 = self.y4Train
                Y3 = self.y5Train
                flag = True
            else:
                X = np.vstack([X,self.xTrain])
                Y1 = np.hstack([Y1,self.y2Train])
                Y2 = np.hstack([Y2,self.y4Train])
                Y3 = np.hstack([Y3,self.y5Train])
        
        X = X[:,1:6,:]
        self.xTrain = np.reshape(X, [-1,self.nCell*self.nWindow]).astype(np.float32)
        self.yTrain = np.concatenate((Y1[:,np.newaxis],Y2[:,np.newaxis],Y3[:,np.newaxis]),1)
        
        # num.of train
        self.nTrain =  int(self.yTrain.shape[0])
        # random train index
        self.batchRandInd = np.random.permutation(self.nTrain)
            

        # test data
        with open(os.path.join(self.features,self.nankaipkls,testNames[0]),'rb') as fp:
            self.xTest = pickle.load(fp)
            _ = pickle.load(fp)
            _ = pickle.load(fp)
            _ = pickle.load(fp)
            _ = pickle.load(fp)
            _ = pickle.load(fp)
            self.y1Test = pickle.load(fp)
            self.y2Test = pickle.load(fp)
            self.y3Test = pickle.load(fp)
            self.y4Test = pickle.load(fp)
            self.y5Test = pickle.load(fp)

        X = self.xTest[:,1:6,:]
        xTest = np.reshape(X, [-1,self.nCell*self.nWindow]).astype(np.float32)
        yTest = np.concatenate((self.y21Test[:,np.newaxis],self.y41Test[:,np.newaxis],self.y51Test[:,np.newaxis]),1)
        
        return xTest, yTest
    # ----
    
    # Load train & test dataset for cycle loss ----
    def loadCycleTrainTestData(self):
        
        with open(os.path.join(self.logPath,'cycle','train_cycleVXY.pkl'), 'rb') as fp:
            self.xCycleTrain = pickle.load(fp)
            self.yCyclebTrain = pickle.load(fp)
            self.yCycleTrain = pickle.load(fp)
        
        # num.of train
        self.nTrain =  int(self.yCycleTrain.shape[0])
        # random train index
        self.batchRandInd = np.random.permutation(self.nTrain)
        
        with open(os.path.join(self.logPath,'cycle','test_cycleVXY.pkl'), 'rb') as fp:
            xCycleTest = pickle.load(fp)
            yCyclebTest = pickle.load(fp)
            yCycleTest = pickle.load(fp)
            
        return xCycleTest, yCyclebTest, yCycleTest
    # ----
    
    # Load exact rireki ----
    def loadNankaiRireki(self):
    
        # nankaifeatue.pkl -> 190.pkl
        
        # X (FFT feature) ----
        fID = 190
        fftpath = os.path.join(self.features,"eval","{}.pkl".format(fID))
        with open(fftpath,"rb") as fp:
            data = pickle.load(fp)
        xEval = np.reshape(np.concatenate([data[0][np.newaxis],data[0][np.newaxis],data[1][np.newaxis],data[1][np.newaxis],data[2][np.newaxis]]),[-1,]) # [50,]
        
        # eq.year for Cycle loss ----        
        rirekipath = os.path.join(self.features,"eval","nankairireki.pkl")
        with open(rirekipath ,'rb') as fp:
            data = pickle.load(fp)
        xrireki = data[fID,:,:]
        yCycleEval = [np.where(xrireki[:,0]>0)[0], np.where(xrireki[:,1]>0)[0], np.where(xrireki[:,2]>0)[0]] # [[eq.year in nk], [eq.year in tnk], [eq.year in tk]]
        
        return xEval, yCycleEval
    # ----
        
    # Make mini-batch dataset ----        
    def nextBatch(self, nBatch=100, isCycle=False):
        
        sInd = nBatch * self.batchCnt
        eInd = sInd + nBatch
        
        if isCycle:
            batchX = self.xCycleTrain[sInd:eInd]
            batchY = self.yCyclebTrain[sInd:eInd]
            batchCycleY = self.yCycleTrain[sInd:eInd]
            
            batchXY = [batchX, batchY, batchCycleY]
            
        else:
            batchX = self.xTrain[sInd:eInd]
            batchY = self.YTrain[sInd:eInd]
            
            batchXY = [batchX, batchY]
     
        if eInd + nBatch > self.nTrain:
            self.batchCnt = 0
        else:
            self.batchCnt += 1
        
        return batchXY
    # ----
    

