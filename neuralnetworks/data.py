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
    def __init__(self, logpath='logs', nCell=5, nClass=10, nWindow=10):
	
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
        self.featurePath = 'features'
    
    # all & part of xy data ----
    def makeCycleData(self, isPart=False):
        '''
        isPart: select part of data
        '''
        
        if isPart:
            tr_randind =  np.random.permutation(xTrain.shape[0])[:int(xTrain.shape[0]*0.01)]
            te_randind =  np.random.permutation(self.xTest.shape[0])[:int(self.xTest.shape[0]*0.01)]
            
            cycle_xTrain = xTrain[tr_randind]
            cycle_yTrain = yTrain[tr_randind]
            cycle_xTest = self.xTest[te_randind]
            cycle_yTest = self.yTest[te_randind]

        myCycle = cycle.Cycle()
        self.loadNankaiRireki
        self.loadTrainTestData()
        
        filename = ['b2b3b4b5b60-100','b2b3b4b5b6105-200','b2b3b4b5b6205-300','tmp300','b2b3b4b5b6400-450'] 

        flag1,flag2 = False,False
        for fID in filename:
            cnt = 0
            logspath = glob.glob(os.path.join('logs',f'{fID}','*.txt'))
            
            for logpath in logspath:
                B,_ = myCycle.loadBV(logpath)
                B = np.concatenate([B[self.nI,np.newaxis],B[self.tnI,np.newaxis],B[self.tI,np.newaxis]],0)
                print(f'{fID}: {len(logspath)-cnt}')
                cnt += 1
                
                '''
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
                            te_paramB = np.vstack([paramB, yb])''' # [num.data,3]
                            
                # train
                # if == isPart
                #for i in np.arange(tr_randind.shape[0]):
                    #yb = cycle_yTrain[i]
                for i in np.arange(self.xTrain.shape[0]):
                    yb = self.yTrain[i]
                    x = self.xTrain[i]


                    if match == 1:
                        pass
                    elif match == 0:

                        pdb.set_trace()
                        if all(B == yb):
                            match = 1 
                            print('train same')
                            print(yb)
                            print(B)

                            myCycle.convV2YearlyData()
                            pJ,_ = myCycle.calcYearMSE(self.xCycleEval)

                            if not flag2:
                                yearMSE = pJ[np.newaxis]
                                paramB = yb
                                X = x
                                flag2 = True
                            else:
                                yearMSE = np.vstack([yearMSE, pJ[np.newaxis]])
                                paramB = np.vstack([paramB, yb])
                                X = np.vstack([X,x])
                                pdb.set_trace()


            with open(os.path.join(self.featurePath,'cycle','train_allcycleVXY_{fID}.pkl'),'wb') as fp:
                pickle.dump(X, fp)
                pickle.dump(paramB, fp)
                pickle.dump(yearMSE, fp)
        '''
        with open(os.path.join(self.featurePath,'cycle','test_cycleVXY.pkl'),'wb') as fp:
            pickle.dump(cycle_xTest, fp)
            pickle.dump(te_paramB, fp)
            pickle.dump(te_yearMSE, fp)
        '''
     # ----
    
    # Load train & test dataset ----
    def loadTrainTestData(self, nameInds=[0]):
        
        # name of train & test pickles
        trainNames = ["b2b3b4b5b6_train{}{}".format(num,self.nClass) for num in np.arange(1,9)]
        #testNames = ["b2b3b4b5b6_test1{}".format(self.nClass)]
        
        # train
        flag = False
        #for di in nameInds:
        for di in np.arange(8):
            # reading train data from pickle
            with open(os.path.join(self.featurePath, 'traintest' ,trainNames[di]),'rb') as fp:
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
                Y1,Y2,Y3 = self.y2Train,self.y4Train,self.y5Train
                flag = True
            else:
                X = np.vstack([X,self.xTrain])
                Y1,Y2,Y3 = np.hstack([Y1,self.y2Train]),np.hstack([Y2,self.y4Train]),np.hstack([Y3,self.y5Train])
        X = X[:,1:6,:]
        self.xTrain = np.reshape(X, [-1,self.nCell*self.nWindow]).astype(np.float32)
        self.yTrain = np.concatenate((Y1[:,np.newaxis],Y2[:,np.newaxis],Y3[:,np.newaxis]),1)
        '''
        # num.of train
        self.nTrain =  int(self.yTrain.shape[0])
        # random train index
        self.batchRandInd = np.random.permutation(self.nTrain)
        
        # test data
        with open(os.path.join(self.featurePath, 'traintest' ,testNames[0]),'rb') as fp:
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
        yTest = np.concatenate((self.y2Test[:,np.newaxis],self.y4Test[:,np.newaxis],self.y5Test[:,np.newaxis]),1)
        
        return xTest, yTest
        '''
    # ----
    
    # Load train & test dataset for cycle loss ----
    def loadCycleTrainTestData(self):
        
        with open(os.path.join(self.featurePath,'cycle','train_cycleVXY.pkl'), 'rb') as fp:
            self.xCycleTrain = pickle.load(fp)
            self.yCyclebTrain = pickle.load(fp)
            self.yCycleTrain = pickle.load(fp)
        
        # num.of train
        self.nTrain =  int(self.yCycleTrain.shape[0])
        # random train index
        self.batchRandInd = np.random.permutation(self.nTrain)
        
        with open(os.path.join(self.featurePath,'cycle','test_cycleVXY.pkl'), 'rb') as fp:
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
        fftpath = os.path.join(self.featurePath,"eval","{}.pkl".format(fID))
        with open(fftpath,"rb") as fp:
            data = pickle.load(fp)
        xEval = np.reshape(np.concatenate([data[0][np.newaxis],data[0][np.newaxis],data[1][np.newaxis],data[1][np.newaxis],data[2][np.newaxis]]),[-1,]) # [50,]
        
        # eq.year for Cycle loss ----        
        rirekipath = os.path.join(self.featurePath,"eval","nankairireki.pkl")
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
            batchY = self.yTrain[sInd:eInd]
            
            batchXY = [batchX, batchY]
     
        if eInd + nBatch > self.nTrain:
            self.batchCnt = 0
        else:
            self.batchCnt += 1
        
        return batchXY
    # ----
    
NankaiData().makeCycleData()
