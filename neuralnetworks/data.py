#-*- coding: utf-8 -*-

import os
import pdb
import pickle
import glob

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import cycle

class NankaiData:
    def __init__(self,nCell=5,nClass=10,nWindow=10):
        #----------------------------- paramters --------------------------------------
		
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
        
        
        # -----------------------------------------------------------------------------

        # ------------------------------- path ----------------------------------------
        self.features = "features"
        self.nankaipkls = "nankaipickles"
        # -----------------------------------------------------------------------------
    
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
        
    #-----------------------------------------------------------------------------#      
    def loadTrainTestData(self,nameInds=[0,1,2,3,4]):
        
        # name of train pickles
        trainNames = ["b2b3b4b5b6_train{}{}".format(num,self.nClass) for num in np.arange(1,9)]
        # name of test pickles
        testNames = ["b2b3b4b5b6_test1{}".format(self.nClass)]
        
        # reading train data from pickle
        with open(os.path.join(self.features,self.nankaipkls,trainNames[nameInds[0]]),'rb') as fp:
            self.x11Train = pickle.load(fp)
            self.y11TrainLabel = pickle.load(fp)
            self.y21TrainLabel = pickle.load(fp)
            self.y31TrainLabel = pickle.load(fp)
            self.y41TrainLabel = pickle.load(fp)
            self.y51TrainLabel = pickle.load(fp)
            self.y11Train = pickle.load(fp)
            self.y21Train = pickle.load(fp)
            self.y31Train = pickle.load(fp)
            self.y41Train = pickle.load(fp)
            self.y51Train = pickle.load(fp)

        # test data
        with open(os.path.join(self.features,self.nankaipkls,testNames[0]),'rb') as fp:
            self.xTest = pickle.load(fp)
            self.y11TestLabel = pickle.load(fp)
            self.y21TestLabel = pickle.load(fp)
            self.y31TestLabel = pickle.load(fp)
            self.y41TestLabel = pickle.load(fp)
            self.y51TestLabel = pickle.load(fp)
            self.y11Test = pickle.load(fp)
            self.y21Test = pickle.load(fp)
            self.y31Test = pickle.load(fp)
            self.y41Test = pickle.load(fp)
            self.y51Test = pickle.load(fp)

        
        #[number of data,]
        self.xTest = self.xTest[:,1:6,:]
        self.xTest = np.reshape(self.xTest,[-1,self.nCell*self.nWindow]).astype(np.float32)
        # test y
        self.yTest = np.concatenate((self.y21Test[:,np.newaxis],self.y41Test[:,np.newaxis],self.y51Test[:,np.newaxis]),1)
        """
        # [number of data, cell(=5,nankai2 & tonakai2 & tokai1), dimention of features(=10)]
        trX = np.concatenate((self.x11Train[:,1:6,:],self.x12Train[:,1:6,:],self.x13Train[:,1:6,:]),0) 
        # mini-batch, [number of data, cell(=5)*dimention of features(=10)]
        self.batchX = np.reshape(trX[sInd:eInd],[-1,self.nCell*self.nWindow])
        # test all targets
        trY1 = np.concatenate((self.y21Train,self.y22Train,self.y23Train),0)
        trY2 = np.concatenate((self.y41Train,self.y42Train,self.y43Train),0)
        trY3 = np.concatenate((self.y51Train,self.y52Train,self.y53Train),0)
        # [number of data(mini-batch), cell(=3)] 
        self.batchY = np.concatenate((trY1[sInd:eInd,np.newaxis],trY2[sInd:eInd,np.newaxis],trY3[sInd:eInd,np.newaxis]),1)
        """
        # test label y
        #self.yTestLabel = np.concatenate((self.y11TestLabel[:,:,np.newaxis],self.y31TestLabel[:,:,np.newaxis],self.y51TestLabel[:,:,np.newaxis]),2)
        # number of train data
        #self.nTrain =  int(self.x11Train.shape[0] + self.x12Train.shape[0] + self.x13Train.shape[0])
        # random train index
        #self.batchRandInd = np.random.permutation(self.nTrain)
        #pdb.set_trace()
        # self.yCycleTest
        
    #-----------------------------------------------------------------------------#    
    def loadNankaiRireki(self):
        
        # nankaifeatue.pkl -> 190.pkl
        
        # X (FFT feature) ----
        fID = 190
        fftpath = os.path.join(self.features,"eval","{}.pkl".format(fID))
        with open(fftpath,"rb") as fp:
            data = pickle.load(fp)
        xfft = np.reshape(np.concatenate([data[0][np.newaxis],data[0][np.newaxis],data[1][np.newaxis],data[1][np.newaxis],data[2][np.newaxis]]),[-1,])
        
        # eq.year for Cycle loss ----        
        rirekipath = os.path.join(self.features,"eval","nankairireki.pkl")
        with open(rirekipath ,'rb') as fp:
            data = pickle.load(fp)
        xrireki = data[fID,:,:]
        xyear = [np.where(xrireki[:,0]>0)[0], np.where(xrireki[:,1]>0)[0], np.where(xrireki[:,2]>0)[0]]

        self.xEval = xfft # [50,]
        self.xCycleEval = xyear # [[eq.year in nk], [eq.year in tnk], [eq.year in tk]]
        
    #-----------------------------------------------------------------------------#
    def nextBatch(self, nBatch=100):
        
        sInd = nBatch * self.batchCnt
        eInd = sInd + nBatch
       
        
        self.batchX = self.X[sInd:eInd,:]
        self.batchY = self.Y[sInd:eInd,:]
 
        pdb.set_trace()
        #trCycleY1
        #trCycleY2
        #trCycleY3
        
        # self.batchCycleY
        
        if eInd + batchSize > self.nTrain:
            self.batchCnt = 0
        else:
            self.batchCnt += 1

    #-----------------------------------------------------------------------------#

#myData = NankaiData(nCell=5,nClass=10,nWindow=10)
#myData.makeCycleData()

