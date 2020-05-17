#-*- coding: utf-8 -*-

import os
import pdb
import pickle

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

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
        
        # -----------------------------------------------------------------------------

        # ------------------------------- path ----------------------------------------
        self.features = "features"
        self.nankaipkls = "nankaipickles"
        self.nankaifeature = "nankairirekiFFT"
        self.nankairireki = "nankairireki.pkl"
        self.rirekiFullPath = os.path.join(self.features,self.nankairireki)
        # -----------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------#      
    def loadTrainTestData(self,nameInds=[0,1,2,3,4]):
        
        # name of train pickles
        trainNames = ["b2b3b4b5b6_train{}{}".format(num,self.nClass) for num in np.arange(1,8)]
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
        
        # train data2
        with open(os.path.join(self.features,self.nankaipkls,trainNames[nameInds[1]]),'rb') as fp:
            self.x12Train = pickle.load(fp)
            self.y12TrainLabel = pickle.load(fp)
            self.y22TrainLabel = pickle.load(fp)
            self.y32TrainLabel = pickle.load(fp)
            self.y42TrainLabel = pickle.load(fp)
            self.y52TrainLabel = pickle.load(fp)
            self.y12Train = pickle.load(fp)
            self.y22Train = pickle.load(fp)
            self.y32Train = pickle.load(fp)
            self.y42Train = pickle.load(fp)
            self.y52Train = pickle.load(fp)
        
        # train data3
        with open(os.path.join(self.features,self.nankaipkls,trainNames[nameInds[2]]),'rb') as fp:
            self.x13Train = pickle.load(fp)
            self.y13TrainLabel = pickle.load(fp)
            self.y23TrainLabel = pickle.load(fp)
            self.y33TrainLabel = pickle.load(fp)
            self.y43TrainLabel = pickle.load(fp)
            self.y53TrainLabel = pickle.load(fp)
            self.y13Train = pickle.load(fp)
            self.y23Train = pickle.load(fp)
            self.y33Train = pickle.load(fp)
            self.y43Train = pickle.load(fp)
            self.y53Train = pickle.load(fp)
        
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
        self.yTest = np.concatenate((self.y11Test[:,np.newaxis],self.y31Test[:,np.newaxis],self.y51Test[:,np.newaxis]),1)
        # test label y
        #self.yTestLabel = np.concatenate((self.y11TestLabel[:,:,np.newaxis],self.y31TestLabel[:,:,np.newaxis],self.y51TestLabel[:,:,np.newaxis]),2)
        # number of train data
        self.nTrain =  int(self.x11Train.shape[0] + self.x12Train.shape[0] + self.x13Train.shape[0])
        # random train index
        self.batchRandInd = np.random.permutation(self.nTrain)
        
        pdb.set_trace()
        # self.yCycleTest
        
    #-----------------------------------------------------------------------------#    
    def loadNankaiRireki(self):
        
        # nankai rireki path (slip velocity V)
        # nankaifeatue.pkl -> 190.pkl
        flag = False
        for fID in np.arange(256):
            # pkl path
            nankairirekiPath = os.path.join(self.features,"rirekifeatures","{}.pkl".format(fID))
            
            with open(nankairirekiPath,"rb") as fp:
                data = pickle.load(fp)
            x = np.reshape(np.concatenate([data[0][np.newaxis],data[0][np.newaxis],data[1][np.newaxis],data[1][np.newaxis],data[2][np.newaxis]]),[-1,])

            if not flag:
                evalX = x[np.newaxis]
                flag = True
            else:
                evalX = np.vstack([evalX,x[np.newaxis]])

        self.xEval = evalX
        
        pdb.set_trace()
        
    #-----------------------------------------------------------------------------#
    def nextBatch(self,batchSize):
        
        sInd = batchSize * self.batchCnt
        eInd = sInd + batchSize
       
        # [number of data, cell(=5,nankai2 & tonakai2 & tokai1), dimention of features(=10)]
        trX = np.concatenate((self.x11Train[:,1:6,:],self.x12Train[:,1:6,:],self.x13Train[:,1:6,:]),0) 
        # mini-batch, [number of data, cell(=5)*dimention of features(=10)]
        self.batchX = np.reshape(trX[sInd:eInd],[-1,self.nCell*self.nWindow])
        # test all targets
        trY1 = np.concatenate((self.y11Train,self.y12Train,self.y13Train),0)
        trY2 = np.concatenate((self.y31Train,self.y32Train,self.y33Train),0)
        trY3 = np.concatenate((self.y51Train,self.y52Train,self.y53Train),0)
        # [number of data(mini-batch), cell(=3)] 
        self.batchY = np.concatenate((trY1[sInd:eInd,np.newaxis],trY2[sInd:eInd,np.newaxis],trY3[sInd:eInd,np.newaxis]),1)
        
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


