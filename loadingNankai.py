# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 23:59:09 2019

@author: yu
"""

import os
import pdb
import pickle

import numpy as np

class NankaiData:
    def __init__(self,nCell=5,nClass=10,nWindow=10,cellInds=[1,3,4]):
        #----------------------------- paramters --------------------------------------
		
        # number of input cell
        self.nCell = nCell
        # number of class
        self.nClass = nClass
        # number of sliding window
        self.nWindow = nWindow
        # init batch count
        self.batchCnt = 0
        # cell index nankai(1,2), tonankai(3,4), tokai(5) -> 3/5
        self.cellInds = cellInds
        
        # -----------------------------------------------------------------------------

        # ------------------------------- path ----------------------------------------
        self.features = "features"
        self.nankaipkls = "nankaipickles"
        self.nankairireki = "nankairireki"
        # -----------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------#      
    def loadTrainTestData(self,nameInds=[0,1,2]):
        
        # name of train pickles
        trainNames = ["b2b3b4b5b6_train{}{}".format(num,self.nClass) for num in np.arange(1,6)]
        # name of test picles
        testNames = ["b2b3b4b5b6_test{}{}".format(num,self.nClass) for num in np.arange(1,3)]
        
        
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
        
		# reading test data from pickle
        with open(os.path.join(self.features,self.nankaipkls,testNames[0]),'rb') as fp:
            self.x1Test = pickle.load(fp)
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

        # test data2
        with open(os.path.join(self.features,self.nankaipkls,testNames[1]),'rb') as fp:
            self.x2Test = pickle.load(fp)
            self.y12TestLabel = pickle.load(fp)
            self.y22TestLabel = pickle.load(fp)
            self.y32TestLabel = pickle.load(fp)
            self.y42TestLabel = pickle.load(fp)
            self.y52TestLabel = pickle.load(fp)
            self.y12Test = pickle.load(fp)
            self.y22Test = pickle.load(fp)
            self.y32Test = pickle.load(fp)
            self.y42Test = pickle.load(fp)
            self.y52Test = pickle.load(fp)
       
        # [number of data, cell(=5,nankai2 & tonakai2 & tokai1), dimention of features(=10)]
        teX = np.concatenate((self.x1Test[:,1:6,:],self.x2Test[:,1:6,:]),0) 
        # [number of data, cell(=5)*dimention of features(=10)]
        self.xTest = np.reshape(teX,[-1,self.nCell*self.nWindow])
        
        # test all labels
        telabel1 = [self.y11TestLabel,self.y21TestLabel,self.y31TestLabel,self.y41TestLabel,self.y51TestLabel]
        telabel2 = [self.y12TestLabel,self.y22TestLabel,self.y32TestLabel,self.y42TestLabel,self.y52TestLabel]
        
        # test class label (y1=nankai, y2=tonakai, y3=tokai)
        self.y1TestLabel = np.concatenate((telabel1[self.cellInds[0]],telabel2[self.cellInds[0]]),0)
        self.y2TestLabel = np.concatenate((telabel1[self.cellInds[1]],telabel2[self.cellInds[1]]),0)
        self.y3TestLabel = np.concatenate((telabel1[self.cellInds[2]],telabel2[self.cellInds[2]]),0)
        # [number of data, number of class(self.nClass), cell(=3)] 
        self.yTestLabel = np.concatenate((self.y1TestLabel[:,:,np.newaxis],self.y2TestLabel[:,:,np.newaxis],self.y3TestLabel[:,:,np.newaxis]),2)
        
        # test all targets
        teY1 = [self.y11Test,self.y21Test,self.y31Test,self.y41Test,self.y51Test]
        teY2 = [self.y12Test,self.y22Test,self.y32Test,self.y42Test,self.y52Test]
        # test target
        self.y1Test = np.concatenate((teY1[self.cellInds[0]],teY2[self.cellInds[0]]),0)
        self.y2Test = np.concatenate((teY1[self.cellInds[1]],teY2[self.cellInds[1]]),0)
        self.y3Test = np.concatenate((teY1[self.cellInds[2]],teY2[self.cellInds[2]]),0)
        # [number of data, cell(=3)] 
        self.yTest = np.concatenate((self.y1Test[:,np.newaxis],self.y2Test[:,np.newaxis],self.y3Test[:,np.newaxis]),1)
        
        # number of train data
        self.nTrain =  self.x11Train.shape[0]
        self.batchRandInd = np.random.permutation(self.nTrain)
        #self.batchRandInd2 = np.random.permutation(self.nTrain)
        #self.batchRandInd3 = np.random.permutation(self.nTrain)

    #-----------------------------------------------------------------------------#    
    def loadNankaiRireki(self):
        
        # nankai rireki path (slip velocity V)
        nankairirekiPath = os.path.join(self.features,self.nankairireki)
        
        with open(nankairirekiPath,"rb") as fp:
            self.evalX = pickle.load(fp)
            
        
    #-----------------------------------------------------------------------------#
    def nextBatch(self,batchSize):
        
        sInd = batchSize * self.batchCnt
        eInd = sInd + batchSize
       
        #pdb.set_trace() 
        # [number of data, cell(=5,nankai2 & tonakai2 & tokai1), dimention of features(=10)]
        trX = np.concatenate((self.x11Train[self.batchRandInd[sInd:eInd],1:6,:],self.x12Train[self.batchRandInd[sInd:eInd],1:6,:],self.x13Train[self.batchRandInd[sInd:eInd],1:6,:]),0) 
        # [number of data, cell(=5)*dimention of features(=10)]
        batchX = np.reshape(trX,[-1,self.nCell*self.nWindow])
        
        # test all labels
        trlabel1 = [self.y11TrainLabel[self.batchRandInd[sInd:eInd]],self.y21TrainLabel[self.batchRandInd[sInd:eInd]],self.y31TrainLabel[self.batchRandInd[sInd:eInd]],self.y41TrainLabel[self.batchRandInd[sInd:eInd]],self.y51TrainLabel[self.batchRandInd[sInd:eInd]]]
        trlabel2 = [self.y12TrainLabel[self.batchRandInd[sInd:eInd]],self.y22TrainLabel[self.batchRandInd[sInd:eInd]],self.y32TrainLabel[self.batchRandInd[sInd:eInd]],self.y42TrainLabel[self.batchRandInd[sInd:eInd]],self.y52TrainLabel[self.batchRandInd[sInd:eInd]]]
        trlabel3 = [self.y13TrainLabel[self.batchRandInd[sInd:eInd]],self.y23TrainLabel[self.batchRandInd[sInd:eInd]],self.y33TrainLabel[self.batchRandInd[sInd:eInd]],self.y43TrainLabel[self.batchRandInd[sInd:eInd]],self.y53TrainLabel[self.batchRandInd[sInd:eInd]]]
        
        # test class label (y1=nankai, y2=tonakai, y3=tokai)
        self.batchlabelY1 = np.concatenate((trlabel1[self.cellInds[0]],trlabel2[self.cellInds[0]],trlabel3[self.cellInds[0]]),0)
        self.batchlabelY2 = np.concatenate((trlabel1[self.cellInds[1]],trlabel2[self.cellInds[1]],trlabel3[self.cellInds[1]]),0)
        self.batchlabelY3 = np.concatenate((trlabel1[self.cellInds[2]],trlabel2[self.cellInds[2]],trlabel3[self.cellInds[2]]),0)
        # [number of data, number of class(self.nClass), cell(=3)] 
        batchlabelY = np.concatenate((self.batchlabelY1[:,:,np.newaxis],self.batchlabelY2[:,:,np.newaxis],self.batchlabelY3[:,:,np.newaxis]),2)
        
        # test all targets
        trY1 = [self.y11Train[self.batchRandInd[sInd:eInd]],self.y21Train[self.batchRandInd[sInd:eInd]],self.y31Train[self.batchRandInd[sInd:eInd]],self.y41Train[self.batchRandInd[sInd:eInd]],self.y51Train[self.batchRandInd[sInd:eInd]]]
        trY2 = [self.y12Train[self.batchRandInd[sInd:eInd]],self.y22Train[self.batchRandInd[sInd:eInd]],self.y32Train[self.batchRandInd[sInd:eInd]],self.y42Train[self.batchRandInd[sInd:eInd]],self.y52Train[self.batchRandInd[sInd:eInd]]]
        trY3 = [self.y13Train[self.batchRandInd[sInd:eInd]],self.y23Train[self.batchRandInd[sInd:eInd]],self.y33Train[self.batchRandInd[sInd:eInd]],self.y43Train[self.batchRandInd[sInd:eInd]],self.y53Train[self.batchRandInd[sInd:eInd]]]
        # test target
        self.batchY1 = np.concatenate((trY1[self.cellInds[0]],trY2[self.cellInds[0]],trY3[self.cellInds[0]]),0)
        self.batchY2 = np.concatenate((trY1[self.cellInds[1]],trY2[self.cellInds[1]],trY3[self.cellInds[0]]),0)
        self.batchY3 = np.concatenate((trY1[self.cellInds[2]],trY2[self.cellInds[2]],trY3[self.cellInds[0]]),0)
        # [number of data, cell(=3)] 
        batchY = np.concatenate((self.batchY1[:,np.newaxis],self.batchY2[:,np.newaxis],self.batchY3[:,np.newaxis]),1)

        if eInd + batchSize > self.nTrain:
            self.batchCnt = 0
        else:
            self.batchCnt += 1
        
        return batchX, batchY, batchlabelY
    #-----------------------------------------------------------------------------#
