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
    def __init__(self,nClass=10):
        #----------------------------- paramters --------------------------------------
		
        self.nClass = nClass
        
        self.batchCnt = 0
        
        self.batchRandInd = np.random.permutation(self.nTrain)
        
        self.features = "features"
        self.nankaipkls = "nankaipickles"
        self.nankairireki = "nankairireki"
    
	#-----------------------------------------------------------------------------#
    def loadTrainTestData(self,nameInds=[0,1,2]):
        
        # name of train pickles
        trainNames = ["b2b3b4b5b6_train{}{}".format(num,self.nClass) for num in np.arange(1,6)]
        # name of test picles
        testNames = ["b2b3b4b5b6_test{}{}".format(num,self.nClass) for num in np.arange(1,2)]
        
        
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
       
        pdb.set_trace()
        self.xTest = np.concatenate((self.x1Test,self.x2Test),0)
        self.y1TestLabel = np.concatenate((self.y21TestLabel,self.y22TestLabel),0)
        self.y2TestLabel = np.concatenate((self.y41TestLabel,self.y42TestLabel),0)
        self.y3TestLabel = np.concatenate((self.y51TestLabel,self.y52TestLabel),0)
        self.y1Test = np.concatenate((self.y21Test,self.y22Test),0)
        self.y2Test = np.concatenate((self.y41Test,self.y42Test),0)
        self.y3Test = np.concatenate((self.y51Test,self.y52Test),0)
    
    #-----------------------------------------------------------------------------#    
    def loadNankaiRireki(self):
        
        # nankai rireki path (slip velocity V)
        nankairirekiPath = os.path.join(self.features,self.nankairireki)
        
        with open(nankairirekiPath,"rb") as fp:
            evalX = pickle.load(fp)
            
        return evalX
        
    #-----------------------------------------------------------------------------#
    def nextBatch(self,batchSize):
        
        sInd = batchSize * self.batchCnt
        eInd = sInd + batchSize
       
        x1,x2,x3= self.x11Train[self.batchRandInd[sInd:eInd]],self.x13Train[self.batchRandInd[sInd:eInd]],self.x14Train[self.batchRandInd[sInd:eInd]]
        y11,y12,y13 = self.y21Train[self.batchRandInd[sInd:eInd]],self.y23Train[self.batchRandInd[sInd:eInd]],self.y24Train[self.batchRandInd[sInd:eInd]]
        y21,y22,y23 = self.y41Train[self.batchRandInd[sInd:eInd]],self.y43Train[self.batchRandInd[sInd:eInd]],self.y44Train[self.batchRandInd[sInd:eInd]]
        y31,y32,y33 = self.y51Train[self.batchRandInd[sInd:eInd]],self.y53Train[self.batchRandInd[sInd:eInd]],self.y54Train[self.batchRandInd[sInd:eInd]]
        yl11,yl12,yl13 = self.y21TrainLabel[self.batchRandInd[sInd:eInd]],self.y23TrainLabel[self.batchRandInd[sInd:eInd]],self.y24TrainLabel[self.batchRandInd[sInd:eInd]]
        yl21,yl22,yl23 = self.y41TrainLabel[self.batchRandInd[sInd:eInd]],self.y43TrainLabel[self.batchRandInd[sInd:eInd]],self.y44TrainLabel[self.batchRandInd[sInd:eInd]]
        yl31,yl32,yl33 = self.y51TrainLabel[self.batchRandInd[sInd:eInd]],self.y53TrainLabel[self.batchRandInd[sInd:eInd]],self.y54TrainLabel[self.batchRandInd[sInd:eInd]]
        
        batchX = np.concatenate((x1,x2,x3),0)
        batchY1,batchY1Label = np.concatenate((y11,y12,y13),0),np.concatenate((yl11,yl12,yl13),0)
        batchY2,batchY2Label = np.concatenate((y21,y22,y23),0),np.concatenate((yl21,yl22,yl23),0)
        batchY3,batchY3Label = np.concatenate((y31,y32,y33),0),np.concatenate((yl31,yl32,yl33),0)
        
        
        if eInd + batchSize > self.nTrain:
            self.batchCnt = 0
        else:
            self.batchCnt += 1
        pdb.set_trace()
        #return batchX,batchY,batchlabelY
        return batchX,batchY1,batchY1Label,batchY2,batchY2Label,batchY3,batchY3Label
    #-----------------------------------------------------------------------------#
