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
         
        #dataNames = ["b2b3b4b5b6_train{}{}".format(num,self.nClass) for num in np.arange(1,9)]
        #pdb.set_trace()
        
        
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
        """ 
        with open(os.path.join("features","rirekifeatures",f"{190}.pkl"),"rb") as fp:
            pred = pickle.load(fp) 
        
        pred = np.concatenate((pred[0][:,np.newaxis],pred[0][:,np.newaxis],pred[1][:,np.newaxis],pred[1][:,np.newaxis],pred[2][:,np.newaxis]),1)

        gt = self.x13Train[:,1:6,:]
        
        #pdb.set_trace() 
        minID = np.argsort(np.sum(np.reshape(abs(pred-np.transpose(gt,[0,2,1])),[-1,50]),1))[:15]

        #pdb.set_trace() 
        for num in np.arange(15):

            #pdb.set_trace() 
            fig, figInds = plt.subplots(nrows=5,sharex=True)
            for figInd in np.arange(len(figInds)):
                figInds[figInd].plot(self.x13Train[minID[num],figInd,:])
            plt.savefig(os.path.join("images","near",f"{np.round(self.y13Train[num],6)}_{np.round(self.y33Train[num],6)}_{np.round(self.y53Train[num],6)}.png"))
            plt.close()
        
        pdb.set_trace() 
        """
        self.xTest = np.reshape(self.xTest,[-1,self.nCell*self.nWindow]).astype(np.float32)
        # test y
        self.yTest = np.concatenate((self.y11Test[:,np.newaxis],self.y31Test[:,np.newaxis],self.y51Test[:,np.newaxis]),1)
        # test label y
        self.yTestLabel = np.concatenate((self.y11TestLabel[:,:,np.newaxis],self.y31TestLabel[:,:,np.newaxis],self.y51TestLabel[:,:,np.newaxis]),2)
        # number of train data
        self.nTrain =  int(self.x11Train.shape[0] + self.x12Train.shape[0] + self.x13Train.shape[0])
        # random train index
        self.batchRandInd = np.random.permutation(self.nTrain)
        
    #-----------------------------------------------------------------------------#    
    def loadNankaiRireki(self):
        
        # nankai rireki path (slip velocity V)
        # nankaifeatue.pkl -> 190.pkl
        flag = False
        for fID in np.arange(256):
            # pkl path
            #nankairirekiPath = os.path.join(self.features,self.nankaifeature,"{}_2.pkl".format(fID))
            nankairirekiPath = os.path.join(self.features,"rirekifeatures","{}.pkl".format(fID))
            
            with open(nankairirekiPath,"rb") as fp:
                data = pickle.load(fp)
            """
            fig, figInds = plt.subplots(nrows=3,sharex=True)
            for figInd in np.arange(len(figInds)):
                figInds[figInd].plot(data[figInd,:])
            plt.savefig(f"{fID}.png")
            plt.close()
            """ 
            x = np.reshape(np.concatenate([data[0][np.newaxis],data[0][np.newaxis],data[1][np.newaxis],data[1][np.newaxis],data[2][np.newaxis]]),[-1,])

            if not flag:
                evalX = x[np.newaxis]
                flag = True
            else:
                evalX = np.vstack([evalX,x[np.newaxis]])

        self.xEval = evalX
        #pdb.set_trace()
    #-----------------------------------------------------------------------------#
    def nextBatch(self,batchSize):
        
        sInd = batchSize * self.batchCnt
        eInd = sInd + batchSize
       
        # [number of data, cell(=5,nankai2 & tonakai2 & tokai1), dimention of features(=10)]
        trX = np.concatenate((self.x11Train[:,1:6,:],self.x12Train[:,1:6,:],self.x13Train[:,1:6,:]),0) 
        # mini-batch, [number of data, cell(=5)*dimention of features(=10)]
        batchX = np.reshape(trX[sInd:eInd],[-1,self.nCell*self.nWindow])
        # test all targets
        trY1 = np.concatenate((self.y11Train,self.y12Train,self.y13Train),0)
        trY2 = np.concatenate((self.y31Train,self.y32Train,self.y33Train),0)
        trY3 = np.concatenate((self.y51Train,self.y52Train,self.y53Train),0)
        # [number of data(mini-batch), cell(=3)] 
        batchY = np.concatenate((trY1[sInd:eInd,np.newaxis],trY2[sInd:eInd,np.newaxis],trY3[sInd:eInd,np.newaxis]),1)
        
        # train all labels, trlabel1 = nankai
        trlabel1 = np.concatenate((self.y11TrainLabel,self.y12TrainLabel,self.y13TrainLabel),0)
        trlabel2 = np.concatenate((self.y31TrainLabel,self.y32TrainLabel,self.y33TrainLabel),0)
        trlabel3 = np.concatenate((self.y51TrainLabel,self.y52TrainLabel,self.y53TrainLabel),0)
        # [number of data, number of class(self.nClass), cell(=3)] 
        batchlabelY = np.concatenate((trlabel1[sInd:eInd,:,np.newaxis],trlabel2[sInd:eInd,:,np.newaxis],trlabel3[sInd:eInd,:,np.newaxis]),2)
        #pdb.set_trace()
        if eInd + batchSize > self.nTrain:
            self.batchCnt = 0
        else:
            self.batchCnt += 1

        return batchX, batchY, batchlabelY
    #-----------------------------------------------------------------------------#


myData = NankaiData(nCell=5,nClass=10,nWindow=10)
myData.loadTrainTestData(nameInds=[0,1,2,3,4])
#b1,b2,b3 = myData.nextBatch(batchSize=1000)
