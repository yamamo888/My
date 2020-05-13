
import os
import glob
import pickle

import numpy as np

import pdb


class PDEData:
    def __init__(self,pdeName='burgers'):
        
        self.dataPath = 'data'
        self.pklPath = '*pkl'
        self.pdeName = pdeName
        
    # Loading train & test dataset ----
    def loadData(self):
    
        if self.pdeName == 'shrodinger':
            
            # x,t,uu dataset
            xtuuPath = os.path.join(self.dataPath,self.pdeName,'x*')
            with open(glob.glob(xtuuPath)[0],'rb') as fp:
                x = pickle.load(fp)
                t = pickle.load(fp)
            
            # exact dataset
            exactPath = os.path.join(self.dataPath,self.pdeName,'Exact*')
            with open(glob.glob(exactPath)[0],'rb') as fp:
                Exact_u = pickle.load(fp)
                Exact_v = pickle.load(fp)
                Exact_h = pickle.load(fp)
            
            #labelPath = os.path.join(self.dataPath,self.pdeName,'label*')
            #with open(glob.glob(labelPath)[0],'rb') as fp:
                #label_h = pickle.load(fp)
            
            # test index
            teInd = [75,100,125]
            
            trxInd = np.ones(x.shape[0], dtype=bool)
            trtInd = np.ones(t.shape[0], dtype=bool)
            
            trxInd[teInd] = False
            trtInd[teInd] = False
            
            # train data x,t,h(x,t)
            self.xtrain = x[trxInd]
            self.ttrain = t[trtInd]
            self.htrain = Exact_h[trxInd,trtInd]
            #self.hlabeltrain = label_h
            
            # test data
            self.xtest = x[teInd]
            self.ttest = t[teInd]
            self.htest = Exact_h[:,teInd]
            pdb.set_trace()
            # train X,Y test X,Y
            return [self.xtrain,self.ttrain], [self.htrain], [self.xtest,self.ttest], [self.htest]
    # ----
    
    # Make label data ----
    def annotateY(self,num_cls=10):
        
        ycls = np.arange(np.min(self.htrain)-1, np.max(self.htrain)+1, num_cls)
        
        flag = False
        for data in self.htest:
            pdb.set_trace()
            oneHot = np.zeros(len(num_cls))
            
            for num in np.arange(num_cls+1):
                if (data >= ycls[num]) & (data < ycls[num+1]):
                    oneHot[num] = 1
                
            if not flag:
                label = oneHot[np.newaxis]
                flag = True
            else:
                label = np.vstack([label,oneHot])
            
            print(oneHot == 1)
            
        return label
    # ----
    
    # mini-batch ----
    def nextBatch(self,batchSize=100):
        
        nxTrain = self.xtrain.shape[0]
        ntTrain = self.ttrain.shape[0]
        
        # random index for x & t
        xInd = np.random.choice(nxTrain,batchSize)
        tInd = np.random.choice(ntTrain,batchSize)
            
        xbatch = self.xtrain[xInd]
        tbatch = self.ttrain[tInd]
        # [x,t]
        batchX = np.concatenate([xbatch[:,np.newaxis],tbatch[:,np.newaxis]],1)
        batchY = self.htrain[xInd,tInd]
        batchlabelY = self.hlabeltrain[xInd,tInd]
        
        return batchX, batchY, batchlabelY
    # ----
