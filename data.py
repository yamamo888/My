
import os
import glob
import glob
import pickle

import pdb


dataPath = 'data'
pdeName = 'shrodinger'
pklPath = '*pkl'


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
# loading train & test dataset ----
def loadData(pdeName='burgers'):

    if pdeName == 'shrodinger':
        
        # x,t,uu dataset
        xtuuPath = os.path.join(dataPath,pdeName,'x*')
        with open(glob.glob(exactPath)[0],'rb') as fp:
            Exact_u = pickle.load(fp)
            Exact_v = pickle.load(fp)
            Exact_h = pickle.load(fp)
        
        # exact dataset for plot
        exactPath = os.path.join(dataPath,pdeName,'Exact*')
        with open(glob.glob(exactPath)[0],'rb') as fp:
            Exact_u = pickle.load(fp)
            Exact_v = pickle.load(fp)
            Exact_h = pickle.load(fp)


        # train X,Y test X,Y
        return [x0,tb], [u0,v0], [x_star], [u_star,v_star,h_star,Exact_h]
# ----

# mini-batch
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
