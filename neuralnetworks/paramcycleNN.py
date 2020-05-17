# -*- coding: utf-8 -*-

import sys
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v1

import random
import pickle
import pdb

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt

import paramNN
import cycle

class ParamCycleNN:
    def __init__(self, keepProbTrain=1.0, lr=1e-3, dInput=50, dOutput=3, trialID=0):
        
        # path ----
        self.modelPath = 'model'
        self.figurePath = 'figure'
        self.logsPath = 'logs'
        self.paramsPath = 'params'
        # ----
        
        # parameter ----
        self.dInput = dInput
        self.dOutput = dOutput
        self.trialID = trialID
        # ----
  
        # ----
        # paramNN
        self.paramNN = paramNN.ParamNN(keepProbTrain=keepProbTrain, lr=lr, dInput=self.dInput, dOutput=self.dOutput)
        # cycle
        self.cycle = cycle.Cycle(logpath=self.logsPath, parampath=self.paramsPath, trialID=self.trialID)
        # Eval data
        self.myData.loadNankaiRireki()
        # ----
        
        # Placeholder ----
        self.x = tf.placeholder(tf.float32,shape=[None, self.dInput])
        self.y = tf.placeholder(tf.float32,shape=[None, self.dOutput])
        self.closs = tf.placeholder(tf.float32,shape=[None])
        # ----
        
        # neural network ----
        self.ppred = self.paramNN.Regress(self.x)
        self.ppred_test = self.paramNN.Regress(self.x, reuse=True)
        self.ppred_eval = self.paramNN.Regress(self.x, reuse=True)
        # ----
        
        # loss ----
        self.ploss = tf.reduce_mean(tf.square(self.y - self.ppred))
        self.ploss_test = tf.reduce_mean(tf.square(self.y - self.ppred_test))
        self.ploss_eval = tf.reduce_mean(tf.square(self.y - self.ppred_eval))
        
        self.pcloss = self.ploss + self.closs
        self.pcloss_test = self.pLoss_test + self.closs
        self.pcloss_eval = self.pLoss_eval + self.closs
        # ----
        
        # optimizer ----
        Vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Regress') 
        self.opt = tf.train.AdamOptimizer(lr).minimize(self.pcloss,var_list=Vars)
        
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1,allow_growth=True)) 
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        # ----
        
        
    # ----
    def train(self, nItr=10000, nBatch=100):
        
        testPeriod = 100
        
        for itr in np.arange(nItr):
            
            self.myData.nextBatch(nBatch)
            
            # pred fric paramters
            pfeed_dict = {self.x:self.batchX, self.y:self.batchY}
            trainPPred, trainPLoss = self.sess.run([self.ppred, self.ploss], pfeed_dict)
            
            # cycle loss
            trainCLoss = self.cycle.loss(trainPPred, self.batchCycleY, itr=itr)
            
            pcfeed_dict = {self.x:self.batchX, self.y:self.batchY, self.closs:trainCLoss}
            trainPCLoss = self.sess.run(self.pcloss, pcfeed_dict)
            
            
            if itr % testPeriod == 0:
                Test = self.test()
                Eval = self.eval()
                
                testPCLoss = Test.testPCLoss
                
                print('tr:%d, trainPLoss:%f, trainPVar:%f' % (i, trainPLoss, trainPVar))
                print('trainCLoss:%f, trainCVar:%f' % (trainCLoss, trainCVar))
                print('trainPCLoss:%f, trainPCVar:%f' % (trainPCLoss, trainPCVar))
           
    # ----
    
    # ----
    def test(self):
        feed_dict={self.x:self.xTest, self.y:self.yTest}
        testPPred, testPLoss = sess.run([self.ppred_test, self.loss_test], feed_dict)
        
        testCLoss, testCPred = self.cycle.loss(testPPred,self.yCycleTest)
     
        pcfeed_dict = {self.x:self.xTest, self.y:self.yTest, self.closs:testCLoss}
        self.testPCLoss = self.sess.run([self.pcloss_test], pcfeed_dict)
        
    # ----
    
    # ----
    def eval(self):
        
        feed_dict={self.x:self.xEval}
        self.evalPred = self.sess.run([self.ppred_eval], feed_dict)
    # ----
    
    # ----
    def restore(self):
        # restore saved model, latest
        savedPath = f'{trialID}'
        savedfullPath = os.path.join(modelPath,savedPath)
        if os.path.exists(savedfullPath):
            saver = tf.train.Saver()
            saver.restore(sess,tf.train.latest_checkpoint(savedfullPath))
            print('---- Success Restore! ----')
    # ----
    
    # ----
    def saver(self):
        saver.save(self.sess, os.path.join(self.modelPath,'paramcycleNN'))
    # ----
         
    
if __name__ == "__main__":
    
    # command argment ----
    # batch size
    nBatch = int(sys.argv[1])
    # iteration of training
    nItr = int(sys.argv[2])
    # trial ID
    trialID = int(sys.argv[3])
    # ----
        
    # parameters ----
    # select nankai data(3/5) 
    nametrInds = [0,1,2,3,4,5,6]
    # random sample loading train data
    nameInds = random.sample(nametrInds,3)
    nCell = 5
    nWindow = 10
    dInput = nCell*nWindow
    dOutput = 3
    keepProbTrain=1.0
    lr = 1e-3
   
    # ----
          
    # model ----
    model = ParamCycleNN(keepProbTrain=keepProbTrain, lr=lr, dInput=dInput, dOutput=dOutput, trialID=trialID)
    model.train(nItr=nItr)
    # ----
    
    
    
    
 