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
import plot

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
        
        # Train & Test data for cycle
        self.xCycleTest, self.yCyclebTest, self.yCycleTest = self.myData.loadCycleTrainTestData()
        # Eval data
        self.xEval, self.yCycleEval = self.myData.loadNankaiRireki()
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
        
        flag = False
        trPL = np.zeros(int(nItr/testPeriod))
        for itr in np.arange(nItr):
            
            batchXY = self.myData.nextBatch(nBatch=nBatch, isCycle=True)
            
            # 1. pred fric paramters
            pfeed_dict = {self.x:batchXY[0], self.y:self.batchXY[1]}
            trainPPred, trainPLoss = self.sess.run([self.ppred, self.ploss], pfeed_dict)
            
            # 2. cycle loss
            trainCLoss, trainCycle = self.cycle.loss(trainPPred, batchXY[2], itr=itr)
            
            # 3. pred + cycle loss
            pcfeed_dict = {self.x:batchXY[0], self.y:batchXY[1], self.closs:trainCLoss}
            _, trainPCPred, trainPCLoss = self.sess.run([self.opt, self.pppred, self.pcloss], pcfeed_dict)
            
            
            if itr % testPeriod == 0:
                Test = self.test()
                Eval = self.eval()
                
                print('itr:%d, trainPLoss:%3f, trainCLoss:%3f, trainPCLoss:%3f' % (itr, trainPLoss, trainCLoss, trainPCLoss))
                print('itr:%d, testPLoss:%3f, testCLoss:%3f, testPCLoss:%3f' % (itr, Test.testPLoss, Test.testCLoss, Test.testPCLoss))
                
                pdb.set_trace()
                trPL[int(itr/testPeriod)] = trainPLoss
                
        # train & test loss
        losses = [trPL]
        params = [trainPPred,trainPCPred, Test.testPPred,Test.testPCPred, Eval.evalPPred,Eval.evalPCPred]
        cycles = [trainCycle, Test.testCycle, Eval.evalCycle]

        return losses, params, cycles
    # ----
    
    # ----
    def test(self):
        
        # 1. pred fric paramters
        feed_dict={self.x:self.xTest, self.y:self.yCyclebTest}
        self.testPPred, self.testPLoss = self.sess.run([self.ppred_test, self.loss_test], feed_dict)
        
        # 2. cycle loss
        testCLoss, testCycle = self.cycle.loss(self.testPPred, self.yCycleTest)
        
        # 3. pred + cycle loss
        pcfeed_dict = {self.x:self.xTest, self.y:self.yCyclebTest, self.closs:testCLoss}
        self.testPCLoss = self.sess.run([self.ppred_test, self.pcloss_test], pcfeed_dict)
        
    # ----
    
    # ----
    def eval(self):
       
        # 1. pred fric paramters
        feed_dict={self.x:self.xEval}
        self.evalPPred = self.sess.run(self.ppred_eval, feed_dict)
        
        # 2. cycle loss
        evalCLoss, evalCycle = self.cycle.loss(self.evalPPred, self.yCycleEval)
     
        # 3. pred + cycle loss
        pcfeed_dict = {self.x:self.xEval, self.closs:evalCLoss}
        self.evalPCPred = self.sess.run(self.ppred_eval, pcfeed_dict)
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
    
    # path ----
    modelPath = "model"
    figurePath = "figure"
    # ----
   
    # parameters ----
    # select nankai data(3/5) 
    nametrInds = [0,1,2,3,4,5,6,7]
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
    losses, params, cycles = model.train(nItr=nItr, nBatch=nBatch)
    # ----
    
    # plot ----
    myPlot = plot.Plot(figurePath=figurePath, trialID=trialID)
    myPlot.Loss(losses)
    # ----
    
    # Re-Make pred rireki ----
    predloss, predcycle = model.cycle.loss(params, model.yCycleEval)
    
    
    # ----
    
    
    
    
    
 