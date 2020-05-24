# -*- coding: utf-8 -*-

import sys
import os
import time

import numpy as np
import tensorflow as tf

import random
import pickle
import pdb

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt

import data
import paramNN
import cycle
import plot

class ParamCycleNN:
    def __init__(self, rateTrain=0.0, lr=1e-3, dInput=50, dOutput=3, trialID=0):
        
        # path ----
        self.modelPath = 'model'
        self.figurePath = 'figure'
        self.logPath = 'logs'
        self.paramPath = 'params'
        # ----
        
        # parameter ----
        self.dInput = dInput
        self.dOutput = dOutput
        self.trialID = trialID
        self.isCycle = True
        # ----
        
        # ----
        # data
        self.myData = data.NankaiData(nCell=nCell, nWindow=nWindow)
        # cycle
        self.cycle = cycle.Cycle(logpath=self.logPath, trialID=self.trialID)
        # Train & Test data for cycle
        self.xCycleTest, self.yCyclebTest, self.yCycleTest = self.myData.loadCycleTrainTestData()
        # Eval data
        self.xEval, self.yCycleEval = self.myData.loadNankaiRireki()
        self.xEval = self.xEval[np.newaxis]
        # ----
        
        # Placeholder ----
        self.x = tf.compat.v1.placeholder(tf.float32,shape=[None, self.dInput])
        self.y = tf.compat.v1.placeholder(tf.float32,shape=[None, self.dOutput])
        self.closs = tf.compat.v1.placeholder(tf.float32,shape=[None])
        # ----
        
        # neural network ----
        self.ppred = self.pcRegress(self.x, isCycle=self.isCycle)
        self.ppred_test = self.pcRegress(self.x, reuse=True, isCycle=self.isCycle)
        self.ppred_eval = self.pcRegress(self.x, reuse=True, isCycle=self.isCycle)
        # ----
        
        # loss ----
        self.ploss = tf.reduce_mean(tf.square(self.y - self.ppred))
        self.ploss_test = tf.reduce_mean(tf.square(self.y - self.ppred_test))
        self.ploss_eval = tf.reduce_mean(tf.square(self.y - self.ppred_eval))
        
        self.pcloss = self.ploss + tf.reduce_mean(self.closs)
        self.pcloss_test = self.ploss_test + tf.reduce_mean(self.closs)
        self.pcloss_eval = self.ploss_eval + tf.reduce_mean(self.closs)
        # ----
        
        # optimizer ----
        Vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,scope='Regress') 
        self.opt = tf.train.AdamOptimizer(lr).minimize(self.pcloss,var_list=Vars)
        print(f'Train values: {Vars}')
        
        config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1,allow_growth=True)) 
        #saver = tf.train.Saver()
        saver = tf.compat.v1.train.Saver()
        self.sess = tf.compat.v1.Session(config=config)
        # ----
        #pdb.set_trace()
        # restore model ----
        ckptpath = os.path.join(self.modelPath, 'pNN')
        ckpt = tf.train.get_checkpoint_state(ckptpath)
        
        if ckpt:
            lastmodel = ckpt.model_checkpoint_path
            saver.restore(self.sess, lastmodel)
            print('>>> Restore pNN model')
        else:
            self.sess.run(tf.global_variables_initializer())
        # ----
    
    # ----
    def pcRegress(self, x, rate=0.0, reuse=False, isCycle=True, trainable=True):
        
        nHidden=128
        
        #pdb.set_trace()
        # paramNN
        pNN = paramNN.ParamNN(isCycle=self.isCycle)
        h = pNN.Regress(self.x, reuse=reuse, isCycle=self.isCycle, trainable=False)
    
        with tf.compat.v1.variable_scope('Regress') as scope:
            if reuse:
                scope.reuse_variables()
                
            # 4th layer
            w4_reg = pNN.weight_variable('w4_reg',[nHidden, self.dOutput], trainable=trainable)
            bias4_reg = pNN.bias_variable('bias4_reg',[self.dOutput], trainable=trainable)
            
            y = pNN.fc(h,w4_reg,bias4_reg,rate)
            
            return y
    # ----
                
    # ----
    def train(self, nItr=10000, nBatch=100):
        
        testPeriod = 100
        trPL,trCL,trPCL = np.zeros(int(nItr/testPeriod)),np.zeros(int(nItr/testPeriod)),np.zeros(int(nItr/testPeriod))
        tePL,teCL,tePCL = np.zeros(int(nItr/testPeriod)),np.zeros(int(nItr/testPeriod)),np.zeros(int(nItr/testPeriod))
        
        for itr in np.arange(nItr):
            # 2: [nBatch,500,3]
            batchXY = self.myData.nextBatch(nBatch=nBatch, isCycle=self.isCycle)
            
            # 1. pred fric paramters
            pfeed_dict = {self.x:batchXY[0], self.y:batchXY[1]}
        
            # paramB, loss
            trainPPred, trainPLoss = self.sess.run([self.ppred, self.ploss], pfeed_dict)
            #pdb.set_trace()
            # 2. cycle loss, [nBatch]
            trainCLoss = self.cycle.loss(trainPPred, batchXY[2], itr=itr, dirpath='train')
            
            # 3. pred + cycle loss
            pcfeed_dict = {self.x:batchXY[0], self.y:batchXY[1], self.closs:trainCLoss}
            
            _, trainPCPred, trainPCLoss = self.sess.run([self.opt, self.ppred, self.pcloss], pcfeed_dict)
            
            #pdb.set_trace()
            if itr % testPeriod == 0:
                self.test(itr=itr)
                self.eval(itr=itr)
                
                print('itr:%d, trainPLoss:%3f, trainCLoss:%3f, trainPCLoss:%3f' % (itr, trainPLoss, np.mean(trainCLoss), trainPCLoss))
                print('itr:%d, testPLoss:%3f, testCLoss:%3f, testPCLoss:%3f' % (itr, self.testPLoss, np.mean(self.testCLoss), self.testPCLoss))
                
                
                
                trPL[int(itr/testPeriod)] = trainPLoss
                trCL[int(itr/testPeriod)] = np.mean(trainCLoss)
                trPCL[int(itr/testPeriod)] = trainPCLoss
                
                tePL[int(itr/testPeriod)] = self.testPLoss
                teCL[int(itr/testPeriod)] = np.mean(self.testCLoss)
                tePCL[int(itr/testPeriod)] = self.testPCLoss
                
        # train & test loss
        losses = [trPL,trCL,trPCL, tePL,teCL,tePCL]
        params = [self.testPPred,self.testPCPred,self.yCyclebTest, self.evalPPred,self.evalPCPred]
        
        return losses, params
    # ----
    
    # ----
    def test(self, itr=0):
        
        # 1. pred fric paramters
        feed_dict={self.x:self.xCycleTest, self.y:self.yCyclebTest}
        self.testPPred, self.testPLoss = self.sess.run([self.ppred_test, self.ploss_test], feed_dict)
        
        # 2. cycle loss
        self.testCLoss = self.cycle.loss(self.testPPred, self.yCycleTest, itr=itr, dirpath='test')
        
        # 3. pred + cycle loss
        pcfeed_dict = {self.x:self.xCycleTest, self.y:self.yCyclebTest, self.closs:self.testCLoss}
        self.testPCPred, self.testPCLoss = self.sess.run([self.ppred_test, self.pcloss_test], pcfeed_dict)
        
    # ----
    
    # ----
    def eval(self, itr=0):
       
        # 1. pred fric paramters
        feed_dict={self.x:self.xEval}
        self.evalPPred = self.sess.run(self.ppred_eval, feed_dict)
        
        # 2. cycle loss
        self.evalCLoss = self.cycle.loss(self.evalPPred, self.yCycleEval, itr=itr, dirpath='eval')
     
        # 3. pred + cycle loss
        pcfeed_dict = {self.x:self.xEval, self.closs:self.evalCLoss}
        self.evalPCPred = self.sess.run(self.ppred_eval, pcfeed_dict)
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
    rateTrain=0.0
    lr = 1e-3
    # ----
          
    # model ----
    model = ParamCycleNN(rateTrain=rateTrain, lr=lr, dInput=dInput, dOutput=dOutput, trialID=trialID)
    losses, params = model.train(nItr=nItr, nBatch=nBatch)
    # ----
    
    # Plot ----
    myPlot = plot.Plot(figurepath=figurePath, trialID=trialID)
    # loss
    myPlot.pcLoss(losses, labels=['trainP','trainC','trainPC','testP','testC','testPC'])
    # exact-pred scatter
    myPlot.epScatter(params, labels=['pNN','pcNN'])
    # ----
    
    # Re-Make pred rireki ----
    print('>>> Eval predB:', params[-1][0])
    
    
    minB,maxB = 0.011,0.0165
    if (minB <= params[-1][0][0]<= maxB) and (minB <= params[-1][0][1] <= maxB) and (minB <= params[-1][0][1] <= maxB):
        pdb.set_trace()
        
        np.savetxt(os.path.join(modelPath, 'params', 'eval', 'logs.csv'), params[-1]*1000000, delimiter=',', fmt='%5f')
                
        # Make logs ----
        lockPath = "Lock.txt"
        lock = str(1)
        with open(lockPath,"w") as fp:
            fp.write(lock)
        
        batFile = 'lastmakelogs.bat'
        os.system(batFile)
    
        sleepTime = 3
        while True:
            time.sleep(sleepTime)
            if os.path.exists(lockPath)==False:
                break
        '''
        # load logs
        loadBV(logpath)
        convV2YearlyData()
        
        gtCycles_nk = np.trim_zeros(gtCycles[ind,:,self.ntI])
        gtCycles_tnk = np.trim_zeros(gtCycles[ind,:,self.tntI])
        gtCycles_tk = np.trim_zeros(gtCycles[ind,:,self.ttI])            
        gt = [gtCycles_nk, gtCycles_tnk, gtCycles_tk]
        
        calcYearMSE(gt)            
        '''
    # ----
    
    
    
    
    
 