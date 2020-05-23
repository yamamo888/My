# -*- coding: utf-8 -*-

import sys
import os

import numpy as np
import tensorflow as tf

import random
import pickle
import pdb

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt

#import data
import datalstm
import cycle
import plot

class ParamCycleNN:
    def __init__(self, rateTrain=0.0, lr=1e-3, dInput=50, trialID=0, dOutput):
        
        # path ----
        self.modelPath = 'model'
        self.figurePath = 'figure'
        self.logPath = 'logs'
        self.paramPath = 'params'
        # ----
        
        # parameter ----
        self.dInput = dInput
        self.dOutput_nk = dOutput[0]
        self.dOutput_tnk = dOutput[1]
        self.dOutput_tk = dOutput[2]
        
        self.trialID = trialID
        self.isCycle = True
        # ----
        
        # ----
        # data
        #self.myData = data.NankaiData(nCell=nCell, nWindow=nWindow)
        self.myData = lstmdata.NankaiData()
        
        # Eval data for cycle
        #self.xCycleTest, self.yCyclenkTest, self.yCycletnkTest, self.yCycletkTest = self.myData.loadCycleTestData()
        # Eval data
        #self.xEval, self.yCycleEval = self.myData.loadNankaiRireki()
        self.xEval, self.yEval, self.seqEval = self.myData.EvalData()
        #self.yCyclenkEval, self.yCycletnkEval, self.yCycletkEval = self.yCycleEval[:,0], self.yCycleEval[:,1], self.yCycleEval[:,2]
        #self.xEval = self.xEval[np.newaxis]
        # ----
        pdb.set_trace()
        # Placeholder ----
        self.x = tf.compat.v1.placeholder(tf.float32,shape=[None, self.dInput])
        self.y_nk = tf.compat.v1.placeholder(tf.float32,shape=[None, self.dOutout_nk])
        self.y_tnk = tf.compat.v1.placeholder(tf.float32,shape=[None, self.dOutout_tnk])
        self.y_tk = tf.compat.v1.placeholder(tf.float32,shape=[None, self.dOutout_tk])
        self.seq = tf.compat.v1.placeholder(tf.int32, shape=[None])
        # ----
        
        # neural network ----
        self.ppred_nk, self.ppred_tnk, self.ppred_tk = self.cRegress(self.x, self.seq, isCycle=self.isCycle)
        self.ppred_nk_test, self.ppred_tnk_test, self.ppred_tk_test = self.cRegress(self.x, self.seq, reuse=True, isCycle=self.isCycle)
        # ----
        
        # loss ----
        self.closs = tf.reduce_mean(tf.square(self.y_nk - self.ppred_nk)) + \
                     tf.reduce_mean(tf.square(self.y_tnk - self.ppred_tnk)) + \
                     tf.reduce_mean(tf.square(self.y_tk - self.ppred_tk))
        
        self.closs_test = tf.reduce_mean(tf.square(self.y_nk_test - self.ppred_nk_test)) + \
                     tf.reduce_mean(tf.square(self.y_tnk_test - self.ppred_tnk_test)) + \
                     tf.reduce_mean(tf.square(self.y_tk_test - self.ppred_tk_test))
        # ----
        
        # optimizer ----
        Vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,scope='Regress') 
        self.opt = tf.train.AdamOptimizer(lr).minimize(self.closs, var_list=Vars)
        print(f'Train values: {Vars}')
        
        config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1,allow_growth=True)) 
        saver = tf.compat.v1.train.Saver()
        self.sess = tf.compat.v1.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        # ----
        #pdb.set_trace()
        '''
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
        '''
    # ----
    #def cRegress(self, x, rate=0.0, reuse=False, isCycle=True, trainable=True):
    def cRegress(self, x, seq, rate=0.0, reuse=False, isCycle=True, trainable=True):
        
        nHidden=128
        
        with tf.compat.v1.variable_scope('Regress') as scope:
            if reuse:
                scope.reuse_variables()
            
            h0 = self.myData.LSTM(x, seq, reuse=reuse)
            
            # 1st layer
            w1_reg = self.weight_variable('w1_reg',[self.dInput, nHidden], trainable=trainable)
            bias1_reg = self.bias_variable('bias1_reg',[nHidden], trainable=trainable)
            h1 = self.fc_relu(h0,w1_reg,bias1_reg,rate)
            
            # 2nd layer
            w2_reg = self.weight_variable('w2_reg',[nHidden, nHidden], trainable=trainable)
            bias2_reg = self.bias_variable('bias2_reg',[nHidden], trainable=trainable)
            h2 = self.fc_relu(h1,w2_reg,bias2_reg,rate)
            
            # 3rd layer 
            w3_reg = self.weight_variable('w3_reg',[nHidden, nHidden], trainable=trainable)
            bias3_reg = self.bias_variable('bias3_reg',[nHidden], trainable=trainable)
            h3 = self.fc_relu(h2,w3_reg,bias3_reg,rate)
          
            # 4th layer
            w41_reg = pNN.weight_variable('w41_reg',[nHidden, self.dOutput_nk], trainable=trainable)
            bias41_reg = pNN.bias_variable('bias41_reg',[self.dOutput_nk], trainable=trainable)
            
            w42_reg = pNN.weight_variable('w42_reg',[nHidden, self.dOutput_tnk], trainable=trainable)
            bias42_reg = pNN.bias_variable('bias42_reg',[self.dOutput_tnk], trainable=trainable)
            
            w43_reg = pNN.weight_variable('w43_reg',[nHidden, self.dOutput_tk], trainable=trainable)
            bias43_reg = pNN.bias_variable('bias43_reg',[self.dOutput_tk], trainable=trainable)
            
            y_nk = pNN.fc(h,w41_reg,bias41_reg,rate)
            y_tnk = pNN.fc(h,w42_reg,bias42_reg,rate)
            y_tk = pNN.fc(h,w43_reg,bias43_reg,rate)
            
            return y_nk, y_tnk, y_tk
    # ----
                
    # ----
    def train(self, nItr=10000, nBatch=100):
        
        testPeriod = 100
        trL,teL = np.zeros(int(nItr/testPeriod)),np.zeros(int(nItr/testPeriod))
        
        for itr in np.arange(nItr):
            
            batchX, batchY, batchSeq = self.myData.nextBatch(nBatch=nBatch, isCycle=self.isCycle)
          
            
            #pfeed_dict = {self.x:self.xCycleTest, self.y_nk:self.yCyclenkTest, self.y_nk:self.yCycletnkTest, self.y_nk:self.yCycletkTest}
            pfeed_dict = {self.x:self.batchX, self.y_nk:self.batchY[0], self.y_nk:batchY[1], self.y_nk:batchY[2], self.seq:batchSeq}
        
            # paramB, loss
            trainCyclenk, trainCycletnk, trainCycletk, trainLoss = \ 
            self.sess.run([self.ppred_nk, self.ppred_tnk, self.ppred_tk, self.ploss], pfeed_dict)
            
            
            if itr % testPeriod == 0:
                self.test()
                
                print('itr:%d, trainLoss:%3f, testPLoss:%3f' % (itr, trainLoss, self.testLoss)
                
                trL[int(itr/testPeriod)] = trainPCLoss
                
                teL[int(itr/testPeriod)] = self.testLoss
                
        # train & test loss
        losses = [trL,teL]
        cycles = [trainCyclenk, trainCycletnk, trainCycletk, self.testCycle, self.evalCycle]

        return losses, cycles
    # ----
    
    # ----
    def test(self, itr=0):
        
        feed_dict={self.x:self.xEval, self.seq:self.seqEval}
        
        self.testCyclenk, self.testCycletnk, self.testCycletk, self.trainLoss = \ 
        self.sess.run([self.ppred_nk_test, self.ppred_tnk_test, self.ppred_tk_test, self.ploss_test], pfeed_dict)
            
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
    model = ParamCycleNN(rateTrain=rateTrain, lr=lr, dInput=dInput, trialID=trialID,  dOutput=[8,8,6])
    losses, params = model.train(nItr=nItr, nBatch=nBatch)
    # ----
    pdb.set_trace()
    # plot ----
    myPlot = plot.Plot(figurePath=figurePath, trialID=trialID)
    myPlot.Loss(losses)
    # ----
    
    