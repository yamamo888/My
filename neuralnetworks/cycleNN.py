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
    def __init__(self, dOutput, rateTrain=0.0, lr=1e-3, nCell=5, trialID=0):
        
        # path ----
        self.modelPath = 'model'
        self.figurePath = 'figure'
        self.logPath = 'logs'
        self.paramPath = 'params'
        # ----
        
        # parameter ----
        self.dInput = 3
        self.dOutput_nk = dOutput[0]
        self.dOutput_tnk = dOutput[1]
        self.dOutput_tk = dOutput[2]
        
        self.trialID = trialID
        # ----
        
        # Dataset ----
        self.myData = datalstm.NankaiData()
        # Train & Test data for cycle
        self.xCycleTest, self.yCyclebTest, self.yCycleTest, self.yCycleseqTest = self.myData.loadIntervalTrainTestData()
        # Eval data
        self.xEval, self.yCycleEval, self.yCycleseqEval = self.myData.IntervalEvalData()
        # ----
        
        # Module ----
        # cycle
        self.cycle = cycle.Cycle(logpath=self.logPath, trialID=self.trialID)
        # ----
     
        # Placeholder ----
        self.x = tf.compat.v1.placeholder(tf.float32,shape=[None, None, self.dInput])
        #self.y_nk = tf.compat.v1.placeholder(tf.float32,shape=[None, self.dOutput_nk])
        #self.y_tnk = tf.compat.v1.placeholder(tf.float32,shape=[None, self.dOutput_tnk])
        #self.y_tk = tf.compat.v1.placeholder(tf.float32,shape=[None, self.dOutput_tk])
        self.y_nk = tf.compat.v1.placeholder(tf.float32,shape=[None, None])
        self.y_tnk = tf.compat.v1.placeholder(tf.float32,shape=[None, None])
        self.y_tk = tf.compat.v1.placeholder(tf.float32,shape=[None, None])
        
        self.seq = tf.compat.v1.placeholder(tf.int32, shape=[None])
        # ----
        
        # neural network ----
        xlstm = self.myData.LSTM(self.x, self.seq)
        xlstm_test = self.myData.LSTM(self.x, self.seq, reuse=True)
        xlstm_eval = self.myData.LSTM(self.x, self.seq, reuse=True)
     
        self.ppred_nk, self.ppred_tnk, self.ppred_tk = self.cRegress(xlstm[1][0], rate=rateTrain)
        self.ppred_nk_test, self.ppred_tnk_test, self.ppred_tk_test = self.cRegress(xlstm_test[1][0], reuse=True)
        self.ppred_nk_eval, self.ppred_tnk_eval, self.ppred_tk_eval = self.cRegress(xlstm_eval[1][0], reuse=True)
        # ----
        
        # loss ----
        self.closs = tf.reduce_mean(tf.square(self.y_nk - self.ppred_nk)) + \
                     tf.reduce_mean(tf.square(self.y_tnk - self.ppred_tnk)) + \
                     tf.reduce_mean(tf.square(self.y_tk - self.ppred_tk))
        
        self.closs_test = tf.reduce_mean(tf.square(self.y_nk - self.ppred_nk_test)) + \
                     tf.reduce_mean(tf.square(self.y_tnk - self.ppred_tnk_test)) + \
                     tf.reduce_mean(tf.square(self.y_tk - self.ppred_tk_test))
        # ----
        
        # optimizer ----
        Vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,scope='Regress') 
        self.opt = tf.train.AdamOptimizer(lr).minimize(self.closs, var_list=Vars)
        print(f'Train values: {Vars}')
        
        config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1,allow_growth=True)) 
        self.sess = tf.compat.v1.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        # ----
         
    # ----
    def weight_variable(self, name, shape, trainable=False):
         return tf.compat.v1.get_variable(name,shape,initializer=tf.random_normal_initializer(stddev=0.1), trainable=trainable)
    # ----
    # ----
    def bias_variable(self, name, shape, trainable=False):
         return tf.compat.v1.get_variable(name,shape,initializer=tf.constant_initializer(0.1), trainable=trainable)
    # ----
    # ----
    def fc_relu(self,inputs,w,b,rate=0.0):
         relu = tf.matmul(inputs,w) + b
         relu = tf.nn.dropout(relu, rate=rate)
         relu = tf.nn.relu(relu)
         return relu
    # ----
    # ----
    def fc(self,inputs,w,b,rate=0.0):
         fc = tf.matmul(inputs,w) + b
         fc = tf.nn.dropout(fc, rate=rate)
         return fc
    # ----
    
    # ----
    def cRegress(self, x, rate=0.0, reuse=False, trainable=True):
        
        nHidden=164
        
        with tf.compat.v1.variable_scope('Regress') as scope:
            if reuse:
                scope.reuse_variables()
            
            dInput = x.get_shape().as_list()[-1]
           
            # 1st layer
            w1_reg = self.weight_variable('w1_reg',[dInput, nHidden], trainable=trainable)
            bias1_reg = self.bias_variable('bias1_reg',[nHidden], trainable=trainable)
            h1 = self.fc_relu(x,w1_reg,bias1_reg,rate)
            
            # 2nd layer
            w2_reg = self.weight_variable('w2_reg',[nHidden, nHidden], trainable=trainable)
            bias2_reg = self.bias_variable('bias2_reg',[nHidden], trainable=trainable)
            h2 = self.fc_relu(h1,w2_reg,bias2_reg,rate)
            
            # 3rd layer 
            w3_reg = self.weight_variable('w3_reg',[nHidden, nHidden], trainable=trainable)
            bias3_reg = self.bias_variable('bias3_reg',[nHidden], trainable=trainable)
            h3 = self.fc_relu(h2,w3_reg,bias3_reg,rate)
          
            # 4th layer
            w41_reg = self.weight_variable('w41_reg',[nHidden, self.dOutput_nk], trainable=trainable)
            bias41_reg = self.bias_variable('bias41_reg',[self.dOutput_nk], trainable=trainable)
            
            w42_reg = self.weight_variable('w42_reg',[nHidden, self.dOutput_tnk], trainable=trainable)
            bias42_reg = self.bias_variable('bias42_reg',[self.dOutput_tnk], trainable=trainable)
            
            w43_reg = self.weight_variable('w43_reg',[nHidden, self.dOutput_tk], trainable=trainable)
            bias43_reg = self.bias_variable('bias43_reg',[self.dOutput_tk], trainable=trainable)
            
            y_nk = self.fc(h3,w41_reg,bias41_reg,rate)
            y_tnk = self.fc(h3,w42_reg,bias42_reg,rate)
            y_tk = self.fc(h3,w43_reg,bias43_reg,rate)
            
            return y_nk, y_tnk, y_tk
    # ----
                
    # ----
    def train(self, nItr=10000, nBatch=100):
        
        testPeriod = 100
        trL,teL = np.zeros(int(nItr/testPeriod)),np.zeros(int(nItr/testPeriod))
        
        for itr in np.arange(nItr):
            
            batchXY = self.myData.nextBatch(nBatch=nBatch)
            pdb.set_trace()
            batchY1 = batchXY[2][:,:,0]
            batchY2 = batchXY[2][:,:,1]
            batchY3 = batchXY[2][:,:,2]
            
            pfeed_dict = {self.x:batchXY[0], self.y_nk:batchY1, self.y_tnk:batchY2, self.y_tk:batchY3, self.seq:batchXY[3]}
            # paramB, loss
            trainCyclenk, trainCycletnk, trainCycletk, trainLoss = \
            self.sess.run([self.ppred_nk, self.ppred_tnk, self.ppred_tk, self.closs], pfeed_dict)
            
            if itr % testPeriod == 0:
                self.test(itr=itr)
                self.eval(itr=itr)
                
                print('itr:%d, trainLoss:%3f, testPLoss:%3f' % (itr, trainLoss, self.testLoss))
                
                trL[int(itr/testPeriod)] = trainLoss
                teL[int(itr/testPeriod)] = self.testLoss
                
        # train & test loss
        losses = [trL,teL]
        cycles = [trainCyclenk, trainCycletnk, trainCycletk, self.testCycle]

        return losses, cycles
    # ----
    
    # ----
    def test(self, itr=0):
        
        yCycleTest_nk = self.yCycleTest[:,:,0]
        yCycleTest_tnk = self.yCycleTest[:,:,1]
        yCycleTest_tk = self.yCycleTest[:,:,2]
        
        pfeed_dict={self.x:self.xCycleTest, self.y_nk:yCycleTest_nk, self.y_tnk:yCycleTest_tnk, self.y_tk:yCycleTest_tk, self.seq:self.yCycleseqTest}    
       
        self.testCyclenk, self.testCycletnk, self.testCycletk, self.testLoss = \
        self.sess.run([self.ppred_nk_test, self.ppred_tnk_test, self.ppred_tk_test, self.closs_test], pfeed_dict)
            
    # ----
    
    # ----
    def eval(self, itr=0):
        # 1. pred fric paramters
        pdb.set_trace()
        pfeed_dict={self.x:self.xEval, self.seq:self.yCycleseqEval}
    
        self.testCyclenk, self.testCycletnk, self.testCycletk = \
        self.sess.run([self.ppred_nk_eval, self.ppred_tnk_eval, self.ppred_tk_eval], pfeed_dict)
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
    nCell = 5
    rateTrain=0.0
    lr = 1e-3
    # ----
          
    # model ----
    model = ParamCycleNN(dOutput=[8,8,6], rateTrain=rateTrain, lr=lr, nCell=nCell, trialID=trialID)
    losses, params = model.train(nItr=nItr, nBatch=nBatch)
    # ----
    pdb.set_trace()
    # plot ----
    myPlot = plot.Plot(figurePath=figurePath, trialID=trialID)
    myPlot.Loss(losses)
    # ----
    
    