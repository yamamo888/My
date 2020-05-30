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

import datalstm
import cycle
import plot

class ParamCycleNN:
    def __init__(self, rateTrain=0.0, lr=1e-3, nCell=5, trialID=0):
        
        # path ----
        self.modelPath = 'model'
        self.figurePath = 'figure'
        self.logPath = 'logs'
        self.paramPath = 'params'
        # ----
        
        # parameter ----
        self.dInput = nCell
        self.dOutput = 3
        self.dOutput_nk = 8
        self.dOutput_tnk = 8
        self.dOutput_tk = 6
        self.trialID = trialID
        isReModel = True
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
        self.seq = tf.compat.v1.placeholder(tf.int32, shape=[None])
        self.y = tf.compat.v1.placeholder(tf.float32,shape=[None, self.dOutput])
        self.closs = tf.compat.v1.placeholder(tf.float32,shape=[None])
        # ----
        
        # neural network ----
        xlstm = self.myData.LSTM(self.x, self.seq)
        xlstm_test = self.myData.LSTM(self.x, self.seq, reuse=True)
        xlstm_eval = self.myData.LSTM(self.x, self.seq, reuse=True)
    
        self.ppred = self.pcRegress(xlstm[1][-1], rate=rateTrain)
        self.ppred_test = self.pcRegress(xlstm_test[1][0], reuse=True)
        self.ppred_eval = self.pcRegress(xlstm_eval[1][0], reuse=True)
        # ----
        
        # loss ----
        self.ploss = tf.square(self.y - self.ppred)
        self.ploss_test = tf.square(self.y - self.ppred_test)
        
        self.pcloss = tf.reduce_mean(self.ploss) +  0.0001 * tf.reduce_mean(self.closs)
        self.pcloss_test = tf.reduce_mean(self.ploss_test) + 0.0001 * tf.reduce_mean(self.closs)  
        # ----
      
        # optimizer ----
        Vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,scope='Regress') 
        self.opt = tf.train.AdamOptimizer(lr).minimize(self.pcloss,var_list=Vars)
       
        config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1,allow_growth=True)) 
        self.sess = tf.compat.v1.Session(config=config)
        # ----
        
        if isReModel:
            saver = tf.compat.v1.train.Saver()
            ckptpath = os.path.join(self.modelPath, 'pNN_rnn')
            ckpt = tf.train.get_checkpoint_state(ckptpath)
        
            lastmodel = ckpt.model_checkpoint_path
            saver.restore(self.sess, lastmodel)
            print('>>> Restore pcNN model')
        else:
            self.sess.run(tf.global_variables_initializer())
            # save model ----
            saver = tf.compat.v1.train.Saver()
            saver.save(self.sess, os.path.join('model', 'pcNN_rnn', 'first'))
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
    def pcRegress(self, x, rate=0.0, reuse=False, isPre=False, trainable=False):
        
        nHidden=64
        
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
            
            if isPre:
                return h3
            else:
                
                # 4th layer
                w4_reg = self.weight_variable('w4_reg',[nHidden, self.dOutput], trainable=True)
                bias4_reg = self.bias_variable('bias4_reg',[self.dOutput], trainable=True)
                
                y = self.fc(h3,w4_reg,bias4_reg,rate)
                
                return y
    # ----
                
    # ----
    def train(self, nItr=10000, nBatch=100):
        
        testPeriod = 100
       
        trPL,trCL,trPCL = np.zeros(int(nItr/testPeriod)),np.zeros(int(nItr/testPeriod)),np.zeros(int(nItr/testPeriod))
        tePL,teCL,tePCL = np.zeros(int(nItr/testPeriod)),np.zeros(int(nItr/testPeriod)),np.zeros(int(nItr/testPeriod))
        
        for itr in np.arange(nItr):
            
            batchXY = self.myData.nextBatch(nBatch=nBatch)
            
            pfeed_dict = {self.x:batchXY[0], self.y:batchXY[1], self.seq:batchXY[3]}
            
            # 1. pred fric paramters ----
            # paramB, loss
            trainPPred, trainPLoss = self.sess.run([self.ppred, self.ploss], pfeed_dict)
            
            # 2. cycle loss, [nBatch] ----
            trainCLoss = self.cycle.loss(trainPPred, batchXY[1], batchXY[2], itr=itr, dirpath='train')
            
            
            pcfeed_dict = {self.x:batchXY[0], self.y:batchXY[1], self.seq:batchXY[3], self.closs:trainCLoss}
            
            # 3. pred + cycle loss ----
            _, trainPCPred, trainPCLoss  = \
            self.sess.run([self.opt, self.ppred, self.pcloss], pcfeed_dict)
           
          
            if itr % testPeriod == 0:
                self.test(itr=itr)
                self.eval(itr=itr)
                
                print('itr:%d, trainPLoss:%3f, trainCLoss:%3f, trainPCLoss:%3f' % (itr, np.mean(trainPLoss), np.mean(trainCLoss), trainPCLoss))
                print('itr:%d, testPLoss:%3f, testCLoss:%3f, testPCLoss:%3f' % (itr,  np.mean(self.testPLoss), np.mean(self.testCLoss), self.testPCLoss))
                
                trPL[int(itr/testPeriod)] = np.mean(trainPLoss)
                trCL[int(itr/testPeriod)] = np.mean(trainCLoss)
                trPCL[int(itr/testPeriod)] = trainPCLoss
                
                tePL[int(itr/testPeriod)] = np.mean(self.testPLoss)
                teCL[int(itr/testPeriod)] = np.mean(self.testCLoss)
                tePCL[int(itr/testPeriod)] = self.testPCLoss
                
        # train & test loss
        losses = [trPL,trCL,trPCL, tePL,teCL,tePCL]
        params = [self.testPPred,self.testPCPred,self.yCyclebTest, self.evalPPred,self.evalPCPred]
        
        return losses, params
    # ----
    
    # ----
    def test(self, itr=0):
        
        feed_dict={self.x:self.xCycleTest, self.y:self.yCyclebTest, self.seq:self.yCycleseqTest}    
           
        # 1. pred fric paramters ----
        self.testPPred, self.testPLoss = self.sess.run([self.ppred_test, self.ploss_test], feed_dict)
        
        # 2. cycle loss ----
        self.testCLoss = self.cycle.loss(self.testPPred, self.yCyclebTest, self.yCycleTest, itr=itr, dirpath='test')
        
        # 3. pred + cycle loss ----
        pcfeed_dict = {self.x:self.xCycleTest, self.y:self.yCyclebTest, self.seq:self.yCycleseqTest, self.closs:self.testCLoss}
      
        self.testPCPred, self.testPCLoss = \
        self.sess.run([self.ppred_test, self.pcloss_test], pcfeed_dict)
     
        print(f'mean: {np.mean(self.testCLoss)}')
        print(f'max: {np.max(self.testCLoss)}')
        print(f'min: {np.min(self.testCLoss)}')
        print('----')
        print('>>> exact')
        print(self.yCyclebTest[:5,:])
        print('>>> pcNN')
        print(self.testPCPred[:5,:])
        
    # ----
    
    # ----
    def eval(self, itr=0):
        # 1. pred fric paramters
        feed_dict={self.x:self.xEval, self.seq:self.yCycleseqEval}
        self.evalPPred = self.sess.run(self.ppred_eval, feed_dict)
        
        # 2. cycle loss
        self.evalCLoss = self.cycle.evalloss(self.evalPPred, self.yCycleEval, itr=itr, dirpath='eval')
        
        print(f'Eval paramB: {self.evalPPred[0]}')
        print(f'Eval CLoss:%3f' % (self.evalCLoss))
        
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
    rateTrain = 0.2
    lr = 1e-3
    # ----
          
    # model ----
    model = ParamCycleNN(rateTrain=rateTrain, lr=lr, nCell=nCell, trialID=trialID)
    losses, params = model.train(nItr=nItr, nBatch=nBatch)
    # ----
    
    Plabels = ['trainP','trainC','trainPC','testP','testC','testPC']
    
    # Plot ----
    myPlot = plot.Plot(figurepath=figurePath, trialID=trialID)
    # loss
    myPlot.pcLoss(losses, labels=Plabels)
    # exact-pred scatter
    myPlot.epScatter(params, labels=['pNN','pcNN'])
    # ----
    
    # Re-Make pred rireki ----
    print('>>> Eval predB:', params[-1][0])
    
    # save lastparam    
    for param,label in zip(params,Plabels):
        np.savetxt(os.path.join('model', 'params', 'eval', f'{trialID}', f'{label}.csv'), param, delimiter=',', fmt='%5f')
                
      
    
    
    
    
 