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
        
        # ----
        # data
        self.myData = datalstm.NankaiData()
        # Eval data, interval & year & max sequence
        self.xEval, self.yEval, self.seqEval = self.myData.IntervalEvalData()
        # ----
        
        # Placeholder ----
        # for LSTM
        self.x = tf.compat.v1.placeholder(tf.float32,shape=[None, None, self.dInput])
        self.y_nk = tf.compat.v1.placeholder(tf.float32,shape=[None, self.dOutput_nk])
        self.y_tnk = tf.compat.v1.placeholder(tf.float32,shape=[None, self.dOutput_tnk])
        self.y_tk = tf.compat.v1.placeholder(tf.float32,shape=[None, self.dOutput_tk])
        self.seq = tf.compat.v1.placeholder(tf.int32, shape=[None])
        # ----
        
        # Feature ----
        xlstm_eval = self.myData.LSTM(self.x, self.seq)
        # ----
        
        # Restore neural network ----
        hidden = self.pcRegress(xlstm_eval[1][-1], self.seq)
        # ----
        
        # optimizer ----
        config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1,allow_growth=True)) 
        self.saver = tf.compat.v1.train.Saver()
        self.sess = tf.compat.v1.Session(config=config)
        # ----
        
        # Restore model ----
        ckptpath = os.path.join(self.modelPath, 'pcNN_rnn')
        ckpt = tf.train.get_checkpoint_state(ckptpath)
        
        lastmodel = ckpt.model_checkpoint_path
        self.saver.restore(self.sess, lastmodel)
        print('>>> Restore pcNN model')
        
        # neural network ----
        self.ppred_nk, self.ppred_tnk, self.ppred_tk = self.cRegress(hidden)
        self.ppred_nk_test, self.ppred_tnk_test, self.ppred_tk_test = self.cRegress(hidden, reuse=True)
        # ----
        
        # loss ----
        self.closs = tf.reduce_mean(tf.square(self.y_nk - self.ppred_nk)) + \
                     tf.reduce_mean(tf.square(self.y_tnk - self.ppred_tnk)) + \
                     tf.reduce_mean(tf.square(self.y_tk - self.ppred_tk))  
        # ----
        
        Vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,scope='Regress') 
        self.opt = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.closs, var_list=Vars)
        
        self.sess.run(tf.compat.v1.variables_initializer(Vars))
        # ----
        
    def weight_variable(self,name,shape,trainable=True):
         return tf.compat.v1.get_variable(name,shape,initializer=tf.random_normal_initializer(stddev=0.1),trainable=trainable)
    # ----
    # ----
    def bias_variable(self,name,shape,trainable=True):
         return tf.compat.v1.get_variable(name,shape,initializer=tf.constant_initializer(0.1),trainable=trainable)
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
    def pcRegress(self, x, seq, rate=0.0, reuse=False, trainable=False):
        '''
        Restore model
        '''
        
        nHidden = 64
        
        with tf.compat.v1.variable_scope('Regress') as scope:
            if reuse:
                scope.reuse_variables()
            
            #h = self.myData.LSTM(x, seq, reuse=reuse)
            #h0 = h[1][-1]
            
            dInput = x.get_shape().as_list()[-1]
          
            # 1st layer
            w1_reg = self.weight_variable('w1_reg',[dInput, nHidden], trainable=trainable)
            bias1_reg = self.bias_variable('bias1_reg',[nHidden], trainable=trainable)
            h1 = self.fc(x,w1_reg,bias1_reg,rate)
            
            # 2nd layer
            w2_reg = self.weight_variable('w2_reg',[nHidden, nHidden], trainable=trainable)
            bias2_reg = self.bias_variable('bias2_reg',[nHidden], trainable=trainable)
            h2 = self.fc_relu(h1,w2_reg,bias2_reg,rate)
            
            # 3rd layer 
            w3_reg = self.weight_variable('w3_reg',[nHidden, nHidden], trainable=trainable)
            bias3_reg = self.bias_variable('bias3_reg',[nHidden], trainable=trainable)
            h3 = self.fc_relu(h2,w3_reg,bias3_reg,rate)
            
            return h3
    # ----
    
    # ----
    def cRegress(self, h, rate=0.0, reuse=False, trainable=True):
        
        nHidden = 64
        
        with tf.compat.v1.variable_scope('Regress') as scope:
            if reuse:
                scope.reuse_variables()
            
            # 4th layer
            w41_reg = self.weight_variable('w41_reg',[nHidden, self.dOutput_nk], trainable=trainable)
            bias41_reg = self.bias_variable('bias41_reg',[self.dOutput_nk], trainable=trainable)
            
            w42_reg = self.weight_variable('w42_reg',[nHidden, self.dOutput_tnk], trainable=trainable)
            bias42_reg = self.bias_variable('bias42_reg',[self.dOutput_tnk], trainable=trainable)
            
            w43_reg = self.weight_variable('w43_reg',[nHidden, self.dOutput_tk], trainable=trainable)
            bias43_reg = self.bias_variable('bias43_reg',[self.dOutput_tk], trainable=trainable)
            
            y_nk = self.fc(h,w41_reg,bias41_reg,rate)
            y_tnk = self.fc(h,w42_reg,bias42_reg,rate)
            y_tk = self.fc(h,w43_reg,bias43_reg,rate)
            
            return y_nk, y_tnk, y_tk
    # ----
                
    # ----
    def train(self, nItr=10000, nBatch=100):
        
        printPeriod = 100
        trL = np.zeros(int(nItr/printPeriod))
        
        for itr in np.arange(nItr):
            #pdb.set_trace()
            pfeed_dict = {self.x:self.xEval, self.y_nk:self.yEval[0][np.newaxis], self.y_tnk:self.yEval[1][np.newaxis], self.y_tk:self.yEval[2][np.newaxis], self.seq:self.seqEval}
            
            # paramB, loss
            trainCyclenk, trainCycletnk, trainCycletk, trainLoss = \
            self.sess.run([self.ppred_nk, self.ppred_tnk, self.ppred_tk, self.closs], pfeed_dict)
            
            if itr % printPeriod == 0:
                
                print('----')
                print('itr:%d, trainLoss:%3f' % (itr, trainLoss))
                print(f'nk cycle: {trainCyclenk[0]}')
                print(f'tnk cycle: {trainCycletnk[0]}')
                print(f'tk cycle: {trainCycletk[0]}')
                
                trL[int(itr/printPeriod)] = trainLoss
                
        # train & test loss
        losses = [trL]
        cycles = [trainCyclenk, trainCycletnk, trainCycletk]

        return losses, cycles
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
    # random sample loading train data
    nCell = 5
    dOutput = [9,9,7]
    rateTrain = 0.0
    lr = 1e-3
    # ----
          
    # model ----
    model = ParamCycleNN(dOutput, rateTrain=rateTrain, lr=lr, nCell=nCell, trialID=trialID)
    losses, cycles = model.train(nItr=nItr, nBatch=nBatch)
    # ----
    
    # plot ----
    myPlot = plot.Plot(figurepath=figurePath, trialID=trialID)
    myPlot.cLoss(losses, labels=['train'])
    # ----
    print('>>> Eval cycles')
    print(cycles[0])
    print(cycles[1])
    print(cycles[2])
    
    
