# -*- coding: utf-8 -*-

import sys
import argparse
import os

import numpy as np
import scipy.io
import tensorflow as tf

import random
import pickle
import pdb

import matplotlib.pylab as plt

import pdedata
import pdeplot


class ParamNN:
    def __init__(self, rateTrain=0.0, lr=1e-3, nBatch=100, trialID=0, dataMode='test'):
        
      
        # parameter ----
        self.xDim = 256
        tDim = 100
        self.nBatch = nBatch
        self.trialID = trialID
        self.yDim = 1
        # ----
        
        # Dataset ----
        self.myData = pdedata.pdeData(pdeMode='burgers', dataMode=dataMode)
        
        self.testX, self.testT, self.testU, self.testNU, self.varU, self.varNU  = self.myData.traintestvaridation()
        # ----        

        # Placeholder ----
        # output param b
        self.y = tf.compat.v1.placeholder(tf.float32,shape=[None, self.yDim])
        # input u
        self.inobs = tf.compat.v1.placeholder(tf.float32,shape=[None, self.xDim, tDim, 1])
        # ----
        
        # neural network ----
        self.predy, self.cnnfeature = self.lambdaNN(self.inobs)
        self.predy_test, self.cnnfeature_test = self.lambdaNN(self.inobs, reuse=True)
        self.predy_vard, self.cnnfeature_vard = self.lambdaNN(self.inobs, reuse=True)
        # ----

        # loss ----
        # param loss
        self.loss = tf.reduce_mean(tf.square(self.y - self.predy))
        self.loss_test = tf.reduce_mean(tf.square(self.y - self.predy_test))
        self.loss_vard = tf.reduce_mean(tf.square(self.y - self.predy_vard))
        # ----
        #pdb.set_trace()
        
        # Optimizer ----
        self.opt = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.loss)
        # ----
        
        # ----
        config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1,allow_growth=True)) 
        self.sess = tf.compat.v1.Session(config=config)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver()
        # ----
        
    # ----
    def weight_variable(self, name, shape):
         return tf.compat.v1.get_variable(name,shape,initializer=tf.random_normal_initializer(stddev=0.1))
    # ----
    # ----
    def bias_variable(self, name, shape):
         return tf.compat.v1.get_variable(name,shape,initializer=tf.constant_initializer(0.1))
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
    def lambdaNN(self, x, rate=0.05, reuse=False):
        
        nHidden = 128
        
        with tf.compat.v1.variable_scope('lambdaNN') as scope:  
            if reuse:
                scope.reuse_variables()
            
            xcnn = self.myData.CNNfeature(x, reuse=reuse)
            #pdb.set_trace()            
            dInput = xcnn.get_shape().as_list()[-1]
            
            # 1st layer
            w1 = self.weight_variable('w1',[dInput, nHidden])
            bias1 = self.bias_variable('bias1',[nHidden])
            #h1 = self.fc_relu(x,w1,bias1,rate)
            h1 = self.fc_relu(xcnn,w1,bias1,rate)
            
            # 2nd layer
            w2 = self.weight_variable('w2',[nHidden, nHidden])
            bias2 = self.bias_variable('bias2',[nHidden])
            h2 = self.fc_relu(h1,w2,bias2,rate)
            
            # 3nd layer
            w3 = self.weight_variable('w3',[nHidden, nHidden])
            bias3 = self.bias_variable('bias3',[nHidden])
            h3 = self.fc_relu(h2,w3,bias3,rate)
            
            # 4nd layer
            w4 = self.weight_variable('w4',[nHidden, self.yDim])
            bias4 = self.bias_variable('bias4',[self.yDim])
            
            # nu
            # static fc
            y = self.fc(h3,w4,bias4,rate)
            #pdb.set_trace() 
            #return y
            return y,xcnn
            
    # ----
    
    
    # ----
    def train(self, nItr=1000):
        
        # parameters ----
        testPeriod = 500
        batchCnt = 0
        #nTrain = 10
        nTrain = 3991
        batchRandInd = np.random.permutation(nTrain)
        # ----
        
        # Start training
        trPL,tePL,varPL = [],[],[]
        
        for itr in range(nItr):
            
            # index
            sInd = self.nBatch * batchCnt
            eInd = sInd + self.nBatch
            index = batchRandInd[sInd:eInd]
            # Get train data
            # [x.shape], [100,], [nbatch, t.shape, x.shape]
            _, _, batchU, batchNU = self.myData.miniBatch(index)
            #pdb.set_trace()
            # y: prameter b
            feed_dict = {self.y:batchNU[:,np.newaxis], self.inobs:batchU}
            
            _, trainParam, trainLoss =\
            self.sess.run([self.opt, self.predy, self.loss], feed_dict)
            
            if eInd + self.nBatch > nTrain:
                batchCnt = 0
                batchRandInd = np.random.permutation(nTrain)
            else:
                batchCnt += 1
            
            # Test & Varidation
            if itr % testPeriod == 0:

                self.test(itr=itr)
                self.varidation(itr=itr)
                
                print('----')
                print('itr: %d, trainLoss:%f' % (itr, trainLoss))
                print(f'train exact: {batchNU[:5]}')
                print(f'train pred: {trainParam[:5]}')
                
                # param loss
                trPL = np.append(trPL,trainLoss)
                tePL = np.append(tePL,self.testLoss)
                varPL = np.append(varPL,self.varLoss)

                # Save model
                #self.saver.save(self.sess, os.path.join('model', f'{dataMode}burgers', f'first_{dataMode}'), global_step=itr)
        
        paramloss = [trPL,tePL,varPL]
        teparams = [self.testX, self.testT, self.testParam]
        varparams = [self.testX, self.testT, self.varParam]
        
        return paramloss, teparams, varparams
    # ----
    
    # ----
    def test(self,itr=0):
        #pdb.set_trace() 
        feed_dict={self.y: self.testNU, self.inobs:self.testU}    
        
        self.testParam, self.testLoss =\
        self.sess.run([self.predy_test, self.loss_test], feed_dict)

        print('itr: %d, testPLoss:%f' % (itr, self.testLoss))
        print(f'test exact: {self.testNU[:5]}')
        print(f'test pred: {self.testParam[:5]}')
    # ----
    
    # ----
    def varidation(self,itr=0):
        
        feed_dict={self.y: self.varNU, self.inobs:self.varU}    
        
        self.varParam, self.varLoss =\
        self.sess.run([self.predy_test, self.loss_test], feed_dict)

        print('itr: %d, varPLoss:%f' % (itr, self.varLoss))
        print(f'varidation exact: {self.varNU}')
        print(f'varidation pred: {self.varParam}')
    # ----
    
    
    
if __name__ == "__main__":
    
    # command argment ----
    parser = argparse.ArgumentParser()

    # iteration of training
    parser.add_argument('--nItr', type=int, default=100)
    # Num of mini-batch
    parser.add_argument('--nBatch', type=int, default=100)
    # datamode (pkl)
    parser.add_argument('--dataMode', required=True, choices=['large', 'middle', 'small'])
    # trial ID
    parser.add_argument('--trialID', type=int, default=0)    
    
    # 引数展開
    args = parser.parse_args()
    
    nItr = args.nItr
    nBatch = args.nBatch
    trialID = args.trialID
    dataMode = args.dataMode
    # ----
    
    # path ----
    modelPath = "model"
    figurePath = "figure"
    # ----
    
    # parameters ----
    # Learning rate
    rateTrain = 0.0
    lr = 1e-3
    # ----
    
    # Training ----
    model = ParamNN(rateTrain=rateTrain, lr=lr, nBatch=nBatch, trialID=trialID, dataMode=dataMode)
    plosses, teparams, varparams = model.train(nItr=nItr)
    # ----
    
    # Plot ----
    myPlot = pdeplot.Plot(dataMode=dataMode, trialID=trialID)
    myPlot.Loss(plosses, labels=['train','test','varid'], savename='param')
    myPlot.paramToU(teparams, savename='tepredparam')
    myPlot.paramToU(varparams, savename='varpredparam')
    
    # ----
 
    
