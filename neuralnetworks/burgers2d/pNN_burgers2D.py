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

import burgers2Ddata
import burgers2Dplot


class ParamNN:
    def __init__(self, rateTrain=0.0, lr=1e-3, nBatch=100, trialID=0, dataMode='test'):
        
      
        # parameter ----
        if dataMode == 'large':
            self.xDim = 25
            self.yDim = 25
        elif dataMode == 'middle':
            self.xDim = 12
            self.yDim = 12
        elif dataMode == 'small':
            self.xDim = 5
            self.yDim = 5
            
        self.tDim = 201
        self.nBatch = nBatch
        self.trialID = trialID
        self.lambdaDim = 2
        # ----
        
        # for plot
        self.myPlot = burgers2Dplot.Plot(dataMode=dataMode, trialID=trialID)
    
        # Dataset ----
        self.myData = burgers2Ddata.Data(pdeMode='burgers2d', dataMode=dataMode)
        # all x,y[51,1] t[201,1]  & x,y,t[xdim or ydim,1] u,v[data,xdim,ydim,tdim] nu[data,]
        self.alltestX, self.alltestY, self.testT, self.testX, self.testY, self.testU, self.testV, self.testNU, self.idx, self.idy = self.myData.traintest()
        #pdb.set_trace()
        # ----        

        # Placeholder ----
        # input u, v (one time) [xdim, ydim, t1]
        self.inobsu = tf.compat.v1.placeholder(tf.float32,shape=[None, self.xDim, self.yDim, 1])
        self.inobsv = tf.compat.v1.placeholder(tf.float32,shape=[None, self.xDim, self.yDim, 1])
        self.inobs = tf.concat([self.inobsu, self.inobsv], -1)
        # output param b
        self.y = tf.compat.v1.placeholder(tf.float32,shape=[None, self.lambdaDim])
        # ----
        
        # neural network ----
        self.predy = self.lambdaNN(self.inobs, rate=rateTrain)
        self.predy_test = self.lambdaNN(self.inobs, reuse=True)
        # ----
        
        # loss ----
        # param loss
        self.loss1 = tf.reduce_mean(tf.square(self.y[:,0] - self.predy[:,0]))
        self.loss2 = tf.reduce_mean(tf.square(self.y[:,1] - self.predy[:,1]))
        
        self.loss1_test = tf.reduce_mean(tf.square(self.y[:,0] - self.predy_test[:,0]))
        self.loss2_test = tf.reduce_mean(tf.square(self.y[:,1] - self.predy_test[:,1]))
        # ----
        
        self.loss = self.loss1 + self.loss2

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
    def weight_variable(self, name, shape, trainable=True):
         return tf.compat.v1.get_variable(name,shape,initializer=tf.random_normal_initializer(stddev=0.1),trainable=trainable)
    # ----
    # ----
    def bias_variable(self, name, shape, trainable=True):
         return tf.compat.v1.get_variable(name,shape,initializer=tf.constant_initializer(0.1),trainable=trainable)
    # ----
    # ----
    def conv2d(self, x, W, b, strides=1):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)
    # ----
    # ----
    def maxpool2d(self, x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')
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
    def lambdaNN(self, x, rate=0.0, reuse=False, trainable=True):
        
        nHidden1 = 8
        nHidden2 = 8
        nHidden3 = 16
        nHidden4 = 16
        nHidden5 = 32
        
        with tf.compat.v1.variable_scope("pre-training-lambdaNN") as scope:
            if reuse:
                scope.reuse_variables()
            #pdb.set_trace() 

            indim = x.get_shape().as_list()[-1]

            # 1st conv layer
            w1 = self.weight_variable('w1', [3,3,indim,nHidden1], trainable=trainable)
            b1 = self.bias_variable('b1', [nHidden1], trainable=trainable)
            conv1 = self.conv2d(x, w1, b1, strides=1)
        
            # 2nd conv layer
            w2 = self.weight_variable('w2', [3,3,nHidden1,nHidden2], trainable=trainable)
            b2 = self.bias_variable('b2', [nHidden2], trainable=trainable)
            conv2 = self.conv2d(conv1, w2, b2, strides=1)
        
            conv2 = self.maxpool2d(conv2)
            
            # 3nd conv layer
            w3 = self.weight_variable('w3', [3,3,nHidden2,nHidden3], trainable=trainable)
            b3 = self.bias_variable('b3', [nHidden3], trainable=trainable)
            conv3 = self.conv2d(conv2, w3, b3, strides=1)
            
            # 4nd conv layer
            w4 = self.weight_variable('w4', [3,3,nHidden3,nHidden4], trainable=trainable)
            b4 = self.bias_variable('b4', [nHidden4], trainable=trainable)
            conv4 = self.conv2d(conv3, w4, b4, strides=1)
        
            conv4 = self.maxpool2d(conv4)
         
            w5 = self.weight_variable('w5', [conv4.get_shape().as_list()[1]*conv4.get_shape().as_list()[2]*conv4.get_shape().as_list()[3], 
                                             nHidden5], trainable=trainable)
            b5 = self.bias_variable('b5', [nHidden5], trainable=trainable)
            
            # 1st full-layer
            reshape_conv4 = tf.reshape(conv4, [-1, w5.get_shape().as_list()[0]])
            
            fc1 = self.fc_relu(reshape_conv4,w5,b5)
            
            # 2nd full-layer
            w6 = self.weight_variable('w6', [nHidden5,self.lambdaDim], trainable=trainable)
            b6 = self.bias_variable('b6', [self.lambdaDim], trainable=trainable)
           
            y = self.fc(fc1,w6,b6)
            
            return y 
    # ----
    
    # ----
    def train(self, nItr=1000):
        
        # parameters ----
        testPeriod = 5
        savemodelPeriod = 100
        batchCnt = 0
        nTrain = 876
        batchRandInd = np.random.permutation(nTrain)
        # for random t index -> select 1/201
        tRandInd = np.random.permutation(self.tDim)
        lambda1 = np.array([1.0])
        # ----
        
        # Start training
        trPL1,trPL2,tePL1,tePL2 = [],[],[],[]
        flag = False
        
        for epoch in range(nItr):
            
            trainLosses1,trainLosses2 = 0,0
            # index
            sInd = self.nBatch * batchCnt
            eInd = sInd + self.nBatch
            index = batchRandInd[sInd:eInd]
            
            # Get train data       
            batchU, batchV, batchNU = self.myData.miniBatch(index)
            batchLambda1 = np.repeat(lambda1, batchNU.shape[0])
            
            # [batch,2]
            batchParam = np.vstack([batchLambda1, batchNU]).T
           
            for itr in range(self.tDim):
                
                feed_dict = {self.y:batchParam, self.inobsu:batchU[:,:,:,tRandInd[itr],None], self.inobsv:batchV[:,:,:,tRandInd[itr],None]}
                
                _, trainParam, trainLoss1, trainLoss2 =\
                self.sess.run([self.opt, self.predy, self.loss1, self.loss2], feed_dict)
                
                # loss^n_t1 + loss^n_t2 + ...
                trainLosses1 += trainLoss1
                trainLosses2 += trainLoss2
                
            if eInd + self.nBatch > nTrain:
                batchCnt = 0
                batchRandInd = np.random.permutation(nTrain)
            else:
                batchCnt += 1
            
            # Test & Varidation
            if itr % testPeriod == 0:

                self.test(epoch=epoch, tRandInd=tRandInd)
                
                trainLosses1 = trainLosses1 / self.tDim
                trainLosses2 = trainLosses2 / self.tDim
                
                # param loss
                trPL1 = np.append(trPL1,trainLosses1)
                trPL2 = np.append(trPL2,trainLosses2)
                
                tePL1 = np.append(tePL1,self.testLosses1)
                tePL1 = np.append(tePL2,self.testLosses2)
                
                print('----')
                print('epoch: %d, trainLoss1:%f, trainLoss2:%f' % (epoch, trainLosses1, trainLosses2))
                
                
                '''
                if not flag:
                    trP = trainParam
                    teP = self.testParam
                    flag = True
                else:
                    trP = np.hstack([trP, trainParam])
                    teP = np.hstack([teP, self.testParam])
                '''
            if itr % savemodelPeriod == 0:
                # Save model
                self.saver.save(self.sess, os.path.join('model', f'{dataMode}burgers2d', f'first_{dataMode}'), global_step=itr)
    
        
        paramlosses1 = [trPL1,tePL1]
        paramlosses2 = [trPL2,tePL2]
        
        return paramlosses1, paramlosses2
    # ----
    
    # ----
    # all t
    def test(self, epoch=0, tRandInd=0):
        
        testLambda1 = np.repeat(np.array([1.0]), self.testNU.shape[0])
        testParam = np.vstack([testLambda1, self.testNU]).T
        pdb.set_trace()
        
        testLosses1,testLosses2 = 0,0
        
        for itr in range(self.tDim):
        
            feed_dict={self.y:testParam, self.inobsu:self.testU[:,:,:,tRandInd[itr],None], self.inobsv:self.testV[:,:,:,tRandInd[itr],None]}    
             
            testParam, testLoss1, testLoss2 =\
            self.sess.run([self.predy_test, self.loss1_test, self.loss2_test], feed_dict)
            
            testLosses1 += testLoss1
            testLosses2 += testLoss2
        
        self.testLosses1 = testLosses1 / self.tDim
        self.testLosses2 = testLosses2 / self.tDim
        
        print('epoch: %d, testLoss1:%f, testLoss2:%f' % (epoch, self.testLossse1, self.testLosses2))
    # ----
     
if __name__ == "__main__":
    
    # command argment ----
    parser = argparse.ArgumentParser()

    # iteration of training
    parser.add_argument('--nItr', type=int, default=100)
    # Num of mini-batch
    parser.add_argument('--nBatch', type=int, default=25)
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
    plosses1, plosses2 = model.train(nItr=nItr)
    # ----
    
    # Plot ----
    model.myPlot.Loss2Data(plosses1, labels=['train','test'], savename='pNN_param1')
    model.myPlot.Loss2Data(plosses2, labels=['train','test'], savename='pNN_param2')
    # ----
