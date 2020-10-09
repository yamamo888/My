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
import pdeburgers
import pdeplot


class ParamNN:
    def __init__(self, rateTrain=0.0, lr=1e-3, nBatch=100, trialID=0, dataMode='test'):
        
      
        # parameter ----
        if dataMode == 'large':
            self.xDim = 50
        elif dataMode == 'middle':
            self.xDim = 25
        elif dataMode == 'small':
            self.xDim = 10
        
        #self.xDim = 256 
        self.tDim = 100
        self.nBatch = nBatch
        self.trialID = trialID
        self.yDim = 1
        # ----
        
        # for Plot ----
        self.myPlot = pdeplot.Plot(dataMode=dataMode, trialID=trialID)
        # ----

        # Dataset ----
        self.myData = pdedata.pdeData(pdeMode='burgers', dataMode=dataMode)
        # [xDim,1], [100,1], [data, xDim, 100], [data,] 
        self.alltestX, self.testX, self.testT, self.testU, self.testNU, self.idx = self.myData.traintest()
        # ----
         
        # Placeholder ----
        # u
        self.inobs = tf.compat.v1.placeholder(tf.float32,shape=[None, self.xDim, self.tDim, 1])
        self.outobs = tf.compat.v1.placeholder(tf.float64,shape=[None, self.xDim, self.tDim])
        # param nu 
        self.y = tf.compat.v1.placeholder(tf.float32,shape=[None, self.yDim])
        # ----
        
        # Restore neural network ----
        # pred nu [ndata,]
        hidden = self.RestorelambdaNN(self.inobs)
        # ----
        
        # optimizer ----
        config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1,allow_growth=True)) 
        self.saver = tf.compat.v1.train.Saver()
        self.sess = tf.compat.v1.Session(config=config)
        # ----
        
        # Restore model ----
        ckptpath = os.path.join('model', f'{dataMode}burgers')
        ckpt = tf.train.get_checkpoint_state(ckptpath)
        
        lastmodel = ckpt.model_checkpoint_path
        self.saver.restore(self.sess, lastmodel)
        print('>>> Restore model')
        # ----
        
        # neural network (nu) ----
        self.predparam, self.predparam_ = self.lambdaNN(hidden, rate=rateTrain)
        # ----
        predparam = tf.cast(tf.reshape(self.predparam, [-1,]), tf.float64)
        
        # PDE ----
        # output: u
        self.predu = self.pde(self.x, self.t, self.param_test, nData=self.testNU.shape[0])
        #self.predu = pdeburgers.burgers(predparam)
        # ----
        
        #pdb.set_trace()
        # space data
        self.indx = tf.compat.v1.placeholder(tf.int32,shape=[None,1])
        self.predu = tf.gather_nd(self.predu, self.indx)
        #self.predu = tf.expand_dims(self.predu, 0) 
        

        # loss param ----
        self.loss_nu = tf.reduce_mean(tf.square(self.y - self.predparam)) 
        # ----
        
        # loss u ----   
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.square(self.outobs - self.predu),2),1))
        # ----
        
        # gradient ----
        self.gradnu = tf.gradients(self.loss_nu, self.inobs)
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
    # 0.01 < x < 5.0 (for moving burgers)
    def outputthes(self, x):
        
        overlambda = tf.constant([0.304])
        overid = tf.where(x>overlambda) # ※　勾配死んでる
        overnum = tf.shape(overid)[0]
        overths = tf.tile(overlambda, [overnum])
        overupdate = tf.tensor_scatter_nd_update(x, overid, overths)
        
        updatex = self.underupdate(overupdate)
        
        return updatex
    # ----
    # ----
    def underupdate(self, xover):
        
        underlambda = tf.constant([0.005])
        underid = tf.where(xover<underlambda)
        undernum = tf.shape(underid)[0]
        underths = tf.tile(underlambda, [undernum])
        underupdate = tf.tensor_scatter_nd_update(xover, underid, underths)
        
        return underupdate
    # ----

    # ----
    def RestorelambdaNN(self, x, reuse=False, trainable=False):
        
        nHidden1 = 8
        nHidden2 = 8
        nHidden3 = 16
        nHidden4 = 16
        nHidden5 = 32
        
        with tf.compat.v1.variable_scope("pre-training-lambdaNN") as scope:
            if reuse:
                scope.reuse_variables()
            #pdb.set_trace() 
            # 1st conv layer
            w1 = self.weight_variable('w1', [3,3,1,nHidden1], trainable=trainable)
            b1 = self.bias_variable('b1', [nHidden1], trainable=trainable)
            conv1 = self.conv2d(x, w1, b1, strides=1)
        
            conv1 = tf.nn.dropout(conv1, rate=0.4)
        
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
            w6 = self.weight_variable('w6', [nHidden5,self.yDim], trainable=trainable)
            b6 = self.bias_variable('b6', [self.yDim], trainable=trainable)
           
            y = self.fc_relu(fc1,w6,b6)
            
            # 0.005 < y < 0.304
            y_ = self.outputthes(y)
            
            return y, y_
    # ----
     
    # ----
    def train(self, nItr=1000):
        
        # parameters ----
        printPeriod = 10
        pdePeriod = 100
        batchCnt = 0
        nTrain = 65
        batchRandInd = np.random.permutation(nTrain)
        # ※手動
        plotnus = [0.005, 0.01, 0.02, 0.05, 0.1, 0.3]
        # ----
        
        # Start training
        teUL = []
        tePL = []
        
        for itr in range(nItr):

            # ※1こずつ？
            flag = False
            cnt = 0
            for ind in batchRandInd:
                #pdb.set_trace()
                feed_dict={self.y:self.testNU[ind,np.newaxis], self.inobs:self.testU[ind,np.newaxis], self.outobs:self.testU[ind,np.newaxis,:,:,0], self.indx:self.idx[:,np.newaxis]} 
          
                if itr <= pdePeriod:
                    _, testParam, testploss, testgrad =\
                    self.sess.run([self.opt, self.predparam, self.loss_nu, self.gradnu], feed_dict)
                
                elif itr > pdePeriod:
                    _, testParam, testploss, testuloss, testGrad =\
                    self.sess.run([self.opt, self.predparam, self.predu, self.loss_nu, self.loss, self.gradnu], feed_dict)
                

                """
                if cnt < printPeriod:
                    
                    if not flag:
                        testPLoss = testploss
                        testULoss = testuloss
                        flag = True
                    else:
                        testPLoss = np.hstack([testPLoss, testploss])
                        testULoss = np.hstack([testULoss, testuloss])
                    
                    cnt += 1
                    
                if itr % printPeriod == 0:
                
                    print('itr: %d, testULoss:%f, testPLoss:%f' % (itr, self.testULoss, self.testPLoss))
                    print('grad %f' % (np.mean(testGrad)))
                    print(testParam)
                    print(self.testNU)

                    tePL = np.append(tePL, testPLoss)
                    teUL = np.append(teUL, testULoss)
                    
                    if np.round(self.testNU[ind],3) in plotnus:
                        
                        exactnu = self.testNU[ind]
                        prednu = testPred[ind]
                        
                        self.myPlot.plotExactPredParam([self.alltestX, self.testt, prednu, exactnu, self.idx], itr=itr, savename='tepredparamode')
        
        paramloss = [tePL]
        ulosses = [teUL]
        
        return paramloss, ulosses"""
        #pdb.set_trace()
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
    #plosses, ulosses = model.train(nItr=nItr)
    model.train(nItr=nItr)
    # ----
    
    # Plot ----
    #model.myPlot.Loss1(plosses, labels=['test'], savename='poNN_param')
    #model.myPlot.Loss1(ulosses, labels=['test'], savename='poNN_u')
    
    #model.myPlot.plotExactPredParam(teparams, xNum=teparams[0].shape[0], savename='lasttepredparamode')
    # ----
 
    
