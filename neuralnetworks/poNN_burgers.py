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
        self.alltestX, self.testx, self.testt, self.testU, self.testNU, self.varx, self.varU, self.varNU  = self.myData.traintestvaridation()
         
        # [testdata, 256] -> [testdata, xdim]
        self.testX = np.reshape(np.tile(self.testx, self.tDim), [-1, self.xDim])
        self.testT = np.reshape(np.tile(self.testt, self.xDim), [-1, self.tDim])
        # ----
         
        # Placeholder ----
        # u
        self.inobs = tf.compat.v1.placeholder(tf.float32,shape=[None, self.xDim, self.tDim, 1])
        self.outobs = tf.compat.v1.placeholder(tf.float32,shape=[None, self.tDim, self.xDim])
        # x,t
        self.x = tf.compat.v1.placeholder(tf.float32,shape=[self.tDim, self.xDim])
        self.t = tf.compat.v1.placeholder(tf.float32,shape=[self.xDim, self.tDim])
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
        self.predparam, self.predparam_pre = self.lambdaNN(hidden)
        # ----
        
        # PDE ----
        # output: u
        self.predu = self.pde(self.x, self.t, self.param_test, nData=self.testNU.shape[0])
        # ----
        
        # loss param ----
        self.loss_nu = tf.reduce_mean(tf.square(self.y - self.param)) 
        # ----

        # loss u ----   
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.square(self.outobs - self.predu),2),1))
        # ----
        
        # gradient
        self.gradnu = tf.gradients(self.loss_nu, self.inobs)[0]

        # Optimizer ----
        #self.opt_nu = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.loss_nu)
        
        lambdaVars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,scope='updatelambdaNN') 
        self.opt = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.loss, var_list=lambdaVars)
        # ----
        
        # ----
        uninitialized = self.sess.run([tf.compat.v1.is_variable_initialized(var) for var in tf.compat.v1.global_variables()])
        uninitializedVars =[v for (v, f) in zip(tf.compat.v1.global_variables(), uninitialized) if not f]
        self.sess.run(tf.compat.v1.variables_initializer(uninitializedVars))
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
    # 0.01 < x < 5.0 (for moving burgers)
    def outputthes(self, x):
        
        overlambda = tf.constant([5.0])
        overid = tf.where(x>overlambda)
        overnum = tf.shape(overid)[0]
        overths = tf.tile(overlambda, [overnum])
        overupdate = tf.tensor_scatter_nd_update(x, overid, overths)
        
        updatex = self.underupdate(overupdate)
        
        return updatex
    # ----
    # ----
    def underupdate(self, xover):
        
        underlambda = tf.constant([0.01])
        underid = tf.where(xover<underlambda)
        undernum = tf.shape(underid)[0]
        underths = tf.tile(underlambda, [undernum])
        underupdate = tf.tensor_scatter_nd_update(xover, underid, underths)
        
        return underupdate
    # ----
    
    # ----
    def RestorelambdaNN(self, x, rate=0.0, reuse=False, trainable=False):
        
        nHidden = 128
        
        with tf.compat.v1.variable_scope('lambdaNN') as scope:  
            if reuse:
                scope.reuse_variables()
            
            # CNN feature
            xcnn = self.myData.CNNfeature(x, reuse=reuse, trainable=trainable)
            
            dInput = xcnn.get_shape().as_list()[-1]
            
            # 1st layer
            w1 = self.weight_variable('w1',[dInput, nHidden], trainable=trainable)
            bias1 = self.bias_variable('bias1',[nHidden], trainable=trainable)
            h1 = self.fc_relu(xcnn,w1,bias1,rate)
            
            # 2nd layer
            w2 = self.weight_variable('w2',[nHidden, nHidden], trainable=trainable)
            bias2 = self.bias_variable('bias2',[nHidden], trainable=trainable)
            h2 = self.fc_relu(h1,w2,bias2,rate)
            
            # 3nd layer
            w3 = self.weight_variable('w3',[nHidden, nHidden], trainable=trainable)
            bias3 = self.bias_variable('bias3',[nHidden], trainable=trainable)
            h3 = self.fc_relu(h2,w3,bias3,rate)
            
            return h3
    # ----
    
    # ----
    def lambdaNN(self, x, rate=0.0, reuse=False, trainable=True):
        
        nHidden = 128
        dOutput = 1
        
        with tf.compat.v1.variable_scope('updatelambdaNN') as scope:
            if reuse:
                scope.reuse_variables()
                
            # 4th layer
            w4_reg = self.weight_variable('w4_reg',[nHidden, dOutput], trainable=trainable)
            bias4_reg = self.bias_variable('bias4_reg',[dOutput], trainable=trainable)
        
            #y = self.fc(x,w4_reg,bias4_reg,rate)
            y = self.fc_relu(x,w4_reg,bias4_reg,rate)
            
            y_thes = self.outputthes(y)
        
            return y_thes, y
    # ----
      
    # ----
    def pde(self, x, t, param, nData=100):
        
        pi = 3.14
    
        # a,bは、すべての u で共通
        tmpa = x - 4.0 * tf.transpose(t) # [t.shape, x.shape]
        tmpb = x - 4.0 * tf.transpose(t) - 2.0 * pi
        # データ数分の t [ndata, t.shape]
        ts = tf.tile(tf.expand_dims(t[0], 0), [nData, 1])
        # データごと(param)に計算 [ndata, t.shape]
        tmpc = 4.0 * param * (ts + 1.0)
            
        # + N dimention [nBatch, t.shape, x.shape]
        a = tf.tile(tf.expand_dims(tmpa, 0), [nData, 1, 1])
        b = tf.tile(tf.expand_dims(tmpb, 0), [nData, 1, 1])
        c = tf.tile(tf.expand_dims(tmpc, -1), [1, 1, self.xDim])
            
        # [nBatch, t.shape, x.shape]
        phi = tf.exp(- a * a / c) + tf.exp(- b * b / c)
        dphi = - 2.0 * a * tf.exp(- a * a / c ) / c - 2.0 * b * tf.exp(- b * b / c) / c
        
        invu = 4.0 - 2.0 * tf.expand_dims(param,1) * dphi / phi

        return invu
    # ----
    
    # ----
    def train(self, nItr=1000):
        
        # parameters ----
        printPeriod = 100
        batchCnt = 0
        nTrain = 1000
        batchRandInd = np.random.permutation(nTrain)
        # ※手動
        plotnus = [0.01, 0.02, 0.05, 0.1, 0.5, 1.0, 2.0, 4.0]
        # ----
        
        # Start training
        teUL = []
        tePL = []
        
        for itr in range(nItr):
            
            # ※1こずつ？
            flag = False
            cnt = 0
            for ind in batchRandInd:
                feed_dict={self.x:self.testX, self.t:self.testT, self.y:self.testNU[ind,np.newaxis], self.inobs:self.testU[ind,np.newaxis], self.outobs:self.testU[ind,np.newaxis,:,:,0].transpose(0,2,1)}    
           
                _, testParam, testPred, testploss, testuloss, grad =\
                self.sess.run([self.opt, self.predparam_test, self.predparam_pre, self.predu_test, self.loss_nu_test, self.loss_test, self.gradnu], feed_dict)
                
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
                    
                    tePL = np.append(tePL, testPLoss)
                    teUL = np.append(teUL, testULoss)
                    
                    if np.round(self.testNU[ind],3) in plotnus:
                        
                        exactnu = self.testNU[ind]
                        prednu = testPred[ind]
                        
                        self.myPlot.plotExactPredParam([self.testx, self.testt, prednu, exactnu], xNum=self.testx.shape[0], itr=itr, savename='tepredparamode')
        
        paramloss = [tePL]
        ulosses = [teUL]
        
        return paramloss, ulosses
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
    plosses, ulosses = model.train(nItr=nItr)
    # ----
    
    # Plot ----
    model.myPlot.Loss1(plosses, labels=['test'], savename='poNN_param')
    model.myPlot.Loss1(ulosses, labels=['test'], savename='poNN_u')
    
    #model.myPlot.plotExactPredParam(teparams, xNum=teparams[0].shape[0], savename='lasttepredparamode')
    # ----
 
    
