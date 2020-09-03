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
            # x,t
            self.xDim = 25 
        elif dataMode == 'middle':
            # x,t
            self.xDim = 20
        elif dataMode == 'small':
            # x,t
            self.xDim = 2
        
        self.tDim = 100
        self.nBatch = nBatch
        self.trialID = trialID
        self.uInput = self.xDim * int(self.tDim/10)   
        # ----
        
        # Dataset ----
        self.myData = pdedata.pdeData(pdeMode='burgers', dataMode=dataMode)
        # [xDim,1], [100,1], [data, xDim, 100], [data,] 
        testx, testt, self.testU, self.testNU = self.myData.traintest()
         
        # [testdata, 256] -> [testdata, xdim]
        self.testX = np.reshape(np.tile(testx, self.tDim), [-1, self.xDim])
        self.testT = np.reshape(np.tile(testt, self.xDim), [-1, self.tDim])
        # ----
         
        # Placeholder ----
        # u
        self.inobs = tf.compat.v1.placeholder(tf.float32,shape=[None, self.xDim, self.tDim, 1])
        self.outobs = tf.compat.v1.placeholder(tf.float32,shape=[None, self.tDim, self.xDim])
        # x,t
        self.x = tf.compat.v1.placeholder(tf.float32,shape=[self.tDim, self.xDim])
        self.t = tf.compat.v1.placeholder(tf.float32,shape=[self.xDim, self.tDim])
        # ----
        # Restore neural network ----
        # pred nu [ndata,]
        hidden = self.RestorelambdaNN(self.inobs)
        hidden_test = self.RestorelambdaNN(self.inobs, reuse=True)
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
        self.param = self.lambdaNN(hidden)
        self.param_test = self.lambdaNN(hidden_test, reuse=True)
        
        # PDE ----
        # output: u
        self.predu, self.predparam = self.pde(self.x, self.t, self.param, nData=self.nBatch)
        # ※ testデータサイズは手動
        self.predu_test, self.predparam_test = self.pde(self.x, self.t, self.param_test, nData=self.testNU.shape[0], reuse=True)
        # ----
        #pdb.set_trace()
        # loss ----   
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.square(self.outobs - self.predu),2),1))
        self.loss_test = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.square(self.outobs - self.predu_test),2),1))
        # ----
        # gradient
        self.gradu = tf.gradients(self.loss, self.inobs)[0]

        # Optimizer ----
        self.opt = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.loss)
        # ----
        
        lambdaVars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,scope='updatelambdaNN') 
        pdeVars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,scope='pde')
        lambdaVars.append(pdeVars)
        self.opt = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.loss, var_list=lambdaVars)
        
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
        
            y = self.fc(x,w4_reg,bias4_reg,rate)
        
            return y  
    # ----
      
    # ----
    def pde(self, x, t, param, nData=100, reuse=False):
        
        pi = 3.14
        
        with tf.compat.v1.variable_scope('pde') as scope:  
            if reuse:
                scope.reuse_variables()
            
            #pdb.set_trace()
        
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

            #return u,param,a,b,c,phi,dphi
            return invu, param
    # ----
    
    # ----
    def train(self, nItr=1000):
        
        # parameters ----
        testPeriod = 100
        batchCnt = 0
        nTrain = 3995
        batchRandInd = np.random.permutation(nTrain)
        # ----
        
        # Start training
        trL,teL = [],[]
        trPL,tePL = [],[]
        flag = False
        for itr in range(nItr):
            
            # index
            sInd = self.nBatch * batchCnt
            eInd = sInd + self.nBatch
            index = batchRandInd[sInd:eInd]
            # Get train data
            # [x.shape], [100,], [nbatch, t.shape, x.shape]
            batchx, batcht, batchU, batchNU = self.myData.nextBatch(index)
            
            # [nbatch,100] -> [nbathc, x.shape]
            batchX = np.reshape(np.tile(batchx, self.tDim), [-1, self.xDim])
            batchT = np.reshape(np.tile(batcht, self.xDim), [-1, self.tDim])
            
            feed_dict = {self.x:batchX, self.t:batchT, self.inobs:batchU[:,:,:,np.newaxis], self.outobs:batchU.transpose(0,2,1)}
           
            _, trainParam, trainPred, trainULoss, grad =\
            self.sess.run([self.opt, self.predparam, self.predu, self.loss, self.gradu], feed_dict)
            
            
            trainPLoss = np.mean(np.square(batchNU - trainParam))

            if eInd + self.nBatch > nTrain:
                batchCnt = 0
                batchRandInd = np.random.permutation(nTrain)
            else:
                batchCnt += 1
            
            # Test
            if itr % testPeriod == 0:

                self.test(itr=itr)
                print('----')
                print('itr: %d, trainULoss:%f, trainPLoss:%f' % (itr, trainULoss, trainPLoss))
                print(f'train exact: {batchNU[:5]}')
                print(f'train pred: {trainParam[:5]}')
                print(np.mean(grad))
                
                # u loss 
                trL = np.append(trL,trainULoss)
                teL = np.append(teL,self.testULoss)
                
                # param loss
                trPL = np.append(trPL,trainPLoss)
                tePL = np.append(tePL,self.testPLoss)

                # Save model
                #self.saver.save(self.sess, os.path.join('model', 'burgers', 'first'), global_step=itr)
        #pdb.set_trace()
        paramloss = [trPL,tePL]
        ulosses = [trL, teL]
    
        return ulosses, paramloss
    # ----
    
    # ----
    def test(self,itr=0):
        
        #pdb.set_trace() 
        feed_dict={self.x:self.testX, self.t:self.testT, self.inobs:self.testU[:,:,:,np.newaxis], self.outobs:self.testU.transpose(0,2,1)}    
        
        self.testParam, self.testPred, self.testULoss =\
        self.sess.run([self.predparam_test, self.predu_test, self.loss_test], feed_dict)

        self.testPLoss = np.mean(np.square(self.testNU-self.testParam))

        #pdb.set_trace()
        print('itr: %d, testULoss:%f, testPLoss:%f' % (itr, self.testULoss, self.testPLoss))
        print(f'test exact: {self.testNU[:5]}')
        print(f'test pred: {self.testParam[:5]}')
       
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
    ulosses, plosses = model.train(nItr=nItr)
    # ----
    
    # Plot ----
    myPlot = pdeplot.Plot(trialID=trialID)
    myPlot.pLoss(ulosses, labels=['train','test'], savename='u')
    myPlot.pLoss(plosses, labels=['train','test'], savename='param')
    # ----
 
    
