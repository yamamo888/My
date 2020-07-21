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
import plot


class ParamNN:
    def __init__(self, rateTrain=0.0, lr=1e-3, nBatch=100, trialID=0, dataMode='test'):
        
      
        # parameter ----
        if dataMode == 'large':
            # x,t
            self.xDim = 25
            tDim = 100
            self.uInput = self.xDim * int(tDim/10)
           
        self.nBatch = nBatch
        self.trialID = trialID
        # ----
        
        # Dataset ----
        self.myData = pdedata.pdeData(pdeMode='burgers', dataMode=dataMode)
        
        testx, testt, testu, self.testNU = self.myData.traintest()
        
        # [1000,100,256] -> [1000,100,xDim]
        idx = np.random.choice(testu.shape[-1], self.xDim, replace=False)
        self.testU = testu[:,:,idx]
        
        # input u [1000,uInput]
        idt = np.random.choice(testu.shape[1], 10, replace=False)
        self.testinU = np.reshape(self.testU[:,idt,:], [-1, self.uInput])
        
        self.testX = np.reshape(np.tile(testx[idx], testt.shape[0]), [-1, self.xDim])
        self.testT = np.reshape(np.tile(testt, self.xDim), [-1, testt.shape[0]])
        # ----
        
        # Placeholder ----
        # u
        self.inobs = tf.compat.v1.placeholder(tf.float32,shape=[None, self.uInput])
        self.outobs = tf.compat.v1.placeholder(tf.float32,shape=[None, tDim, self.xDim])
        # x,t
        self.x = tf.compat.v1.placeholder(tf.float32,shape=[tDim, self.xDim])
        self.t = tf.compat.v1.placeholder(tf.float32,shape=[self.xDim, tDim])
        # ----
        
        # PDE ----
        # output: u
        self.predu, self.predparam, self.a,self.b,self.c,self.phi,self.dphi = self.pde(self.x, self.t, self.inobs, nData=self.nBatch)
        self.predu_test, self.predparam_test, self.a_test,self.b_test,self.c_test,self.phi_test,self.dphi_test = self.pde(self.x, self.t, self.inobs, nData=1000, reuse=True)
        # ----
        #pdb.set_trace()
        # loss ----
        self.loss = tf.reduce_mean(tf.square(self.outobs - self.predu))
        self.loss_test = tf.reduce_mean(tf.square(self.outobs - self.predu_test))
        # ----
        
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
    def lambdaNN(self, x, rate=0.0, reuse=False):
        
        nHidden = 64
        
        with tf.compat.v1.variable_scope('lambdaNN') as scope:  
            if reuse:
                scope.reuse_variables()
            
            dInput = x.get_shape().as_list()[-1]
            dOutput = 1
            
            # 1st layer
            w1 = self.weight_variable('w1',[dInput, nHidden])
            bias1 = self.bias_variable('bias1',[nHidden])
            h1 = self.fc_relu(x,w1,bias1,rate)
            
            # 2nd layer
            w2 = self.weight_variable('w2',[nHidden, dOutput])
            bias2 = self.bias_variable('bias2',[dOutput])
            # nu
            y = self.fc(h1,w2,bias2,rate)
        
            return y
    # ----
    
    # ----
    def pde(self, x, t, u, nData=100, reuse=False):
        
        pi = 3.14
        
        with tf.compat.v1.variable_scope('pde') as scope:  
            if reuse:
                scope.reuse_variables()
            
            pdb.set_trace()
            
            # pred nu [ndata,]
            param = self.lambdaNN(u, reuse=reuse)
            
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
            
            return u,param,a,b,c,phi,dphi
    # ----
    
    # ----
    def train(self, nItr=1000):
        
        # parameters ----
        testPeriod = 500
        batchCnt = 0
        nTrain = int(5000 * 0.8)
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
            batchX = np.reshape(np.tile(batchx, batcht.shape[0]), [-1, batchx.shape[0]])
            batchT = np.reshape(np.tile(batcht, batchx.shape[0]), [-1, batcht.shape[0]])
            
            # ※ 工夫の余地あり(input)
            # random u(t) for feature (ｄａｔａ増やしたいときは第二引数+) 
            idt = np.random.choice(batchU.shape[1], 10, replace=False)
            batchinU = np.reshape(batchU[:,idt,:], [-1, self.uInput])
            
            #pdb.set_trace()
            feed_dict = {self.x:batchX, self.t:batchT, self.inobs:batchinU, self.outobs:batchU}
           
            _, trainParam, trainPred, trainLoss, traina,trainb,trainc,trainphi,traindphi =\
            self.sess.run([self.opt, self.predparam, self.predu, self.loss, self.a, self.b, self.c,self.phi,self.dphi], feed_dict)
            
            if eInd + self.nBatch > nTrain:
                batchCnt = 0
                batchRandInd = np.random.permutation(nTrain)
            else:
                batchCnt += 1
            
            # Test
            if itr % testPeriod == 0:

                self.test(itr=itr)
                
                print('----')
                print('itr: %d, trainLoss:%f' % (itr, trainLoss))
                
                trL = np.append(trL,trainLoss)
                teL = np.append(teL,self.testLoss)
                
                if not flag:
                    trP = trainParam
                    teP = self.testlambda
                    
                    flag = True
                else:
                    trP = np.vstack([trP, trainParam])
                    teP = np.vstack([teP, self.testParam])
                    
                pdb.set_trace()
                # Save model
                #self.saver.save(self.sess, os.path.join('model', 'burgers', 'first'), global_step=itr)
        
        # x,t
        data = [self.XT, self.testXY[1], self.testinvPred, self.testPreds]
        lambdas = [trP,teP]
        lambdasloss = [trPL,tePL]
        losses = [trL, teL]
    
        return losses, data, lambdas, lambdasloss
    # ----
    
    # ----
    def test(self,itr=0):
        pdb.set_trace()
        
        feed_dict={self.x:self.testX, self.t:self.testT, self.inobs:self.testinU, self.outobs:self.testU}    
        
        self.testParam, self.testPred, self.testLoss =\
        self.sess.run([self.predparam_test, self.predu_test, self.loss_test], feed_dict)
         
        print('itr: %d, testLoss:%f' % (itr, self.testLoss))
       
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
    losses = model.train(nItr=nItr)
    # ----
    
    # Plot ----
    myPlot = plot.Plot(trialID=trialID)
    myPlot.pLoss(losses, labels=['train','test'])
    # ----
 
    