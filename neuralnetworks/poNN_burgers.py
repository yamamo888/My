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
        # for varidation data
        self.varX = np.reshape(np.tile(self.varx, self.tDim), [-1, self.xDim])
        self.varT = np.reshape(np.tile(self.testt, self.xDim), [-1, self.tDim])
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
        hidden_test = self.RestorelambdaNN(self.inobs, reuse=True)
        hidden_vard = self.RestorelambdaNN(self.inobs, reuse=True)
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
        self.param_vard = self.lambdaNN(hidden_vard, reuse=True)
        # ----
        
        # PDE ----
        # output: u
        self.predu, self.predparam = self.pde(self.x, self.t, self.param, nData=self.nBatch)
        self.predu_test, self.predparam_test = self.pde(self.x, self.t, self.param_test, nData=self.testNU.shape[0], reuse=True)
        self.predu_vard, self.predparam_vard = self.pde(self.x, self.t, self.param_vard, nData=1, reuse=True)
        # ----
        #pdb.set_trace()

        # loss param ----
        self.loss_nu = tf.reduce_mean(tf.square(self.y - self.param)) 
        self.loss_nu_test = tf.reduce_mean(tf.square(self.y - self.param_test)) 
        self.loss_nu_vard = tf.reduce_mean(tf.square(self.y - self.param_vard)) 
        # ----

        # loss u ----   
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.square(self.outobs - self.predu),2),1))
        self.loss_test = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.square(self.outobs - self.predu_test),2),1))
        self.loss_vard = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.square(self.outobs - self.predu_vard),2),1))
        # ----
        
        # gradient
        self.gradu = tf.gradients(self.loss, self.inobs)[0]

        # Optimizer ----
        self.opt_nu = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.loss_nu)
        self.opt = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.loss)
        # ----
        
        lambdaVars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,scope='updatelambdaNN') 
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
        
            #y = self.fc(x,w4_reg,bias4_reg,rate)
            y = self.fc_relu(x,w4_reg,bias4_reg,rate)
        
            return y  
    # ----
      
    # ----
    def pde(self, x, t, param, nData=100, reuse=False):
        
        pi = 3.14
        # 桁落ち防ぐため
        zero_flow = tf.math.exp(tf.constant([[-10.0]]))
        

        # a,bは、すべての u で共通
        tmpa = x - 4.0 * tf.transpose(t) # [t.shape, x.shape]
        tmpb = x - 4.0 * tf.transpose(t) - 2.0 * pi
        # データ数分の t [ndata, t.shape]
        ts = tf.tile(tf.expand_dims(t[0], 0), [nData, 1])
        # データごと(param)に計算 [ndata, t.shape]
        tmpc = 4.0 * (param + zero_flow) * (ts + 1.0)
            
        # + N dimention [nBatch, t.shape, x.shape]
        a = tf.tile(tf.expand_dims(tmpa, 0), [nData, 1, 1])
        b = tf.tile(tf.expand_dims(tmpb, 0), [nData, 1, 1])
        c = tf.tile(tf.expand_dims(tmpc, -1), [1, 1, self.xDim])
            
        # [nBatch, t.shape, x.shape]
        phi = tf.exp(- a * a / c) + tf.exp(- b * b / c)
        dphi = - 2.0 * a * tf.exp(- a * a / c ) / c - 2.0 * b * tf.exp(- b * b / c) / c
        
        invu = 4.0 - 2.0 * tf.expand_dims(param,1) * dphi / (phi + tf.tile(tf.expand_dims(zero_flow,1),[1,self.tDim,self.xDim]))

        return invu, param
    # ----
    
    # ----
    def train(self, nItr=1000):
        
        # parameters ----
        testPeriod = 100
        batchCnt = 0
        nTrain = 3991
        batchRandInd = np.random.permutation(nTrain)
        # ----
        
        # Start training
        trL,teL,varL = [],[],[]
        trPL,tePL,varPL = [],[],[]
        trPUL,tePUL,varPUL = [],[],[]
        
        for itr in range(nItr):
            
            # index
            sInd = self.nBatch * batchCnt
            eInd = sInd + self.nBatch
            index = batchRandInd[sInd:eInd]
            # Get train data
            # [x.shape], [100,], [nbatch, t.shape, x.shape]
            batchx, batcht, batchU, batchNU = self.myData.miniBatch(index)
            
            # [nbatch,100] -> [nbathc, x.shape]
            batchX = np.reshape(np.tile(batchx, self.tDim), [-1, self.xDim])
            batchT = np.reshape(np.tile(batcht, self.xDim), [-1, self.tDim])
            #pdb.set_trace() 
            feed_dict = {self.x:batchX, self.t:batchT, self.y:batchNU[:,np.newaxis], self.inobs:batchU[:,:,:,np.newaxis], self.outobs:batchU.transpose(0,2,1)}
            
            _,_, trainParam, trainPred, trainULoss, grad =\
            self.sess.run([self.opt_nu, self.opt, self.predparam, self.predu, self.loss, self.gradu], feed_dict)

            '''
            # ※手動
            if itr < 2500:
                _, trainParam, trainPred, trainULoss, grad =\
                self.sess.run([self.opt_nu, self.predparam, self.predu, self.loss, self.gradu], feed_dict)
            elif itr >= 2500:
                _, trainParam, trainPred, trainULoss, grad =\
                self.sess.run([self.opt, self.predparam, self.predu, self.loss, self.gradu], feed_dict)
            '''
            
            trainPLoss = np.mean(np.square(batchNU - trainParam))

            if eInd + self.nBatch > nTrain:
                batchCnt = 0
                batchRandInd = np.random.permutation(nTrain)
            else:
                batchCnt += 1
            
            # Test
            if itr % testPeriod == 0:
                
                # pred nu -> u (py)
                params = [batchx, batcht, trainParam, batchNU]
                invU = self.myPlot.paramToU(params, xNum=self.xDim)
                trainPULoss = np.mean(np.sum(np.sum(np.square(batchU - invU),2),1))
                
                self.test(itr=itr)
                self.varidation(itr=itr)
                
                print('----')
                print('itr: %d, trainULoss:%f, trainPLoss:%f' % (itr, trainULoss, trainPLoss))
                print(f'train exact: {batchNU[:5]}')
                print(f'train pred: {trainParam[:5]}')
                #print(np.mean(grad))
                
                # u loss 
                trL = np.append(trL,trainULoss)
                teL = np.append(teL,self.testULoss)
                varL = np.append(varL,self.varULoss)
                
                # param -> u loss
                trPUL = np.append(trPUL,trainPULoss)
                tePUL = np.append(tePUL,self.testPULoss)
                varPUL = np.append(varPUL,self.varPULoss)
                
                # param loss
                trPL = np.append(trPL,trainPLoss)
                tePL = np.append(tePL,self.testPLoss)
                varPL = np.append(varPL,self.varPLoss)
            
            #if itr % savemodelPeriod == 0:
                # Save model
                #self.saver.save(self.sess, os.path.join('model', 'burgers', 'first'), global_step=itr)
        
        #pdb.set_trace()
        paramloss = [trPL, tePL, varPL]
        ulosses = [trL, teL, varL]
        pulosses = [trPUL, tePUL, varPUL]
    
        teparams = [self.testx, self.testt, self.testParam, self.testNU]
        varparams = [self.varx, self.testt, self.varParam, self.varNU]
        
        return paramloss, ulosses, pulosses, teparams, varparams
    # ----
    
    # ----
    def test(self,itr=0):
        
        #pdb.set_trace() 
        feed_dict={self.x:self.testX, self.t:self.testT, self.y:self.testNU, self.inobs:self.testU, self.outobs:self.testU[:,:,:,0].transpose(0,2,1)}    
        
        self.testParam, self.testPred, self.testULoss =\
        self.sess.run([self.predparam_test, self.predu_test, self.loss_test], feed_dict)

        self.testPLoss = np.mean(np.square(self.testNU-self.testParam))

        # return nu -> u ---
        params = [self.testx, self.testt, self.testParam, self.testNU]
        invU = self.myPlot.paramToU(params, xNum=self.xDim) 
        self.testPULoss = np.mean(np.sum(np.sum(np.square(self.testU[:,:,:,0] - invU),2),1))
        
        prednu_maemax = self.testParam[np.argmax(np.square(self.testNU - self.testParam))]
        exactnu_maemax = self.testNU[np.argmax(np.square(self.testNU - self.testParam))]
        
        prednu_maemin = self.testParam[np.argmin(np.square(self.testNU - self.testParam))]
        exactnu_maemin = self.testNU[np.argmin(np.square(self.testNU - self.testParam))]

        prednu_maxmin = np.vstack([prednu_maemin, prednu_maemax])
        exactnu_maxmin = np.vstack([exactnu_maemin, exactnu_maemax])

        self.myPlot.plotExactPredParam([self.testx, self.testt, prednu_maxmin, exactnu_maxmin], xNum=self.testx.shape[0], itr=itr, savename='tepredparamode')
        # ---
        
        print('itr: %d, testULoss:%f, testPLoss:%f' % (itr, self.testULoss, self.testPLoss))
        print(f'test exact: {self.testNU[:5]}')
        print(f'test pred: {self.testParam[:5]}')
       
    # ----
    
    # ----
    def varidation(self,itr=0):
        #pdb.set_trace() 
        feed_dict={self.x:self.testX, self.t:self.testT, self.y:self.varNU, self.inobs:self.varU, self.outobs:self.varU[:,:,:,0].transpose(0,2,1)}    
        
        self.varParam, self.varPred, self.varULoss =\
        self.sess.run([self.predparam_vard, self.predu_vard, self.loss_vard], feed_dict)

        self.varPLoss = np.mean(np.square(self.varNU-self.varParam))

        # return nu -> u ---
        params = [self.varx, self.testt, self.varParam, self.varNU]
        invU = self.myPlot.paramToU(params, xNum=self.xDim)
        self.varPULoss = np.mean(np.sum(np.sum(np.square(self.varU[:,:,:,0] - invU),2),1))
    
        self.myPlot.plotExactPredParam(params, xNum=self.varx.shape[0], itr=itr, savename='varpredparamode')
        # ---
    
        print('itr: %d, varULoss:%f, varPLoss:%f' % (itr, self.varULoss, self.varPLoss))
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
    plosses, ulosses, pulosses, teparams, varparams = model.train(nItr=nItr)
    # ----
    
    # Plot ----
    model.myPlot.Loss(plosses, labels=['train','test','varid'], savename='poNN_param')
    model.myPlot.Loss(ulosses, labels=['train','test','varid'], savename='poNN_u')
    model.myPlot.Loss(pulosses, labels=['train','test','varid'], savename='poNN_invu')
    
    model.myPlot.plotExactPredParam(teparams, xNum=teparams[0].shape[0], savename='lasttepredparamode')
    model.myPlot.plotExactPredParam(varparams, xNum=varparams[0].shape[0], savename='lastvarpredparamode')
    # ----
 
    
