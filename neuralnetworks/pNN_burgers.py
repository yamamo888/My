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
        tDim = 100
        self.nBatch = nBatch
        self.trialID = trialID
        self.yDim = 1
        # ----
        
        # for plot
        self.myPlot = pdeplot.Plot(dataMode=dataMode, trialID=trialID)
    
        # Dataset ----
        self.myData = pdedata.pdeData(pdeMode='burgers', dataMode=dataMode)
        
        #self.testX, self.testT, self.testU, self.testNU, self.varU, self.varNU  = self.myData.traintestvaridation(ismask=True)
        # [256,1] [xDim,1] [100,1] [data,xDim,100,1]
        self.alltestX, self.testX, self.testT, self.testU, self.testNU, self.varX, self.varU, self.varNU  = self.myData.traintestvaridation()
        #pdb.set_trace()
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
        testPeriod = 100
        savemodelPeriod = 500
        batchCnt = 0
        #nTrain = 10
        nTrain = 3991
        batchRandInd = np.random.permutation(nTrain)
        # ----
        
        # Start training
        trPL,tePL,varPL = [],[],[]
        trUL,teUL,varUL = [],[],[]
        
        for itr in range(nItr):
            
            # index
            sInd = self.nBatch * batchCnt
            eInd = sInd + self.nBatch
            index = batchRandInd[sInd:eInd]
            # Get train data
            # [x.shape], [100,], [nbatch, t.shape, x.shape]
            batchX, batchT, batchU, batchNU = self.myData.miniBatch(index)
            #pdb.set_trace()
            # y: prameter b
            #feed_dict = {self.y:batchNU[:,np.newaxis], self.inobs:batchU}
            feed_dict = {self.y:batchNU[:,np.newaxis], self.inobs:batchU[:,:,:,np.newaxis]}
            
            _, trainParam, trainLoss =\
            self.sess.run([self.opt, self.predy, self.loss], feed_dict)
            
            if eInd + self.nBatch > nTrain:
                batchCnt = 0
                batchRandInd = np.random.permutation(nTrain)
            else:
                batchCnt += 1
            
            # Test & Varidation
            if itr % testPeriod == 0:

                # train return nu -> u
                #params = [self.testX, self.testT, trainParam]
                params = [batchX, batchT, trainParam, batchNU]
                #pdb.set_trace()
                invU = self.myPlot.paramToU(params, xNum=self.xDim)
                
                trainULoss = np.mean(np.sum(np.sum(np.square(batchU - invU),2),1))

                self.test(itr=itr)
                self.varidation(itr=itr)
                
                print('----')
                print('itr: %d, trainLoss:%f, trainULoss:%f' % (itr, trainLoss, trainULoss))
                print(f'train exact: {batchNU[:5]}')
                print(f'train pred: {trainParam[:5]}')
                
                # param loss
                trPL = np.append(trPL,trainLoss)
                tePL = np.append(tePL,self.testLoss)
                varPL = np.append(varPL,self.varLoss)
                
                # invu loss
                trUL = np.append(trUL, trainULoss)
                teUL = np.append(teUL, self.testULoss)
                varUL = np.append(varUL, self.varULoss)

            if itr % savemodelPeriod == 0:
                # Save model
                self.saver.save(self.sess, os.path.join('model', f'{dataMode}burgers', f'first_{dataMode}'), global_step=itr)
        #pdb.set_trace()
        paramloss = [trPL,tePL,varPL]
        uloss = [trUL,teUL,varUL]
        
        teparams = [self.testX, self.testT, self.testParam, self.testNU]
        varparams = [self.testX, self.testT, self.varParam, self.varNU]
        
        return paramloss, uloss, teparams, varparams
    # ----
    
    # ----
    def test(self,itr=0):
        
        feed_dict={self.y: self.testNU, self.inobs:self.testU}    
        
        self.testParam, self.testLoss =\
        self.sess.run([self.predy_test, self.loss_test], feed_dict)
        
        #pdb.set_trace()
        # return nu -> u
        params = [self.testX, self.testT, self.testParam, self.testNU]
        invU = self.myPlot.paramToU(params, xNum=self.xDim)
        
        self.testULoss = np.mean(np.sum(np.sum(np.square(self.testU[:,:,:,0] - invU),2),1))
        #pdb.set_trace()
        
        prednu_maemax = self.testParam[np.argmax(np.square(self.testNU - self.testParam))]
        exactnu_maemax = self.testNU[np.argmax(np.square(self.testNU - self.testParam))]
        
        prednu_maemin = self.testParam[np.argmin(np.square(self.testNU - self.testParam))]
        exactnu_maemin = self.testNU[np.argmin(np.square(self.testNU - self.testParam))]

        prednu_maxmin = np.vstack([prednu_maemin, prednu_maemax])
        exactnu_maxmin = np.vstack([exactnu_maemin, exactnu_maemax])

        self.myPlot.plotExactPredParam([self.testX, self.testT, prednu_maxmin, exactnu_maxmin], xNum=self.testX.shape[0], itr=itr, savename='tepredparam')

        print('itr: %d, testPLoss:%f, testULoss:%f' % (itr, self.testLoss, self.testULoss))
        print(f'test exact: {self.testNU[:5]}')
        print(f'test pred: {self.testParam[:5]}')
    # ----
    
    # ----
    def varidation(self,itr=0):
        
        feed_dict={self.y: self.varNU, self.inobs:self.varU}    
        
        self.varParam, self.varLoss =\
        self.sess.run([self.predy_test, self.loss_test], feed_dict)

        #pdb.set_trace() 
        # return nu -> u
        params = [self.varX, self.testT, self.varParam, self.varNU]
        invU = self.myPlot.paramToU(params, xNum=self.xDim)
        
        self.varULoss = np.mean(np.sum(np.sum(np.square(self.varU[:,:,:,0] - invU),2),1))
    
        self.myPlot.plotExactPredParam(params, xNum=self.varX.shape[0], itr=itr, savename='varpredparam')
        
        print('itr: %d, varPLoss:%f, varULoss:%f' % (itr, self.varLoss, self.varULoss))
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
    plosses, ulosses, teparams, varparams = model.train(nItr=nItr)
    # ----
    
    # Plot ----
    #myPlot = pdeplot.Plot(dataMode=dataMode, trialID=trialID)
    model.myPlot.Loss(plosses, labels=['train','test','varid'], savename='pNN_param')
    model.myPlot.Loss(ulosses, labels=['train','test','varid'], savename='pNN_u')
    model.myPlot.plotExactPredParam(teparams, xNum=teparams[0].shape[0], savename='lasttepredparam')
    model.myPlot.plotExactPredParam(varparams, xNum=varparams[0].shape[0], savename='lastvarpredparam')
    # ----
    
 
    
