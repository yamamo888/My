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
    def __init__(self, rateTrain=0.0, lr=1e-3, index=0, dataMode='test', isExactModel=False):
        
      
        # parameter ----
        if dataMode == 'large':
            self.xDim = 50
        elif dataMode == 'middle':
            self.xDim = 25
        elif dataMode == 'small':
            self.xDim = 10
        
        self.tDim = 100
        self.index = index
        self.yDim = 1
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
        self.predparam = self.lambdaNN(self.inobs)
        # ----
        
        # optimizer ----
        config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1,allow_growth=True)) 
        self.saver = tf.compat.v1.train.Saver()
        self.sess = tf.compat.v1.Session(config=config)
        # ----
        #pdb.set_trace()
        # Restore model ----
        if isExactModel:
            ckptpath = os.path.join('model', f'{dataMode}burgers_exact{self.index}')
            ckpt = tf.train.get_checkpoint_state(ckptpath)
            
            lastmodel = ckpt.model_checkpoint_path
            self.saver.restore(self.sess, lastmodel)
            print('>>> Restore test model')
        else: 
            ckptpath = os.path.join('model', f'{dataMode}burgers')
            ckpt = tf.train.get_checkpoint_state(ckptpath)
            
            lastmodel = ckpt.model_checkpoint_path
            self.saver.restore(self.sess, lastmodel)
            print('>>> Restore train model')
        # ----
         
        # float32 -> float64
        self.predparam = tf.cast(self.predparam, tf.float64)
        
        # PDE ----
        self.placeparam = tf.compat.v1.placeholder(tf.float64, shape=[None, self.yDim])
        # output: u
        self.predu = pdeburgers.burgers(self.placeparam)
        # ----
        
        # space data -> [none, self.xDim, t] ----
        self.indx = tf.compat.v1.placeholder(tf.int32,shape=[self.xDim,1])
        trans_predu = tf.transpose(self.predu, perm=[1,0,2])
        # [100(x),data,t]
        gather_predu = tf.gather_nd(trans_predu, self.indx)
        # [data,self.xDim,t]
        space_predu = tf.transpose(gather_predu, perm=[1,0,2])
        # ----

        # loss param ----
        self.loss_nu = tf.reduce_mean(tf.square(tf.cast(self.y, tf.float64) - self.placeparam)) 
        # ----
        
        # loss u ----   
        self.loss = tf.reduce_mean(tf.square(self.outobs - space_predu))
        # ----
        
        # gradient ----
        self.alpha = tf.compat.v1.placeholder(tf.float64, shape=[1])
        self.gradnu = tf.gradients(self.loss, self.placeparam)[0]
        
        self.nextparam = self.placeparam - (self.gradnu * self.alpha)
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
    def lambdaNN(self, x, reuse=False, trainable=False):
        
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
           
            y = self.fc(fc1,w6,b6)
            
         
            return y
    # ----
     
    # ----
    def clstrain(self, nItr=1000, nEpoch=3, alpha=0.01, nCls=0):
       
        pmin = 0.005
        pmax = 0.305
        clsWidth = (pmax - pmin) / nCls
        firstClsCenter = pmin + (clsWidth / 2)
       
        totalgrads,totalllosses,totalpreParam = [],[],[]
        for epoch in range(nEpoch):
            
            #pdb.set_trace()
            
            if pmin == pmax:
                break
            
            # paramters ----
            pcls = np.arange(pmin, pmax, clsWidth)
            pcls = np.append(pcls, pmax+0.001)
            # ----
            
            grads,llosses,preParam = [],[],[]
            for itr in range(nItr):
                
                if itr == 0:
                    predParam = 0.05
                elif itr > 0:
                    #pdb.set_trace()
                    predParam = preParam[itr-1]
               
                feed_dict={self.y:self.testNU[self.index,np.newaxis], self.inobs:self.testU[self.index,np.newaxis], 
                           self.outobs:self.testU[self.index,np.newaxis,:,:,0], self.indx:self.idx[:,np.newaxis], 
                           self.placeparam:np.array([predParam])[:,None], self.alpha:np.array([alpha])}
          
                grad, nextParam, lloss, vloss = self.sess.run([self.gradnu, self.nextparam, self.loss_nu, self.loss], feed_dict)
                
                # start 0,1,2.. cnt == num. of class 
                cntNum = [cnt for cnt in range(pcls.shape[0]-1) if pcls[cnt] <= nextParam < pcls[cnt+1]]
                
                # nextParam < 0.005 or nextParam > 0.305 -> 1 or -1 class
                if nextParam < pmin:
                    cntNum = [0]
                elif nextParam > pmax:
                    cntNum = [pcls.shape[0]-2]
                
                # ※ 4 調整要るかも
                nextParam = np.round(cntNum[0] * clsWidth + firstClsCenter,4)
                
                preParam = np.append(preParam, nextParam)
                
                # if Stop update nextParam
                if itr > 1 and preParam[itr-2] == preParam[itr]:
                    #pdb.set_trace()
                    # ※ 4 ちょうせいいるかも
                    pmin = np.round(nextParam - (clsWidth/2), 6)
                    pmax = np.round(nextParam + (clsWidth/2), 6)
                    
                    print('Fin update lambda...')
                    break
                
                grads = np.append(grads, grad)
                llosses = np.append(llosses, lloss)
                
                print('----')
                print('exact lambda: %.8f predlambda: %.8f' % (self.testNU[self.index], preParam[itr-1]))
                print('lambda mse: %.10f' % (lloss))
                print('v mse: %.10f' % (vloss))
                print('gradient (closs/param): %f' % (grad))
        
            totalgrads = np.append(totalgrads,grads)
            totalllosses = np.append(totalllosses, llosses)
            totalpreParam = np.append(totalpreParam, preParam)
        
        return [llosses], [grads], np.round(preParam[0],6)
    # ----
    
    # ----
    def randomtrain(self, nItr=1000, alpha=0.01):
       
        # paramters ----
        pmin = 0.005
        pmax = 0.305
        print('>>> random mode')
        # all
        randomarray = np.arange(pmin,pmax,0.0001)
        # ----
        
        grads,llosses,preParam = [],[],[]
        for itr in range(nItr):
            
            if isEveryRandomParam and itr == 0:
                predParam = random.choice(randomarray)
            if (isEveryRandomParam and itr > 0) or (isCls and itr > 0):
                #pdb.set_trace()
                predParam = preParam[itr-1]
            if not isEveryRandomParam and itr == 0:
                predParam = 0.05
            
            
            feed_dict={self.y:self.testNU[self.index,np.newaxis], self.inobs:self.testU[self.index,np.newaxis], 
                       self.outobs:self.testU[self.index,np.newaxis,:,:,0], self.indx:self.idx[:,np.newaxis], 
                       self.placeparam:np.array([predParam])[:,None], self.alpha:np.array([alpha])}
      
            
            grad, nextParam, lloss, vloss = self.sess.run([self.gradnu, self.nextparam, self.loss_nu, self.loss], feed_dict)
            
            if itr % 10 == 0:
                nextParam = np.array([[random.choice(randomarray)]])
            
            preParam = np.append(preParam, nextParam)
            
       
            grads = np.append(grads, grad)
            llosses = np.append(llosses, lloss)
            
            print('----')
            print('exact lambda: %.8f predlambda: %.8f' % (self.testNU[self.index], preParam[itr-1]))
            print('lambda mse: %.10f' % (lloss))
            print('v mse: %.10f' % (vloss))
            print('gradient (closs/param): %f' % (grad))
    
        return [llosses], [grads], np.round(preParam[0],6)
    # ----
   
    
if __name__ == "__main__":
    
    # command argment ----
    parser = argparse.ArgumentParser()

    # iteration of training
    parser.add_argument('--nItr', type=int, default=100)
    # datamode (pkl)
    parser.add_argument('--dataMode', required=True, choices=['large', 'middle', 'small'])
    # index test 2=0.01
    parser.add_argument('--index', type=int, default=2)
    # alpha * grad
    parser.add_argument('--alpha', type=float, default=0.0001)
    # select fine-tuned model exact(only test data) == flag or train data
    parser.add_argument('--exactflag', action='store_true')
    # select random param == flag or not random param
    parser.add_argument('--everyrandomflag', action='store_true')
    # classification param == flag
    parser.add_argument('--clsflag', action='store_true')
    # num of class
    parser.add_argument('--nCls', type=int, choices=[10, 50, 100])
    
    
    # 引数展開
    args = parser.parse_args()
    
    nItr = args.nItr
    dataMode = args.dataMode
    index = args.index
    alpha = args.alpha
    isExactModel = args.exactflag
    isEveryRandomParam = args.everyrandomflag
    isCls = args.clsflag
    nCls = args.nCls
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
    model = ParamNN(rateTrain=rateTrain, lr=lr, index=index,
                    dataMode=dataMode, isExactModel=isExactModel)
    
    if isEveryRandomParam:
        llosses, grads, preparam = model.clstrain(nItr=nItr, alpha=alpha, nCls=nCls)    
    elif isCls: 
        llosses, grads, preparam = model.clstrain(nItr=nItr, alpha=alpha, nCls=nCls)
    # ----
    
    # Plot ----
    myPlot = pdeplot.Plot(dataMode=dataMode, trialID=index)
    myPlot.Loss1(llosses, labels=['test'], savename=f'poNN_testloss_cls_{preparam}')
    myPlot.Loss1(grads, labels=['test'], savename=f'poNN_testgrad_cls_{preparam}')
    # ----
