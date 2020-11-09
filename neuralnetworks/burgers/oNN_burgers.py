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
    def __init__(self, rateTrain=0.0, lr=1e-3, index=0, dataMode='test'):
        
      
        # parameter ----
        if dataMode == 'large':
            self.xDim = 50
        elif dataMode == 'middle':
            self.xDim = 25
        elif dataMode == 'small':
            self.xDim = 10
        
        #self.xDim = 256 
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
        self.outobs = tf.compat.v1.placeholder(tf.float64,shape=[None, self.xDim, self.tDim])
        # param nu 
        self.y = tf.compat.v1.placeholder(tf.float32,shape=[None, self.yDim])
        # ----
         
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
        
        # loss u ----   
        self.loss = tf.reduce_mean(tf.square(self.outobs - space_predu))
        # ----
        
        # gradient ----
        self.alpha = tf.compat.v1.placeholder(tf.float64, shape=[1])
        self.gradnu = tf.gradients(self.loss, self.placeparam)[0]
        
        self.nextparam = self.placeparam - (self.gradnu * self.alpha)
        # ----

    # ----
    def train(self, nItr=1000, alpha=0.01, isEveryRandomParam=False):
        
    
        grads = []
        llosses = []
        preParam = []
        if isEveryRandomParam:
            
            print('>>> random mode')
    
            # 0.01
            #randomarray = np.arange(0.005,0.009,0.0001)
            #randomarray = np.arange(0.01,0.1,0.0001)
            #randomarray = np.arange(0.11,0.3,0.0001)
            # 0.005
            #randomarray = np.arange(0.005,0.014,0.0001)
            #randomarray = np.arange(0.015,0.095,0.0001)
            #randomarray = np.arange(0.10,0.3,0.0001)
            # 0.30
            #randomarray = np.arange(0.291,0.305,0.0001)
            #randomarray = np.arange(0.21,0.290,0.0001)
            #randomarray = np.arange(0.005,0.20,0.0001)
            # all
            randomarray = np.arange(0.005,0.305,0.0001)
        
        for itr in range(nItr):
                
                if itr == 0:
                    predParam = np.array([[random.choice(randomarray)]])
                else:
                    predParam = preParam[itr-1]
                
                feed_dict={self.y:self.testNU[self.index,np.newaxis], self.outobs:self.testU[self.index,np.newaxis,:,:,0],
                           self.indx:self.idx[:,np.newaxis], 
                           self.placeparam:np.array([predParam])[:,None], self.alpha:np.array([alpha])}
  
                grad, nextParam, lloss, vloss = self.sess.run([self.gradnu, self.nextparam, self.loss_nu, self.loss], feed_dict)
                
                # if itr % 10 == 0
                #nextParam = np.array([[random.choice(randomarray)]])
                
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
    # select random param == flag or not random param
    parser.add_argument('--everyrandomflag', action='store_true')
    # classification param == flag
    parser.add_argument('--clsflag', action='store_true')
    
    
    # 引数展開
    args = parser.parse_args()
    
    nItr = args.nItr
    dataMode = args.dataMode
    index = args.index
    alpha = args.alpha
    isEveryRandomParam = args.everyrandomflag
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
    model = ParamNN(rateTrain=rateTrain, lr=lr, index=index, dataMode=dataMode)
    llosses, grads, preparam = model.train(nItr=nItr, alpha=alpha, isEveryRandomParam=isEveryRandomParam)
    # ----
    
    # Plot ----
    myPlot = pdeplot.Plot(dataMode=dataMode, trialID=index)
    myPlot.Loss1(llosses, labels=['test'], savename=f'poNN_testloss_{preparam}')
    myPlot.Loss1(grads, labels=['test'], savename=f'poNN_testgrad_{preparam}')
    # ----
