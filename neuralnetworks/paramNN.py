# -*- coding: utf-8 -*-

import sys
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v1

import random
import pickle
import pdb

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt

import data


class ParamNN:
    def __init__(self, keepProbTrain=1.0, lr=1e-3, dInput=50, dOutput=3,
                 nCell=5, nWindow=10):
        
        # Select train data *pkl
        self.nametrInds = [0,1,2,3,4,5,6,7]
        # random sample loading train data
        nameInds = random.sample(self.nametrInds,3)
        
        # Dataset ----
        self.myData = data.NankaiData(nCell=nCell, nWindow=nWindow)
        self.xTest, self.yTest = self.myData.loadTrainTestData(nameInds=nameInds)
        # ----
        
        # parameter ----
        self.dInput = dInput
        self.dOutput = dOutput
        # ----
        
        pdb.set_trace()
      
        # Placeholder ----
        self.x = tf.placeholder(tf.float32,shape=[None, self.dInput])
        self.y = tf.placeholder(tf.float32,shape=[None, self.dOutput])
        # ----
        
        self.pred = self.Regress(self.x, keepProb=keepProbTrain)
        self.loss = tf.reduce_mean(tf.square(self.y - self.pred))
        
        Vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Regress') 
        self.opt = tf.train.AdamOptimizer(lr).minimize(self.loss,var_list=Vars)
        
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1,allow_growth=True)) 
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        
        # save model
        saver = tf.train.Saver()
        saver.save(self.sess, os.path.join('model','paramNN'))
                
    
    # ----
    def weight_variable(self,name,shape):
         return tf.get_variable(name,shape,initializer=tf.random_normal_initializer(stddev=0.1))
    # ----
    # ----
    def bias_variable(self,name,shape):
         return tf.get_variable(name,shape,initializer=tf.constant_initializer(0.1))
    # ----
    # ----
    def fc_relu(self,inputs,w,b,keepProb):
         relu = tf.matmul(inputs,w) + b
         relu = tf.nn.dropout(relu, keepProb)
         relu = tf.nn.relu(relu)
         return relu
    # ----
    # ----
    def fc(self,inputs,w,b,keepProb):
         fc = tf.matmul(inputs,w) + b
         fc = tf.nn.dropout(fc, keepProb)
         return fc
    # ----
    # ----
    def Regress(self, x_r, keepProb=1.0, reuse=False, isCycle=False):
        
        pdb.set_trace()
        
        nHidden=128
        
        with tf.variable_scope('Regress') as scope:  
            if reuse:
                scope.reuse_variables()
    
            # 1st layer
            w1_reg = self.weight_variable('w1_reg',[self.dInput, nHidden])
            bias1_reg = self.bias_variable('bias1_reg',[nHidden])
            h1 = self.fc_relu(x_r,w1_reg,bias1_reg,keepProb)
            
            # 2nd layer
            w2_reg = self.weight_variable('w2_reg',[nHidden, nHidden])
            bias2_reg = self.bias_variable('bias2_reg',[nHidden])
            h2 = self.fc_relu(h1,w2_reg,bias2_reg,keepProb)
            
            # 3rd layer 
            w3_reg = self.weight_variable('w3_reg',[nHidden, nHidden])
            bias3_reg = self.bias_variable('bias3_reg',[nHidden])
            h3 = self.fc_relu(h2,w3_reg,bias3_reg,keepProb)
            
            # 4th layer
            w4_reg = self.weight_variable('w4_reg',[nHidden, self.dOutput])
            bias4_reg = self.bias_variable('bias4_reg',[self.dOutput])
            
            y = self.fc(h3,w4_reg,bias4_reg,keepProb)
            
            if isCycle:
                return h3
            else:
                return y
    # ----
        
    # ----
    def train(self, nItr=1000, nBatch=100):
      
        filePeriod = 100
            
        # Start training
        flag = False
        for i in range(nItr):
            
            # Get mini-batch
            batchXY = self.myData.nextBatch(nBatch=nBatch)
            
            feed_dict = {self.x:batchXY[0], self.y:batchXY[1]}
         
            # Change nankai date
            if i % filePeriod == 0:
                nameInds = random.sample(self.nametrInds,3) 
                self.myData.loadTrainTestData(nameInds=nameInds)
            
            # parameter loss
            _, trainPred, trainLoss = self.sess.run([self.opt, self.pred, self.ploss], feed_dict)
            
            # print
            if i % 1000 == 0:
                print('itr: %d, trainLoss:%f' % (i, trainLoss))
    # ----


if __name__ == "__main__":
    
    # command argment ----
    # batch size
    nBatch= int(sys.argv[1])
    # iteration of training
    nItr = int(sys.argv[2])
    # ----
    
    # path ----
    modelPath = "model"
    figurePath = "figure"
    # ----
    
    # parameters ----
    nCell = 5
    nWindow = 10
    dInput = nCell*nWindow
    dOutput = 3
    # Learning rate
    keepProbTrain=1.0
    lr = 1e-3
    # ----
    
    # Training ----
    model = ParamNN(keepProbTrain=keepProbTrain, lr=lr, dInput=dInput, dOutput=dOutput,
                    nCell=nCell, nWindow=nWindow)
    model.train(nItr=nItr, nBatch=nBatch)
    # ----
    
    
    
    
 