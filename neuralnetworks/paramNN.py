# -*- coding: utf-8 -*-

import sys
import os

import numpy as np
import tensorflow as tf

import random
import pickle
import pdb

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt

import data
import plot

class ParamNN:
    def __init__(self, rateTrain=0.0, lr=1e-3, dInput=50, dOutput=3,
                 nCell=5, nWindow=10, isCycle=False):
        
        # parameter ----
        # Select train data *pkl
        self.nametrInds = [0,1,2,3,4,5,6,7]
        # random sample loading train data
        nameInds = random.sample(self.nametrInds,3)
        self.dInput = dInput
        self.dOutput = dOutput
        # ----
      
        
        # Dataset ----
        if isCycle == False:
            self.myData = data.NankaiData(nCell=nCell, nWindow=nWindow)
            self.xTest, self.yTest = self.myData.loadTrainTestData(nameInds=nameInds)
        # ----
            # Placeholder ----
            self.x = tf.compat.v1.placeholder(tf.float32,shape=[None, self.dInput])
            self.y = tf.compat.v1.placeholder(tf.float32,shape=[None, self.dOutput])
            # ----
            
            # neural network ----
            self.pred = self.Regress(self.x, rate=rateTrain, trainable=True)
            self.pred_test = self.Regress(self.x, rate=rateTrain, reuse=True, trainable=True)
            # ----
            
            # loss ----
            self.loss = tf.reduce_mean(tf.square(self.y - self.pred))
            self.loss_test = tf.reduce_mean(tf.square(self.y - self.pred_test))
            # ----
    
            Vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,scope='Regress') 
            self.opt = tf.train.AdamOptimizer(lr).minimize(self.loss,var_list=Vars)
            
            config = tf.compat.v1.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1,allow_growth=True)) 
            self.sess = tf.compat.v1.Session(config=config)
            self.sess.run(tf.global_variables_initializer())
            
            # save model ----
            #saver = tf.train.Saver()
            saver = tf.compat.v1.train.Saver()
            saver.save(self.sess, os.path.join('model', 'pNN', 'first'))
            # ----
                    
    
    # ----
    def weight_variable(self, name, shape, trainable=False):
         return tf.compat.v1.get_variable(name,shape,initializer=tf.random_normal_initializer(stddev=0.1), trainable=trainable)
    # ----
    # ----
    def bias_variable(self, name, shape, trainable=False):
         return tf.compat.v1.get_variable(name,shape,initializer=tf.constant_initializer(0.1), trainable=trainable)
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
    def Regress(self, x_r, rate=0.0, reuse=False, isCycle=False, trainable=False):
        
        nHidden=128
        
        with tf.compat.v1.variable_scope('Regress') as scope:  
            if reuse:
                scope.reuse_variables()
            #pdb.set_trace()
            # 1st layer
            w1_reg = self.weight_variable('w1_reg',[self.dInput, nHidden], trainable=trainable)
            bias1_reg = self.bias_variable('bias1_reg',[nHidden], trainable=trainable)
            h1 = self.fc_relu(x_r,w1_reg,bias1_reg,rate)
            
            # 2nd layer
            w2_reg = self.weight_variable('w2_reg',[nHidden, nHidden], trainable=trainable)
            bias2_reg = self.bias_variable('bias2_reg',[nHidden], trainable=trainable)
            h2 = self.fc_relu(h1,w2_reg,bias2_reg,rate)
            
            # 3rd layer 
            bias4_reg = self.bias_variable('bias4_reg',[self.dOutput])
            w3_reg = self.weight_variable('w3_reg',[nHidden, nHidden], trainable=trainable)
            bias3_reg = self.bias_variable('bias3_reg',[nHidden], trainable=trainable)
            h3 = self.fc_relu(h2,w3_reg,bias3_reg,rate)
            
            if isCycle:
                return h3
            else:
                # 4th layer
                w4_reg = self.weight_variable('w4_reg',[nHidden, self.dOutput], trainable=trainable)
                bias4_reg = self.bias_variable('bias4_reg',[self.dOutput], trainable=trainable)
            
                y = self.fc_relu(h3,w4_reg,bias4_reg,rate)
            
                return y
    # ----
        
    # ----
    def train(self, nItr=1000, nBatch=100):
      
        filePeriod = 100
        testPeriod = 100
            
        # Start training
        flag = False
        trL,teL = np.zeros(int(nItr/testPeriod)),np.zeros(int(nItr/testPeriod))

        for i in range(nItr):
            
            # Get mini-batch
            batchXY = self.myData.nextBatch(nBatch=nBatch)
            
            feed_dict = {self.x:batchXY[0], self.y:batchXY[1]}
         
            # Change nankai date
            if i % filePeriod == 0:
                nameInds = random.sample(self.nametrInds,3) 
                self.myData.loadTrainTestData(nameInds=nameInds)
            
            # parameter loss
            _, trainPred, trainLoss = self.sess.run([self.opt, self.pred, self.loss], feed_dict)
            
            # Test
            if i % testPeriod == 0:

                self.test()

                print('itr: %d, trainLoss:%f, testLoss:%f' % (i, trainLoss, self.testLoss))
                
                trL[int(i/testPeriod)] = trainLoss
                teL[int(i/testPeriod)] = self.testLoss

        losses = [trL,teL]
        params = [self.yTest,self.testPred]

        return losses, params
    # ----
    
    # ----
    def test(self):
        
        # 1. pred fric paramters
        feed_dict={self.x:self.xTest, self.y:self.yTest}
        self.testPred, self.testLoss = self.sess.run([self.pred_test, self.loss_test], feed_dict)
        
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
    rateTrain=0.0
    lr = 1e-3
    # ----
    
    # Training ----
    model = ParamNN(rateTrain=rateTrain, lr=lr, dInput=dInput, dOutput=dOutput,
                    nCell=nCell, nWindow=nWindow)
    losses, params = model.train(nItr=nItr, nBatch=nBatch)
    # ----

    # Plot ----
    myPlot = plot.Plot()
    myPlot.Loss(losses, labels=['train','test'])
    # ----

    # Save ----
    with open(os.path.join(modelPath,'pNN','testY_exactpred.pkl'), 'wb') as fp:
        pickle.dump(params[0], fp)
        pickle.dump(params[1], fp)
    
    
    
    
 
