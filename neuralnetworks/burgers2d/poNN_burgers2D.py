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
import pdeburgers2d
import burgers2dplot


class ParamNN:
    def __init__(self, rateTrain=0.0, lr=1e-3, index=0, dataMode='test'):
        
      
        # parameter ----
        if dataMode == 'large':
            self.xDim = 25
            self.yDim = 25
        elif dataMode == 'middle':
            self.xDim = 12
            self.yDim = 12
        elif dataMode == 'small':
            self.xDim = 5
            self.yDim = 5
        
        self.tDim = 201
        self.index = index
        self.yDim = 2
        # ----
    
        # Dataset ----
        self.myData = pdedata.pdeData(pdeMode='burgers2d', dataMode=dataMode)
        
        self.alltestX, self.alltestY, self.testT, self.testX, self.testY, self.testU, self.testV, self.testNU, self.idx, self.idy = self.myData.traintest()
        # ----
         
        # Placeholder ----
        # u
        self.inobsu = tf.compat.v1.placeholder(tf.float32,shape=[None, self.xDim, self.yDim, self.tDim, 1])
        self.inobsv = tf.compat.v1.placeholder(tf.float32,shape=[None, self.xDim, self.yDim, self.tDim, 1]) 
        self.inobs = tf.concat([self.inobsu, self.inobsv], -1)
        
        self.outobsu = tf.compat.v1.placeholder(tf.float64,shape=[None, self.xDim, self.yDim, self.tDim])
        self.outobsv = tf.compat.v1.placeholder(tf.float64,shape=[None, self.xDim, self.yDim, self.tDim])
        
        # param nu 
        self.y = tf.compat.v1.placeholder(tf.float32,shape=[None, self.yDim])
        # ----
        
        # Restore neural network ----
        # pred nu [ndata,]
        self.predparam = self.lambdaNN(self.inobs)
        # ----
        
        # for predparam
        self.placeparam = tf.compat.v1.placeholder(tf.float64, shape=[None, self.yDim])
        
        # optimizer ----
        config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1,allow_growth=True)) 
        self.saver = tf.compat.v1.train.Saver()
        self.sess = tf.compat.v1.Session(config=config)
        # ----
        #pdb.set_trace()
        # Restore model ----
        ckptpath = os.path.join('model', f'{dataMode}burgers2d')
        ckpt = tf.train.get_checkpoint_state(ckptpath)
        
        lastmodel = ckpt.model_checkpoint_path
        self.saver.restore(self.sess, lastmodel)
        print('>>> Restore train model')
        # ----
         
        # float32 -> float64
        self.predparam = tf.cast(self.predparam, tf.float64)
        
        # for predparam
        self.placeparam1 = tf.compat.v1.placeholder(tf.float64, shape=[None, 1])
        self.placeparam2 = tf.compat.v1.placeholder(tf.float64, shape=[None, 1])
        
        # PDE ----
        # output: u
        self.predu = pdeburgers2d.burgers(self.placeparam1, self.placeparam2)
        self.predu_first = pdeburgers2d.burgers(self.predparam[0], self.predparam[1])
        # ----
        
        # space data -> [none, self.xDim, t] ----
        self.indx = tf.compat.v1.placeholder(tf.int32,shape=[self.xDim,1])
        self.indy = tf.compat.v1.placeholder(tf.int32,shape=[self.yDim,1])
        # for u
        trans_predu = tf.transpose(self.predu, perm=[1,0,2,3])
        trans_predu_first = tf.transpose(self.predu_first, perm=[1,0,2,3])
        # [100(x),data,y,t]
        gather_predu = tf.gather_nd(trans_predu, self.indx)
        gather_predu_first = tf.gather_nd(trans_predu_first, self.indx)
        # for v
        trans_predv = tf.transpose(gather_predu, perm=[2,1,0,3])
        trans_predv_first = tf.transpose(gather_predu_first, perm=[2,1,0,3])
        
        gather_predv = tf.gather_nd(trans_predv, self.indx)
        gather_predv_first = tf.gather_nd(trans_predv_first, self.indx)
        
        # [data,self.xDim,t]
        tmp_trans = tf.transpose(gather_predv, perm=[1,0,2,3])
        tmp_trans_first = tf.transpose(gather_predv_first, perm=[1,0,2,3])
        
        space_preduv = tf.transpose(tmp_trans, perm=[0,2,1,3])
        space_preduv_first = tf.transpose(tmp_trans_first, perm=[0,2,1,3])
        # ----
        pdb.set_trace()
        # loss param ----
        self.loss1 = tf.reduce_mean(tf.square(tf.cast(self.y[:,0], tf.float64) - self.placeparam1))
        self.loss2 = tf.reduce_mean(tf.square(tf.cast(self.y[:,1], tf.float64) - self.placeparam2))
        self.loss1_first = tf.reduce_mean(tf.square(tf.cast(self.y[:,0], tf.float64) - self.predparam[:,0]))
        self.loss2_first = tf.reduce_mean(tf.square(tf.cast(self.y[:,1], tf.float64) - self.predparam[:,1]))
        # ----
        
        # loss uv ----   
        self.lossuv = tf.reduce_mean(tf.square(self.outobs - space_preduv))
        self.lossuv_first = tf.reduce_mean(tf.square(self.outobs - space_preduv_first))
        # ----
        
        # gradient ----
        self.alpha = tf.compat.v1.placeholder(tf.float64, shape=[1])
        self.grad1 = tf.gradients(self.loss, self.placeparam1)[0]
        self.grad2 = tf.gradients(self.loss, self.placeparam2)[0]
        
        self.grad1_first = tf.gradients(self.loss1_first, self.predparam[:,0])[0] # for first time(use predict parameter)
        self.grad2_first = tf.gradients(self.loss2_first, self.predparam[:,1])[0] # for first time(use predict parameter)
        
        self.nextparam1 = self.placeparam1 - (self.grad1 * self.alpha)
        self.nextparam2 = self.placeparam2 - (self.grad2 * self.alpha)
        
        self.nextparam1_first = self.predparam[:,0] - (self.grad1_first * self.alpha) # for first time
        self.nextparam2_first = self.predparam[:,1] - (self.grad2_first * self.alpha) # for first time
        # ----
        
        # graph ----
        self.run_options = tf.RunOptions(output_partition_graphs=True)
        self.run_metadata = tf.RunMetadata()
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
    def train(self, nItr=1000, alpha=0.01):
       
        grads1,grads2,losses1,losses2,uvlosses,preParam1,preParam2 = [],[],[],[],[],[],[]
        for itr in range(nItr):
            
            if itr == 0:
                # ※dummy for placeparam
                predParam = 0.03
                
                testParam = np.concatenate([np.array([1.0])[None], self.testNU[None,self.index], 0])
                
                feed_dict={self.y:testParam, self.inobsu:self.testU[self.index,None], self.inobsv:self.testV[self.index,None], 
                           self.outobsu:self.testU[self.index,None,:,:,0], self.outobsv:self.testV[self.index,None,:,:,0], 
                           self.indx:self.idx[:,None], self.indy:self.idx[:,None],
                           self.placeparam1:np.array([predParam])[:,None], self.placeparam2:np.array([predParam])[:,None], 
                           self.alpha:np.array([alpha])}
                
                # call *_fitst only NN
                grad1, grad2, nextParam1, nextParam2, loss1, loss2, uvloss =\
                self.sess.run([self.grad1_first, self.grad2_first, self.nextparam1_first, self.nextparam2_first, self.loss1_first, self.loss2_first, self.lossuv_first], feed_dict)
              
                pdb.set_trace()
            
            elif itr > 0:
                # for lambda1, lambda2(nu)
                predParam1 = preParam1[itr-1]
                predParam2 = preParam2[itr-1]
            
                feed_dict={self.y:testParam, self.inobsu:self.testU[self.index,None], self.inobsv:self.testV[self.index,None], 
                           self.outobsu:self.testU[self.index,None,:,:,0], self.outobsv:self.testV[self.index,None,:,:,0],
                           self.indx:self.idx[:,None], self.indy:self.idx[:,None],
                           self.placeparam1:np.array([predParam1])[:,None], self.placeparam2:np.array([predParam2])[:,None],
                           self.alpha:np.array([alpha])}
          
                grad1, grad2, nextParam1, nextParam2, loss1, loss2, uvloss =\
                self.sess.run([self.grad1, self.grad2, self.nextparam1, self.nextparam2, self.loss1, self.loss2, self.lossuv], feed_dict)
                
            preParam1 = np.append(preParam1, nextParam1)
            preParam2 = np.append(preParam2, nextParam2)
            
            grads1 = np.append(grads1, grad1)
            grads2 = np.append(grads2, grad2)
            
            losses1 = np.append(losses1, loss1)
            losses2 = np.append(losses2, loss2)
            
            uvlosses = np.append(uvlosses, uvloss)
            
            print('----')
            print('exact lambda: %.8f predlambda1: %.8f, predlambda2: %.8f' % (self.testNU[self.index], preParam1[itr-1], preParam2[itr-1]))
            print('lambda1 mse: %.10f, lambda2 mse: %.10f' % (loss1, loss2))
            print('v mse: %.10f' % (uvloss))
            print('gradient (closs/param1): %f, (closs/param2): %f' % (grad1, grad2))
    
        return [losses1, losses2], [grads1, grads2]
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
    
    # 引数展開
    args = parser.parse_args()
    
    nItr = args.nItr
    dataMode = args.dataMode
    index = args.index
    alpha = args.alpha
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
    
    llosses, grads = model.train(nItr=nItr, alpha=alpha)
    pdb.set_trace()
    # Plot ----
    myPlot = burgers2dplot.Plot(dataMode=dataMode, trialID=index)
    myPlot.Loss1Data(llosses[0], labels=['test'], savename=f'poNN_testloss1')
    myPlot.Loss1Data(llosses[1], labels=['test'], savename=f'poNN_testloss2')
    
    myPlot.Loss1Data(grads[0], labels=['test'], savename=f'poNN_testgrad1')
    myPlot.Loss1Data(grads[1], labels=['test'], savename=f'poNN_testgrad2')
    
    # ----

