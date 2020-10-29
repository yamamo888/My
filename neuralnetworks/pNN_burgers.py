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
        
        # [100,1] [xDim,1] [100,1] [data,xDim,100,1]
        self.alltestX, self.testX, self.testT, self.testU, self.testNU, self.idx = self.myData.traintest()
        #pdb.set_trace() 
        #np.savetxt(os.path.join('model','params',f'pNNexacttest_{dataMode}.txt'), self.testNU, fmt='%2f')
        # ----        

        # Placeholder ----
        # output param b
        self.y = tf.compat.v1.placeholder(tf.float32,shape=[None, self.yDim])
        # input u
        self.inobs = tf.compat.v1.placeholder(tf.float32,shape=[None, self.xDim, tDim, 1])
        # ----
        
        # neural network ----
        self.predy = self.lambdaNN(self.inobs, rate=rateTrain)
        self.predy_test = self.lambdaNN(self.inobs, reuse=True)
        # ----

        # loss ----
        # param loss
        #self.loss = tf.reduce_mean(tf.square(self.y - self.predy_))
        #self.loss_test = tf.reduce_mean(tf.square(self.y - self.predy_test_))
        self.loss = tf.reduce_mean(tf.square(self.y - self.predy))
        self.loss_test = tf.reduce_mean(tf.square(self.y - self.predy_test))
        # ----

        self.gradnu = tf.gradients(self.loss, self.inobs)
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
    def weight_variable(self, name, shape, trainable=True):
         return tf.compat.v1.get_variable(name,shape,initializer=tf.random_normal_initializer(stddev=0.1),trainable=trainable)
    # ----
    # ----
    def bias_variable(self, name, shape, trainable=True):
         return tf.compat.v1.get_variable(name,shape,initializer=tf.constant_initializer(0.1),trainable=trainable)
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
    # 0.005 < x < 0.304 (for moving burgers)
    def outputthes(self, x):
        
        overlambda = tf.constant([0.304])
        overid = tf.where(x>overlambda)
        overnum = tf.shape(overid)[0]
        overths = tf.tile(overlambda, [overnum])
        overupdate = tf.tensor_scatter_nd_update(x, overid, overths)
        
        updatex = self.underupdate(overupdate)
        
        return updatex
    # ----
    # ----
    def underupdate(self, xover):
        
        underlambda = tf.constant([0.005])
        underid = tf.where(xover<underlambda)
        undernum = tf.shape(underid)[0]
        underths = tf.tile(underlambda, [undernum])
        underupdate = tf.tensor_scatter_nd_update(xover, underid, underths)
        
        return underupdate
    # ----
    
    # ----
    def lambdaNN(self, x, rate=0.0, reuse=False, trainable=True):
        
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
        
            #conv1 = tf.nn.dropout(conv1, rate=0.4)
            
            # 2nd conv layer
            w2 = self.weight_variable('w2', [3,3,nHidden1,nHidden2], trainable=trainable)
            b2 = self.bias_variable('b2', [nHidden2], trainable=trainable)
            conv2 = self.conv2d(conv1, w2, b2, strides=1)
        
            conv2 = self.maxpool2d(conv2)
            
            #conv2 = tf.nn.dropout(conv2, rate=0.4)
            
            # 3nd conv layer
            w3 = self.weight_variable('w3', [3,3,nHidden2,nHidden3], trainable=trainable)
            b3 = self.bias_variable('b3', [nHidden3], trainable=trainable)
            conv3 = self.conv2d(conv2, w3, b3, strides=1)
            
            #conv3 = tf.nn.dropout(conv3, rate=0.4)
            
            # 4nd conv layer
            w4 = self.weight_variable('w4', [3,3,nHidden3,nHidden4], trainable=trainable)
            b4 = self.bias_variable('b4', [nHidden4], trainable=trainable)
            conv4 = self.conv2d(conv3, w4, b4, strides=1)
        
            conv4 = self.maxpool2d(conv4)
            
            #conv4 = tf.nn.dropout(conv4, rate=0.4)
          
            w5 = self.weight_variable('w5', [conv4.get_shape().as_list()[1]*conv4.get_shape().as_list()[2]*conv4.get_shape().as_list()[3], 
                                             nHidden5], trainable=trainable)
            b5 = self.bias_variable('b5', [nHidden5], trainable=trainable)
            
            # 1st full-layer
            reshape_conv4 = tf.reshape(conv4, [-1, w5.get_shape().as_list()[0]])
            
            fc1 = self.fc_relu(reshape_conv4,w5,b5)

            #fc1 = tf.nn.dropout(fc1, rate=0.8)
            
            # 2nd full-layer
            w6 = self.weight_variable('w6', [nHidden5,self.yDim], trainable=trainable)
            b6 = self.bias_variable('b6', [self.yDim], trainable=trainable)
           
            y = self.fc(fc1,w6,b6)
            
            # 0.005 < y < 0.304
            # あかん
            #y = self.outputthes(y)
            
            return y 
    # ----
    
    
    # ----
    def train(self, nItr=1000):
        
        # parameters ----
        testPeriod = 5
        savemodelPeriod = 100
        batchCnt = 0
        nTrain = 235
        batchRandInd = np.random.permutation(nTrain)
        # ----
        
        # Start training
        trPL,tePL = [],[]
        trUL,teUL = [],[]
        
        flag = False
        for itr in range(nItr):
            
            # index
            sInd = self.nBatch * batchCnt
            eInd = sInd + self.nBatch
            index = batchRandInd[sInd:eInd]
            # Get train data
            # [x.shape], [100,], [nbatch, t.shape, x.shape]
            batchX, batchT, batchU, batchNU = self.myData.miniBatch(index)
            # y: prameter b
            #feed_dict = {self.y:batchNU[:,np.newaxis], self.inobs:batchU}
            feed_dict = {self.y:batchNU[:,np.newaxis], self.inobs:batchU[:,:,:,np.newaxis]}
            
            _, trainParam, trainLoss, trainGrad =\
            self.sess.run([self.opt, self.predy, self.loss, self.gradnu], feed_dict)
            
            if eInd + self.nBatch > nTrain:
                batchCnt = 0
                batchRandInd = np.random.permutation(nTrain)
            else:
                batchCnt += 1
            
            # Test & Varidation
            if itr % testPeriod == 0:

                # train return nu -> u
                #params = [self.testX, self.testT, trainParam]
                params = [self.alltestX, batchT, trainParam, self.idx]
                #pdb.set_trace()
                invU = self.myPlot.paramToU2(params)
                #pdb.set_trace() 
                trainULoss = np.mean(np.mean(np.mean(np.square(batchU - invU),2),1))

                self.test(itr=itr)
                
                print('----')
                print('itr: %d, trainLoss:%f, trainULoss:%f' % (itr, trainLoss, trainULoss))
                print('grad {:.16f}'.format(np.mean(trainGrad)))
                
                #print(trainParam[:10])
                #print(batchNU[:10])
                #print(np.mean(trainGrad))
                #print(f'train exact: {batchNU[:5]}')
                #print(f'train pred: {trainParam[:5]}')
                
                # param loss
                trPL = np.append(trPL,trainLoss)
                tePL = np.append(tePL,self.testLoss)
                
                # invu loss
                trUL = np.append(trUL, trainULoss)
                teUL = np.append(teUL, self.testULoss)

                if not flag:
                    trP = trainParam
                    teP = self.testParam
                    flag = True
                else:
                    trP = np.hstack([trP, trainParam])
                    teP = np.hstack([teP, self.testParam])

            if itr % savemodelPeriod == 0:
                # Save model
                self.saver.save(self.sess, os.path.join('model', f'{dataMode}burgers', f'first_{dataMode}'), global_step=itr)
        #pdb.set_trace()
        paramloss = [trPL,tePL]
        uloss = [trUL,teUL]
        predparams = [trP, teP]
        
        teparams = [self.alltestX, self.testT, self.testParam, self.testNU]
        
        
        return paramloss, uloss, teparams, predparams
    # ----
    
    # ----
    def test(self,itr=0):
        feed_dict={self.y: self.testNU, self.inobs:self.testU}    
         
        #pdb.set_trace() 
        self.testParam, self.testLoss =\
        self.sess.run([self.predy_test, self.loss_test], feed_dict)
        
        # return nu -> u
        params = [self.alltestX, self.testT, self.testParam, self.idx]
        invU = self.myPlot.paramToU2(params)
        
        self.testULoss = np.mean(np.mean(np.mean(np.square(self.testU[:,:,:,0] - invU),2),1))
        
        if np.isnan(self.testULoss):
            pdb.set_trace()

        # plot ----
        printnus = [0.005, 0.01, 0.02, 0.05, 0.1, 0.3]
        printindex = [i for i,e in enumerate(self.testNU) if np.round(e,3) in printnus]

        exactnus = self.testNU[printindex]
        prednus = self.testParam[printindex]
        
        #pdb.set_trace()

        self.myPlot.plotExactPredParam([self.alltestX, self.testT, prednus, exactnus, self.idx], itr=itr, savename='tepredparam')

        print('itr: %d, testPLoss:%f, testULoss:%f' % (itr, self.testLoss, self.testULoss))
        #pdb.set_trace()
        #print(f'test exact: {self.testNU[:5]}')
        #print(f'test pred: {self.testParam[:5]}')
    # ----
    
     
if __name__ == "__main__":
    
    # command argment ----
    parser = argparse.ArgumentParser()

    # iteration of training
    parser.add_argument('--nItr', type=int, default=100)
    # Num of mini-batch
    parser.add_argument('--nBatch', type=int, default=25)
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
    plosses, ulosses, teparams, params  = model.train(nItr=nItr)
    # ----
    
    # Plot ----
    model.myPlot.Loss(plosses, labels=['train','test'], savename='pNN_param')
    model.myPlot.Loss(ulosses, labels=['train','test'], savename='pNN_u')
    #model.myPlot.plotExactPredParam(teparams, xNum=teparams[0].shape[0], savename='lasttepredparam')
    # ----

    # save txt ----
    #np.savetxt(os.path.join('model','params',f'pNNtrain_{dataMode}{trialID}.txt'), params[0], fmt='%2f')
    #np.savetxt(os.path.join('model','params',f'pNNtest_{dataMode}{trialID}.txt'), params[1], fmt='%2f')
    # ----
    
 
    
