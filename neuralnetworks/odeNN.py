# -*- coding: utf-8 -*-

import sys
import os

import numpy as np
import tensorflow as tf

import random
import pickle
import pdb

import matplotlib.pylab as plt

import data
import plot

class ParamCycleNN:
    def __init__(self, rateTrain=0.0, lr=1e-3, nCell=5, trialID=0):
        
        # path ----
        self.modelPath = 'model'
        self.figurePath = 'figure'
        # ----
        
        # parameter ----
        dInput = 6
        self.dOutput = 3
        self.trialID = trialID
        # ----
            
        # func ----
        # data
        self.myData = data.NankaiData()
        # Test data, interval/seq/onehotyear/b 
        self.xTest, self.seqTest, self.yYearTest, self.yTest = self.myData.TrainTest()
        # Eval data nankai rireki
        self.xEval, self.seqEval, self.yYearEval, self.yEval = self.myData.Eval()
        
        # plot
        self.myPlot = plot.Plot(figurepath=figurePath, trialID=trialID)
        # ----
        
        # Placeholder ----
        # interval
        # pred paramb + vt-1
        self.odex = tf.compat.v1.placeholder(tf.float32,shape=[None, None, dInput])
        # vt
        self.odey = tf.compat.v1.placeholder(tf.float32,shape=[None, self.dOutput])
        # ----
        
        # neural network ----
        self.Vt = self.odeNN(self.odex)
        self.Vt_test = self.odeNN(self.odex, reuse=True)
        # ----
        
        # loss ----
        self.odeloss = tf.square(self.odey - self.Vt)
        self.odeloss_test = tf.square(self.odey - self.Vt_test)
        # ----
        #pdb.set_trace()
        # optimizer ----
        odeVars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,scope='odeNN') 
        self.optODE = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.odeloss, var_list=odeVars)
        # ----
        
        # ----
        config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1,allow_growth=True)) 
        self.sess = tf.compat.v1.Session(config=config)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver()
        # ----
    
    # ----
    def weight_variable(self,name,shape,trainable=True):
         return tf.compat.v1.get_variable(name,shape,initializer=tf.random_normal_initializer(stddev=0.1),trainable=trainable)
    # ----
    # ----
    def bias_variable(self,name,shape,trainable=True):
         return tf.compat.v1.get_variable(name,shape,initializer=tf.constant_initializer(0.1),trainable=trainable)
    # ----
    def fc_sigmoid(self,inputs,w,b,rate=0.0):
        sigmoid = tf.matmul(inputs,w) + b
        sigmoid = tf.nn.dropout(sigmoid, rate=rate)
        sigmoid = tf.nn.sigmoid(sigmoid)
        return sigmoid
    # ----
    def fc_relu(self,inputs,w,b,rate=0.0):
         relu = tf.matmul(inputs,w) + b
         relu = tf.nn.dropout(relu, rate=rate)
         relu = tf.nn.relu(relu)
         return relu 
    # ----
    def fc(self,inputs,w,b,rate=0.0):
         fc = tf.matmul(inputs,w) + b
         fc = tf.nn.dropout(fc, rate=rate)
         return fc
    # ----

    # ----
    def odeNN(self, x, reuse=False):
        '''
        2 layears LSTM
        input -> 1st LSTM(GRU) -> fc -> LSTM(GRU) -> output
        '''

        nHidden = 10
        
        with tf.compat.v1.variable_scope("odeNN") as scope:
            if reuse:
                scope.reuse_variables()
            
            # 1st LSTM
            cell = tf.compat.v1.nn.rnn_cell.GRUCell(nHidden)
            #cell = tf.compat.v1.nn.rnn_cell.LSTMCell(nHidden, use_peepholes=True)
            # 1st lstm
            lstm_output, _ = tf.compat.v1.nn.dynamic_rnn(cell=cell, inputs=x, dtype=tf.float32)
            
            weight = self.weight_variable('weight1', [nHidden,nHidden])
            bias = self.bias_variable('bias1', [nHidden])
            pdb.set_trace()
            # hidden
            h = tf.matmul(lstm_output[-1], weight) + bias
            
            # 2nd lstm
            lstm_output, _ = tf.compat.v1.nn.dynamic_rnn(cell=h, inputs=x, dtype=tf.float32)
            
            weight = self.weight_variable('weight', [nHidden,self.dOutput])
            bias = self.bias_variable('bias', [self.dOutput])
            
            # output
            y = tf.matmul(lstm_output[-1], weight) + bias
            
            return y
    # ----
                     
    # ----
    def train(self, nItr=10000):
        #pdb.set_trace()
        
        # parameters ----
        nYear = 8000
        testPeriod = 3
        nBatch = 5
        batchCnt = 0
        #nTrain = 80068
        nTrain = 100
        batchRandInd = np.random.permutation(nTrain)
        
        V0 = np.array([0.0, 0.0, 0.0])
        # ----
        
        # for loss
        trL,teL = [],[]
        for itr in np.arange(nItr):
            
            # mini-batch ----
            sInd = nBatch * batchCnt
            eInd = sInd + nBatch
            index = batchRandInd[sInd:eInd]
            # Get train data x(interval), seq, y(param), year(onehot)
            batchXY = self.myData.nextBatch(index)
            # ----
            
            print('----')
            
            trCL = []    
            trC = np.zeros([nBatch, nYear, self.dOutput])
            # ODE LSTM
            for nbatch in np.arange(nBatch):
                
                #pdb.set_trace()
                flag = False
                for num in np.arange(nYear):
                    # v0
                    if num == 0:
                        # batchXY[2]:b
                        x = np.hstack([batchXY[2][nbatch], V0])[np.newaxis,np.newaxis]
                    # vt
                    else:
                        x = np.hstack([batchXY[2][nbatch], predVt[0]])[np.newaxis,np.newaxis]
                    
                    feed_dict = {self.odex:x, self.odey:batchXY[-1][nbatch,num,np.newaxis]}
                    
                    _, predVt, trainLoss =\
                    self.sess.run([self.optODE, self.Vt, self.odeloss], feed_dict)
            
                    if not flag:
                        # all year loss v1 + v2, ..., + vt loss
                        trLs = np.sum(trainLoss)
                        # predVt v1, ..., vt
                        predVts = predVt
                        flag = True
                    else:
                        trLs = np.hstack([trLs, np.sum(trainLoss)])
                        predVts = np.vstack([predVts, predVt])
                
                print('itr:%d, nbatch:%d, trainlstmLoss:%3f' % (itr, nbatch, np.sum(trLs)))
                
                # each itr loss, [nbatch,]
                trCL = np.append(trCL, np.sum(trLs))
                # pred rireki, [nbatch,]
                trC[nbatch] = predVts
    
            # Test & Evaluation
            if itr % testPeriod == 0:
                self.test(itr=itr)
                self.eval(itr=itr)
                
                print('====')  
                print('itr:%d, nbatch:%d, trainLoss:%3f, testLoss:%3f' % (itr, nbatch, np.mean(trCL), np.mean(self.teCL)))
                print('====')
                # total loss [itr,]
                trL = np.append(trL, np.mean(trCL))
                teL = np.append(teL, np.mean(self.teCL))
                
                # rireki plot ----
                minC = trC[np.argmin(trCL)]
                maxC = trC[np.argmax(trCL)]
                mingt = batchXY[2][np.argmin(trCL)]
                maxgt = batchXY[2][np.argmax(trCL)]
                
                self.myPlot.scatterRireki(mingt, minC, path=f'train{self.trialID}', label=f'min_{itr}_{min(trCL)}')
                self.myPlot.scatterRireki(maxgt, maxC, path=f'train{self.trialID}', label=f'max_{itr}_{max(trCL)}')
                # ----
                
                # rireki plot (test) ----
                C0 = self.teC[0]
                #C50 = self.teCL[50]
                #C100 = self.teCL[100]
                
                gt0 = self.yCycleTest[0]
                #gt50 = self.yCycleTest[50]
                #gt100 = self.yCycleTest[100]
                
                self.myPlot.scatterRireki(gt0, C0, path=f'test{self.trialID}', label=f'0_{itr}_{min(trCL)}')
                #self.myPlot.scatterRireki(gt50, C50, path=f'test{self.trialID}', label=f'50_{itr}_{max(trCL)}')
                #self.myPlot.scatterRireki(gt100, C100, path=f'test{self.trialID}', label=f'100_{itr}_{max(trCL)}')
                # ----
                
                # rireki plot (eval nankairireki) ----
                self.myPlot.scatterRireki(self.yYearEval, self.evC, path=f'eval{self.trainID}', label=f'{itr}')
                # ----
                
                # Save model
                self.saver.save(self.sess, os.path.join('model', 'odeNN', 'first'), global_step=itr)
        
        # train & test loss
        losses = [trL,teL]
    
        return losses
    # ----
    
    # ----  
    def test(self, itr=0):
        
        # parameters ----
        nYear = 8000
        V0 = np.array([0.0, 0.0, 0.0])
        # ----
        
        self.teCL = []
        self.teC = np.zeros([self.yTest.shape[0], nYear, self.dOutput])
        
        print('----')
        
        for nbatch in np.arange(self.yTest.shape[0]):
            
            flag = False
            for num in np.arange(nYear):
                if num == 0:
                    x = np.hstack([self.yTest[nbatch], V0])[np.newaxis,np.newaxis]
                else:
                    x = np.hstack([self.yTest[nbatch], predVt[0]])[np.newaxis,np.newaxis]
            
                feed_dict = {self.odex:x, self.odey:self.yYearTest[nbatch,num,np.newaxis]}
                
                predVt, testLoss =\
                self.sess.run([self.Vt_test, self.odeloss], feed_dict)
                  
                if not flag:
                    teLs = np.sum(testLoss)
                    predVts = predVt
                    flag = True
                else:
                    teLs = np.hstack([teLs, np.sum(testLoss)])
                    predVts = np.vstack([predVts, predVt])
                    
                    
            print('itr:%d, nbatch:%d, testlstmLoss:%3f' % (itr, nbatch, np.sum(teLs)))
            # each itr loss, [nbatch,]
            self.teCL = np.append(self.teCL, np.sum(teLs))
            # pred rireki, [nYear,]
            self.teC[nbatch] = predVts
    # ----
    
    # ----
    def eval(self, itr=0):
        nYear = 1400
        V0 = np.array([0.0, 0.0, 0.0])
        self.evC = np.zeros([nYear, self.dOutput])
        
        flag = False
        for num in np.arange(nYear):
            if num == 0:
                pdb.set_trace()
                x = np.hstack([self.yEval, V0])[np.newaxis]
            else:
                x = np.hstack([self.yEval, predVt[0]])[np.newaxis]
            
            feed_dict = {self.odex:x}
            
            predVt =\
            self.sess.run([self.Vt_eval], feed_dict)
            pdb.set_trace()
            
            if not flag:
                predVts = predVt
                flag = True
            else:
                predVts = np.vstack([predVts, predVt]) # [1400,3]
        
        self.evC = predVts
    # ----
    
if __name__ == "__main__":
    
    # command argment ----
    # iteration of training
    nItr = int(sys.argv[1])
    # trial ID
    trialID = int(sys.argv[2])
    # ----
    
    # path ----
    figurePath = "figure"
    # ----
   
    # parameters ----
    # random sample loading train data
    rateTrain = 0.0
    lr = 1e-2
    # ----
          
    # model ----
    model = ParamCycleNN(rateTrain=rateTrain, lr=lr, trialID=trialID)
    losses = model.train(nItr=nItr)
    # ----
    
    # plot ----
    myPlot = plot.Plot(figurepath=figurePath, trialID=trialID)
    myPlot.pLoss(losses, labels=['eval'])
    # ----
   
    
  
    