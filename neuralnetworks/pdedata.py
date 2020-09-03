# -*- coding: utf-8 -*-

import os
import glob
import pdb
import pickle

import numpy as np
import scipy.io
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import tensorflow as tf

import plot

np.random.seed(1234)


class pdeData:
    def __init__(self, pdeMode='test', dataMode='test'):
        '''
        datamode: small(256/100) or middle(256/50) or large(256/10)
                  Num of X.
        '''
        
        
        self.pdeMode = 'burgers'
        self.modelPath = 'model'
        self.dataMode = dataMode
    
    # ---- 
    def burgers(self, nu=0.01):
        '''
        # viscosity (parameter)
        nu = 0.01 (default)
        '''
        
        # datapoint
        # Num.of time
        tNum = 100 
        # range of time
        tMin = 0.0
        tMax = 1.0
        # time
        t = np.linspace(tMin, tMax, tNum) # [100,]
        
        # Num.of space
        xNum = 256
        # range of space
        xMin = 0.0 # xMin > 0
        xMax = 2.0 * np.pi
        # space
        x = np.linspace(xMin, xMax, xNum) # [256,]
        # observation
        obsu = np.zeros([xNum, tNum])

        for j in range (0, tNum):
            for i in range (0, xNum):
                a = ( x[i] - 4.0 * t[j] )
                b = ( x[i] - 4.0 * t[j] - 2.0 * np.pi )
                c = 4.0 * nu * ( t[j] + 1.0 )
                #pdb.set_trace()
                phi = np.exp ( - a * a / c ) + np.exp ( - b * b / c )
                dphi = - 2.0 * a * np.exp ( - a * a / c ) / c \
                       - 2.0 * b * np.exp ( - b * b / c ) / c
                
                obsu[i,j] = 4.0 - 2.0 * nu * dphi / phi
        #pdb.set_trace()
        return x, t, obsu
    # ---- 
    
    # ----
    def saveXTU(self):
        '''
        minnu, maxnu, swnu 変更で、データ量・データ範囲変更可能.
        '''
        # default
        defaultnu = 0.01
        minnu = 0.005 # minnu < 0.0004　は対応できません(理由不明)
        maxnu = 5.001
        swnu = 0.001
        #swnu = 1.0
        # 一様分布
        samplenu = np.arange(minnu, maxnu, swnu)
        
        cnt = 0
        flag = False
        for nu in samplenu:
            cnt += 1
            print(cnt)
            # x:[256,] t:[100,] obsu:[256,100]
            x, t, obsu = self.burgers(nu=nu)
            
            if not flag:
                X = x
                T = t
                U = obsu.T[np.newaxis]
                NU = np.array([nu])
                flag = True
            else:
                X = np.vstack([X, x]) # [data, 256]
                T = np.vstack([T, t]) # [data, 100]
                U = np.vstack([U, obsu.T[np.newaxis]]) # [data, 100, 256]
                NU = np.hstack([NU, np.array([nu])]) # [data,]
        
        # X,T: 全データ数分同じデータconcat
        with open(os.path.join(self.modelPath, self.pdeMode, 'XTUNU.pkl'), 'wb') as fp:
            pickle.dump(X, fp)
            pickle.dump(T, fp)
            pickle.dump(U, fp)
            pickle.dump(NU, fp)
    # ----
        
    
    # ----
    def maketraintest(self):
        
        ind = np.ones(4996, dtype=bool)
        # train data index
        trainidx = np.random.choice(4996, int(4996*0.8), replace=False).tolist()
        ind[trainidx] = False
        vec = np.arange(4996)
        # test data index
        testidx = vec[ind]
        
        imgspath = glob.glob(os.path.join('model','burgers_small_01','IMG*'))
        
        flag = False
        with open(imgpath, 'rb') as fp:
            X = pickle.load(fp)
            T = pickle.load(fp)
            U = pickle.load(fp)
            NU = pickle.load(fp)
        
        trU = U[trainidx,:]
        teU = U[testidx,:]
        trNU = NU[trainidx]
        teNU = NU[testidx]
        #pdb.set_trace() 
        
        with open(os.path.join(self.modelPath, self.pdeMode, f'IMGtrainXTUNU_small_01.pkl'), 'wb') as fp:
            pickle.dump(X, fp)
            pickle.dump(T, fp)
            pickle.dump(trU, fp)
            pickle.dump(trNU, fp)
            
        with open(os.path.join(self.modelPath, self.pdeMode, f'IMGtestXTUNU_small_01.pkl'), 'wb') as fp:
            pickle.dump(X, fp)
            pickle.dump(T, fp)
            pickle.dump(teU, fp)
            pickle.dump(teNU, fp)
    
    # ----    
    def traintest(self):
        
        # train data
        with open(os.path.join(self.modelPath, self.pdeMode, f'IMGtrainXTUNU_{self.dataMode}_01.pkl'), 'rb') as fp:
            self.trainX = pickle.load(fp)
            self.trainT = pickle.load(fp)
            self.trainU = pickle.load(fp)
            self.trainNU = pickle.load(fp)
        
        # test data
        # testX,testT: 同じX,Tがtestデータ分
        with open(os.path.join(self.modelPath, self.pdeMode, f'IMGtestXTUNU_{self.dataMode}_01.pkl'), 'rb') as fp:
            testX = pickle.load(fp)
            testT = pickle.load(fp)
            testU = pickle.load(fp)
            testNU = pickle.load(fp)
        
        #pdb.set_trace() 
        return testX, testT, testU, testNU
    # ----

    # ----
    def makeImg(self,x,t,u,label='test',name='large'):
            
        X, T = np.meshgrid(x,t) #[100,256]
        #pdb.set_trace() 
        X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None])) # [25600,2]
        u_star = u.flatten()[:,None] # [25600,1]         
    
   
        U_star = griddata(X_star, u_star.flatten(), (X, T), method='cubic')
    
        img = plt.imshow(U_star.T, interpolation='nearest', cmap='rainbow', 
                      extent=[t.min(), t.max(), x.min(), x.max()], 
                      origin='lower', aspect='auto')
        
        plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
        plt.tick_params(bottom=False,left=False,right=False,top=False)
        plt.axis('off')
        
        plt.savefig(os.path.join('figure',f'burgers_{name}',f'{label}.png'))
        
    # ----


     # ----
    def weight_variable(self,name,shape,trainable=True):
         return tf.compat.v1.get_variable(name,shape,initializer=tf.random_normal_initializer(stddev=0.1),trainable=trainable)
    # ----
    # ----
    def bias_variable(self,name,shape,trainable=True):
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
    def CNNfeature(self, x, reuse=False, trainable=True):
        
        with tf.compat.v1.variable_scope("CNN") as scope:
            if reuse:
                scope.reuse_variables()
    
            # 1st conv layer
            w1 = self.weight_variable('w1', [5,5,1,24])
            b1 = self.bias_variable('b1', [24])
            conv1 = self.conv2d(x, w1, b1, strides=2)
        
            conv1 = self.maxpool2d(conv1)
        
            # 2nd conv layer
            w2 = self.weight_variable('w2', [5,5,24,24])
            b2 = self.bias_variable('b2', [24])
            conv2 = self.conv2d(conv1, w2, b2, strides=2)
        
            conv2 = self.maxpool2d(conv2)
            
            #pdb.set_trace() 
            
            #w3 = self.weight_variable('w3', [24*32*24, 32])
            w3 = self.weight_variable('w3', [conv2.get_shape().as_list()[1]*conv2.get_shape().as_list()[2]*conv2.get_shape().as_list()[3], 32], trainable=trainable)
            b3 = self.bias_variable('b3', [32], trainable=trainable)
            
            # 1st full-layer
            reshape_conv2 = tf.reshape(conv2, [-1, w3.get_shape().as_list()[0]])
            
            fc1 = self.fc_relu(reshape_conv2,w3,b3)
            
            w4 = self.weight_variable('w4', [32, 32], trainable=trainable)
            b4 = self.bias_variable('b4', [32], trainable=trainable)
            fc2 = self.fc_relu(fc1,w4,b4)
            
            return fc2
    # ----
    
    # ----
    def nextBatch(self, index):
        
        #pdb.set_trace()

        batchX = self.trainX
        batchT = self.trainT
        batchU = self.trainU[index]
        batchNU = self.trainNU[index]
        
        return batchX, batchT, batchU, batchNU
    # ----
    

#myData = pdeData(dataMode='small')
#myData.maketraintest()

'''
num1 = 0
num2 = 500
name = 'small'

with open(os.path.join('model','burgers','XTUNU.pkl'), 'rb') as fp:
    X = pickle.load(fp)
    T = pickle.load(fp)
    U = pickle.load(fp)
    NU = pickle.load(fp)

# making space X (large, middle small)
if name == 'large':
    xSize = 25
elif name == 'middle':
    xSize = 20
elif name == 'small':
    xSize = 2
#pdb.set_trace()
idx = np.random.choice(X[0].shape[0], xSize, replace=False)
print(name)
flag = False
for i in np.arange(NU.shape[0]):
    #pdb.set_trace()
    label = NU[i].astype(str)
    print(label)

    if not flag:
        Us = U[i,:,idx][np.newaxis] # [x,t]
        flag = True
    else:
        Us = np.vstack([Us,U[i,:,idx][np.newaxis]])

with open(os.path.join('model',f'burgers_{name}_01',f'IMGXTUNU_{name}.pkl'), 'wb') as fp:
    pickle.dump(X[0,idx,np.newaxis], fp)
    pickle.dump(T[0,:,np.newaxis], fp)
    pickle.dump(Us, fp)
    pickle.dump(NU, fp)
'''
#myPlot = plot.Plot(figurepath='figure', trialID=0)
#myPlot.udata(xt, trainXY, testXY[1], testXY, testXY)

