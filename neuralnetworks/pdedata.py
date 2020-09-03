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
        #minnu = 0.05 # minnu < 0.0004　は対応できません(理由不明)
        minnu = 0.01
        #maxnu = 0.015
        maxnu = 5.001
        swnu = 0.001
        
        # 一様分布
        samplenu = np.arange(minnu, maxnu, swnu)
        
        cnt = 0
        flag = False
        for nu in samplenu:
            cnt += 1
            print(cnt)
            # x:[256,] t:[100,] obsu:[256,100]
            x, t, obsu = self.burgers(nu=nu)
            
            if nu == defaultnu:
                with open(os.path.join(self.modelPath, self.pdeMode, 'burgeres_default001.pkl'), 'wb') as fp:
                    pickle.dump(x, fp)
                    pickle.dump(t, fp)
                    pickle.dump(obsu, fp)
                    pickle.dump(nu, fp)
            else:
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
            pickle.dump(X[0], fp) #[256,]
            pickle.dump(T[0], fp) #[100,]
            pickle.dump(U, fp) 
            pickle.dump(NU, fp)
    # ----
        
    # ----
    def maketraintest(self, name):
        
        nData = 4990
        ind = np.ones(nData, dtype=bool)
        # train data index
        trainidx = np.random.choice(nData, int(nData*0.8), replace=False).tolist()
        ind[trainidx] = False
        vec = np.arange(nData)
        # test data index
        testidx = vec[ind]
        
        imgpath = glob.glob(os.path.join('model','burgers',f'IMGXTUNU_{name}.pkl'))
        #pdb.set_trace()
        # space data, 256 -> xDim
        with open(imgpath[0], 'rb') as fp:
            X = pickle.load(fp)
            T = pickle.load(fp)
            U = pickle.load(fp)
            NU = pickle.load(fp)
            
        with open(os.path.join(self.modelPath, self.pdeMode, 'XTUNU.pkl'), 'rb') as fp:
            allX = pickle.load(fp)
            _ = pickle.load(fp)
            allU = pickle.load(fp)
            _ = pickle.load(fp)
        
        trU = U[trainidx,:]
        teU = U[testidx,:]
        allteU = allU[testidx,:]
        trNU = NU[trainidx]
        teNU = NU[testidx]
        
        #pdb.set_trace()
        
        with open(os.path.join(self.modelPath, self.pdeMode, f'IMGtrainXTUNU_{name}.pkl'), 'wb') as fp:
            pickle.dump(X, fp)
            pickle.dump(T, fp)
            pickle.dump(trU, fp)
            pickle.dump(trNU, fp)
            
        with open(os.path.join(self.modelPath, self.pdeMode, f'IMGtestXTUNU_{name}.pkl'), 'wb') as fp:
            pickle.dump(allX, fp)
            pickle.dump(T, fp)
            pickle.dump(allteU, fp)
            pickle.dump(teNU, fp)
            pickle.dump(X, fp)
            pickle.dump(teU, fp)
    # ----
    
    # ----    
    def traintestvaridation(self):
        
        # train data
        with open(os.path.join(self.modelPath, self.pdeMode, f'IMGtrainXTUNU_{self.dataMode}.pkl'), 'rb') as fp:
            self.trainX = pickle.load(fp)
            self.trainT = pickle.load(fp)
            self.trainU = pickle.load(fp)
            self.trainNU = pickle.load(fp)
        
        # test data
        # testX,testT: 同じX,Tがtestデータ分
        with open(os.path.join(self.modelPath, self.pdeMode, f'IMGtestXTUNU_{self.dataMode}.pkl'), 'rb') as fp:
            testX = pickle.load(fp)
            testT = pickle.load(fp)
            testU = pickle.load(fp)
            testNU = pickle.load(fp)
        
        # varidation data
        with open(os.path.join(self.modelPath, self.pdeMode, f'IMGvarXTUNU_{self.dataMode}.pkl'), 'rb') as fp:
            varX = pickle.load(fp)
            varT = pickle.load(fp)
            varU = pickle.load(fp)
            varNU = pickle.load(fp)
    
        pdb.set_trace()
        
        return testX, testT, testU, testNU, varX, varT, varU, varNU
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
            pdb.set_trace() 
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
    def miniBatch(self, index):
        
        #pdb.set_trace()

        batchX = self.trainX
        batchT = self.trainT
        batchU = self.trainU[index]
        batchNU = self.trainNU[index]
        
        return batchX, batchT, batchU, batchNU
    # ----
    
    # ----
    def Batch(self):
        
        batchX = self.trainX
        batchT = self.trainT
        batchU = self.trainU
        batchNU = self.trainNU
        
        return batchX, batchT, batchU, batchNU
    # ----
        
    

#name = 'large'
#myData = pdeData(dataMode='small')
'''
#[1]
myData.saveXTU()

#[2]
print(name)

with open(os.path.join('model','burgers','XTUNU.pkl'), 'rb') as fp:
    X = pickle.load(fp)
    T = pickle.load(fp)
    U = pickle.load(fp)
    NU = pickle.load(fp)

# making space X (large, middle small)
if name == 'large':
    xSize = 50
elif name == 'middle':
    xSize = 25
elif name == 'small':
    xSize = 10

idx = np.random.choice(X.shape[0], xSize, replace=False)

# for train & test ----
flag = False
for i in np.arange(NU.shape[0]):
  
    label = NU[i].astype(str)
    print(label)

    if not flag:
        Us = U[i,:,idx][np.newaxis] # [x,t]
        flag = True
    else:
        Us = np.vstack([Us,U[i,:,idx][np.newaxis]])

with open(os.path.join('model','burgers',f'IMGXTUNU_{name}.pkl'), 'wb') as fp:
    pickle.dump(X[idx,np.newaxis], fp)
    pickle.dump(T[:,np.newaxis], fp)
    pickle.dump(Us, fp)
    pickle.dump(NU, fp)
# ----
        
# for varidation ----
with open(os.path.join('model','burgers','burgeres_default001.pkl'), 'rb') as fp:
    varX = pickle.load(fp) #[256,]
    varT = pickle.load(fp) #[100,]
    varU = pickle.load(fp) #[256,100]
    varNU = pickle.load(fp) # 0.01
    
with open(os.path.join('model','burgers',f'IMGvarXTUNU_{name}.pkl'), 'wb') as fp:
    pickle.dump(varX[:,np.newaxis], fp)
    pickle.dump(varT[:,np.newaxis], fp)
    pickle.dump(varU, fp)
    pickle.dump(varNU, fp)
    pickle.dump(varX[idx,np.newaxis], fp)
    pickle.dump(varU[idx], fp)
# ----
'''
#[3]
#myData.maketraintest(name=name)

#myPlot = plot.Plot(figurepath='figure', trialID=0)
#myPlot.udata(xt, trainXY, testXY[1], testXY, testXY)

