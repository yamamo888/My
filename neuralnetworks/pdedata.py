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
        
        #imgpath = glob.glob(os.path.join('model','burgers',f'randommaskIMGXTUNU_{name}.pkl'))
        imgpath = glob.glob(os.path.join('model','burgers',f'IMGXTUNU_{name}.pkl'))
        
        #pdb.set_trace()
        # space data, 256 -> xDim
        with open(imgpath[0], 'rb') as fp:
            allX = pickle.load(fp)
            T = pickle.load(fp)
            U = pickle.load(fp)
            NU = pickle.load(fp)
            X = pickle.load(fp)
            allU = pickle.load(fp)
             
        trU = U[trainidx,:]
        trallU = allU[trainidx,:]
        teallU = allU[testidx,:]
        teU = U[testidx,:]
        trNU = NU[trainidx]
        teNU = NU[testidx]
        
        
        with open(os.path.join(self.modelPath, self.pdeMode, f'IMGtrainXTUNU_{name}.pkl'), 'wb') as fp:
            pickle.dump(allX, fp)
            pickle.dump(T, fp)
            pickle.dump(trU, fp)
            pickle.dump(trNU, fp)
            pickle.dump(X, fp)
            pickle.dump(trallU, fp)
            
        #with open(os.path.join(self.modelPath, self.pdeMode, f'randommaskIMGtestXTUNU_{name}.pkl'), 'wb') as fp:    
        with open(os.path.join(self.modelPath, self.pdeMode, f'IMGtestXTUNU_{name}.pkl'), 'wb') as fp:
            pickle.dump(allX, fp)
            pickle.dump(T, fp)
            pickle.dump(teU, fp)
            pickle.dump(teNU, fp)
            pickle.dump(X, fp)
            pickle.dump(teallU, fp)
    # ----
    
    # ----    
    def traintestvaridation(self):
        
        # train data
        with open(os.path.join(self.modelPath, self.pdeMode, f'IMGtrainXTUNU_{self.dataMode}.pkl'), 'rb') as fp:
            self.alltrainX = pickle.load(fp) #[xDim,]
            self.trainT = pickle.load(fp) #[100,]
            self.trainU = pickle.load(fp) #[256,100]
            self.trainNU = pickle.load(fp) #[data]
            self.trainX = pickle.load(fp) #[xDim,]
        
        # test data
        # testX,testT: 同じX,Tがtestデータ分
        with open(os.path.join(self.modelPath, self.pdeMode, f'IMGtestXTUNU_{self.dataMode}.pkl'), 'rb') as fp:
            alltestX = pickle.load(fp)
            testT = pickle.load(fp)
            _ = pickle.load(fp)
            testNU = pickle.load(fp)
            testX = pickle.load(fp) # mask X
            testU = pickle.load(fp)
        
        # varidation data
        with open(os.path.join(self.modelPath, self.pdeMode, f'IMGvarXTUNU_{self.dataMode}.pkl'), 'rb') as fp:
            _ = pickle.load(fp) # X
            _ = pickle.load(fp) # T
            _ = pickle.load(fp) # mask data
            varNU = pickle.load(fp)
            varX = pickle.load(fp) # mask X
            varU = pickle.load(fp) # all data
        # [256,1] [100,1] [data,256,100,1]
        return alltestX, testX, testT, testU[:,:,:,np.newaxis], testNU[:,np.newaxis], varX, varU[np.newaxis,:,:,np.newaxis], np.array([varNU])[np.newaxis]
    # ----
     
    # ----
    def makeImg(self,x,t,u,label='test',name='large'):
            
        X, T = np.meshgrid(x,t) #[100,256]
        #pdb.set_trace() 
        X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None])) # [25600,2]
        # flatten: [100,256]
        u_star = u.flatten()[:,None] # [25600,1]         
    
        U_star = griddata(X_star, u_star.flatten(), (X, T), method='cubic')
    
        img = plt.imshow(U_star.T, interpolation='nearest', cmap='gray', origin='lower')
    
        plt.xlabel('t')
        plt.ylabel('x')
        plt.title(f'nu={label}')
        
        plt.colorbar(img, shrink=0.3)
        
        plt.savefig(os.path.join('figure',f'burgers_{name}',f'{label}.png'))
        plt.close()
        
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
            #pdb.set_trace() 
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
        
    
name = 'small'
#name = 'middle'
#name = 'large'
myData = pdeData(dataMode='small')
#[3]
#myData.maketraintest(name=name)

#[1]
#myData.saveXTU()

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

tmpidx = np.random.choice(X.shape[0], xSize, replace=False)
idx = np.sort(tmpidx)

# for train & test ----
flag = False
for i in np.arange(NU.shape[0]):
  
    label = np.round(NU[i],5).astype(str)
    print(label)
    
    if name == 'large':
        myData.makeImg(X,T,U[i,:,:],label=label,name='exact')
        
    myData.makeImg(X[idx],T,U[i,:,idx].T,label=label,name=f'{name}')
        
    pdb.set_trace()    
    
    if not flag:
        Us = U[i,:,idx][np.newaxis] # [x,t]
        flag = True
    else:
        Us = np.vstack([Us,U[i,:,idx][np.newaxis]])

with open(os.path.join('model','burgers',f'IMGXTUNU_{name}.pkl'), 'wb') as fp:
    pickle.dump(X[:,np.newaxis], fp)
    pickle.dump(T[:,np.newaxis], fp)
    pickle.dump(Us, fp)
    pickle.dump(NU, fp)
    pickle.dump(X[idx,np.newaxis], fp)
    pickle.dump(Us, fp)
# ----


