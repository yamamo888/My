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
    def savetraintest(self, size=10, savepklname='test.pkl'):
        '''
        xSize: small -> 1/1000, middle -> 1/100, large -> 1/10
        '''
        
        with open(os.path.join(self.modelPath, self.pdeMode, 'XTUNU.pkl'), 'rb') as fp:
            X = pickle.load(fp) # [256,1]
            T = pickle.load(fp) # [256,1]
            # 画像サイズが
            U = pickle.load(fp) # [500, 371, 498, 3]
            NU = pickle.load(fp) # [500]
            
        # train data
        # Num.of data
        nData = X.shape[0]
        nTrain = int(nData*0.8)
        
        #xSize = int(X.shape[1] / size)
        xSize = size
        
        # select x
        # [1] static x (とりあえずこっち) [2] random
        idx = np.random.choice(X.shape[1], xSize, replace=False)
         
        # Index of traindata
        trainID = np.random.choice(X.shape[0], nTrain, replace=False)
        # expectiong trainID
        allind = np.arange(X.shape[0])
        ind = np.ones(nData,dtype=bool)
        ind[trainID] = False
        testID = allind[ind]
        
        # train data
        trainX = X[trainID][:,idx] # [traindata,xsize]
        trainT = T[trainID]
        pdb.set_trace()
        flag = False
        for id in idx:
            allu = U[trainID]
            u = allu[:,:,id]
            
            if not flag:
                trainU = u[:,:,np.newaxis]
                flag = True
            else:
                trainU = np.concatenate([trainU, u[:,:,np.newaxis]],2)
                
        trainNU = NU[trainID]
        
        # test data
        testX = X[testID]
        testT = T[testID] 
        testU = U[testID] # [testdata,100,256]
        testNU = NU[testID] # [testdata]
        
        with open(os.path.join(self.modelPath, self.pdeMode, f'trainXTUNU_{savepklname}.pkl'), 'wb') as fp:
            pickle.dump(trainX, fp)
            pickle.dump(trainT, fp)
            pickle.dump(trainU, fp)
            pickle.dump(trainNU, fp)
        
        
        with open(os.path.join(self.modelPath, self.pdeMode, f'testXTUNU_{savepklname}.pkl'), 'wb') as fp:
            pickle.dump(testX, fp)
            pickle.dump(testT, fp)
            pickle.dump(testU, fp)
            pickle.dump(testNU, fp)
        
    # ----
    
    # ----
    def maketraintest(self):
        
        ind = np.ones(496, dtype=bool)
        # train data index
        trainidx = np.random.choice(496, int(496*0.8), replace=False).tolist()
        ind[trainidx] = False
        vec = np.arange(496)
        # test data index
        testidx = vec[ind]
        
        imgspath = glob.glob(os.path.join('model','burgers_large','IMG*'))
        
        flag = False
        for imgpath in imgspath:
            with open(imgpath, 'rb') as fp:
                X = pickle.load(fp)
                T = pickle.load(fp)
                U = pickle.load(fp)
                NU = pickle.load(fp)
            
            # 圧縮
            if U.shape[1] > 372:
                for uind in trainidx:

                    tru = cv2.resize(U[uind], (498,371))
                    pdb.set_trace()

                    if not flag:
                       trU = tru[np.newaxis]
                       flag = True
                    else:
                        trU = np.vstack([trU,tru[np.newaxis]])
                
                for uind in testidx:

                    teu = cv2.resize(U[uind], (498,371))

                    if not flag:
                       teU = teu[np.newaxis]
                       flag = True
                    else:
                        teU = np.vstack([teU,teu[np.newaxis]])
            
            else:
                trU = U[trainidx]
                teU = U[testidx]
            
            #pdb.set_trace() 
            print(U.shape)
            #trU = U[trainidx,:371,:498,:]
            #teU = U[testidx,:371,:498,:]
            trNU = NU[trainidx]
            teNU = NU[testidx]
            
            if not flag:
                trUs = trU
                trNUs = trNU
                teUs = teU
                teNUs = teNU
                flag = True
            else:
                trUs = np.vstack([trUs,trU])
                trNUs = np.hstack([trNUs,trNU])
                teUs = np.vstack([teUs,teU])
                teNUs = np.hstack([teNUs,teNU])
        
        #pdb.set_trace() 
        with open(os.path.join(self.modelPath, self.pdeMode, f'IMGtrainXTUNU_large.pkl'), 'wb') as fp:
            pickle.dump(X, fp)
            pickle.dump(T, fp)
            pickle.dump(trUs, fp)
            pickle.dump(trNUs, fp)
            
        with open(os.path.join(self.modelPath, self.pdeMode, f'IMGtestXTUNU_large.pkl'), 'wb') as fp:
            pickle.dump(X, fp)
            pickle.dump(T, fp)
            pickle.dump(teUs, fp)
            pickle.dump(teNUs, fp)
        
    
    # ----    
    def traintest(self):
        
        # train data
        #with open(os.path.join(self.modelPath, self.pdeMode, f'trainXTUNU_{self.dataMode}.pkl'), 'rb') as fp:
        with open(os.path.join(self.modelPath, self.pdeMode, f'IMGtrainXTUNU_large.pkl'), 'rb') as fp:
            self.trainX = pickle.load(fp)
            self.trainT = pickle.load(fp)
            self.trainU = pickle.load(fp)
            self.trainNU = pickle.load(fp)
        
        # test data
        # testX,testT: 同じX,Tがtestデータ分
        #with open(os.path.join(self.modelPath, self.pdeMode, f'testXTUNU_{self.dataMode}.pkl'), 'rb') as fp:
        with open(os.path.join(self.modelPath, self.pdeMode, f'IMGtestXTUNU_large.pkl'), 'rb') as fp:
            testX = pickle.load(fp)
            testT = pickle.load(fp)
            testu = pickle.load(fp)
            testnu = pickle.load(fp)
        
        teidx = np.random.choice(testu.shape[0], 250, replace=False)

        testU = testu[teidx.tolist()]
        testNU = testnu[teidx.tolist()]
        
        return testX[0], testT[0], testU, testNU
        #return self.trainU, self.trainNU, testX[0], testT[0], testU, testNU
    
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

    def delSpace(self,imgpath):

        img = cv2.imread(imgpath[0])
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.threshold(img_gray, 254, 255, cv2.THRESH_BINARY)[1]

        contours = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

        x1 = []
        y1 = []
        x2 = []
        y2 = []
        for i in range(1, len(contours)):
            ret = cv2.boundingRect(contours[i])
            x1.append(ret[0])
            y1.append(ret[1])
            x2.append(ret[0]+ret[2])
            y2.append(ret[1]+ret[3])

        x1_min = min(x1)
        y1_min = min(y1)
        x2_max = max(x2)
        y2_max = max(y2)

        delimg = img[y1_min:y2_max, x1_min:x2_max]

        return delimg

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
    def CNNfeature(self, x, reuse=False):
        
        with tf.compat.v1.variable_scope("CNN") as scope:
            if reuse:
                scope.reuse_variables()
            #pdb.set_trace() 
    
            # 1st conv layer
            w1 = self.weight_variable('w1', [5,5,3,24])
            b1 = self.bias_variable('b1', [24])
            conv1 = self.conv2d(x, w1, b1, strides=2)
        
            conv1 = self.maxpool2d(conv1)
        
            # 2nd conv layer
            w2 = self.weight_variable('w2', [5,5,24,24])
            b2 = self.bias_variable('b2', [24])
            conv2 = self.conv2d(conv1, w2, b2, strides=2)
        
            conv2 = self.maxpool2d(conv2)
            
             
            w3 = self.weight_variable('w3', [24*32*24, 32])
            b3 = self.bias_variable('b3', [32])
            
            # 1st full-layer
            reshape_conv2 = tf.reshape(conv2, [-1, w3.get_shape().as_list()[0]])
            
            fc1 = self.fc_relu(reshape_conv2,w3,b3)
            
            w4 = self.weight_variable('w4', [32, 32])
            b4 = self.bias_variable('b4', [32])
            fc2 = self.fc_relu(fc1,w4,b4)
            
            return fc2
    # ----
    
    # ----
    def nextBatch(self, index):
        
        #pdb.set_trace()

        batchX = self.trainX[0]
        batchT = self.trainT[0]
        batchU = self.trainU[index]
        batchNU = self.trainNU[index]
        
        return batchX, batchT, batchU, batchNU
    # ----
    

#myData = pdeData(dataMode='small')
#myData.maketraintest()
#trainXY, testXY, xt = myData.Loadingburgers()
#trainXY, testXY, xt = myData.burgers()
#myData.saveXTU()

#Size = [200, 150, 100, 50]
#Name = ['large','middle','small']
#Name = ['LLL','LL','LM','LS']

#for size, name in zip(Size,Name):
    #myData.savetraintest(size=size, savepklname=name)

#trU, trNU, teX, teT, teU, teNU = myData.traintest()

'''
num1 = 4500
num2 = 4500
name = 'large'

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
#for i in np.arange(NU.shape[0])[num1:num2]:
for i in np.arange(NU.shape[0])[num1:]:
    #pdb.set_trace()
    label = NU[i].astype(str)
    print(label)
    print(i)
    
    myData.makeImg(X[0,idx,np.newaxis],T[0,:,np.newaxis],U[i,:,idx].T,label=label,name=name)
    #myData.makeImg(X[idx,:,np.newaxis],T[0,:,np.newaxis],U[i],label=label,name=name)
    #pdb.set_trace()
    imgpath = glob.glob(os.path.join('figure',f'burgers_{name}','*png'))
    img = [s for s in imgpath if label in s]
    #pdb.set_trace()
    tmpimg = myData.delSpace(img)

    if not flag:
        Img = tmpimg[np.newaxis]
        flag = True
    else:
        Img = np.vstack([Img,tmpimg[np.newaxis]])

#pdb.set_trace()
with open(os.path.join('model',f'burgers_{name}',f'IMGXTUNU_{name}_{num1}{num2}.pkl'), 'wb') as fp:
    pickle.dump(X[0,:,np.newaxis], fp)
    pickle.dump(T[0,:,np.newaxis], fp)
    pickle.dump(Img, fp)
    #pickle.dump(NU[num1:num2], fp)
    pickle.dump(NU[num1:], fp)
'''
#myPlot = plot.Plot(figurepath='figure', trialID=0)
#myPlot.udata(xt, trainXY, testXY[1], testXY, testXY)

