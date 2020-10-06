# -*- coding: utf-8 -*-

import os
import glob
import pdb
import pickle

import numpy as np
import scipy.io
from scipy.interpolate import griddata


import matplotlib as mpl
mpl.use('Agg')

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
    def burgers2(self, NU=0.01):
        
        NT = 100
        NX = 100
        TMIN = 0.0
        TMAX = 0.5
        XMAX = 2.0*np.pi
        t = np.linspace(TMIN, TMAX, NT) # [100,]
        
        # Increments
        DT = TMAX/(NT-1)
        DX = XMAX/(NX-1)
        
        # Initialise data structures
        u = np.zeros((NX,NT))
        x = np.zeros(NX)
        ipos = np.zeros(NX)
        ineg = np.zeros(NX)
        
        # Periodic boundary conditions
        for i in range(0,NX):
            x[i] = i*DX
            ipos[i] = i+1
            ineg[i] = i-1
        
        ipos[NX-1] = 0
        ineg[0] = NX-1
        
        
        # Initial conditions
        for i in range(0,NX):
            phi = np.exp( -(x[i]**2)/(4*NU) ) + np.exp( -(x[i]-2*np.pi)**2 / (4*NU) )
            dphi = -(0.5*x[i]/NU)*np.exp( -(x[i]**2) / (4*NU) ) - (0.5*(x[i]-2*np.pi) / NU )*np.exp(-(x[i]-2*np.pi)**2 / (4*NU) )
            u[i,0] = -2*NU*(dphi/phi) + 4
        
        # Numerical solution
        for n in range(0,NT-1):
            for i in range(0,NX):
                u[i,n+1] = (u[i,n]-u[i,n]*(DT/DX)*(u[i,n]-u[int(ineg[i]),n])+
                 NU*(DT/DX**2)*(u[int(ipos[i]),n]-2*u[i,n]+u[int(ineg[i]),n]))
        #pdb.set_trace() 
        return x, t, u
    # ---- 
        
    # ----
    def saveXTU(self):
        '''
        minnu, maxnu, swnu 変更で、データ量・データ範囲変更可能.
        '''
        # default
        defaultnu = 0.01
        #minnu = 0.01 # burgers
        #maxnu = 5.001
        minnu = 0.005 # burgers2
        maxnu = 0.305
        swnu = 0.001
        
        # 一様分布
        samplenu = np.arange(minnu, maxnu, swnu)
        
        cnt = 0
        flag = False
        for nu in samplenu:
            cnt += 1
            print(cnt)
            #x, t, obsu = self.burgers(nu=nu)
            # x:[100,] t:[100,] obsu:[x,t]
            x, t, obsu = self.burgers2(NU=nu)
            #pdb.set_trace() 
            self.makeImg(x,t,obsu.T,label=f'{np.round(nu,3)}',name='exact')
            #pdb.set_trace() 
            if not flag:
                X = x
                T = t
                U = obsu[np.newaxis]
                NU = np.array([nu])
                flag = True
            else:
                X = np.vstack([X, x]) # [data, 100]
                T = np.vstack([T, t]) # [data, 100]
                U = np.vstack([U, obsu[np.newaxis]]) # [data, x, t]
                NU = np.hstack([NU, np.array([nu])]) # [data,]
    
        # X,T: 全データ数分同じデータconcat
        with open(os.path.join(self.modelPath, self.pdeMode, 'XTUNU2.pkl'), 'wb') as fp:
            pickle.dump(X[0], fp) #[256,] or [100,]
            pickle.dump(T[0], fp) #[100,]
            pickle.dump(U, fp) 
            pickle.dump(NU, fp)
    # ----
        
    # ----
    def maketraintest(self, name):
        
        nData = 300
        ind = np.ones(nData, dtype=bool)
        # train data index
        trainidx = np.random.choice(nData, int(nData*0.8), replace=False)
        # del 0.005 0.01 0.02 0.03
        trainidx = [s for s in trainidx if s not in [0, 5, 15, 95, 295]]
        ind[trainidx] = False
        vec = np.arange(nData)
        # test data index
        testidx = vec[ind]
        
        imgpath = glob.glob(os.path.join('model','burgers',f'IMGXTUNU2_{name}.pkl'))
        
        # space data, 256 -> xDim
        with open(imgpath[0], 'rb') as fp:
            allX = pickle.load(fp)
            T = pickle.load(fp)
            U = pickle.load(fp)
            NU = pickle.load(fp)
            X = pickle.load(fp)
            allU = pickle.load(fp)
            idx = pickle.load(fp)
             
        #pdb.set_trace()
        trU = U[trainidx,:]
        teU = U[testidx,:]
        trallU = allU[trainidx,:]
        teallU = allU[testidx,:]
        trNU = NU[trainidx]
        teNU = NU[testidx]
        
        
        with open(os.path.join(self.modelPath, self.pdeMode, f'IMGtrainXTUNU2_{name}.pkl'), 'wb') as fp:
            pickle.dump(allX, fp)
            pickle.dump(T, fp)
            pickle.dump(trU, fp)
            pickle.dump(trNU, fp)
            pickle.dump(X, fp)
            pickle.dump(trallU, fp)
            pickle.dump(idx, fp)
            
        with open(os.path.join(self.modelPath, self.pdeMode, f'IMGtestXTUNU2_{name}.pkl'), 'wb') as fp:
            pickle.dump(allX, fp)
            pickle.dump(T, fp)
            pickle.dump(teU, fp)
            pickle.dump(teNU, fp)
            pickle.dump(X, fp)
            pickle.dump(teallU, fp)
            pickle.dump(idx, fp)
    # ----
    
    # ----    
    def traintest(self):
        
        # train data
        with open(os.path.join(self.modelPath, self.pdeMode, f'IMGtrainXTUNU2_{self.dataMode}.pkl'), 'rb') as fp:
            self.alltrainX = pickle.load(fp) #[xDim,]
            self.trainT = pickle.load(fp) #[100,]
            self.trainU = pickle.load(fp) #[256,100]
            self.trainNU = pickle.load(fp) #[data]
            self.trainX = pickle.load(fp) #[xDim,]
        
        # test data
        # testX,testT: 同じX,Tがtestデータ分
        with open(os.path.join(self.modelPath, self.pdeMode, f'IMGtestXTUNU2_{self.dataMode}.pkl'), 'rb') as fp:
            alltestX = pickle.load(fp)
            testT = pickle.load(fp)
            testU = pickle.load(fp)
            testNU = pickle.load(fp)
            testX = pickle.load(fp)
            _ = pickle.load(fp)
            idx = pickle.load(fp)
        
        # [100,1] [100,1] [data,100,100,1]
        return alltestX, testX, testT, testU[:,:,:,np.newaxis], testNU[:,np.newaxis], idx
    # ----
     
    # ----
    def makeImg(self,x,t,u,label='test',name='large'):
        #pdb.set_trace()
        X, T = np.meshgrid(x,t)
        
        X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
        
        u_star = u.flatten()[:,None]     
    
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
        
   
#name = 'small'
#name = 'middle'
#name = 'large'
#myData = pdeData(dataMode='small')
#[3]
#myData.maketraintest(name=name)

#[1]
#myData.saveXTU()

'''
#[2]
print(name)

with open(os.path.join('model','burgers','XTUNU2.pkl'), 'rb') as fp:
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
    #pdb.set_trace()    
    myData.makeImg(X[idx],T,U[i,idx,:].T,label=label,name=f'{name}')
        
    if not flag:
        Us = U[i,idx,:][np.newaxis] # [x,t]
        flag = True
    else:
        Us = np.vstack([Us,U[i,idx,:][np.newaxis]])

with open(os.path.join('model','burgers',f'IMGXTUNU2_{name}.pkl'), 'wb') as fp:
    pickle.dump(X[:,np.newaxis], fp)
    pickle.dump(T[:,np.newaxis], fp)
    pickle.dump(Us, fp)
    pickle.dump(NU, fp)
    pickle.dump(X[idx,np.newaxis], fp)
    pickle.dump(U, fp)
    pickle.dump(idx, fp)
#----
'''

