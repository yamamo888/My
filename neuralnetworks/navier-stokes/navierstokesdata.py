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


class Data:
    def __init__(self, pdeMode='test', dataMode='test'):
        
        self.pdeMode = pdeMode
        self.dataMode = dataMode
        
        self.modelPath = 'model'
        
        # parameter ----
        self.num_gridx = 90
        self.num_gridy = 60
        
        self.delta = 0.1
        self.delta_t = 0.05
        
        object_sizex = 5
        object_sizey = 10
        object_posix = 30
        object_posiy = 30
        
        self.object_startx = object_posix - object_sizex // 2 + 1
        self.object_endx =  self.object_startx + object_sizex
        self.object_starty = object_posiy - object_sizey // 2 + 1
        self.object_endy =  self.object_starty + object_sizey
        self.num_vx = self.num_gridx+2
        self.num_vy = self.num_gridy+2
        self.num_px = self.num_vx+1
        self.num_py = self.num_vy+1
        self.num_Ax, self.num_Ay = self.num_px-2, self.num_py-2
        # ----
        
    # ---- 
    
    # ----
    def ConvectionTerm(self, u, v, flag_v, u_old, v_old):
        for i in range(1, self.num_vy-1):
            for j in range(1, self.num_vx-1):
                if flag_v[i, j] >= 1: continue
                if u_old[i, j] >= 0 and v_old[i, j] >= 0:
                    u[i, j] -= (u_old[i, j] * (u_old[i, j] - u_old[i, j-1]) + v_old[i, j] * (u_old[i, j] - u_old[i-1, j])) * self.delta_t / self.delta
                    v[i, j] -= (u_old[i, j] * (v_old[i, j] - v_old[i, j-1]) + v_old[i, j] * (v_old[i, j] - v_old[i-1, j])) * self.delta_t / self.delta
                elif u_old[i, j] < 0 and v_old[i, j] >= 0:
                    u[i, j] -= (u_old[i, j] * (u_old[i, j+1] - u_old[i, j]) + v_old[i, j] * (u_old[i, j] - u_old[i-1, j])) * self.delta_t / self.delta
                    v[i, j] -= (u_old[i, j] * (v_old[i, j+1] - v_old[i, j]) + v_old[i, j] * (v_old[i, j] - v_old[i-1, j])) * self.delta_t / self.delta
                elif u_old[i, j] >= 0 and v_old[i, j] < 0:
                    u[i, j] -= (u_old[i, j] * (u_old[i, j] - u_old[i, j-1]) + v_old[i, j] * (u_old[i+1, j] - u_old[i, j])) * self.delta_t / self.delta
                    v[i, j] -= (u_old[i, j] * (v_old[i, j] - v_old[i, j-1]) + v_old[i, j] * (v_old[i+1, j] - v_old[i, j])) * self.delta_t / self.delta
                else:
                    u[i, j] -= (u_old[i, j] * (u_old[i, j+1] - u_old[i, j]) + v_old[i, j] * (u_old[i+1, j] - u_old[i, j])) * self.delta_t / self.delta
                    v[i, j] -= (u_old[i, j] * (v_old[i, j+1] - v_old[i, j]) + v_old[i, j] * (v_old[i+1, j] - v_old[i, j])) * self.delta_t / self.delta
        return
    # ----
    # ----  
    def DiffusionTerm(self, Re, v, v_old, flag_v):
        #pdb.set_trace()
        for i in range(1, self.num_vy-1):
            for j in range(1, self.num_vx-1):
                if flag_v[i, j] >= 1: continue
                v[i, j] += (v_old[i+1, j] + v_old[i-1, j] + v_old[i, j+1] + v_old[i, j-1] - 4*v_old[i, j]) * self.delta_t / (self.delta**2 * Re)
        return
    # ----
    # ----
    def DiverV(self, s, u, v, flag_v):
        for i in range(1, self.num_py-1):
            for j in range(1, self.num_px-1):
                if flag_v[i, j] >= 3 or flag_v[i-1, j] >= 3 or flag_v[i-1, j-1] >= 3 or flag_v[i, j-1] >= 3:
                    continue
                if flag_v[i, j] == 2:
                    if i == self.num_py-2: u[i, j], v[i, j] = u[i-1, j], v[i-1, j]
                    else:  u[i, j], v[i, j] = u[i, j-1], v[i, j-1]
                if flag_v[i-1, j] == 2:
                    if i-1 == 0: u[i-1, j], v[i-1, j] = u[i, j], v[i, j]
                    else: u[i-1, j], v[i-1, j] = u[i-1, j-1], v[i-1, j-1]
                if flag_v[i, j-1] == 2:
                    if i == self.num_py-2: u[i, j-1], v[i, j-1] = u[i-1, j-1], v[i-1, j-1]
                    else: u[i, j-1], v[i, j-1] = u[i, j], v[i, j]
                if flag_v[i-1, j-1] == 2:
                    if i-1 == 0: u[i-1, j-1], v[i-1, j-1] = u[i, j-1], v[i, j-1]
                    else: u[i-1, j-1], v[i-1, j-1] = u[i-1, j], v[i-1, j]
                s[i, j] = (
                    u[i-1, j] - u[i-1, j-1] + u[i, j] - u[i, j-1] + \
                    v[i, j-1] - v[i-1, j-1] + v[i, j] - v[i-1, j]
                ) * self.delta / (2*self.delta_t)
        return
    # ----  
    # ----
    def Cholesky(self, p, s, flag_p, A):
        s = s[1:-1, 1:-1] * -1
        b = s.reshape((-1,))
        
        L = np.linalg.cholesky(A)
        t = np.linalg.solve(L, b)
        x = np.linalg.solve(L.T.conj(), t)
        x = x.reshape((self.num_Ay, self.num_Ax))
        for i in range(1, self.num_py-1):
            for j in range(1, self.num_px-1):
                if flag_p[i, j] >= 1: continue
                p[i, j] = x[i-1, j-1]
                if flag_p[i+1, j] == 2: p[i+1, j] = x[i-1, j-1]
                if flag_p[i-1, j] == 2: p[i-1, j] = x[i-1, j-1]
                if flag_p[i, j+1] == 2: p[i, j+1] = x[i-1, j-1]
                if flag_p[i, j-1] == 2: p[i, j-1] = x[i-1, j-1]
        return
    # ----
    # ----   
    def PressureTerm(self, p, u, v, flag_p, flag_v, A):
        s = np.zeros((self.num_py, self.num_px))
        self.DiverV(s, u, v, flag_v)
        self.Cholesky(p, s, flag_p, A)
        return
    # ----   
    # ----   
    def UpdateV(self, u, v, p, flag_v):
        for i in range(1, self.num_vy-1):
            for j in range(1, self.num_vx-1):
                if flag_v[i, j] >= 1: continue
                u[i, j] -= (p[i, j+1] - p[i, j] + p[i+1, j+1] - p[i+1, j]) * self.delta_t / (2*self.delta)
                v[i, j] -= (p[i+1, j] - p[i, j] + p[i+1, j+1] - p[i, j+1]) * self.delta_t / (2*self.delta)
        return
    # ----
    # ----   
    def NablaV(self, u, v, flag_v):
        nablav = 0
        for i in range(1, self.num_py-1):
            for j in range(1, self.num_px-1):
                if flag_v[i, j] >= 3: continue
                elif flag_v[i-1, j] >= 3: continue
                elif flag_v[i-1, j-1] >= 3: continue
                elif flag_v[i, j-1] >= 3: continue
                nablav += u[i-1, j] - u[i-1, j-1] + u[i, j] - u[i, j-1] + \
                    v[i, j-1] - v[i-1, j-1] + v[i, j] - v[i-1, j]
        return nablav
    # ----
    # ----
    def initA(self, flag_p):
        N = self.num_Ax * self.num_Ay 
        A = np.eye(N) * 4
        for i in range(N):
            if flag_p[i // self.num_Ax + 1, i % self.num_Ax + 1] >= 2:
                A[i, i] = 1
                continue
            Apij = [
                [i, i-self.num_Ax, i // self.num_Ax, i % self.num_Ax + 1],
                [i, i-1, i // self.num_Ax + 1, i % self.num_Ax],
                [i, i+1, i //self.num_Ax + 1, i% self.num_Ax + 2],
                [i, i+self.num_Ax, i // self.num_Ax + 2, i % self.num_Ax + 1]
            ]
            for Ai, Aj, pi, pj in Apij:
                if flag_p[pi, pj] == 2:
                    A[Ai, Ai] -= 1
                if Aj < 0 or Aj >= N:
                    continue
                if flag_p[pi, pj] >= 1 or pi == 0 or pi == self.num_py-1 \
                        or pj == 0 or pj == self.num_px-1:
                    A[Ai, Aj] = 0
                else:
                    A[Ai, Aj] = -1
        return A
    # ----
    # ----
    def navierstokesSimulate(self, Re):
        
        # parameters ----
        # default
        #Re = 100
        u_value = 0.98 
        v_value = 0.02 
        
        time_step = 1000
        # ----
        
        #[1]init boundary
        #u, v,[num_gridx+2, num_gridy+2] p[num_gridx+3,num_gridy+3]
        u = np.zeros((self.num_vy, self.num_vx))
        u[0,:], u[:,0], u[-1,:] = u_value, u_value, u_value
        v = np.zeros((self.num_vy, self.num_vx))
        v[0,:], v[:,0], v[-1,:] = v_value, v_value, v_value
        p = np.zeros((self.num_py, self.num_px))
        #flag_v, flag_p
        flag_v = np.zeros((self.num_vy, self.num_vx))
        flag_v[:, -1] = 2
        flag_v[0, :], flag_v[-1, :], flag_v[:, 0] = 1, 1, 1
        flag_p = np.zeros((self.num_py, self.num_px))
        flag_p[:, -1] = 2
        flag_p[0, :], flag_p[-1:, :], flag_p[:, 0] = 1, 1, 1
        #flag_v, flag_p
        flag_v[self.object_starty:self.object_endy, self.object_startx:self.object_endx] = 1
        flag_v[self.object_starty+1:self.object_endy-1, self.object_startx+1:self.object_endx-1] = 3
        flag_p[self.object_starty+1:self.object_endy, self.object_startx+1:self.object_endx] = 2
        flag_p[self.object_starty+2:self.object_endy-1, self.object_startx+2:self.object_endx-1] = 3
        #create A
        A = self.initA(flag_p)
        
        print(f"t=0 divergence v: {self.NablaV(u, v, flag_v)}")
        
        # start t > 0
        flag = False
        for t in np.arange(time_step):
        
            u_old = u.copy()
            v_old = v.copy()
            
            if not flag:
                U = u_old[:,:,None]
                V = v_old[:,:,None]
                # ※ first zeros?
                P = np.zeros((self.num_py, self.num_px))[:,:,None]
                flag = True
            else:
                U = np.concatenate([U, u[:,:,None]],2)
                V = np.concatenate([V, v[:,:,None]],2)
                P = np.concatenate([P, p[:,:,None]],2)
                
            self.ConvectionTerm(u, v, flag_v, u_old, v_old)
            self.DiffusionTerm(Re, u, u_old, flag_v)
            self.DiffusionTerm(Re, v, v_old, flag_v)
            self.PressureTerm(p, u, v, flag_p, flag_v, A)
            self.UpdateV(u, v, p, flag_v)
            
            if t % 100 == 0:
                print(f"t={t+1} divergence v: {self.NablaV(u, v, flag_v)}")
        
        return U,V,P
    # ----
    
    # ----
    def saveXYTUVP(self):
        
        # parameters ----
        # ※調整？
        minRe = 40
        maxRe = 300
        swRe = 1
        # 一様分布
        sampleRe = np.arange(minRe, maxRe, swRe)
      
        nx = 92 
        ny = 62 
        nt = 1000
       
        x = np.arange(nx)
        y = np.arange(ny)
        t = np.arange(nt)
        # ----
        
        cnt = 0
        flag = False
        for re in sampleRe:
            
            cnt += 1
            print(cnt)
            
            # u,v[y(62),x(92),t(1000)] p[y(63),x(93),t(1000)]
            u, v, p = self.navierstokesSimulate(Re=re)
            
            if not flag:
                U = u[None]
                V = v[None]
                P = p[None]
                Re = np.array([re])
                flag = True
            else:
                U = np.vstack([U, u[None]])
                V = np.vstack([V, v[None]])
                P = np.vstack([P, p[None]])
                Re = np.hstack([Re, np.array([re])]) #[sampleRe]
        
        #pdb.set_trace()
        # save data
        with open(os.path.join(self.modelPath, self.pdeMode, 'XYTUVP.pkl'), 'wb') as fp:
            pickle.dump(x, fp) 
            pickle.dump(y, fp) 
            pickle.dump(t, fp) 
            pickle.dump(U, fp)
            pickle.dump(V, fp)
            pickle.dump(P, fp)
            pickle.dump(Re, fp)
    # ----
    
    # ----
    def maketraintest(self, name):
        
        nData = 260
        
        ind = np.ones(nData, dtype=bool)
        # train data index
        trainidx = np.random.choice(nData, int(nData*0.8), replace=False)
        # ※index変更? del 100
        trainidx = [s for s in trainidx if s not in [100]]
        ind[trainidx] = False
        vec = np.arange(nData)
        # test data index
        testidx = vec[ind]
        
        imgpath = glob.glob(os.path.join(self.modelPath, self.pdeMode, f'IMGXYTUVP_{name}.pkl'))
        
        with open(imgpath[0], 'rb') as fp:
            allX = pickle.load(fp)
            allY = pickle.load(fp)
            allT = pickle.load(fp)
            U = pickle.load(fp)
            V = pickle.load(fp)
            P = pickle.load(fp)
            Re = pickle.load(fp)            
            idx = pickle.load(fp)
            idy = pickle.load(fp)

        trU = U[trainidx]
        teU = U[testidx]
        trV = V[trainidx]
        teV = V[testidx]
        trP = P[trainidx]
        teP = P[testidx]  
        trRe = Re[trainidx]
        teRe = Re[testidx]
    
        # save train & test
        with open(os.path.join(self.modelPath, self.pdeMode, f'IMGtrainXYTUVPNU_{name}.pkl'), 'wb') as fp:
            pickle.dump(allX, fp)
            pickle.dump(allY, fp)
            pickle.dump(allT, fp)
            pickle.dump(trU, fp)
            pickle.dump(trV, fp)
            pickle.dump(trP, fp)
            pickle.dump(trRe, fp)
            pickle.dump(idx, fp)
            pickle.dump(idy, fp)
            
        with open(os.path.join(self.modelPath, self.pdeMode, f'IMGtestXYTUVPNU_{name}.pkl'), 'wb') as fp:
            pickle.dump(allX, fp)
            pickle.dump(allY, fp)
            pickle.dump(allT, fp)
            pickle.dump(teU, fp)
            pickle.dump(teV, fp)
            pickle.dump(teP, fp)
            pickle.dump(teRe, fp)
            pickle.dump(idx, fp)
            pickle.dump(idy, fp)
    # ----
    
    # ----    
    def traintest(self):
        
        # train data
        with open(os.path.join(self.modelPath, self.pdeMode, f'IMGtrainXYTUVPNU_{self.dataMode}.pkl'), 'rb') as fp:
            self.alltrainX = pickle.load(fp) 
            self.alltrainY = pickle.load(fp) 
            self.alltrainT = pickle.load(fp) 
            self.trainU = pickle.load(fp) 
            self.trainV = pickle.load(fp) 
            self.trainP = pickle.load(fp) 
            self.trainRe = pickle.load(fp) 
            self.trainX = pickle.load(fp)
            self.trainY = pickle.load(fp)
            
        # test data
        with open(os.path.join(self.modelPath, self.pdeMode, f'IMGtestXYTUVPNU_{self.dataMode}.pkl'), 'rb') as fp:
            alltestX = pickle.load(fp)
            alltestY = pickle.load(fp)
            alltestT = pickle.load(fp)
            testU = pickle.load(fp)
            testV = pickle.load(fp)
            testP = pickle.load(fp)
            testRe = pickle.load(fp)
            testX = pickle.load(fp)
            testY = pickle.load(fp)
            
        return alltestX,alltestY,alltestT, testX,testY, testU,testV,testP, testRe
    # ----
    
    # ----
    def miniBatch(self, index):
        
        batchX = self.trainX
        batchY = self.trainY
        batchT = self.trainT
        batchU = self.trainU[index]
        batchV = self.trainV[index]
        batchP = self.trainP[index]
        batchNU = self.trainNU[index]
        
        return batchX, batchY, batchT, batchU, batchV, batchP, batchNU
    # ----
   
    
name = 'small'
#name = 'middle'
#name = 'large'
myData = Data(pdeMode='nv', dataMode=name)

#[1]
#myData.saveXYTUVP()


#[2]
with open(os.path.join('model','nv','XYTUVP.pkl'), 'rb') as fp:
    X = pickle.load(fp)
    Y = pickle.load(fp)
    T = pickle.load(fp)
    U = pickle.load(fp)
    V = pickle.load(fp)
    P = pickle.load(fp)
    Re = pickle.load(fp)

# making space X (large, middle small)
if name == 'large':
    xSize = 46
    ySize = 31
elif name == 'middle':
    xSize = 23
    ySize = 15
elif name == 'small':
    xSize = 9
    ySize = 6

# ※ok?                
tmpidx = np.random.choice(X.shape[0], xSize, replace=False)
tmpidy = np.random.choice(Y.shape[0], ySize, replace=False)

idx = np.sort(tmpidx)
idy = np.sort(tmpidy)

# make space data
pUy = U[:,idy,:,:] # [y,x,t]
pVy = V[:,idy,:,:]
pPy = P[:,idy,:,:]

pU = pUy[:,:,idx,:]
pV = pVy[:,:,idx,:]
pP = pPy[:,:,idx,:]

with open(os.path.join('model', 'nv', f'IMGXYTUVP_{name}.pkl'), 'wb') as fp:
     pickle.dump(X[:,None], fp)
     pickle.dump(Y[:,None], fp)
     pickle.dump(T[:,None], fp)
     pickle.dump(pU, fp)
     pickle.dump(pV, fp)
     pickle.dump(pP, fp)
     pickle.dump(Re, fp)
     pickle.dump(idx, fp)
     pickle.dump(idy, fp)


#[3]
myData.maketraintest(name=name)

    
    
       

    
  
 