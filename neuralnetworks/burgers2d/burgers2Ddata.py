# coding: utf-8

import os
import glob
import pdb
import pickle

import numpy as np

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import burgers2Dplot


np.random.seed(1234)


class Data:
    def __init__(self, pdeMode='test', dataMode='test'):
        '''
        datamode: small(256/100) or middle(256/50) or large(256/10)
                  Num of X.
        '''

        self.pdeMode = pdeMode
        self.dataMode = dataMode
    
        self.modelPath = 'model'
        
    
    def burgers2D(self, NU=0.1):
        
        # parameters ----
        #NT = 501
        #NX = 101
        #NY = 101
        
        NT = 301
        NX = 81
        NY = 81
        
        TMIN = 0.0
        TMAX = 0.5
        XMAX = 2
        YMAX = 2
        
        DT = TMAX/(NT-1)
        DX = XMAX/(NX-1)
        DY = YMAX/(NY-1)
       
        u = np.ones((NX, NY, NT))
        v = np.ones((NX, NY, NT))
        # ----
        
        '''
        # ※うまくいかない
        # Initial Condition
        for i in np.arange(NX):
            x[i] = i*DX
            for j in np.arange(NY):
                y[j] = j*DY
                if (int((NX-1)/4) <= i < int((NX-1)/2)) or (int((NY-1)/4) <= j < int((NY-1)/2)):
                    u[i,j,0] = 2
                    v[i,j,0] = 2
                    
        # Boundary Condition
        for n in np.arange(NT):
            for i in np.arange(NX):
                u[i,0,n] = 1
                u[i,NY-1,n] = 1
                v[i,0,n] = 1
                v[i,NY-1,n] = 1
            for j in np.arange(NY):
                u[0,j,n] = 1
                u[NX-1,j,n] = 1
                v[0,j,n] = 1
                v[NX-1,j,n] = 1
        '''
        # Boundary conditions
        u[0,:,:] = u[NX-1,:,:] = u[:,0,:] = u[:,NY-1,:] = 1
        v[0,:,:] = v[NX-1,:,:] = v[:,0,:] = v[:,NY-1,:] = 1
       
        # Initial conditions
        #u[:,:,:] = v[:,:,:] = 1
        u[int((NX-1)/4):int((NX-1)/2),int((NY-1)/4):int((NY-1)/2),0] = 2
        v[int((NX-1)/4):int((NX-1)/2),int((NY-1)/4):int((NY-1)/2),0] = 2
       
        #pdb.set_trace()
        # Numerical Solution
        for n in range(0,NT-1):
            for i in range(1,NX-1):
                for j in range(1,NY-1):
                    u[i,j,n+1] = (u[i,j,n]
                    -DT*((u[i,j,n]/DX)*(u[i,j,n]-u[i-1,j,n])+(v[i,j,n]/DY)*(u[i,j,n]-u[i,j-1,n]))
                    +DT*NU*((u[i-1,j,n]-2*u[i,j,n]+u[i+1,j,n])/DX**2
                    +(u[i,j-1,n]-2*u[i,j,n]+u[i,j+1,n])/DY**2))
                    
                    v[i,j,n+1] = (v[i,j,n]
                    -DT*((u[i,j,n]/DX)*(v[i,j,n]-v[i-1,j,n])+(v[i,j,n]/DY)*(v[i,j,n]-v[i,j-1,n]))
                    +DT*NU*((v[i-1,j,n]-2*v[i,j,n]+v[i+1,j,n])/DX**2
                    +(v[i,j-1,n]-2*v[i,j,n]+v[i,j+1,n])/DY**2))
        
        
        #self.plot_3D(u,x,y,0,'Figure1')
        #self.plot_3D(u,x,y,50,'Figure2')
        #self.plot_3D(v,x,y,0,'Figure3')
        #self.plot_3D(v,x,y,50,'Figure4')
        
        return u, v
        
    
    def plot_3D(self, u,x,y,time,title):
       """
       Plots the 2D velocity field
       """
    
       fig=plt.figure(figsize=(11,7),dpi=100)
       ax=fig.gca(projection='3d')
       ax.set_xlabel('x (m)')
       ax.set_ylabel('y (m)')
      
       X,Y=np.meshgrid(x,y)
       surf=ax.plot_surface(X,Y,u[:,:,time],rstride=2,cstride=2)
       plt.title(title)
       #plt.savefig(f'test_{title}.png')
       plt.show()
       plt.close()
       
    
    # ----
    def saveXYTUV(self):
        
        # parameters ----
        defaultnu = 0.1
        #minnu = 0.0001
        #minnu = 0.028
        #minnu = 0.058
        minnu = 0.088
        #maxnu = 0.028
        #maxnu = 0.058
        #maxnu = 0.088
        maxnu = 0.1101

        swnu = 0.0001
       
        #NT = 501
        #NX = 101
        #NY = 101
        
        NT = 301
        NX = 81
        NY = 81
        
        TMIN = 0.0
        TMAX = 0.5
        XMAX = 2                     
        YMAX = 2
        
        DX = XMAX/(NX-1)
        DY = YMAX/(NY-1)
       
        x = np.zeros(NX)
        y = np.zeros(NY)
        t = np.linspace(TMIN, TMAX, NT)
        # ----
    
        # 一様分布
        samplenu = np.arange(minnu, maxnu, swnu)
        
        cnt = 0
        flag = False
        for nu in samplenu:
            cnt += 1
            print(cnt)
            
            obsu, obsv = self.burgers2D(NU=np.around(nu,5))
            
            if not flag:
                U = obsu[np.newaxis]
                V = obsv[np.newaxis]
                NU = np.array([nu])
                flag = True
            else:
                U = np.vstack([U, obsu[np.newaxis]])
                V = np.vstack([V, obsv[np.newaxis]])
                NU = np.hstack([NU, np.array([nu])])
        
        # X Loop
        for i in range(0, NX):
           x[i] = i*DX
    
        # Y Loop
        for j in range(0, NY):
           y[j] = j*DY
         
        #pdb.set_trace()
        with open(os.path.join(self.modelPath, self.pdeMode, f'XYTUVNU_{maxnu}.pkl'), 'wb') as fp:
            pickle.dump(x, fp, protocol=4)
            pickle.dump(y, fp, protocol=4)
            pickle.dump(t, fp, protocol=4)
            pickle.dump(U, fp, protocol=4)
            pickle.dump(V, fp, protocol=4)
            pickle.dump(NU, fp, protocol=4)
    # ----

    def concatdata(self):

        data = glob.glob(os.path.join('model','burgers2d_small','*pkl'))
        
        flag = False
        for i in [2,0,1,3]:
            f = data[i]
            print(f)
            #pdb.set_trace()

            with open(f, 'rb') as fp:
                x = pickle.load(fp)
                y = pickle.load(fp)
                t = pickle.load(fp)
                u = pickle.load(fp)
                v = pickle.load(fp)
                nu = pickle.load(fp)
            #pdb.set_trace()
            if not flag:
                us = u
                vs = v
                nus = nu
                flag = True
            else:
                us = np.concatenate([us, u], 0)
                vs = np.concatenate([vs, v], 0)
                nus = np.concatenate([nus, nu],0)

        #pdb.set_trace()
        with open(os.path.join(self.modelPath, self.pdeMode, f'XYTUVNU_small.pkl'), 'wb') as fp:
            pickle.dump(x, fp, protocol=4)
            pickle.dump(y, fp, protocol=4)
            pickle.dump(t, fp, protocol=4)
            pickle.dump(us, fp, protocol=4)
            pickle.dump(vs, fp, protocol=4)
            pickle.dump(nus, fp, protocol=4)


    
    # ----
    def maketraintest(self, name):
    
        nData = 1101
        ind = np.ones(nData, dtype=bool)
        # train data index
        trainidx = np.random.choice(nData, int(nData*0.8), replace=False)
        # del 0.0001 0.001 0.01 0.1
        trainidx = [s for s in trainidx if s not in [0,9,99,999]]
        ind[trainidx] = False
        vec = np.arange(nData)
        # test data index
        testidx = vec[ind]
        
        # loading sparse data 
        sparsepath = glob.glob(os.path.join(self.modelPath, self.pdeMode, f'SparseXYTUVNU_{name}.pkl'))
        # space data, 256 -> xDim
        with open(sparsepath[0], 'rb') as fp:
            x = pickle.load(fp)
            y = pickle.load(fp)
            t = pickle.load(fp)
            pu = pickle.load(fp)
            pv = pickle.load(fp)
            nu = pickle.load(fp)
            idx = pickle.load(fp)
            idy = pickle.load(fp)
             
        #pdb.set_trace()
        trU = pu[trainidx,:]
        teU = pu[testidx,:]
        trV = pv[trainidx,:]
        teV = pv[testidx,:]
        trNU = nu[trainidx]
        teNU = nu[testidx]
        
        
        with open(os.path.join(self.modelPath, self.pdeMode, f'SparsetrainXYTUVNU_{name}.pkl'), 'wb') as fp:
            pickle.dump(x, fp, protocol=4)
            pickle.dump(y, fp, protocol=4)
            pickle.dump(t, fp, protocol=4)
            pickle.dump(trU, fp, protocol=4)
            pickle.dump(trV, fp, protocol=4)
            pickle.dump(trNU, fp, protocol=4)
            pickle.dump(idx, fp, protocol=4)
            pickle.dump(idy, fp, protocol=4)
            
        with open(os.path.join(self.modelPath, self.pdeMode, f'SparsetestXYTUVNU_{name}.pkl'), 'wb') as fp:
            pickle.dump(x, fp, protocol=4)
            pickle.dump(y, fp, protocol=4)
            pickle.dump(t, fp, protocol=4)
            pickle.dump(teU, fp, protocol=4)
            pickle.dump(teV, fp, protocol=4)
            pickle.dump(teNU, fp, protocol=4)
            pickle.dump(idx, fp, protocol=4)
            pickle.dump(idy, fp, protocol=4)
    # ----

    # ----
    def miniBatch(self, index):
        
        batchX = self.train


   
#name = 'small'
#name = 'middle'
name = 'large'
myData = Data(pdeMode='burgers2d', dataMode=name)

#myData.burgers2D()

#[1]
#myData.saveXYTUV()

#[2]'
#myData.concatdata()

'''
#[2]
print(name)
with open(os.path.join(myData.modelPath, myData.pdeMode, 'XYTUVNU.pkl'), 'rb') as fp:
    X = pickle.load(fp)
    Y = pickle.load(fp)
    T = pickle.load(fp)
    U = pickle.load(fp)
    V = pickle.load(fp)
    NU = pickle.load(fp)

# making space X (large, middle small)
if name == 'large':
    xSize = 40
    ySize = 40
    #xSize = 25
    #ySize = 25
elif name == 'middle':
    xSize = 20
    ySize = 20
    #xSize = 12
    #ySize = 12
elif name == 'small':
    xSize = 8
    ySize = 8
    #xSize = 5
    #ySize = 5

tmpidx = np.random.choice(X.shape[0], xSize, replace=False)
tmpidy = np.random.choice(Y.shape[0], ySize, replace=False)

idx = np.sort(tmpidx)
idy = np.sort(tmpidy)

# make sparse data
pU = U[:,:,idy,:]
pV = V[:,:,idy,:]

pU = pU[:,idx,:,:]
pV = pV[:,idx,:,:]

#pdb.set_trace()
with open(os.path.join(myData.modelPath, myData.pdeMode, f'SparseXYTUVNU_{name}_.pkl'), 'wb') as fp:
    pickle.dump(X[:,None], fp, protocol=4)
    pickle.dump(Y[:,None], fp, protocol=4)
    pickle.dump(T[:,None], fp, protocol=4)
    pickle.dump(pU, fp, protocol=4)
    pickle.dump(pV, fp, protocol=4)
    pickle.dump(NU, fp, protocol=4)
    pickle.dump(idx, fp, protocol=4)
    pickle.dump(idy, fp, protocol=4)
#----
'''
#[3]
myData.maketraintest(name=name)

