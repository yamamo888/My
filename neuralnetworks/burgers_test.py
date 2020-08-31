# -*- coding: utf-8 -*-

import os
import argparse
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


# ----    
def burgersPy(x,t,nu):
    
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

    return obsu
# ----

# ----        
def burgersTensor(nData=1,xDim=256, x,t,nu):
    
    pi = 3.14
        
    # a,bは、すべての u で共通
    tmpa = x - 4.0 * tf.transpose(t) # [t.shape, x.shape]
    tmpb = x - 4.0 * tf.transpose(t) - 2.0 * pi
    # データ数分の t [ndata, t.shape]
    ts = tf.tile(tf.expand_dims(t[0], 0), [nData, 1])
    # データごと(param)に計算 [ndata, t.shape]
    tmpc = 4.0 * nu * (ts + 1.0)
    
    # + N dimention [nBatch, t.shape, x.shape]
    a = tf.tile(tf.expand_dims(tmpa, 0), [nData, 1, 1])
    b = tf.tile(tf.expand_dims(tmpb, 0), [nData, 1, 1])
    c = tf.tile(tf.expand_dims(tmpc, -1), [1, 1, xDim])
    
    # [nBatch, t.shape, x.shape]
    phi = tf.exp(- a * a / c) + tf.exp(- b * b / c)
    dphi = - 2.0 * a * tf.exp(- a * a / c ) / c - 2.0 * b * tf.exp(- b * b / c) / c

    obsu = 4.0 - 2.0 * tf.expand_dims(nu,1) * dphi / phi
    
    return obsu
# ----
    
# ----
def plotImg(x,t,u,label='test',dirname='exact'):
        
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
    
    plt.savefig(os.path.join('figure', dirname, f'{label}.png'))    
# ----

    
if __name__ == "__main__":
    
    
    # command argment ----
    parser = argparse.ArgumentParser()
    # burgers mode (python var or tensorflow var)
    parser.add_argument('--mode', required=True, choices=['python', 'tensorflow'])
    # burgers param nu (ex) 0.1 0.01 1.0
    parser.add_argument('--nu', type=float, default=1.0, nargs='*')

    args = parser.parse_args()
    
    mode = args.mode
    NUs = args.nu
    # ----
    
    # datapoint x,t ----
    # Num.of time
    tNum = 100 
    # range of time
    tMin = 0.0
    tMax = 1.0
    # time
    T = np.linspace(tMin, tMax, tNum) # [100,]
    
    # Num.of space
    xNum = 256
    # range of space
    xMin = 0.0
    xMax = 2.0 * np.pi
    # space
    X = np.linspace(xMin, xMax, xNum) # [256,]
    # ----
    
    # Call burgers by python or tensorflow ----
    if mode == 'python':
        
        for NU in NUs:
            
            U = burgersPy(nData=NUs.shape[0],xDim=xNum, X,T,NU)
            
            if not flag:
                Us = U[np.newaxis]
                flag = True
            else:
                Us = np.vstack([Us, U[np.newaxis]])
    
    elif mode == 'tensorflow':
        
        # graph ----
        # placeholder
        x = tf.compat.v1.placeholder(tf.float32,shape=[tDim, xDim])
        t = tf.compat.v1.placeholder(tf.float32,shape=[xDim, tDim])
        nu = tf.compat.v1.placeholder()
        
        # u = PDE(x,t,nu)
        u_op = burgersTensor(x,t,nu)
        
        config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1,allow_growth=True)) 
        sess = tf.compat.v1.Session(config=config)
        sess.run(tf.compat.v1.global_variables_initializer())
        # ----
        pdb.set_trace()
        feed_dict = {x:X, t:T, nu:NUs}
        Us = sess.run(u_op, feed_dict)
    # ----
    
    # Plot ----
    for ni in np.arange(NUs.shape[0]):
        print(f'>>> plot {nu[ni]}')
        plotImg(X,T,Us[ni],dirname=mode,label=NUs[ni])
    # ----
    
    