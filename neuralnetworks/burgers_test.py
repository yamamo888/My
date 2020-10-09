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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import cv2

from matplotlib import animation, rc
import sympy
from sympy.utilities.lambdify import lambdify

import tensorflow as tf
import math

# ----    
def burgersPy(x,t,nu, xNum=256,tNum=100):
    
    # observation
    obsu = np.zeros([xNum, tNum])

    for j in range (0, tNum):
        for i in range (0, xNum):
            a = ( x[i] - 4.0 * t[j] )
            b = ( x[i] - 4.0 * t[j] - 2.0 * np.pi )
            c = 4.0 * nu * ( t[j] + 1.0 )
                
           
            phi = np.exp ( - a * a / c ) + np.exp ( - b * b / c )
            dphi = - 2.0 * a * np.exp ( - a * a / c ) / c \
                   - 2.0 * b * np.exp ( - b * b / c ) / c
            
            
            obsu[i,j] = 4.0 - 2.0 * nu * dphi / phi
      
    return obsu
# ----

# ----
def burgersPy2(x,t,nu, xNum=256,tNum=100):
    
    NT = 100
    NX = 150
    TMAX = 0.5
    XMAX = 2.0*np.pi
    NU = 0.01
    
    # Increments
    DT = TMAX/(NT-1)
    DX = XMAX/(NX-1)
    
    # Initialise data structures
    u = np.zeros((NX,NT))
    u_analytical = np.zeros((NX,NT))
    x = np.zeros(NX)
    t = np.zeros(NT)
    ipos = np.zeros(NX)
    ineg = np.zeros(NX)
    
    # Periodic boundary conditions
    for i in range(0,NX):
        x[i] = i*DX
        ipos[i] = i+1
        ineg[i] = i-1
    
    ipos[NX-1] = 0
    ineg[0] = NX-1
    
    #pdb.set_trace()
    
    # Initial conditions
    for i in range(0,NX):
        phi = np.exp( -(x[i]**2)/(4*NU) ) + np.exp( -(x[i]-2*np.pi)**2 / (4*NU) )
        dphi = -(0.5*x[i]/NU)*np.exp( -(x[i]**2) / (4*NU) ) - (0.5*(x[i]-2*np.pi) / NU )*np.exp(-(x[i]-2*np.pi)**2 / (4*NU) )
        # [100,]
        u[i,0] = -2*NU*(dphi/phi) + 4
        
    #pdb.set_trace()
    # u [100(x), 151(t)]
    # Numerical solution
    for n in range(0,NT-1):
        for i in range(0,NX):
            u[i,n+1] = (u[i,n]-u[i,n]*(DT/DX)*(u[i,n]-u[int(ineg[i]),n])+
             NU*(DT/DX**2)*(u[int(ipos[i]),n]-2*u[i,n]+u[int(ineg[i]),n]))
            
    pdb.set_trace()
    
# ----   

# ----       
def burgersTensor2(nu):
    """
    w = tf.Variable([[0.5]])

    ans = tf.matmul(nu,w)
    ans = tf.square(10.0 - ans)
    
    return ans, w
    """
    
    # init  
    NX = 100
   
    i0 = tf.constant(0)
    n0 = tf.constant(0)
    
    dummyu = tf.expand_dims(tf.expand_dims(tf.constant([-1.0], tf.float64),0),0) #※ これなくす方法わからん
    dummyuxt = tf.cast(tf.convert_to_tensor((np.ones([NX,NX])*-1)[np.newaxis]), tf.float64) 
    #pdb.set_trace()
    # [output] 3 i:iteration, u0: [none(データ数),x,1], nu: [none(データ数),1]
    # データ数分の初期t=0のときの観測 ui
    initu = tf.while_loop(condinit, bodyinit, loop_vars=[i0, dummyu, nu],
                          shape_invariants=[i0.get_shape(), tf.TensorShape([None,None,1]), tf.TensorShape([None,1])])
    
    # 1 del dummyu [none,101,1] -> [none,100,1]
    initu = tf.slice(initu[1], [0,1,0], [-1,NX,1])
    
    # Numerical solution 3: u[none(data),x,t] ex) [1,100,100]
    #simulateu = tf.while_loop(cond, body, loop_vars=[n0, initu, dummyuxt, nu],
                              #shape_invariants=[n0.get_shape(), tf.TensorShape([None,None,1]), tf.TensorShape([None,None,None]), tf.TensorShape([None,1])])
    simulateu = tf.while_loop(cond, body, loop_vars=[n0, initu, dummyuxt, initu, nu],
                              shape_invariants=[n0.get_shape(), tf.TensorShape([None,None,1]), tf.TensorShape([None,None,None]), tf.TensorShape([None,None,1]), tf.TensorShape([None,1])])
    
    #return initu
    return simulateu
    
    #return initu, simulateu
    # output u[x,t]
    #return simulateu[2][:,1:] # del dummy
    
# ----       
def condinit(i, _, nu):
    NX = 100
    return i < NX

def bodyinit(i, init_u, nu):
    
    # ----
    pi = tf.constant(3.141592653589793, tf.float64)
    xMin = tf.constant(0.0, tf.float64)
    xMax = tf.constant(2.0 * 3.141592653589793, tf.float64)
    xNum = tf.constant(100)
   
    x = tf.linspace(xMin, xMax, xNum)
    # ----
    # float32 -> float64 for nan
    phi1 = tf.cast(-(x[i]**2)/(4.0*nu), tf.float64)
    phi2 = tf.cast(-(x[i]-2.0*pi)**2 / (4.0*nu), tf.float64)
    phi = tf.exp(phi1) + tf.exp(phi2)
    
    dphi1 = tf.cast(-(x[i]**2) / (4.0*nu), dtype=tf.float64)
    dphi2 = tf.cast(-(x[i] -2.0 * pi)**2 / (4.0*nu), dtype=tf.float64)
    dphi = -(0.5*x[i]/nu) * tf.exp(dphi1) - (0.5* (x[i] -2.0*pi) / nu ) * tf.exp(dphi2)
    
    # u[:,0] [none,1]
    u = -2.0 * nu * (dphi/phi) + 4.0
    
    # [none(データ数),none(x),1] 0:初期値, i=0, i=1, ... -> [none, 101(初期値-1入ってる), 1]
    init_u = tf.concat([init_u, tf.expand_dims(u,2)], 1)
    
    pdb.set_trace()
    
    return [i+1, init_u, nu]
# ----       

# ----    
# outside for n range(NT)
#def cond(n, ux, uxt, nu):

def cond(n, ux, uxt, _, nu):
    NT = 100
    return n < NT

def body(n, ux, uxt, updateux, nu):
    '''
    ux == initu (first time)
    '''
    #pdb.set_trace()
    
    NX = 100
   
    i0 = tf.constant(0)
    dummyu = tf.expand_dims(tf.expand_dims(tf.constant([-1.0], tf.float64),0),0) #※ これなくす方法わからん
    
    # 0:i, 1:ut[none(data),x,1], 2:ut+1(simulate)[none(data),x,1], 3:nu[none(data),1] 
    updateux = tf.while_loop(xcond, xbody, loop_vars=[i0, ux, dummyu, nu],
                       shape_invariants=[i0.get_shape(), tf.TensorShape([None,None,1]), tf.TensorShape([None,None,1]), tf.TensorShape([None,1])])
    
    #pdb.set_trace()
    # 1 del dummyu [none,101,1] -> [none,100,1]
    updateux = tf.slice(updateux[2], [0,1,0], [-1,NX,1])
    
    # copy ux <- updateux
    #ux = updateux
    
    # u[data,x,t]
    #uxt = tf.concat([uxt, updateux], 2)
    
    #return [n+1, ux, uxt, nu]
    return [n+1, ux, uxt, updateux, nu]
    
# ----       

# ----    
# inside for i range(XT)
def xcond(i, ux, updateu, nu):
    NX = 100
    return i < NX

def xbody(i, ut, updateu, nu):
    
    #pdb.set_trace()
    
    # ※手動
    DT = tf.constant(0.005050505050505051, tf.float64)
    DX = tf.constant(0.06346651825433926, tf.float64)
    
    ipos = tf.concat([tf.range(1,100),tf.constant([0])],0)
    ineg = tf.concat([tf.constant([99]),tf.range(0,99)],0)
    
    # ut+1 = ut * nu ... [none(data),1]
    tmpu = ut[i] - ut[i] * (DT/DX) * (ut[i] - ut[ineg[i]]) + nu * (DT/DX**2) * (ut[ipos[i]] - 2.0 * ut[i] + ut[ineg[i]])
    
    # [none,none(101?),1]
    updateu = tf.concat([updateu, tf.expand_dims(tmpu,2)], 1)
    pdb.set_trace()
    return [i+1, ut, updateu, nu]
# ----       


'''
def body():
    
    overlambda = tf.constant([5.0])
    overid = tf.where(x>overlambda)
    overnum = tf.shape(overid)[0]
    overths = tf.tile(overlambda, [overnum])
    overupdate = tf.tensor_scatter_nd_update(x, overid, overths)
    
    updatex = underupdate(overupdate)
    
    return updatex
    
def underupdate(xover):
    
    underlambda = tf.constant([0.01])
    underid = tf.where(xover<underlambda)
    undernum = tf.shape(underid)[0]
    underths = tf.tile(underlambda, [undernum])
    underupdate = tf.tensor_scatter_nd_update(xover, underid, underths)
    
    return underupdate

    
    overlambda = tf.constant([5.0])
    underlambda = tf.constant([0.01])
    
    # [1] x < 0.01 yes 0.01 [2] no 0.01 < x < 5 yes x no 5
    #return tf.where(x<underlambda, updateunder(x), overths(x))
    return tf.where(x<underlambda, updateunder(x), func(x))


def updateunder(x):
    
    init_val = (0,0)
    count, update = tf.while_loop(cond, body, init_val)
    
    return update

def overths(x):
    
    overlambda = tf.constant([5.0])
    underlambda = tf.constant([0.01])
    
    return tf.where(underlambda<x<overlambda, x, updateover(x))

'''
# ----

# ----
def plotImg(x,t,u,label='test',dirname='exact'):
        
    X, T = np.meshgrid(x,t) #[100,256]
    
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None])) # [25600,2]
    # flatten: [100,256]
    u_star = u.flatten()[:,None] # [25600,1]         

   
    U_star = griddata(X_star, u_star.flatten(), (X, T), method='cubic')
    pdb.set_trace()
    img = plt.imshow(U_star.T, interpolation='nearest', cmap='gray',
                     origin='lower')
    
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title(f'nu={label}')
     
    plt.colorbar(img,shrink=0.3)
    
    fpath = os.path.join('figure', f'burgers_{dirname}')
    isdir = os.path.exists(fpath)
    if not isdir:
        os.makedirs(fpath)
    
    plt.savefig(os.path.join(fpath, f'{label}.png'))
    plt.close()
# ----
    
if __name__ == "__main__":
    
    # command argment ----
    parser = argparse.ArgumentParser()
    # burgers mode (python var or tensorflow var)
    parser.add_argument('--mode', required=True, choices=['python', 'tensorflow'])
    # burgers param nu<-[0.05,10] (ex) 0.1 1.0 10.0
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
    tMax = 0.5
    # time
    T = np.linspace(tMin, tMax, tNum) # [100,]
    
    # Num.of space
    xNum = 100
    # range of space
    xMin = 0.0
    xMax = 2.0 * np.pi
    # space
    X = np.linspace(xMin, xMax, xNum) # [256,]
    
    NX = 100
    XMAX = 2.0*np.pi
    DX = XMAX/(NX-1)
      
    x = np.zeros(NX)
        
    for i in range(0,NX):
        x[i] = i*DX
        

    # Num.of data
    nData = len(NUs)
    # ----
    
    # Call burgers by python or tensorflow ----
    if mode == 'python':
        
        flag = False
        for NU in NUs:
            print(f'>>> start burgers py var nu={NU}')
            # [x.shape,t.shape]
            #U = burgersPy(X,T,NU, xNum=xNum,tNum=tNum)
            U = burgersPy2(X,T,NU, xNum=xNum,tNum=tNum)
            pdb.set_trace()
            if not flag:
                Us = U.T[np.newaxis]
                flag = True
            else:
                Us = np.vstack([Us, U.T[np.newaxis]])
    
    elif mode == 'tensorflow':
        
        nu = tf.compat.v1.placeholder(tf.float64,shape=[None,1])
        ind = tf.compat.v1.placeholder(tf.int32,shape=[None,1])
        
        #initu_op = burgersTensor2(nu)
        simulateu_op = burgersTensor2(nu)
        
        #initu_op, simulateu_op = burgersTensor2(nu)
        # [none,none]
        #simulateu_op, w_op = burgersTensor2(nu)
        #gather_op = tf.gather_nd(simulateu_op, ind)
        
        #pdb.set_trace()
        #grad = tf.gradients(initu_op, nu)
        #grad = tf.gradients(simulateu_op, nu)
        #grad = tf.gradients(gather_op, nu)
        
        
        #pdb.set_trace()
        config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1,allow_growth=True)) 
        sess = tf.compat.v1.Session(config=config)
        sess.run(tf.compat.v1.global_variables_initializer())
        
        
        for i in np.arange(1):
            
            feed_dict = {nu:np.array(NUs)[:,np.newaxis], ind:np.array([0,2])[:,np.newaxis]}
            #pdb.set_trace()
            #updataX, gradX = sess.run([initu_op,grad], feed_dict)
            updateX = sess.run([simulateu_op], feed_dict)
            #updataX = sess.run([initu_op, simulateu_op], feed_dict)
            #updataX, gatherX, gradX = sess.run([simulateu_op, gather_op, grad], feed_dict)
            
            print(updateX)
            #print(gatherX)
            #print(gradX)
            #print(updataX[0,:10])
        pdb.set_trace()
            
        '''
        # graph ----
        # placeholder
        x = tf.compat.v1.placeholder(tf.float32,shape=[tNum, xNum])
        t = tf.compat.v1.placeholder(tf.float32,shape=[xNum, tNum])
        nu = tf.compat.v1.placeholder(tf.float32,shape=[nData,1])
        pi = tf.compat.v1.placeholder(tf.float32,shape=[1])
        
        # u = PDE(x,t,nu)
        u_op, a_op, b_op, c_op, tmpa_op, tmpb_op, tmpc_op, phi_op, tmpnu_op = burgersTensor(x,t,nu,pi, nData=nData,xDim=xNum)
        
        config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1,allow_growth=True)) 
        sess = tf.compat.v1.Session(config=config)
        sess.run(tf.compat.v1.global_variables_initializer())
        # ----
        Xgrid = np.reshape(np.tile(X, tNum), [-1, xNum])
        Tgrid = np.reshape(np.tile(T, xNum), [-1, tNum])
           
        feed_dict = {x:Xgrid, t:Tgrid, pi:np.array([np.pi]), nu:np.array([NUs]).T}
        Us, A, B, C, tmpA, tmpB, tmpC, Phi, tmpNU = sess.run([u_op, a_op, b_op, c_op, tmpa_op, tmpb_op, tmpc_op, phi_op, tmpnu_op], feed_dict)
        
        print(Phi)
        pdb.set_trace()
        '''
    # ----
    
    '''
    # ----
   
      
    # Plot ----
    #for ni in np.arange(testNU.shape[0]):
    for ni in np.arange(len(NUs)):
        print(f'>>> plot nu={NUs[ni]}')
        plotImg(X,T,Us[ni],dirname=mode,label=NUs[ni])
        
    # ----
    '''
    