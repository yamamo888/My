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
def burgers(nu):
    
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
    simulateu = tf.while_loop(cond, body, loop_vars=[n0, initu, dummyuxt, nu],
                              shape_invariants=[n0.get_shape(), tf.TensorShape([None,None,1]), tf.TensorShape([None,None,None]), tf.TensorShape([None,1])])
    
    # del dummy [none,x,200] -> [none,x,100]
    simulateu = tf.slice(simulateu[2], [0,0,100], [-1,NX,100])
    
    # push initu & pop last u
    simulateu = tf.concat([initu,simulateu],2)

    return simulateu[:,:,:-1]
# ----       

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
    
    #pdb.set_trace()
    
    return [i+1, init_u, nu]
# ----       

# ----    
# outside for n range(NT)
def cond(n, ux, uxt, nu):
    NT = 100
    return n < NT

def body(n, ux, uxt, nu):
    
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
    
    # 1 del dummyu [none,101,1] -> [none,100,1]
    updateux = tf.slice(updateux[2], [0,1,0], [-1,NX,1])
    
    # copy ux <- updateux
    ux = updateux
    
    # u[data,x,t]
    uxt = tf.concat([uxt, updateux], 2)
    
    return [n+1, ux, uxt, nu]
# ----       

# ----    
# inside for i range(XT)
def xcond(i, ux, updateu, nu):
    NX = 100
    return i < NX

def xbody(i, ut, updateu, nu):
    # ※手動
    DT = tf.constant(0.005050505050505051, tf.float64)
    DX = tf.constant(0.06346651825433926, tf.float64)
    
    ipos = tf.concat([tf.range(1,100),tf.constant([0])],0)
    ineg = tf.concat([tf.constant([99]),tf.range(0,99)],0)
    
    # ut+1 = ut * nu ... [none(data),1]
    tmpu = tf.slice(ut, [0,i,0], [-1,1,1]) - tf.slice(ut, [0,i,0], [-1,1,1]) * (DT/DX) * (tf.slice(ut,[0,i,0], [-1,1,1]) - 
                    tf.slice(ut, [0,ineg[i],0], [-1,1,1])) + nu * (DT/DX**2) * (tf.slice(ut, [0,ipos[i],0], [-1,1,1]) 
                    - 2.0 * tf.slice(ut,[0,i,0], [-1,1,1]) + tf.slice(ut, [0,ineg[i],0], [-1,1,1]))
    
    #pdb.set_trace()
    
    # [none,none(101),1]
    updateu = tf.concat([updateu, tmpu], 1)
    
    return [i+1, ut, updateu, nu]
# ----       
