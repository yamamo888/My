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
    dummyu = tf.constant([-1.0], tf.float64) #※ これなくす方法わからん
    dummyuxt = tf.cast(tf.convert_to_tensor(np.arange(NX)[:,np.newaxis]), tf.float64) 
    
    # Initial conditions t=0
    initu = tf.while_loop(condinit, bodyinit, loop_vars=[i0, dummyu, nu],
                          shape_invariants=[i0.get_shape(), tf.TensorShape([None]), tf.TensorShape([None])])
    
    # Numerical solution 3: u[:,0] [none,1]
    simulateu = tf.while_loop(cond, body, loop_vars=[n0, tf.expand_dims(initu[1],1), dummyuxt, nu],
                              shape_invariants=[n0.get_shape(), tf.TensorShape([None,None]), tf.TensorShape([None,None]), tf.TensorShape([None])])
    
    
    #return initu, simulateu
    # output u[x,t]
    return simulateu[2][:,1:] # del dummy

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
    
    # u[:,0]
    u = -2.0 * nu * (dphi/phi) + 4.0
    
    # [none,] 0:初期値, i=0, i=1, ...
    init_u = tf.concat([init_u, u], 0)
    
    return [i+1, init_u[init_u>0.0], nu]
# ----       

# ----    
# outside for n range(NT)
def cond(n, ux, uxt, nu):
    NT = 100
    return n < NT

def body(n, ux, uxt, nu):
    
    i0 = tf.constant(0)
    dummyu = tf.constant([-1.0], dtype=tf.float64) #※ これなくす方法わからん
 
    
    # 0:i, 1:ut, 2:ut+1(simulate), 3:nu 
    updateux = tf.while_loop(xcond, xbody, loop_vars=[i0, ux, dummyu, nu],
                       shape_invariants=[i0.get_shape(), tf.TensorShape([None,None]), tf.TensorShape([None]), tf.TensorShape([None])])
    
    updateux = tf.expand_dims(updateux[2][updateux[2]>0.0],1) # del dummy
    
    # copy ux <- updateux
    ux = updateux
    
    # u[x,t]
    uxt = tf.concat([uxt, updateux], 1)
    
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
    
    tmpu = ut[i] - ut[i] * (DT/DX) * (ut[i] - ut[ineg[i]]) + nu * (DT/DX**2) * (ut[ipos[i]] - 2.0 * ut[i] + ut[ineg[i]])
  
    # [none,]
    updateu = tf.concat([updateu, tmpu], 0)
    
    return [i+1, ut, updateu, nu]
# ----       
