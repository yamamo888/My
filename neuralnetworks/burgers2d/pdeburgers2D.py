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


NX = 101
NY = 101
NT = 501

TMAX = 0.5
XMAX = 2
YMAX = 2

DT = tf.cast(TMAX/(NT-1), tf.int8)
DX = tf.cast(XMAX/(NX-1), tf.int8)
DY = tf.cast(YMAX/(NY-1), tf.int8)
  

# ----
# lambda1,lambda2: pred param
def burgers2d(lambda1,lambda2):
    
    # parameter ----
    indx1 = tf.cast((NX-1)/4, tf.int8)
    indx2 = tf.cast((NX-1)/2, tf.int8)
    indy1 = tf.cast((NX-1)/4, tf.int8)
    indy2 = tf.cast((NY-1)/2, tf.int8)
    
    onesu = tf.ones([NX,NY,NT])[None]
    onesv = tf.ones([NX,NY,NT])[None] 
    
    t0 = tf.constant(0)
    dummyuxyt = tf.constant([-1.0], tf.float64)[None,None,None,:]
    dummyvxyt = tf.constant([-1.0], tf.float64)[None,None,None,:]
    # ----
    pdb.set_trace()
    
    # Initial condition ----
    initu = tf.slice(onesu, [:,indx1,indy1,:], [:,indx2,indy2,:])
    initv = tf.slice(onesv, [:,indx1,indy1,:], [:,indx2,indy2,:])
    # ----
    
    # Numerical solution ----
    simulateuv = tf.while_loop(cond, body, loop_vars=[t0, initu, initv, dummyuxyt, dummyvxyt, lambda1, lambda2],
                              shape_invariants=[t0.get_shape(), 
                                                tf.TensorShape([1,None,None,None]), tf.TensorShape([1,None,None,None]),
                                                tf.TensorShape([1,None,None,None]), tf.TensorShape([1,None,None,None]), 
                                                tf.TensorShape(1,None), tf.TensorShape(1,None)])
    # ----
    
    pdb.set_trace()
    
    return simulateu
# ----
    
# t Loop ----
def cond(t, _, _, _):
    return t < NT-1

def body(t, inituxyt, initvxyt, uxyt, vxyt, lambda1, lambda2):
    
    # parameter ----
    i0 = tf.constant(0)
    dummyuxy = tf.constant([-1.0], tf.float64)[None,:,None,None]
    dummyvxy = tf.constant([-1.0], tf.float64)[None,:,None,None]
    # ----
    
    # index 0,1:limit 2,3:u,v[i,j,n], 4,5:u,v[i,j,n+1]
    output = tf.while_loop(xcond, xbody, loop_vars=[i0, t, inituxyt, initvxyt, dummyuxy, dummyvxy, lambda1, lambda2],
                             shape_invariants=[i0.get_shape(), i0.get_shape(), 
                                               tf.TensorShape([1,None,None,None]), tf.TensorShape([1,None,None,None]),
                                               tf.TensorShape([1,None,None,None]), tf.TensorShape([1,None,None,None]),
                                               tf.TensorShape(1,None), tf.TensorShape(1,None)])
    
    pdb.set_trace()
    # 1 del dummy [0(del),x1,x2,...] 
    tmpuxy = tf.slice(output[4], [0,1,0,0], [-1,NX,1,1])
    tmpvxy = tf.slice(output[5], [0,1,0,0], [-1,NX,1,1])
    
    # copy inituxyt <- tmpuxy u,v[data,x,t,tn] -< u,v[data,x,t,tn+1] 
    inituxyt = tmpuxy
    initvxyt = tmpvxy
    
    # u,v[data,xdim,ydim,tdim+1]
    uxyt = tf.concat([inituxyt, tmpuxy], -1)
    vxyt = tf.concat([initvxyt, tmpvxy], -1)
    
    return [t+1, ux, uxyt, vxyt, lambda1, lambda2]
# ----

# x Loop ----
def xcond(i, _, _, _, _):
    return 1 < i < NX-1

def xbody(i, t, inituxyt, initvxyt, updateuxy, updatevxy, lambda1, lambda2):
    
    # parameter ----
    j0 = tf.constant(0)
    dummyuy = tf.constant([-1.0], tf.float64)[None,None,:,None]
    dummyvy = tf.constant([-1.0], tf.float64)[None,None,:,None]
    # ----
    
    # index 0,1,2:limit 3,4:u,v[i,j,n], 5,6:u,v[i,j,n+1]
    output = tf.while_loop(ycond, ybody, loop_vars=[j0, i, t, inituxyt, initvxyt, dummyuy, dummyvy, lambda1, lambda2],
                             shape_invariants=[j0.get_shape(), j0.get_shape(), j0.get_shape(), 
                                               tf.TensorShape([1,None,None,None]), tf.TensorShape([1,None,None,None]), 
                                               tf.TensorShape([1,None,None,None]), tf.TensorShape([1,None,None,None]), 
                                               tf.TensorShape(1,None), tf.TensorShape(1,None)])
    
    pdb.set_trace()
    # 1 del dummy [0(del),y1,y2...]
    tmpuxy = tf.slice(output[5], [0,0,1,0], [-1,1,NY,1])
    tmpvxy = tf.slice(output[6], [0,0,1,0], [-1,1,NY,1])
    
    # u,v[data,xdim+1,ydim,1]
    updateuxy = tf.concat([updateuxy, tmpuxy], 1)
    updatevxy = tf.concat([updatevxy, tmpvxy], 1)
    
    return [i+1, t, inituxyt, initvxyt, updateuxy, updatevxy, lambda1, lambda2]
# ----

# y Loop ----
def ycond(j, _, _, _, _, _):
     return 1 < j < NX-1
    
def ybody(j, i, t, inituxyt, initvxyt, updateuy, updatevy, lambda1, lambda2):
    
    pdb.set_trace()
    # initxyt [data,x,y,t]
    tmpuy = tf.slice(inituxyt, [0,i,j,t], [-1,1,1,1]) # u(i,j,n)
    - DT*(tf.slice(inituxyt, [0,i,j,t], [-1,1,1,1])/DX)*lambda1*(tf.slice(inituxyt, [0,i,j,t], [-1,1,1,1])-tf.slice(inituxyt, [0,i-1,j,t], [-1,1,1,1]) #2項目
    + (tf.slice(initvxyt, [0,i,j,t], [-1,1,1,1])/DY)*lambda1*(tf.slice(inituxyt, [0,i,j,t], [-1,1,1,1])-tf.slice(inituxyt, [0,i,j-1,t], [-1,1,1,1])) #3項目
    + DT*lambda2*((tf.slice(inituxyt, [0,i-1,j,t], [-1,1,1,1])-2*tf.slice(inituxyt, [0,i,j,t], [-1,1,1,1])+tf.slice(inituxyt, [0,i+1,j,t], [-1,1,1,1]))/DX**2 #4項目 
    + (tf.slice(inituxyt, [0,i,j-1,t], [-1,1,1,1])-2*tf.slice(inituxyt, [0,i,j,t], [-1,1,1,1])+tf.slice(inituxyt, [0,i,j+1,t], [-1,1,1,1]))/DY**2)
    
    tmpvy = tf.slice(initvxyt, [0,i,j,t], [-1,1,1,1]) 
    - DT*(tf.slice(inituxyt, [0,i,j,t], [-1,1,1,1])/DX)*lambda1*(tf.slice(initvxyt, [0,i,j,t], [-1,1,1,1])-tf.slice(initvxyt, [0,i-1,j,t], [-1,1,1,1]) 
    + (tf.slice(initvxyt, [0,i,j,t], [-1,1,1,1])/DY)*lambda1*(tf.slice(initvxyt, [0,i,j,t], [-1,1,1,1])-tf.slice(initvxyt, [0,i,j-1,t], [-1,1,1,1]))
    + DT*lambda2*((tf.slice(initvxyt, [0,i-1,j,t], [-1,1,1,1])-2*tf.slice(initvxyt, [0,i,j,t], [-1,1,1,1])+tf.slice(initvxyt, [0,i+1,j,t], [-1,1,1,1]))/DX**2 
    + (tf.slice(initvxyt, [0,i,j-1,t], [-1,1,1,1])-2*tf.slice(initvxyt, [0,i,j,t], [-1,1,1,1])+tf.slice(initvxyt, [0,i,j+1,t], [-1,1,1,1]))/DY**2)
    
    # u,v[data,1,ydim+1,1]
    updateuy = tf.concat([updateuy, tmpuy], 2)
    updatevy = tf.concat([updatevy, tmpvy], 2)
    
    
    return [j+1, i, t, inituxyt, initvxyt, updateuy, updatevy, lambda1, lambda2]
# ----
    
    
    
    