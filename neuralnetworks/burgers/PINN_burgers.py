# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
#from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import cv2

import argparse
import time
import pdb

import burgersdata


np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x, u, layers, lb, ub):
        
        self.lb = lb
        self.ub = ub
        
        self.x = x[:,0]
        self.t = x[:,1]
        self.u = u
        
        self.layers = layers
        
        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        # Initialize parameters
        self.lambda1 = tf.Variable([-6.0], dtype=tf.float32) # 0.0?
        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, 1])
      
        self.u_pred = self.net_u(self.x_tf, self.t_tf)
        self.f_pred = self.net_f(self.x_tf, self.t_tf)
        
        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_pred))
        
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 100000, # 学習回数？
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
    
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
    
    # ----
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        
        return Y
            
    def net_u(self, x, t):  
        u = self.neural_net(tf.concat([x,t],1), self.weights, self.biases)
        return u
    
    def net_f(self, x, t):
        '''
        x: [none,1]
        t: [none,1]
        '''
        
        lambda1 = tf.exp(self.lambda1)
        u = self.net_u(x,t)
        
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_t + u*u_x - lambda1*u_xx
        
        return f
    
    def callback(self, loss, lambda_1):
        print('Loss: %e, l1: %.5f' % (loss, np.exp(lambda_1)))
        
    
    # ----
    def train(self, nItr):
        #pdb.set_trace()
            
        tf_dict = {self.x_tf:self.x[:,None], self.t_tf:self.t[:,None], self.u_tf:self.u}
        
        for it in range(nItr):
            
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                loss_value = self.sess.run(self.loss, tf_dict)
                lambda_1_value = self.sess.run(self.lambda1)
                
                print('It: %d, Loss: %.3e, Lambda_1: %.6f' % (it, loss_value, lambda_1_value))
                
        
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss, self.lambda1],
                                loss_callback = self.callback)
    # ----
        
    def predict(self, X_star):
        
        tf_dict = {self.x_tf: X_star[:,0:1], self.t_tf: X_star[:,1:2]}
        
        u_star = self.sess.run(self.u_pred, tf_dict) # [25600,1]
        f_star = self.sess.run(self.f_pred, tf_dict)
        #pdb.set_trace()
        return u_star, f_star

    
if __name__ == "__main__": 
     
    # command argment ----
    parser = argparse.ArgumentParser()

    # iteration of training
    parser.add_argument('--nItr', type=int, default=0)
    # datamode (pkl)
    parser.add_argument('--dataMode', required=True, choices=['large', 'middle', 'small'])
    # index test 2==0.01
    parser.add_argument('--index', type=int, default=2)
    # trial ID
    parser.add_argument('--trialID', type=int, default=0)    
    
    # 引数展開
    args = parser.parse_args()
    
    nItr = args.nItr
    dataMode = args.dataMode
    index = args.index
    trialID = args.trialID
    # ----

    
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    
    # Dataset ----
    myData = burgersdata.Data(pdeMode='burgers', dataMode=dataMode)
    allx, partx, allt, testU, testNU, idx = myData.traintest()
    #pdb.set_trace()
    # select nu & u
    nu = testNU[index]
    exactU = testU[index]
    
    X, T = np.meshgrid(partx,allt) #[100,256]
    #X, T = np.meshgrid(allx,allt) #[100,256]
    
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None])) # [xDim*t,2]
    u_star = exactU.flatten()[:,None] # [xDim*t,1]

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)       
    # ----
    #pdb.set_trace()
    # train ----
    model = PhysicsInformedNN(X_star, u_star, layers, lb, ub)
    model.train(nItr=nItr)
    # -----
    
    # predict ----
    u_pred, f_pred = model.predict(X_star)
            
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    
    lambda_value = model.sess.run(model.lambda1)
    lambda_value = np.exp(lambda_value)
    
    error_lambda = np.abs(lambda_value - nu)/nu * 100
    mse_lambda = np.square(nu - lambda_value)
    
    print('Exact lambda: %.3f, Predict lambda: %f' %(nu, lambda_value))
    print('TestPLoss: %.10f' %(mse_lambda))
    print('Error u: %e' % (error_u))
    print('Error l1: %.5f%%' % (error_lambda))                          
    # ----