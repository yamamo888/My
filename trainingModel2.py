# -*- coding: utf-8 -*-

import sys
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v1

import random
import pickle
import pdb

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt

import makingData
import loadingNankai2
import Plot as myPlot

# -------------------------- command argment ----------------------------------
myArgs = input("Please specify arguments(space separator): ").split()
print(myArgs)

# toy data
if int(myArgs[0]) == 0:
    # 0: toydata var., 1: nankai var.
    dataMode = int(sys.argv[1])
    # Model type 0: ordinary regression, 1: anhor-based, 2: atr-nets
    methodModel = int(sys.argv[2])
    # noize of x1, x2
    sigma = float(sys.argv[3])
    # number of class
    # nankai nClass = 11 or 21 or 51
    nClass = int(sys.argv[4])
    # number of rotation -> sin(pNum*pi) & cos(pNum*pi)
    pNum = int(sys.argv[5])
    # number of layer for Regression NN
    depth = int(sys.argv[6])
    # batch size
    batchSize = int(sys.argv[7])
    # data size
    nData = int(sys.argv[8])
    # rate of training
    trainRatio = float(sys.argv[9])
    # l1loss
    l1Mode = int(sys.argv[10])
    # l2loss
    l2Mode = int(sys.argv[11])
    # trial ID
    trialID = sys.argv[12]

# nankai
elif int(myArgs[0]) == 1:
    # 0: toydata var., 1: nankai var.
    dataMode = int(sys.argv[1])
    # Model type 0: ordinary regression, 1: anchor-based, 2: atr-nets
    methodModel = int(sys.argv[2])
    # nankai nClass = 11 or 21 or 51
    nClass = int(sys.argv[3])
    # number of layer for Regression NN
    # 3 or 4 or 5
    depth = int(sys.argv[4])
    # batch size
    batchSize = int(sys.argv[5])
    # alpha
    alphaMode = float(sys.argv[6])
    # l1loss
    l1Mode = int(sys.argv[7])
    # l2loss
    l2Mode = int(sys.argv[8])
    # constant multiple
    #cm100 = int(sys.argv[9])
    # trial ID
    trialID = sys.argv[9]
# -----------------------------------------------------------------------------

# ------------------------------- path ----------------------------------------
featurePath = "features"
resultPath = "results"
modelPath = "models"
visualPath = "visualization"

if dataMode == 0:
    savePath = "toypickles"
    testPath = "toytests"
else:
    savePath = "nankaipickles"
    testPath = "nankaitests"
    
pickleFullPath = os.path.join(resultPath,savePath)
# -----------------------------------------------------------------------------

# --------------------------- parameters --------------------------------------
# number of nankai cell(input)
nCell = 5
# number of sliding window
nWindow = 10

# node of 1 hidden
nHidden = 128
# node of 2 hidden
nHidden2 = 128
    
# node of 1 hidden
nRegHidden = 128
# node of 2 hidden
nRegHidden2 = 128
# node of 3 hidden
nRegHidden3 = 128
# node of 4 hidden
nRegHidden4 = 128

if methodModel == 2:
    isATR = True
else:
    isATR = False

  
# maximum of target variables
yMax = 6
# miinimum of target variables
yMin = 2

# maximum of nankai
nkMax = 0.0125
# minimum of nankai
nkMin = 0.017
# maximum of tonankai & tokai
tkMax = 0.012
# minimum of tonankai & tokai
tkMin = 0.0165

# Toy
if dataMode == 0:
    print(pNum)
    
    dInput = 2
    dOutput = 1
    # round decimal 
    limitdecimal = 3
    # Width class
    beta = np.round((yMax - yMin) / nClass,limitdecimal)
    dataName = f"toy_{trialID}"
    nTraining = 5000
    # not training alpha
    istrainAlpha = True
# Nankai
else:
    dInput = nCell*nWindow
    dOutput = 3
    # round decimal 
    limitdecimal = 6
    # Width class
    beta = np.round((nkMax - nkMin) / nClass,limitdecimal)
    dataName = f"nankai_{trialID}"
    nTraining = 1000
    # not training alpha
    istrainAlpha = False

# Center variable of the first class
first_cls_center = np.round(yMin + (beta / 2),limitdecimal)
# Center variable of the first class in nankai
first_cls_center_nk = np.round(nkMin + (beta / 2),limitdecimal)
# Center variable of the first class in tonankai & tokai
first_cls_center_tk = np.round(tkMin + (beta / 2),limitdecimal)

# select nankai data(3/5) 
nametrInds = [0,1,2,3,4,5,6]
# random sample loading train data
nameInds = random.sample(nametrInds,3) 

# dropout
keepProbTrain = 1.0
# Learning rate
lr = 1e-3
# nankai file change timing
filePeriod = nTraining / 100
# test count
#testPeriod = 100
# validation count
valPeriod = 100
# if plot == True
isPlot = False
# if save model == True
isSaveModel = True
# if restore model == True
isRestoreModel = False 
# if l1loss == True
if l1Mode == 1:
    isL1 = True
# if l2loss == True
elif l2Mode == 1:
    isL2 = True

# -----------------------------------------------------------------------------

# --------------------------- data --------------------------------------------
# select toydata or nankaidata
if dataMode == 0:    
    myData = makingData.toyData(trainRatio=trainRatio, nData=nData, pNum=pNum, sigma=sigma)
    myData.createData(trialID,beta=beta)
else:
    myData = loadingNankai2.NankaiData(nCell=nCell,nClass=nClass,nWindow=nWindow)
    myData.splitTrainValidation(nameInds=nameInds)
    #myData.loadNankaiRireki()
    if nClass == 10:
        nClass = 11
    elif nClass == 20:
        nClass = 21
    else:
        nClass =  51
# -----------------------------------------------------------------------------

#------------------------- placeholder ----------------------------------------
# input of placeholder for classification
x_cls = tf.placeholder(tf.float32,shape=[None,dInput])
# input of placeholder for regression
x_reg = tf.placeholder(tf.float32,shape=[None,dInput])
x_reg_test = tf.placeholder(tf.float32,shape=[None,dInput])
# GT output of placeholder (target)
y = tf.placeholder(tf.float32,shape=[None,dOutput])
alpha_base = tf.placeholder(tf.float32)

if dataMode == 0:
    # GT output of label
    y_label = tf.placeholder(tf.int32,shape=[None,nClass])
else:
    y_label = tf.placeholder(tf.int32,shape=[None,nClass,dOutput])

# -----------------------------------------------------------------------------

#-----------------------------------------------------------------------------#      
def weight_variable(name,shape):
     return tf.get_variable(name,shape,initializer=tf.random_normal_initializer(stddev=0.1))
#-----------------------------------------------------------------------------#
def bias_variable(name,shape):
     return tf.get_variable(name,shape,initializer=tf.constant_initializer(0.1))
#-----------------------------------------------------------------------------#
def alpha_variable(name,shape):
    #alphaInit = tf.random_normal_initializer(mean=0.5,stddev=0.1)
    
    if alphaMode == 0.01:
        mean = 0.01
    elif alphaMode == 1:
        mean = 1
    elif alphaMode == 10:
        mean = 10
    elif alphaMode == 20:
        mean = 20
    
    alphaInit = tf.random_normal_initializer(mean=mean,stddev=0)
    return tf.get_variable(name,shape,initializer=alphaInit)
#-----------------------------------------------------------------------------#
def fc_sigmoid(inputs,w,b,keepProb):
    sigmoid = tf.matmul(inputs,w) + b
    sigmoid = tf.nn.dropout(sigmoid,keepProb)
    sigmoid = tf.nn.sigmoid(sigmoid)
    return sigmoid
#-----------------------------------------------------------------------------#
def fc_relu(inputs,w,b,keepProb):
     relu = tf.matmul(inputs,w) + b
     relu = tf.nn.dropout(relu, keepProb)
     relu = tf.nn.relu(relu)
     return relu
#-----------------------------------------------------------------------------#
def fc(inputs,w,b,keepProb):
     fc = tf.matmul(inputs,w) + b
     fc = tf.nn.dropout(fc, keepProb)
     return fc
#-----------------------------------------------------------------------------#
def Classify(x, reuse=False, keepProb=1.0,isNankai=False):
    
    """
    4 layer fully-connected classification networks.
    Activation: relu -> relu -> none
    Dropout: keepProb
    
    Args:
        x: input data (feature vector, shape=[None, number of dimention])
        reuse=False: Train, reuse=True: Evaluation & Test (variables sharing)
    
    Returns:
        y: predicted target variables of class (one-hot vector)
    """
    with tf.variable_scope('Classify') as scope:  
        if reuse:
            scope.reuse_variables()
        
        # 1st layer
        w1 = weight_variable('w1',[dInput,nHidden])
        bias1 = bias_variable('bias1',[nHidden])
        h1 = fc_relu(x,w1,bias1,keepProb)
        
        # 2nd layer
        w2 = weight_variable('w2',[nHidden,nHidden2])
        bias2 = bias_variable('bias2',[nHidden2])
        h2 = fc_relu(h1,w2,bias2,keepProb) 
       
        # Toy
        if dataMode == 0:
            # 3rd layar
            w3 = weight_variable('w3',[nHidden2,nClass])
            bias3 = bias_variable('bias3',[nClass])
            y = fc(h2,w3,bias3,keepProb)
        # Nankai
        else:
            w3_1 = weight_variable('w3_1',[nHidden2,nClass])
            bias3_1 = bias_variable('bias3_1',[nClass])
            
            w3_2 = weight_variable('w3_2',[nHidden2,nClass])
            bias3_2 = bias_variable('bias3_2',[nClass])
            
            w3_3 = weight_variable('w3_3',[nHidden2,nClass])
            bias3_3 = bias_variable('bias3_3',[nClass])
            
            y1 = fc(h2,w3_1,bias3_1,keepProb)
            y2 = fc(h2,w3_2,bias3_2,keepProb)
            y3 = fc(h2,w3_3,bias3_3,keepProb)
            # [number of data, number of class, cell(=3)]
            y = tf.concat((tf.expand_dims(y1,2),tf.expand_dims(y2,2),tf.expand_dims(y3,2)),2)
    
        # shape=[None,number of class]
        return y
#-----------------------------------------------------------------------------#
def Regress(x_reg,reuse=False,isATR=False,depth=0,keepProb=1.0):
    
    """
    Fully-connected regression networks.
    
    Activation of Atr-nets: relu -> relu -> sigmoid
    Activation of Baseline regression & anchor-based: relu -> relu -> none
    Dropout: keepProb
    
    Args:
        x: input data (feature vector or residual, shape=[None, number of dimention])
        reuse=False: Train, reuse=True: Evaluation & Test (variables sharing)
        isR=False : atr-nets, isR=True : ordinary regression & anchor-based (in order to change output activation.)
        depth=3: 3layer, depth=4: 4layer, depth=5: 5layer
    
    Returns:
        y: predicted target variables or residual
    """
    
    with tf.variable_scope("Regress") as scope:  
        if reuse:
            scope.reuse_variables()

        # 1st layer
        w1_reg = weight_variable('w1_reg',[dInput,nRegHidden])
        bias1_reg = bias_variable('bias1_reg',[nRegHidden])
        h1 = fc_relu(x_reg,w1_reg,bias1_reg,keepProb)
        
        if depth == 3:
            # 2nd layer
            w2_reg = weight_variable('w2_reg',[nRegHidden,dOutput])
            bias2_reg = bias_variable('bias2_reg',[dOutput])
            
            if isATR:
                # shape=[None,number of dimention (y)]
                return fc_sigmoid(h1,w2_reg,bias2_reg,keepProb),w1_reg,w2_reg
            else:
                return fc(h1,w2_reg,bias2_reg,keepProb),w1_reg,w2_reg
        # ---------------------------------------------------------------------
        elif depth == 4:
            # 2nd layer
            w2_reg = weight_variable('w2_reg',[nRegHidden,nRegHidden2])
            bias2_reg = bias_variable('bias2_reg',[nRegHidden2])
            h2 = fc_relu(h1,w2_reg,bias2_reg,keepProb)
            
            # 3rd layer
            w3_reg = weight_variable('w3_reg',[nRegHidden2,dOutput])
            bias3_reg = bias_variable('bias3_reg',[dOutput])
            
            if isATR:
                return fc_sigmoid(h2,w3_reg,bias3_reg,keepProb),w1_reg,w2_reg
            else:
                return fc(h2,w3_reg,bias3_reg,keepProb),w1_reg,w2_reg
        # ---------------------------------------------------------------------
        elif depth == 5:
            # 2nd layer
            w2_reg = weight_variable('w2_reg',[nRegHidden,nRegHidden2])
            bias2_reg = bias_variable('bias2_reg',[nRegHidden2])
            h2 = fc_relu(h1,w2_reg,bias2_reg,keepProb)
            
            # 3rd layer 
            w3_reg = weight_variable('w3_reg',[nRegHidden2,nRegHidden3])
            bias3_reg = bias_variable('bias3_reg',[nRegHidden3])
            h3 = fc_relu(h2,w3_reg,bias3_reg,keepProb)
            
            # 4th layer
            w4_reg = weight_variable('w4_reg',[nRegHidden3,dOutput])
            bias4_reg = bias_variable('bias4_reg',[dOutput])
            
            if isATR:
                return fc_sigmoid(h3,w4_reg,bias4_reg,keepProb),w1_reg,w2_reg
            else:
                return fc(h3,w4_reg,bias4_reg,keepProb),w1_reg,w2_reg 
#-----------------------------------------------------------------------------#
def ResidualRegress(x_reg,reuse=False,isATR=False,depth=0,keepProb=1.0):
    
    """
    Fully-connected regression networks.
    
    Activation of Atr-nets: relu -> relu -> sigmoid
    Activation of Baseline regression & anchor-based: relu -> relu -> none
    Dropout: keepProb
    
    Args:
        x: input data (feature vector or residual, shape=[None, number of dimention])
        reuse=False: Train, reuse=True: Evaluation & Test (variables sharing)
        isR=False : atr-nets, isR=True : ordinary regression & anchor-based (in order to change output activation.)
        depth=3: 3layer, depth=4: 4layer, depth=5: 5layer
    
    Returns:
        y: predicted target variables or residual
    """
    
    with tf.variable_scope("ResidualRegress") as scope:  
        if reuse:
            scope.reuse_variables()

        # 1st layer
        w1_reg = weight_variable('w1_reg',[dOutput + dInput,nRegHidden])
        bias1_reg = bias_variable('bias1_reg',[nRegHidden])
        h1 = fc_relu(x_reg,w1_reg,bias1_reg,keepProb)
        
        if depth == 3:
            # 2nd layer
            w2_reg = weight_variable('w2_reg',[nRegHidden,dOutput])
            bias2_reg = bias_variable('bias2_reg',[dOutput])
            
            
            if isATR:
                # shape=[None,number of dimention (y)]
                return fc_sigmoid(h1,w2_reg,bias2_reg,keepProb),w1_reg,w2_reg
            else:
                return fc(h1,w2_reg,bias2_reg,keepProb),w1_reg,w2_reg
        # ---------------------------------------------------------------------
        elif depth == 4:
            # 2nd layer
            w2_reg = weight_variable('w2_reg',[nRegHidden,nRegHidden2])
            bias2_reg = bias_variable('bias2_reg',[nRegHidden2])
            h2 = fc_relu(h1,w2_reg,bias2_reg,keepProb)
            
            # 3rd layer
            w3_reg = weight_variable('w3_reg',[nRegHidden2,dOutput])
            bias3_reg = bias_variable('bias3_reg',[dOutput])
            
            if isATR:
                return fc_sigmoid(h2,w3_reg,bias3_reg,keepProb),w1_reg,w2_reg
            else:
                return fc(h2,w3_reg,bias3_reg,keepProb),w1_reg,w2_reg
        # ---------------------------------------------------------------------
        elif depth == 5:
            # 2nd layer
            w2_reg = weight_variable('w2_reg',[nRegHidden,nRegHidden2])
            bias2_reg = bias_variable('bias2_reg',[nRegHidden2])
            h2 = fc_relu(h1,w2_reg,bias2_reg,keepProb)
            
            # 3rd layer 
            w3_reg = weight_variable('w3_reg',[nRegHidden2,nRegHidden3])
            bias3_reg = bias_variable('bias3_reg',[nRegHidden3])
            h3 = fc_relu(h2,w3_reg,bias3_reg,keepProb)
            
            # 4th layer
            w4_reg = weight_variable('w4_reg',[nRegHidden3,dOutput])
            bias4_reg = bias_variable('bias4_reg',[dOutput])
            
            if isATR:
                return fc_sigmoid(h3,w4_reg,bias4_reg,keepProb),w1_reg,w2_reg
            else:
                return fc(h3,w4_reg,bias4_reg,keepProb),w1_reg,w2_reg 
#-----------------------------------------------------------------------------#
def CreateRegInputOutput(x,y,cls_score,isEval=False):
    
    """
    Create input vector(=cls_center_x) & anchor-based method GT output(=r) for Regress.
    
    Args:
        x: feature vector (input data) 
        cls_score: output in Classify (one-hot vector of predicted y class)
    
    Returns:
        pred_cls_center: center variable of class
        r: residual for regression (gt anchor-based) 
        cls_cener_x: center variable of class for regression input
    """
    if dataMode == 0:

        # Max class of predicted class
        pred_maxcls = tf.expand_dims(tf.cast(tf.argmax(cls_score,axis=1),tf.float32),1)  
        # Center variable of class        
        pred_cls_center = pred_maxcls * beta + first_cls_center
    
    else:
        # Max class of predicted class
        pred_maxcls1 = tf.expand_dims(tf.cast(tf.argmax(cls_score[:,:,0],axis=1),tf.float32),1)  
        # Max class of predicted class
        pred_maxcls2 = tf.expand_dims(tf.cast(tf.argmax(cls_score[:,:,1],axis=1),tf.float32),1)  
        # Max class of predicted class
        pred_maxcls3 = tf.expand_dims(tf.cast(tf.argmax(cls_score[:,:,2],axis=1),tf.float32),1)

        # Center variable of class for nankai       
        pred_cls_center1 = pred_maxcls1 * beta + first_cls_center_nk
        # Center variable of class for tonaki        
        pred_cls_center2 = pred_maxcls2 * beta + first_cls_center_tk
        # Center variable of class for tokai       
        pred_cls_center3 = pred_maxcls3 * beta + first_cls_center_tk
        # [number of data, cell(=3)] 
        pred_cls_center = tf.concat((pred_cls_center1,pred_cls_center2,pred_cls_center3),1)
    
    
    if isEval:
        return pred_cls_center, cls_center_x
    else:
        
        # residual = objective - center variavle of class 
        r = y - pred_cls_center
        # feature vector + center variable of class
        cls_center_x =  tf.concat((pred_cls_center,x),axis=1)
        
        return pred_cls_center, r, cls_center_x
#-----------------------------------------------------------------------------#
def TruncatedResidual(r,alpha_base,reuse=False):
    """
    Truncated range of residual by sigmoid function.
    
    Args:
        r: residual
        reuse=False: Train, reuse=True: Evaluation & Test (alpha sharing)
    
    Returns:
        r_at: trauncated range of residual
        alpha: traincated adjustment parameter
    """
    with tf.variable_scope('TrResidual') as scope:  
        if reuse:
            scope.reuse_variables()
        
        alpha = alpha_variable("alpha",[dOutput]) 
        alpha_final = tf.multiply(alpha,alpha_base)
        
        if istrainAlpha:
            r_at = 1/(1 + tf.exp(- alpha_final * r))
            return r_at, alpha_final
        else:
            r_at = 1/(1 + tf.exp(- alpha * r))
            return r_at, alpha
#-----------------------------------------------------------------------------#
#reduce_res_op = Reduce(reg_res,alpha,reuse=True)
def Reduce(r_at,param,reuse=False):
    """
    Reduce truncated residual(=r_at) to residual(=pred_r).
    
    Args:
        r_at: truncated residual
        param: truncated adjustment param (alpha)
        reuse=False: Train, reuse=True: Evaluation & Test (alpha sharing)
    
    Returns:
        pred_r: reduce residual 
    """
    with tf.variable_scope('TrResidual') as scope:  
        if reuse:
            scope.reuse_variables()

        #pred_r = (-1/param) * tf.log((1/r_at) - 1)
        pred_r = 1/param * tf.log(r_at/(1-r_at + 1e-8))
        
        return pred_r
#-----------------------------------------------------------------------------#    
def Loss(y,predict,isCE=False):
    """
    Loss function for Regress & Classify & alpha.
    Regress & alpha -> Mean Absolute Loss(MAE), Classify -> Cross Entropy(CE)
    
    Args:
        y: ground truth
        predict: predicted y
        isR=False: CE, isR=True: MAE
    """
    if isCE:
        if dataMode == 0:
            return tf.losses.softmax_cross_entropy(y,predict)
        else:
            return tf.losses.softmax_cross_entropy(y[:,:,0],predict[:,:,0]) + tf.losses.softmax_cross_entropy(y[:,:,1],predict[:,:,1]) + tf.losses.softmax_cross_entropy(y[:,:,2],predict[:,:,2])
    else:
        return tf.reduce_mean(tf.square(y - predict)) 
#-----------------------------------------------------------------------------#
def LossGroup(weight): 
    group_weight = tf.reduce_sum(tf.square(weight),axis=0)
    return group_weight
#-----------------------------------------------------------------------------#
def Optimizer(loss,name_scope="Regress"):
    """
    Optimizer for Regress & Classify & alpha.
    
    Args:
        loss: loss function
        name_scope: "Regress" or "Classify" or "TrResidual"
    """
    Vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=name_scope) 
    opt = tf.train.AdamOptimizer(lr).minimize(loss,var_list=Vars)
    return opt
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
# =================== Classification networks =========================== #
# IN -> x_cls: feature vector x = x1 + x2
# OUT -> cls_op: one-hot vector
cls_op = Classify(x_cls,keepProb=keepProbTrain)
cls_op_test = Classify(x_cls,reuse=True,keepProb=1.0)
#cls_op_eval = Classify(x_cls,reuse=True,keepProb=1.0)

# ============== Make Regression Input & Output ========================= #
# IN -> x_cls: feature vector x = x1 + x2, cls_op: one-hot vector
# OUT -> pred_cls_center: center of predicted class, res: gt residual, reg_in: x + pred_cls_center
pred_cls_center, res, reg_in = CreateRegInputOutput(x_cls,y,cls_op)
pred_cls_center_test, res_test, reg_in_test = CreateRegInputOutput(x_cls,y,cls_op_test)
#pred_cls_center_eval = CreateRegInputOutput(x_cls,y,cls_op_eval,isEval=True)
# ====================== Regression networks ============================ #
# IN -> x_reg: feature vector x = x1 + x2 (only baseline) or x + predicted center of class, 
#       isATR: bool (if ATR-Nets, isATR=True), depth: number of layer (command args)
# OUT -> reg_res: predicted of target variables y (baseline), predicted residual (Anchor-based), predicted truncated residual (ATR-Nets)
reg_res,res_w1,res_w2 = ResidualRegress(reg_in,isATR=isATR,depth=depth,keepProb=keepProbTrain)
reg_res_test,res_w1_test,res_w2_test = ResidualRegress(reg_in_test,reuse=True,isATR=isATR,depth=depth,keepProb=1.0)
#reg_res_eval = ResidualRegress(reg_in_eval,reuse=True,isATR=isATR,depth=depth,keepProb=1.0)

reg_y,reg_w1,reg_w2 = Regress(x_reg,isATR=isATR,depth=depth,keepProb=keepProbTrain)
reg_y_test,reg_w1_test,reg_w2_test = Regress(x_reg_test,reuse=True,isATR=isATR,depth=depth,keepProb=1.0)
#reg_y_eval = Regress(x_reg_eval,reuse=True,isATR=isATR,depth=depth,keepProb=1.0)

# ======================= L1 & L2 Loss ==================================== #
# ridge w residual var
res_l1 = tf.reduce_sum(tf.abs(res_w1)) + tf.reduce_sum(tf.abs(res_w2))
res_l1_test = tf.reduce_sum(tf.abs(res_w1_test)) + tf.reduce_sum(tf.abs(res_w2_test))
# l1 regression var
reg_l1 = tf.reduce_sum(tf.abs(reg_w1)) + tf.reduce_sum(tf.abs(reg_w2))
reg_l1_test = tf.reduce_sum(tf.abs(reg_w1_test)) + tf.reduce_sum(tf.abs(reg_w2_test))

# lass w
res_l2 = tf.nn.l2_loss(res_w1) + tf.nn.l2_loss(res_w2)
res_l2_test = tf.nn.l2_loss(res_w1_test) + tf.nn.l2_loss(res_w2_test)

reg_l2 = tf.nn.l2_loss(reg_w1) + tf.nn.l2_loss(reg_w2)
reg_l2_test = tf.nn.l2_loss(reg_w1_test) + tf.nn.l2_loss(reg_w2_test)

# =================== Truncated residual ================================ #
# IN -> res: residual, [None,1]
# OUT -> res_at: truncated range residual, [None,1], alpha: truncated parameter, [1]  
res_atr, alpha = TruncatedResidual(res,alpha_base)
res_atr_test, alpha_test = TruncatedResidual(res_test,alpha_base,reuse=True)

# ================== Reduce truncated residual ========================== #
# IN -> reg_res: predicted truncated regression
# OUT -> reduce_res: reduced residual, [None,1] 
reduce_res_op = Reduce(reg_res,alpha,reuse=True)
reduce_res_op_test = Reduce(reg_res_test,alpha_test,reuse=True)

# predicted y by ATR-Nets
pred_y = pred_cls_center + reduce_res_op
pred_y_test = pred_cls_center_test + reduce_res_op_test

# ============================= Loss ==================================== #
# Classification loss
# gt label (y_label) vs predicted label (cls_op)
loss_cls = Loss(y_label,cls_op,isCE=True)
loss_cls_test = Loss(y_label,cls_op_test,isCE=True)

# Ordinary regression loss train & test
# gt value (y) vs predicted value (reg_res)
loss_reg = Loss(y,reg_y)
loss_reg_test = Loss(y,reg_y_test)

# Ridge, Ordinary regression + L1regularization
loss_reg_l1 = loss_reg + reg_l1
loss_reg_l1_test = loss_reg_test + reg_l1_test
# Lasso, Ordinary regression + L2regularization
loss_reg_l2 = loss_reg + reg_l2
loss_reg_l2_test = loss_reg_test + reg_l2_test

# Regression loss for Anchor-based
# gt residual (res) vs predicted residual (res_op)
loss_anc = Loss(res,reg_res)
loss_anc_test = Loss(res_test,reg_res_test)

# Ridge, Ordinary regression + L1regularization
loss_anc_l1 = loss_anc + res_l1 
loss_anc_l1_test = loss_anc_test + res_l1_test

# Lasso, Ordinary regression + L2regularization
loss_anc_l2 = loss_anc + res_l2
loss_anc_l2_test = loss_anc_test + res_l2_test

# Regression loss for Atr-nets
# gt truncated residual (res_at) vs predicted truncated residual (res_op)
loss_atr = Loss(res_atr,reg_res)
loss_atr_test = Loss(res_atr_test,reg_res_test)

# Ridge, Ordinary regression + L1regularization
loss_atr_l1 = loss_atr + res_l1 
loss_atr_l1_test = loss_atr_test + res_l1_test 

# Lasso, Ordinary regression + L2regularization
loss_atr_l2 = loss_atr + res_l2
loss_atr_l2_test = loss_atr_test + res_l2_test


# Training alpha loss
# gt value (y) vs predicted value (pred_yz = pred_cls_center + reduce_res)
#loss_alpha = Loss(y,pred_y)
#loss_alpha_test = Loss(y,pred_y_test)
#_, var_train = tf.nn.moments(pred_y,[0])
#_, var_test = tf.nn.moments(pred_y_test,[0])
grad_x = tf.gradients(pred_y,x_cls)
grad_x_test = tf.gradients(pred_y,x_cls)
max_grad_x = tf.reduce_max(tf.abs(grad_x))
max_grad_x_test = tf.reduce_max(tf.abs(grad_x_test))
_, var_train = tf.nn.moments(grad_x[0],[0])
_, var_test = tf.nn.moments(grad_x_test[0],[0])
loss_alpha = max_grad_x #tf.reduce_sum(var_train)
loss_alpha_test = Loss(y,pred_y_test) + max_grad_x_test #tf.reduce_sum(var_test)

# ========================== Optimizer ================================== #
# for classification 
trainer_cls = Optimizer(loss_cls,name_scope="Classify")

# for Baseline regression
trainer_reg = Optimizer(loss_reg,name_scope="Regress")

# for Baseline regression + L1
trainer_reg_l1 = Optimizer(loss_reg_l1,name_scope="Regress")

# for Baseline regression + L2
trainer_reg_l2 = Optimizer(loss_reg_l2,name_scope="Regress")

# for Anchor-based regression
trainer_anc = Optimizer(loss_anc,name_scope="ResidualRegress")

# for Anchor-based regression + L1
trainer_anc_l1 = Optimizer(loss_anc_l1,name_scope="ResidualRegress")

# for Anchor-based regression + L2
trainer_anc_l2 = Optimizer(loss_anc_l2,name_scope="ResidualRegress")

# for Atr-nets regression
trainer_atr = Optimizer(loss_atr,name_scope="ResidualRegress")

# for Atr-nets regression
trainer_atr_l1 = Optimizer(loss_atr,name_scope="ResidualRegress")

# for Atr-nets regression/
trainer_atr_l2 = Optimizer(loss_atr,name_scope="ResidualRegress")

# for alpha training in atr-nets
trainer_alpha = Optimizer(loss_alpha,name_scope="TrResidual")

#------------------------------------------------------------------------ # 
if isSaveModel:
    # save model, every test steps
    saver = tf.train.Saver(max_to_keep=0)
#------------------------------------------------------------------------ # 
#------------------------------------------------------------------------ #
config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1,allow_growth=True)) 
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

if isRestoreModel:
    
    # restore saved model, latest
    savedfilePath = "{}{}".format(savePath,methodModel)
    savedmodelDir = os.path.join(modelPath,savedfilePath)
    if os.path.exists(savedmodelDir):
        saver = tf.train.Saver()
        saver.restore(sess,tf.train.latest_checkpoint(savedmodelDir))
        # select model
        #saver.restore(sess,os.path.join(savedmodelDir,"model_0_1-10000"))

#------------------------------------------------------------------------ #
#------------------------------------------------------------------------ #
# start training
flag = False
for i in range(nTraining):
    
    # Get mini-batch
    batchX,batchY,batchlabelY = myData.nextBatch(batchSize)
    #pdb.set_trace()
    if dataMode == 1:
    # ------------------------------------------------------------------- #
        # Change nankai date
        if i % filePeriod == 0:
            nameInds = random.sample(nametrInds,3) 
            myData.splitTrainValidation(nameInds=nameInds)
    # ------------------------------------------------------------------- #    
        
    # ==================== Ordinary regression ========================== #
    if methodModel == 0:
        
        if l1Mode == 1:        
            # l1 + ordinary regression
            _, trainPred, trainRegLoss = sess.run([trainer_reg_l1, reg_y, loss_reg_l1], feed_dict={x_reg:batchX, y:batchY}) 
        elif l2Mode == 1:
            # l2 + ordinary regression
            _, trainPred, trainRegLoss = sess.run([trainer_reg_l2, reg_y, loss_reg_l2], feed_dict={x_reg:batchX, y:batchY})
        
        else:
            # ordinary regression
            _, trainPred, trainRegLoss = sess.run([trainer_reg, reg_y, loss_reg], feed_dict={x_reg:batchX, y:batchY})
        # -------------------- Validation ------------------------------------- #
        if i % valPeriod == 0:   
            
            if l1Mode == 1:
                # l1 + regression
                testPred, testRegLoss = sess.run([reg_y_test, loss_reg_l1_test], feed_dict={x_reg_test:myData.xVal, y:myData.yVal})
            elif l2Mode == 1:
                # l2 + regression
                testPred, testRegLoss = sess.run([reg_y_test, loss_reg_l1_test], feed_dict={x_reg_test:myData.xVal, y:myData.yVal})
            else:
                # regression
                testPred, testRegLoss = sess.run([reg_y_test, loss_reg_test], feed_dict={x_reg_test:myData.xVal, y:myData.yVal})
        
            trainTotalVar = np.var(np.square(batchY - trainPred))
            testTotalVar = np.var(np.square(myData.yVal - testPred))
            #testBottom25 = np.sort(np.mean(np.square(myData.yVal - testPred),1))[:testPred.shape[0]*0.25]
            #testBottom75 = np.sort(np.mean(np.square(myData.yVal - testPred),1))[:testPred.shape[0]*0.75]
            #pdb.set_trace()
            print("tr:%d, trainRegLoss:%f, trainTotalVar:%f" % (i, trainRegLoss, trainTotalVar))
            print("itr:%d, testRegLoss:%f, testTotalVar:%f" % (i, testRegLoss, testTotalVar)) 
            # save model
            if isSaveModel:
                savemodelPath =  "{}{}".format(savePath,methodModel)
                modelfileName = "model_{}_{}".format(methodModel,trialID)
                savemodelDir = os.path.join(modelPath,savemodelPath)
                saver.save(sess,os.path.join(savemodelDir,modelfileName),global_step=i)
                
                # update checkpoint
                f = open(os.path.join(savemodelDir,"log.txt"),"a")
                f.write(modelfileName + "-" +  "{}".format(i) + "\n")
                #f.write("trainLoss:{},trainVar:{},testLoss:{},testVar:{}\n".format(trainRegLoss,trainTotalVar,testRegLoss,testTotalVar))
                f.close()
            
            # for saving test data 
            with open(os.path.join(pickleFullPath,"test_{}.pkl".format(savefilePath)),"wb") as fp:
                pickle.dump(myData.xVal,fp)
                pickle.dump(myData.yVal,fp)
                pickle.dump(testPred,fp)

            if not flag:
                trainRegLosses,testRegLosses = trainRegLoss[np.newaxis],testRegLoss[np.newaxis]
                flag = True
            else:
                trainRegLosses,testRegLosses = np.hstack([trainRegLosses,trainRegLoss[np.newaxis]]),np.hstack([testRegLosses,testRegLoss[np.newaxis]])
        
            # save nankai params
            savefilePath = "{}_{}_{}_{}_{}_{}_{}_{}".format(i,dataName,methodModel,depth,batchSize,l1Mode,l2Mode,trialID)

            myPlot.Plot_loss(0,0,0,0,trainRegLosses, testRegLosses,testPeriod,isPlot=isPlot, methodModel=methodModel, savefilePath=savefilePath)
    
    # ==================== Anchor-based regression ====================== #
    elif methodModel == 1:
        if l1Mode == 1:
            _, _, trainClsCenter, trainResPred, trainClsLoss, trainResLoss, trainRes = sess.run([trainer_cls, trainer_anc_l1, pred_cls_center, reg_res, loss_cls, loss_anc_l1, res],feed_dict={x_cls:batchX, y:batchY, y_label:batchlabelY})
        elif l2Mode == 1:
            _, _, trainClsCenter, trainResPred, trainClsLoss, trainResLoss, trainRes = sess.run([trainer_cls, trainer_anc_l2, pred_cls_center, reg_res, loss_cls, loss_anc_l2, res],feed_dict={x_cls:batchX, y:batchY, y_label:batchlabelY})
        else:
            _, _, trainClsCenter, trainResPred, trainClsLoss, trainResLoss, trainRes = sess.run([trainer_cls, trainer_anc, pred_cls_center, reg_res, loss_cls, loss_anc, res],feed_dict={x_cls:batchX, y:batchY, y_label:batchlabelY})
        # -------------------- Validation ------------------------------------- #
        if i % valPeriod == 0:
            if l1Mode == 1:
                testClsLoss, testResLoss, testClsCenter, testResPred  = sess.run([loss_cls_test, loss_anc_l1_test, pred_cls_center_test, reg_res_test], feed_dict={x_cls:myData.xVal, y:myData.yVal, y_label:myData.ylabelVal})
            elif l2Mode == 1:
                testClsLoss, testResLoss, testClsCenter, testResPred  = sess.run([loss_cls_test, loss_anc_l2_test, pred_cls_center_test, reg_res_test], feed_dict={x_cls:myData.xVal, y:myData.yVal, y_label:myData.ylabelVal})
            else:
                testClsLoss, testResLoss, testClsCenter, testResPred  = sess.run([loss_cls_test, loss_anc_test, pred_cls_center_test, reg_res_test], feed_dict={x_cls:myData.xVal, y:myData.yVal, y_label:myData.ylabelVal})
            
            # Reduce
            trainPred = trainClsCenter + trainResPred
            testPred = testClsCenter + testResPred     
        
            # total loss (mean) & variance
            trainTotalLoss = np.mean(np.square(batchY - trainPred))
            trainTotalVar  = np.var(np.square(batchY - trainPred))
            testTotalLoss  = np.mean(np.square(myData.yVal - testPred))
            testTotalVar  = np.var(np.square(myData.yVal - testPred))
            
            print("itr:%d,trainClsLoss:%f,trainRegLoss:%f, trainTotalLoss:%f, trainTotalVar:%f" % (i,trainClsLoss,trainResLoss, trainTotalLoss, trainTotalVar))
            print("itr:%d,testClsLoss:%f,testRegLoss:%f, testTotalLoss:%f, testTotalVar:%f" % (i,testClsLoss,testResLoss, testTotalLoss, testTotalVar)) 
            
            # save model
            if isSaveModel:
                savemodelPath =  "{}{}".format(savePath,methodModel)
                modelfileName = "model_{}_{}".format(methodModel,trialID)
                savemodelDir = os.path.join(modelPath,savemodelPath)
                saver.save(sess,os.path.join(savemodelDir,modelfileName),global_step=i)
                
                # update checkpoint
                f = open(os.path.join(savemodelDir,"log.txt"),"a")
                f.write(modelfileName + "-" +  "{}".format(i) + "\n")
                #f.write("trainVar:{},testVar:{}\n".format(trainTotalVar,testTotalVar))
                f.close()
            
            # for saving test data 
            with open(os.path.join(pickleFullPath,"test_{}.pkl".format(savefilePath)),"wb") as fp:
                pickle.dump(myData.xVal,fp)
                pickle.dump(myData.yVal,fp)
                pickle.dump(testClsCenter,fp)
                pickle.dump(testResPred,fp)
            
            if not flag:
                trainResLosses,testResLosses = trainResLoss[np.newaxis],testResLoss[np.newaxis]
                trainClassLosses,testClassLosses = trainClsLoss[np.newaxis],testClsLoss[np.newaxis]
                trainTotalLosses, testTotalLosses = trainTotalLoss[np.newaxis],testTotalLoss[np.newaxis]
                flag = True
            else:
                trainResLosses,testResLosses = np.hstack([trainResLosses,trainResLoss[np.newaxis]]),np.hstack([testResLosses,testResLoss[np.newaxis]])
                trainClassLosses,testClassLosses = np.hstack([trainClassLosses,trainClsLoss[np.newaxis]]),np.hstack([testClassLosses,testClsLoss[np.newaxis]])
                trainTotalLosses,testTotalLosses = np.hstack([trainTotalLosses,trainTotalLoss[np.newaxis]]),np.hstack([testTotalLosses,testTotalLoss[np.newaxis]])
        
            savefilePath = "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(i,dataName,methodModel,nClass,depth,batchSize,l1Mode,l2Mode,trialID)
           
            myPlot.Plot_loss(trainTotalLosses, testTotalLosses, trainClassLosses, testClassLosses, trainResLosses, testResLosses, valPeriod, isPlot=isPlot, methodModel=methodModel, savefilePath=savefilePath)
    # ======================== Atr-Nets ================================= #
    elif methodModel == 2:

        if i==0:
            alpha_base_value = 0.1
        if istrainAlpha:
            _, _, _, trainClsCenter, trainResPred, trainAlpha, trainClsLoss, trainResLoss, trainAlphaLoss, trainRResPred, grad_x_value = \
            sess.run([trainer_cls, trainer_atr, trainer_alpha, pred_cls_center, reg_res, alpha, loss_cls, loss_atr, loss_alpha, reduce_res_op, grad_x],feed_dict={x_cls:batchX, y:batchY, y_label:batchlabelY})
            
        
        else:
            if l1Mode == 1:
                 # fixing alpha
                 _, _, trainClsCenter, trainCls, trainResPred, trainAlpha, trainClsLoss, trainResLoss, trainRResPred = \
                 sess.run([trainer_cls, trainer_atr_l1, pred_cls_center, cls_op, reg_res, alpha, loss_cls, loss_atr, reduce_res_op],feed_dict={x_cls:batchX, y:batchY, y_label:batchlabelY,alpha_base:alpha_base_value})

            elif l2Mode == 1:
                 # fixing alpha
                 _, _, trainClsCenter, trainCls, trainResPred, trainAlpha, trainClsLoss, trainResLoss, trainRResPred = \
                 sess.run([trainer_cls, trainer_atr_l2, pred_cls_center, cls_op, reg_res, alpha, loss_cls, loss_atr_l2, reduce_res_op],feed_dict={x_cls:batchX, y:batchY, y_label:batchlabelY,alpha_base:alpha_base_value})

            else:
                 # fixing alpha
                 _, _, trainClsCenter, trainCls, trainResPred, trainAlpha, trainClsLoss, trainResLoss, trainRResPred = \
                 sess.run([trainer_cls, trainer_atr, pred_cls_center, cls_op, reg_res, alpha, loss_cls, loss_atr, reduce_res_op],feed_dict={x_cls:batchX, y:batchY, y_label:batchlabelY,alpha_base:alpha_base_value})


            #_, trainAlpha, trainAlphaLoss, grad_x_value, max_grad_x_value = \
            #sess.run([trainer_alpha, alpha, loss_alpha, grad_x, max_grad_x],feed_dict={x_cls:batchX, y:batchY, y_label:batchlabelY})

        alpha_base_value = np.max([0.01,trainTotalLoss])
        
            
        # -------------------- Validation ------------------------------------- #
        if i % valPeriod == 0:
            #------------
            # entropy 
            if dataMode == 0:
                e = np.exp(trainCls - np.tile(np.max(trainCls,axis=1,keepdims=True),[1,nClass]))
                #prob=np.exp(trainCls)/np.tile(np.sum(np.exp(trainCls),axis=1,keepdims=True),[1,nClass])
                prob = e/np.tile(np.sum(e,axis=1,keepdims=True),[1,nClass])
                entropy = np.mean(np.sum(prob*np.log(prob+10e-5),axis=1))
                print(f"entropy = {entropy}")
            
            if l1Mode == 1:
                testClsCenter, testResPred, testAlpha, testClsLoss, testResLoss, testAlphaLoss, testRResPred = \
                sess.run([pred_cls_center_test, reg_res_test, alpha_test, loss_cls_test, loss_atr_l1_test, loss_alpha_test, reduce_res_op_test],feed_dict={x_cls:myData.xVal, y:myData.yVal, y_label:myData.ylabelVal,alpha_base:alpha_base_value})
            elif l2Mode == 1:
                testClsCenter, testResPred, testAlpha, testClsLoss, testResLoss, testAlphaLoss, testRResPred = \
                sess.run([pred_cls_center_test, reg_res_test, alpha_test, loss_cls_test, loss_atr_l2_test, loss_alpha_test, reduce_res_op_test],feed_dict={x_cls:myData.xVal, y:myData.yVal, y_label:myData.ylabelVal,alpha_base:alpha_base_value})
            else:
                testClsCenter, testResPred, testAlpha, testClsLoss, testResLoss, testAlphaLoss, testRResPred = \
                sess.run([pred_cls_center_test, reg_res_test, alpha_test, loss_cls_test, loss_atr_test, loss_alpha_test, reduce_res_op_test],feed_dict={x_cls:myData.xVal, y:myData.yVal, y_label:myData.ylabelVal,alpha_base:alpha_base_value})
            
            TrainResidualat = 1/(1+np.exp(-trainAlpha * trainResPred))
            TrainBigResidual = np.where((0.0==TrainResidualat)|(TrainResidualat==1.0))
            bigResidualpar = TrainBigResidual[0].shape[0] / batchY.shape[0]
            
            TestTrRes = 1/(1+np.exp(-testAlpha * testResPred))
            TestBigResidual = np.where((0.0==TestTrRes)|(TestTrRes==1.0))
            TestbigResidualpar = TestBigResidual[0].shape[0] / myData.yVal.shape[0]
            
            # Recover
            trainPred = trainClsCenter + trainRResPred
            testPred = testClsCenter + testRResPred
    
            # total loss (mean) & variance
            trainTotalLoss = np.mean(np.square(batchY - trainPred))
            trainTotalVar  = np.var(np.square(batchY - trainPred))
            testTotalLoss  = np.mean(np.square(myData.yVal - testPred))
            testTotalVar  = np.var(np.square(myData.yVal - testPred))
            
            print("Test Alpha", testAlpha)
            print("BigTrainResidual割合", bigResidualpar)
            print("BigTestResidual割合", TestbigResidualpar)
            print("-----------------------------------")
            print("itr:%d,trainClsLoss:%f,trainRegLoss:%f, trainTotalLoss:%f, trainTotalVar:%f" % (i,trainClsLoss,trainResLoss, trainTotalLoss, trainTotalVar))
            print("itr:%d,testClsLoss:%f,testRegLoss:%f, testTotalLoss:%f, testTotalVar:%f" % (i,testClsLoss,testResLoss, testTotalLoss, testTotalVar)) 
            
            # save model
            if isSaveModel:
                strnClass = nClass -1
                savemodelPath =  "{}{}{}".format(savePath,methodModel,strnClass)
                modelfileName = "model_{}_{}".format(methodModel,trialID)
                savemodelDir = os.path.join(modelPath,savemodelPath)
                saver.save(sess,os.path.join(savemodelDir,modelfileName),global_step=i)
                
                # update checkpoint
                f = open(os.path.join(savemodelDir,"log.txt"),"a")
                f.write(modelfileName + "-" +  "{}".format(i) + "\n")
                #f.write("trainVar:{},testVar:{}\n".format(trainTotalVar,testTotalVar))
                f.close()
            # for saving test data 
            with open(os.path.join(pickleFullPath,"test_{}.pkl".format(savefilePath)),"wb") as fp:
                pickle.dump(myData.xVal,fp)
                pickle.dump(myData.yVal,fp)
                pickle.dump(testClsCenter,fp)
                pickle.dump(testResPred,fp)
            
            if not flag:
                trainResLosses,testResLosses = trainResLoss[np.newaxis],testResLoss[np.newaxis]
                trainClassLosses,testClassLosses = trainClsLoss[np.newaxis],testClsLoss[np.newaxis]
                trainTotalLosses,testTotalLosses = trainTotalLoss[np.newaxis],testTotalLoss[np.newaxis]
                flag = True
            else:
                trainResLosses,testResLosses = np.hstack([trainResLosses,trainResLoss[np.newaxis]]),np.hstack([testResLosses,testResLoss[np.newaxis]])
                trainClassLosses,testClassLosses = np.hstack([trainClassLosses,trainClsLoss[np.newaxis]]),np.hstack([testClassLosses,testClsLoss[np.newaxis]])
                trainTotalLosses,testTotalLosses = np.hstack([trainTotalLosses,trainTotalLoss[np.newaxis]]),np.hstack([testTotalLosses,testTotalLoss[np.newaxis]])

            savefilePath = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(i,dataName,methodModel,nClass,depth,batchSize,testAlpha,l1Mode,l2Mode,trialID)
            

            myPlot.Plot_loss(trainTotalLosses, testTotalLosses, trainClassLosses, testClassLosses, trainResLosses, testResLosses, valPeriod, isPlot=isPlot, methodModel=methodModel, savefilePath=savefilePath)
# ------------------------- plot loss & toydata ------------------------- #
if methodModel == 0:
    
    if dataMode == 0:
        savefilePath = "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(dataName,methodModel,sigma,pNum,depth,batchSize,nData,trainRatio,trialID)
        myPlot.Plot_3D(myData.xTest[:,0],myData.xTest[:,1],myData.yTest,testPred, isPlot=isPlot, savefilePath=savefilePath)
        
    else:
        savefilePath = "{}_{}_{}_{}_{}_{}_{}_{}".format(i,dataName,methodModel,depth,batchSize,l1Mode,l2Mode,trialID)
        #myPlot.Plot_Scatter(myData.yTest,testPred,isPlot=isPlot,savefilePath=savefilePath)
        
    myPlot.Plot_loss(0,0,0,0,trainRegLosses, testRegLosses,valPeriod,isPlot=isPlot, methodModel=methodModel, savefilePath=savefilePath)
    
    # index of minimum variladation loss
    minInd = (np.argsort(testRegLosses)[0] + 1) * 100
    savefilePath = "{}_{}_{}_{}_{}_{}_{}_{}".format(minInd,dataName,methodModel,depth,batchSize,l1Mode,l2Mode,trialID)
    # loading test data
    with open(os.path.join(os.path.join(featurePath,testPath),"test_{}.pkl".format(savefilePath)),"wb") as fp:
        pickle.dump(xTest,fp)
        pickle.dump(yTest,fp)
        pickle.dump(yPred,fp)

    with open(os.path.join(pickleFullPath,"test_{}.pkl".format(savefilePath)),"wb") as fp:
            pickle.dump(batchY,fp)
            pickle.dump(trainPred,fp)
            pickle.dump(myData.yVar,fp)
            pickle.dump(testPred,fp)
            pickle.dump(trainTotalVar,fp)
            pickle.dump(testTotalVar,fp)
            pickle.dump(trainRegLosses,fp)
            pickle.dump(testRegLosses,fp)
    
        
    # ----------------------------------------------------------------------- #
elif methodModel == 1:
    
    if dataMode == 0:
        savefilePath = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(dataName,methodModel,sigma,nClass,pNum,depth,batchSize,nData,trainRatio,trialID)
        myPlot.Plot_3D(myData.xTest[:,0],myData.xTest[:,1],myData.yTest,testPred, isPlot=isPlot, savefilePath=savefilePath)
    else:
        savefilePath = "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(i,dataName,methodModel,nClass,depth,batchSize,l1Mode,l2Mode,trialID)
        #myPlot.Plot_Scatter(myData.yTest,testPred,isPlot=isPlot,savefilePath=savefilePath)
        
    myPlot.Plot_loss(trainTotalLosses, testTotalLosses, trainClassLosses, testClassLosses, trainResLosses, testResLosses, valPeriod, isPlot=isPlot, methodModel=methodModel, savefilePath=savefilePath)
    
    # -------------------- Test ----------------------------------------- #
    # index of minimum validation loss
    pdb.set_trace()
    minValloss = (np.argsort(testTotalLosses)[0] + 1) * 100
    savefilePath = "{}_{}_{}_{}_{}_{}_{}_{}".format(minInd,dataName,methodModel,depth,batchSize,l1Mode,l2Mode,trialID) 
    # loading test data
    with open(os.path.join(os.path.join(featurePath,testPath),"test_{}.pkl".format(savefilePath)),"wb") as fp:
        pickle.dump(xTest,fp)
        pickle.dump(yTest,fp)
        pickle.dump(yPred,fp)

    with open(os.path.join(pickleFullPath,"test_{}.pkl".format(savefilePath)),"wb") as fp:
        pickle.dump(batchY,fp)
        pickle.dump(trainPred,fp)
        pickle.dump(myData.yVal,fp)
        pickle.dump(testPred,fp)
        pickle.dump(trainTotalVar,fp)
        pickle.dump(testTotalVar,fp)
        pickle.dump(trainClassLosses,fp)
        pickle.dump(testClassLosses,fp)
        pickle.dump(trainResLosses,fp)
        pickle.dump(testResLosses,fp)
        pickle.dump(trainTotalLosses,fp)
        pickle.dump(testTotalLosses,fp)
    
    
# ----------------------------------------------------------------------- #
elif methodModel == 2: 
    
    if dataMode == 0:
        savefilePath = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(dataName,methodModel,sigma,nClass,pNum,depth,batchSize,nData,trainRatio,testAlpha,trialID)
        myPlot.Plot_3D(myData.xTest[:,0],myData.xTest[:,1],myData.yTest,testPred, isPlot=isPlot, savefilePath=savefilePath)
    else:
        savefilePath = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(i,dataName,methodModel,nClass,depth,batchSize,testAlpha,l1Mode,l2Mode,trialID)
        myPlot.Plot_Scatter(myData.yTest,testPred,isPlot=isPlot,savefilePath=savefilePath)
        
    myPlot.Plot_loss(trainTotalLosses, testTotalLosses, trainClassLosses, testClassLosses, trainResLosses, testResLosses, valPeriod, isPlot=isPlot, methodModel=methodModel, savefilePath=savefilePath)
    
    # index of minimum validation loss
    minValloss = (np.argsort(testTotalLosses)[0]
    savefilePath = "{}_{}_{}_{}_{}_{}_{}_{}".format(minInd,dataName,methodModel,depth,batchSize,l1Mode,l2Mode,trialID) 
    # loading test data
    with open(os.path.join(os.path.join(featurePath,testPath),"test_{}.pkl".format(savefilePath)),"wb") as fp:
        pickle.dump(xTest,fp)
        pickle.dump(yTest,fp)
        pickle.dump(yPred,fp)

    with open(os.path.join(pickleFullPath,"test_{}.pkl".format(savefilePath)),"wb") as fp:
        pickle.dump(batchY,fp)
        pickle.dump(trainPred,fp)
        pickle.dump(myData.yVal,fp)
        pickle.dump(testPred,fp)
        pickle.dump(trainTotalVar,fp)
        pickle.dump(testTotalVar,fp)
        pickle.dump(trainClassLosses,fp)
        pickle.dump(testClassLosses,fp)
        pickle.dump(trainResLosses,fp)
        pickle.dump(testResLosses,fp)
        pickle.dump(trainTotalLosses,fp)
        pickle.dump(testTotalLosses,fp)
# ----------------------------------------------------------------------- #  
