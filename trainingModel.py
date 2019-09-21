# -*- co    ding: utf-8 -*-

import sys
import os

import numpy as np
import tensorflow as tf

import pickle
import pdb

import makingData
import loadingNankai
import Plot as myPlot

# -------------------------- command argment ----------------------------------
# Model type 0: ordinary regression, 1: anhor-based, 2: atr-nets
methodModel = int(sys.argv[1])
# noize of x1, x2
sigma = np.float(sys.argv[2])
# number of class
# nankai nClass = 10 or 12 or 20 or 50
nClass = int(sys.argv[3])
# number of rotation -> sin(pNum*pi) & cos(pNum*pi)
pNum = int(sys.argv[4])
# number of layer for Regression NN
depth = int(sys.argv[5])
# batch size
batchSize = int(sys.argv[6])
# trial ID
trialID = int(sys.argv[7])
# data size
nData = int(sys.argv[8])
trainRatio = float(sys.argv[9])
# 0: toydata var., 1: nankai var.
dataMode = int(sys.argv[10])
# -----------------------------------------------------------------------------

# ------------------------------- path ----------------------------------------
results = "results"
pickles = "pickles"
pickleFullPath = os.path.join(results,pickles)
# -----------------------------------------------------------------------------

# --------------------------- parameters --------------------------------------
# number of nankai cell
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
    dInput = 2
    # node of output in Regression 
    nRegOutput = 1    
    # round decimal 
    limitdecimal = 3
    # Width class
    beta = np.round((yMax - yMin) / nClass,limitdecimal)

# Nankai
else:
    dInput = nCell*nWindow
    # node of output in Regression 
    nRegOutput = 3    
    # round decimal 
    limitdecimal = 6
    # Width class
    beta = np.round((nkMax - nkMin) / nClass,limitdecimal)

# Center variable of the first class
first_cls_center = np.round(yMin + (beta / 2),limitdecimal)
# Center variable of the first class in nankai
first_cls_center_nk = np.round(nkMin + (beta / 2),limitdecimal)
# Center variable of the first class in tonankai & tokai
first_cls_center_tk = np.round(tkMin + (beta / 2),limitdecimal)


# select nankai data 
nameInds = [0,1,2] 

# dropout
keepProbTrain = 1.0
# Learning rate
lr = 1e-3
# number of training
nTraining = 100000
# test count
testPeriod = 500
# if plot == True
isPlot = True
# -----------------------------------------------------------------------------

# --------------------------- data --------------------------------------------
# Get train & test data, shape=[number of data, dimention]
print(pNum)
# select toydata or nankaidata
if dataMode == 0:    
    myData = makingData.toyData(trainRatio=trainRatio, nData=nData, pNum=pNum, sigma=sigma)
    myData.createData(trialID,beta=beta)
    myData.loadNankaiRireki()
else:
    myData = loadingNankai.NankaiData(nClass=nClass)
    myData.loadTrainTestData(nameInds=nameInds)
# -----------------------------------------------------------------------------

#------------------------- placeholder ----------------------------------------
# input of placeholder for classification
x_cls = tf.placeholder(tf.float32,shape=[None,dInput])
# input of placeholder for regression
x_reg = tf.placeholder(tf.float32,shape=[None,dInput])
x_reg_test = tf.placeholder(tf.float32,shape=[None,dInput])
# GT output of placeholder (target)
y = tf.placeholder(tf.float32,shape=[None,1])
# GT output of label
y_label = tf.placeholder(tf.int32,shape=[None,nClass])

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
    alphaInit = tf.random_normal_initializer(mean=20,stddev=0)
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
def Classify(x,reuse=False, keepProb=1.0):
    
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
        
        # 3rd layar
        w3 = weight_variable('w3',[nHidden2,nClass])
        bias3 = bias_variable('bias3',[nClass])
        y = fc(h2,w3,bias3,keepProb)
        
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
            w2_reg = weight_variable('w2_reg',[nRegHidden,nRegOutput])
            bias2_reg = bias_variable('bias2_reg',[nRegOutput])
            
            if isATR:
                # shape=[None,number of dimention (y)]
                return fc_sigmoid(h1,w2_reg,bias2_reg,keepProb)
            else:
                return fc(h1,w2_reg,bias2_reg,keepProb)
        # ---------------------------------------------------------------------
        elif depth == 4:
            # 2nd layer
            w2_reg = weight_variable('w2_reg',[nRegHidden,nRegHidden2])
            bias2_reg = bias_variable('bias2_reg',[nRegHidden2])
            h2 = fc_relu(h1,w2_reg,bias2_reg,keepProb)
            
            # 3rd layer
            w3_reg = weight_variable('w3_reg',[nRegHidden2,nRegOutput])
            bias3_reg = bias_variable('bias3_reg',[nRegOutput])
            
            if isATR:
                return fc_sigmoid(h2,w3_reg,bias3_reg,keepProb)
            else:
                return fc(h2,w3_reg,bias3_reg,keepProb)
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
            w4_reg = weight_variable('w4_reg',[nRegHidden3,nRegOutput])
            bias4_reg = bias_variable('bias4_reg',[nRegOutput])
            
            if isATR:
                return fc_sigmoid(h3,w4_reg,bias4_reg,keepProb)
            else:
                return fc(h3,w4_reg,bias4_reg,keepProb) 
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
        w1_reg = weight_variable('w1_reg',[nRegOutput + dInput,nRegHidden])
        bias1_reg = bias_variable('bias1_reg',[nRegHidden])
        h1 = fc_relu(x_reg,w1_reg,bias1_reg,keepProb)
        
        if depth == 3:
            # 2nd layer
            w2_reg = weight_variable('w2_reg',[nRegHidden,nRegOutput])
            bias2_reg = bias_variable('bias2_reg',[nRegOutput])
            
            
            if isATR:
                # shape=[None,number of dimention (y)]
                return fc_sigmoid(h1,w2_reg,bias2_reg,keepProb)
            else:
                return fc(h1,w2_reg,bias2_reg,keepProb)
        # ---------------------------------------------------------------------
        elif depth == 4:
            # 2nd layer
            w2_reg = weight_variable('w2_reg',[nRegHidden,nRegHidden2])
            bias2_reg = bias_variable('bias2_reg',[nRegHidden2])
            h2 = fc_relu(h1,w2_reg,bias2_reg,keepProb)
            
            # 3rd layer
            w3_reg = weight_variable('w3_reg',[nRegHidden2,nRegOutput])
            bias3_reg = bias_variable('bias3_reg',[nRegOutput])
            
            if isATR:
                return fc_sigmoid(h2,w3_reg,bias3_reg,keepProb)
            else:
                return fc(h2,w3_reg,bias3_reg,keepProb)
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
            w4_reg = weight_variable('w4_reg',[nRegHidden3,nRegOutput])
            bias4_reg = bias_variable('bias4_reg',[nRegOutput])
            
            if isATR:
                return fc_sigmoid(h3,w4_reg,bias4_reg,keepProb)
            else:
                return fc(h3,w4_reg,bias4_reg,keepProb) 
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
    pdb.set_trace()
    # Max class of predicted class
    pred_maxcls = tf.expand_dims(tf.cast(tf.argmax(cls_score,axis=1),tf.float32),1)  
    # Center variable of class        
    pred_cls_center = pred_maxcls * beta + first_cls_center
    # feature vector + center variable of class
    cls_center_x =  tf.concat((pred_cls_center,x),axis=1)
    # residual = objective - center variavle of class 
    r = y - pred_cls_center
    
    if isEval:
        return pred_cls_center
    else:
        return pred_cls_center, r, cls_center_x
#-----------------------------------------------------------------------------#
def TruncatedResidual(r,reuse=False):
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
        
        alpha = alpha_variable("alpha",[1]) 
        
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
        return tf.losses.softmax_cross_entropy(y,predict)
    else:
        return tf.reduce_mean(tf.abs(y - predict))
#-----------------------------------------------------------------------------#
def LossGroup(self,weight1): 
    group_weight = tf.reduce_sum(tf.square(weight1),axis=0)
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

# ============== Make Regression Input & Output ========================= #
# IN -> x_cls: feature vector x = x1 + x2, cls_op: one-hot vector
# OUT -> pred_cls_center: center of predicted class, res: gt residual, reg_in: x + pred_cls_center
pred_cls_center, res, reg_in = CreateRegInputOutput(x_cls,y,cls_op)
pred_cls_center_test, res_test, reg_in_test = CreateRegInputOutput(x_cls,y,cls_op_test)
pred_cls_center_eval = CreateRegInputOutput(x_cls,y,cls_op_test,isEval=True)
# ====================== Regression networks ============================ #
# IN -> x_reg: feature vector x = x1 + x2 (only baseline) or x + predicted center of class, 
#       isATR: bool (if ATR-Nets, isATR=True), depth: number of layer (command args)
# OUT -> reg_res: predicted of target variables y (baseline), predicted residual (Anchor-based), predicted truncated residual (ATR-Nets)
reg_res = ResidualRegress(reg_in,isATR=isATR,depth=depth,keepProb=keepProbTrain)
reg_res_test = ResidualRegress(reg_in_test,reuse=True,isATR=isATR,depth=depth,keepProb=1.0)

reg_y = Regress(x_reg,isATR=isATR,depth=depth,keepProb=keepProbTrain)
reg_y_test = Regress(x_reg_test,reuse=True,isATR=isATR,depth=depth,keepProb=1.0)

# =================== Truncated residual ================================ #
# IN -> res: residual, [None,1]
# OUT -> res_at: truncated range residual, [None,1], alpha: truncated parameter, [1]  
res_atr, alpha = TruncatedResidual(res)
res_atr_test, alpha_test = TruncatedResidual(res_test,reuse=True)

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

# Baseline regression loss train & test
# gt value (y) vs predicted value (reg_res)
loss_reg = Loss(y,reg_y)
loss_reg_test = Loss(y,reg_y_test)

# Regression loss for Anchor-based
# gt residual (res) vs predicted residual (res_op)
loss_anc = Loss(res,reg_res)
loss_anc_test = Loss(res_test,reg_res_test)

# Regression loss for Atr-nets
# gt truncated residual (res_at) vs predicted truncated residual (res_op)
loss_atr = Loss(res_atr,reg_res)
loss_atr_test = Loss(res_atr_test,reg_res_test)

# Training alpha loss
# gt value (y) vs predicted value (pred_yz = pred_cls_center + reduce_res)
loss_alpha = Loss(y,pred_y)
loss_alpha_test = Loss(y,pred_y_test)

# ========================== Optimizer ================================== #
# for classification 
trainer_cls = Optimizer(loss_cls,name_scope="Classify")

# for Baseline regression
trainer_reg = Optimizer(loss_reg,name_scope="Regress")

# for Anchor-based regression
trainer_anc = Optimizer(loss_anc,name_scope="ResidualRegress")

# for Atr-nets regression
trainer_atr = Optimizer(loss_atr,name_scope="ResidualRegress")

# for alpha training in atr-nets
trainer_alpha = Optimizer(loss_alpha,name_scope="TrResidual")

#------------------------------------------------------------------------ # 
config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1,allow_growth=True)) 
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
# save model
saver = tf.train.Saver()
#------------------------------------------------------------------------ #
#------------------------------------------------------------------------ #
# start training
flag = False
for i in range(nTraining):
    
    # Get mini-batch
    batchX,batchY,batchlabelY = myData.nextBatch(batchSize)
    
    # ==================== Baseline regression ========================== #
    if methodModel == 0:
        # regression
        _, trainPred, trainRegLoss = sess.run([trainer_reg, reg_y, loss_reg], feed_dict={x_reg:batchX, y:batchY})
        
        # -------------------- Test ------------------------------------- #
        if i % testPeriod == 0:   
            # regression
            testPred, testRegLoss = sess.run([reg_y_test, loss_reg_test], feed_dict={x_reg_test:myData.xTest, y:myData.yTest})
        
            trainTotalVar  = np.var(np.abs(batchY - trainPred))
            testTotalVar = np.var(np.abs(myData.yTest - testPred))
            
            print("itr:%d, trainRegLoss:%f, trainTotalVar:%f" % (i, trainRegLoss, trainTotalVar))
            print("itr:%d, testRegLoss:%f, testTotalVar:%f" % (i, testRegLoss, testTotalVar)) 
            
            
            if not flag:
                trainRegLosses,testRegLosses = trainRegLoss[np.newaxis],testRegLoss[np.newaxis]
                flag = True
            else:
                trainRegLosses,testRegLosses = np.hstack([trainRegLosses,trainRegLoss[np.newaxis]]),np.hstack([testRegLosses,testRegLoss[np.newaxis]])
        
    # ==================== Anchor-based regression ====================== #
    elif methodModel == 1:
        _, _, trainClsCenter, trainResPred, trainClsLoss, trainResLoss, trainRes = sess.run([trainer_cls, trainer_anc, pred_cls_center, reg_res, loss_anc, loss_cls, res],feed_dict={x_cls:batchX, y:batchY, y_label:batchlabelY})
        
        # -------------------- Test ------------------------------------- #
        if i % testPeriod == 0:

            testClsLoss, testResLoss, testClsCenter, testResPred  = sess.run([loss_cls_test, loss_anc_test, pred_cls_center_test, reg_res_test], feed_dict={x_cls:myData.xTest, y:myData.yTest, y_label:myData.yTestLabel})

            
            # Reduce
            trainPred = trainClsCenter + trainResPred
            testPred = testClsCenter + testResPred     
            
            # total loss (mean) & variance
            trainTotalLoss = np.mean(np.abs(batchY - trainPred))
            trainTotalVar  = np.var(np.abs(batchY - trainPred))
            testTotalLoss  = np.mean(np.abs(myData.yTest - testPred))
            testTotalVar  = np.var(np.abs(myData.yTest - testPred))
            
            print("itr:%d,trainClsLoss:%f,trainRegLoss:%f, trainTotalLoss:%f, trainTotalVar:%f" % (i,trainClsLoss,trainResLoss, trainTotalLoss, trainTotalVar))
            print("itr:%d,testClsLoss:%f,testRegLoss:%f, testTotalLoss:%f, testTotalVar:%f" % (i,testClsLoss,testResLoss, testTotalLoss, testTotalVar)) 
            
            if not flag:
                trainResLosses,testResLosses = trainResLoss[np.newaxis],testResLoss[np.newaxis]
                trainClassLosses,testClassLosses = trainClsLoss[np.newaxis],testClsLoss[np.newaxis]
                trainTotalLosses, testTotalLosses = trainTotalLoss[np.newaxis],testTotalLoss[np.newaxis]
                flag = True
            else:
                trainResLosses,testResLosses = np.hstack([trainResLosses,trainResLoss[np.newaxis]]),np.hstack([testResLosses,testResLoss[np.newaxis]])
                trainClassLosses,testClassLosses = np.hstack([trainClassLosses,trainClsLoss[np.newaxis]]),np.hstack([testClassLosses,testClsLoss[np.newaxis]])
                trainTotalLosses,testTotalLosses = np.hstack([trainTotalLosses,trainTotalLoss[np.newaxis]]),np.hstack([testTotalLosses,testTotalLoss[np.newaxis]])
        
    # ======================== Atr-Nets ================================= #
    elif methodModel == 2:

        #_, _, _, trainClsCenter, trainResPred, trainAlpha, trainClsLoss, trainResLoss, trainAlphaLoss, trainRResPred = \
        #sess.run([trainer_cls, trainer_atr, trainer_alpha, pred_cls_center, reg_res, alpha, loss_cls, loss_atr, loss_alpha, reduce_res_op],feed_dict={x_cls:batchX, y:batchY, y_label:batchlabelY})
        _, _, trainClsCenter, trainResPred, trainAlpha, trainClsLoss, trainResLoss, trainRResPred = \
        sess.run([trainer_cls, trainer_atr, pred_cls_center, reg_res, alpha, loss_cls, loss_atr, reduce_res_op],feed_dict={x_cls:batchX, y:batchY, y_label:batchlabelY})
        '''
        _, trainAlpha, trainAlphaLoss = \
        sess.run([trainer_alpha, alpha, loss_alpha],feed_dict={x_cls:batchX, y:batchY, y_label:batchlabelY})
        '''

        # -------------------- Test ------------------------------------- #
        if i % testPeriod == 0:
        
            testClsCenter, testResPred, testAlpha, testClsLoss, testResLoss, testAlphaLoss, testRResPred = \
            sess.run([pred_cls_center_test, reg_res_test, alpha_test, loss_cls_test, loss_atr_test, loss_alpha_test, reduce_res_op_test],feed_dict={x_cls:myData.xTest, y:myData.yTest, y_label:myData.yTestLabel})
            
            # Recover
            trainPred = trainClsCenter + trainRResPred
            testPred = testClsCenter + testRResPred
        
            # total loss (mean) & variance
            trainTotalLoss = np.mean(np.abs(batchY - trainPred))
            trainTotalVar  = np.var(np.abs(batchY - trainPred))
            testTotalLoss  = np.mean(np.abs(myData.yTest - testPred))
            testTotalVar  = np.var(np.abs(myData.yTest - testPred))
            
            TrainResidualat = 1/(1+np.exp(-trainAlpha * trainResPred))
            TrainBigResidual = np.where((0.0==TrainResidualat)|(TrainResidualat==1.0))
            bigResidualpar = TrainBigResidual[0].shape[0] / batchY.shape[0]
            
            TestTrRes = 1/(1+np.exp(-testAlpha * testResPred))
            TestBigResidual = np.where((0.0==TestTrRes)|(TestTrRes==1.0))
            TestbigResidualpar = TestBigResidual[0].shape[0] / myData.yTest.shape[0]
            
            print("Test Alpha", testAlpha)
            print("BigTrainResidual割合", bigResidualpar)
            print("BigTestResidual割合", TestbigResidualpar)
            print("-----------------------------------")
            print("itr:%d,trainClsLoss:%f,trainRegLoss:%f, trainTotalLoss:%f, trainTotalVar:%f" % (i,trainClsLoss,trainResLoss, trainTotalLoss, trainTotalVar))
            print("itr:%d,testClsLoss:%f,testRegLoss:%f, testTotalLoss:%f, testTotalVar:%f" % (i,testClsLoss,testResLoss, testTotalLoss, testTotalVar)) 
            
            if not flag:
                trainResLosses,testResLosses = trainResLoss[np.newaxis],testResLoss[np.newaxis]
                trainClassLosses,testClassLosses = trainClsLoss[np.newaxis],testClsLoss[np.newaxis]
                trainTotalLosses, testTotalLosses = trainTotalLoss[np.newaxis],testTotalLoss[np.newaxis]
                flag = True
            else:
                trainResLosses,testResLosses = np.hstack([trainResLosses,trainResLoss[np.newaxis]]),np.hstack([testResLosses,testResLoss[np.newaxis]])
                trainClassLosses,testClassLosses = np.hstack([trainClassLosses,trainClsLoss[np.newaxis]]),np.hstack([testClassLosses,testClsLoss[np.newaxis]])
                trainTotalLosses,testTotalLosses = np.hstack([trainTotalLosses,trainTotalLoss[np.newaxis]]),np.hstack([testTotalLosses,testTotalLoss[np.newaxis]])

# ------------------------- save model  --------------------------------- #
"""
modelFileName = "model_{}_{}_{}_{}_{}_{}.ckpt".format(methodModel,sigma,nClass,pNum,nTrain,nTest)
modelPath = "models"
modelfullPath = os.path.join(modelPath,modelFileName)
saver.save(sess,modelfullPath)
"""
# ------------------------- plot loss & toydata ------------------------- #
if methodModel == 0:
    myPlot.Plot_loss(0,0,0,0,trainRegLosses, testRegLosses,testPeriod,isPlot=isPlot, methodModel=methodModel, sigma=sigma, nClass=0, alpha=0, pNum=pNum, depth=depth)
    myPlot.Plot_3D(myData.xTest[:,0],myData.xTest[:,1],myData.yTest,testPred, isPlot=isPlot, methodModel=methodModel, sigma=sigma, nClass=0, alpha=0, pNum=pNum, depth=depth, isTrain=0)
    myPlot.Plot_3D(batchX[:,0],batchX[:,1],batchY,trainPred, isPlot=isPlot, methodModel=methodModel, sigma=sigma, nClass=0, alpha=0, pNum=pNum, depth=depth, isTrain=1)
    
    with open(os.path.join(pickleFullPath,"test_{}_{}_{}_{}_{}.pkl".format(methodModel,sigma,nClass,pNum,depth)),"wb") as fp:
            pickle.dump(batchY,fp)
            pickle.dump(trainPred,fp)
            pickle.dump(myData.yTest,fp)
            pickle.dump(testPred,fp)

elif methodModel == 1:
    myPlot.Plot_loss(trainTotalLosses, testTotalLosses, trainClassLosses, testClassLosses, trainResLosses, testResLosses, testPeriod, isPlot=isPlot, methodModel=methodModel, sigma=sigma, nClass=nClass, alpha=0, pNum=pNum, depth=depth)   
    myPlot.Plot_3D(myData.xTest[:,0],myData.xTest[:,1],myData.yTest,testPred, isPlot=isPlot, methodModel=methodModel, sigma=sigma, nClass=nClass, alpha=0, pNum=pNum, depth=depth, isTrain=0)
    myPlot.Plot_3D(batchX[:,0],batchX[:,1],batchY,trainPred, isPlot=isPlot, methodModel=methodModel, sigma=sigma, nClass=nClass, alpha=0, pNum=pNum, depth=depth, isTrain=1)
    
    with open(os.path.join(pickleFullPath,"test_{}_{}_{}_{}_{}.pkl".format(methodModel,sigma,nClass,pNum,depth)),"wb") as fp:
            pickle.dump(batchY,fp)
            pickle.dump(trainPred,fp)
            pickle.dump(myData.yTest,fp)
            pickle.dump(testPred,fp)
    
elif methodModel == 2: 
    myPlot.Plot_loss(trainTotalLosses, testTotalLosses, trainClassLosses, testClassLosses, trainResLosses, testResLosses, testPeriod,isPlot=isPlot, methodModel=methodModel, sigma=sigma, nClass=nClass, alpha=testAlpha, pNum=pNum, depth=depth)    
    myPlot.Plot_3D(myData.xTest[:,0],myData.xTest[:,1],myData.yTest,testPred, isPlot=isPlot, methodModel=methodModel, sigma=sigma, nClass=nClass, alpha=testAlpha, pNum=pNum, depth=depth, isTrain=0)
    myPlot.Plot_3D(batchX[:,0],batchX[:,1],batchY,trainPred, isPlot=isPlot,methodModel=methodModel,sigma=sigma,nClass=nClass,alpha=trainAlpha,pNum=pNum,depth=depth,isTrain=1)
    
    with open(os.path.join(pickleFullPath,"test_{}_{}_{}_{}_{}_{}.pkl".format(methodModel,sigma,nClass,testAlpha,pNum,depth)),"wb") as fp:
            pickle.dump(batchY,fp)
            pickle.dump(trainPred,fp)
            pickle.dump(myData.yTest,fp)
            pickle.dump(testPred,fp)
# ----------------------------------------------------------------------- #  
