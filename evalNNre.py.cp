#-*- coding: utf-8 -*-
"""
Created on Wed May 30 17:32:45 2018

@author: yu
"""

# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pdb
import tensorflow as tf
from tensorflow.python.ops import nn_ops
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
from scipy.spatial.distance import correlation
from scipy.stats import pearsonr
import pickle
import glob
import EarthQuakePlateModelKDE_FFT as eqp


class Trainingeqp():
    
    def __init__(self,CellMode=1,classMode=10,datapickleMode=1):
        
        # 引数
        # 予測するセル
        self.CellMode = int(sys.argv[1])
        # クラス数(\sigma) 
        self.classMode = int(sys.argv[2])
        # 使うpicklefile指定
        self.datapickleMode = int(sys.argv[3])
        # 使うセル
        if CellMode == 23456:
            bInd=[1,2,3,4,5]
            
        if classMode == 10:
            nClass = 11 
        elif classMode == 12:
            nClass = 12
        elif classMode == 20:
            nClass = 21
        elif classMode == 50:
            nClass = 51
        
        # Path
        self.featuresPath = './features'
        self.modelsPath = './models'
        self.trainingPath = './training'
        
        # 読み取るpicklePath 
        self.dataPath = 'b{}b{}_us'.format(bInd[0]+1,bInd[1]+1)
        self.picklePath = 'listxydatab{}b{}_'.format(bInd[0]+1,bInd[1]+1)
        self.trainingpicklePath = 'listtraintestdatab{}b{}_'.format(bInd[0]+1,bInd[1]+1)

        # picklefile
        xypicklefileName = self.picklePath + str(nClass) + 'u.pkl'
        trainpicklefileName = self.trainingpicklePath + str(nClass) + 'u.pkl'
        
        # picklePathとクラス数組み合わせてPath作成
        picklefullPath = os.path.join(xypicklefileName)
        trainpicklefullPath = os.path.join(trainpicklefileName)
            
        
        # parameter setting
        self.nCell = 5
        self.bInd = bInd
        self.nWindow = 10
        # 回帰と分類の隠れ層
        self.nRegHidden = 128
        self.nRegHidden2 = 128
        self.nRegHidden3 = 128
        self.nRegHidden4 = 128
        # 回帰と分類の出力
        self.nReg = 3
        self.nClass = nClass
        # b1+b2+特徴量(回帰の入力)
        self.b1b2X = self.nReg + (self.nWindow*self.nCell)
        self.lr = 1e-4
        
        self.myData = eqp.Data(fname='yV*', trainRatio=0.8, nCell=8,
                               sYear=2000, bInd=bInd, eYear=10000, 
                               isWindows=True, isClass=True,CellMode=CellMode,datapickleMode=datapickleMode,classMode=classMode,nClass=nClass,
              featuresPath=self.featuresPath,dataPath=self.dataPath,
              trainingpicklePath=trainpicklefullPath,picklePath=picklefullPath)

#-----------------------------------------------------------------------------------------------------
    def weight_variable(self,name,shape):
         return tf.get_variable(name,shape,initializer=tf.random_normal_initializer(stddev=0.1))
    
    def bias_variable(self,name,shape):
         return tf.get_variable(name,shape,initializer=tf.constant_initializer(0.1))
         
    def fc_relu(self,inputs, w, b, keepProb):
         relu = tf.matmul(inputs,w) + b
         relu = tf.nn.dropout(relu, keepProb)
         relu = tf.nn.relu(relu)
         return relu
        
    def fc_sigmoid(self,inputs,w,b,keepProb):
         sigmoid = tf.matmul(inputs,w) + b
         sigmoid = tf.nn.dropout(sigmoid,keepProb)
         sigmoid = tf.nn.sigmoid(sigmoid)
         return sigmoid

    def fc(self, inputs,w, b, keepProb):
         fc = tf.matmul(inputs,w) + b
         fc = tf.nn.dropout(fc, keepProb)
         return fc
    
    def Loss(self,y,predict):
         return tf.reduce_mean(tf.abs(y - predict))
        
    def Optimizer(self,loss,name_scope="regression"):
         regressionVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=name_scope) 
         trainer_regression = tf.train.AdamOptimizer(self.lr).minimize(loss,var_list=regressionVars)
         return trainer_regression 
        
#-----------------------------------------------------------------------------------------------------
    def regression(self,x,reuse=False,isRR=False,name_scope="regression",depth=0):
        with tf.variable_scope(name_scope) as scope:  
            keepProb = 1.0
            if reuse:
                keepProb = 1.0            
                scope.reuse_variables()
            
            w1_regression = self.weight_variable('w1_regression',[self.nCell*self.nWindow,self.nRegHidden])
            bias1_regression = self.bias_variable('bias1_regression',[self.nRegHidden])
            
            fc1 = self.fc_relu(x,w1_regression,bias1_regression,keepProb)
            
            
            if isRegression and depth == 3:
                
                w2_regression = self.weight_variable('w2_regression',[self.nRegHidden,self.nReg])
                bias2_regression = self.bias_variable('bias2_regression',[self.nReg])
                
                return self.fc(fc1,w2_regression,bias2_regression,keepProb)

            elif isRegression and depth == 4:

                w2_regression = self.weight_variable('w2_regression',[self.nRegHidden,self.nRegHidden2])
                bias2_regression = self.bias_variable('bias2_regression',[self.nRegHidden2])
                fc2 = self.fc_relu(fc1,w2_regression,bias2_regression,keepProb)
                
                w3_regression = self.weight_variable('w3_regression',[self.nRegHidden2,self.nReg])
                bias3_regression = self.bias_variable('bias3_regression',[self.nReg])
                
                return self.fc(fc2,w3_regression,bias3_regression,keepProb)

            elif isRegression and depth == 5:
                
                w2_regression = self.weight_variable('w2_regression',[self.nRegHidden,self.nRegHidden2])
                bias2_regression = self.bias_variable('bias2_regression',[self.nRegHidden2])
                fc2 = self.fc_relu(fc1,w2_regression,bias2_regression,keepProb)
                
                w3_regression = self.weight_variable('w3_regression',[self.nRegHidden2,self.nRegHidden3])
                bias3_regression = self.bias_variable('bias3_regression',[self.nRegHidden3])
                fc3 = self.fc_relu(fc2,w3_regression,bias3_regression,keepProb)
            
                w4_regression = self.weight_variable('w4_regression',[self.nRegHidden3,self.nReg])
                bias4_regression = self.bias_variable('bias4_regression',[self.nReg])

                return self.fc(fc3,w4_regression,bias4_regression,keepProb) 
            
#-------------------------------------------------------------------------------------------------------
class Plot():
    """
    Visualization: Point cloud of evaluation data is blue with 3 axes of (x1, x2, y)
    Predicted value
    """
    def __init__(self,isPlot=False,nClass=1,alpha=1):
        self.isPlot = isPlot
        self.nClass = nClass
        self.alpha = alpha
        self.visualPath = "visualization"
    
    def Plot_loss(self,trainRegLoss, testRegLoss, testPeriod):
        if self.isPlot:
            # lossPlot
            plt.plot(np.arange(trainRegLosses.shape[0]),trainRegLosses,label="trainRegLosses",color="c")
            plt.plot(np.arange(testRegLosses.shape[0]),testRegLosses,label="testRegLosses",color="pink")
            
            plt.ylim([0,0.5])
            plt.xlabel("iteration x {}".format(testPeriod))
            
            plt.legend()
            fullPath = os.path.join(self.visualPath,"Loss_{}.png".format(self.alpha))
            plt.savefig(fullPath)

#---------------------
#---------------------
if __name__ == "__main__":
    
    isWindows = False
    isPlot = True
    
    testPeriod = 500
    # Mode
    CellMode = int(sys.argv[1])
    classMode = int(sys.argv[2])
    datapicleMode = int(sys.argv[3])
    depth = int(sys.argv[4])
    
    mytraining = Trainingeqp(CellMode=CellMode,classMode=classMode)

    
    # parameter
    nCell = mytraining.nCell
    bInd = mytraining.bInd
    nWindow = mytraining.nWindow
    nClass = mytraining.nClass
    nReg = mytraining.nReg
    batchSize = 2000
    nTraining = 100000

    # b1b2両方を出力したいときは True
    isRegression = True
    
    if isRegression:
        
        ################## Regression #########################
        x = tf.placeholder(tf.float32,shape=[None,nCell*nWindow]) 
        y1_ = tf.placeholder(tf.float32,shape=[None,1]) 
        y2_ = tf.placeholder(tf.float32,shape=[None,1]) 
        y3_ = tf.placeholder(tf.float32,shape=[None,1]) 
        #val = tf.constant(100.,name="val") 
        #y1,y2,y3 = y1_ * val, y2_ * val, y3_ * val


        y = tf.concat((y1_,y2_,y3_),1)
        #y = y_ * val
        
        predict_op= mytraining.regression(x,depth=depth)
        predict_test_op= mytraining.regression(x,reuse=True,depth=depth)
         
        loss = mytraining.Loss(y,predict_op)
        
        trainer_regression = mytraining.Optimizer(loss)
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1,allow_growth=True)) 
        
        # training
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        #---------------------
        
        # cellの数指定
        #testX = mytraining.myData.xTest
        #testX = testX[:,1:6,:]
        #testX = np.reshape(testX,[-1,nCell*nWindow])
        
        evalationNankaifullPath = os.path.join("191.pkl")
        with open(evalationNankaifullPath,"rb") as fp:
            testX = pickle.load(fp)
        testX = np.reshape(testX,[1,-1])
        testX = np.concatenate([testX[:,:10],testX[:,:20],testX[:,10:20],testX[:,20:]],axis=1)
        
        #testY1Label = mytraining.myData.y1TestLabel
        #testY2Label = mytraining.myData.y2TestLabel
        #testY3Label = mytraining.myData.y3TestLabel
        #testY1 = mytraining.myData.y1Test[:,np.newaxis]
        #testY2 = mytraining.myData.y2Test[:,np.newaxis]
        #testY3 = mytraining.myData.y3Test[:,np.newaxis]
        #testY1,testY2,testY3 = testY1*10,testY2*10,testY2*10
        
        # Start training
        flag = False
        for i in range(nTraining):
            #reduce_val = 1/100
             
            batchX, batchY1,batchY1Label,batchY2,batchY2Label,batchY3,batchY3Label = mytraining.myData.nextBatch(batchSize)
            batchX = batchX[:,1:6,:]
            batchX = np.reshape(batchX,[-1,nCell*nWindow])
            batchY1,batchY2,batchY3 = batchY1[:,np.newaxis],batchY2[:,np.newaxis],batchY3[:,np.newaxis]
            #batchY1,batchY2,batchY3 = batchY1*10,batchY2*10,batchY3*10
            
            _, predTrainex, trainRegLoss= sess.run([trainer_regression,predict_op,loss],feed_dict={x:batchX,y1_:batchY1,y2_:batchY2,y3_:batchY3})
            
            #predTrain = predTrainex * reduce_val

            # Test
            if i % testPeriod == 0:
                # regression(範囲を狭く)
                predTestex = sess.run(predict_test_op,feed_dict={x:testX})
                
                #predTest = predTestex * reduce_val
                
                print(predTestex[:10,0])
                print(predTestex[:10,1])
                print(predTestex[:10,2])
                
                with open('./visualization/eval/TestR_{}_{}.pickle'.format(i,depth),'wb') as fp:
                        pickle.dump(predTestex[:,0],fp)
                        pickle.dump(predTestex[:,1],fp)
                        pickle.dump(predTestex[:,2],fp)        


                

                
                 
                
                
                


