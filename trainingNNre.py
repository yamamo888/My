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
        self.nRegHidden = 440
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
    #---------------------------------------------    
        
    def Optimizer(self,loss,name_scope="regression"):
         regressionVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=name_scope) 
         trainer_regression = tf.train.AdamOptimizer(self.lr).minimize(loss,var_list=regressionVars)
         return trainer_regression 
        
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    def regression(self,x,reuse=False,isRR=False,name_scope="regression"):
        with tf.variable_scope(name_scope) as scope:  
            keepProb = 1.0
            if reuse:
                keepProb = 1.0            
                scope.reuse_variables()
            
            w1_regression = self.weight_variable('w1_regression',[self.nCell*self.nWindow,self.nRegHidden])
            bias1_regression = self.bias_variable('bias1_regression',[self.nRegHidden])
            
            fc1 = self.fc_relu(x,w1_regression,bias1_regression,keepProb)
            
            w12_regression = self.weight_variable('w12_regression',[self.nRegHidden,self.nReg])
            bias12_regression = self.bias_variable('bias12_regression',[self.nReg])
         
            y = self.fc(fc1,w12_regression,bias12_regression,keepProb)
            
            if isRR:
                return y,w1_regression,w12_regression
            else:
                return y

#---------------------
#---------------------
if __name__ == "__main__":
    
    isWindows = False
    
    
    # Mode
    CellMode = int(sys.argv[1])
    classMode = int(sys.argv[2])
    datapicleMode = int(sys.argv[3])
    ramda = np.float(sys.argv[4])
    
    
    mytraining = Trainingeqp(CellMode=CellMode,classMode=classMode)

    
    # parameter
    nCell = mytraining.nCell
    bInd = mytraining.bInd
    nWindow = mytraining.nWindow
    nClass = mytraining.nClass
    nReg = mytraining.nReg
    batchSize = 1000
    

    # b1b2両方を出力したいときは True
    isRegression = True
    
    if isRegression:
        
        ################## Regression #########################
        x = tf.placeholder(tf.float32,shape=[None,nCell*nWindow]) 
        y1_ = tf.placeholder(tf.float32,shape=[None,1]) 
        y2_ = tf.placeholder(tf.float32,shape=[None,1]) 
        y3_ = tf.placeholder(tf.float32,shape=[None,1]) 
        
        y_ = tf.concat((y1_,y2_,y3_),1)
        
        predict_op= mytraining.regression(x)
        predict_test_op= mytraining.regression(x,reuse=True)
         
        loss = mytraining.Loss(y_,predict_op)
        loss_test = mytraining.Loss(y_,predict_test_op)
        
        trainer_regression = mytraining.Optimizer(loss)
        
        # l2制約
        predict_l2_op,w1,w2= mytraining.regression(x,isRR=True,name_scope="L2")
        predict_l2_test_op,w1_test,w2_test= mytraining.regression(x,reuse=True,isRR=True,name_scope="L2")
        
        l2 = mytraining.Loss(y_,predict_l2_op)
        l2_test = mytraining.Loss(y_,predict_l2_test_op)
        
        loss_l2 = l2 + ramda * (tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2))
        loss_l2_test = l2_test + ramda * (tf.nn.l2_loss(w1_test) + tf.nn.l2_loss(w2_test))
        
        trainer_l2_regression = mytraining.Optimizer(loss_l2,name_scope="L2")
        
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1,allow_growth=True)) 
        
        # training
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        #---------------------
        
        # cellの数指定
        testX = mytraining.myData.xTest
        testX = testX[:,1:6,:]
        testX = np.reshape(testX,[-1,nCell*nWindow])
        
        testY1Label = mytraining.myData.y1TestLabel
        testY2Label = mytraining.myData.y2TestLabel
        testY3Label = mytraining.myData.y3TestLabel
        testY1 = mytraining.myData.y1Test[:,np.newaxis]
        testY2 = mytraining.myData.y2Test[:,np.newaxis]
        testY3 = mytraining.myData.y3Test[:,np.newaxis]
        
        # Start training
        for i in range(80000):
            
            
            batchX, batchY1,batchY1Label,batchY2,batchY2Labeli,batchY3,batchY3Label = mytraining.myData.nextBatch(batchSize)
            batchX = batchX[:,1:6,:]
            batchX = np.reshape(batchX,[-1,nCell*nWindow])


            batchY1,batchY2,batchY3 = batchY1[:,np.newaxis],batchY2[:,np.newaxis],batchY3[:,np.newaxis]
            _,RegLoss= sess.run([trainer_regression,loss],feed_dict={x:batchX,y1_:batchY1,y2_:batchY2,y3_:batchY3})
            # l2
            _,RegL2Loss = sess.run([trainer_l2_regression,loss_l2],feed_dict={x:batchX,y1_:batchY1,y2_:batchY2,y3_:batchY3})
                    
            # Test
            if i % 500 == 0:
                # regression(範囲を狭く)
                testRegLoss,predTest = sess.run([loss_test,predict_test_op],feed_dict={x:testX,y1_:testY1,y2_:testY2,y3_:testY3})
                # l2 regression
                testRegL2Loss,yTestL2Pred = sess.run([loss_l2_test,predict_l2_test_op],feed_dict={x:testX,y1_:testY1,y2_:testY2,y3_:testY3})
                
                print("itr:%d,trainRegLoss:%f,testRegLoss:%f" % (i,RegLoss,testRegLoss))
                print("------------------------------------")
                print('b1regression',np.mean(np.abs(testY1-predTest[:,0][:,np.newaxis]))) 
                print('b1regVar',np.var(np.abs(testY1-predTest[:,0][:,np.newaxis])))
                print("------------------------------------")
                print('b2regression',np.mean(np.abs(testY2-predTest[:,1][:,np.newaxis]))) 
                print('b2regVar',np.var(np.abs(testY1-predTest[:,1][:,np.newaxis])))
                print("------------------------------------")
                print('b3regression',np.mean(np.abs(testY3-predTest[:,2][:,np.newaxis]))) 
                print('b3regVar',np.var(np.abs(testY3-predTest[:,2][:,np.newaxis])))
                print("------------------------------------")
                print("------------------------------------")
                print('b1l2regression',np.mean(np.abs(testY1-yTestL2Pred[:,0][:,np.newaxis]))) 
                print('b1l2regVar',np.var(np.abs(testY1-yTestL2Pred[:,0][:,np.newaxis])))
                print("------------------------------------")
                print('b2l2regression',np.mean(np.abs(testY2-yTestL2Pred[:,1][:,np.newaxis]))) 
                print('b2l2regVar',np.var(np.abs(testY1-yTestL2Pred[:,1][:,np.newaxis])))
                print("------------------------------------")
                print('b3l2regression',np.mean(np.abs(testY3-yTestL2Pred[:,2][:,np.newaxis]))) 
                print('b3l2regVar',np.var(np.abs(testY3-yTestL2Pred[:,2][:,np.newaxis])))
                print("------------------------------------")
                
                with open('./visualization/residual/TestR_trainalpha_{}{}.pickle'.format(i,CellMode),'wb') as fp:
                        pickle.dump(predTest[:,0],fp)
                        pickle.dump(predTest[:,1],fp)
                        pickle.dump(predTest[:,2],fp)        
                        pickle.dump(testY1,fp)
                        pickle.dump(testY2,fp)
                        pickle.dump(testY3,fp)
                with open('./visualization/residual/TestR_L2_trainalpha_{}{}{}.pickle'.format(i,CellMode,ramda),'wb') as fp:
                        pickle.dump(yTestL2Pred[:,0],fp)
                        pickle.dump(yTestL2Pred[:,1],fp)
                        pickle.dump(yTestL2Pred[:,2],fp)        
                        pickle.dump(testY1,fp)
                        pickle.dump(testY2,fp)
                        pickle.dump(testY3,fp)
        
                
                 
                
                
                


