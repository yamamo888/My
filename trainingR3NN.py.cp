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
    
    def __init__(self,methodModel=1,CellMode=1,classMode=10,datapickleMode=1):
        
        # 南海トラフ巨大地震履歴を使う時はTrue
        self.isEval = False
        # 評価データを正規化するとき
        self.isNormalization = False
        # 引数
        self.methodModel = methodModel
        # 予測するセル
        self.CellMode = CellMode
        # クラス数(\sigma) 
        self.classMode = classMode
        # 使うpicklefile指定
        self.datapickleMode = datapickleMode
        # 使うセル
        if CellMode == 23456:
            bInd=[1,2,3,4,5]
        
        # クラス数
        if classMode == 10:
            nClass = 11
        elif classMode == 12:
            nClass = 12
        elif classMode == 20:
            nClass = 21
        elif classMode == 50:
            nClass = 50
        
        
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
        
        self.nCell = 5
            
        self.bInd = bInd
        self.nWindow = 10
        # 回帰と分類の隠れ層
        self.clsHidden = 128 #128 
        self.clsHidden2 = 128 #64
        self.nRegHidden = 128
        self.nRegHidden2 = 128
        self.nRegHidden3 = 128
        # 回帰と分類の出力
        self.nReg = 3
        self.nClass = nClass
        # b1+b2+特徴量(回帰の入力)
        self.b1b2X = self.nReg + (self.nWindow*self.nCell)
        # クラス幅切り捨て位
        limitdecimal = 6
        # 元の大きさ
        self.sbn = 0.0125
        self.ebn = 0.017
        self.sb = 0.012
        self.eb = 0.0165
        
        #self.sbn,self.ebn,self.sb,self.eb = self.sbn*10,self.ebn*10,self.sb*10,self.eb*10
        # クラス幅
        self.beta = round((self.eb-self.sb)/self.nClass,limitdecimal)
        
        # 狭める範囲
        self.aW = 1/20
        # 学習データ数
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
            
    def softmax(self,inputs,w,b,keepProb):
         softmax = tf.matmul(inputs,w) + b
         softmax = tf.nn.dropout(softmax, keepProb)
         softmax = tf.nn.softmax(softmax)
         return softmax
        
    def alpha_variable(self,name,shape):
        alphaMean = self.aW * (1/self.beta)
        #alphaInit = tf.random_normal_initializer(mean=alphaMean,stddev=0.1)
        #alphaInit = tf.random_normal_initializer(mean=0.5,stddev=0.1)
        alphaInit = tf.random_normal_initializer(mean=10,stddev=0.1)
        return tf.get_variable(name,shape,initializer=alphaInit)
    
    def fc_sigmoid(self,inputs,w,b,keepProb):
        sigmoid = tf.matmul(inputs,w) + b
        sigmoid = tf.nn.dropout(sigmoid,keepProb)
        sigmoid = tf.nn.sigmoid(sigmoid)
        return sigmoid
    
    def fc_relu(self,inputs, w, b, keepProb):
         relu = tf.matmul(inputs,w) + b
         relu = tf.nn.dropout(relu, keepProb)
         relu = tf.nn.relu(relu)
         return relu
        
    def fc(self, inputs,w, b, keepProb):
         fc = tf.matmul(inputs,w) + b
         fc = tf.nn.dropout(fc, keepProb)
         return fc 
    #--------------------------------------------------------------------------
    def Classification(self,x,reuse=False):
        with tf.variable_scope('Classification') as scope:  
            keepProb = 1.0
            if reuse:
                keepProb = 1.0            
                scope.reuse_variables()
             
            #input -> hidden
            w1 = self.weight_variable('w1',[self.nCell*self.nWindow,self.clsHidden])
            bias1 = self.bias_variable('bias1',[self.clsHidden])
            
            h = self.fc_relu(x,w1,bias1,keepProb) 
             
            w2 = self.weight_variable('w2',[self.clsHidden,self.clsHidden2])
            bias2 = self.bias_variable('bias2',[self.clsHidden2])
            
            h2 = self.fc_relu(h,w2,bias2,keepProb) 
            
            #hidden -> output
            w3_1 = self.weight_variable('w3_1',[self.clsHidden2,self.nClass])
            bias3_1 = self.bias_variable('bias3_1',[self.nClass])
            
            w3_2 = self.weight_variable('w3_2',[self.clsHidden2,self.nClass])
            bias3_2 = self.bias_variable('bias3_2',[self.nClass])
            
            w3_3 = self.weight_variable('w3_3',[self.clsHidden2,self.nClass])
            bias3_3 = self.bias_variable('bias3_3',[self.nClass])
            
            y1 = self.fc(h2,w3_1,bias3_1,keepProb)
            y2 = self.fc(h2,w3_2,bias3_2,keepProb)
            y3 = self.fc(h2,w3_3,bias3_3,keepProb)
            
            return y1, y2, y3
    #---------------------------------------------------------------------------        
    def Regression(self,x,reuse=False,isRR=False,depth=0):
        with tf.variable_scope('Regression') as scope:  
            keepProb = 1.0
            if reuse:
                keepProb = 1.0            
                scope.reuse_variables()

            w1_regression = self.weight_variable('w1_regression',[self.b1b2X,self.nRegHidden])
            bias1_regression = self.bias_variable('bias1_regression',[self.nRegHidden])
            fc1 = self.fc_relu(x,w1_regression,bias1_regression,keepProb)
             
            if depth == 3:
                
                w2_regression = self.weight_variable('w2_regression',[self.nRegHidden,self.nReg])
                bias2_regression = self.bias_variable('bias2_regression',[self.nReg])
                
                if isRR:
                    return self.fc(fc1,w2_regression,bias2_regression,keepProb)
                else:
                    return self.fc_sigmoid(fc1,w2_regression,bias2_regression,keepProb)

            elif depth == 4:

                w2_regression = self.weight_variable('w2_regression',[self.nRegHidden,self.nRegHidden2])
                bias2_regression = self.bias_variable('bias2_regression',[self.nRegHidden2])
                fc2 = self.fc_relu(fc1,w2_regression,bias2_regression,keepProb)
                
                w3_regression = self.weight_variable('w3_regression',[self.nRegHidden2,self.nReg])
                bias3_regression = self.bias_variable('bias3_regression',[self.nReg])
                
                if isRR:
                    return self.fc(fc2,w3_regression,bias3_regression,keepProb)
                else:
                    return self.fc_sigmoid(fc2,w3_regression,bias3_regression,keepProb)

            elif depth == 5:
                
                w2_regression = self.weight_variable('w2_regression',[self.nRegHidden,self.nRegHidden2])
                bias2_regression = self.bias_variable('bias2_regression',[self.nRegHidden2])
                fc2 = self.fc_relu(fc1,w2_regression,bias2_regression,keepProb)
                
                w3_regression = self.weight_variable('w3_regression',[self.nRegHidden2,self.nRegHidden3])
                bias3_regression = self.bias_variable('bias3_regression',[self.nRegHidden3])
                fc3 = self.fc_relu(fc2,w3_regression,bias3_regression,keepProb)
            
                w4_regression = self.weight_variable('w4_regression',[self.nRegHidden3,self.nReg])
                bias4_regression = self.bias_variable('bias4_regression',[self.nReg])
                
                if isRR:
                    return self.fc(fc3,w4_regression,bias4_regression,keepProb) 
                else:
                    return self.fc_sigmoid(fc3,w4_regression,bias4_regression,keepProb)
            
    #-------------------------------------------------------------------------- 
    def FeatureVector(self,x,cls_center1,cls_center2,cls_center3):
        cls_center_all = tf.concat((cls_center1,cls_center2,cls_center3),axis=1)
        feature_vector = tf.concat((cls_center_all,x),axis=1)
        return cls_center_all,feature_vector


    def ClassCenter_Residual(self,y_class,predict,start_center):
        # 予測したクラスの最大クラス
        pred_class_maxclass = tf.expand_dims(tf.cast(tf.argmax(predict,axis=1),tf.float32),1)  
        # 選択したクラスの中心値       
        predict_class_center = pred_class_maxclass * self.beta + start_center
        # 残差
        residual = y_class - predict_class_center
        return predict_class_center,residual
    #---------------------------------------------    
    def Magnify(self,res,reuse=False):
        with tf.variable_scope('Magnify') as scope:  
            if reuse:
                scope.reuse_variables()
            alpha = self.alpha_variable("alpha",[1]) 
            magnify_residual = 1/(1 + tf.exp(- alpha * res))
            return magnify_residual,alpha
    
    def Reduce(self,pred_r_mag,mag_param,reuse=False):
        with tf.variable_scope('Magnify') as scope:  
            if reuse:
                scope.reuse_variables()
            predict_residual = (-1/mag_param) * tf.log((1/pred_r_mag) - 1)
            return predict_residual
    #---------------------------------------------    
            
    def Loss(self,y,predict,isRegression=False,isAlpha=False):
        if isRegression:
            return tf.reduce_mean(tf.abs(y - predict))
        elif isAlpha:
            return tf.reduce_mean(tf.abs(y - predict))
        else:
            return tf.losses.softmax_cross_entropy(y,predict)
    #---------------------------------------------    
        
    def Optimizer(self,loss,isRegression=False,isAlpha=False):
        
        if isRegression:
            regressionVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="Regression") 
            trainer_regression = tf.train.AdamOptimizer(self.lr).minimize(loss,var_list=regressionVars)
            return trainer_regression 
        
        elif isAlpha:
            alphaVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="Magnify") 
            trainer_alpha = tf.train.AdamOptimizer(self.lr).minimize(loss,var_list=alphaVars)
            #trainer_alpha = tf.train.GradientDescentOptimizer(self.lr).minimize(loss,var_list=alphaVars)
            return trainer_alpha 
        
        else:
            classificationVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="Classification") 
            trainer = tf.train.AdamOptimizer(self.lr).minimize(loss,var_list=classificationVars)
            return trainer
    
#---------------------------------------------    
class Plot():
    """
    Visualization: Point cloud of evaluation data is blue with 3 axes of (x1, x2, y)
    Predicted value
    """
    def __init__(self,isPlot=False,methodModel=1,nClass=1,alpha=1):
        self.isPlot = isPlot
        self.methodModel = methodModel
        self.nClass = nClass
        self.alpha = alpha
        self.visualPath = "visualization"
    
          
    def Plot_loss(self,trainTotalLosses, testTotalLosses, trainClassLosses, testClassLosses,trainRegLoss, testRegLoss, testPeriod, cell=0):
        if self.isPlot:
            # lossPlot
            plt.plot(np.arange(trainTotalLosses.shape[0]),trainTotalLosses,label="trainTotalLosses",color="r")
            plt.plot(np.arange(testTotalLosses.shape[0]),testTotalLosses,label="testTotalLosses",color="g")
            plt.plot(np.arange(trainClassLosses.shape[0]),trainClassLosses,label="trainClassLosses",color="b")
            plt.plot(np.arange(testClassLosses.shape[0]),testClassLosses,label="testClassLosses",color="k")
            plt.plot(np.arange(trainRegLosses.shape[0]),trainRegLosses,label="trainRegLosses",color="c")
            plt.plot(np.arange(testRegLosses.shape[0]),testRegLosses,label="testRegLosses",color="pink")
            
            plt.ylim([0,0.5])
            plt.xlabel("iteration x {}".format(testPeriod))
            
            plt.legend()
            fullPath = os.path.join(self.visualPath,"Loss_{}_{}_{}_{}.png".format(self.methodModel,self.nClass,self.alpha,cell))
            plt.savefig(fullPath)
            plt.close()

#---------------------
#---------------------
if __name__ == "__main__":
    
    isWindows = False
    isPlot = True
    
    # Mode
    methodModel = int(sys.argv[1])
    CellMode = int(sys.argv[2])
    classMode = int(sys.argv[3])
    datapickleMode = int(sys.argv[4])
    depth = float(sys.argv[5])
    testPeriod = 500
    # どれぐらいの領域を正規化するか

    mytraining = Trainingeqp(CellMode=CellMode,classMode=classMode,datapickleMode=datapickleMode)
    
    isEval = mytraining.isEval
    isNormalization = mytraining.isNormalization
    # parameter
    nCell = mytraining.nCell
    bInd = mytraining.bInd
    nWindow = mytraining.nWindow
    nClass = mytraining.nClass
    nReg = mytraining.nReg
    batchSize = 2000
    nTraining = 300000
    # 区間幅
    beta = mytraining.beta
    # bの始まり
    sbn = mytraining.sbn
    sb = mytraining.sb
    # 0クラス目の中心値
    sCent_nankai = sbn + (beta/2)
    sCent = sb + (beta/2)
    # しんのセル
    nI,tnI,tI = 0,1,2
    #----------------------------------------------------------------------
    x = tf.placeholder(tf.float32, shape=[None,nCell*nWindow]) 
    y1_class_label = tf.placeholder(tf.float32, shape=[None,nClass])
    y2_class_label = tf.placeholder(tf.float32, shape=[None,nClass])
    y3_class_label = tf.placeholder(tf.float32, shape=[None,nClass])
    y1_class = tf.placeholder(tf.float32, shape=[None,1])
    y2_class = tf.placeholder(tf.float32, shape=[None,1]) 
    y3_class = tf.placeholder(tf.float32, shape=[None,1])  
    y_class_all = tf.concat((y1_class,y2_class,y3_class),axis=1)
    #----------------------------------------------------------------------
    
    if isEval:
            
        evalationNankaifullPath = "nankairireki.pkl"
        with open(evalationNankaifullPath,"rb") as fp:
            nankairireki = pickle.load(fp)
        testX = nankairireki[190,:,:]
        pdb.set_trace()

        
    else:
        # テストデータ
        testX = mytraining.myData.xTest
        testX = testX[:,1:6,:]
        testX = np.reshape(testX,[-1,nCell*nWindow])
            
        testY1Label = mytraining.myData.y1TestLabel
        testY2Label = mytraining.myData.y2TestLabel
        testY3Label = mytraining.myData.y3TestLabel
            
        testY1 = mytraining.myData.y1Test[:,np.newaxis]
        testY2 = mytraining.myData.y2Test[:,np.newaxis]
        testY3 = mytraining.myData.y3Test[:,np.newaxis]
        #testY1,testY2,testY3 = testY1*10,testY2*10,testY3*10
    if methodModel == 1:

        predict_class1_op,predict_class2_op,predict_class3_op = mytraining.Classification(x)
        predict_class1_test_op,predict_class2_test_op,predict_class3_test_op = mytraining.Classification(x,reuse=True)

        loss1 = mytraining.Loss(y1_class_label,predict_class1_op)
        loss2 = mytraining.Loss(y2_class_label,predict_class2_op)
        loss3 = mytraining.Loss(y3_class_label,predict_class3_op)
        
        loss1_test = mytraining.Loss(y1_class_label,predict_class1_test_op)
        loss2_test = mytraining.Loss(y2_class_label,predict_class2_test_op)
        loss3_test = mytraining.Loss(y3_class_label,predict_class3_test_op)
        # all Classification loss
        loss_class_all = loss1 + loss2 + loss3
        loss_class_test_all = loss1_test + loss2_test + loss3_test
        
        trainer_classification = mytraining.Optimizer(loss_class_all)
        #----------------------------------------------------------------------
        pred_class1_center,residual1_op = mytraining.ClassCenter_Residual(y1_class,predict_class1_op,sCent_nankai)
        pred_class2_center,residual2_op = mytraining.ClassCenter_Residual(y2_class,predict_class2_op,sCent)
        pred_class3_center,residual3_op = mytraining.ClassCenter_Residual(y3_class,predict_class3_op,sCent)

        pred_class1_center_test,residual1_test_op = mytraining.ClassCenter_Residual(y1_class,predict_class1_op,sCent_nankai)
        pred_class2_center_test,residual2_test_op = mytraining.ClassCenter_Residual(y2_class,predict_class2_op,sCent)
        pred_class3_center_test,residual3_test_op = mytraining.ClassCenter_Residual(y3_class,predict_class3_op,sCent)
        
        residual_all = tf.concat((residual1_op,residual2_op,residual3_op),axis=1)
        residual_all_test = tf.concat((residual1_test_op,residual2_test_op,residual3_test_op),axis=1)

        # all Class Center
        class_center_all,class_center_x = mytraining.FeatureVector(x,pred_class1_center,pred_class2_center,pred_class3_center) 
        class_center_all_test,class_center_x_test = mytraining.FeatureVector(x,pred_class1_center_test,pred_class2_center_test,pred_class3_center_test)

        predict_regression_residual_op = mytraining.Regression(class_center_x,isRR=True,depth=depth)
        predict_regression_residual_test_op = mytraining.Regression(class_center_x_test,reuse=True,isRR=True,depth=depth)
        
        loss_regression = mytraining.Loss(residual_all,predict_regression_residual_op,isRegression=True)
        loss_regression_test = mytraining.Loss(residual_all_test,predict_regression_residual_test_op,isRegression=True) 
        
        trainer_regression =  mytraining.Optimizer(loss_regression,isRegression=True) 

        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1,allow_growth=True)) 
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        
        flag=False
        for i in range(nTraining):
            batchX, batchY1,batchY1Label,batchY2,batchY2Label,batchY3,batchY3Label = mytraining.myData.nextBatch(batchSize)
            # 特徴量ベクトルに変更 
            batchX = batchX[:,1:6,:]
            
            batchX = np.reshape(batchX,[-1,nCell*nWindow])
            batchY1,batchY2,batchY3 = batchY1[:,np.newaxis],batchY2[:,np.newaxis],batchY3[:,np.newaxis]
               
            _, trainClassLoss1,trainClassLoss2,trainClassLoss3,trainClassCenter1,trainClassCenter2,trainClassCenter3 = sess.run([trainer_classification,loss1,loss2,loss3,pred_class1_center,pred_class2_center,pred_class3_center],feed_dict={x:batchX,y1_class:batchY1,y2_class:batchY2,y3_class:batchY3,y1_class_label:batchY1Label,y2_class_label:batchY2Label,y3_class_label:batchY3Label})

            # regression
            _, trainRegLoss, TrainPred = sess.run([trainer_regression,loss_regression,predict_regression_residual_op],feed_dict={x:batchX,y1_class:batchY1,y2_class:batchY2,y3_class:batchY3,y1_class_label:batchY1Label,y2_class_label:batchY2Label,y3_class_label:batchY3Label})
            

            yTrainPred1 = trainClassCenter1 + TrainPred[:,nI][:,np.newaxis]
            yTrainPred2 = trainClassCenter2 + TrainPred[:,tnI][:,np.newaxis]
            yTrainPred3 = trainClassCenter3 + TrainPred[:,tI][:,np.newaxis]
            # total loss b1
            trainTotalLoss1 = np.mean(np.abs(batchY1-yTrainPred1))
            trainTotalLoss2 = np.mean(np.abs(batchY2-yTrainPred2))
            trainTotalLoss3 = np.mean(np.abs(batchY3-yTrainPred3))
            
            # Test
            if i % testPeriod == 0:
                if isEval:
                    # classification for evaluation                   
                    predCenterTest1,predCenterTest2,predCenterTest3 = sess.run([predict_class1_max_center_test_op,predict_class2_max_center_test_op,predict_class3_max_center_test_op], feed_dict={x:testX})
                    # regression
                    exp_r_test= sess.run(exp_regression_test,feed_dict={x:testX})
                
                else:
                    # classication
                    testClassLoss1,testClassLoss2,testClassLoss3,testClassCenter1,testClassCenter2,testClassCenter3 = sess.run([loss1_test,loss2_test,loss3_test,pred_class1_center_test,pred_class2_center_test,pred_class3_center_test],feed_dict={x:testX,y1_class:testY1,y2_class:testY2,y3_class:testY3,y1_class_label:testY1Label,y2_class_label:testY2Label,y3_class_label:testY3Label})    
                    # regression
                    testRegLoss, TestPred = sess.run([loss_regression_test,predict_regression_residual_test_op],feed_dict={x:testX,y1_class:testY1,y2_class:testY2,y3_class:testY3,y1_class_label:testY1Label,y2_class_label:testY2Label,y3_class_label:testY3Label})
                    
                    yTestPred1 = testClassCenter1 + TestPred[:,nI][:,np.newaxis]
                    yTestPred2 = testClassCenter2 + TestPred[:,tnI][:,np.newaxis]
                    yTestPred3 = testClassCenter3 + TestPred[:,tI][:,np.newaxis]

                    # total loss b1
                    testTotalLoss1 = np.mean(np.abs(testY1-yTestPred1))
                    testTotalLoss2 = np.mean(np.abs(testY2-yTestPred2))
                    testTotalLoss3 = np.mean(np.abs(testY3-yTestPred3))

                
                if isEval:
                    with open('./visualization/residual/EvalCR3_{}{}{}.pickle'.format(i,CellMode,classMode),'wb') as fp:
                            pickle.dump(np.round(b1_cr,7),fp)
                            pickle.dump(np.round(b2_cr,7),fp)
                            pickle.dump(np.round(b3_cr,7),fp)
                else:
                    print("GT",testY1[:10])
                    print("Cls+Res",yTestPred1[:10])
                    print("Cls",testClassCenter1[:10])
                    print("Res",TestPred[:10,0])
                    print("------------------------------------")
                    print('b1regression',np.mean(np.abs(testY1-yTestPred1))) 
                    print('b1regVar',np.var(np.abs(testY1-yTestPred1)))
                    print("------------------------------------")
                    print('b2regression',np.mean(np.abs(testY2-yTestPred2))) 
                    print('b2regVar',np.var(np.abs(testY2-yTestPred2)))
                    print("------------------------------------")
                    print('b3regression',np.mean(np.abs(testY3-yTestPred3))) 
                    print('b3regVar',np.var(np.abs(testY3-yTestPred3)))
                    print("---------------")
                    print("itr:%d,trainClassLoss1:%f,trainClassLoss2:%f,trainClassLoss3:%f, trainRegLoss:%f" % (i,trainClassLoss1,trainClassLoss2,trainClassLoss3,trainRegLoss))
                    print("itr:%d,trainTotalLoss1:%f,trainTotalLoss2:%f, trainTotalLoss2:%f" % (i,trainTotalLoss1,trainTotalLoss1, trainTotalLoss3))
                    print("-----------------------------------")
                    print("itr:%d,testClassLoss1:%f,testClassLoss2:%f,testClassLoss3:%f, testRegLoss:%f" % (i,testClassLoss1,testClassLoss2,testClassLoss3,testRegLoss))
                    print("itr:%d,testTotalLoss1:%f,testTotalLoss2:%f, testTotalLoss2:%f" % (i,testTotalLoss1,testTotalLoss1, testTotalLoss3))

                if not flag:
                    trainRegLosses,testRegLosses = trainRegLoss[np.newaxis],testRegLoss[np.newaxis]
                    trainClassLosses1,testClassLosses1 = trainClassLoss1[np.newaxis],testClassLoss1[np.newaxis]
                    trainTotalLosses1, testTotalLosses1 = trainTotalLoss1[np.newaxis],testTotalLoss1[np.newaxis]
                    trainClassLosses2,testClassLosses2 = trainClassLoss2[np.newaxis],testClassLoss2[np.newaxis]
                    trainTotalLosses2, testTotalLosses2= trainTotalLoss2[np.newaxis],testTotalLoss2[np.newaxis]
                    trainClassLosses3,testClassLosses3 = trainClassLoss3[np.newaxis],testClassLoss3[np.newaxis]
                    trainTotalLosses3, testTotalLosses3 = trainTotalLoss3[np.newaxis],testTotalLoss3[np.newaxis]
                    flag = True
                else:
                    trainRegLosses,testRegLosses = np.hstack([trainRegLosses,trainRegLoss[np.newaxis]]),np.hstack([testRegLosses,testRegLoss[np.newaxis]])
                    trainClassLosses1,testClassLosses1 = np.hstack([trainClassLosses1,trainClassLoss1[np.newaxis]]),np.hstack([testClassLosses1,testClassLoss1[np.newaxis]])
                    trainTotalLosses1,testTotalLosses1 = np.hstack([trainTotalLosses1,trainTotalLoss1[np.newaxis]]),np.hstack([testTotalLosses1,testTotalLoss1[np.newaxis]])
                    trainClassLosses2,testClassLosses2 = np.hstack([trainClassLosses2,trainClassLoss2[np.newaxis]]),np.hstack([testClassLosses2,testClassLoss2[np.newaxis]])
                    trainTotalLosses2,testTotalLosses2 = np.hstack([trainTotalLosses2,trainTotalLoss2[np.newaxis]]),np.hstack([testTotalLosses2,testTotalLoss2[np.newaxis]])
                    trainClassLosses3,testClassLosses3 = np.hstack([trainClassLosses3,trainClassLoss3[np.newaxis]]),np.hstack([testClassLosses3,testClassLoss3[np.newaxis]])
                    trainTotalLosses3,testTotalLosses3 = np.hstack([trainTotalLosses3,trainTotalLoss3[np.newaxis]]),np.hstack([testTotalLosses3,testTotalLoss3[np.newaxis]])
                 
                with open('./visualization/residual/TestCR_{}_{}_{}.pickle'.format(i,classMode,depth),'wb') as fp:
                        pickle.dump(yTestPred1,fp)
                        pickle.dump(yTestPred2,fp)
                        pickle.dump(yTestPred3,fp)        
                        pickle.dump(testY1,fp)
                        pickle.dump(testY2,fp)
                        pickle.dump(testY3,fp)
                """
                with open('./visualization/residual/TrainCR_{}_{}_{}.pickle'.format(i,classMode,depth),'wb') as fp:
                        pickle.dump(yTrainPred1,fp)
                        pickle.dump(yTrainPred2,fp)
                        pickle.dump(yTrainPred3,fp)
                        pickle.dump(batchY1,fp)
                        pickle.dump(batchY2,fp)
                        pickle.dump(batchY3,fp)"""     
         
    elif methodModel == 2:
         
        predict_class1_op,predict_class2_op,predict_class3_op = mytraining.Classification(x)
        predict_class1_test_op,predict_class2_test_op,predict_class3_test_op = mytraining.Classification(x,reuse=True)
        
        loss1 = mytraining.Loss(y1_class_label,predict_class1_op)
        loss2 = mytraining.Loss(y2_class_label,predict_class2_op)
        loss3 = mytraining.Loss(y3_class_label,predict_class3_op)
        
        loss1_test = mytraining.Loss(y1_class_label,predict_class1_test_op)
        loss2_test = mytraining.Loss(y2_class_label,predict_class2_test_op)
        loss3_test = mytraining.Loss(y3_class_label,predict_class3_test_op)
        # all Classification loss
        loss_class_all = loss1 + loss2 + loss3
        loss_class_test_all = loss1_test + loss2_test + loss3_test
        
        trainer_classification = mytraining.Optimizer(loss_class_all)
        #----------------------------------------------------------------------
        pred_class1_center,residual1_op = mytraining.ClassCenter_Residual(y1_class,predict_class1_op,sCent_nankai)
        pred_class2_center,residual2_op = mytraining.ClassCenter_Residual(y2_class,predict_class2_op,sCent)
        pred_class3_center,residual3_op = mytraining.ClassCenter_Residual(y3_class,predict_class3_op,sCent)

        pred_class1_center_test,residual1_test_op = mytraining.ClassCenter_Residual(y1_class,predict_class1_op,sCent_nankai)
        pred_class2_center_test,residual2_test_op = mytraining.ClassCenter_Residual(y2_class,predict_class2_op,sCent)
        pred_class3_center_test,residual3_test_op = mytraining.ClassCenter_Residual(y3_class,predict_class3_op,sCent)
        
        # all GT residual
        residual_all = tf.concat((residual1_op,residual2_op,residual3_op),axis=1) 
        residual_all_test = tf.concat((residual1_test_op,residual2_test_op,residual3_test_op),axis=1) 
        # all Class Center
        class_center_all,class_center_x = mytraining.FeatureVector(x,pred_class1_center,pred_class2_center,pred_class3_center) 
        class_center_all_test,class_center_x_test = mytraining.FeatureVector(x,pred_class1_center_test,pred_class2_center_test,pred_class3_center_test)

        predict_regression_residual_op = mytraining.Regression(class_center_x,depth=depth)
        predict_regression_residual_test_op = mytraining.Regression(class_center_x_test,reuse=True,depth=depth)
        
        magnify_residual,alpha_op = mytraining.Magnify(residual_all)
        magnify_residual_test,alpha_test_op = mytraining.Magnify(residual_all_test,reuse=True)
        
        loss_regression = mytraining.Loss(magnify_residual,predict_regression_residual_op,isRegression=True)
        loss_regression_test = mytraining.Loss(magnify_residual_test,predict_regression_residual_test_op,isRegression=True) 
        
        trainer_regression =  mytraining.Optimizer(loss_regression,isRegression=True) 
        #---------------------------------------------------------------------- 
        reduce_residual =  mytraining.Reduce(predict_regression_residual_op,alpha_op)
        reduce_residual_test =  mytraining.Reduce(predict_regression_residual_test_op,alpha_test_op,reuse=True)

        pred_y = class_center_all +  reduce_residual
        pred_y_test = class_center_all_test + reduce_residual_test
        

        loss_alpha =  mytraining.Loss(y_class_all,pred_y,isAlpha=True)
        loss_alpha_test =  mytraining.Loss(y_class_all,pred_y_test,isAlpha=True)
        
        trainer_alpha =  mytraining.Optimizer(loss_alpha,isAlpha=True)
        #---------------------------------------------------------------------- 
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1,allow_growth=True)) 
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

            
        #---------------------
        # Start training
        flag=False
        for i in range(nTraining):
            batchX, batchY1,batchY1Label,batchY2,batchY2Label,batchY3,batchY3Label = mytraining.myData.nextBatch(batchSize)
            # 特徴量ベクトルに変更 
            batchX = batchX[:,1:6,:]
            if isNormalization: 
                minbX = batchX.min(axis=2)[:,:,np.newaxis]
                maxbX = batchX.max(axis=2)[:,:,np.newaxis]     
                batchX = (batchX-minbX)/(maxbX-minbX)
            
            batchX = np.reshape(batchX,[-1,nCell*nWindow])
            batchY1,batchY2,batchY3 = batchY1[:,np.newaxis],batchY2[:,np.newaxis],batchY3[:,np.newaxis]   
            #batchY1,batchY2,batchY3 =  batchY1*10,batchY2*10,batchY3*10
            _,trainClassCenter,trainClassLoss1,trainClassLoss2,trainClassLoss3, Residual = sess.run([trainer_classification,class_center_all,loss1,loss2,loss3,residual_all],feed_dict={x:batchX,y1_class:batchY1,y2_class:batchY2,y3_class:batchY3,y1_class_label:batchY1Label,y2_class_label:batchY2Label,y3_class_label:batchY3Label})
            # regression
            _, trainRegLoss, TrainPred = sess.run([trainer_regression,loss_regression,predict_regression_residual_op],feed_dict={x:batchX,y1_class:batchY1,y2_class:batchY2,y3_class:batchY3,y1_class_label:batchY1Label,y2_class_label:batchY2Label,y3_class_label:batchY3Label})
            # alpha
            _, trainAlphaLoss, TrainAlpha = sess.run([trainer_alpha, loss_alpha, alpha_op],feed_dict={x:batchX,y1_class:batchY1,y2_class:batchY2,y3_class:batchY3,y1_class_label:batchY1Label,y2_class_label:batchY2Label,y3_class_label:batchY3Label})
            # Recover
            yTrainPred = trainClassCenter + (-1/TrainAlpha) * np.log((1/TrainPred) - 1)
            batchY = np.concatenate((batchY1,batchY2,batchY3),1)
            
            # total loss
            trainTotalLoss1 = np.mean(np.abs(batchY1-yTrainPred[:,nI][:,np.newaxis]))
            trainTotalLoss2 = np.mean(np.abs(batchY2-yTrainPred[:,tnI][:,np.newaxis]))
            trainTotalLoss3 = np.mean(np.abs(batchY3-yTrainPred[:,tI][:,np.newaxis]))
                 
            # Test & Evaluation
            if i % testPeriod == 0:
                
                if isEval:
                    testClassLoss1,testClassLoss2,testClassLoss3,testClassCenter = sess.run([loss1_test,loss2_test,loss3_test,class_center_all_test],feed_dict={x:testX})    
                    # regression
                    res1,testRegLoss,TestPred = sess.run([residual1_test_op,loss_regression_test,predict_regression_residual_test_op],feed_dict={x:testX})
                    # alpha
                    testAlphaLoss, TestAlpha = sess.run([loss_alpha_test, alpha_test_op],feed_dict={x:testX})
                    yTest = np.concatenate((testY1,testY2,testY3),axis=1)  
                    # Reduce
                    yTestPred = testClassCenter + (-1/TestAlpha) * np.log((1/TestPred) - 1)
                    
                    with open('./visualization/eval/TestATR_{}_{}_{}_{}_{}.pickle'.format(i,classMode,np.round(TrainAlpha,5),depth,10),'wb') as fp:
                            pickle.dump(yTestPred[:,0],fp)
                            pickle.dump(yTestPred[:,1],fp)
                            pickle.dump(yTestPred[:,2],fp)        

                else:
                    # classication
                    testClassLoss1,testClassLoss2,testClassLoss3,testClassCenter = sess.run([loss1_test,loss2_test,loss3_test,class_center_all_test],feed_dict={x:testX,y1_class:testY1,y2_class:testY2,y3_class:testY3,y1_class_label:testY1Label,y2_class_label:testY2Label,y3_class_label:testY3Label})    
                    # regression
                    res1,testRegLoss,TestPred = sess.run([residual1_test_op,loss_regression_test,predict_regression_residual_test_op],feed_dict={x:testX,y1_class:testY1,y2_class:testY2,y3_class:testY3,y1_class_label:testY1Label,y2_class_label:testY2Label,y3_class_label:testY3Label})
                    # alpha
                    testAlphaLoss, TestAlpha = sess.run([loss_alpha_test, alpha_test_op],feed_dict={x:testX,y1_class:testY1,y2_class:testY2,y3_class:testY3,y1_class_label:testY1Label,y2_class_label:testY2Label,y3_class_label:testY3Label})
                    yTest = np.concatenate((testY1,testY2,testY3),axis=1)  
                    # Reduce
                    yTestPred = testClassCenter + (-1/TestAlpha) * np.log((1/TestPred) - 1)
                    # total loss
                    testTotalLoss1  = np.mean(np.abs(testY1-yTestPred[:,nI][:,np.newaxis]))
                    testTotalLoss2  = np.mean(np.abs(testY2-yTestPred[:,tnI][:,np.newaxis]))
                    testTotalLoss3  = np.mean(np.abs(testY3-yTestPred[:,tI][:,np.newaxis]))
                
                    anc = testClassCenter[:10,0][np.newaxis] + res1[:10][np.newaxis]

                    print("ancVar",np.var(np.abs(testY1[:10][np.newaxis]-anc)))
                    print("テストAlpha")
                    print(TestAlpha)
                    print("------------------------------------")
                    print('b1regression',np.mean(np.abs(testY1-yTestPred[:,0][:,np.newaxis]))) 
                    print('b1regVar',np.var(np.abs(testY1-yTestPred[:,0][:,np.newaxis])))
                    print("------------------------------------")
                    print('b2regression',np.mean(np.abs(testY2-yTestPred[:,1][:,np.newaxis]))) 
                    print('b2regVar',np.var(np.abs(testY2-yTestPred[:,1][:,np.newaxis])))
                    print("------------------------------------")
                    print('b3regression',np.mean(np.abs(testY3-yTestPred[:,2][:,np.newaxis]))) 
                    print('b3regVar',np.var(np.abs(testY3-yTestPred[:,2][:,np.newaxis])))
                    print("---------------")
                    print("itr:%d,trainClassLoss1:%f,trainClassLoss2:%f,trainClassLoss3:%f, trainRegLoss:%f" % (i,trainClassLoss1,trainClassLoss2,trainClassLoss3,trainRegLoss))
                    print("itr:%d,trainTotalLoss1:%f,trainTotalLoss2:%f, trainTotalLoss2:%f" % (i,trainTotalLoss1,trainTotalLoss1, trainTotalLoss3))
                    print("-----------------------------------")
                    print("itr:%d,testClassLoss1:%f,testClassLoss2:%f,testClassLoss3:%f, testRegLoss:%f" % (i,testClassLoss1,testClassLoss2,testClassLoss3,testRegLoss))
                    print("itr:%d,testTotalLoss1:%f,testTotalLoss2:%f, testTotalLoss2:%f" % (i,testTotalLoss1,testTotalLoss1, testTotalLoss3))
                
                if not flag:
                    trainRegLosses,testRegLosses = trainRegLoss[np.newaxis],testRegLoss[np.newaxis]
                    trainClassLosses1,testClassLosses1 = trainClassLoss1[np.newaxis],testClassLoss1[np.newaxis]
                    trainTotalLosses1, testTotalLosses1 = trainTotalLoss1[np.newaxis],testTotalLoss1[np.newaxis]
                    trainClassLosses2,testClassLosses2 = trainClassLoss2[np.newaxis],testClassLoss2[np.newaxis]
                    trainTotalLosses2, testTotalLosses2= trainTotalLoss2[np.newaxis],testTotalLoss2[np.newaxis]
                    trainClassLosses3,testClassLosses3 = trainClassLoss3[np.newaxis],testClassLoss3[np.newaxis]
                    trainTotalLosses3, testTotalLosses3 = trainTotalLoss3[np.newaxis],testTotalLoss3[np.newaxis]
                    flag = True
                else:
                    trainRegLosses,testRegLosses = np.hstack([trainRegLosses,trainRegLoss[np.newaxis]]),np.hstack([testRegLosses,testRegLoss[np.newaxis]])
                    trainClassLosses1,testClassLosses1 = np.hstack([trainClassLosses1,trainClassLoss1[np.newaxis]]),np.hstack([testClassLosses1,testClassLoss1[np.newaxis]])
                    trainTotalLosses1,testTotalLosses1 = np.hstack([trainTotalLosses1,trainTotalLoss1[np.newaxis]]),np.hstack([testTotalLosses1,testTotalLoss1[np.newaxis]])
                    trainClassLosses2,testClassLosses2 = np.hstack([trainClassLosses2,trainClassLoss2[np.newaxis]]),np.hstack([testClassLosses2,testClassLoss2[np.newaxis]])
                    trainTotalLosses2,testTotalLosses2 = np.hstack([trainTotalLosses2,trainTotalLoss2[np.newaxis]]),np.hstack([testTotalLosses2,testTotalLoss2[np.newaxis]])
                    trainClassLosses3,testClassLosses3 = np.hstack([trainClassLosses3,trainClassLoss3[np.newaxis]]),np.hstack([testClassLosses3,testClassLoss3[np.newaxis]])
                    trainTotalLosses3,testTotalLosses3 = np.hstack([trainTotalLosses3,trainTotalLoss3[np.newaxis]]),np.hstack([testTotalLosses3,testTotalLoss3[np.newaxis]])
                
                
                with open('./visualization/residual/TestATR_{}_{}_{}_{}_{}.pickle'.format(i,classMode,np.round(TrainAlpha,5),depth,10),'wb') as fp:
                        pickle.dump(yTestPred[:,0],fp)
                        pickle.dump(yTestPred[:,1],fp)
                        pickle.dump(yTestPred[:,2],fp)        
                        pickle.dump(testY1,fp)
                        pickle.dump(testY2,fp)
                        pickle.dump(testY3,fp)
                """
                with open('./visualization/residual/TrainATR_{}_{}_{}_{}_{}.pickle'.format(i,classMode,np.round(TestAlpha,5),depth,10),'wb') as fp:
                        pickle.dump(yTrainPred[:,0],fp)
                        pickle.dump(yTrainPred[:,1],fp)
                        pickle.dump(yTrainPred[:,2],fp)
                        pickle.dump(batchY1,fp)
                        pickle.dump(batchY2,fp)
                        pickle.dump(batchY3,fp)"""        
                 
    #-----------------------------------------------------------------------
    """
    if methodModel == 1 or methodModel == 2:
        myPlot = Plot(isPlot=isPlot,methodModel=methodModel, nClass=nClass, alpha=depth) 
        myPlot.Plot_loss(trainTotalLosses1, testTotalLosses1, trainClassLosses1, testClassLosses1, trainRegLosses, testRegLosses, testPeriod,cell=1)
        myPlot.Plot_loss(trainTotalLosses2, testTotalLosses2, trainClassLosses2, testClassLosses2, trainRegLosses, testRegLosses, testPeriod,cell=2)
        myPlot.Plot_loss(trainTotalLosses3, testTotalLosses3, trainClassLosses3, testClassLosses3, trainRegLosses, testRegLosses, testPeriod,cell=3)"""

        
        
                
                 
                
                
                

