# -*- co    ding: utf-8 -*-

import os
import sys
import glob
import pickle
import pdb
import time
import random
import string

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import tensorflow as tf

class Data():
    """
    Data(x1,x2,y)
    target variable: y = U(ymin,ymax)
    object variable: x1 = sin(y) + 1/log(y) + noise, x2 = cos(y) + log(y) + noise
    noise : 0±sigma(to decide by oneself)
    """
    #def __init__(self,sigma,nClass,nTrain,nTest):
    def __init__(self,sigma,nClass):
        # 学習データ割合
        trainRatio = 0.8
        # 全データ数
        self.nData = 8000
        # 学習データ数
        self.nTrain = int(self.nData * trainRatio)
        # 評価データ数
        self.nTest = int(self.nData - self.nTrain)
        #self.nTrain = nTrain
        #self.nTest = nTest
        self.batchSize = 200
        self.batchCnt = 0
        self.batchRandInd = np.random.permutation(self.nTrain)
        # ノイズ平均、分散(1e-5)
        mu = 0
        sigma = np.float(sigma)
        # ノイズ
        self.noise = np.random.normal(mu,sigma,self.nData)
        # クラス数(大・中・小)
        self.nClass = nClass
        
    #---------------------------------------------    
    def TrainTest(self,pNum=6):
        # 目的変数の最小、最大値
        targetMin = 2
        targetMax = 6
        # 小数以下丸める
        limitDecimal = 6
        
        # Spiral Staircase 螺旋階段データ作成
        y = np.random.uniform(targetMin,targetMax,self.nData)
        x1 = np.sin(pNum * y) + 1 / np.log(y) + self.noise
        x2 = np.cos(pNum * y) + np.log(y) + self.noise
        
        # 全データを学習データと評価データに分割
        x1Train = x1[:self.nTrain][:,np.newaxis]
        x2Train = x2[:self.nTrain][:,np.newaxis]
        yTrain = y[:self.nTrain][:,np.newaxis]
        x1Test = x1[self.nTrain:][:,np.newaxis]
        x2Test = x2[self.nTrain:][:,np.newaxis]
        yTest = y[self.nTrain:][:,np.newaxis]


        #[データ数、次元数]
        return x1Train, x2Train, yTrain, x1Test, x2Test, yTest, y
    #---------------------------------------------    
    
    def Anotation(self,target):
        
        # 目的変数の最大、最小値
        yMin = target.min()
        yMax = target.max()
        # 四捨五入
        limitdecimal = 3
        # クラス刻み幅
        beta = np.round(np.abs(yMax - yMin)/self.nClass,limitdecimal)
        yClass = np.arange(yMin,yMax + beta,beta)
        
        flag = False
        for nInd in np.arange(target.shape[0]):
            tmpY = target[nInd]
            oneHot = np.zeros(len(yClass))
            ind = 0
            # (最小、最大]
            for threY in yClass:
                if (tmpY > threY) & (tmpY <= threY + beta):
                          oneHot[ind] = 1            
                ind += 1
            # 最小値は0番目のクラスにする
            if target[nInd] == yMin:
                oneHot[0] = 1
            # 最大値が一番最後のクラスにラベルされるのを戻す
            if target[nInd] == yMax:
                oneHot[-2] = 1
            
            tmpY = oneHot[np.newaxis] 
                  
            if not flag:
                Ylabel = tmpY
                flag = True
            else:
                Ylabel = np.vstack([Ylabel,tmpY])
        # 値が入っていないクラスを削除
        if len(yClass) == self.nClass+1:
            Ylabel = Ylabel[:,:-1]
        
        YTrainlabel = Ylabel[:self.nTrain]
        YTestlabel = Ylabel[self.nTrain:]
        #[データ数、クラス数]
        return YTrainlabel,YTestlabel,yMin,yMax
    #---------------------------------------------    
    def nextBatch(self,objectTrain,targetTrain,targetTrainlabel):
        sInd = self.batchSize * self.batchCnt
        eInd = sInd + self.batchSize
        
        batchX = objectTrain[self.batchRandInd[sInd:eInd],:]
        batchY = targetTrain[self.batchRandInd[sInd:eInd],:]
        batchlabelY = targetTrainlabel[self.batchRandInd[sInd:eInd],:]
        
        if eInd + self.batchSize > self.nTrain:
            self.batchCnt = 0
        else:
            self.batchCnt += 1
        
        return batchX,batchY,batchlabelY
 
#--------------------------------------------    
#---------------------------------------------    
class DeepModel():
    """
    Assessment: L1 loss for learning and assessment
    """
    def __init__(self,methodModel,dInput=2,nInput=10,nClass=10,yMin=0,yMax=1):
        # 引数で手法切り替え［naiveregression=0,MVR=1］
        self.methodMode = methodModel
        #self.ramda = ramda
        # Classificationの次元数
        self.nInput = nInput
        self.dInput = dInput
        self.nHidden = 128
        self.nHidden2 = 128
        self.nClass = nClass
        # Regressionの次元数
        self.nReg = 1
        if methodModel == 0:
            self.nInputReg = dInput
        else:
            self.nInputReg = self.nReg + dInput
        self.nRegHidden = 128
        self.nRegHidden2 = 128
        self.nRegHidden3 = 128
        self.nRegHidden4 = 128
        # yの最小、最大値
        sb = yMin
        eb = yMax
        limitdecimal = 3
        # クラス幅
        self.beta = np.round((eb - sb) / self.nClass,limitdecimal)
        # 一番初めのクラスの中心値
        self.sCenter = np.round(sb + (self.beta / 2),limitdecimal)
        # 狭める範囲
        self.aW = 1/20
        # 学習データ数
        self.lr = 1e-4
        
        
    #---------------------------------------------    
      
    def weight_variable(self,name,shape):
         return tf.get_variable(name,shape,initializer=tf.random_normal_initializer(stddev=0.1))
    
    def bias_variable(self,name,shape):
         return tf.get_variable(name,shape,initializer=tf.random_normal_initializer(stddev=0.1))

    def alpha_variable(self,name,shape):
        alphaMean = self.aW * (1/self.beta)
        #alphaInit = tf.random_normal_initializer(mean=alphaMean,stddev=0.1)
        alphaInit = tf.random_normal_initializer(mean=0.5,stddev=0.1)
        return tf.get_variable(name,shape,initializer=alphaInit)
     
    def fc_sigmoid(self,inputs,w,b,keepProb):
        sigmoid = tf.matmul(inputs,w) + b
        sigmoid = tf.nn.dropout(sigmoid,keepProb)
        sigmoid = tf.nn.sigmoid(sigmoid)
        return sigmoid

    def fc_relu(self,inputs,w,b,keepProb):
         relu = tf.matmul(inputs,w) + b
         relu = tf.nn.dropout(relu, keepProb)
         relu = tf.nn.relu(relu)
         return relu
     
    def fc(self,inputs,w,b,keepProb):
         fc = tf.matmul(inputs,w) + b
         fc = tf.nn.dropout(fc, keepProb)
         return fc
    
    #---------------------------------------------    
    
    def Classification(self,x,reuse=False):
        with tf.variable_scope('Classification') as scope:  
            keepProb = 1.0
            if reuse:
                keepProb = 1.0            
                scope.reuse_variables()
            
            #input -> hidden
            w1 = self.weight_variable('w1',[self.dInput,self.nHidden])
            bias1 = self.bias_variable('bias1',[self.nHidden])
            
            h1 = self.fc_relu(x,w1,bias1,keepProb) 
            
            w2 = self.weight_variable('w2',[self.nHidden,self.nHidden2])
            bias2 = self.bias_variable('bias2',[self.nHidden2])
            
            h2 = self.fc_relu(h1,w2,bias2,keepProb) 
            
            #hidden -> output
            w3 = self.weight_variable('w3',[self.nHidden2,self.nClass])
            bias3 = self.bias_variable('bias3',[self.nClass])
            
            y = self.fc(h2,w3,bias3,keepProb)
            
            return y
    #---------------------------------------------    
                    
    def Regression(self,x,reuse=False,isRR=False,name_scope="Regression",depth=0):
        with tf.variable_scope(name_scope) as scope:  
            keepProb = 1.0
            if reuse:
                keepProb = 1.0            
                scope.reuse_variables()
            w1_regression = self.weight_variable('w1_regression',[self.nInputReg,self.nRegHidden])
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
            
            
    #---------------------------------------------    
    def FeatrureVector(self,x,predict):
    
        # 予測したクラスの最大クラス
        pred_class_maxclass = tf.expand_dims(tf.cast(tf.argmax(predict,axis=1),tf.float32),1)  
        # 選択したクラスの中心値       
        predict_class_center = pred_class_maxclass * self.beta + self.sCenter
        # regression input(中央値＋特徴量)
        class_center_x =  tf.concat((predict_class_center,x),axis=1)
        # 残差
        residual = y_class - predict_class_center
        
        return predict_class_center,residual,class_center_x
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
            
    def Loss(self,y,predict,ramda=0.1,gl=0.1,isRegression=False,isGL=False,isAlpha=False):
        if isRegression:
            #return tf.reduce_mean(tf.square(y - predict))
            return tf.reduce_mean(tf.abs(y - predict))
        elif isGL:
            return tf.reduce_mean(tf.square(y - predict)) + (ramda * tf.reduce_sum(tf.sqrt(gl)))
        elif isAlpha:
            return tf.reduce_mean(tf.abs(y - predict))
        else:
            return tf.losses.softmax_cross_entropy(y,predict)
    #---------------------------------------------    
        
    def LossGroup(self,weight1): 
        group_weight = tf.reduce_sum(tf.square(weight1),axis=0)

        return group_weight
    
    def Optimizer(self,loss,isRegression=False,isGL=False,name_scope="Regression"):
        
        if isRegression:
            regressionVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=name_scope) 
            trainer_regression = tf.train.AdamOptimizer(self.lr).minimize(loss,var_list=regressionVars)
            return trainer_regression 
        
        if isGL:
            regressionVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=name_scope) 
            trainer_regression = tf.train.AdamOptimizer(self.lr).minimize(loss,var_list=regressionVars)
            return trainer_regression 
        
        else:
            classificationVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="Classification") 
            trainer = tf.train.AdamOptimizer(self.lr).minimize(loss,var_list=classificationVars)
            return trainer
    
#---------------------------------------------    
#---------------------------------------------    
class Plot():
    """
    Visualization: Point cloud of evaluation data is blue with 3 axes of (x1, x2, y)
    Predicted value
    """
    def __init__(self,isPlot=False,isRR=False,methodModel=1,sigma=0.1,nClass=1,alpha=1,pNum=6, nTrain=0, nTest=0, depth=0):
        self.isPlot = isPlot
        self.isRR = isRR
        self.methodModel = methodModel
        table = str.maketrans("", "" , string.punctuation + ".")
        sigma = str(sigma).translate(table)
        self.nClass = nClass
        self.alpha = alpha
        self.visualPath = "visualization"
        self.lossPath = "loss"
        self.pNum = pNum
        self.nTrain = nTrain
        self.nTest = nTest
        self.depth = depth
    
    def Plot_3D(self,x1Test,x2Test,yTest,yPred,isTrain=0):
        if self.isPlot:
             
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_zlabel("y")
            # 評価データplot
            ax.plot(np.reshape(x1Test,[-1,]),np.reshape(x2Test,[-1,]),np.reshape(yTest,[-1,]),"o",color="b",label="GT")
            # 予測値plot
            ax.plot(np.reshape(x1Test,[-1,]),np.reshape(x2Test,[-1,]),np.reshape(yPred,[-1,]),"o",color="r",label="Pred")
            plt.legend()
            fullPath = os.path.join(self.visualPath,"Pred_{}_{}_{}_{}_{}_{}_{}_{}_{}.png".format(self.methodModel,sigma,self.nClass,self.alpha, self.pNum, self.nTrain, self.nTest,self.depth,isTrain))
            plt.savefig(fullPath)
                
            
    def Plot_loss(self,trainTotalLosses, testTotalLosses, trainClassLosses, testClassLosses,trainRegLosses, testRegLosses, testPeriod):
        if self.isPlot:
            if self.methodModel==2 or self.methodModel==1:
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
                fullPath = os.path.join(self.visualPath,self.lossPath,"Loss_{}_{}_{}_{}_{}_{}_{}.png".format(self.methodModel,sigma,self.nClass,self.alpha, self.pNum, self.nTrain, self.nTest))
            else:
                plt.plot(np.arange(trainClassLosses.shape[0]),trainClassLosses,label="trainRegLosses",color="c")
                #plt.plot(np.arange(testClassLosses.shape[0]),testClassLosses,label="testRegLosses",color="pink")
                
                plt.ylim([0,0.5])
                plt.xlabel("iteration x {}".format(testPeriod))
            
                plt.legend()
                fullPath = os.path.join(self.visualPath,self.lossPath,"Loss_{}_{}_{}_{}_{}_{}.png".format(self.methodModel,sigma,self.nClass,self.alpha, self.pNum, self.nTrain, self.nTest))
            
            plt.savefig(fullPath)

    def Plot_Alpha(self,trainAlpha,testAlpha,testPeriod):
        if self.isPlot:
            plt.close()
            plt.plot(np.arange(trainAlpha.shape[1]),trainAlpha.T,label="trainAlpha",color="deepskyblue")
            plt.plot(np.arange(testAlpha.shape[1]),testAlpha.T,label="testAlpha",color="orange")
            
            plt.xlabel("iteration x {}".format(testPeriod))
            
            plt.legend()
            fullPath = os.path.join(self.visualPath,"Alpha_{}_{}_{}_{}_{}.png".format(self.methodModel,sigma,self.nClass,self.alpha, self.pNum))
            plt.savefig(fullPath)

        
            
if __name__ == "__main__":
    
    # 手法切り替え [naiveRegression=0,MVR=1]
    methodModel = int(sys.argv[1])
    # ノイズ分散
    sigma = np.float(sys.argv[2])
    # Classification クラス数
    nClass = int(sys.argv[3])
    # plotするときはTrue
    isPlot = True
    # l2をplotするとき
    pNum = int(sys.argv[4])
    # regression hidden
    depth = int(sys.argv[5])
     
    testPeriod = 500
    
    # nTrain,nTest:1000,7000 3200,800 500,3500 6400,1600
    """
    nTrain = 1000
    nTest = 7000
    dataPath = "trainData"
    toydataPath = "toyData_{}_{}_{}_{}.pickle".format(sigma,pNum,nTrain,nTest)
    toyDatafullPath = os.path.join(dataPath,toydataPath)
    with open(toyDatafullPath,"rb") as fp:
        x1Train = pickle.load(fp)
        x2Train = pickle.load(fp)
        yTrain = pickle.load(fp)
        x1Test = pickle.load(fp)
        x2Test = pickle.load(fp)
        yTest = pickle.load(fp)
        yData = pickle.load(fp)"""
    #-----------------------------------------------------------------------
    # データ作成
    #myData = Data(sigma,nClass,nTrain,nTest)
    myData = Data(sigma,nClass)
    x1Train,x2Train,yTrain,x1Test,x2Test,yTest,yData = myData.TrainTest(pNum=pNum)
    # クラスラベル付け
    yTrainlabel,yTestlabel,yMin,yMax = myData.Anotation(yData)
    # 入力ベクトル作成:[データ数、次元数(x,y)
    xTrain,xTest = np.hstack([x1Train,x2Train]),np.hstack([x1Test,x2Test])
    
    
    nTrain = xTrain.shape[0]
    nTest = xTest.shape[0]
    """
    # toyDataの保管&情報
    dataPath = "trainData"
    toydataPath = "toyData_{}_{}_{}_{}.pickle".format(sigma,pNum,nTrain,nTest)
    toyDatafullPath = os.path.join(dataPath,toydataPath)
     
    # データ保存
    with open(toyDatafullPath,"wb") as fp:
        pickle.dump(x1Train,fp)
        pickle.dump(x2Train,fp)
        pickle.dump(yTrain,fp)
        pickle.dump(x1Test,fp)
        pickle.dump(x2Test,fp)
        pickle.dump(yTest,fp)
        pickle.dump(yData,fp)
    """
    
    #-----------------------------------------------------------------------
    # input要素数,input次元数
    nInput = xTrain.shape[0]
    dInput = xTrain.shape[1]
    #yMax = np.maximum(np.max(yTrain),np.max(yTest))
    #yMin = np.minimum(np.min(yTrain),np.max(yTest))
    
    myDeepModel = DeepModel(methodModel,dInput=dInput,nInput=nInput,nClass=nClass,yMin=yMin,yMax=yMax)
    # learning rate
    lr = myDeepModel.lr
    # クラスの刻み幅
    beta = myDeepModel.beta
    # 学習回数
    nTraining = 300000
    nReg = myDeepModel.nReg
    # input of placeholder
    x = tf.placeholder(tf.float32,shape=[None,dInput])
    y_class = tf.placeholder(tf.float32,shape=[None,1])
    y_class_label = tf.placeholder(tf.int32,shape=[None,nClass])
    pred_residual = tf.placeholder(tf.float32,shape=[None,nReg])
    # Baseline regression & Baseline regression + Group Lasso
    if methodModel == 0:
        predict_regression_op = myDeepModel.Regression(x,isRR=True,depth=depth)
        predict_regression_test_op = myDeepModel.Regression(x,reuse=True,isRR=True,depth=depth)
        
        loss = myDeepModel.Loss(y_class,predict_regression_op,isRegression=True)
        loss_test = myDeepModel.Loss(y_class,predict_regression_test_op,isRegression=True)
        
        trainer_regression = myDeepModel.Optimizer(loss,isRegression=True)

        # L2制約
        #predict_regression_gl_op,w1,w2 = myDeepModel.GroupLassoRegression(x)
        #predict_regression_gl_test_op,w1_test,w2_test = myDeepModel.GroupLassoRegression(x,reuse=True)
        
        # l1
        #gl_op = tf.reduce_sum(tf.abs(w1)) + tf.reduce_sum(tf.abs(w2))
        #gl_test_op = tf.reduce_sum(tf.abs(w1_test)) + tf.reduce_sum(tf.abs(w2_test))
        
        # l2
        #gl_op = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
        #gl_test_op = tf.nn.l2_loss(w1_test) + tf.nn.l2_loss(w2_test)
        
        #gl_op = myDeepModel.LossGroup(w1)
        #gl_test_op = myDeepModel.LossGroup(w1_test)
        
        #loss_gl = myDeepModel.Loss(y_class,predict_regression_gl_op,ramda,gl_op,isGL=True)
        #loss_gl_test = myDeepModel.Loss(y_class,predict_regression_gl_test_op,ramda,gl_test_op,isGL=True)
        
        #trainer_gl_regression = myDeepModel.Optimizer(loss_gl,isGL=True,name_scope="GLRegression")
        
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1,allow_growth=True)) 
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        
        flag = False
        for i in range(nTraining):
            batchX,batchY,batchlabelY = myData.nextBatch(xTrain,yTrain,yTrainlabel)
            # regression
            _,yTrainPred,RegLoss = sess.run([trainer_regression,predict_regression_op,loss],feed_dict={x:batchX,y_class:batchY,y_class_label:batchlabelY})
            # l2 regression
            #_,yTrainL2Pred,RegL2Loss,Weights = sess.run([trainer_gl_regression,predict_regression_gl_op,loss_gl,w1],feed_dict={x:batchX,y_class:batchY,y_class_label:batchlabelY})
            
            if i % testPeriod == 0:
                # regression
                testRegLoss,yTestPred = sess.run([loss_test,predict_regression_test_op],feed_dict={x:xTest,y_class:yTest,y_class_label:yTestlabel})
                # l2 regression
                #testRegL2Loss,yTestL2Pred,GLTest  = sess.run([loss_gl_test,predict_regression_gl_test_op,gl_test_op],feed_dict={x:xTest,y_class:yTest,y_class_label:yTestlabel})
                
                print("itr:%d,trainRegLoss:%f,testRegLoss:%f" % (i,RegLoss,testRegLoss))
                print("Naivemean",np.mean(np.abs(yTest-yTestPred)))
                print("Naivevar",np.var(np.abs(yTest-yTestPred)))
                #----------------------------------------------------------------------
                """
                print("itr:%d,trainRegL2Loss:%f,testRegL2Loss:%f" % (i,RegL2Loss,testRegL2Loss))
                print("NaiveL2mean",np.mean(np.abs(yTest-yTestL2Pred)))
                print("NaiveL2var",np.var(np.abs(yTest-yTestL2Pred)))
                print(GLTest[:10])
                print(np.min(GLTest))
                print(np.max(GLTest))
                """
                if not flag:
                    trainRegLosses,testRegLosses = RegLoss[np.newaxis],testRegLoss[np.newaxis]
                    flag = True
                else:
                    trainRegLosses,testRegLosses = np.hstack([trainRegLosses,RegLoss[np.newaxis]]),np.hstack([testRegLosses,testRegLoss[np.newaxis]])
                
                with open('./predData/TestR_{}_{}_{}_{}_{}_{}.pickle'.format(i,methodModel,nClass,pNum,depth,yTest.shape[0]),'wb') as fp:
                        pickle.dump(x1Test,fp)
                        pickle.dump(x2Test,fp)
                        pickle.dump(yTestPred,fp)
                        pickle.dump(yTest,fp)
    
    # Not Truncate
    elif methodModel == 1:
        predict_classification_op = myDeepModel.Classification(x)
        predict_classification_test_op = myDeepModel.Classification(x,reuse=True)
        
        loss = myDeepModel.Loss(y_class_label,predict_classification_op)
        loss_test = myDeepModel.Loss(y_class_label,predict_classification_test_op)
        trainer_classification = myDeepModel.Optimizer(loss)
        #----------------------------------------------------------------------
        pred_class_center,residual_op,classCenter_x = myDeepModel.FeatrureVector(x,predict_classification_op)
        pred_class_center_test,residual_test_op,classCenter_x_test = myDeepModel.FeatrureVector(x,predict_classification_test_op)
        
        predict_regression_op = myDeepModel.Regression(classCenter_x,isRR=True,depth=depth)
        predict_regression_test_op = myDeepModel.Regression(classCenter_x_test,reuse=True,isRR=True,depth=depth)
        
        loss_regression = myDeepModel.Loss(residual_op,predict_regression_op,isRegression=True)
        loss_regression_test = myDeepModel.Loss(residual_test_op,predict_regression_test_op,isRegression=True) 
        trainer_regression = myDeepModel.Optimizer(loss_regression,isRegression=True) 
        #---------------------------------------------------------------------- 
        
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1,allow_growth=True)) 
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # start Training
        flag = False
        for i in range(nTraining):
            batchX,batchY,batchlabelY = myData.nextBatch(xTrain,yTrain,yTrainlabel)
            # classication
            _,trainClassCenter,trainClassLoss = sess.run([trainer_classification,pred_class_center,loss],feed_dict={x:batchX,y_class:batchY,y_class_label:batchlabelY})
            # regression
            _, trainRegLoss, TrainPred = sess.run([trainer_regression,loss_regression,predict_regression_op],feed_dict={x:batchX,y_class:batchY,y_class_label:batchlabelY})
            # Recover
            yTrainPred = trainClassCenter + TrainPred
            # total loss
            trainTotalLoss = np.mean(np.abs(batchY-yTrainPred))
            trainTotalVar  = np.var(np.abs(batchY-yTrainPred))
            # Test
            if i % testPeriod == 0:
                # classication
                testClassLoss,testClassCenter = sess.run([loss_test,pred_class_center_test],feed_dict={x:xTest,y_class:yTest,y_class_label:yTestlabel})    
                # regression
                testRegLoss,TestPred = sess.run([loss_regression_test,predict_regression_test_op],feed_dict={x:xTest,y_class:yTest,y_class_label:yTestlabel})
                # Reduce
                yTestPred = testClassCenter + TestPred 
                # total loss & var
                testTotalLoss  = np.mean(np.abs(yTest-yTestPred))
                testTotalVar  = np.var(np.abs(yTest-yTestPred))
                #-----------------------------------------------------------------------
                print("中心値")
                print(testClassCenter[:10])
                print("---------------")
                print("予測 y")
                print(yTestPred[:10])
                print("---------------")
                print("Res",TestPred[:10])
                print("真値")
                print(yTest[:10])
                print("---------------")
                print("itr:%d,trainClsLoss:%f,trainRegLoss:%f, trainTotalLoss:%f, trainTotalVar:%f" % (i,trainClassLoss,trainRegLoss, trainTotalLoss, trainTotalVar))
                print("-----------------------------------")
                print("itr:%d,testClsLoss:%f,testRegLoss:%f, testTotalLoss:%f, testTotalVar:%f" % (i,testClassLoss,testRegLoss, testTotalLoss, testTotalVar)) 
                #-----------------------------------------------------------------------
                
                if not flag:
                    trainRegLosses,testRegLosses = trainRegLoss[np.newaxis],testRegLoss[np.newaxis]
                    trainClassLosses,testClassLosses = trainClassLoss[np.newaxis],testClassLoss[np.newaxis]
                    trainTotalLosses, testTotalLosses = trainTotalLoss[np.newaxis],testTotalLoss[np.newaxis]
                    flag = True
                else:
                    trainRegLosses,testRegLosses = np.hstack([trainRegLosses,trainRegLoss[np.newaxis]]),np.hstack([testRegLosses,testRegLoss[np.newaxis]])
                    trainClassLosses,testClassLosses = np.hstack([trainClassLosses,trainClassLoss[np.newaxis]]),np.hstack([testClassLosses,testClassLoss[np.newaxis]])
                    trainTotalLosses,testTotalLosses = np.hstack([trainTotalLosses,trainTotalLoss[np.newaxis]]),np.hstack([testTotalLosses,testTotalLoss[np.newaxis]])
                """
                with open('./predData/TestCR_{}_{}_{}_{}_{}_{}.pickle'.format(i,methodModel,nClass,pNum,depth,yTest.shape[0]),'wb') as fp:
                        pickle.dump(x1Test,fp)
                        pickle.dump(x2Test,fp)
                        pickle.dump(yTestPred,fp)
                        pickle.dump(yTest,fp)"""
    
        
    # MVR
    else:
        predict_classification_op = myDeepModel.Classification(x)
        predict_classification_test_op = myDeepModel.Classification(x,reuse=True)
        
        loss = myDeepModel.Loss(y_class_label,predict_classification_op)
        loss_test = myDeepModel.Loss(y_class_label,predict_classification_test_op)
        trainer_classification = myDeepModel.Optimizer(loss)
        #----------------------------------------------------------------------
        pred_class_center,residual_op,classCenter_x = myDeepModel.FeatrureVector(x,predict_classification_op)
        pred_class_center_test,residual_test_op,classCenter_x_test = myDeepModel.FeatrureVector(x,predict_classification_test_op)
        
        predict_regression_op = myDeepModel.Regression(classCenter_x,depth=depth)
        predict_regression_test_op = myDeepModel.Regression(classCenter_x_test,reuse=True,depth=depth)
        
        magnify_residual,alpha_op = myDeepModel.Magnify(pred_residual)
        magnify_residual_test,alpha_test_op = myDeepModel.Magnify(pred_residual,reuse=True)
        
        loss_regression = myDeepModel.Loss(magnify_residual,predict_regression_op,isRegression=True)
        loss_regression_test = myDeepModel.Loss(magnify_residual_test,predict_regression_test_op,isRegression=True) 
        trainer_regression = myDeepModel.Optimizer(loss_regression,isRegression=True) 
        #---------------------------------------------------------------------- 
        reduce_residual = myDeepModel.Reduce(predict_regression_op,alpha_op)
        reduce_residual_test = myDeepModel.Reduce(predict_regression_test_op,alpha_test_op,reuse=True)

        pred_y = pred_class_center +  reduce_residual
        pred_y_test = pred_class_center_test + reduce_residual_test

        loss_alpha = myDeepModel.Loss(y_class,pred_y,isAlpha=True)
        loss_alpha_test = myDeepModel.Loss(y_class,pred_y_test,isAlpha=True)

        trainer_alpha = myDeepModel.Optimizer(loss_alpha,isRegression=True,name_scope="Magnify")
        
        grads = tf.gradients(loss_alpha,alpha_op)
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1,allow_growth=True)) 
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # start Training
        flag = False
        for i in range(nTraining):
            batchX,batchY,batchlabelY = myData.nextBatch(xTrain,yTrain,yTrainlabel)
            # classication
            _,trainClassCenter,trainClassLoss, Residual = sess.run([trainer_classification,pred_class_center,loss,residual_op],feed_dict={x:batchX,y_class:batchY,y_class_label:batchlabelY})
            # regression
            _, trainRegLoss, TrainPred = sess.run([trainer_regression,loss_regression,predict_regression_op],feed_dict={x:batchX,y_class:batchY,y_class_label:batchlabelY,pred_residual:Residual})
            # alpha
            _, trainAlphaLoss, TrainAlpha, Traingrad = sess.run([trainer_alpha, loss_alpha, alpha_op, grads],feed_dict={x:batchX,y_class:batchY,y_class_label:batchlabelY,pred_residual:Residual})
            # Recover
            yTrainPred = trainClassCenter + (-1/TrainAlpha) * np.log((1/TrainPred) - 1)
            # total loss
            trainTotalLoss = np.mean(np.abs(batchY-yTrainPred))
            trainTotalVar = np.var(np.abs(batchY-yTrainPred))
            TrainResidualat = 1/(1+np.exp(-TrainAlpha*Residual))
            TrainBigResidual = np.where((0.0==TrainResidualat)|(TrainResidualat==1.0))
            bigResidualpar = TrainBigResidual[0].shape[0] / batchY.shape[0]
            # Test
            if i % testPeriod == 0:
                # classication
                testClassLoss,testClassCenter,TestResidual = sess.run([loss_test,pred_class_center_test,residual_test_op],feed_dict={x:xTest,y_class:yTest,y_class_label:yTestlabel})    
                # regression
                testRegLoss,TestPred = sess.run([loss_regression_test,predict_regression_test_op],feed_dict={x:xTest,y_class:yTest,y_class_label:yTestlabel,pred_residual:TestResidual})
                # alpha
                testAlphaLoss, TestAlpha, Testgrad = sess.run([loss_alpha_test, alpha_test_op, grads],feed_dict={x:xTest,y_class:yTest,y_class_label:yTestlabel,pred_residual:TestResidual})
                # Reduce
                yTestPred = testClassCenter + (-1/TestAlpha) * np.log((1/TestPred) - 1)
                # total loss
                testTotalLoss  = np.mean(np.abs(yTest-yTestPred))
                testTotalVar = np.var(np.abs(yTest-yTestPred))
                
                TestResidualat = 1/(1+np.exp(-TestAlpha*TestResidual))
                TestBigResidual = np.where((0.0==TestResidualat)|(TestResidualat==1.0))
                TestbigResidualpar = TestBigResidual[0].shape[0] / yTest.shape[0]
                #-----------------------------------------------------------------------
                print("テストAlpha")
                print(TestAlpha)
                print("BigTrainResidual割合")
                print(bigResidualpar)
                print("BigTestResidual割合")
                print(TestbigResidualpar)
                print("-----------------------------------")
                
                """
                print("Test勾配")
                print(Testgrad)
                print("中心値")
                print(testClassCenter[:10])
                print("---------------")
                print("予測 y")
                print(yTestPred[:10])
                print("---------------")
                print("真値")
                print(yTest[:10])
                print("---------------")
                """
                print("itr:%d,trainMagLoss:%f,trainRegLoss:%f, trainTotalLoss:%f, trainTotalVar:%f" % (i,trainClassLoss,trainRegLoss, trainTotalLoss, trainTotalVar))
                print("-----------------------------------")
                print("itr:%d,testMagLoss:%f,testRegLoss:%f, testTotalLoss:%f, testTotalVar:%f" % (i,testClassLoss,testRegLoss, testTotalLoss, testTotalVar)) 
                #-----------------------------------------------------------------------
                if not flag:
                    trainRegLosses,testRegLosses = trainRegLoss[np.newaxis],testRegLoss[np.newaxis]
                    trainClassLosses,testClassLosses = trainClassLoss[np.newaxis],testClassLoss[np.newaxis]
                    trainTotalLosses, testTotalLosses = trainTotalLoss[np.newaxis],testTotalLoss[np.newaxis]
                    trainAlpha,testAlpha = TrainAlpha[np.newaxis],TestAlpha[np.newaxis]
                    flag = True
                else:
                    trainRegLosses,testRegLosses = np.hstack([trainRegLosses,trainRegLoss[np.newaxis]]),np.hstack([testRegLosses,testRegLoss[np.newaxis]])
                    trainClassLosses,testClassLosses = np.hstack([trainClassLosses,trainClassLoss[np.newaxis]]),np.hstack([testClassLosses,testClassLoss[np.newaxis]])
                    trainTotalLosses,testTotalLosses = np.hstack([trainTotalLosses,trainTotalLoss[np.newaxis]]),np.hstack([testTotalLosses,testTotalLoss[np.newaxis]])
                    trainAlpha,testAlpha = np.hstack([trainAlpha,TrainAlpha[np.newaxis]]),np.hstack([testAlpha,TestAlpha[np.newaxis]])
                
                with open('./predData/ATR_{}_{}_{}_{}_{}_{}_{}.pickle'.format(i,methodModel,nClass,pNum,depth,TestAlpha,yTest.shape[0]),'wb') as fp:
                        pickle.dump(x1Test,fp)
                        pickle.dump(x2Test,fp)
                        pickle.dump(yTestPred,fp)
                        pickle.dump(yTest,fp)
    #-----------------------------------------------------------------------
    # モデル保存
    """
    modelFileName = "model_{}_{}_{}_{}_{}_{}.ckpt".format(methodModel,sigma,nClass,pNum,nTrain,nTest)
    modelPath = "models"
    modelfullPath = os.path.join(modelPath,modelFileName)
    saver.save(sess,modelfullPath)
    """
    # Proposed
    if methodModel == 2:
        # 可視化
        myPlot = Plot(isPlot=isPlot,methodModel=methodModel,sigma=sigma, nClass=nClass, alpha=TestAlpha, pNum=pNum, nTrain=nTrain, nTest=nTest,depth=depth)
        #myPlot.Plot_loss(trainTotalLosses, testTotalLosses, trainClassLosses, testClassLosses, trainRegLosses, testRegLosses, testPeriod)
        myPlot.Plot_3D(x1Test,x2Test,yTest,yTestPred)
        myPlot.Plot_3D(batchX[:,0],batchX[:,1],batchY,yTrainPred,isTrain=1)
        #myPlot.Plot_Alpha(trainAlpha,testAlpha,testPeriod)
    
    elif methodModel == 1:
        myPlot = Plot(isPlot=isPlot,methodModel=methodModel,sigma=sigma, nClass=nClass, alpha=0, pNum=pNum, nTrain=nTrain, nTest=nTest,depth=depth)
        
        # regression + classification
        #myPlot.Plot_loss(trainTotalLosses, testTotalLosses, trainClassLosses, testClassLosses, trainRegLosses, testRegLosses, testPeriod)
        myPlot.Plot_3D(x1Test,x2Test,yTest,yTestPred)
        myPlot.Plot_3D(batchX[:,0],batchX[:,1],batchY,yTrainPred,isTrain=1)
    else:
        myPlot = Plot(isPlot=isPlot,methodModel=methodModel,sigma=sigma, nClass=nClass, alpha=0, pNum=pNum, nTrain=nTrain, nTest=nTest,depth=depth) 
        # regression
        #myPlot.Plot_loss(0, 0, trainRegLosses, testRegLosses, RegLoss, testRegLoss, testPeriod)
        myPlot.Plot_3D(x1Test,x2Test,yTest,yTestPred)
        myPlot.Plot_3D(batchX[:,0],batchX[:,1],batchY,yTrainPred,isTrain=1)
        #regression + l2
        #myPlot.Plot_loss(0, 0, trainRegL2Losses, testRegL2Losses, RegL2Loss, testRegL2Loss, testPeriod)
        #myPlot.Plot_3D(x1Test,x2Test,yTest,yTestL2Pred)
        #myPlot.Plot_3D(batchX[:,0],batchX[:,1],batchY,yTrainL2Pred,isTrain=1)
