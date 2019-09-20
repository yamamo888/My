# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 17:14:14 2018

@author: yu
"""
# -*- coding: utf-8 -*-
import shutil
import os
import sys
import numpy as np
import pdb
import tensorflow as tf
from tensorflow.python.ops import nn_ops
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
import pickle
import pywt
import glob

import pandas as pd
import seaborn as sns
from scipy import stats
import random
import itertools
import math
import scipy.optimize

#########################################
class EarthQuakePlateModel:
        
    def __init__(self,dataPath,logName,nCell=8,nYear=10000):
        

        # Path
        self.logPath = './logs'
        self.features = 'features'
        self.visualPath = 'visualization' 
        self.dataPath = dataPath 
        self.logName = logName
        self.logFullPath = os.path.join(self.logPath,self.dataPath,logName)
        # パラメータ
        self.nCell = nCell
        self.nYear = nYear
        self.yInd = 1
        self.vInds = [2,3,4,5,6,7,8,9]
        self.yV = np.zeros([nYear,nCell])
        
        
        
    #--------------------------

    #--------------------------
    #データの読み込み
    def loadABLV(self):
        self.data = open(self.logFullPath).readlines()
        
        # A, B, Lの取得
        self.A = np.zeros(self.nCell)
        self.B = np.zeros(self.nCell)
        self.L = np.zeros(self.nCell)
        
        for i in np.arange(1,self.nCell+1):
            tmp = np.array(self.data[i].strip().split(",")).astype(np.float32)
            self.A[i-1] = tmp[0]
            self.B[i-1] = tmp[1]
            self.L[i-1] = tmp[4]
            
        
        # Vの開始行取得
        isRTOL = [True if self.data[i].count('value of RTOL')==1 else False for i in np.arange(len(self.data))]
        vInd = np.where(isRTOL)[0][0]+1
        
        # Vの値の取得（vInd行から最終行まで）
        flag = False
        for i in np.arange(vInd,len(self.data)):
            tmp = np.array(self.data[i].strip().split(",")).astype(np.float32)
            
            if not flag:
                self.V = tmp
                flag = True
            else:
                self.V = np.vstack([self.V,tmp])

    #--------------------------
    
    #--------------------------
    # Vを年単位のデータに変換
    """ 
    def convV2YearlyData(self): 
        for year in np.arange(self.nYear):
            if np.sum(np.floor(self.V[:,self.yInd])==year):
                self.yV[year,:] = np.mean(self.V[np.floor(self.V[:,self.yInd])==year,self.vInds[0]:],axis=0)
        self.yV = self.yV.T"""
        
    def convV2YearlyData(self):
        # 初めの観測した年
        sYear = np.floor(self.V[0,self.yInd])
        
        # 観測データがない年には観測データの１つ前のデータを入れる(累積)
        for year in np.arange(sYear,self.nYear):
            # 観測データがある場合
            if np.sum(np.floor(self.V[:,self.yInd])==year):
                
                # 観測データがあるときはそのまま代入
                self.yV[int(year)] = self.V[np.floor(self.V[:,self.yInd])==year,self.vInds[0]:]
            
            # 観測データがない場合
            else:
                # その1つ前の観測データを入れる
                self.yV[int(year)] = self.yV[int(year)-1,:]
        # 累積速度から、速度データにする
        deltaV = self.yV[self.yInd:]-self.yV[:-self.yInd]
        # 一番最初のデータをappendして、10000年にする
        self.yV = np.concatenate((self.yV[np.newaxis,0],deltaV),0)
        self.yV = self.yV.T
        
    #--------------------------
        
    #--------------------------
    # Vの生データのプロット
    def plotV(self,isPlotShow=False,isYearly=True,prefix='yV'):
        """
        self.yV = self.yV[1:6]
        
        if isPlotShow:
            plt.close()
            fig, figInds = plt.subplots(nrows=5, sharex=True)
            #pdb.set_trace()
            
            for figInd in np.arange(len(figInds)):
                #figInds[figInd].plot(self.V[:,self.yInd],self.V[:,self.vInds[0]+figInd])
                #if isYearly:
                figInds[figInd].plot(np.arange(self.nYear), self.yV[figInd,:].T)

            fullPath = os.path.join(self.visualPath,"V_{}.png".format(self.logName))
            plt.savefig(fullPath)
            #fig.show()
            #plt.show()            
        """
        #fullPath = os.path.join(self.features,'b2b3b4Files','train1',"{}".format(self.logName))
        fullPath = os.path.join(self.features,'train1',"{}".format(self.logName))
        with open(fullPath,'rb') as fp:
            y = pickle.load(fp)
            yV = pickle.load(fp)
        pdb.set_trace()
        
        fullPath = os.path.join("FFT_{}.png".format(self.logName))
        yV = yV[1:6,:]
      
        fig, figInds = plt.subplots(nrows=5, sharex=True)
        for figInd in np.arange(len(figInds)):
            figInds[figInd].plot(np.arange(10), yV[figInd,:])
        
        plt.savefig(fullPath)
        
        
        """ 
        # pklデータの保存
        fullPath = os.path.join(self.features,self.dataPath,"{}.pkl".format(self.logName))
        with open(fullPath,'wb') as fp:
            pickle.dump(log.B,fp)
            pickle.dump(self.yV,fp)
            #pickle.dump(self.yVkde,fp)
            #pickle.dump(self.yVfft,fp)
            #pickle.dump(self.X,fp)
        
        #pdb.set_trace()

        listy = []
        fullPath = os.path.join(self.features,self.dataPath,"{}.pkl".format(self.logName))
        with open(fullPath,'rb') as fp:
            trueB = pickle.load(fp)
            yV = pickle.load(fp)
        #pdb.set_trace() 
        logNum = int(self.logName[1])-1 
        yV = yV[logNum,2000:]
        
        plt.plot(yV)
        plt.savefig('V{}.png'.format(self.logName))
        plt.close() 
        # 任意の数字
        d1 = np.array(np.where(yV>1)).T
        d2 = np.append(np.array(np.where(yV>1))[0,1:],[0])[:,np.newaxis]
        delta = d2 - d1
        mean = int(np.mean(delta[:-1,0]))
        print(mean)
        listy.append(mean)
        print(listy)"""
    
    #--------------------------
    

    

    #------------------------------------
    # イベントデータ（デルタ関数）を、KDEで滑らかにつなげる
    def KDE(self, v_divid = 10.0, bw = 0.01):

        flag = False
        for cellInd in np.arange(self.nCell):

            #　速度vから地震が何回起きたか相対的に取得
            v_width = self.yV[cellInd,:].max() / v_divid
            eqp_num = np.floor(self.yV[cellInd,:] / v_width)
                
            # list(float) -> list(int) -> array
            eqp_tmp = list(map(int,eqp_num))
            eqp_num = np.array(eqp_tmp)

            # 年数を任意の回数増やした(0回のデータは消える)
            eqp_v = np.repeat(np.arange(0,self.nYear),eqp_num)
            
            # KDE
            x_grid = np.arange(0,self.nYear)
            kde_model = stats.gaussian_kde(eqp_v,bw_method=bw)
            kde_model = kde_model(x_grid)
            kde_model = kde_model[np.newaxis,:]

            if not flag:
                self.yVkde = kde_model
                flag = True
            else:
                self.yVkde = np.concatenate((self.yVkde,kde_model),axis=0)
        """ 
        fullPath = os.path.join("KDE_{}.png".format(self.logName))

        fig, figInds = plt.subplots(nrows=3, sharex=True)
        for figInd in np.arange(len(figInds)):
            figInds[figInd].plot(np.arange(8000), self.yVfft[figInd,:])
        
        plt.savefig(fullPath)"""
    #--------------------------

    #--------------------------
    # 周波数特徴量の抽出
    def FFT(self,widthWindow=25,eFrq=250, sYear=2000, eYear=10000):

        # FFTの計算
        self.yVfft = np.abs(np.fft.fft(self.yVkde[:,sYear:eYear]))
        
        
        #----------------------
        # スペクトラムをスライディングウィンドウごとに平均をとった特徴量の抽出
        flag = False
        for cnt in np.arange(int(eFrq/widthWindow)):
            sInd = widthWindow * cnt + 1
            eInd = sInd + widthWindow
            
            # ウィンドウのスペクトラムの平均(周波数スペクトル)（ピークに反応できない場合）
            #平均や最小値をとったりする（次元数を増やす必要がある）
            #X = np.mean(self.yVfft[:,sInd:eInd],axis=1)
            X = np.max(self.yVfft[:,sInd:eInd],axis=1)
            X = X[np.newaxis]

            if not flag:
                self.X = X
                flag = True
            else:
                self.X = np.concatenate((self.X,X),axis=0)

        self.X = self.X.T
        """
        fullPath = os.path.join(self.features,"b2b3b4b5b6200300","{}.pkl".format(self.logName))
        with open(fullPath,'wb') as fp:
            pickle.dump(log.B,fp)
            pickle.dump(self.X,fp)
        
        """
        fullPath = os.path.join("FFT_{}.png".format(self.logName))

        fig, figInds = plt.subplots(nrows=6, sharex=True)
        for figInd in np.arange(len(figInds)):
            figInds[figInd].plot(np.arange(250), self.yVfft[figInd,:250])
        
        plt.savefig(fullPath)
    
    #--------------------------
    def Specgram(self,width=200,nStride=100,nYear=8000):

        # Window数
        nWindow = int((nYear-width)/nStride)
        
        #スペクトル初期化(セル,ウィンドウ数,ウインド幅)
        self.yVkdeSpec = np.zeros([self.nCell,nWindow,width])
        # 最初の2000年は信ぴょう性がないので省く
        self.yVkde = self.yVkde[:,2000:]

        flag = False
        ## 刻み幅ずつ刻みながら窓幅分のデータをフーリエ変換
        for fftInd in range(nWindow):
            # フレーム取り出し
            frame = self.yVkde[:,fftInd*nStride:fftInd*nStride+width]
            # FFT
            yVkdeSpec = np.abs(np.fft.fft(frame)) 

            # abs(SFTP)**2
            #yVkdeSpec = np.abs(np.fft.fft(frame))**2
            
            if not flag:
                flag = True
                self.yVkdeSpec = yVkdeSpec[:,np.newaxis,:]
            else:
                self.yVkdeSpec = np.concatenate((self.yVkdeSpec,yVkdeSpec[:,np.newaxis,:]),1)

    #--------------------------

    #-------------------------- 
    # スペクトラム画像のプロット
    def PlotSpecgram_KDE(self,width=200,nYear=8000):
        
        self.yVkdeSpec = self.yVkdeSpec.transpose(0,2,1)
        
        # スペクトラム画像
        # vmin vmaxでRGBの範囲が決まる(濃淡に変化)
        # 5000/100=50hzの範囲
        plt.close()
        fig,figInds = plt.subplots(nrows=self.nCell,sharex=True)
        for figInd in np.arange(len(figInds)):    
            figInds[figInd].imshow(self.yVkdeSpec[figInd,1:100,:], cmap='gray',vmin=np.min(self.yVkdeSpec[figInd,1:100,:]),vmax=np.max(self.yVkdeSpec[figInd,1:100,:]), aspect='auto')
             
        # 保存
        SpecfullPath = os.path.join(self.visualPath,'specgram',"yVkde_{}.png".format(self.logName))
        plt.savefig(SpecfullPath)
        
        # 元kde
        plt.close()
        fig,figInds = plt.subplots(nrows=self.nCell,sharex=True)
        for figInd in np.arange(len(figInds)):
            figInds[figInd].plot(np.arange(0,nYear),self.yVkde[figInd,:])

        # Save
        KDEfullPath = os.path.join(self.visualPath,'kde',"yVkde_{}.png".format(self.logName))
        plt.savefig(KDEfullPath)
#########################################

#########################################
class Data:
    ############ 引数説明 #################################
    #fname:Class EarthQuakePlateModelで保存したpicklefileすべて
    #trainRatio:学習とテストデータの分け方
    #nCell,sYear,bInd,eYear:このクラスで使うparameter
    #isWindow:Windows用
    #isClass:classification用
    #cellMode,datapickleMode:mainに詳細,Modeの設定
    #dataPath:pickledataが入っているディレクトリ
    #xydataPath:X,Yが保存してあるpickleファイル指定
    #traintestdataPath:TrainTestが保存してあるpickleファイル指定
    #####################################################
        
    #####[3] cellMode（出力するセルのパラメータ[b]指定） #########
    # cellMode=12: (2次元の出力）
    # cellMode=123: (3次元の出力)
    #####################################################
    
    ####[4] datapickleMode （pickleが３種類）######################
    # datapickleMode=1(3つめ): Load Train,Test data
    # datapickelMode=2(2つめ): Save Train,Test data Load X,Y
    # datapickleMode=3(１つめ): Save X,Y Load yV etc...
    ###########################################################
    
    def __init__(self,fname='kde_fft_log_10*',trainRatio=0.8, nCell=8, 
                 sYear=2000, bInd=0, eYear=10000, isWindows=True, isClass=False, 
                 CellMode=1,datapickleMode=1,classMode=1,nClass=10,featuresPath='./features', dataPath='datab1',
                 trainingpicklePath='traintestdatab1.pkl',picklePath='xydatab1.pkl'):
        
         
        # pklファイルの一覧
        self.tPath = 'train4'
        fullPath = os.path.join(featuresPath,dataPath,self.tPath,fname)
        files = glob.glob(fullPath)
        self.nData = len(files)
        """
        
        self.nTrain = np.floor(self.nData * trainRatio).astype(int)
        self.oneTrain = np.floor(self.nTrain/6).astype(int)
        self.nTest = self.nData - self.nTrain
        self.oneTest = np.floor(self.nTest/2).astype(int)
        
        # fileをシャッフル
        random.shuffle(files)
        

        testfiles = files[:self.nTest]
        trainfiles = files[self.nTest:]
        
        test1 = testfiles[:self.oneTest]
        test2 = testfiles[self.oneTest:]

        train1 = trainfiles[:self.oneTrain]
        train2 = trainfiles[self.oneTrain:self.oneTrain*2]
        train3 = trainfiles[self.oneTrain*2:self.oneTrain*3]
        train4 = trainfiles[self.oneTrain*3:self.oneTrain*4]
        train5 = trainfiles[self.oneTrain*4:self.oneTrain*5]
        train6 = trainfiles[self.oneTrain*5:]
      pdb.set_trace() 
        # ファイル移動
        for i in test1:
            shutil.move(i,os.path.join('features','test1',i.split('/')[2]))
        for i in test2:
            shutil.move(i,os.path.join('features','test2',i.split('/')[2]))
        
        for i in train1:
            shutil.move(i,os.path.join('features','train1',i.split('/')[2]))
        for i in train2:
            shutil.move(i,os.path.join('features','train2',i.split('/')[2]))
        for i in train3:
            shutil.move(i,os.path.join('features','train3',i.split('/')[2]))
        for i in train4:
            shutil.move(i,os.path.join('features','train4',i.split('/')[2]))
        for i in train5:
            shutil.move(i,os.path.join('features','train5',i.split('/')[2]))
        for i in train6:
            shutil.move(i,os.path.join('features','train6',i.split('/')[2]))
        """
        self.nTrain = 100476
        #self.nTrain = 103030
        # データの領域確保
        #self.nData = 90601
        # b1b2b3b4用
        #self.nData = 28561 
        # バッチの初期化(mode=1の時のため)
        #self.batch12Cnt = 0
        #self.batch23Cnt = 0
        #self.batch34Cnt = 0
        self.batchCnt = 0
        
        self.batchRandInd = np.random.permutation(self.nTrain)
        #testRandInd = np.random.permutation(371293)
        
        picklefiles = 'b2b3b4pickles'
        #traintestfullPath = os.path.join(featuresPath,dataPath,trainingpicklePath)
        # b2b3b4用
        xyfullPath = os.path.join(featuresPath,picklefiles,picklePath)
        #pdb.set_trace()
        self.CellMode = CellMode
        
        
        if datapickleMode == 1:
            #train x,ytestのpickleファイル読み込み
                    
            if CellMode == 12 or CellMode == 23 or CellMode == 34 or CellMode == 45 or CellMode == 56 or CellMode == 67 or CellMode == 78:
                with open(traintestfullPath,'rb') as fp:
                    self.xTrain = pickle.load(fp)
                    self.y1Train = pickle.load(fp)
                    self.y2Train = pickle.load(fp)
                    self.y1TrainLabel = pickle.load(fp)
                    self.y2TrainLabel = pickle.load(fp)
                    
                    self.xTest = pickle.load(fp)
                    self.y1Test = pickle.load(fp)
                    self.y2Test = pickle.load(fp)
                    self.y1TestLabel = pickle.load(fp)
                    self.y2TestLabel = pickle.load(fp)
                    
            elif CellMode == 23456:
                picklesPath = 'b2b3b4b5b6pickles' 
                train1 = 'b2b3b4b5b6_train1{}u'.format(classMode)
                train2 = 'b2b3b4b5b6_train2{}u'.format(classMode)
                train3 = 'b2b3b4b5b6_train3{}u'.format(classMode)
                train4 = 'b2b3b4b5b6_train4{}u'.format(classMode)
                train5 = 'b2b3b4b5b6_train5{}u'.format(classMode)
                train6 = 'b2b3b4b5b6_train6{}u'.format(classMode)
                test1 = 'b2b3b4b5b6_test1{}u'.format(classMode)
                test2 = 'b2b3b4b5b6_test2{}u'.format(classMode)
                
                with open(os.path.join(featuresPath,picklesPath,train1),'rb') as fp:
                    self.x11Train = pickle.load(fp)
                    self.y11TrainLabel = pickle.load(fp)
                    self.y21TrainLabel = pickle.load(fp)
                    self.y31TrainLabel = pickle.load(fp)
                    self.y41TrainLabel = pickle.load(fp)
                    self.y51TrainLabel = pickle.load(fp)
                    self.y11Train = pickle.load(fp)
                    self.y21Train = pickle.load(fp)
                    self.y31Train = pickle.load(fp)
                    self.y41Train = pickle.load(fp)
                    self.y51Train = pickle.load(fp)
                """
                with open(os.path.join(featuresPath,picklesPath,train2),'rb') as fp:
                    self.x12Train = pickle.load(fp)
                    self.y12TrainLabel = pickle.load(fp)
                    self.y22TrainLabel = pickle.load(fp)
                    self.y32TrainLabel = pickle.load(fp)
                    self.y42TrainLabel = pickle.load(fp)
                    self.y52TrainLabel = pickle.load(fp)
                    self.y12Train = pickle.load(fp)
                    self.y22Train = pickle.load(fp)
                    self.y32Train = pickle.load(fp)
                    self.y42Train = pickle.load(fp)
                    self.y52Train = pickle.load(fp)
                """
                with open(os.path.join(featuresPath,picklesPath,train3),'rb') as fp:
                    self.x13Train = pickle.load(fp)
                    self.y13TrainLabel = pickle.load(fp)
                    self.y23TrainLabel = pickle.load(fp)
                    self.y33TrainLabel = pickle.load(fp)
                    self.y43TrainLabel = pickle.load(fp)
                    self.y53TrainLabel = pickle.load(fp)
                    self.y13Train = pickle.load(fp)
                    self.y23Train = pickle.load(fp)
                    self.y33Train = pickle.load(fp)
                    self.y43Train = pickle.load(fp)
                    self.y53Train = pickle.load(fp)
                
                with open(os.path.join(featuresPath,picklesPath,train4),'rb') as fp:
                    self.x14Train = pickle.load(fp)
                    self.y14TrainLabel = pickle.load(fp)
                    self.y24TrainLabel = pickle.load(fp)
                    self.y34TrainLabel = pickle.load(fp)
                    self.y44TrainLabel = pickle.load(fp)
                    self.y54TrainLabel = pickle.load(fp)
                    self.y14Train = pickle.load(fp)
                    self.y24Train = pickle.load(fp)
                    self.y34Train = pickle.load(fp)
                    self.y44Train = pickle.load(fp)
                    self.y54Train = pickle.load(fp)
                """
                
                with open(os.path.join(featuresPath,picklesPath,train5),'rb') as fp:
                    self.x14Train = pickle.load(fp)
                    self.y14TrainLabel = pickle.load(fp)
                    self.y24TrainLabel = pickle.load(fp)
                    self.y34TrainLabel = pickle.load(fp)
                    self.y44TrainLabel = pickle.load(fp)
                    self.y54TrainLabel = pickle.load(fp)
                    self.y14Train = pickle.load(fp)
                    self.y24Train = pickle.load(fp)
                    self.y34Train = pickle.load(fp)
                    self.y44Train = pickle.load(fp)
                    self.y54Train = pickle.load(fp)
                with open(os.path.join(featuresPath,picklesPath,train6),'rb') as fp:
                    self.x15Train = pickle.load(fp)
                    self.y15TrainLabel = pickle.load(fp)
                    self.y25TrainLabel = pickle.load(fp)
                    self.y35TrainLabel = pickle.load(fp)
                    self.y45TrainLabel = pickle.load(fp)
                    self.y55TrainLabel = pickle.load(fp)
                    self.y15Train = pickle.load(fp)
                    self.y25Train = pickle.load(fp)
                    self.y35Train = pickle.load(fp)
                    self.y45Train = pickle.load(fp)
                    self.y55Train = pickle.load(fp)
                """
                with open(os.path.join(featuresPath,picklesPath,t/est1),'rb') as fp:
                    self.x1Test = pickle.load(fp)
                    self.y11TestLabel = pickle.load(fp)
                    self.y21TestLabel = pickle.load(fp)
                    self.y31TestLabel = pickle.load(fp)
                    self.y41TestLabel = pickle.load(fp)
                    self.y51TestLabel = pickle.load(fp)
                    self.y11Test = pickle.load(fp)
                    self.y21Test = pickle.load(fp)
                    self.y31Test = pickle.load(fp)
                    self.y41Test = pickle.load(fp)
                    self.y51Test = pickle.load(fp)
                
                with open(os.path.join(featuresPath,picklesPath,test2),'rb') as fp:
                    self.x2Test = pickle.load(fp)
                    self.y12TestLabel = pickle.load(fp)
                    self.y22TestLabel = pickle.load(fp)
                    self.y32TestLabel = pickle.load(fp)
                    self.y42TestLabel = pickle.load(fp)
                    self.y52TestLabel = pickle.load(fp)
                    self.y12Test = pickle.load(fp)
                    self.y22Test = pickle.load(fp)
                    self.y32Test = pickle.load(fp)
                    self.y42Test = pickle.load(fp)
                    self.y52Test = pickle.load(fp)
                """
                self.xTest = self.x1Test
                self.y1TestLabel = self.y21TestLabel
                self.y2TestLabel = self.y41TestLabel
                self.y3TestLabel = self.y51TestLabel
                self.y1Test = self.y21Test
                self.y2Test = self.y41Test
                self.y3Test = self.y51Test
                
                """
                #pdb.set_trace()
                self.xTest = np.concatenate((self.x1Test,self.x2Test),0)
                self.y1TestLabel = np.concatenate((self.y21TestLabel,self.y22TestLabel),0)
                self.y2TestLabel = np.concatenate((self.y41TestLabel,self.y42TestLabel),0)
                self.y3TestLabel = np.concatenate((self.y51TestLabel,self.y52TestLabel),0)
                self.y1Test = np.concatenate((self.y21Test,self.y22Test),0)
                self.y2Test = np.concatenate((self.y41Test,self.y42Test),0)
                self.y3Test = np.concatenate((self.y51Test,self.y52Test),0)
                
                """
                self.xTest = self.x1Test
                self.y1TestLabel = self.y11TestLabel
                self.y2TestLabel = self.y21TestLabel
                self.y3TestLabel = self.y31TestLabel
                self.y4TestLabel = self.y41TestLabel
                self.y5TestLabel = self.y51TestLabel
                self.y1Test = self.y11Test
                self.y2Test = self.y21Test
                self.y3Test = self.y31Test
                self.y4Test = self.y41Test
                self.y5Test = self.y51Test
                """ 

        
        
            elif CellMode == 234:
                picklesPath = 'b2b3b4pickles' 
                train1 = 'b2b3b4_train1{}u'.format(classMode)
                train2 = 'b2b3b4_train2{}u'.format(classMode)
                train3 = 'b2b3b4_train3{}u'.format(classMode)
                train4 = 'b2b3b4_train4{}u'.format(classMode)
                train5 = 'b2b3b4_train5{}u'.format(classMode)
                train6 = 'b2b3b4_train6{}u'.format(classMode)
                train7 = 'b2b3b4_train7{}u'.format(classMode)
                train8 = 'b2b3b4_train8{}u'.format(classMode)
                test1 = 'b2b3b4_test1{}u'.format(classMode)
                test2 = 'b2b3b4_test2{}u'.format(classMode)
                test12345 = 'b1b2b3b4b5_{}u'.format(classMode)
                
                with open(os.path.join(featuresPath,picklesPath,train1),'rb') as fp:
                    self.x11Train = pickle.load(fp)
                    self.y11TrainLabel = pickle.load(fp)
                    self.y21TrainLabel = pickle.load(fp)
                    self.y31TrainLabel = pickle.load(fp)
                    self.y11Train = pickle.load(fp)
                    self.y21Train = pickle.load(fp)
                    self.y31Train = pickle.load(fp)
                with open(os.path.join(featuresPath,picklesPath,train2),'rb') as fp:
                    self.x12Train = pickle.load(fp)
                    self.y12TrainLabel = pickle.load(fp)
                    self.y22TrainLabel = pickle.load(fp)
                    self.y32TrainLabel = pickle.load(fp)
                    self.y12Train = pickle.load(fp)
                    self.y22Train = pickle.load(fp)
                    self.y32Train = pickle.load(fp)
                with open(os.path.join(featuresPath,picklesPath,train3),'rb') as fp:
                    self.x13Train = pickle.load(fp)
                    self.y13TrainLabel = pickle.load(fp)
                    self.y23TrainLabel = pickle.load(fp)
                    self.y33TrainLabel = pickle.load(fp)
                    self.y13Train = pickle.load(fp)
                    self.y23Train = pickle.load(fp)
                    self.y33Train = pickle.load(fp)
                
                with open(os.path.join(featuresPath,picklesPath,train4),'rb') as fp:
                    self.x14Train = pickle.load(fp)
                    self.y14TrainLabel = pickle.load(fp)
                    self.y24TrainLabel = pickle.load(fp)
                    self.y34TrainLabel = pickle.load(fp)
                    self.y14Train = pickle.load(fp)
                    self.y24Train = pickle.load(fp)
                    self.y34Train = pickle.load(fp)
                """ 
                with open(os.path.join(featuresPath,picklesPath,train5),'rb') as fp:
                    self.x15Train = pickle.load(fp)
                    self.y15TrainLabel = pickle.load(fp)
                    self.y25TrainLabel = pickle.load(fp)
                    self.y35TrainLabel = pickle.load(fp)
                    self.y15Train = pickle.load(fp)
                    self.y25Train = pickle.load(fp)
                    self.y35Train = pickle.load(fp)
                with open(os.path.join(featuresPath,picklesPath,train6),'rb') as fp:
                    self.x16Train = pickle.load(fp)
                    self.y16TrainLabel = pickle.load(fp)
                    self.y26TrainLabel = pickle.load(fp)
                    self.y36TrainLabel = pickle.load(fp)
                    self.y16Train = pickle.load(fp)
                    self.y26Train = pickle.load(fp)
                    self.y36Train = pickle.load(fp)
                with open(os.path.join(featuresPath,picklesPath,train7),'rb') as fp:
                    self.x17Train = pickle.load(fp)
                    self.y17TrainLabel = pickle.load(fp)
                    self.y27TrainLabel = pickle.load(fp)
                    self.y37TrainLabel = pickle.load(fp)
                    self.y17Train = pickle.load(fp)
                    self.y27Train = pickle.load(fp)
                    self.y37Train = pickle.load(fp)
                with open(os.path.join(featuresPath,picklesPath,train8),'rb') as fp:
                    self.x18Train = pickle.load(fp)
                    self.y18TrainLabel = pickle.load(fp)
                    self.y28TrainLabel = pickle.load(fp)
                    self.y38TrainLabel = pickle.load(fp)
                    self.y18Train = pickle.load(fp)
                    self.y28Train = pickle.load(fp)
                    self.y38Train = pickle.load(fp)
                
                with open(os.path.join(featuresPath,picklesPath,test12345),'rb') as fp:
                    self.xTest = pickle.load(fp)
                    self.y_1TestLabel = pickle.load(fp)
                    self.y_2TestLabel = pickle.load(fp)
                    self.y_3TestLabel = pickle.load(fp)
                    self.y_1Test = pickle.load(fp)
                    self.y_2Test = pickle.load(fp)
                    self.y_3Test = pickle.load(fp)
                
                #pdb.set_trace()
                self.xTest = self.xTest[testRandInd[:3000]]
                self.y1TestLabel = self.y1TestLabel[testRandInd[:3000]]
                self.y2TestLabel = self.y2TestLabel[testRandInd[:3000]]
                self.y3TestLabel = self.y3TestLabel[testRandInd[:3000]]
                self.y1Test = self.y1Test[testRandInd[:3000]]
                self.y2Test = self.y2Test[testRandInd[:3000]]
                self.y3Test = self.y3Test[testRandInd[:3000]]
                
                self.x_2Test = self.xTest[testRandInd[:10000]]
                self.y_12TestLabel = self.y_1TestLabel[testRandInd[:10000]]
                self.y_22TestLabel = self.y_2TestLabel[testRandInd[:10000]]
                self.y_32TestLabel = self.y_3TestLabel[testRandInd[:10000]]
                self.y_12Test = self.y_1Test[testRandInd[:10000]]
                self.y_22Test = self.y_2Test[testRandInd[:10000]]
                self.y_32Test = self.y_3Test[testRandInd[:10000]]
                """ 
                with open(os.path.join(featuresPath,picklesPath,test1),'rb') as fp:
                    self.x11Test = pickle.load(fp)
                    self.y11TestLabel = pickle.load(fp)
                    self.y21TestLabel = pickle.load(fp)
                    self.y31TestLabel = pickle.load(fp)
                    self.y11Test = pickle.load(fp)
                    self.y21Test = pickle.load(fp)
                    self.y31Test = pickle.load(fp)
                with open(os.path.join(featuresPath,picklesPath,test2),'rb') as fp:
                    self.x12Test = pickle.load(fp)
                    self.y12TestLabel = pickle.load(fp)
                    self.y22TestLabel = pickle.load(fp)
                    self.y32TestLabel = pickle.load(fp)
                    self.y12Test = pickle.load(fp)
                    self.y22Test = pickle.load(fp)
                    self.y32Test = pickle.load(fp)
                
                #pdb.set_trace()
                
                self.xTest = np.concatenate((self.x11Test,self.x12Test),0)
                self.y1Test = np.concatenate((self.y11Test,self.y12Test),0)
                self.y2Test = np.concatenate((self.y21Test,self.y22Test),0)
                self.y3Test = np.concatenate((self.y31Test,self.y32Test),0)
                self.y1TestLabel = np.concatenate((self.y11TestLabel,self.y12TestLabel),0)
                self.y2TestLabel = np.concatenate((self.y21TestLabel,self.y22TestLabel),0)
                self.y3TestLabel = np.concatenate((self.y31TestLabel,self.y32TestLabel),0)
                 
                """
                self.xTest = self.x11Test[:10000]
                self.y1TestLabel = self.y11TestLabel[:10000]
                self.y2TestLabel = self.y21TestLabel[:10000]
                self.y3TestLabel = self.y31TestLabel[:10000]
                self.y1Test = self.y11Test[:10000]
                self.y2Test = self.y21Test[:10000]
                self.y3Test = self.y31Test[:10000]
                """
            elif CellMode == 1234:
                
                b1b2picklePath = 'b1b2_{}u'.format(classMode)
                b2b3picklePath = 'b2b3_{}u'.format(classMode)
                b3b4picklePath = 'b3b4_{}u'.format(classMode)
                
                b1b2b3b4dataPath = 'b1b2b3b4_u'
                b1b2b3b4picklePath = 'b1b2b3b4_{}u'.format(classMode)
                
                # 各X,YpickledataがあるPath 
                b1b2fullPath = os.path.join(featuresPath,b1b2b3b4dataPath,b1b2picklePath)
                b2b3fullPath = os.path.join(featuresPath,b1b2b3b4dataPath,b2b3picklePath)
                b3b4fullPath = os.path.join(featuresPath,b1b2b3b4dataPath,b3b4picklePath)
                b1b2b3b4fullPath = os.path.join(featuresPath,b1b2b3b4dataPath,b1b2b3b4picklePath)
                
                # バッチの初期化(b1b2)
                self.batchRandIndb1b2 = np.random.permutation(self.nData)
                # バッチの初期化(b2b3)
                self.batchRandIndb2b3 = np.random.permutation(self.nData)
                # バッチの初期化(b3b4)
                self.batchRandIndb3b4 = np.random.permutation(self.nData)
                

                with open(b1b2fullPath,'rb') as fp:
                    self.b1b2xTrain = pickle.load(fp)
                    self.y1TrainLabel_12 = pickle.load(fp)
                    self.y2TrainLabel_12 = pickle.load(fp)
                    self.y3TrainLabel_12 = pickle.load(fp)
                    self.y4TrainLabel_12 = pickle.load(fp)
                    self.y1Train_12 = pickle.load(fp)
                    self.y2Train_12 = pickle.load(fp)
                    self.y3Train_12 = pickle.load(fp)
                    self.y4Train_12 = pickle.load(fp)
                
                with open(b2b3fullPath,'rb') as fp:
                    self.b2b3xTrain = pickle.load(fp)
                    self.y1TrainLabel_23 = pickle.load(fp)
                    self.y2TrainLabel_23 = pickle.load(fp)
                    self.y3TrainLabel_23 = pickle.load(fp)
                    self.y4TrainLabel_23 = pickle.load(fp)
                    self.y1Train_23 = pickle.load(fp)
                    self.y2Train_23 = pickle.load(fp)
                    self.y3Train_23 = pickle.load(fp)
                    self.y4Train_23 = pickle.load(fp)
                
                with open(b3b4fullPath,'rb') as fp:
                    self.b3b4xTrain = pickle.load(fp)
                    self.y1TrainLabel_34 = pickle.load(fp)
                    self.y2TrainLabel_34 = pickle.load(fp)
                    self.y3TrainLabel_34 = pickle.load(fp)
                    self.y4TrainLabel_34 = pickle.load(fp)
                    self.y1Train_34 = pickle.load(fp)
                    self.y2Train_34 = pickle.load(fp)
                    self.y3Train_34 = pickle.load(fp)
                    self.y4Train_34 = pickle.load(fp)
                
                with open(b1b2b3b4fullPath,'rb') as fp:
                    self.b1b2b3b4xTest = pickle.load(fp)
                    self.y1TestLabel = pickle.load(fp)
                    self.y2TestLabel = pickle.load(fp)
                    self.y3TestLabel = pickle.load(fp)
                    self.y4TestLabel = pickle.load(fp)
                    self.y1Test = pickle.load(fp)
                    self.y2Test = pickle.load(fp)
                    self.y3Test = pickle.load(fp)
                    self.y4Test = pickle.load(fp)
        
        elif datapickleMode == 2:
            #　XとYのpickleファイル読み込み
            #　XとYのpickleファイル読み込み
            
            if CellMode == 12 or CellMode == 23 or CellMode == 34 or CellMode == 45 or CellMode == 56 or CellMode == 67 or CellMode == 78:
                with open(xyfullPath,'rb') as fp:
                    self.X = pickle.load(fp)
                    Y1 = pickle.load(fp)
                    Y2 = pickle.load(fp)
                
                    labelY1 = pickle.load(fp)
                    labelY2 = pickle.load(fp)
                    
                """
                # XとYの正規化(Yが小さすぎるため,そのもののbを可視化したいときはYの正規化だけはずす）
                self.minY = np.min(self.Y)
                self.maxY = np.max(self.Y)
                self.Y = (self.Y - self.minY)/(self.maxY-self.minY)
                
                self.minX = np.min(self.X)
                self.maxX = np.max(self.X)
                self.X = (self.X - self.minX)/(self.maxX-self.minX)
                self.X = (self.X-np.mean(self.X,axis=0))*100
                self.Y = self.Y * 100 - 1
                """
                self.nTrain = np.floor(self.nData * trainRatio).astype(int)
                self.nTest = self.nData - self.nTrain
                
                # ミニバッチの初期化
                self.batchCnt = 0
                self.batchRandInd = np.random.permutation(self.nTrain)
            
                # 学習データとテストデータ数
                self.nTrain = np.floor(self.nData * trainRatio).astype(int)
                self.nTest = self.nData - self.nTrain
                
                # ランダムにインデックスをシャッフル
                self.randInd = np.random.permutation(self.nData)

                
                # 学習データ
                self.xTrain = self.X[self.randInd[0:self.nTrain]]
                  
                self.y1Train,self.y1TrainLabel = Y1[self.randInd[0:self.nTrain]],labelY1[[self.randInd[0:self.nTrain]]] 
                self.y2Train,self.y2TrainLabel = Y2[self.randInd[0:self.nTrain]],labelY2[[self.randInd[0:self.nTrain]]]
                
                # 評価データ
                self.xTest = self.X[self.randInd[self.nTrain:]] 
                
               
                self.y1Test,self.y1TestLabel = Y1[self.randInd[self.nTrain:]],labelY1[[self.randInd[self.nTrain:]]] 
                self.y2Test,self.y2TestLabel = Y2[self.randInd[self.nTrain:]],labelY2[[self.randInd[self.nTrain:]]] 
                
                
                # 学習とテストデータの保存
                with open(traintestfullPath,'wb') as fp:
                    pickle.dump(self.xTrain,fp)
                    pickle.dump(self.y1Train,fp)
                    pickle.dump(self.y2Train,fp)
                    pickle.dump(self.y1TrainLabel,fp)
                    pickle.dump(self.y2TrainLabel,fp)
                    
                    pickle.dump(self.xTest,fp)
                    pickle.dump(self.y1Test,fp)
                    pickle.dump(self.y2Test,fp)
                    pickle.dump(self.y1TestLabel,fp)
                    pickle.dump(self.y2TestLabel,fp)
            
            if CellMode == 123:
                with open(xyfullPath,'rb') as fp:
                    self.X = pickle.load(fp)
                    Y1 = pickle.load(fp)
                    Y2 = pickle.load(fp)
                    Y3 = pickle.load(fp)
                    
                    labelY1 = pickle.load(fp)
                    labelY2 = pickle.load(fp)
                    labelY3 = pickle.load(fp)

                self.nTrain = np.floor(self.nData,trainRatio).astype(int)
                self.nTest = self.nData - self.nTrain
                
                self.batchCnt = 0
                self.batchRandInd = np.random.permutation(self.nTrain)

                self.xTrain = self.X[self.randInd[0:self.nTrain]]
                self.y1Train,self.y1TrainLabel = Y1[self.randInd[0:self.nTrain]],labelY1[self.randInd[0:self.nTrain]]
                self.y2Train,self.y2TrainLabel = Y2[self.randInd[0:self.nTrain]],labelY2[self.randInd[0:self.nTrain]]
                self.y3Train,self.y3TrainLabel= Y3[self.randInd[0:self.nTrain]],labelY3[self.randInd[0:self.nTrain]]

                self.xTest = self.X[self.randInd[self.nTrain:]]
                self.y1Test = Y1[self.randInd[self.nTrain:]],labelY1[self.randInd[self.nTrain:]]
                self.y2Test = Y2[self.randInd[self.nTrain:]],labelY1[self.randInd[self.nTrain:]]
                self.y3Test = Y3[self.randInd[self.nTrain:]],labelY1[self.randInd[self.nTrain:]]


                with open(traintestfullPath,'wb') as fp:
                    pickle.dump(self.xTrain,fp)
                    pickle.dump(self.y1Train,fp)
                    pickle.dump(self.y2Train,fp)
                    pickle.dump(self.y3Train,fp)
                    pickle.dump(self.xTest,fp)
                    pickle.dump(self.y1Test,fp)
                    pickle.dump(self.y2Test,fp)
                    pickle.dump(self.y3Test,fp)
        
        elif datapickleMode == 3:
            flag = False
            #tPath = 'test2'
            # データの読み込み
            for fID in np.arange(self.nData):
                
                if isWindows:
                    file = files[fID].split('\\')[1]
                else:
                    file = files[fID].split('/')[3]
                    
                fullPath = os.path.join(featuresPath,dataPath,self.tPath,file)
                #pdb.set_trace()
                with open(fullPath,'rb') as fp:
                    tmpY = pickle.load(fp)
                    tmpX = pickle.load(fp)
                     
                    # 5Crle
                    tmpY1 = tmpY[1]
                    tmpY2 = tmpY[2]
                    tmpY3 = tmpY[3]
                    tmpY4 = tmpY[4]
                    tmpY5 = tmpY[5]
                  
                    if isClass:
                        Y1 = tmpY[1][np.newaxis]
                        Y2 = tmpY[2][np.newaxis]
                        Y3 = tmpY[3][np.newaxis]
                        Y4 = tmpY[4][np.newaxis]
                        Y5 = tmpY[5][np.newaxis]
                        
                        sBn = 0.0125
                        eBn = 0.017
                        sB = 0.012
                        eB = 0.0165
            
                        iBn = round((eBn-sBn)/nClass,6)
                        Bsn = np.arange(sBn,eBn,iBn)
                        
                        iB = round((eB-sB)/nClass,6)
                        Bs = np.arange(sB,eB,iB)
                        
                        oneHot1 = np.zeros(len(Bsn))#0.001,0.0015,...0.00165
                        oneHot2 = np.zeros(len(Bsn))#0.001,0.0015,...0.00165 
                        oneHot3 = np.zeros(len(Bs))#0.001,0.0015,...0.00165
                        oneHot4 = np.zeros(len(Bs))#0.001,0.0015,...0.00165
                        oneHot5 = np.zeros(len(Bs))#0.001,0.0015,...0.00165
                        #pdb.set_trace()
                        #b1
                        ind = 0
                        for threB in Bsn:
                            if (tmpY1 >= threB) & (tmpY1 < threB + iBn):
                                oneHot1[ind] = 1            
                            ind += 1
                        tmpY1 = oneHot1 
                        tmpY1 = tmpY1[np.newaxis]
                        
                        #b2
                        ind = 0
                        for threB in Bsn:
                            if (tmpY2 >= threB) & (tmpY2 < threB + iBn):
                                oneHot2[ind] = 1            
                            ind += 1
                        tmpY2 = oneHot2 
                        tmpY2 = tmpY2[np.newaxis]
                        
                        #b3
                        ind = 0
                        for threB in Bs:
                            if (tmpY3 >= threB) & (tmpY3 < threB + iB):
                                oneHot3[ind] = 1            
                            ind += 1
                        tmpY3 = oneHot3 
                        tmpY3 = tmpY3[np.newaxis]
                        
                        # b4
                        ind = 0
                        for threB in Bs:
                            if (tmpY4 >= threB) & (tmpY4 < threB + iB):
                                oneHot4[ind] = 1            
                            ind += 1
                        tmpY4 = oneHot4 
                        tmpY4 = tmpY4[np.newaxis]
                        
                        # b5
                        ind = 0
                        for threB in Bs:
                            if (tmpY5 >= threB) & (tmpY5 < threB + iB):
                                oneHot5[ind] = 1            
                            ind += 1
                        tmpY5 = oneHot5 
                        tmpY5 = tmpY5[np.newaxis]
                        
                    if not flag:
                        self.X = tmpX[np.newaxis].astype(np.float32)
                        self.Y1 = tmpY1.astype(np.float32)
                        self.Y2 = tmpY2.astype(np.float32)
                        self.Y3 = tmpY3.astype(np.float32)
                        self.Y4 = tmpY4.astype(np.float32)
                        self.Y5 = tmpY5.astype(np.float32)
                        self.trueY1 = Y1.astype(np.float32)
                        self.trueY2 = Y2.astype(np.float32)
                        self.trueY3 = Y3.astype(np.float32)
                        self.trueY4 = Y4.astype(np.float32)
                        self.trueY5 = Y5.astype(np.float32)
                        flag = True
                    else:
                        self.X = np.concatenate((self.X,tmpX[np.newaxis]),axis=0)
                        self.Y1 = np.concatenate((self.Y1, tmpY1.astype(np.float32)),axis=0)
                        self.Y2 = np.concatenate((self.Y2, tmpY2.astype(np.float32)),axis=0)
                        self.Y3 = np.concatenate((self.Y3, tmpY3.astype(np.float32)),axis=0)
                        self.Y4 = np.concatenate((self.Y4, tmpY4.astype(np.float32)),axis=0)
                        self.Y5 = np.concatenate((self.Y5, tmpY5.astype(np.float32)),axis=0)
                        self.trueY1 = np.concatenate((self.trueY1,Y1.astype(np.float32)),axis=0)           
                        self.trueY2 = np.concatenate((self.trueY2,Y2.astype(np.float32)),axis=0)
                        self.trueY3 = np.concatenate((self.trueY3,Y3.astype(np.float32)),axis=0)
                        self.trueY4 = np.concatenate((self.trueY4,Y4.astype(np.float32)),axis=0)
                        self.trueY5 = np.concatenate((self.trueY5,Y5.astype(np.float32)),axis=0)
                        print(fID) 
                    """
                    tmpY1 = tmpY[1]
                    tmpY2 = tmpY[2]
                    tmpY3 = tmpY[3]
                  
                    if isClass:
                        Y1 = tmpY[1][np.newaxis]
                        Y2 = tmpY[2][np.newaxis]
                        Y3 = tmpY[3][np.newaxis]
                        
                        sB = 0.012
                        eB = 0.018
            
                        iB = round((eB-sB)/nClass,6)
                        Bs = np.arange(sB,eB,iB)
                        
                        oneHot1 = np.zeros(len(Bs))#0.001,0.0015,...0.00165
                        oneHot2 = np.zeros(len(Bs))#0.001,0.0015,...0.00165 
                        oneHot3 = np.zeros(len(Bs))#0.001,0.0015,...0.00165
                       
                        #b1
                        ind = 0
                        for threB in Bs:
                            if (tmpY1 >= threB) & (tmpY1 < threB + iB):
                                oneHot1[ind] = 1            
                            ind += 1
                        tmpY1 = oneHot1 
                        tmpY1 = tmpY1[np.newaxis]
                        
                        #b2
                        ind = 0
                        for threB in Bs:
                            if (tmpY2 >= threB) & (tmpY2 < threB + iB):
                                oneHot2[ind] = 1            
                            ind += 1
                        tmpY2 = oneHot2 
                        tmpY2 = tmpY2[np.newaxis]
                        
                        #b3
                        ind = 0
                        for threB in Bs:
                            if (tmpY3 >= threB) & (tmpY3 < threB + iB):
                                oneHot3[ind] = 1            
                            ind += 1
                        tmpY3 = oneHot3 
                        tmpY3 = tmpY3[np.newaxis]
                        
                        
                    if not flag:
                        self.X = tmpX[np.newaxis].astype(np.float32)
                        self.Y1 = tmpY1.astype(np.float32)
                        self.Y2 = tmpY2.astype(np.float32)
                        self.Y3 = tmpY3.astype(np.float32)
                        self.trueY1 = Y1.astype(np.float32)
                        self.trueY2 = Y2.astype(np.float32)
                        self.trueY3 = Y3.astype(np.float32)
                        flag = True
                    else:
                        self.X = np.concatenate((self.X,tmpX[np.newaxis]),axis=0)
                        self.Y1 = np.concatenate((self.Y1, tmpY1.astype(np.float32)),axis=0)
                        self.Y2 = np.concatenate((self.Y2, tmpY2.astype(np.float32)),axis=0)
                        self.Y3 = np.concatenate((self.Y3, tmpY3.astype(np.float32)),axis=0)
                        self.trueY1 = np.concatenate((self.trueY1,Y1.astype(np.float32)),axis=0)           
                        self.trueY2 = np.concatenate((self.trueY2,Y2.astype(np.float32)),axis=0)
                        self.trueY3 = np.concatenate((self.trueY3,Y3.astype(np.float32)),axis=0)
                        print(fID) 
            with open(xyfullPath,'wb') as fp:
                pickle.dump(self.X,fp)
                pickle.dump(self.Y1,fp)
                pickle.dump(self.Y2,fp)
                pickle.dump(self.Y3,fp)
                pickle.dump(self.trueY1,fp)
                pickle.dump(self.trueY2,fp)
                pickle.dump(self.trueY3,fp)

            """
            if CellMode == 1234 or CellMode == 12 or CellMode == 23 or CellMode == 34 or CellMode == 45 or CellMode == 56 or CellMode == 67 or CellMode == 78:
                with open(xyfullPath,'wb') as fp:
                    pickle.dump(self.X,fp)
                    pickle.dump(self.Y1,fp)
                    pickle.dump(self.Y2,fp)
                    pickle.dump(self.Y3,fp)
                    pickle.dump(self.Y4,fp)
                                
                    pickle.dump(self.trueY1,fp)
                    pickle.dump(self.trueY2,fp)
                    pickle.dump(self.trueY3,fp)
                    pickle.dump(self.trueY4,fp)
            with open(xyfullPath,'wb') as fp:
                pickle.dump(self.X,fp)
                pickle.dump(self.Y1,fp)
                pickle.dump(self.Y2,fp)
                pickle.dump(self.Y3,fp)
                pickle.dump(self.Y4,fp)
                pickle.dump(self.Y5,fp)
                                
                pickle.dump(self.trueY1,fp)
                pickle.dump(self.trueY2,fp)
                pickle.dump(self.trueY3,fp)
                pickle.dump(self.trueY4,fp)
                pickle.dump(self.trueY5,fp)
        
    #------------------------------------
    
    #------------------------------------
    # ミニバッチの取り出し
    #def nextBatch(self,batchSize12,batchSize23,batchSize34,isTensorflow=True):
    def nextBatch(self,batchSize,isTensorflow=True):
        
        if self.CellMode == 23456:
            sInd = batchSize * self.batchCnt
            eInd = sInd + batchSize
            """
            x1,x2= self.x11Train[self.batchRandInd[sInd:eInd]],self.x13Train[self.batchRandInd[sInd:eInd]]
            y11,y12 = self.y21Train[self.batchRandInd[sInd:eInd]],self.y23Train[self.batchRandInd[sInd:eInd]]
            y21,y22 = self.y41Train[self.batchRandInd[sInd:eInd]],self.y43Train[self.batchRandInd[sInd:eInd]]
            y31,y32 = self.y51Train[self.batchRandInd[sInd:eInd]],self.y53Train[self.batchRandInd[sInd:eInd]]
            yl11,yl12 = self.y21TrainLabel[self.batchRandInd[sInd:eInd]],self.y23TrainLabel[self.batchRandInd[sInd:eInd]]
            yl21,yl22 = self.y41TrainLabel[self.batchRandInd[sInd:eInd]],self.y43TrainLabel[self.batchRandInd[sInd:eInd]]
            yl31,yl32 = self.y51TrainLabel[self.batchRandInd[sInd:eInd]],self.y53TrainLabel[self.batchRandInd[sInd:eInd]]
            
            batchX = np.concatenate((x1,x2),0)
            batchY1,batchY1Label = np.concatenate((y11,y12),0),np.concatenate((yl11,yl12),0)
            batchY2,batchY2Label = np.concatenate((y21,y22),0),np.concatenate((yl21,yl22),0)
            batchY3,batchY3Label = np.concatenate((y31,y32),0),np.concatenate((yl31,yl32),0)
            
            """ 
            x1,x2,x3= self.x11Train[self.batchRandInd[sInd:eInd]],self.x13Train[self.batchRandInd[sInd:eInd]],self.x14Train[self.batchRandInd[sInd:eInd]]
            y11,y12,y13 = self.y21Train[self.batchRandInd[sInd:eInd]],self.y23Train[self.batchRandInd[sInd:eInd]],self.y24Train[self.batchRandInd[sInd:eInd]]
            y21,y22,y23 = self.y41Train[self.batchRandInd[sInd:eInd]],self.y43Train[self.batchRandInd[sInd:eInd]],self.y44Train[self.batchRandInd[sInd:eInd]]
            y31,y32,y33 = self.y51Train[self.batchRandInd[sInd:eInd]],self.y53Train[self.batchRandInd[sInd:eInd]],self.y54Train[self.batchRandInd[sInd:eInd]]
            yl11,yl12,yl13 = self.y21TrainLabel[self.batchRandInd[sInd:eInd]],self.y23TrainLabel[self.batchRandInd[sInd:eInd]],self.y24TrainLabel[self.batchRandInd[sInd:eInd]]
            yl21,yl22,yl23 = self.y41TrainLabel[self.batchRandInd[sInd:eInd]],self.y43TrainLabel[self.batchRandInd[sInd:eInd]],self.y44TrainLabel[self.batchRandInd[sInd:eInd]]
            yl31,yl32,yl33 = self.y51TrainLabel[self.batchRandInd[sInd:eInd]],self.y53TrainLabel[self.batchRandInd[sInd:eInd]],self.y54TrainLabel[self.batchRandInd[sInd:eInd]]
            
            batchX = np.concatenate((x1,x2,x3),0)
            batchY1,batchY1Label = np.concatenate((y11,y12,y13),0),np.concatenate((yl11,yl12,yl13),0)
            batchY2,batchY2Label = np.concatenate((y21,y22,y23),0),np.concatenate((yl21,yl22,yl23),0)
            batchY3,batchY3Label = np.concatenate((y31,y32,y33),0),np.concatenate((yl31,yl32,yl33),0)
            
            """
            x1,x2= self.x11Train[self.batchRandInd[sInd:eInd]],self.x13Train[self.batchRandInd[sInd:eInd]]
            y11,y12 = self.y11Train[self.batchRandInd[sInd:eInd]],self.y13Train[self.batchRandInd[sInd:eInd]]
            y21,y22 = self.y21Train[self.batchRandInd[sInd:eInd]],self.y23Train[self.batchRandInd[sInd:eInd]]
            
            #y31,y32 = self.y31Train[self.batchRandInd[sInd:eInd]],self.y33Train[self.batchRandInd[sInd:eInd]]
            #y41,y42 = self.y41Train[self.batchRandInd[sInd:eInd]],self.y43Train[self.batchRandInd[sInd:eInd]]
            #y51,y52 = self.y51Train[self.batchRandInd[sInd:eInd]],self.y53Train[self.batchRandInd[sInd:eInd]]
            yl11,yl12 = self.y11TrainLabel[self.batchRandInd[sInd:eInd]],self.y13TrainLabel[self.batchRandInd[sInd:eInd]]
            yl21,yl22 = self.y21TrainLabel[self.batchRandInd[sInd:eInd]],self.y23TrainLabel[self.batchRandInd[sInd:eInd]]
            #yl31,yl32 = self.y31TrainLabel[self.batchRandInd[sInd:eInd]],self.y33TrainLabel[self.batchRandInd[sInd:eInd]]
            #yl41,yl42 = self.y41TrainLabel[self.batchRandInd[sInd:eInd]],self.y43TrainLabel[self.batchRandInd[sInd:eInd]]
            #yl51,yl52 = self.y51TrainLabel[self.batchRandInd[sInd:eInd]],self.y53TrainLabel[self.batchRandInd[sInd:eInd]]
            
            batchX = np.concatenate((x1,x2),0)
            batchY1,batchY1Label = np.concatenate((y11,y12),0),np.concatenate((yl11,yl12),0)
            batchY2,batchY2Label = np.concatenate((y21,y22),0),np.concatenate((yl21,yl22),0)
            batchY3,batchY3Label = np.concatenate((y31,y32),0),np.concatenate((yl31,yl32),0)
            #batchY4,batchY4Label = np.concatenate((y41,y42),0),np.concatenate((yl41,yl42),0)
            #batchY5,batchY5Label = np.concatenate((y51,y52),0),np.concatenate((yl51,yl52),0)
            """
            
            if eInd+batchSize > self.nTrain:
                self.batchCnt = 0
            else:
                self.batchCnt += 1
            #pdb.set_trace() 
            #return batchX,batchY1,batchY1Label,batchY2,batchY2Label,batchY3,batchY3Label,batchY4,batchY4Label,batchY5,batchY5Label
            return batchX,batchY1,batchY1Label,batchY2,batchY2Label,batchY3,batchY3Label
        
        if self.CellMode == 234:
            
            sInd = batchSize * self.batchCnt
            eInd = sInd + batchSize
            """
            x1,x2,x3,x4,x5,x6,x7,x8 = self.x11Train[self.batchRandInd[sInd:eInd]],self.x12Train[self.batchRandInd[sInd:eInd]],self.x13Train[self.batchRandInd[sInd:eInd]],self.x14Train[self.batchRandInd[sInd:eInd]],self.x15Train[self.batchRandInd[sInd:eInd]],self.x16Train[self.batchRandInd[sInd:eInd]],self.x17Train[self.batchRandInd[sInd:eInd]],self.x18Train[self.batchRandInd[sInd:eInd]]
            y11,y12,y13,y14,y15,y16,y17,y18 = self.y11Train[self.batchRandInd[sInd:eInd]],self.y12Train[self.batchRandInd[sInd:eInd]],self.y13Train[self.batchRandInd[sInd:eInd]],self.y14Train[self.batchRandInd[sInd:eInd]],self.y15Train[self.batchRandInd[sInd:eInd]],self.y16Train[self.batchRandInd[sInd:eInd]],self.y17Train[self.batchRandInd[sInd:eInd]],self.y18Train[self.batchRandInd[sInd:eInd]]
            yl11,yl12,yl13,yl14,yl15,yl16,yl17,yl18 = self.y11TrainLabel[self.batchRandInd[sInd:eInd]],self.y12TrainLabel[self.batchRandInd[sInd:eInd]],self.y13TrainLabel[self.batchRandInd[sInd:eInd]],self.y14TrainLabel[self.batchRandInd[sInd:eInd]],self.y15TrainLabel[self.batchRandInd[sInd:eInd]],self.y16TrainLabel[self.batchRandInd[sInd:eInd]],self.y17TrainLabel[self.batchRandInd[sInd:eInd]],self.y18TrainLabel[self.batchRandInd[sInd:eInd]]
            y21,y22,y23,y24,y25,y26,y27,y28 = self.y21Train[self.batchRandInd[sInd:eInd]],self.y22Train[self.batchRandInd[sInd:eInd]],self.y23Train[self.batchRandInd[sInd:eInd]],self.y24Train[self.batchRandInd[sInd:eInd]],self.y25Train[self.batchRandInd[sInd:eInd]],self.y26Train[self.batchRandInd[sInd:eInd]],self.y27Train[self.batchRandInd[sInd:eInd]],self.y28Train[self.batchRandInd[sInd:eInd]]
            yl21,yl22,yl23,yl24,yl25,yl26,yl27,yl28 = self.y21TrainLabel[self.batchRandInd[sInd:eInd]],self.y22TrainLabel[self.batchRandInd[sInd:eInd]],self.y23TrainLabel[self.batchRandInd[sInd:eInd]],self.y24TrainLabel[self.batchRandInd[sInd:eInd]],self.y25TrainLabel[self.batchRandInd[sInd:eInd]],self.y26TrainLabel[self.batchRandInd[sInd:eInd]],self.y27TrainLabel[self.batchRandInd[sInd:eInd]],self.y28TrainLabel[self.batchRandInd[sInd:eInd]]
            y31,y32,y33,y34,y35,y36,y37,y38 = self.y31Train[self.batchRandInd[sInd:eInd]],self.y32Train[self.batchRandInd[sInd:eInd]],self.y33Train[self.batchRandInd[sInd:eInd]],self.y34Train[self.batchRandInd[sInd:eInd]],self.y35Train[self.batchRandInd[sInd:eInd]],self.y36Train[self.batchRandInd[sInd:eInd]],self.y37Train[self.batchRandInd[sInd:eInd]],self.y38Train[self.batchRandInd[sInd:eInd]]
            yl31,yl32,yl33,yl34,yl35,yl36,yl37,yl38 = self.y31TrainLabel[self.batchRandInd[sInd:eInd]],self.y32TrainLabel[self.batchRandInd[sInd:eInd]],self.y33TrainLabel[self.batchRandInd[sInd:eInd]],self.y34TrainLabel[self.batchRandInd[sInd:eInd]],self.y35TrainLabel[self.batchRandInd[sInd:eInd]],self.y36TrainLabel[self.batchRandInd[sInd:eInd]],self.y37TrainLabel[self.batchRandInd[sInd:eInd]],self.y38TrainLabel[self.batchRandInd[sInd:eInd]]
            """
            
            x1,x2,x3,x4 = self.x11Train[self.batchRandInd[sInd:eInd]],self.x12Train[self.batchRandInd[sInd:eInd]],self.x13Train[self.batchRandInd[sInd:eInd]],self.x14Train[self.batchRandInd[sInd:eInd]]
            y11,y12,y13,y14 = self.y11Train[self.batchRandInd[sInd:eInd]],self.y12Train[self.batchRandInd[sInd:eInd]],self.y13Train[self.batchRandInd[sInd:eInd]],self.y14Train[self.batchRandInd[sInd:eInd]]
            yl11,yl12,yl13,yl14 = self.y11TrainLabel[self.batchRandInd[sInd:eInd]],self.y12TrainLabel[self.batchRandInd[sInd:eInd]],self.y13TrainLabel[self.batchRandInd[sInd:eInd]],self.y14TrainLabel[self.batchRandInd[sInd:eInd]]
            y21,y22,y23,y24 = self.y21Train[self.batchRandInd[sInd:eInd]],self.y22Train[self.batchRandInd[sInd:eInd]],self.y23Train[self.batchRandInd[sInd:eInd]],self.y24Train[self.batchRandInd[sInd:eInd]]
            yl21,yl22,yl23,yl24 = self.y21TrainLabel[self.batchRandInd[sInd:eInd]],self.y22TrainLabel[self.batchRandInd[sInd:eInd]],self.y23TrainLabel[self.batchRandInd[sInd:eInd]],self.y24TrainLabel[self.batchRandInd[sInd:eInd]]
            y31,y32,y33,y34 = self.y31Train[self.batchRandInd[sInd:eInd]],self.y32Train[self.batchRandInd[sInd:eInd]],self.y33Train[self.batchRandInd[sInd:eInd]],self.y34Train[self.batchRandInd[sInd:eInd]]
            yl31,yl32,yl33,yl34 = self.y31TrainLabel[self.batchRandInd[sInd:eInd]],self.y32TrainLabel[self.batchRandInd[sInd:eInd]],self.y33TrainLabel[self.batchRandInd[sInd:eInd]],self.y34TrainLabel[self.batchRandInd[sInd:eInd]]
            """            
            batchX = np.concatenate((x1,x2,x3,x4,x5,x6,x7,x8),0)
            batchY1,batchY1Label = np.concatenate((y11,y12,y13,y14,y15,y16,y17,y18),0),np.concatenate((yl11,yl12,yl13,yl14,yl15,yl16,yl17,yl18),0)
            batchY2,batchY2Label = np.concatenate((y21,y22,y23,y24,y25,y26,y27,y28),0),np.concatenate((yl21,yl22,yl23,yl24,yl25,yl26,yl27,yl28),0)
            batchY3,batchY3Label = np.concatenate((y31,y32,y33,y34,y35,y36,y37,y38),0),np.concatenate((yl31,yl32,yl33,yl34,yl35,yl36,yl37,yl38),0)
            """
            batchX = np.concatenate((x1,x2,x3,x4),0)
            batchY1,batchY1Label = np.concatenate((y11,y12,y13,y14),0),np.concatenate((yl11,yl12,yl13,yl14),0)
            batchY2,batchY2Label = np.concatenate((y21,y22,y23,y24),0),np.concatenate((yl21,yl22,yl23,yl24),0)
            batchY3,batchY3Label = np.concatenate((y31,y32,y33,y34),0),np.concatenate((yl31,yl32,yl33,yl34),0)
            
           
            if eInd+batchSize > self.nTrain:
                self.batchCnt = 0
            else:
                self.batchCnt += 1
            
            #:pdb.set_trace() 
            return batchX,batchY1,batchY1Label,batchY2,batchY2Label,batchY3,batchY3Label

        
        #------------------------------------
        #------------------------------------
        if self.CellMode == 1234:
            
            s12Ind = batchSize12 * self.batch12Cnt
            e12Ind = s12Ind + batchSize12
            # b2b3
            s23Ind = batchSize23 * self.batch23Cnt
            e23Ind = s23Ind + batchSize23
            # b3b4
            s34Ind = batchSize34 * self.batch34Cnt
            e34Ind = s34Ind + batchSize34
            
            # b1b2
            batchX1 = self.b1b2xTrain[self.batchRandIndb1b2[s12Ind:e12Ind]]
            batchY11,batchY11Label = self.y1Train_12[self.batchRandIndb1b2[s12Ind:e12Ind]],self.y1TrainLabel_12[self.batchRandIndb1b2[s12Ind:e12Ind]]
            batchY21,batchY21Label = self.y2Train_12[self.batchRandIndb1b2[s12Ind:e12Ind]],self.y2TrainLabel_12[self.batchRandIndb1b2[s12Ind:e12Ind]]
            batchY31,batchY31Label = self.y3Train_12[self.batchRandIndb1b2[s12Ind:e12Ind]],self.y3TrainLabel_12[self.batchRandIndb1b2[s12Ind:e12Ind]]
            batchY41,batchY41Label = self.y4Train_12[self.batchRandIndb1b2[s12Ind:e12Ind]],self.y4TrainLabel_12[self.batchRandIndb1b2[s12Ind:e12Ind]]
 
            #b2b3
            batchX2 = self.b2b3xTrain[self.batchRandIndb2b3[s23Ind:e23Ind]]
            batchY12,batchY12Label = self.y1Train_23[self.batchRandIndb2b3[s23Ind:e23Ind]],self.y1TrainLabel_23[self.batchRandIndb2b3[s23Ind:e23Ind]]
            batchY22,batchY22Label = self.y2Train_23[self.batchRandIndb2b3[s23Ind:e23Ind]],self.y2TrainLabel_23[self.batchRandIndb2b3[s23Ind:e23Ind]]
            batchY32,batchY32Label = self.y3Train_23[self.batchRandIndb2b3[s23Ind:e23Ind]],self.y3TrainLabel_23[self.batchRandIndb2b3[s23Ind:e23Ind]]
            batchY42,batchY42Label = self.y4Train_23[self.batchRandIndb2b3[s23Ind:e23Ind]],self.y4TrainLabel_23[self.batchRandIndb2b3[s23Ind:e23Ind]]
            
            #b3b4
            batchX3 = self.b3b4xTrain[self.batchRandIndb3b4[s34Ind:e34Ind]]
            batchY13,batchY13Label = self.y1Train_34[self.batchRandIndb3b4[s34Ind:e34Ind]],self.y1TrainLabel_34[self.batchRandIndb3b4[s34Ind:e34Ind]]
            batchY23,batchY23Label = self.y2Train_34[self.batchRandIndb3b4[s34Ind:e34Ind]],self.y2TrainLabel_34[self.batchRandIndb3b4[s34Ind:e34Ind]]
            batchY33,batchY33Label = self.y3Train_34[self.batchRandIndb3b4[s34Ind:e34Ind]],self.y3TrainLabel_34[self.batchRandIndb3b4[s34Ind:e34Ind]]
            batchY43,batchY43Label = self.y4Train_34[self.batchRandIndb3b4[s34Ind:e34Ind]],self.y4TrainLabel_34[self.batchRandIndb3b4[s34Ind:e34Ind]]

            if e12Ind+batchSize12 > self.nData:
                self.batch12Cnt = 0
            else:
                self.batch12Cnt += 1
            
            if e23Ind+batchSize23 > self.nData:
                self.batch23Cnt = 0
            else:
                self.batch23Cnt += 1
            
            if e34Ind+batchSize34 > self.nData:
                self.batch34Cnt = 0
            else:
                self.batch34Cnt += 1
            #pdb.set_trace() 
            return batchX1,batchY11,batchY11Label,batchY21,batchY21Label,batchY31,batchY31Label,batchY41,batchY41Label,batchX12,batchY12,batchY12Label,batchY22,batchY22Label,batchY32,batchY32Label,batchY42,batchY42Label,batchX13,batchY13,batchY13Label,batchY23,batchY23Label,batchY33,batchY33Label,batchY43,batchY43Label
    #------------------------------------

#########################################

############## MAIN #####################
if __name__ == "__main__":
    
    
    isWindows = False

    # Mode 設定
    # fileMode:使用するファイル指定 
    fileMode = int(sys.argv[1])
    # CellMode:使うセル指定
    CellMode = int(sys.argv[2])
    # 取り出すpickleの種類
    datapickleMode = int(sys.argv[3])
    # クラス数指定(sigma) 
    classMode = int(sys.argv[4])
    
    # 取り出す 
    if fileMode == 12: 
        fname = 'R6*'
        #fname = 'yV*.txt'

    if fileMode == 23456 or fileMode == 1234 or fileMode == 234 or fileMode == 23 or fileMode == 34 or fileMode == 45 or fileMode ==56 or fileMode == 67 or fileMode == 78:
        fname = 'log*'
    
    """ 
    if dataMode == 123:
        dataPath = 'b1b2b3_us'   
        picklePath = 'xydatab1b2b3_u.pkl'
        trainingpicklePath = 'traintestdatab1b2b3_u.pkl'
        fname = 'yV*'"""
            
    if CellMode == 12:
        bInd=[0,1]
    elif CellMode == 23:
        bInd=[1,2]
    elif CellMode == 34:
        bInd=[2,3]
    elif CellMode == 45:
        bInd=[3,4]
    elif CellMode == 56:
        bInd=[4,5]
    elif CellMode == 67:
        bInd=[5,6]
    elif CellMode == 78:
        bInd=[6,7] 
    elif CellMode == 234:
        bInd=[1,2,3]
    elif CellMode == 1234:
        bInd=[0,1,2,3]
    elif CellMode == 12345:
        bInd=[0,1,2,3,4]
    elif CellMode == 23456:
        bInd=[1,2,3,4,5]

    # クラス数
    if classMode == 10:
        nClass = 10 
    elif classMode == 12:
        nClass = 12
    elif classMode == 20:
        nClass = 20
    elif classMode == 50:
        nClass = 50
    """ 
    # pickleFilesPath
    dataPath = 'b{}b{}_u'.format(bInd[0]+1,bInd[1]+1)
    picklePath = 'b{}b{}_{}u'.format(bInd[0]+1,bInd[1]+1,classMode)
    trainingpicklePath = 'listtraintestdatab{}b{}_u.pkl'.format(bInd[0],bInd[1])
    """
    # b2b3b4用picklePath
    #dataPath = 'b{}b{}b{}_u'.format(bInd[0]+1,bInd[1]+1,bInd[2]+1)

    # b1b2b3b4用pickleFilesPath
    #dataPath = 'b{}b{}b{}b{}_u'.format(bInd[0]+1,bInd[1]+1,bInd[2]+1,bInd[3]+1)
    #picklePath = 'b{}b{}b{}b{}_{}u'.format(bInd[0]+1,bInd[1]+1,bInd[2]+1,bInd[3]+1,classMode)
    
    # b1b2b3b4b5用
    #dataPath = 'b{}b{}b{}b{}b{}_u'.format(bInd[0]+1,bInd[1]+1,bInd[2]+1,bInd[3]+1,bInd[3]+1)
    #picklePath = 'b{}b{}b{}b{}b{}_{}u'.format(bInd[0]+1,bInd[1]+1,bInd[2]+1,bInd[3]+1,bInd[5]+1,classMode)
    # dataPath
    logsPath = './logs'
    dataPath = 'b2b3b4b5b6FILES'
    tPath = 'train4'
    #filePath = os.path.join(logsPath,dataPath,fname) 
    filePath = os.path.join(dataPath,tPath,fname) 
    picklePath = 'b{}b{}b{}_{}{}u'.format(bInd[0]+1,bInd[1]+1,bInd[2]+1,tPath,classMode)
    #filePath = os.path.join('./features','train1',fname) 
    #picklePath = 'b{}b{}b{}b{}b{}_{}{}u'.format(bInd[0]+1,bInd[1]+1,bInd[2]+1,bInd[3]+1,bInd[4]+1,tPath,classMode)
    pdb.set_trace() 
    #Reading load log.txt
    if isWindows:
        files = glob.glob('logsb{}\\log_*.txt'.format(dataMode))
    else:
        files = glob.glob(filePath)
    for fID in np.arange(len(files)):
        print('reading',files[fID])

        if isWindows:
            file = files[fID].split('\\')[1]
        else:
            #pdb.set_trace()
            file = files[fID].split('/')[2]

        pdb.set_trace()
        fullPath = os.path.join(dataPath,tPath,file) 
        with open(fullPath,'rb') as fp:
            y = pickle.load(fp)
            yV = pickle.load(fp)
        
        
        # 地震プレートモデル用のオブジェクト
        #log = EarthQuakePlateModel(dataPath,file,nCell=8,nYear=10000)
        #log.loadABLV()
        #log.convV2YearlyData()
        
        # KDE
        #log.KDE()
        # FFT
        #log.FFT(widthWindow=10,eFrq=100)
        # Specgram
        #log.Specgram(width=500,nStride=10,nYear=8000)
        # SpecgramPlot
        #log.PlotSpecgram_KDE(width=500,nYear=8000)
        
        # 保存
        #log.plotV(isPlotShow=True,isYearly=False,prefix='yV')
        
        ############# Data作成の手順 #########################
        # 1. data=...の中のMode,Pathを指定する
        # 2. datapickleMode　を　3-2-1の順番でpickleを保存と読み込み
        #####################################################
    """
    #このファイルから直接Mode指定するときは、dataインスタンスの値を変更する必要がある（コマンドからは反応しない）
    data = Data(fname=fname,trainRatio=0.8, nCell=8, 
                sYear=2000, bInd=bInd, eYear=10000, isWindows=isWindows, isClass=True,
                CellMode=1,datapickleMode=datapickleMode,classMode=1,nClass=nClass,featuresPath='features', dataPath=dataPath,
                trainingpicklePath="train",picklePath=picklePath)"""
