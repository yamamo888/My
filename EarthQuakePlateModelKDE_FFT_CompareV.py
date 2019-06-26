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
        self.logFullPath = os.path.join(self.logPath,logName)
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
        state_year = 2000
        nankai_ind,tonankai_ind,tokai_ind = 1,3,5
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
        self.yV = self.yV[state_year:,:].T
        
        # それぞれのセルに分ける
        nankaiyV,tonankaiyV,tokaiyV = self.yV[nankai_ind,:],self.yV[tonankai_ind,:],self.yV[tokai_ind,:]
        # 地震が発生したとこだけ
        slip_nankaiyV,slip_tonankaiyV,slip_tokaiyV = np.abs(np.where(nankaiyV>1)[0][1:] - np.where(nankaiyV>1)[0][:-1]), np.abs(np.where(tonankaiyV>1)[0][1:] - np.where(tonankaiyV>1)[0][:-1]), np.abs(np.where(tokaiyV>1)[0][1:] - np.where(tokaiyV>1)[0][:-1])
        # 地震間隔ベクトル
        # file名:ID
        file_name = self.logName.split(".")[0]
        
        return slip_nankaiyV,slip_tonankaiyV,slip_tokaiyV,file_name
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
        pdb.set_trace()

class CompareData:
    def __init__(self,standdata,comparedata):
        # 基準データと比較データの長さ取得
        self.standdata = standdata
        self.comparedata = comparedata
        self.len_stand = standdata.shape[0]
        self.len_compare = comparedata.shape[0]
    
    def Compare2Vec(self):
        
        # 基準データ>比較データ
        if self.len_stand > self.len_compare:
            compare_vec = np.abs(self.standdata[:self.len_compare] - self.comparedata)
            # 余った地震間隔
            residual_vec = np.abs(self.standdata[self.len_compare:])
        else:
            compare_vec = np.abs(self.standdata - self.comparedata[:self.len_stand])
            # 余った地震間隔
            residual_vec = np.abs(self.comparedata[self.len_stand:])
        
        # 余った地震間隔も合計
        compare_sum = np.sum(compare_vec) + np.sum(residual_vec)
        # 地震間隔比較ベクトルの合計
        compare_sum_non_residual = np.sum(compare_vec)
        
        return compare_sum,compare_sum_non_residual
        

#########################################

############## MAIN #####################
if __name__ == "__main__":
    
    """
    23456 23456 10,12,20,50
    """
    
    isWindows = True

    # Mode 設定
    # fileMode:使用するファイル指定 
    fileMode = int(sys.argv[1])
    # CellMode:使うセル指定
    CellMode = int(sys.argv[2])
    # クラス数指定(sigma) 
    classMode = int(sys.argv[3])

    if fileMode == 23456 or fileMode == 1234 or fileMode == 234 or fileMode == 23 or fileMode == 34 or fileMode == 45 or fileMode ==56 or fileMode == 67 or fileMode == 78:
        fname = '*txt'

    if CellMode == 23456:
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
   
    logsPath = './logs'
    dataPath = 'b2b3b4Files'
    tPath = 'train4'
    
    standardID = "stand"
    
    filePath = os.path.join(logsPath,fname) 
    picklePath = 'b{}b{}b{}_{}{}u'.format(bInd[0]+1,bInd[1]+1,bInd[2]+1,tPath,classMode)
    
    #Reading load log.txt
    if isWindows:
        files = glob.glob(filePath)
    
    else:
        files = glob.glob(filePath)
    
    flag = False
    for fID in np.arange(len(files)):
        print('reading',files[fID])

        if isWindows:
            file = files[fID].split('\\')[1]
        else:
            file = files[fID].split('/')[4]
        
        # 地震プレートモデル用のオブジェクト
        log = EarthQuakePlateModel(dataPath,file,nCell=8,nYear=10000)
        log.loadABLV()
        # 比較データ取得
        nankaiyV,tonankaiyV,tokaiyV,IDs = log.convV2YearlyData()
        """
        # KDE
        log.KDE()
        # FFT
        log.FFT(widthWindow=10,eFrq=100)
        """
        if standardID in IDs:    
            # 基準データ
            standard_nankaiyV = nankaiyV
            standard_tonankaiyV = tonankaiyV 
            standard_tokaiyV = tokaiyV
            stand_IDs = IDs
        else:
            compare_nankai = CompareData(standard_nankaiyV,nankaiyV)
            compare_tonankai = CompareData(standard_tonankaiyV,tonankaiyV)
            compare_tokai = CompareData(standard_tokaiyV,tokaiyV)
            compare_IDs = IDs
            
            # 各領域の地震間隔誤差と合計、はみ出した地震間隔を足した合計
            sum_nankai,sum_nankai_non_residual = compare_nankai.Compare2Vec()
            sum_tonankai,sum_tonankai_non_residual = compare_tonankai.Compare2Vec()
            sum_tokai,sum_tokai_non_residual = compare_tokai.Compare2Vec()
            
            # 各セルの地震間隔ベクトルを格納
            if not flag:
                nankaiSum,nankaiNonResidual = sum_nankai,sum_nankai_non_residual
                tonankaiSum,tonankaiNonResidual = sum_tonankai,sum_tonankai_non_residual
                tokaiSum,tokaiNonResidual = sum_tokai,sum_tokai_non_residual
                relateIDs = compare_IDs
                flag = True
            else:
                nankaiSum,nankaiNonResidual = np.vstack([nankaiSum,sum_nankai]),np.vstack([nankaiNonResidual,sum_nankai_non_residual]) 
                tonankaiSum,tonankaiNonResidual = np.vstack([tonankaiSum,sum_tonankai]),np.vstack([tonankaiNonResidual,sum_tonankai_non_residual])
                tokaiSum,tokaiNonResidual = np.vstack([tokaiSum,sum_tokai]),np.vstack([tokaiNonResidual,sum_tokai_non_residual])
                relateIDs = np.vstack([relateIDs,compare_IDs])
           
    # 近い順に並べ替えたindexを取得
    nankaiSumInd,nankaiNonResidualInd = np.argsort(np.reshape(nankaiSum,-1))[0],np.argsort(np.reshape(nankaiNonResidual,-1))[0]
    tonankaiSumInd,tonankaiNonResidualInd = np.argsort(np.reshape(tonankaiSum,-1))[0],np.argsort(np.reshape(tonankaiNonResidual,-1))[0]
    tokaiSumInd,tokaiNonResidualInd = np.argsort(np.reshape(tokaiSum,-1))[0],np.argsort(np.reshape(tokaiNonResidual,-1))[0]
    
    # 一番近いもののファイル名取得
    nankaiIDs,nankaiNonResidualIDs = str(relateIDs[nankaiSumInd]),str(relateIDs[nankaiNonResidualInd])
    tonankaiIDs,tonankaiNonResidualIDs = str(relateIDs[tonankaiSumInd]),str(relateIDs[tonankaiNonResidualInd])
    tokaiIDs,tokaiNonResidualIDs = str(relateIDs[tokaiSumInd]),str(relateIDs[tokaiNonResidualInd])
    
    IDsFileName = "yVIDs.txt"
    # ファイルに書き出し
    fp = open(IDsFileName,"a+")
    fp.write(nankaiIDs) 
    fp.write(nankaiNonResidualIDs)
    fp.write(tonankaiIDs) 
    fp.write(tonankaiNonResidualIDs)
    fp.write(tokaiIDs) 
    fp.write(tokaiNonResidualIDs)
        
    
    