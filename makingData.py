# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:51:43 2019

@author: yu
"""
import os
import pickle
import pdb

import numpy as np

class toyData:
	def __init__(self, trainRatio=0.8, nData=4000, yMin=2, yMax=6,pNum=5,sigma=0,nClass=10):
		#----------------------------- paramters --------------------------------------
		self.trainRatio = trainRatio # training rate
		self.nData = nData # number of data
		self.nTrain = int(self.nData * self.trainRatio) # number of train data
		self.nTest = int(self.nData - self.nTrain) # number of test data
		self.yMin = yMin
		self.yMax = yMax
		self.pNum = pNum #Rotation of explanatory variable x1, x2
		self.sigma = sigma
		self.nClass = nClass

		self.batchCnt = 0


		self.dataPath = "trainData"
	# -----------------------------------------------------------------------------

	#------------------------------------------------------------------------------
	"""
	Split data into train data and test data.

	Returns:
		train,test data
	"""
	def createData(self, trialID=1, beta=1):
		toydataPath = "toyData_{}_{}_{}_{}_{}.pickle".format(self.sigma,self.pNum,self.nTrain,self.nTest,trialID)
		toyDatafullPath = os.path.join(self.dataPath,toydataPath)

		# reading data from pickle
		if os.path.exists(toyDatafullPath):	 
			with open(toyDatafullPath,"rb") as fp:
				self.x1Train = pickle.load(fp)
				self.x2Train = pickle.load(fp)
				self.yTrain = pickle.load(fp)
				self.x1Test = pickle.load(fp)
				self.x2Test = pickle.load(fp)
				self.yTest = pickle.load(fp)
				self.yAll= pickle.load(fp)
				self.batchRandInd = pickle.load(fp)
		else:
			# Make target variable, y ~ U(x) U: i.i.d.
			self.yAll = np.random.uniform(self.yMin,self.yMax,self.nData) 
			#x1 = np.sin(self.pNum * self.yAll) + 1 / np.log(self.yAll) + np.random.normal(scale=self.sigma)
			#x2 = np.cos(self.pNum * self.yAll) + np.log(self.yAll) + np.random.normal(scale=self.sigma)
			x1 = np.sin(self.pNum * self.yAll) + np.random.normal(scale=self.sigma)
			x2 = np.cos(self.pNum * self.yAll) + np.random.normal(scale=self.sigma)
			
			# split all data to train & test data
			self.x1Train = x1[:self.nTrain][:,np.newaxis]
			self.x2Train = x2[:self.nTrain][:,np.newaxis]
			self.yTrain = self.yAll[:self.nTrain][:,np.newaxis]
			self.x1Test = x1[self.nTrain:][:,np.newaxis]
			self.x2Test = x2[self.nTrain:][:,np.newaxis]
			self.yTest = self.yAll[self.nTrain:][:,np.newaxis]

			# batch random index
			self.batchRandInd = np.random.permutation(self.nTrain)

			with open(toyDatafullPath,"wb") as fp:
				pickle.dump(self.x1Train,fp)
				pickle.dump(self.x2Train,fp)
				pickle.dump(self.yTrain,fp)
				pickle.dump(self.x1Test,fp)
				pickle.dump(self.x2Test,fp)
				pickle.dump(self.yTest,fp)
				pickle.dump(self.yAll,fp)
				pickle.dump(self.batchRandInd,fp)


		self.xTrain = np.concatenate([self.x1Train,self.x2Train],1)
		self.xTest = np.concatenate([self.x1Test,self.x2Test],1)


		self.annotate(beta)
	#-----------------------------------------------------------------------------	  

	#-----------------------------------------------------------------------------	  
	def annotate(self,beta=1):
		# class
		yClass = np.arange(self.yMin,self.yMax + beta, beta) 
	 
		flag = False
		for nInd in np.arange(self.yAll.shape[0]):
			tmpY = self.yAll[nInd]
			oneHot = np.zeros(len(yClass)-1)

			# (最小、最大]
			for cInd in range(len(yClass)-1):
				if (tmpY >= yClass[cInd]) & (tmpY < yClass[cInd+1]): 
					oneHot[cInd] = 1			

			'''
			# 最小値は0番目のクラスにする
			if self.yAll[nInd] == self.yMin:
				oneHot[0] = 1
			# 最大値が一番最後のクラスにラベルされるのを戻す
			if self.yAll[nInd] == self.yMax:
				oneHot[-2] = 1
			'''
			
			tmpY = oneHot[np.newaxis] 
				  
			if not flag:
				yLabel = tmpY
				flag = True
			else:
				yLabel = np.vstack([yLabel,tmpY])
			
		'''	
		# 値が入っていないクラスを削除
		if len(yClass) == self.nClass + 1:
			yLabel = yLabel[:,:-1]
		'''
		
		self.yTrainLabel = yLabel[:self.nTrain]
		self.yTestLabel = yLabel[self.nTrain:]

	#-----------------------------------------------------------------------------#
	def nextBatch(self, batchSize):	
		sInd = batchSize * self.batchCnt
		eInd = sInd + batchSize

        batchX = self.xTrain[self.batchRandInd[sInd:eInd],:]
		batchY = self.yTrain[self.batchRandInd[sInd:eInd],:]
		batchlabelY = self.yTrainLabel[self.batchRandInd[sInd:eInd],:]
        
		if eInd + batchSize > self.nTrain:
			self.batchCnt = 0
		else:
			self.batchCnt += 1

		return batchX,batchY,batchlabelY
	#-----------------------------------------------------------------------------#
		
