# -*- coding: utf-8 -*-

import os

import numpy as np
import seaborn as sns

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt

import pdb

class Plot:
      def __init__(self, figurepath='figure', trialID=0):
          
          # cell index
          self.ntI = 0
          self.tntI = 1
          self.ttI = 2
          
          self.figurePath = figurepath
          self.trialID = trialID
      
      # Plot loss ----
      def pcLoss(self, data, labels):
          #pdb.set_trace()
          
          sns.set_style('dark')
          # train
          plt.plot(data[0], linewidth=2, label=labels[0])
          plt.plot(data[1], linewidth=2, label=labels[1])
          plt.plot(data[2], linewidth=2, label=labels[2])
          
          # test
          plt.plot(data[3], linewidth=2, label=labels[3])
          plt.plot(data[4], linewidth=2, label=labels[4])
          plt.plot(data[5], linewidth=2, label=labels[5])
          
          plt.title('trPLoss,trCLoss,trPCLoss: %f %f %f\n teLoss,teCLoss,tePCLoss: %f %f %f' 
                    % (data[0][-1], data[1][-1], data[2][-1], data[3][-1], data[4][-1], data[5][-1]))
          
          plt.xlabel('iteration')
          plt.ylabel('# of data')
          plt.legend()

          losspath = os.path.join(self.figurePath, 'loss', f'{self.trialID}.png')
          plt.savefig(losspath)
          plt.close()
      # ----
      
      # Plot loss ----
      def cLoss(self, data, labels):
          #pdb.set_trace()
          
          sns.set_style('dark')
      
          plt.plot(data[0], linewidth=2, label=labels[0])
          
          plt.title('evCycleLoss: %f' % (data[0]))
          
          plt.xlabel('iteration')
          plt.ylabel('# of data')
          plt.legend()

          losspath = os.path.join(self.figurePath, 'loss', f'{self.trialID}.png')
          plt.savefig(losspath)
          plt.close()
      # ----
      
      
      # Plot exact-pred scatter ----
      def epScatter(self, params, labels):
          '''
          params[-1] : test exaxt paramb
          '''
          
          sns.set_style('dark')
          
          # cycle loss pred
          fig, figInds = plt.subplots(ncols=3, sharex=True, sharey=True)
              
          for figInd in np.arange(len(figInds)):
              figInds[figInd].scatter(params[2][:,figInd], params[1][:,figInd], c='black')
              
          fig.suptitle(f'{labels[1]}')
          
          pcpath = os.path.join(self.figurePath, 'gtpred', f'{self.trialID}_{labels[1]}.png')
          plt.savefig(pcpath)
          plt.close()  
      # ----
      
      
      # Plot rireki ----
      def Rireki(self, gt, pred):
        
        sns.set_style('dark')
        
        predV,gtV = np.zeros([1400,3]), np.zeros([1400,3])
        
        predV[pred[self.ntI].tolist(), self.ntI] = 5
        predV[pred[self.tntI].tolist(), self.tntI] = 5
        predV[pred[self.ttI].tolist(), self.ttI] = 5
        
        gtV[gt[self.ntI].tolist(), self.ntI] = 5
        gtV[gt[self.tntI].tolist(), self.tntI] = 5
        gtV[gt[self.ttI].tolist(), self.ttI] = 5
       
        colors = ["coral","skyblue","coral","skyblue","coral","skyblue"]
        plot_data = [gtV[:,self.ntI],predV[:,self.ntI],gtV[:,self.tntI],predV[:,self.tntI],gtV[:,self.ttI],predV[:,self.ttI]]
          
        fig = plt.figure()
        fig, axes = plt.subplots(nrows=6,sharex="col")
        for row,(color,data) in enumerate(zip(colors,plot_data)):
            axes[row].plot(np.arange(1400), data, color=color)
         
        rirekipath = os.path.join(self.figurePath, 'rireki', f'{self.trialID}')
        plt.savefig(rirekipath)
        plt.close()
      # ----
      
  
