# -*- coding: utf-8 -*-

import os

import numpy as np
import seaborn as sns

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
          
          sns.set_style('dark')
      
          plt.plot(data, linewidth=2, label=labels[0])
          
          plt.title('evCycleLoss: %f' % (data[-1]))
          
          plt.xlabel('iteration')
          plt.ylabel('# of data')
          plt.legend()

          losspath = os.path.join(self.figurePath, 'loss', f'{self.trialID}_{labels[0]}.png')
          plt.savefig(losspath)
          plt.close()
      # ----
      
      # Plot loss ----
      def pLoss(self, data, labels):
          
          sns.set_style('dark')
      
          plt.plot(data[0], linewidth=2, label=labels[0])
          plt.plot(data[1], linewidth=2, label=labels[1])
          
          plt.title('trLoss, teLoss: %f %f ' % (data[0][-1], data[1][-1]))
          
          plt.xlabel('iteration')
          plt.ylabel('# of data')
          plt.legend()
          
          losspath = os.path.join(self.figurePath, 'loss', f'{self.trialID}.png')
          plt.savefig(losspath)
          plt.close()
      # ----
      
      # Plot exact-pred scatter ----
      def epScatter(self, params, labels, isP=False):
          '''
          params[-1] : test exaxt paramb
          '''
          
          sns.set_style('dark')
          
          # cycle loss pred
          fig, figInds = plt.subplots(ncols=3, sharex=True, sharey=True)
          
          if isP:
              for figInd in np.arange(len(figInds)):
                  figInds[figInd].scatter(params[1][:,figInd], params[2][:,figInd], c='black')
                  
              fig.suptitle(f'{labels[0]}')
              
          else:    
              for figInd in np.arange(len(figInds)):
                  figInds[figInd].scatter(params[2][:,figInd], params[1][:,figInd], c='black')
                  
              fig.suptitle(f'{labels[1]}')
          
          pcpath = os.path.join(self.figurePath, 'gtpred', f'{self.trialID}.png')
          plt.savefig(pcpath)
          plt.close()  
      # ----
      
      # Plot rireki ----
      def Rireki(self, gt, pred, label='pred'):
        
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
        
        rirekipath = os.path.join(self.figurePath, 'rireki', f'{label}_{self.trialID}.png')
        plt.savefig(rirekipath)
        plt.close()
        
        print(pred)
        print(gt)
      # ----
      
      # Scatter rireki ----
      def scatterRireki(self, gt, pred, path='none', label='pred'):
          
          sns.set_style('dark')
          
          fig = plt.figure()
          fig, axes = plt.subplots(nrows=3,sharex="col")
          for figInd in np.arange(3):     
              axes[figInd].plot(np.arange(gt.shape[0]), gt[:,figInd], color="coral", marker='.', linestyle='None')
              axes[figInd].plot(np.arange(gt.shape[0]), pred[:,figInd], color="skyblue", marker='.', ms=0.5, linestyle='None')
        
          rirekipath = os.path.join(self.figurePath, 'srireki', path, f'{label}_{self.trialID}.png')
          plt.savefig(rirekipath)
          plt.close()
      # ----
        
      # ----
      def feature2D(self, data, label, range):
          #pdb.set_trace()
          sns.heatmap(data[np.newaxis], vmin=range[0], vmax=range[1])
          rirekipath = os.path.join(self.figurePath, 'feature', f'{label}.png')
          plt.savefig(rirekipath)
          plt.close()
      # ----
      
      # ----
      def BoxPlot(self, data, label='pred'):
          
          sns.set_style("dark")
           
          fig,ax = plt.subplots()
          ax.boxplot([data],sym='d',patch_artist=True,boxprops=dict(facecolor='lightblue',color='gray'),medianprops=dict(color='gray'),labels=['pcNN'])
          ax.set_ylabel('MAE')
            
          bpath = os.path.join(self.figurePath, 'box', f'{label}_{self.trialID}.png')
          plt.savefig(bpath)
          plt.close()
      # ----
        
          
          
          
          
      
  
