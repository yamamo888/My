# -*- coding: utf-8 -*-

import os

import numpy as np
import seaborn as sns

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt



class Plot:
      def __init__(self, figurepath='figure', trialID=0):
          
          # cell index
          self.ntI = 0
          self.tntI = 1
          self.ttI = 2
          
          self.figurePath = figurepath
          self.trialID = trialID
      
      # Plot loss ----
      def Loss(self, data):
          
          plt.plot(data)
          
          losspath = os.path.join(self.figurePath, 'loss', f'{self.trialID}')
          plt.savefig(losspath)
          plt.close()
      # ----
      
      # Plot rireki ----
      def Rireki(self, gt, pred):
        
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
      
  