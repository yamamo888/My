# coding: utf-8

import os

import numpy as np
import seaborn as sns

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
import matplotlib as cm

import warnings
warnings.simplefilter('ignore')

import pdb


class Plot:
    def __init__(self, figurepath='figure', dataMode='test', trialID=0):

          self.figurePath = figurepath
          self.dataMode = dataMode
          self.trialID = trialID
    
    # Plot loss (only one) ----
    def Loss1Data(self, data, labels, savename='loss'):
    
          sns.set_style('dark')

          plt.plot(data[0], linewidth=5, color='dimgrey', label=labels[0])
         
          plt.title('lLoss: %.10f, vLoss: %.10f' % (data[0][-1], data[0][0]))

          plt.xlabel('iteration',fontsize='18')
          plt.ylabel('loss',fontsize='18')
          plt.legend(fontsize='18')

          losspath = os.path.join(self.figurePath, 'loss', f'{savename}_{self.dataMode}_{self.trialID}.png')
          plt.savefig(losspath)
          plt.close()
    # ----
      
    # Plot loss (two data)----
    def Loss2Data(self, data, labels, savename='loss'):
    
          sns.set_style('dark')

          plt.plot(data[0], linewidth=5, color='navy', label=labels[0])
          plt.plot(data[1], linewidth=3, color='black', label=labels[1])

          plt.title('trLoss, teLoss: %f %f' % (data[0][-1], data[1][-1]))
    
          plt.xlabel('iteration')
          plt.ylabel('# of data')
          plt.legend()
    
          losspath = os.path.join(self.figurePath, 'loss', f'{savename}_{self.dataMode}_{self.trialID}.png')
          plt.savefig(losspath)
          plt.close()
    # ----