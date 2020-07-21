# -*- coding: utf-8 -*-

import os

import numpy as np
import seaborn as sns

import matplotlib.pylab as plt

from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

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
      
      # ----
      def figsize(self, scale, nplots=1):
          # Get this from LaTeX using \the\textwidth
          fig_width_pt = 390.0
          # Convert pt to inch
          inches_per_pt = 1.0/72.27                       
           # Aesthetic ratio (you could change this)
          golden_mean = (np.sqrt(5.0)-1.0)/2.0           
          # width in inches
          fig_width = fig_width_pt*inches_per_pt*scale    
          # height in inches
          fig_height = nplots*fig_width*golden_mean             
          fig_size = [fig_width,fig_height]
          
          return fig_size
      # ----
      
      # u(t,x) ----
      #def udata(self, xt, trainxt, testu, predu, predinvu):
      def udata(self, xt, testu, predu, predinvu):
          '''
          xt: list 0:x, 1:t
          trainxt: list 0:train x, 1: train t
          testu: all u data
          '''
          
          # for plot data
          X, T = np.meshgrid(xt[0],xt[1]) #[100,256]
          X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
          # ground truth
          U_star = griddata(X_star, testu.flatten(), (X, T), method='cubic')
          # predict test (NN)
          U_pred = griddata(X_star, predu.flatten(), (X, T), method='cubic')
          # predict test (ODE)
          U_predinv = griddata(X_star, predinvu.flatten(), (X, T), method='cubic')
          
          
          fig = plt.figure(figsize=self.figsize(1.0, 1.4))
          ax = fig.add_subplot(111)
          
          ## ground truth ##
          ax.axis('off')
          gs0 = gridspec.GridSpec(1, 2)
          gs0.update(top=1-0.06, bottom=1-1.0/3.0+0.06, left=0.15, right=0.85, wspace=0)
          ax = plt.subplot(gs0[:, :])
          
          h = ax.imshow(U_star.T, interpolation='nearest', cmap='rainbow', 
                        extent=[xt[1].min(), xt[1].max(), xt[0].min(), xt[0].max()], 
                        origin='lower', aspect='auto')
          
          divider = make_axes_locatable(ax)
          cax = divider.append_axes("right", size="5%", pad=0.05)
          fig.colorbar(h, cax=cax)
          # train data
          #line = np.linspace(xt[1].min(), xt[1].max(), 2)[:,None]
          #for ind in np.arange(trainxt[0].shape[0]):
              #ax.plot(data, line, 'w-', linewidth = 5)
              #ax.axhline(trainxt[0][ind], ls="-", color="w", linewidth = 0.3)
          ##
          
          ## pred u (output neural network) ##
          gs1 = gridspec.GridSpec(1, 3)
          gs1.update(top=1-1.0/3.0-0.1, bottom=1.0-2.0/3.0, left=0.15, right=0.85, wspace=0)
          ax = plt.subplot(gs1[:, :])
    
          h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
                        extent=[xt[:,1].min(), xt[:,1].max(), xt[:,0].min(), xt[:,0].max()], 
                        origin='lower', aspect='auto')
          divider = make_axes_locatable(ax)
          cax = divider.append_axes("right", size="5%", pad=0.05)
          fig.colorbar(h, cax=cax)
          ##
          
          ## pred u (output ODE) ##
          gs2 = gridspec.GridSpec(1, 4)
          gs2.update(top=1-1.0/3.0-0.1, bottom=1.0-2.0/3.0, left=0.15, right=0.85, wspace=0)
          ax = plt.subplot(gs2[:, :])
    
          h = ax.imshow(U_predinv.T, interpolation='nearest', cmap='rainbow',
                        extent=[xt[:,1].min(), xt[:,1].max(), xt[:,0].min(), xt[:,0].max()], 
                        origin='lower', aspect='auto')
          divider = make_axes_locatable(ax)
          cax = divider.append_axes("right", size="5%", pad=0.05)
          fig.colorbar(h, cax=cax)
          ##
          
          upath = os.path.join(self.figurePath, 'u', f'{self.trialID}.png')
          plt.savefig(upath)
          plt.close()
          
      # ----
            
          
          
          
          
      
  
