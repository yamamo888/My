# -*- coding: utf-8 -*-

import os

import numpy as np
import seaborn as sns

import matplotlib.pylab as plt

from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

import warnings
warnings.simplefilter('ignore')

import pdb

class Plot:
      def __init__(self, figurepath='figure', dataMode='test', trialID=0):

          # cell index
          self.ntI = 0
          self.tntI = 1
          self.ttI = 2

          self.figurePath = figurepath
          self.dataMode = dataMode
          self.trialID = trialID

      # Plot loss (two data)----
      def Loss(self, data, labels, savename='loss'):

          sns.set_style('dark')

          plt.plot(data[0], linewidth=2, label=labels[0])
          plt.plot(data[1], linewidth=2, label=labels[1])

          plt.title('trLoss, teLoss: %f %f' % (data[0][-1], data[1][-1]))

          plt.xlabel('iteration')
          plt.ylabel('# of data')
          plt.legend()

          losspath = os.path.join(self.figurePath, 'loss', f'{savename}_{self.dataMode}_{self.trialID}.png')
          plt.savefig(losspath)
          plt.close()
      # ----
      
      # Plot loss (only one) ----
      def Loss1(self, data, labels, savename='loss'):

          sns.set_style('dark')

          plt.plot(data[0], linewidth=2, label=labels[0])
         
          plt.title('Loss: %f' % (data[0][-1]))

          plt.xlabel('iteration')
          plt.ylabel('# of data')
          plt.legend()

          losspath = os.path.join(self.figurePath, 'loss', f'{savename}_{self.dataMode}_{self.trialID}.png')
          plt.savefig(losspath)
          plt.close()
      # ----
      
      # nu -> u(t,x) ----
      def paramToU(self, params, xNum=256, tNum=100):
          
          x = params[0]
          t = params[1]
          nus = params[2]
          
          
          # 0.01 < x < 5.0
          nus = np.where(np.round(nus,3)<0.01, 0.01, np.where(np.round(nus,3)>5.0, 5.0, nus))
          
          if any(nus) < 0.01:
              pdb.set_trace()
          
          flag = False
          for nu in nus:
              #print(f'start param nu:{nu} -> inv u')
              # observation
              obsu = np.zeros([xNum, tNum])
    
              for j in range (0, tNum):
                  for i in range (0, xNum):
                      a = ( x[i] - 4.0 * t[j] )
                      b = ( x[i] - 4.0 * t[j] - 2.0 * np.pi )
                      
                      c = 4.0 * nu * ( t[j] + 1.0 )
    
                      phi = np.exp ( - a * a / c ) + np.exp ( - b * b / c )
                      dphi = - 2.0 * a * np.exp ( - a * a / c ) / c \
                             - 2.0 * b * np.exp ( - b * b / c ) / c
    
                      obsu[i,j] = 4.0 - 2.0 * nu * dphi / phi
            
              # for pNN_burgers
              if not flag:
                  predobsu = obsu[np.newaxis]
                  flag = True
              else:
                  predobsu = np.vstack([predobsu, obsu[np.newaxis]])

          return predobsu
      # ----
      
      # param NN ----
      def plotExactPredParam(self, params, xNum=256, tNum=100, itr=0, savename='test'):
          
          x = params[0]
          t = params[1]
          prednus = params[2]
          exactnus = params[3]
          
          predus = self.paramToU([x,t,prednus], xNum=xNum, tNum=tNum)
          exactus = self.paramToU([x,t,exactnus], xNum=xNum, tNum=tNum)
          
          for predu,exactu,prednu,exactnu in zip(predus,exactus,prednus,exactnus):
              # plot u
              self.Uimg(x ,t, exactu, predu, label=f'{exactnu}_{prednu}_{itr}', savename=savename)
      # ----
      
      # ----
      def CycleExactPredParam(self, params, xNum=256, tNum=100, itr=0, savename='test'):
          
          x = params[0]
          t = params[1]
          prednus = params[2]
          exactnus = params[3]
          cycleloss = params[4]
          grads = params[5]
          
          predus = self.paramToU([x,t,prednus], xNum=xNum, tNum=tNum)
          exactus = self.paramToU([x,t,exactnus], xNum=xNum, tNum=tNum)
          
          for predu,exactu,prednu,exactnu in zip(predus,exactus,prednus,exactnus):
              # plot u
              self.Uimg(x ,t, exactu, predu, label=f'{exactnu}_{prednu}_{itr}', label2=f'{cycleloss}_{grads}', savename=savename, isCycle=True)
      # ----
      
      # u(t,x) ----
      def Uimg(self, x, t, exactu, predu, label='test', label2='test', savename='u', isCycle=False):
          #pdb.set_trace() 
          X, T = np.meshgrid(x,t) #[100,256]

          X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None])) # [25600,2]
      
          us = [exactu.T,predu.T]
        
          # u of exact nu (row0) -> pred nu (row1)
          fig, axes = plt.subplots(nrows=2)
          for row,u in enumerate(us):
        
              # flatten: [100,256]
              u_star = u.flatten()[:,None] # [25600,1]
              # [100,256]
              U_star = griddata(X_star, u_star.flatten(), (X, T), method='cubic')
    
              img = axes[row].imshow(U_star.T, interpolation='nearest', cmap='gray',
                        extent=[t.min(), t.max(), x.min(), x.max()],
                        origin='lower', aspect='auto')
              pdb.set_trace()
              if isCycle:
                  if row == 0:
                      titlelabel = 'exact nu='
                      axes[row].set_title('%s %3f' % (titlelabel, np.float(label.split('_')[row][1:-1])))
                  elif row == 1:
                      titlelabel = ['predict nu=', 'cyclecloss', 'grad']
                      axes[row].set_title('%s %3f %s %5f %s %3f' % (titlelabel[0], np.float(label.split('_')[row][1:-1]), titlelabel[1], np.float(label2.split('_')[0]), titlelabel[2], np.float(label2.split('_')[1])))

              else:
                  if row == 0:
                      titlelabel = 'exact nu='
                  elif row == 1:
                      titlelabel = 'predict nu='
                  axes[row].set_title('%s %5f' % (titlelabel, np.float(label.split('_')[row][1:-1])))
            
              divider1 = make_axes_locatable(axes[row])
              cax1 = divider1.append_axes("right", size="2%", pad=0.1)
              plt.colorbar(img,cax=cax1)
            
              axes[row].set_xlabel('t', fontsize=10)
              axes[row].set_ylabel('u(t,x)', fontsize=10)
        
          plt.tight_layout()
        
          fpath = os.path.join('figure', f'burgers_{savename}_{self.dataMode}')
          isdir = os.path.exists(fpath)
      
          if not isdir:
              os.makedirs(fpath)
          plt.savefig(os.path.join(fpath, f'{label}.png'))
          plt.close()
      # ----
