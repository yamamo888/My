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

      # Plot loss ----
      def Loss(self, data, labels, savename='loss'):

          sns.set_style('dark')

          plt.plot(data[0], linewidth=2, label=labels[0])
          plt.plot(data[1], linewidth=2, label=labels[1])
          plt.plot(data[2], linewidth=2, label=labels[1])

          plt.title('trLoss, teLoss, varLoss: %f %f %f' % (data[0][-1], data[1][-1], data[1][-1]))

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
          
          flag = False
          for nu in nus:
              print(f'start param nu:{nu} -> inv u')
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
      
      # ----
      def plotExactPredParam(self, params, xNum=256, tNum=100, savename='test'):
          
          x = params[0]
          t = params[1]
          prednus = params[2]
          exactnus = params[3]
          
          predus = self.paramToU([x,t,prednus], xNum=xNum, tNum=tNum)
          exactus = self.paramToU([x,t,exactnus], xNum=xNum, tNum=tNum)
          
          for predu,exactu,prednu,exactnu in zip([predus,exactus,prednus,exactnus]):
              # plot u
              self.Uimg(x ,t, exactu, predu, label=f'{exactnu}_{prednu}', savename=savename)
      # ----
      
      # u(t,x) ----
      def Uimg(self, x, t, exactu, predu, label='test', savename='u'):

          X, T = np.meshgrid(x,t) #[100,256]

          X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None])) # [25600,2]
          
          us = [exactu,predu]
          #pdb.set_trace()
          fig, axes = plt.subplots(nrows=2,sharex="col")
          for row,u in enumerate(us):
              u_star = u.flatten()[:,None] # [25600,1]
              # [100,256]
              U_star = griddata(X_star, u_star.flatten(), (X, T), method='cubic')

              img = axes[row].imshow(U_star.T, interpolation='nearest', cmap='gray',
                         extent=[t.min(), t.max(), x.min(), x.max()],
                         origin='lower', aspect='auto')

              divider1 = make_axes_locatable(axes[row])
              cax1 = divider1.append_axes("right", size="2%", pad=0.1)
              plt.colorbar(img,cax=cax1)
              
          #fig.canvas.draw()
          #axpos1 = axes[0].get_position() # 上の図の描画領域
          #axpos2 = ax2.get_position() # 下の図の描画領域
          #幅をax1と同じにする
          #ax2.set_position([axpos2.x0, axpos2.y0, axpos1.width, axpos2.height])
              
          plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
          plt.tick_params(bottom=False,left=False,right=False,top=False)
          plt.axis('off')

          fpath = os.path.join('figure', f'burgers_{savename}_{self.dataMode}')
          isdir = os.path.exists(fpath)
          if not isdir:
              os.makedirs(fpath)

          plt.savefig(os.path.join(fpath, f'{label}.png'))
      # ----
