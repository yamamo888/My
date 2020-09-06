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
      def paramToU(self, params, xNum=256, tNum=100, savename='exact', isPlot=False):
          
          x = params[0]
          t = params[1]
          nus = params[2]
          exactnu = params[3]
          
          flag = False
          losses = []
          for enu, nu in zip(exactnu,nus):
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
              
              if isPlot:
                  
                  loss = np.square(enu - nu)
                  losses = np.append(losses, loss)
                  
                  # plot u
                  self.Uimg(x ,t, obsu, label=nu, savename=savename)
              # for pNN_burgers
              else:
                  
                  if not flag:
                      predobsu = obsu[np.newaxis]
                      flag = True
                  else:
                      predobsu = np.vstack([predobsu, obsu[np.newaxis]])
        
          if not isPlot:
              return predobsu
          else:
              exactu_loss = np.hstack([exactnu, losses[:,np.newaxis]])
              # save loss
              np.savetxt(os.path.join('result',f'burgers_{savename}_{self.dataMode}.csv'), exactu_loss, fmt='%4f', delimiter=',')
                  
      # ----

      # u(t,x) ----
      def Uimg(self, x, t, u, label='test', savename='u'):

          X, T = np.meshgrid(x,t) #[100,256]

          X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None])) # [25600,2]
          u_star = u.flatten()[:,None] # [25600,1]

          # [100,256]
          U_star = griddata(X_star, u_star.flatten(), (X, T), method='cubic')

          #pdb.set_trace()

          img = plt.imshow(U_star.T, interpolation='nearest', cmap='gray',
                           extent=[t.min(), t.max(), x.min(), x.max()],
                           origin='lower', aspect='auto')

          #plt.colorbar()
          plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
          plt.tick_params(bottom=False,left=False,right=False,top=False)
          plt.axis('off')

          fpath = os.path.join('figure', f'burgers_{savename}_{self.dataMode}')
          isdir = os.path.exists(fpath)
          if not isdir:
              os.makedirs(fpath)

          plt.savefig(os.path.join(fpath, f'{label}.png'))
      # ----
