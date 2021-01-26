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
    
    # ----
    def UVimg(self, x, y, obs, label='test', savename='test'):
        '''
        u(x,y) or v(x,y) of t time
        '''
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        X, Y = np.meshgrid(x,y)
        
        ax.plot_surface(X, Y, obs, cmap=cm.gray)
        
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$$')
        
        
        figurefullpath = os.path.join(self.figurePath, f'{savename}_{self.dataMode}', f'{label}.png')
        plt.savefig()
        
    
    
    # ----    