# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:51:45 2019

@author: yu
"""

import os
import pdb

import string

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

# ------------------------------- path ----------------------------------------
# save .eps path
visualPath = "visualization"
# save loss path
lossPath = "loss"
# save scatter paht
scatterPath = "scatter"
# -----------------------------------------------------------------------------

#-----------------------------------------------------------------------------#      
def Plot_3D(x1,x2,yGT,yPred,isPlot=False,savefilePath=0):

    """
    Visualization: Point cloud of evaluation data is blue with 3 axes of (x1, x2, y)
    Predicted value
    """
    
    table = str.maketrans("", "" , string.punctuation + ".")
    sigma = str(sigma).translate(table)
    
    if isPlot:
         
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("y")
        # 評価データplot
        ax.plot(np.reshape(x1,[-1,]),np.reshape(x2,[-1,]),np.reshape(yGT,[-1,]),"o",color="b",label="GT")
        # 予測値plot
        ax.plot(np.reshape(x1,[-1,]),np.reshape(x2,[-1,]),np.reshape(yPred,[-1,]),"o",color="r",label="Pred")
        plt.legend()
        fullPath = os.path.join(visualPath,"Pred_{}.eps".format(savefilePath))
        
        plt.savefig(fullPath)
    
#-----------------------------------------------------------------------------#              
def Plot_loss(trainTotalLosses, testTotalLosses, trainClassLosses, testClassLosses, trainRegLosses, testRegLosses, testPeriod, isPlot=False, savefilePath):
    if isPlot:
        if methodModel==2 or methodModel==1:
            # lossPlot
            plt.plot(np.arange(trainTotalLosses.shape[0]),trainTotalLosses,label="trainTotalLosses",color="r")
            plt.plot(np.arange(testTotalLosses.shape[0]),testTotalLosses,label="testTotalLosses",color="g")
            #plt.plot(np.arange(trainClassLosses.shape[0]),trainClassLosses,label="trainClassLosses",color="b")
            #plt.plot(np.arange(testClassLosses.shape[0]),testClassLosses,label="testClassLosses",color="k")
            #plt.plot(np.arange(trainRegLosses.shape[0]),trainRegLosses,label="trainRegLosses",color="c")
            #plt.plot(np.arange(testRegLosses.shape[0]),testRegLosses,label="testRegLosses",color="pink")
        
            plt.ylim([0,2.5])
            plt.xlabel("iteration x {}".format(testPeriod))
            plt.legend()
            
        else:
            
            plt.plot(np.arange(trainRegLosses.shape[0]),trainRegLosses,label="trainRegLosses",color="c")
            plt.plot(np.arange(testRegLosses.shape[0]),testRegLosses,label="testRegLosses",color="pink")
         
            plt.ylim([0,2.5])
            plt.xlabel("iteration x {}".format(testPeriod))
            plt.legend()
            
        fullPath = os.path.join(visualPath,lossPath,"Loss_{}.eps".format(savefilePath))
        plt.savefig(fullPath)
        plt.close()
#-----------------------------------------------------------------------------#      
def Plot_Scatter(gt,pred, isPlot=False, savefilePath):
    if isPlot:
        for cellInd in np.arange(3):
            fig = plt.figure(figsize=(9,6))    
            # 45° line
            line = np.arange(np.min(gt),np.max(gt)+0.001,0.001)
            # scatter
            plt.plot(gt[:,cellInd],pred[:,cellInd],".",color="black",linestyle="None",ms=10)
            # line
            plt.plot(line,line,"-",color="red",linewidth=4)
            
            plt.xlabel('ground truth',fontsize=22)
            plt.ylabel('predict',fontsize=22)
                
            plt.ylim([np.min(gt),np.max(gt)])
            plt.xlim([np.min(gt),np.max(gt)])
            
            fig.subplots_adjust(left=0.2,bottom=0.2)
            plt.tick_params(labelsize=22)
        
            savePath = os.path.join("Scatter_{}.png".format(savefilePath))    
            plt.savefig(savePath)
        
            plt.close()
#-----------------------------------------------------------------------------#      
def Plot_Alpha(trainAlpha,testAlpha,testPeriod, isPlot=False, savefilePath):
    if isPlot:
        plt.close()
        plt.plot(np.arange(trainAlpha.shape[1]),trainAlpha.T,label="trainAlpha",color="deepskyblue")
        plt.plot(np.arange(testAlpha.shape[1]),testAlpha.T,label="testAlpha",color="orange")
        
        plt.xlabel("iteration x {}".format(testPeriod))
        
        plt.legend()
        fullPath = os.path.join(visualPath,"Alpha_{}.eps".format(savefilePath))
        
        plt.savefig(fullPath)
