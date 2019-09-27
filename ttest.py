import numpy as np
import pdb
from scipy import stats
import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
import time
import pandas as pd
import seaborn as sns
import os
import sys


#------------------------------------------------------
class Plot():
    def __init__(self,visualizationpath):
        self.visualPath = visualizationpath

    def Line(self,data):

        x = np.arange(np.min(data),np.max(data)+0.001,0.001)
        y = x
        return x,y

    def Scatter(self,test1,test2,test3,pred1,pred2,pred3,pro1,pro2,pro3,depth=0,nClass=0,isAnchor=False):

        x1,y1 = self.Line(test1)
        x2,y2 = self.Line(test2)
        x3,y3 = self.Line(test3)
        if isAnchor:
            plt.plot(test1,pred1,".",color="y",linestyle="None",label='Anchor-based')
        
        else:
            plt.plot(test1,pred1,".",color='m',linestyle="None",label='Regression')

        plt.plot(test1,pro1,".",color='c',linestyle="None",label='ATR-Nets')
        plt.plot(x1,y1,"-",color="black",linewidth=4)
        plt.xlabel('ground truth')
        plt.ylabel('predict')
        plt.ylim([np.min(test1),np.max(test1)])
        plt.xlim([np.min(test1),np.max(test1)])
        plt.legend(loc="best")
        savePath = os.path.join(self.visualPath,"nankai2_{}_{}.png".format(depth,nClass))
        plt.savefig(savePath)
        plt.close()
        
        if isAnchor:
            plt.plot(test2,pred2,".",color="y",linestyle="None",label='Anchor-based')
        
        else:
            plt.plot(test2,pred2,".",color='m',linestyle="None",label='Regression')


        plt.plot(test2,pro2,".",color='c',linestyle="None",label='ATR-Nets')
        plt.plot(x2,y2,"-",color="black",linewidth=4)
        plt.xlabel('ground truth')
        plt.ylabel('predict')
        plt.ylim([np.min(test2),np.max(test2)])
        plt.xlim([np.min(test2),np.max(test2)])
        plt.legend(loc="best")
        savePath = os.path.join(self.visualPath,"tonankai2_{}_{}.png".format(depth,nClass))
        plt.savefig(savePath)
        plt.close()
        
        if isAnchor:
            plt.plot(test3,pred3,".",color="y",linestyle="None",label='Anchor-based')
        
        else:
            plt.plot(test3,pred3,".",color='m',linestyle="None",label='Regression')

        plt.plot(test3,pro3,".",color='c',linestyle="None",label='ATR-Nets')
        plt.plot(x3,y3,"-",color="black",linewidth=4)
        plt.xlabel('ground truth')
        plt.ylabel('predict')
        plt.ylim([np.min(test3),np.max(test3)])
        plt.xlim([np.min(test3),np.max(test3)])
        plt.legend(loc="best")
        savePath = os.path.join(self.visualPath,"tokai2_{}_{}.png".format(depth,nClass))
        plt.savefig(savePath)
        plt.close()
    
    def Average6Variable(self,test,pred,pro):
        
        print("Mean",np.mean(np.abs(test-pred)))
        print("Var",np.var(np.abs(test-pred)))
        print("ProposedMean",np.mean(np.abs(test-pro)))
        print("ProposedVar",np.var(np.abs(test-pro)))
        print("--------------------------------")
        
        """
    def Average6Variable(self,test1,test2,test3,pred1,pred2,pred3,pro1,pro2,pro3):
        
        tests = [test1,test2,test3]
        preds = [pred1,pred2,pred3]
        pros = [pro1,pro2,pro3]
        cnt = 1
        for test,pred,pro in zip(tests,preds,pros):
            print(cnt)
            print("Mean",np.mean(np.abs(test-pred)))
            print("Var",np.var(np.abs(test-pred)))
            print("ProposedMean",np.mean(np.abs(test-pro)))
            print("ProposedVar",np.var(np.abs(test-pro)))
            print("--------------------------------")
            cnt +=1
    """    
        
    """ 
    def EvaluationT(self,pred1,pred2,pred3,pro1,pro2,pro3):

        size = int(pred1.shape[0]/2)
        pnum = 0.05
        for cnt in np.arange(int(pred1.shape[0]/size)):
            sInd = size * cnt
            eInd = sInd + size
            t1,p1 = stats.ttest_ind(pro1[sInd:eInd],pred1[sInd:eInd])
            t2,p2 = stats.ttest_ind(pro2[sInd:eInd],pred2[sInd:eInd])
            t3,p3 = stats.ttest_ind(pro3[sInd:eInd],pred3[sInd:eInd])
    
            if p1<pnum and p2<pnum and p3<pnum:
                print("有意")
                print("p1:%f"%p1)
                print("p2:%f"%p2)
                print("p3:%f"%p3)
            else:
                print('有意でない')
                print("p1:%f"%p1)
                print("p2:%f"%p2)
                print("p3:%f"%p3)
            
            time.sleep(1)
    """
    def EvaluationT(self,pred1,pro1):

        size = int(pred1.shape[0])
        pnum = 0.01
        for cnt in np.arange(int(pred1.shape[0]/size)):
            sInd = size * cnt
            eInd = sInd + size
            t1,p1 = stats.ttest_rel(pro1[sInd:eInd],pred1[sInd:eInd])

            if p1<pnum:
                print("有意")
                print("p1:%f"%p1)
            else:
                print('有意でない')
                print("p1:%f"%p1)
    
            time.sleep(1)

if __name__ == "__main__":
    
    visualization = "visualization"
    residual = "residual"
    toy = "toy" 
    myPlot = Plot(visualization)
    # Anchor & ATR-Nets
    for j in np.arange(2):
        
        if j==0:
            nClass=10
        elif j==1:
            nClass=20
        print("Class",nClass)
        for nh in np.arange(3,6,1):
            print("layer",nh)

            picklefullPath = os.path.join(visualization,)
            with open(picklefullPath,'rb') as fp:
                    b1 = pickle.load(fp)
                    b2 = pickle.load(fp)
                    b3 = pickle.load(fp)

            """
            picklefullPath = os.path.join(visualization,residual,"TestR_299500_23456_{}.pickle".format(nh))
            with open(picklefullPath,'rb') as fp:
                    b1 = pickle.load(fp)
                    b2 = pickle.load(fp)
                    b3 = pickle.load(fp)
                    testY1 = pickle.load(fp)
                    testY2 = pickle.load(fp)
                    testY3 = pickle.load(fp)
                
            b1 = b1[:,np.newaxis]
            b2 = b2[:,np.newaxis]
            b3 = b3[:,np.newaxis]
           """ 
            
            
            picklefullPath = os.path.join(visualization,"toy2","TestCR_299500_1_20_5_{}_7000.pickle".format(nh))
            with open(picklefullPath,'rb') as fp:
                    b1 = pickle.load(fp)
                    testY1 = pickle.load(fp)
             
            picklefullPath = os.path.join(visualization,"toy2","TestCR_299500_1_10_5_{}_7000.pickle".format(nh))
            with open(picklefullPath ,'rb') as fp:
                    pb1 = pickle.load(fp)
                    testY1 = pickle.load(fp)
            """
            picklefullPath = os.path.join(visualization,residual,"TestATR_299500_10_{}.pickle".format(nh))
            with open(picklefullPath,'rb') as fp:
                    b1 = pickle.load(fp)
                    b2 = pickle.load(fp)
                    b3 = pickle.load(fp)
                    testY1 = pickle.load(fp)
                    testY2 = pickle.load(fp)
                    testY3 = pickle.load(fp)
            b1 = b1[:,np.newaxis]
            b2 = b2[:,np.newaxis]
            b3 = b3[:,np.newaxis]
            
            picklefullPath = os.path.join(visualization,residual,"TestATR_299500_20_{}.pickle".format(nh))
            with open(picklefullPath ,'rb') as fp:
                    pb1 = pickle.load(fp)
                    pb2 = pickle.load(fp)
                    pb3 = pickle.load(fp)
                    testY1 = pickle.load(fp)
                    testY2 = pickle.load(fp)
                    testY3 = pickle.load(fp)
            pb1 = pb1[:,np.newaxis]
            pb2 = pb2[:,np.newaxis]
            pb3 = pb3[:,np.newaxis]
            """ 
            #isAnchor = True 
            #myPlot.Scatter(testY1,testY2,testY3,b1,b2,b3,pb1,pb2,pb3,depth=nh,nClass=nClass,isAnchor=True)
            #myPlot.Average6Variable(testY1,testY2,testY3,b1,b2,b3,pb1,pb2,pb3)
            #myPlot.EvaluationT(b1,b2,b3,pb1,pb2,pb3)
            myPlot.Average6Variable(testY1,b1,pb1)
            #myPlot.EvaluationT(b1,pb1)
