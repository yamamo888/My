# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
from scipy import stats

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pylab as plt
import seaborn as sns

import pickle
import glob
import pdb
import random
import itertools
from natsort import natsorted
import time

#########################################
class EarthQuakePlateModel:
        
    def __init__(self,dataPath,logName,nCell=8,nYear=10000):
        # full path
        self.logFullPath = os.path.join(dataPath,logName)
        # ---- paramters ---- #
        self.nCell = nCell
        self.nYear = nYear
        # gt Year
        self.Year = 1400
        self.yInd = 1
        self.vInds = [2,3,4,5,6,7,8,9]
        self.yV = np.zeros([nYear,nCell])
        # nakai, tonakai, tokai 8cell simulation
        self.nI,self.tnI,self.tI = 2,4,5
        # size of slip velocity
        self.slip = 1
        # ------------------ #
    
    #--------------------------
    #データの読み込み
    def loadABLV(self):
        self.data = open(self.logFullPath).readlines()
        
        # B の取得
        self.B = np.zeros(self.nCell)
        
        for i in np.arange(1,self.nCell+1):
            tmp = np.array(self.data[i].strip().split(",")).astype(np.float32)
            self.B[i-1] = tmp[1]
            
        # Uの開始行取得
        isRTOL = [True if self.data[i].count('value of RTOL')==1 else False for i in np.arange(len(self.data))]
        vInd = np.where(isRTOL)[0][0]+1
        
        # Uの値の取得（vInd行から最終行まで）
        flag = False
        for i in np.arange(vInd,len(self.data)):
            tmp = np.array(self.data[i].strip().split(",")).astype(np.float32)
            
            if not flag:
                self.V = tmp
                flag = True
            else:
                self.V = np.vstack([self.V,tmp])
        
        return self.B
    #--------------------------
    # Vを年単位のデータに変換
    def convV2YearlyData(self):
        # 初めの観測した年
        sYear = np.floor(self.V[0,self.yInd])
        
        # 観測データがない年には観測データの１つ前のデータを入れる(累積)
        for year in np.arange(sYear,self.nYear):
            # 観測データがある場合
            if np.sum(np.floor(self.V[:,self.yInd])==year):
                
                # 観測データがあるときはそのまま代入
                self.yV[int(year)] = self.V[np.floor(self.V[:,self.yInd])==year,self.vInds[0]:]
                
            # 観測データがない場合
            else:
                # その1つ前の観測データを入れる
                self.yV[int(year)] = self.yV[int(year)-1,:]
            
        # 累積速度から、速度データにする
        deltaV = self.yV[self.yInd:]-self.yV[:-self.yInd]
        # 一番最初のデータをappendして、10000年にする
        self.yV = np.concatenate((self.yV[np.newaxis,0],deltaV),0)
        self.yV = self.yV[2000:]
       
    #--------------------------
    
    #--------------------------
    # 正規分布で比較
    def comparedGauss(self,gt,fID=0,label=0,savePath="none",isPlot=False):
        
        # predicted earth (8000,3)
        pred = np.concatenate((self.yV[:,self.nI][:,np.newaxis],self.yV[:,self.tnI][:,np.newaxis],self.yV[:,self.tI][:,np.newaxis]),1)
        
        # ----
        # 何年間の誤差を許容するか
        sigma = 50
        # 予測した地震年数　(all cell(重複なし))
        pYear = np.unique(np.where(pred>self.slip)[0])
        # 最後1400年間は引く必要がある
        pYear = pYear[pYear<8000-self.Year]
        # 真値の地震年数
        gnYear = np.where(gt[:,0]>self.slip)[0]
        gtnYear = np.where(gt[:,1]>self.slip)[0]
        gtYear = np.where(gt[:,2]>self.slip)[0]
        # ----
        
        flag = False
        # 予測した地震の回数分ずらしていく
        # for sYear in pYear:
        for sYear in np.arange(8000-self.Year): 
            # 予測した地震の年数 + 1400
            eYear = sYear + self.Year
            
            # 予測した地震年数 
            pnYear = np.where(pred[sYear:eYear,0]>self.slip)[0]
            ptnYear = np.where(pred[sYear:eYear,1]>self.slip)[0]
            ptYear = np.where(pred[sYear:eYear,2]>self.slip)[0]
            
            # gaussian distance for year of gt - year of pred (gYears.shape, pred.shape)
            ndist = self.gauss(gnYear,pnYear.T,sigma)
            tndist = self.gauss(gtnYear,ptnYear.T,sigma)
            tdist = self.gauss(gtYear,ptYear.T,sigma)
            
            # 予測誤差の合計, 回数で割ると当てずっぽうが小さくなる
            nYearError = sum(ndist.max(1)/pnYear.shape[0])
            tnYearError = sum(tndist.max(1)/ptnYear.shape[0])
            tYearError = sum(tdist.max(1)/ptYear.shape[0])
            
            # 予測誤差の合計 (all cells)
            yearError = nYearError + tnYearError + tYearError 
            
            if not flag:
                yearErrors = yearError
                nYearErrors = nYearError
                tnYearErrors = tnYearError
                tYearErrors = tYearError
                flag = True
            else:
                yearErrors = np.hstack([yearErrors,yearError])
                nYearErrors = np.hstack([nYearErrors,nYearError])
                tnYearErrors = np.hstack([tnYearErrors,tnYearError])
                tYearErrors = np.hstack([tYearErrors,tYearError])
        
        # 最小誤差開始修了年数(1400年)取得
        sInd = np.argmax(yearErrors)
        eInd = sInd + self.Year
        #pdb.set_trace()
        print(f"開始年:{sInd}\n")
        print("----")
        
        # 最小誤差確率　nankai,tonankai,tokai,allcell
        self.minNankai = nYearErrors[sInd]
        self.minTonankai = tnYearErrors[sInd]
        self.minTokai = tYearErrors[sInd]
        self.minError = yearErrors[sInd]
        
        # plot of result
        self.Plot(gt,pred,sInd=sInd,eInd=eInd,Year=self.Year,label=label,savePath=savePath,isPlot=isPlot)
        
        return self.minError,self.minNankai,self.minTonankai,self.minTokai
    #--------------------------
        
    
    #--------------------------
    def gauss(self,gtY,predY,sigma=0):
        
        # predict matrix for matching times of gt eq.
        predYs = predY.repeat(gtY.shape[0],0).reshape(-1,gtY.shape[0])
        # gt var.
        gtYs = gtY.repeat(predY.shape[0],0).reshape(-1,predY.shape[0])
        
        gauss = np.exp(-(gtYs - predYs.T)**2/(2*sigma**2))
        
        return gauss
    #--------------------------
        
    #--------------------------
    def Plot(self,gt,pred,sInd=0,eInd=0,Year=0,label=0,savePath="none",isPlot=False):
        if isPlot:
            #cnt = 4368
            #cnt = 588
            pred[np.where(pred>self.slip)] = 30
            
            #pdb.set_trace()
            # gt & pred 1枚ずつ重ねる, 1400year
            sns.set()
            fig, figInds = plt.subplots(nrows=3, sharex=True)
            for figInd in np.arange(len(figInds)):
                figInds[figInd].plot(np.arange(Year),gt[:,figInd],color='#ff7f00')
                figInds[figInd].plot(np.arange(Year),pred[sInd:eInd,figInd])
        
            fig.suptitle(f"nkb:{np.round(self.B[self.nI],6)} tnkb:{np.round(self.B[self.tnI],6)} tkb:{np.round(self.B[self.tI],6)} {sInd+2000}start \n {np.round(self.minError,6)} nk:{np.round(self.minNankai,6)} tnk:{np.round(self.minTonankai,6)} tk:{np.round(self.minTokai,6)}")
            #plt.legend()
            
            plt.savefig(os.path.join(savePath,f"{label}.png"))
            plt.close()
            
            
            """
            # ---- param b ---- #     
            sns.set()
            for i in np.arange(3):
                cellName = ["nk","tnk","tk"][i]
                #plt.plot(y[:,i],marker=".",color="coral",linestyle="None",label="ATR-Nets")
                sns.distplot(y[:,i],kde=False,label="OR")
                plt.title(f"Sample of paramter b in {cellName}")
                plt.legend()
                plt.savefig(f"OR_{cellName}.png")
                plt.close()
            # ----------------- #
            
            
            # label list (str) -> label array (int)
            labels = np.array(labelNum).astype(int)
            # select all paramb based on labelNum
            labelInd = np.where(labels==labelMode)
            
            # select predict & ground truth slip velocity & fileID  based on labels
            # shape=[Num. of labelInd,1400,cell(3)] all the same
            predY = allpredY[labelInd]
            trueY = tfiles[labelInd]
            #pdb.set_trace()
            fileID = [fileNum[l] for l in labelInd[0]]
            # zero matrix shape=[8000,cell(=3)]
            zeroGT = np.zeros([predY.shape[0],predY.shape[1],predY.shape[2]])
            # gt year 1400 -> 8000
            zeroGT[:,sInd:eInd] = trueY
            # ---------------------- Plot --------------------------------------- #
            #fig = plt.figure()
            # 8000 year var
            sns.set()
            for num in np.arange(zeroGT.shape[0]):
                # one gt
                gtY = zeroGT[num]
                #pdb.set_trace()
                fig, figInds = plt.subplots(nrows=3, sharex=True)
                for figInd in np.arange(len(figInds)):
                    # gt.shape=[1400,3]
                    figInds[figInd].plot(np.arange(8000),gtY[:,figInd],color='#ff7f00')
                    # pred.shape=[8000,3]
                    figInds[figInd].plot(np.arange(8000),predY[0,:,figInd])
        
                fig.suptitle(f"nankai:{np.round(paramB[2],6)} tonankai:{np.round(paramB[4],6)} tokai:{np.round(paramB[5],6)} {sInd+2000}start")
                
                plt.savefig(os.path.join(savePath,"one8000",f"{fileID[num]}_all_{labelMode}.png"))
                plt.close()
            
            #pdb.set_trace()
            # ----------------------------------------------------------------------- #
            # ----------------------- all Plot -------------------------------------- #
            sns.set()
            #plt.hold(True);
            for num in np.arange(zeroGT.shape[0]):
                # one gt
                gtY = zeroGT[num]
                
                fig, figInds = plt.subplots(nrows=3, sharex=True)
                for figInd in np.arange(len(figInds)):
                    # gt.shape=[1400,3]
                    figInds[figInd].plot(np.arange(8000),gtY[:,figInd],color='#ff7f00')
                    # pred.shape=[8000,3]
                    figInds[figInd].plot(np.arange(8000),predY[0,:,figInd])
        
            fig.suptitle(f"nankai:{np.round(paramB[2],6)} tonankai:{np.round(paramB[4],6)} tokai:{np.round(paramB[5],6)}")
            plt.savefig(os.path.join(savePath,"all8000",f"all_{labelMode}.png"))
            plt.close()
            # ----------------------------------------------------------------------- #
            """
            
# -------------------------    
if __name__ == "__main__":
    
    isPlot = False
    
    # ---- Command ---- #
    # number of file 0(OR), 20 or 200 or 500 or 1000
    fileMode = int(sys.argv[1])
    # ----------------- #
    
    # ---- path ---- #
    if fileMode == 0:
        fileMode = "OR"
        
    logsPath = f"evalML{fileMode}"
    saveImgPath = f"images{fileMode}"
    saveTxtPath = f"error{fileMode}"
    
    visualPath = 'visualization' 
    dirlogsPath = "evalMLs"
    savefname = "paramb" 
    
    fname = '*.txt'
    #fname = '*.pkl'
    
    # for making simulated log file bat
    featureFile = "featureV.bat"
    lockFile = "LockCompare.txt"
    
    filePath = os.path.join(dirlogsPath,logsPath,fname)
    filePath2 = os.path.join(saveTxtPath,"errors*")
    dirPath = os.path.dirname(filePath)
    
    files = glob.glob(filePath)
    files2 = glob.glob(filePath2)
    # --------------- #
    
    # ---- parameters ---- #
    # ファイルに使う
    ni,ti,tni = 1,2,3
    sleepTime = 3
    # 何個データを取得するか
    eNum = 5
    # -------------------- #
    
    # ---- loading files ---- #
    sortfiles = []
    # files:通し番号のデータ,sortfiles:自然順のデータ
    for path in natsorted(files):
        sortfiles.append(path)
   
    # 真の南海トラフ巨大地震履歴
    with open(os.path.join("nankairirekifeature","nankairireki.pkl"), "rb") as fp:
        tfiles = pickle.load(fp)
    # ----------------------- #
    
    # ----------------------- #
    flag = False
    #for fID in np.arange(len(files)):
    for fID in [1,190]:
        
        file = [s for s in sortfiles if f"first_{fID}_" in s][0].split("\\")[2]
        print('reading',file)
        
        # ---- labels ---- #
        id = file.split("_")[1]
        strParam1 = file.split("_")[2]
        strParam2 = file.split("_")[3]
        strParam3 = file.split("_")[4].split(".")[0]
        # --------------- #
        
        # gt V
        tfile = tfiles[int(id)-1]
        
        # 地震プレートモデル用のオブジェクト
        log = EarthQuakePlateModel(dirPath,file,nCell=8,nYear=10000)
        paramB = log.loadABLV()
        log.convV2YearlyData()
        
        # return minimum interval start Year, end Year, predicted (simulated eq.)
        error, nkerror, tnkerror, tkerror = log.comparedGauss(tfile,label=f"{int(id)}_{strParam1}_{strParam2}_{strParam3}",savePath=saveImgPath,isPlot=isPlot)
        
        # 最小年数を取得
        if not flag:
            errors = error
            nkerrors = nkerror
            tnkerrors = tnkerror
            tkerrors = tkerror
         
            flag = True
        else:
            errors = np.hstack([errors,error])
            nkerrors = np.hstack([nkerrors,nkerror])
            tnkerrors = np.hstack([tnkerrors,tnkerror])
            tkerrors = np.hstack([tkerrors,tkerror])
    # ※　better worseが逆かも
    betterInd = np.argsort(errors)[:eNum]
    better = errors[betterInd]
    nkbetter = nkerrors[betterInd]
    tnkbetter = tnkerrors[betterInd]
    tkbetter = tkerrors[betterInd]
    
    betters = np.round(np.hstack([betterInd,better,nkbetter,tnkbetter,tkbetter]).tolist(),6)
    
    worseInd = np.argsort(errors)[::-1][:eNum]
    worse = errors[worseInd]
    nkworse = nkerrors[worseInd]
    tnkworse = tnkerrors[worseInd]
    tkworse = tkerrors[worseInd]

    worses = np.round(np.hstack([worseInd,worse,nkworse,tnkworse,tkworse]).tolist(),6)
    
    inds = np.argsort(errors)[::-1]
    sort_errors = np.sort(errors)[::-1]
    
    #pdb.set_trace()
    # save better & worse
    #np.savetxt(os.path.join(saveTxtPath,f"worse_{sigma}.txt"),betters,fmt="%.6f")
    #np.savetxt(os.path.join(saveTxtPath,f"better_{sigma}.txt"),worses,fmt="%.6f")
    
    #np.savetxt(os.path.join(saveTxtPath,f"ind_{sigma}.txt"),inds,fmt="%.0f")
    #np.savetxt(os.path.join(saveTxtPath,f"errors_{100}.txt"),sort_errors,fmt="%.6f")
    
    print(f"年数誤差確率:{errors}")
    """
    # ---- errors hist ---- #
    data = np.loadtxt(files2[0])
    
    sns.set()
    sns.distplot(data,kde=False)
    plt.savefig(os.path.join(saveTxtPath,files2[0].split("\\")[1].split(".")[0]+".png"))
    plt.close()      
    # -------------- #
    """
    
    
        