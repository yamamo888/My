

import os
import sys
import concurrent.futures
import subprocess

import glob
import shutil
import pickle
import pdb
import time

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pylab as plt

import numpy as np

from natsort import natsorted

import makingData as myData
import ResultEnKF as myResult

# -------------------------- command argument --------------------------- #
# 0:2000年以降, 1:類似度
mimMode = int(sys.argv[1])
# gt & obs name of cell, 2 or 4 or 5 or 123
cell = int(sys.argv[2])
# noize of obs prefer to big
sigma = float(sys.argv[3])
# 5:deltaU vあり, 4:deltaU vなし or U vあり, 3: U vなし  ※ 3cellには対応していない
vMode = int(sys.argv[4])
# 0:deltaU, 1: U
featureMode = int(sys.argv[5])
# ----------------------------------------------------------------------- #

# ----------------------------- Path ------------------------------------ #
# In first ensamble file & logs file
dirPath = "logs"
# In paramHM* file
paramPath = "parFile"
# save txt
savePath = "savetxt"

# gt V
featuresPath = "nankairirekifeature"
firstEnName = "first*"
fileName = "*.txt"
K8_AV2File = "K8_AV2.txt"

# only one cell
if cell == 2 or cell == 4 or cell == 5:
    paramCSV = "Param1File.csv"
    batFile = "PyToC1.bat"
    
# multi cells
elif cell == 245:
    paramCSV= "Param3File.csv"
    batFile = "PyToC3.bat"
# ----------------------------------------------------------------------- #
        
# --------------------------- parameter --------------------------------- #

# 南海トラフ巨大地震履歴期間
gt_Year = 1400
# シミュレーションの安定した年
state_Year = 2000
# シミュレータの年数
nYear = 10000

# only one cell ---------------------------  
# select gt & obs cell, nankai(2), tonankai(4), tokai(5)
if cell == 2 or cell == 4 or cell == 5:
    # number of all param U,Uex,Th,V,b
    if vMode == 5:
        nParam = 5
    elif vMode == 4:
        nParam = 4
    elif vMode == 3:
        nParam = 3
    # number of cell
    nCell = 1
    # gt number of cell
    gt_nCell = 1

# gt cell index
if cell == 2:
    gtcell = 0
elif cell == 4:
    gtcell = 1
elif cell == 5:
    gtcell = 2
    
            
# 3 cell ----------------------------------
elif cell == 123:
    nParam = 5
    nCell = 3
    gt_nCell = 3
    # indec of each cell (gt)
    ntI,tntI,ttI = 0,1,2
    # index of each cell (simulation var)
    nI,tnI,tI = 2,4,5

# ※ 小さすぎるから0以上にする必要がある
# slip velocity?
slip = 0
# ※
# Ut - Ut-1の目安
deltau = 1
# 観測誤差平均
mu = 0
# 観測誤差分散
small_sigma = 0.1

# reading file start & end line
Sfl = 4
Efl = 12

if vMode == 5:
    # b index after updating
    bInd = nCell * (nParam-2)
    # U index after updating (same all nCell)
    uInd = 0
    # Ut-1 index
    uxInd = 1
    # theta index after updating (same all nCell)
    thInd = 1
    # V index after updating (same al nCell)
    VInd = 2

# limit decimal
limitNum = 6
# ----------------------------------------------------------------------- #

# =============================================================================
# # 一期予測 #
# =============================================================================
# 現在の状態の推定値
def Odemetry(Xt):
     
    # one-cell ----------------------------------------------------------------
    if nCell == 1:
        
        # select parameters
        if vMode == 5:
            yU,yUex,yth,yV,paramb = Xt[:,0],Xt[:,1],Xt[:,2],Xt[:,3],Xt[:,4]
        elif vMode == 4 and featureMode == 0: # deltaU vなし
            yU,yUex,yth,paramb = Xt[:,0],Xt[:,1],Xt[:,2],Xt[:,3]
        elif vMode == 4 and featureMode == 1: # U vあり
            yU,yth,yV,paramb = Xt[:,0],Xt[:,1],Xt[:,2],Xt[:,3]
        elif vMode == 3: # U vなし
            yU,yth,paramb = Xt[:,0],Xt[:,1],Xt[:,2]
        
        # システムノイズ(時間変化しないパラメータに与える？),W:[データ数(N)]:アンサンブル平均の0.1%(正規乱数)
        # アンサンブル平均はセルごと[Cell(1),]
        West_t = np.random.normal(0,0.01*np.mean(paramb,axis=0),nCell)
        
        # parambにシステムノイズ付加(West_tをアンサンブルメンバー数に増やした)
        paramb = paramb + np.repeat(West_t,paramb.shape[0])
        
        if vMode == 5:
            # タイムステップtの予報値:(2.8)[アンサンブルメンバー数(l),Cell数分のU,V,th,paramb(8*5)]
            Xf_t = np.vstack([yU,yUex,yth,yV,paramb]).T
        elif vMode == 4 and featureMode == 0: # deltaU vなし
            # [l,8*4]
            Xf_t = np.vstack([yU,yUex,yth,paramb]).T    
        elif vMode == 4 and featureMode == 1: # U vあり
            # [l,8*4]
            Xf_t = np.vstack([yU,yth,yV,paramb]).T
        elif vMode == 3: # U vなし
            # [l,8*3]
            Xf_t = np.vstack([yU,yth,paramb]).T
    
        # タイムステップtの予報アンサンブルで標本誤差共分散行列で誤差共分散行列を近似
        Xfhat_t = np.mean(Xf_t,axis=0) #(2.9)[Cell*(U+V+th+b)]
        EPSf_t = Xf_t - Xfhat_t #(2.10) [アンサンブルメンバー数(l),Cell*(U+b+th+V)]
        
        # 予測誤差共分散行列(カルマンゲインに使用)
        Pf_t = np.cov(EPSf_t.T) #[要素数,要素数]
    # -------------------------------------------------------------------------
        
    elif nCell == 3:
        yU,yUex,yth,yV,paramb = Xt[:,:nCell],Xt[:,nCell:nCell+nCell],Xt[:,nCell+nCell:nCell+nCell+nCell],Xt[:,nCell+nCell+nCell:nCell+nCell+nCell+nCell],Xt[:,:-nCell]
        # [Cell(8),]
        West_t = np.random.normal(0,0.01*np.mean(paramb,axis=0),nCell)
        
        # parambにシステムノイズ付加(West_tをアンサンブルメンバー数に増やした)
        paramb = paramb + np.reshape(np.repeat(West_t,paramb.shape[0]),[-1,nCell])
    
        # タイムステップtの予報値:(2.8)[アンサンブルメンバー数(l),Cell数分のU,V,th,paramb(8*5)]
        Xf_t = np.concatenate([yU,yUex,yth,yV,paramb],1)
        
        # タイムステップtの予報アンサンブルで標本誤差共分散行列で誤差共分散行列を近似
        Xfhat_t = np.mean(Xf_t,axis=0) #(2.9)[Cell*(U+V+th+b)]
        EPSf_t = Xf_t - Xfhat_t #(2.10) [アンサンブルメンバー数(l),Cell*(U+b+th+V)]
        
        # 予測誤差共分散行列(カルマンゲインに使用)
        Pf_t = np.cov(EPSf_t.T) #[要素数,要素数]        

    if vMode == 5:
        return Xf_t, Pf_t, yU, yUex, yth, yV, paramb
    elif vMode == 4 and featureMode == 0: # deltaU vなし
        return Xf_t, Pf_t, yU, yUex, yth, paramb
    elif vMode == 4 and featureMode == 1: # U vあり
        return Xf_t, Pf_t, yU, yth, yV, paramb
    elif vMode == 3: # U vなし
        return Xf_t, Pf_t, yU, yth, paramb
    
# =============================================================================
#            # 予報値更新 #
# =============================================================================

#------------------------------------------------------------------------------        
# カルマンフィルタ計算
def KalmanFilter(Pf_t):
    
    # M:真のU[nCell,], N:全変数数
    M,N = gt_nCell, nCell*nParam
    # 観測誤差共分散行列 R:[M,M]
    R =  np.diag(np.array(list([sigma])*M))
    # 観測行列[M,N],
    # y:[deltaU1,deltaU2,deltaU3],x:[U1,..,UN,Uex1,..,UexN,..,B1,B2,...,BN]
    H = np.zeros([M,N]) # [3,40]
    
    if nCell == 1:
        if vMode == 5 or (vMode == 4 and featureMode == 0): # deltaU なし　U vあり
            H[0][0] = np.float(1)
            H[0][1] = np.float(-1)
        elif (vMode == 4 and featureMode == 1) or vMode == 3 : # deltaU vなし & U v なし
            H[0][0] = np.float(1)
            pdb.set_trace()
        
    if nCell == 3:
        # Ut=1,Ut-1=-1,それ以外0,deltaU作成
        H[0,nI] = np.float(1)
        H[0,nCell+nI] = np.float(-1)
        H[1,tnI] = np.float(1)
        H[1,nCell+tnI] = np.float(-1)
        H[2,tI] = np.float(1)
        H[2,nCell+tI] = np.float(-1)
    
    # カルマンフィルタ　Pf_t:[N,N]*H:[M,N]->[N,M]
    K_t = np.matmul(np.matmul(Pf_t,H.T),np.linalg.inv((np.matmul(np.matmul(H,Pf_t),H.T)+R))) #(2.13)
    
    return K_t, H 
#------------------------------------------------------------------------------
def UpData(y,paramb,Yo_t,H,Xf_t,K_t):
    """
    予報アンサンブルメンバーを観測値とデータの予測値の残差を用いて修正
    Args:
        r_t:観測誤差
    """
    # アンサンブルの数(データ数)
    lNum = paramb.shape[0]
    dNum = y.shape[0]
    
    # アンサンブル分
    flag = False
    for lInd in np.arange(lNum):
        # 観測誤差
        r_t = np.random.normal(mu,small_sigma,dNum)
        # 観測値
        Yo_t = (y + r_t).T #(2.2) [Cell(3)]
        
        #  第2項：[アンサンブル数,]
        residual = Yo_t + r_t - np.matmul(H,Xf_t[lInd,:])
        tmp = Xf_t[lInd,:] + np.matmul(K_t,residual) #(2.12)
        if not flag:
            Xa_t = tmp[np.newaxis]
            flag = True
        else:
            Xa_t = np.concatenate((Xa_t,tmp[np.newaxis]),0)
    #[アンサンブル数,M]
    return Xa_t # [Ensambles,40(8*5)]
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
if __name__ == "__main__":
     #-------------------------------------------------------------------------
                        # アンサンブルカルマンフィルタ 開始 #
     #-------------------------------------------------------------------------
     for tfID in np.arange(1):
        
        print("-----------------------------------")
        
        # ------------------------ path ------------------------------------- #
        # dirpath for each logs
        logsPath = "{}".format(tfID)
        # before dirpath
        exlogsPath = "{}".format(tfID-1)
        # fullPath
        filePath = os.path.join(dirPath,logsPath,fileName)
        # ------------------------------------------------------------------- #
        
        # ------------------------ gt path ---------------------------------- #
        # gt simulated one file fullPath
        gtfilePath = os.path.join("gtlogs",logsPath,fileName)
        # gt simulated file
        gtfiles = glob.glob(gtfilePath)[0]
        # only file name
        gtfile = os.path.basename(gtfiles)
        # ------------------------------------------------------------------- #

        # ------------- 真の南海トラフ巨大地震履歴 特徴量 (simulated) ------------ #
        
        # loading U,theta,V,B [number of data,10]
        gU,gth,gV,gB = myData.loadABLV("gtlogs",logsPath,gtfile)
        # [80000,8]
        gtyU,gtyUex,gtth,gtyV,gtyYear = myData.convV2YearlyData(gU,gth,gV,nYear,cell=cell,cnt=0)
        
        # ※ 特徴量 ------------------------------------------------------------
        # 地震が起きた年数を特徴量がUの時も使いたいから置いておいてください。
        # deltaU = Ut - Ut-1
        gtdeltaU = np.vstack([gtyU[1:] - gtyU[:-1],np.zeros(8)])
        # [8000,3]
        gtdeltaU = np.hstack([gtdeltaU[:,2,np.newaxis],gtdeltaU[:,4,np.newaxis],gtdeltaU[:,5,np.newaxis]])
        # [1400,3]
        gtdeltaU = gtdeltaU[:1400,:]
        
        gtU = gtyU - gtyUex
        # [1400,3]
        gtU = gtU[:1400,:]
        # ------------------------------------------------------------------- #
            
        gtth = gtth[:1400,:]
        gtyV = gtyV[:1400,:]
        gtyU = gtyU[:1400,:]
        
        # gt eq. in all cell
        gtJ = np.unique(np.where(gtyV>0)[0])
        # ------------------------------------------------------------------- #         

        # ------------------------------------------------------------------- #        
        # ----------- 同化 (期間 : 真の南海トラフ履歴に合わせた1400年間) ----------- #
        # ------------------------------------------------------------------- #
        plotparams = [] # parameters for plot
        plotyears = [] # years for plot
        plotgt = [] # gt for plot
        cnt,iS = 0,0
        while True:
            
            # ------ file 読み込み, 最初は初期アンサンブル読み取り (logs\\*) ------- # 
            files = glob.glob(filePath)
            
            if iS > 0: # not first ensemble
                files = [s for s in files if "log_{}_".format(iS-1) in s]
                
                if files == []:
                    print("---- ALL KILL!! ----\n")
                else:
                    print(f"==== 【{iS} times】 ====")
            # --------------------------------------------------------------- #
           
            # make next logs directory
            if not os.path.exists(os.path.join(dirPath,"{}".format(iS))):
                os.mkdir(os.path.join(dirPath,"{}".format(iS)))
               
            # =================== Ensemble member 作成 ====================== #
            Xt = np.zeros([len(files),nCell*nParam])
            fcnt = 0
            jisins,emptybox = [],[]
            flag,flag1,flag2,flag3,flag4 = False,False,False,False,False
            
            for fID in np.arange(len(files)):
                # fID : Ensamble member
                print('reading',files[fID])
                
                # --------------- loading ansanble data --------------------- #
                file = os.path.basename(files[fID])
                logFullPath = os.path.join(dirPath,logsPath,file)
                data = open(logFullPath).readlines()
                # ----------------------------------------------------------- #
                
                # ----------------------- Error ----------------------------- #
                if data == []:
                    myData.Empty(logFullPath) # 発散したデータ(logs)を捨てる
                # 〃
                elif "value of RTOL" in data[-1]:
                    myData.Empty(logFullPath)
                # ----------------------------------------------------------- #
                
                else:
                    # loading U,theta,V,B [number of data,10]
                    U,th,V,B = myData.loadABLV(dirPath,logsPath,file)
                    
                    # --------------------- Error --------------------------- #
                    myData.Negative(V,logFullPath,fID) #すべり速度マイナス判定
                    # ------------------------------------------------------- #
                    
                    # - 1.1 SpinUP(match first time gt eq & simulation eq.) - #
                    if iS == 0:
                        print("---- Spin UP Start! ----")
                        
                        # Ut,tht,Vt, [10000,8] yYear: eq. of years
                        yU,yUex,yth,yV,yYear = myData.convV2YearlyData(U,th,V,nYear,cell=cell,cnt=iS)
                        # U,th,V [1400,8], yUex [8,]
                        yU,yUex,yth,yV = myData.MinErrorNankai(gtdeltaU,yU,yUex,yth,yV,cell=cell,mimMode=mimMode)
                        
                        # ※ 特徴量 -------------------------------------------
                        if featureMode == 0:
                            # deltaU = Ut - Ut-1 [1400,8]
                            deltaU = np.vstack([yU[1:] - yU[:-1],np.zeros(8)])
                        
                        elif featureMode == 1:
                            # [1400,3] for plot
                            predyU = np.hstack([yU[:,2,np.newaxis],yU[:,4,np.newaxis],yU[:,5,np.newaxis]])
                        # --------------------------------------------------- #
                        
                        # --------------------------------------------------- #
                        """
                        # ------------- plot gt & pred ---------------------- #
                        print("---- start plot! ----")
                        fig, figInds = plt.subplots(nrows=3, sharex=True)
                        for figInd in np.arange(len(figInds)):
                            figInds[figInd].plot(np.arange(1400),gtyU[:,figInd],color='#ff7f00')
                            figInds[figInd].plot(np.arange(1400),predyU[:,figInd],color='skyblue')
                            
                        plt.savefig(os.path.join("images",f"U_{fID}.png"))
                        plt.close()
                        
                        fig, figInds = plt.subplots(nrows=3, sharex=True)
                        for figInd in np.arange(len(figInds)):
                            figInds[figInd].plot(np.arange(1400),gtdeltaU[:,figInd],color='#ff7f00')
                            # ※ はじめだけyUexが考慮されていない
                            figInds[figInd].plot(np.arange(1400),deltaU[:,figInd],color='skyblue')
                            
                        plt.savefig(os.path.join("images",f"deltaU_{fID}.png"))
                        plt.close()
                        
                        fig, figInds = plt.subplots(nrows=3, sharex=True)
                        for figInd in np.arange(len(figInds)):
                            figInds[figInd].plot(np.arange(1400),gtyV[:,figInd],color='#ff7f00')
                            figInds[figInd].plot(np.arange(1400),yV[:,figInd],color='skyblue')
                            
                        plt.savefig(os.path.join("images",f"V_{fID}.png"))
                        plt.close()
                        # --------------------------------------------------- #
                        """
                        # ------------- gt & predict select cell ------------ #
                        if cell == 2 or cell == 4 or cell == 5:
                            
                            # ※　本当に地震が起きた時? Uの時もこの指標使う?
                            # first eq. year (During 0-1400 year) 
                            #　jInd = np.where(yV[:,cell]>slip)[0][0]
                            jInd = np.where(deltaU[:,gtcell]>deltau)[0][0]
                            
                            # ------------- predict feature ----------------- #
                            if vMode == 0: # deltaU Vあり
                                # ※
                                # Vt one kalman, shape=[nParam,]
                                Vt = np.hstack([yU[jInd,cell],yUex[cell],yth[jInd,cell],yV[jInd,cell],B[cell]])
                                # Vt all cell, shape=[8,5]
                                Vt_all = np.hstack([yU[jInd][:,np.newaxis],yUex[:,np.newaxis],yth[jInd][:,np.newaxis],yV[jInd][:,np.newaxis],B[:,np.newaxis]])
                            elif vMode == 4 and featureMode == 0: # deltaU vなし
                                Vt = np.hstack([yU[jInd,cell],yUex[cell],yth[jInd,cell],B[cell]])
                                Vt_all = np.hstack([yU[jInd][:,np.newaxis],yUex[:,np.newaxis],yth[jInd][:,np.newaxis],B[:,np.newaxis]])
                            elif vMode == 4 and featureMode == 1: # U vあり                        
                                Vt = np.hstack([yU[jInd,cell],yth[jInd,cell],yV[jInd,cell],B[cell]])
                                Vt_all = np.hstack([yU[jInd][:,np.newaxis],yth[jInd][:,np.newaxis],yV[jInd][:,np.newaxis],B[:,np.newaxis]])
                            elif vMode == 3: # U vなし
                                Vt = np.hstack([yU[jInd,cell],yth[jInd,cell],B[cell]])
                                Vt_all = np.hstack([yU[jInd][:,np.newaxis],yth[jInd][:,np.newaxis],B[:,np.newaxis]])
                            # ----------------------------------------------- #
                            
                            # ------------- predict label ------------------- #
                            # jisin or non jisin -> True or False, shape=[8,]
                            tmpJ = [j > slip for j in yV[jInd,:]] 
                            # True, False -> 1,0, shape=[8,]
                            jisins = [1 if jisin else 0 for jisin in tmpJ]
                            # ----------------------------------------------- #
                            
                            # -------------------- gt ----------------------- #
                            if featureMode == 0:
                                # [1,]
                                TrueU = gtdeltaU[np.where(gtdeltaU[:,gtcell]>deltau)[0][0],gtcell]
                            elif featureMode == 1:
                                # ※ この指標は正しいのか
                                TrueU = gtU[np.where(gtdeltaU[:,gtcell]>deltau)[0][0],gtcell]
                            # ----------------------------------------------- #
                            
                        # one Ensamble -> Ensambles
                        if not flag1:
                            Xt = Vt[np.newaxis,:]
                            Xt_all = Vt_all[np.newaxis,:]
                            aInds = jInd + state_Year
                            jLabel = jisins 
                            flag1 = True
                        else:
                            # [アンサンブル数(N),Ut,Ut-1,th,V,b(5)] * cell
                            Xt = np.concatenate((Xt,Vt[np.newaxis,:]),0)
                            # 8 cell ver, shape=[ensemble,8,params(=5)]
                            Xt_all = np.concatenate((Xt_all,Vt_all[np.newaxis,:]),0)
                            aInds = np.vstack([aInds, jInd + state_Year])
                            # [アンサンブル数(N), Cell(5)]
                            jLabel = np.vstack([jLabel,jisins])
                    # ------------------------------------------------------- #
                    
                    # - 1.2 最初以外は、同化年数以降の年を格納して、同化年数を決める - #
                    else:
                        # shape=[10000,8]
                        yU, yth, yV, yYear = myData.convV2YearlyData(U,th,V,nYear,cell=cell,cnt=iS)
                        
                        if vMode == 5 or (vMode == 4 and featureMode == 1):
                            # Vt one kalman all year [u,th,v,10000,8]
                            uthv_all = np.concatenate([yU[:,:,np.newaxis],yth[:,:,np.newaxis],yV[:,:,np.newaxis]],2)
                        elif (vMode == 4 and featureMode == 0) or vMode == 3: # deltaU vなし
                            uthv_all = np.concatenate([yU[:,:,np.newaxis],yth[:,:,np.newaxis]],2)
                            
                        # eq. simulaion time step t
                        if not flag2:
                            uthvs_all = uthv_all[np.newaxis]
                            bs = B
                            pJ = yYear[0]
                            flag2 = True
                        else:
                            # uthvs.shape=[ensenbles,u/th/v,10000(year),cells]
                            uthvs_all = np.vstack([uthvs_all,uthv_all[np.newaxis]])
                            bs = np.vstack([bs,B])
                            # first eq. year
                            pJ = np.vstack([pJ,yYear[0]])
                            
                        fcnt += 1
                        # last file (save all Ensemble time-step)
                        if fcnt == len(files):
                            # concate gt eq. year
                            pJ = np.hstack([pJ.reshape(-1,),gtJ+state_Year])
                            # minimum　assimulation year
                            jInd = int(np.sort(pJ[pJ>np.max(aInds)])[0])
                            # all same assimulation year (次のシミュレーションのために必要)
                            aInds = np.repeat(jInd,len(files))[:,np.newaxis]
                            
                            if cell == 2 or cell == 4 or cell == 5:
                                # ※
                                # gt deltaU or U, shape=()   
                                if vMode == 5 or (vMode == 4 and featureMode == 0):
                                    # [ensamble,params(U/Uex/Th/V/b=5)] nextUexは下で実装
                                    Xt = np.concatenate([uthvs_all[:,jInd,cell,uInd,np.newaxis],nextUex[:,cell,np.newaxis],uthvs_all[:,jInd,cell,thInd:],bs[:,cell,np.newaxis]],1)
                                    TrueU = gtdeltaU[jInd-state_Year,gtcell]
                                    # shape=[ensamble,cell(=8),params(U/Uex/Th/V/b=5)]
                                    Xt_all = np.concatenate([uthvs_all[:,jInd,:,uInd,np.newaxis],nextUex[:,:,np.newaxis],uthvs_all[:,jInd,:,thInd:],bs[:,:,np.newaxis]],2)
                                        
                                elif (vMode == 4 and featureMode == 1) or vMode == 3: # deltaU vなし
                                    Xt = np.concatenate([uthvs_all[:,jInd,cell,uInd,np.newaxis],uthvs_all[:,jInd,cell,thInd:],bs[:,cell,np.newaxis]],1)
                                    Xt_all = np.concatenate([uthvs_all[:,jInd,:,uInd,np.newaxis],uthvs_all[:,jInd,:,thInd:],bs[:,:,np.newaxis]],2)
                                    TrueU = gtU[jInd-state_Year,gtcell]
                            
                            for eID in np.arange(Xt_all.shape[0]):
                                yV = Xt_all[eID,:,-2]
                                # jisin or non jisin -> True or False list of 8cell
                                tmpJ = [j > slip for j in yV] 
                                # True, False -> 1,0
                                jisins = [1 if jisin else 0 for jisin in tmpJ]
                                # one Ensamble -> Ensambles
                                if not flag3:
                                    jLabel = jisins 
                                    flag3 = True
                                else:
                                    # [アンサンブル数(N), Cell(5)]
                                    jLabel = np.vstack([jLabel,jisins])
                    # ------------------------------------------------------- #

            # =============================================================== #
            
            # ---------------------------- Error ---------------------------- #
            if files == emptybox: # すべてのファイルが死んだとき
                break
            # --------------------------------------------------------------- # 
                
            # -------------------------- EnKF 計算 -------------------------- #
            # 2.次ステップまでの時間積分・システムノイズ付加
            # 3.標本誤差共分散行列(タイムステップt)
            # IN:Ensambles,[number of Ensambles,parameters(nCell*5(Ut,Ut-1,th,V,B))]
            if vMode == 5:
                Xf_t, Pf_t, yU, yexU, yth, yV, noizeb = Odemetry(Xt)
            elif vMode == 4 and featureMode == 0: # deltaU vなし
                Xf_t, Pf_t, yU, yexU, yth, noizeb = Odemetry(Xt)
            elif vMode == 4 and featureMode == 1: # U vあり
                Xf_t, Pf_t, yU, yexU, yth, noizeb = Odemetry(Xt)
            elif vMode == 3: # U vなし
                Xf_t, Pf_t, yU, yth, noizeb = Odemetry(Xt)

            # 4.カルマンゲイン
            K_t, H = KalmanFilter(Pf_t)
            # 5.予報アンサンブル更新
            Xal_t = UpData(np.array([TrueU]),noizeb,TrueU,H,Xf_t,K_t)
            # --------------------------------------------------------------- # 
            
            # ------------------ save Kalman & parameters ------------------- #
            # K_t [num. of param,1], Xt & Xal_t [ensembles,num. of params]
            KUUexThV = np.reshape(np.append(K_t.T,np.append(Xt,Xal_t)),[-1,nParam])
            plotparams.append(KUUexThV)
            plotyears.append(jInd) # 最初は同化年数が異なるため ignore ok!
            plotgt.append(TrueU)
            # --------------------------------------------------------------- #
            
            # --------------- save txt Kalman & parameters ------------------ #
            np.savetxt(os.path.join(savePath,"0",f"EnKFparams_{iS}.txt"),plotparams[iS],"%.5f")
            np.savetxt(os.path.join(savePath,"0",f"EnKFyears_{iS}.txt"),np.array([plotyears[iS]]),"%.0f")
            np.savetxt(os.path.join(savePath,"0",f"EnKFTrueU_{iS}.txt"),np.array([plotgt[iS]]),"%.5f")
            # --------------------------------------------------------------- #
            
            FLAG = False
            rmInds = [] # マイナス値が入ったインデックス
            rmlogs = [] # シミュレーションできなかったloファイル
            for lNum in np.arange(Xal_t.shape[0]):
                
                # Ut-1を取り除く, parfileHM*に必要がないから, [4(parameter)*8(cell),]
                Xa_t = np.hstack([Xal_t[:,:nCell],Xal_t[:,nCell+nCell:]])[lNum,:]
                UThV = np.reshape(Xa_t[:bInd],[-1,nCell]) 
                # yUexは2000年以前のyU
                startyU = Xal_t[:,nCell:nCell+nCell]
                
                # Ut-1を取り除く, all cell ver., [8,4]
                Xa_t_all = np.concatenate([Xt_all[:,:,:nCell],Xt_all[:,:,nCell+nCell:]],2)[lNum,:]
                
                # ※ deltaUのときは2000年以前のUを足し、元のスケールに戻してシミュレーション
                if featureMode == 1:
                    rescaleU = Xa_t_all[:,uInd] + startyU[:,lNum]
                    Xa_t_all[:,uInd] = rescaleU
                    
                # Get paramb, 小数第7位四捨五入
                updateB = np.round(Xa_t[bInd:],limitNum) # [b1,..,bN].T
                # Get U
                updateU = np.round(UThV[uInd],limitNum)
                # Get theta, [cell,]
                updateTh = np.round(UThV[thInd],limitNum)
                
                print("----")
                print(f"TrueU:{np.round(TrueU,6)}\n")
                print("before,update:\n")
                print(f"Kalman:{K_t}\n")

                if featureMode == 0:
                    # Get V, [cell,], 小さすぎるから四捨五入なし
                    updateV = UThV[VInd]
                    print(f"B:{np.round(Xt[lNum][-1],limitNum)},{updateB}\nU:{np.round(Xt[lNum][0],limitNum)},{updateU}\nUex:{np.round(Xt[lNum][1],limitNum)}\ntheta:{np.round(Xt[lNum][2],limitNum)},{updateTh}\nV:{Xt[lNum][-2]},{updateV}\n")
                elif featureMode == 1: 
                    print(f"B:{np.round(Xt[lNum][-1],limitNum)},{updateB}\nU:{np.round(Xt[lNum][0],limitNum)},{updateU}\nUex:{np.round(Xt[lNum][1],limitNum)}\ntheta:{np.round(Xt[lNum][2],limitNum)},{updateTh}\n")
                
                if cell == 2 or cell == 4 or cell == 5:
                    # 更新したセルのパラメータとそのほかをconcat
                    Xa_t_all[cell] = Xa_t
                    # save index of minus parameters
                    if np.any(Xa_t_all<0):
                        rmInds.append(lNum)
                
                # ----------------------- Xt-1 作成手順 ---------------------- #
                # 1 parfileをアンサンブル分作成
                # 2 batchファイルでファイル番号(Label etc...)受け渡し
                # 2.5 エラー処理：マイナス値が出たアンサンブルメンバーは削除
                # 3 受け渡された番号のparfileを読み取りsimulation実行
                # ----------------------------------------------------------- #
                
                # ========================= 1 =============================== #
                # defaultparfileファイルを読み込む
                with open("parfileHM031def.txt","r") as fp:
                    alllines = fp.readlines()
                # parfileHM031の改行コード削除
                alllines = [alllines[i].strip().split(",") for i in np.arange(len(alllines))]
                # 計算ステップ指定 (各データで異なる)
                alllines[0][0] = str(int(aInds[lNum]) + 1)
                alllines[0][1] = str(1500 + state_Year)
                
                # パラメータ設定行抽出
                lines = alllines[Sfl:Efl]
                # Input b,U,th,V in all cell
                for nl in np.arange(len(lines)): # 8 cell times
                    # b, U, theta, V
                    inlines = lines[nl]
                    pdb.set_trace()
                    outlines = Xa_t_all[nl]
                    # v あり        
                    if featureMode == 0:    
                        inlines[1],inlines[-3],inlines[-2],inlines[-1] = str(np.round(outlines[-1],limitNum)),str(np.round(outlines[0],limitNum)),str(np.round(outlines[1],limitNum)),str(outlines[2])
                    # v なし
                    elif featureMode == 1: 
                        inlines[1],inlines[-3],inlines[-2] = str(np.round(outlines[-1],limitNum)),str(np.round(outlines[0],limitNum)),str(np.round(outlines[1],limitNum))

                # Save parfileHM031 -> parfileHM0*
                parFilePath = os.path.join(paramPath,f"{tfID}",f"parfileHM{iS}_{lNum}.txt")
                # 書式を元に戻す
                alllines = [','.join(alllines[i]) + '\n' for i in np.arange(len(alllines))]
                with open(parFilePath,"w") as fp:
                    for line in alllines:
                        fp.write(line)
                
                if not FLAG:
                    FLAG=True
                    # iS: ensamble itentical
                    # lNum: ensamble member
                    # tfID: 0-256(historical of eq.(=directory))
                    parNum = np.hstack([iS,lNum,tfID])
                else:
                    parNum = np.vstack([parNum,np.hstack([iS,lNum,tfID])])                    
                # =========================================================== #
            
            # ========================== 2.5 ================================ #
            inds = np.ones(jLabel.shape[0],dtype=bool)
            inds[rmInds] = False
            jLabel = jLabel[inds]
            parNum = parNum[inds]
            # =============================================================== #
            
            # next Uex, 次のシミュレーションのUexに使う
            nextUex = np.round(Xt_all[:,:,uInd],limitNum)[inds]
            
            # ========================== 2 ================================== #
            if cell == 2 or cell == 4 or cell == 5:
                cLabel = jLabel[:,cell][:,np.newaxis]
                # concat jisin or non-jisin & year & ensamble member
                fileLabel = np.hstack([cLabel,parNum,jLabel])
            elif cell == 245:
                fileLabel = np.hstack([jLabel,parNum])

            # parFile番号格納
            data = np.c_[fileLabel]
            np.savetxt(paramCSV,data,delimiter=",",fmt="%.0f")
            # =============================================================== #
                    
            # ========================== 3 ================================== #
            # all parHM* files
            parallfiles = glob.glob(os.path.join(paramPath,str(tfID),f"*HM{iS}_*.txt")) 
            
            # get file index not in rm files
            # inds: True or False (rmInds==False)
            # Xt_all.shape=[ensambles,cell,paramters]
            fInds = inds.astype(int) # 0 or 1            
            parallfiles = (np.array(parallfiles)[fInds==1]).tolist()
            # =============================================================== #
            
            # ========================== 4 ================================== #
            
            # ---- Making Lock.txt 
            lockPath = "Lock.txt"
            lock = str(1)
            with open(lockPath,"w") as fp:
                fp.write(lock)
            # --------------------
            
            os.system(batFile)
            cnt += 1
                
            sleepTime = 3
            # lockファイル作成時は停止
            while True:
                time.sleep(sleepTime)
                if os.path.exists(lockPath)==False:
                    break
            # =============================================================== #
            
            # 通し番号を１つ増やす 0回目, １回目 ...
            iS += 1
            # --------------------------------------------------------------- # 