

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

# -------------------------- command argument --------------------------- #
# 0:EnKF, 1:PF
mode = int(sys.argv[1])
# 0:2000年以降, 1:類似度
mimMode = int(sys.argv[2])
# gt & obs name of cell, 2 or 4 or 5 or 123
cell = int(sys.argv[3])
# noize of obs prefer to big
sigma = float(sys.argv[4])

# ----------------------------------------------------------------------- #

# ----------------------------- Path ------------------------------------ #
# In first ensamble file & logs file
dirPath = "logs"
# In paramHM* file
paramPath = "parFile"
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

isWindows = True

# 南海トラフ巨大地震履歴期間
gt_Year = 1400
# シミュレーションの安定した年
state_Year = 2000
# シミュレータの年数
nYear = 10000

#初めに地震発生した年(真値)
#sYear = np.where(gtU>0)[0][0]

# only one cell ---------------------------  
# select gt & obs cell, nankai(2), tonankai(4), tokai(5)
if cell == 2 or cell == 4 or cell == 5:
    # number of all param U,Uex,Th,V,b
    nParam = 5
    # number of cell
    nCell = 1
    # gt number of cell
    gt_nCell = 1
            
# 3 cell ----------------------------------
elif cell == 123:
    nParam = 5
    nCell = 3
    gt_nCell = 3
    # indec of each cell (gt)
    ntI,tntI,ttI = 0,1,2
    # index of each cell (simulation var)
    nI,tnI,tI = 2,4,5

# slip velocity?
slip = 0
# 観測誤差平均
mu = 0
# 観測誤差分散
small_sigma = 0.1

# reading file start & end line
Sfl = 4
Efl = 12

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
    #pdb.set_trace()        
    # one-cell
    if nCell == 1:
        yU,yUex,yth,yV,paramb = Xt[:,0],Xt[:,1],Xt[:,2],Xt[:,3],Xt[:,4]
    
        # システムノイズ(時間変化しないパラメータに与える？),W:[データ数(N)]:アンサンブル平均の0.1%(正規乱数)
        # アンサンブル平均はセルごと[Cell(1),]
        West_t = np.random.normal(0,0.01*np.mean(paramb,axis=0),nCell)
        
        # parambにシステムノイズ付加(West_tをアンサンブルメンバー数に増やした)
        paramb = paramb + np.repeat(West_t,paramb.shape[0])
    
        # タイムステップtの予報値:(2.8)[アンサンブルメンバー数(l),Cell数分のU,V,th,paramb(8*5)]
        Xf_t = np.vstack([yU,yUex,yth,yV,paramb]).T
    
        # タイムステップtの予報アンサンブルで標本誤差共分散行列で誤差共分散行列を近似
        Xfhat_t = np.mean(Xf_t,axis=0) #(2.9)[Cell*(U+V+th+b)]
        EPSf_t = Xf_t - Xfhat_t #(2.10) [アンサンブルメンバー数(l),Cell*(U+b+th+V)]
        
        # 予測誤差共分散行列(カルマンゲインに使用)
        Pf_t = np.cov(EPSf_t.T) #[要素数,要素数]
        #pdb.set_trace()
        
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

    return Xf_t, Pf_t, yU, yUex, yth, yV, paramb
    
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
    #pdb.set_trace()    
    
    if nCell == 1:
        H[0][0] = np.float(1)
        H[0][1] = np.float(-1)
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
    #pdb.set_trace()
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
    #pdb.set_trace()
    # アンサンブル分
    flag = False
    for lInd in np.arange(lNum):
        # 観測誤差
        r_t = np.random.normal(mu,small_sigma,dNum)
        # 観測値
        Yo_t = (y + r_t).T #(2.2) [Cell(3)]
        #print("GT: {}".format(y))        
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
    #--------------------------------------------------------------------------
                        # アンサンブルカルマンフィルタ 開始 #
    #--------------------------------------------------------------------------
    # 1 確実に起きた地震
    # 190 全て起きた地震
    #for tfID in np.arange(0,256): # 0始まり
    for tfID in [0,1,190]:
        
        print("-----------------------------------")
        print("------ {} historical eq data ------".format(tfID))
        
        # ------------------------ path ------------------------------------- #
        # dirpath for each logs
        logsPath = "{}".format(tfID)
        # before dirpath
        exlogsPath = "{}".format(tfID-1)
        # fullPath
        filePath = os.path.join(dirPath,logsPath,fileName)
        # ------------------------------------------------------------------- #
        
        # ------------------------------------------------------------------- #
        # 次のディレクトリに初期アンサンブルを移動
        if tfID > 0 and os.path.exists(os.path.dirname(filePath)):
            firstEnFiles = glob.glob(os.path.join(dirPath,exlogsPath,firstEnName))
            newPath = os.path.join(os.path.dirname(filePath))
            
            [shutil.copy2(firstEnFiles[i],newPath) for i in np.arange(len(firstEnFiles))]
        # ------------------------------------------------------------------- #
        
        # ---------------- 真の南海トラフ巨大地震履歴 V------------------------- #
        with open(os.path.join(featuresPath,"nankairireki.pkl"), "rb") as fp:
            nkfiles = pickle.load(fp)
        # 一番地震が発生している slip-velocity V, shape=[Year(1400),Cell(3)]
        # 発生年数取得 & slip velocity
        gtV = nkfiles[tfID,:,:]
        
        with open(os.path.join(featuresPath,"NankaiTimeModel.pkl"), "rb") as fp:
            tfiles = pickle.load(fp)
        
        # for Ensemble KL, U
        gtU = tfiles[tfID,:,:]
        
        # deltaU -> slip velocity 
        gtUV = np.vstack([np.zeros(3)[np.newaxis], gtU[1:,:] - gtU[:-1,:]])
        
        # gt eq. in all cell
        gtJ = np.unique(np.where(gtV>0)[0])    
        # ------------------------------------------------------------------ #
        
        # ----- 同化 (期間 -> 真の南海トラフ履歴に合わせた1400年間) ----- #
        cnt,iS = 0,0
        while True:
            
            # ------ file 読み込み, 最初は初期アンサンブル読み取り (logs\\*) ------ # 
            if isWindows:
                files = glob.glob(filePath)
            else:
                files = glob.glob(filePath)
            
            if iS > 0: # not first ensemble
                files = [s for s in files if "log_{}_".format(iS-1) in s]
                
                if files == []:
                    print("---- ALL KILL!! ----\n")
                else:
                    print("++++++++++++++++++++++")
                    print("【{} times】".format(iS))
                    # Get log files only before one year
                    
            # --------------------------------------------------------------- #
           
            # make logs directory
            if not os.path.exists(os.path.join(dirPath,"{}".format(iS))):
                os.mkdir(os.path.join(dirPath,"{}".format(iS)))
               
            # =================== Ensemble member 作成 ===========================#
            Xt = np.zeros([len(files),nCell*nParam])
            fcnt = 0
            jisins,emptybox = [],[]
            flag,flag1,flag2,flag3 = False,False,False,False
            for fID in np.arange(len(files)):
                # fID == Ensamble member
                print('reading',files[fID])
                
                file = os.path.basename(files[fID])
                logFullPath = os.path.join(dirPath,logsPath,file)
                data = open(logFullPath).readlines()
                
                # 発散したデータ(logs)を捨てる
                if data == []:
                    myData.Empty(logFullPath)
                # 〃
                elif "value of RTOL" in data[-1]:
                    myData.Empty(logFullPath)
                
                else:
                    # loading U,theta,V,B [number of data,10]
                    U,th,V,B = myData.loadABLV(dirPath,logsPath,file)
                    
                    # すべり速度マイナス判定
                    myData.Negative(V,logFullPath,fID,isWindows)
                    
                    # -- 1.1 SpinUP (= match first time gt eq & simulation eq.)-- #
                    if iS == 0:
                        print("Spin UP Start!")
                    
                        # U,th,V,地震発生年(すべて) 取得(タイムステップt-1), [10000,8]
                        # yYear: eq. of year
                        # syU: 2000年以前で差が出るU
                        yU,yUex,yth,yV,yYear,syU = myData.convV2YearlyData(U,th,V,nYear,cell=cell,cnt=iS)
                        
                        # ------------------------------------------------------- #
                        # U,th,V = [1400,8]
                        yU,yUex,yth,yV = myData.MinErrorNankai(gtV,yU,yUex,yth,yV,cell=cell,mimMode=mimMode)
                        #pdb.set_trace()
                        # 2000年目をはじめとして累積変位調整
                        yU = yU - syU
                        yUex = yUex - syU
                        # ------------------------------------------------------- #
                        
                        # select cell
                        if cell == 2 or cell == 3 or cell == 4:
                            # first eq. year (During 0-1400 year) 
                            jInd = np.where(yV[:,cell]>slip)[0][0]
                            #pdb.set_trace()
                            # Vt one kalman, shape=[nParam,]
                            Vt = np.hstack([yU[jInd,cell],yUex[cell],yth[jInd,cell],yV[jInd,cell],B[cell]])
                            # Vt all cell, shape=[8,]
                            Vt_all = np.hstack([yU[jInd][:,np.newaxis],yUex[:,np.newaxis],yth[jInd][:,np.newaxis],yV[jInd][:,np.newaxis],B[:,np.newaxis]])
                            # gt U (84 year), shape=[cell,] 0年目も考慮すべきだがU=0なので問題なし, cellの指定はシミュレーションに合わせてるので-2必要
                            TrueU = gtUV[np.where(gtUV[:,cell-2]>slip)[0][0],cell-2]
                            
                            # jisin or non jisin -> True or False, shape=[8,]
                            tmpJ = [j > slip for j in yV[jInd,:]] 
                            # セルに関係なく共通
                            # True, False -> 1,0, shape=[8,]
                            jisins = [1 if jisin else 0 for jisin in tmpJ]
                        
                        # one Ensamble -> Ensambles
                        if not flag1:
                            Xt = Vt[np.newaxis,:]
                            Xt_all = Vt_all[np.newaxis,:]
                            startyU = syU[:,np.newaxis]
                            aInds = jInd + state_Year
                            jLabel = jisins 
                            flag1 = True
                        else:
                            # [アンサンブル数(N),Ut,Ut-1,th,V,b(5)] * cell
                            Xt = np.concatenate((Xt,Vt[np.newaxis,:]),0)
                            # 8 cell ver., shape=[ensemble,8,params(=5)]
                            Xt_all = np.concatenate((Xt_all,Vt_all[np.newaxis,:]),0)
                            aInds = np.vstack([aInds, jInd + state_Year])
                            # 基準のU, shape=[8,ensemble]
                            startyU = np.hstack([startyU,syU[:,np.newaxis]])
                            # [アンサンブル数(N), Cell(5)]
                            jLabel = np.vstack([jLabel,jisins])
                    # ------------------------------------------------------- #
                    
                    # 最初以外は、同化年数以降の年を格納して、同化年数を決める
                    else:
                        # shape=[10000,8]
                        yU, yth, yV, yYear = myData.convV2YearlyData(U,th,V,nYear,cell=cell,cnt=iS)
                        
                        # Vt one kalman all year [u,th,v,10000,8]
                        uthv_all = np.concatenate([yU[:,:,np.newaxis],yth[:,:,np.newaxis],yV[:,:,np.newaxis]],2)
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
                            
                            if cell == 2 or cell == 3 or cell == 4:
                                # shape=[ensamble,params(U/Uex/Th/V/b=5)]
                                Xt = np.concatenate([uthvs_all[:,jInd,cell,uInd,np.newaxis],nextUex[:,cell,np.newaxis],uthvs_all[:,jInd,cell,thInd:],bs[:,cell,np.newaxis]],1)
                                # gt UV, shape=()
                                TrueU = gtU[jInd-state_Year,cell-2]
                                
                            # shape=[ensamble,cell(=8),params(U/Uex/Th/V/b=5)]
                            Xt_all = np.concatenate([uthvs_all[:,jInd,:,uInd,np.newaxis],nextUex[:,:,np.newaxis],uthvs_all[:,jInd,:,thInd:],bs[:,:,np.newaxis]],2)
                            
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
            
            # すべてのファイルが死んだとき
            if files == emptybox:
                break
                
            # -------------------------- EnKF 計算 -------------------------- #
            # 2.次ステップまでの時間積分・システムノイズ付加
            # 3.標本誤差共分散行列(タイムステップt)
            # IN:Ensambles,[number of Ensambles,parameters(nCell*5(Ut,Ut-1,th,V,B))]
            Xf_t, Pf_t, yU, yexU, yth, yV, noizeb = Odemetry(Xt)
            
            # 4.カルマンゲイン
            K_t, H = KalmanFilter(Pf_t)
            # 5.予報アンサンブル更新
            Xal_t = UpData(np.array([TrueU]),noizeb,TrueU,H,Xf_t,K_t)
            # --------------------------------------------------------------- # 
            
            
            FLAG = False
            rmInds = [] # マイナス値が入ったインデックス
            rmlogs = [] # シミュレーションできなかったloファイル
            for lNum in np.arange(Xal_t.shape[0]):
                
                # [[Ut1,..,Utn],[th1,..,th3],[V1,..,Vn],]
                # Ut-1を取り除く, parfileHM*に必要がないから, shape=[4(parameter)*8(cell),]
                Xa_t = np.hstack([Xal_t[:,:nCell],Xal_t[:,nCell+nCell:]])[lNum,:]
                UThV = np.reshape(Xa_t[:bInd],[-1,nCell]) #[uthv(3),ncell(8)]
                
                # Ut-1を取り除く, all cell ver., shape=[8,4]
                Xa_t_all = np.concatenate([Xt_all[:,:,:nCell],Xt_all[:,:,nCell+nCell:]],2)[lNum,:]
                
                # 2000年以前のUを足し、元のスケールに戻してシミュレーション
                rescaleU = Xa_t_all[:,uInd] + startyU[:,lNum]
                Xa_t_all[:,uInd] = rescaleU
                
                # Get paramb, 小数第7位四捨五入
                updateB = np.round(Xa_t[bInd:],limitNum) # [b1,..,bN].T
                # Get U
                updateU = np.round(UThV[uInd],limitNum)
                # Get theta, [cell,]
                updateTh = np.round(UThV[thInd],limitNum)
                # Get V, [cell,], 小さすぎるから四捨五入なし
                updateV = UThV[VInd]
                
                if cell == 2 or cell == 4 or cell == 5:
                    # 更新したセルのパラメータとそのほかをconcat
                    Xa_t_all[cell] = Xa_t
                    # save index of minus parameters
                    if np.any(Xa_t_all<0):
                        rmInds.append(lNum)
                
                print("----")
                print(f"TrueU:{np.round(TrueU,6)}\n")
                print("before,update:\n")
                print(f"B:{np.round(Xt[lNum][-1],limitNum)},{updateB}\nU:{np.round(Xt[lNum][0],limitNum)},{updateU}\nUex:{np.round(Xt[lNum][1],limitNum)}\ntheta:{np.round(Xt[lNum][2],limitNum)},{updateTh}\nV:{Xt[lNum][-2]},{updateV}\n")
                
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
                    outlines = Xa_t_all[nl]
                    inlines[1],inlines[-3],inlines[-2],inlines[-1] = str(np.round(outlines[3],limitNum)),str(np.round(outlines[0],limitNum)),str(np.round(outlines[1],limitNum)),str(outlines[2])

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
                # =============================================================== #
            
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
            while True:
                
                # ---- Making Lock.txt 
                lockPath = "Lock.txt"
                lock = str(1)
                with open(lockPath,"w") as fp:
                    fp.write(lock)
                # --------------------    
                    
                # ---- Do bat
                process = subprocess.Popen(batFile)
                # --------------------    
                
                # --------------------    
                try:
                    #process.wait(timeout=data.shape[0]*3)
                    process.wait(timeout=2)
                    print("wait...")
                    
                    cnt += 1
                    sleepTime = 3
                    # lockファイル作成時は停止
                    while True:
                        time.sleep(sleepTime)
                        if os.path.exists(lockPath)==False:
                            break
                # --------------------    

                # --------------------    
                except subprocess.TimeoutExpired:
                    
                    # 一旦プロセス殺す
                    process.kill()
                    
                    # get newest logs file 
                    rmlogFile = os.path.basename(max(allfiles,key=os.path.getctime))

                    # save rm log file for dumping
                    rmlogs.append(rmlogFile)

                    rmlabels = rmlogFile.split("_")[1:]

                    # rm log label -> rm parHM* file
                    rmparamFile = parallfiles in rmlabels
                    (np.array(parallfiles)[fInds==1]).tolist()
                    pdb.set_trace()
                    myData.Empty(rmparamFile)    
                # --------------------    

                # ---- get label of last loading parHM* 
                lastLabel = parallfiles[-1][-7:]
                # all logs files
                allfiles = glob.glob(filePath)
                # lastLabel in allfiles != []
                lastlogfile = [s for s in allfiles if lastLabel in s]
                # --------------------    
                #pdb.set_trace()
                # ---- 最後までシミュレーションできたら終了
                if lastlogfile != []:
                    print("---- Finish Simulation! ----/n")
                    break
                # --------------------    
            # =============================================================== #
            
            # =============================================================== #
            
            # 通し番号を１つ増やす 0回目, １回目 ...
            iS += 1
        # -------------------------------------------------------- #

