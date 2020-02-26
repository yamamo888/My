# -*- coding: utf-8 -*-

import os
import sys
import glob
import pickle
import pdb

import seaborn as sns
import matplotlib.pylab as plt
import numpy as np

import makingData as myData

# path ----------
logsPath = "logs"
paramPath = "parFile"
dirPath = "last"
featuresPath = "nankairirekifeature"
txtPath = "*txt"
# ---------------

filePath = os.path.join(logsPath,dirPath,txtPath)
file = glob.glob(filePath)

# parameter ----
nCell = 8
Sfl = 4
Efl = 12
limitNum = 6
ntI,tntI,ttI = 0,1,2
nYear = 1400
slip = 1
# --------------

# -----------------------------------------------------------------------------
def MinErrorNankai(gt,yU,yth,yV,pY,cell=0,gtcell=0,nCell=0):
   
    # ----
    # 昭和以降の地震年数
    gYear_nk = gt[ntI][-3:]
    gYear_tnk = gt[tntI][-3:]
    gYear_tk = gt[ttI][-2:]
    # ----
    
    # 予測した地震年数 only one-cell
    pYear_nk = pY[ntI]
    pYear_tnk = pY[tntI]
    pYear_tk = pY[ttI]
    
    pred = np.zeros([nYear,cell])
    gt = np.zeros([nYear,cell])
    
    pred[pYear_nk,ntI] = 2
    pred[pYear_tnk,tntI] = 2
    pred[pYear_tk,ttI] = 2
    
    gt[gYear_nk,ntI] = 2
    gt[gYear_tnk,tntI] = 2
    gt[gYear_tk,ttI] = 2
    
    
    # gaussian distance for year of gt - year of pred (gYears.shape, pred.shape)
    # for each cell
    ndist_nk = gauss(gYear_nk,pYear_nk.T)
    ndist_tnk = gauss(gYear_tnk,pYear_tnk.T)
    ndist_tk = gauss(gYear_tk,pYear_tk.T)

    # 予測誤差の合計, 回数で割ると当てずっぽうが小さくなる
    # for each cell
    yearError_nk = sum(ndist_nk.max(1)/pYear_nk.shape[0])
    yearError_tnk = sum(ndist_tnk.max(1)/pYear_tnk.shape[0])
    yearError_tk = sum(ndist_tk.max(1)/pYear_tk.shape[0])

    # for all cell
    maxSim = yearError_nk + yearError_tnk + yearError_tk

    print(">>>>>>>>\n")                
    print(f"最大類似度:{np.round(maxSim,6)}\n")
    print(">>>>>>>>\n")
    
    sns.set_style("dark")
    fig, figInds = plt.subplots(nrows=3, sharex=True)
    for figInd in np.arange(len(figInds)):
        figInds[figInd].plot(np.arange(1400), pred[:,figInd], color="skyblue")
        figInds[figInd].plot(np.arange(1400), gt[:,figInd], alpha=0.5, color="coral")
        
    plt.suptitle(f"{np.round(maxSim,5)}")
    plt.savefig("resultPF.png")
    plt.close()
        
    pdb.set_trace()
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def gauss(gtY,predY,sigma=100):

    # predict matrix for matching times of gt eq.
    predYs = predY.repeat(gtY.shape[0],0).reshape(-1,gtY.shape[0])
    # gt var.
    gtYs = gtY.repeat(predY.shape[0],0).reshape(-1,predY.shape[0])

    gauss = np.exp(-(gtYs - predYs.T)**2/(2*sigma**2))

    return gauss
# -----------------------------------------------------------------------------
    
"""
flag = False
for file in files:
    data = open(file).readlines()
        
    B = np.zeros(nCell)
    UTV = np.zeros([nCell,3])
    
    for sl in np.arange(4,12,1):
        tmpB = np.array([data[sl].split(",")[1]]).astype(float)
        tmpUTV = np.array([data[sl].split(",")[-3:]]).astype(float)
        #pdb.set_trace()
        B[sl-4] = tmpB
        UTV[sl-4] = tmpUTV

    if not flag:
        Bs = B
        UTVs = UTV[np.newaxis]
        flag = True
    else:
        Bs = np.vstack([Bs,B])
        UTVs = np.vstack([UTVs,UTV[np.newaxis]])

print("B:",np.round(np.mean(Bs,0),6))
for cell in np.arange(3):
    print(f"UTV in {cell}:",np.mean(UTVs[:,:,cell],0))

B_all = np.mean(Bs,0)
yU_rYear = np.mean(UTVs[:,:,0],0)
yth_rYear = np.mean(UTVs[:,:,1],0)
yV_rYear = np.mean(UTVs[:,:,2],0)


# making parHM* files ---------------------------------------------------------
with open("parfileHM031def.txt","r") as fp:
    alllines = fp.readlines()
# parfileHM031の改行コード削除
alllines = [alllines[i].strip().split(",") for i in np.arange(len(alllines))]

# 計算ステップ指定 (各データで異なる)
alllines[0][0] = str(1107)
alllines[0][1] = str(1400)
#pdb.set_trace()
# パラメータ設定行抽出
lines = alllines[Sfl:Efl]
for nl in np.arange(len(lines)): # 8 cell times
    # B, U, theta, V
    inlines = lines[nl]
    inlines[1] = str(np.round(B_all[nl],limitNum))
    inlines[-3] = str(yU_rYear[nl])
    inlines[-2] = str(yth_rYear[nl])
    inlines[-1] = str(yV_rYear[nl])
#pdb.set_trace()
# Save parfileHM031 -> parfileHM0*
parFilePath = os.path.join(paramPath,"0","100_0.txt")
# 書式を元に戻す
alllines = [','.join(alllines[i]) + '\n' for i in np.arange(len(alllines))]
with open(parFilePath,"w") as fp:
    for line in alllines:
        fp.write(line)
# -----------------------------------------------------------------------------
"""

for tfID in [190]:
    
    # ----------------- 真の南海トラフ巨大地震履歴 V------------------------- #
    with open(os.path.join(featuresPath,"nankairireki.pkl"), "rb") as fp:
        nkfiles = pickle.load(fp)
    
    # 発生年数取得 & slip velocity (all 30)
    gtV = nkfiles[tfID,:,:]
    
    gtJ = np.unique(np.where(gtV>0)[0])
    gtJ_nk = np.where(gtV[:,ntI]>0)[0]
    gtJ_tnk = np.where(gtV[:,tntI]>0)[0]
    gtJ_tk = np.where(gtV[:,ttI]>0)[0]
    gtJs = [gtJ_nk,gtJ_tnk,gtJ_tk]
    
    U,th,V,B = myData.loadABLV(logsPath,dirPath,os.path.basename(file[0]))
    yU, yth, yV, pJ_all = myData.convV2YearlyData(U,th,V,nYear,cell=245,cnt=1) 
    MinErrorNankai(gtJs,yU,yth,yV,pJ_all,cell=245,nCell=nCell)
    









