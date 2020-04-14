# -*- coding: utf-8 -*-


import os
import sys
import time

import matplotlib.pylab as plt
import numpy as np


# parameters ------------------------------------------------------------------
# In paramHM* file
paramPath = "parFile"
paramCSV = "ParamFilePF.csv"
batFile = "PyToCPF.bat"

gt_Year = 1400
state_Year = 2000

# reading file start & end line
Sfl = 4
Efl = 12
# limit decimal
limitNum = 6
# -----------------------------------------------------------------------------

def updateParameters(yU,yth,yV,B,ssYears,tfID=0,iS=0,nP=0):
    
    # --------------------------- Xt-1 作成手順 ---------------------- #
        # 1 parfileをアンサンブル分作成
        # 2 batchファイルでファイル番号(Label etc...)受け渡し
        # 3 受け渡された番号のparfileを読み取りsimulation実行
    # --------------------------------------------------------------- #
    FLAG = False
    for lNum in np.arange(nP): # perticleの分
        #pdb.set_trace()
        # ========================= 1 =============================== #
        # Reading default parfile
        with open("parfileHM031def.txt","r") as fp:
            alllines = fp.readlines()
        # parfileHM031の改行コード削除
        alllines = [alllines[i].strip().split(",") for i in np.arange(len(alllines))]
        # ※ gtの発生年数に合わせる
        # 計算ステップ指定 (各データで異なる)
        alllines[0][0] = str(ssYears[lNum][0] + state_Year)
        alllines[0][1] = str(ssYears[lNum][0] + state_Year + gt_Year -1)

        # パラメータ設定行抽出
        lines = alllines[Sfl:Efl]
        for nl in np.arange(len(lines)): # 8 cell times
            # B, U, theta, V
            inlines = lines[nl]
            inlines[1] = str(np.round(B[nl][lNum],limitNum))
            inlines[-3] = str(yU[lNum][nl])
            inlines[-2] = str(yth[lNum][nl])
            inlines[-1] = str(yV[lNum][nl])
        #pdb.set_trace()
        # Save parfileHM031 -> parfileHM0*
        parFilePath = os.path.join(paramPath,f"{tfID}",f"parfileHM{iS}_{lNum}.txt")
        # 書式を元に戻す
        alllines = [','.join(alllines[i]) + '\n' for i in np.arange(len(alllines))]
        with open(parFilePath,"w") as fp:
            for line in alllines:
                fp.write(line)

        if not FLAG:
            # iS: 同化回数, tfID: 0-256(historical of eq.(=directory))
            fileLabel = np.hstack([iS,lNum,tfID])
            FLAG=True
        else:
            fileLabel = np.vstack([fileLabel,np.hstack([iS,lNum,tfID])])
        # =========================================================== #

    # ========================== 2 ================================== #
    # parFile番号格納
    data = np.c_[fileLabel]
    np.savetxt(paramCSV,data,delimiter=",",fmt="%.0f")
    # =============================================================== #

    # ========================== 3 ================================== #
    # ---- Making Lock.txt
    lockPath = "Lock.txt"
    lock = str(1)
    with open(lockPath,"w") as fp:
        fp.write(lock)
    # --------------------

    os.system(batFile)

    sleepTime = 3
    # lockファイル作成時は停止
    while True:
        time.sleep(sleepTime)
        if os.path.exists(lockPath)==False:
            break
    # =============================================================== #
