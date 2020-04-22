# -*- coding: utf-8 -*-


import os
import sys
import time
import pdb

import matplotlib.pylab as plt
import numpy as np


# parameters ------------------------------------------------------------------
# In paramHM* file
paramCSV = "updateB.csv"
batFile = "featureV.bat"
# -----------------------------------------------------------------------------

def updateParameters(B):
    
    B = np.round(B*1000000)
    # Save parameter b 
    np.savetxt(os.path.join('updateB.csv'),B,delimiter=",",fmt="%d")
    pdb.set_trace()
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
