# -*- coding: utf-8 -*-

import os
import numpy as np

class ParamCycleNN:
    def __init__(self, savepath='none', ):
        
        self.trialID = trialID
        self.savepath = savepath
        
    def simulate(self, params):
        
        # for loading batfile
        np.savetxt(os.path.join(self.savepath, 'logs.csv'))
        # copy
        np.savetxt(os.path.join(self.savepath, ''))
        
        # Make logs ----
        lockPath = "Lock.txt"
        lock = str(1)
        with open(lockPath,"w") as fp:
            fp.write(lock)
        
        batFile = 'makelogs.bat'
        os.system(batFile)
    
        sleepTime = 3
        while True:
            time.sleep(sleepTime)
            if os.path.exists(lockPath)==False:
                break
        # ----
        
