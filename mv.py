# -*- coding: utf-8 -*-

import os
import shutil

import numpy as np
import pdb

# Path ------------------------------------------------------------------------
logsPath = "logs"
dirPath = "b2b3b4b5b6205-300"
mvPath = "tmp"
# -----------------------------------------------------------------------------

# Reading ---------------------------------------------------------------------
with open("Top_190_path.txt") as f:
    pathFile = [line.strip().split("/")[-1] for line in f.readlines()]
# -----------------------------------------------------------------------------

# Move ------------------------------------------------------------------------    
for file in pathFile:
    if os.path.exists(os.path.join(logsPath,dirPath,file)):
        shutil.move(os.path.join(logsPath,dirPath,file),os.path.join(logsPath,mvPath,file))
# -----------------------------------------------------------------------------
