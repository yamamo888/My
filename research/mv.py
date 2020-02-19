# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import pdb

cell = sys.argv[1]

# Path ------------------------------------------------------------------------
logsPath = "logs"
dirPath = "b2b3b4b5b6400-495"
mvPath = f"tmp400_{cell}"
# -----------------------------------------------------------------------------

# Reading ---------------------------------------------------------------------
with open(f"path_190_b2b3b4b5b6400-450_{cell}.txt") as f:
    pathFile = [line.strip().split("/")[-1] for line in f.readlines()]
# -----------------------------------------------------------------------------

# Move ------------------------------------------------------------------------    
for file in pathFile:
    if os.path.exists(os.path.join(logsPath,dirPath,file)):
        shutil.move(os.path.join(logsPath,dirPath,file),os.path.join(logsPath,mvPath,file))
# -----------------------------------------------------------------------------
