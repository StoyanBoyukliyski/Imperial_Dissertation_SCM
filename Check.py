# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 17:03:44 2020

@author: StoyanBoyukliyski
"""

import Cholesky as Ch
import time
import numpy as np

time_one = time.time()
n = 150
m = 150
dx = 5
dy = 5
mTR = Ch.MatrixBuilder(0.1,n, m,dx,dy)
time_two = time.time()
print(time_two-time_one)

L = np.linalg.cholesky(mTR)

time_three = time.time()
print(time_three-time_two)
