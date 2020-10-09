# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 14:31:23 2020

@author: StoyanBoyukliyski
"""

import Cholesky as Ch
import numpy as np


select = 0.1
n = 3
m = 3
dx = 20
dy = 20

Lower, Cor = Ch.MatrixBuilder(select, n, m, dx, dy)
attempt = []
StdMDim = np.linalg.det(Cor)
standarddev = []
for _ in range(1000):
    X = np.random.normal(0, 1, n*m)
    Y = np.matmul(Lower, X)
    standarddev.append(np.std(Y))
    attempt.append((np.matmul(np.matmul(X, Cor),X)-np.mean(np.matmul(Lower,X)))/(n*m-1))

meanattempt = np.mean(attempt)
meanstandard = np.mean(standarddev)
print("XQX = ", meanattempt)
print(StdMDim)
print("LX = ", meanstandard)
