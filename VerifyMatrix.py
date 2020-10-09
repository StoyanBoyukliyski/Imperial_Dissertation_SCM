# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 12:57:26 2020

@author: StoyanBoyukliyski
"""

import Cholesky as Ch
import numpy as np
import matplotlib.pyplot as plt

select = 0.01
n = 100
m = 1

dx = 500
dy = 500

Cholesky = Ch.MatrixBuilder(select, n, m, dx, dy)

L = np.linalg.cholesky(Cholesky)


for j in range(5):
    x = np.random.normal(0,1,n)
    y = np.matmul(L,x)
    plt.plot(y)
print(y)