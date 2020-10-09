# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 21:36:36 2020

@author: StoyanBoyukliyski
"""
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import Cholesky as Ch

size = 1000
initx = 10
inity = 10
Lx = 100
Ly = 100
select = 0.1
x = np.linspace(initx, initx+Lx, size)
y = np.linspace(inity, inity+Ly, size)
e = np.random.normal(0,1,size)
L, Cg = Ch.FastMatrixBuilder(select, x,y)

y = np.matmul(L,e)
