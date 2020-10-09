# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 21:40:44 2020

@author: StoyanBoyukliyski
"""

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import Cholesky as Ch
import matplotlib.pyplot as plt
import scipy.stats as stats
select = 0.1
n= 1
m = 100
Lx = 2
Ly = 2
if n ==1:
    dx = 0
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
else:
    dx = Lx/(n-1)
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
dy = Ly/(m-1)
x = [r*dx for r in range(n)]
y = [z*dy for z in range(m)]
X, Y = np.meshgrid(x,y)

L, Cg = Ch.MatrixBuilder(select, n, m, dx, dy)

standarddev = []
number = []
listing = []
stdavg =[]
for j in range(1000):
    e = np.random.normal(0,1, n*m)
    Er= np.matmul(L,e)
    listing.append(list(Er))
    standarddev.append(np.std(Er))
    stdavg.append(np.mean(standarddev))
    number.append(j)
    Er = np.reshape(Er, (n,m))
    if n ==1:
        ax.scatter(Y,np.transpose(Er))
    else:
        ax.scatter(X,Y,Er)
        
        
sigma = []
ax1= fig.add_subplot(1,2,2)
ax1.plot(number,standarddev)
ax1.plot(number, stdavg)
figr = plt.figure()
r = 1
last = np.zeros(len(np.transpose(listing)[0]))
for j in np.transpose(listing):
    ax2 = figr.add_subplot(1, n*m, r)
    ax2.hist(last-j, bins = 20)
    sigma.append(np.std(last-j))
    r = r + 1
    last = j

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(sigma[1:])