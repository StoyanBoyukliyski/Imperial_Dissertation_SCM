# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 12:13:04 2020

@author: StoyanBoyukliyski
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import lognorm
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

import matplotlib.animation as animation
import Cholesky as Ch
import RandomFields as RF

Lx = Ch.Lx
Ly = Ch.Ly
discx = 10
discy = 10

n = discx +1
m = discy +1
initdist = Ch.initdist
dx = Lx/(n)
dy = Ly/(m)

distx = np.linspace(0,Lx, n)
disty = np.linspace(0,Ly, m)

DX,DY = np.meshgrid(distx,disty)
DZ = np.zeros((n,m))
CX = DX + dx/2
CY = DY + dy/2
CXp = CX[:-1,:-1]
CYp = CY[:-1,:-1]
CX = np.reshape(CX, (n)*(m))
CY = np.reshape(CY, (n)*(m))
CXp = np.reshape(CXp, (n-1)*(m-1))
CYp = np.reshape(CYp, (n-1)*(m-1))
Cg = np.zeros((discx*discy,discx*discy))
C= np.zeros((discx,discx))
matrices = []

select = str(0.1)
x = np.linspace(initdist, Lx + initdist, n)
y = np.linspace(0, Ly, m)
RF.ax.plot(CXp,CYp, "r.")
RF.ax.plot_wireframe(DX, DY, DZ)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_wireframe(DX, DY, DZ)
#ax.plot(CXp,CYp, "ro")
#ax.plot(RF.X, RF.Y, "g.")
ax.set_title("3D plot of Intensity Measures", {'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'})
ax.set_ylabel("Distance in North (km)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad=20)
ax.set_xlabel("Distance in East (km)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)

if float(select) < 1:
    parb = 8.5 + 17.2*float(select)
else:
    parb = 22 + 3.7*float(select)


def Rho(h):
    rho = np.exp(-3*h/parb)
    return rho


for z in range(discy):
        for i in range(discx):
            x3 = dx*i
            x4 = dx*(i+1)
            y3 = 0
            y4 = dy
            r = np.random.uniform(x3,x4,1000)
            f = np.random.uniform(y3,y4,1000)
            for j in range(i,discx):
                x1 = i*dx
                x2 = (i+1)*dx
                y1 = z*dy
                y2 = (z+1)*dy
                x = np.random.uniform(x1,x2,1000)
                y = np.random.uniform(y1,y2,1000)
                distance = np.sqrt((x-r)**2+(y-f)**2)
                Rhoavg = np.average(Rho(distance))
                C[i,j] = Rhoavg
        matrices.append(np.array(C))

for k in range(discy):
    Cg[(k)*discx:(k+1)*discx, (k)*discx:(k+1)*discx] = matrices[0]
    for z in range(k+1,discy):
        C =  matrices[z-k]
        Cr = C + np.transpose(C) - np.diag(np.diag(C))
        Cg[(k)*discx:(k+1)*discx, (z)*discx:(z+1)*discx] = Cr


Cg = Cg + np.transpose(Cg)  - np.diag(np.diag(Cg))

