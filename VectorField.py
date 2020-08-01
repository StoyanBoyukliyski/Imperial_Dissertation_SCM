# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 14:46:56 2020

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
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import SourceGeometry as SG
import ChiouandYoung2014 as CY14
import Cholesky as Ch
import PlotGeometry as PG

Lx = Ch.Lx
Ly = Ch.Ly
n= Ch.n  
m = Ch.m
initx = Ch.initx
inity = Ch.inity
dx = Ch.dx
dy = Ch.dy
T= SG.T
Tfita= SG.Tfita
beta = SG.beta
W = SG.W
L = SG.L
xl = SG.xl
xw = SG.xw

x = np.linspace(initx,initx + Lx, n)
y = np.linspace(inity,inity + Ly, m)

select = Ch.select
slc = Ch.slc
scale = SG.M

LogF = CY14.LognormalFunct(slc,x,y)

ResidualWth = np.random.normal(0,1, n*m)
ResidualBtw = np.random.normal(0,1,1)

Mtr = Ch.MatrixBuilder(select, m, n, dx, dy)
L = np.linalg.cholesky(Mtr)
ResidualWth = np.matmul(L,ResidualWth)

Res = ResidualWth*LogF[3]
Zerbtw = LogF[0] + ResidualBtw*LogF[4]*(1+LogF[2])
Zerwth = Zerbtw + Res


PG.ax.set_zlim([-W,np.max(Zerwth)*scale])
X,Y = np.meshgrid(x,y)
Mean = np.reshape(LogF[0], np.shape(X))
Zerbtw = np.reshape(Zerbtw, np.shape(X))
Zerwth = np.reshape(Zerwth, np.shape(X))
Res= np.reshape(Res, np.shape(X))


#PG.ax.plot_wireframe(X, Y, Mean*scale, color='green', linewidth = 0.6, label = "Mean Value")
PG.ax.plot_wireframe(X, Y, Zerbtw*scale + abs(np.min(np.reshape(Zerbtw*scale, np.size(Zerbtw)))), color = "blue", linewidth = 0.6, label = "Mean + Btw")
PG.ax.contourf(X, Y, Zerbtw, cmap = "viridis")
#PG.ax.plot_wireframe(X, Y, Zerwth*scale, color = "black", linewidth = 1, label = "Mean + Btw + Wth")
PG.ax.legend(loc = "upper right", prop = {"size": 7})
PG.ax.grid(linestyle = "--")
PG.ax.set_title("3D plot of Intensity Measures", {'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'})
PG.ax.set_zlabel("Intesity Measure (g)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)
PG.ax.set_ylabel("Distance in North (km)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad=20)
PG.ax.set_xlabel("Distance in East (km)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)


fig2 = plt.figure(4)
ax1 = fig2.add_subplot(111)
ax1.hist(ResidualWth, bins = 40)
ax1.set_title("Corellated Residual Sampling", {'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'})
ax1.set_ylabel("Number of samples in Bin",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad=20)
ax1.set_xlabel("Residual Value",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)


fig = plt.figure(3)
ax = fig.add_subplot(1, 2, 1, projection='3d')
#ax.plot_wireframe(X, Y, Mean, color='green', linewidth = 0.6, label = "Mean Value")
#ax.plot_wireframe(X, Y, Zerbtw, color = "blue", linewidth = 0.6, label = "Mean + Btw")
ax.plot_wireframe(X, Y, Zerwth, color = "blue", linewidth = 1, label = "Mean + Btw + Wth")

ax.legend(loc = "upper right", prop = {"size": 7})
ax.grid(linestyle = "--")
ax.set_title("3D plot of Intensity Measures", {'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'})
ax.set_zlabel("Intesity Measure (g)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)
ax.set_ylabel("Distance in North (km)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad=20)
ax.set_xlabel("Distance in East (km)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)

ax1 = fig.add_subplot(2, 2, 2)
ax1.plot(X[1, :n],np.exp(Mean[1, :n]), color = "black", linewidth = 0.6, label = "Mean Value")
ax1.plot(X[1, :n],np.exp(Zerbtw[1, :n]), color = "green",  linewidth = 0.6, label = "Mean + Btw Residual")
ax1.plot(X[1, :n],np.exp(Zerwth[1, :n]), color = "red",linewidth = 1, label = "Mean + Wth + Btw Residual")
ax1.grid(linestyle = "--")
ax1.set_yscale("log")
ax1.set_title("Intensity Measures 2D", {'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'})
ax1.set_xlabel("Distance in North (km)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad=20)
ax1.set_ylabel("Intesity Measure (g)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)
ax1.legend(loc = "upper right", prop = {"size": 7})
ax2 = fig.add_subplot(2, 2, 4)
CS = ax2.contourf(X,Y,Res, cmap = "viridis")
ax2.contour(X, Y, Res, cmap = "viridis")
fig.colorbar(CS)
ax2.set_title("Contour Plot of Residuals", {'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'})
ax2.set_ylabel("Distance in North (km)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad=20)
ax2.set_xlabel("Distance in East (km)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)
plt.tight_layout()
plt.show()