# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 18:07:36 2020

@author: StoyanBoyukliyski
"""
import ChiouandYoung2014 as CY
import Cholesky as Ch
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

data = pd.read_csv("C:\\Users\\StoyanBoyukliyski\\OneDrive\\Desktop\\MScDissertation\PythonFiles\\RegressionCoefficients.csv")
data.head()
data = data.set_index("Period(s)")

Lx = Ch.Lx
Ly = Ch.Ly
n= Ch.n    
m = Ch.m
initdist = Ch.initdist
dx = Lx/(n-1)
dy = Ly/(m-1)

select = str(0.1)
slc = data.loc[select]
x = np.linspace(initdist, Lx + initdist, n)
y = np.linspace(0, Ly, m)

X1 = np.random.normal(0,1, n*m)
ResidualBtw = np.random.normal(0,1,1)

L = np.linalg.cholesky(Ch.Cg)

Mean = []
StdWth = []
StdBtw = []
between = CY.CalcStdBetween(slc)
for j in x:
    meanval = CY.CalculateY(slc,j)
    (NLo, within) = CY.CalcStdWithin(slc)
    StdBtw.append((1+NLo)*between)
    StdWth.append(within)
    Mean.append(meanval)

X,Y = np.meshgrid(x,y)         
Mean = np.array([Mean,]*len(y))
StdBtw = np.array([StdBtw,]*len(y))
StdWth = np.array([StdWth,]*len(y))


X1 = np.random.normal(0,1, n*m)
ResidualBtw = np.random.normal(0,1,1)
Yer = np.matmul(L,X1)

Mean = np.reshape(Mean, n*m)
StdBtw = np.reshape(StdBtw, n*m)
StdWth = np.reshape(StdWth,n*m)

Res = np.exp(Yer*StdWth)
Zerbtw = Mean*np.exp(ResidualBtw*StdBtw)
Zerwth = Zerbtw*Res

Zerbtw = np.reshape(Zerbtw, (m,n))
Zerwth = np.reshape(Zerwth, (m,n))
Mean = np.reshape(Mean, (m,n))
Res = np.reshape(Res, (m,n))



def plotter():
    fig2 = plt.figure(2)
    ax1 = fig2.add_subplot(111)
    ax1.hist(Yer, bins = 40)
    ax1.set_title("Corellated Residual Sampling", {'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'})
    ax1.set_ylabel("Number of samples in Bin",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad=20)
    ax1.set_xlabel("Residual Value",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)
    
    
    fig = plt.figure(3)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_wireframe(X, Y, Mean, color='black', linewidth = 0.6, label = "Mean Value")
    ax.plot_wireframe(X, Y, Zerbtw, color = "green", linewidth = 0.6, label = "Mean + Btw")
    ax.plot_wireframe(X, Y, Zerwth, color = "red", linewidth = 1, label = "Mean + Btw + Wth")
    ax.legend(loc = "upper right", prop = {"size": 7})
    ax.grid(linestyle = "--")
    ax.set_title("3D plot of Intensity Measures", {'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'})
    ax.set_zlabel("Intesity Measure (g)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)
    ax.set_ylabel("Distance in North (km)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad=20)
    ax.set_xlabel("Distance in East (km)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)
    
    ax1 = fig.add_subplot(2, 2, 2)
    ax1.plot(X[1, :n],Mean[1, :n], color = "black", linewidth = 0.6, label = "Mean Value")
    ax1.plot(X[1, :n],Zerbtw[1, :n], color = "green",  linewidth = 0.6, label = "Mean + Btw Residual")
    ax1.plot(X[1, :n],Zerwth[1, :n], color = "red",linewidth = 1, label = "Mean + Wth + Btw Residual")
    ax1.grid(linestyle = "--")
    ax1.set_yscale("log")
    ax1.set_title("Intensity Measures 2D", {'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'})
    ax1.set_xlabel("Distance in North (km)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad=20)
    ax1.set_ylabel("Intesity Measure (g)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)
    ax1.legend(loc = "upper right", prop = {"size": 7})
    ax2 = fig.add_subplot(2, 2, 4)
    CS = ax2.contourf(X,Y,np.log(Res), cmap = "viridis")
    ax2.contour(X, Y, np.log(Res), cmap = "viridis")
    fig.colorbar(CS)
    ax2.set_title("Contour Plot of Residuals", {'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'})
    ax2.set_ylabel("Distance in North (km)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad=20)
    ax2.set_xlabel("Distance in East (km)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)
    plt.tight_layout()
    plt.show()
    
plotter()