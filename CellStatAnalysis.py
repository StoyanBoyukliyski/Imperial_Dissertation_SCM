# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 14:05:25 2020

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
import math as ma

import RandomFields as RF
import ChiouandYoung2014 as CY
import GlobalDiscretization as GD
import Cholesky as Ch
import JayaramBaker2009 as JB


discx = GD.discx
discy = GD.discy
dx = RF.dx
dy = RF.dy
Lx = RF.Lx
Ly = RF.Ly
lensegx = Lx/discx
lensegy = Ly/discy
matrixofmatrix = []
Correlations = GD.Cg

n= discx  
m = discy

Cg = np.zeros((n*m,n*m))
C= np.zeros((n,n))
matrices = []

select = str(0.1)

for z in range(m):
        for i in range(n):
            for j in range(i,n):
                C[i,j] = JB.Rho(np.sqrt(((z)*dy)**2 + (dx*(j-i))**2),select)
        matrices.append(np.array(C))

for k in range(m):
    Cg[(k)*n:(k+1)*n, (k)*n:(k+1)*n] = matrices[0]
    for z in range(k+1,m):
        C =  matrices[z-k]
        Cr = C + np.transpose(C) - np.diag(np.diag(C))
        Cg[(k)*n:(k+1)*n, (z)*n:(z+1)*n] = Cr
        
del matrices
Cg = Cg + np.transpose(Cg) - np.identity(n*m) 

Lower = np.linalg.cholesky(Cg)

data = pd.read_csv("C:\\Users\\StoyanBoyukliyski\\OneDrive\\Desktop\\MScDissertation\PythonFiles\\RegressionCoefficients.csv")
data.head()
data = data.set_index("Period(s)")

select = str(0.1)
slc = data.loc[select]

IM = RF.Zerwth
MeansIM = []
StdSIM = []
StandardDevC = []
MeanC = []
Samples = []
Sampler1 = np.random.normal(0,1,1)
Sampler1 = np.ones((np.size(RF.Zerwth,0),np.size(RF.Zerwth,1)))*Sampler1
MainSamples = np.zeros((np.size(RF.Zerwth,0),np.size(RF.Zerwth,1)))
Sampler = np.reshape(RF.X1, (Ch.n, Ch.m))
MeanGlobal = np.zeros((np.size(RF.Zerwth,0),np.size(RF.Zerwth,1)))
StdGlobal = np.zeros((np.size(RF.Zerwth,0),np.size(RF.Zerwth,1)))
StdGlobalwth = np.zeros((np.size(RF.Zerwth,0),np.size(RF.Zerwth,1)))
StdGlobalbtw = np.zeros((np.size(RF.Zerwth,0),np.size(RF.Zerwth,1)))

Error = np.random.normal(0,1,n*m)
Yer = np.matmul(Lower,Error)

for k in range(discy):
    for j in range(discx):
        dist = (j+1/2)*(Lx/discx)
        meanval = CY.CalculateY(slc,dist)
        stdvalbtw = CY.CalcStdBetween(slc)
        stdvalwth = CY.CalcStdWithin(slc)[1]
        MeanC.append(meanval)
        
        Rb = ma.ceil(k*(Ly/discy)/dy)
        Rt = ma.floor((k+1)*(Ly/discy)/dy)+1
        Fb = ma.ceil(j*(Lx/discx)/dx)
        Ft = ma.floor((j+1)*(Lx/discx)/dx)+1
        IMval = IM[Rb:Rt,Fb:Ft]
        SamplingGlobal = Sampler[Rb:Rt,Fb:Ft]
        sample = Yer[j+k*discx]
        Samples.append(sample)
        K = np.ones((np.size(IMval,0),np.size(IMval,1)))
        matrixofmatrix.append(np.array(IMval))
        IMval = IMval.reshape(np.size(IMval))
        MeanIM = np.mean(IMval)
        StdIM = np.std(IMval)
        MeanGlobal[Rb:Rt,Fb:Ft] = MeanGlobal[Rb:Rt,Fb:Ft] + K*MeanIM
        StdGlobal[Rb:Rt,Fb:Ft] = StdGlobal[Rb:Rt,Fb:Ft] + K*StdIM
        StdGlobalwth[Rb:Rt,Fb:Ft] = StdGlobalwth[Rb:Rt,Fb:Ft] + K*stdvalwth
        StdGlobalbtw[Rb:Rt,Fb:Ft] = StdGlobalbtw[Rb:Rt,Fb:Ft] + K*stdvalbtw
        MainSamples[Rb:Rt,Fb:Ft] = MainSamples[Rb:Rt,Fb:Ft] + K*sample
        MeansIM.append(MeanIM)
        StdSIM.append(StdIM)
        '''
        plt.figure()
        plt.hist(IMval, bins = 10)
        '''

Corl = np.diag(Correlations)
GD.ax.plot_wireframe(RF.X, RF.Y, RF.Mean)
GD.ax.scatter3D(GD.CXp, GD.CYp, MeansIM, color = "red")
GD.ax.plot_wireframe(RF.X, RF.Y, MeanGlobal, color="green")

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.plot_wireframe(RF.X, RF.Y, IM)
ax1 = fig.add_subplot(122, projection='3d')
ax1.plot_wireframe(RF.X, RF.Y, MeanGlobal, color="green")
ax1.plot_wireframe(RF.X, RF.Y, MeanGlobal*np.exp(StdGlobalwth*MainSamples+StdGlobalbtw*Sampler1), color = "red")

