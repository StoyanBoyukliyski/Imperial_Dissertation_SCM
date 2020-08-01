# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 14:05:25 2020

@author: StoyanBoyukliyski
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as ma

import ChiouandYoung2014 as CY
import GlobalDiscretization as GD
import Cholesky as Ch
import JayaramBaker2009 as JB
import VectorField as VF


VF.ax.plot(GD.CXp,GD.CYp, "r.")
VF.ax.plot_wireframe(GD.DX, GD.DY, GD.DZ)
VF.ax.plot(GD.CXp,GD.CYp, "r.")


discx = GD.discx
discy = GD.discy
dx = VF.dx
dy = VF.dy
lensegx = VF.Lx/discx
lensegy = VF.Ly/discy
matrixofmatrix = []
Correlations = GD.Cg

n= discx  
m = discy

select = VF.select

Cg = Ch.MatrixBuilder(select, m, n, dx, dy)
Lower = np.linalg.cholesky(Cg)
ResidualWth = np.reshape(VF.ResidualWth, (Ch.n,Ch.m))


StdGlobalwth = np.zeros((np.size(VF.Zerwth,0),np.size(VF.Zerwth,1)))
StdGlobalbtw = np.zeros((np.size(VF.Zerwth,0),np.size(VF.Zerwth,1)))


def CreatePlot(figure, k,j, Rb, Rt, Fb, Ft, totalstats, z=1):
    stats = totalstats[Rb:Rt,Fb:Ft]
    stats = totalstats[Rb:Rt,Fb:Ft]
    mean = np.mean(stats)
    std = np.std(stats)
    axnu = figure.add_subplot(discy, discx, k*(discx) +j +1)
    stats = np.reshape(stats, np.size(stats))
    axnu.hist(stats, bins= 15)
    axnu.plot([mean,mean],[0,20], "r-", linewidth = 2)
    if z == 1:
        axnu.plot([0,0],[0,20], "k-", linewidth = 2)
    else:
        pass
    axnu.text(mean, 0, "Mean = " + str("{:.2f}".format(mean)) + "\n" + "Std = " + str("{:.2f}".format(std)), fontsize = 7)
    axnu.axis("off")
    return mean, std

FigStatError = plt.figure(7)
FigFunct = plt.figure(8)
FigTotal = plt.figure(9)

MeanEr = []
StdEr = []
MeanF = []
StdF = []

'''
Method 1:
    Calculate the mean and standard deviation at central point location
'''



def Method1(discx,discy):
    MeanC = []
    StandardC = []
    MeanMethod1 = np.zeros(np.shape(VF.Zerwth))
    StdMethod1 = np.zeros(np.shape(VF.Zerwth))
    for k in range(discy):
        for j in range(discx):
            distx = (j+1/2)*(VF.Lx/discx) + Ch.initx
            disty = (k+1/2)*(VF.Ly/discy) + Ch.inity
            Rb = ma.ceil(k*(VF.Ly/discy)/dy)
            Rt = ma.floor((k+1)*(VF.Ly/discy)/dy)+1
            Fb = ma.ceil(j*(VF.Lx/discx)/dx)
            Ft = ma.floor((j+1)*(VF.Lx/discx)/dx)+1
            K = np.ones(np.shape(MeanMethod1[Rb:Rt,Fb:Ft]))
        
            LognFunct = CY.LognormalFunct(VF.slc, distx, disty)
            meanm1 = np.float(LognFunct[0])
            stdm1 = np.float(LognFunct[1])
            
            MeanMethod1[Rb:Rt,Fb:Ft] = MeanMethod1[Rb:Rt,Fb:Ft] + K*meanm1
            StdMethod1[Rb:Rt,Fb:Ft] = StdMethod1[Rb:Rt,Fb:Ft] + K*stdm1
            
            MeanC.append(meanm1)
            StandardC.append(stdm1)
            
    return MeanC, StandardC,MeanMethod1, StdMethod1


def Method2(discx, discy):
    MeanMeanGlobal = np.zeros((np.size(VF.Zerwth,0),np.size(VF.Zerwth,1)))
    StdMeanGlobal = np.zeros((np.size(VF.Zerwth,0),np.size(VF.Zerwth,1)))
    MeanOfMeanVector = [] 
    StdOfMeanVector  = []
    StatError = np.reshape(VF.Res, (Ch.n,Ch.m))
    StatFunct = np.reshape(VF.Zerbtw, (Ch.n,Ch.m))
    StatTotal = np.reshape(VF.Zerwth, (Ch.n,Ch.m))
    for k in range(discy):
        for j in range(discx):
            Rb = ma.ceil(k*(VF.Ly/discy)/dy)
            Rt = ma.floor((k+1)*(VF.Ly/discy)/dy)+1
            Fb = ma.ceil(j*(VF.Lx/discx)/dx)
            Ft = ma.floor((j+1)*(VF.Lx/discx)/dx)+1
            
            MeanCell = VF.Mean[Rb:Rt,Fb:Ft]
            K = np.ones(np.shape(MeanCell))
            MeanCell = MeanCell.reshape(np.size(MeanCell))
            MeanOfMean = np.mean(MeanCell)
            StdOfMean = np.std(MeanCell)
            
            MeanMeanGlobal[Rb:Rt,Fb:Ft] = MeanMeanGlobal[Rb:Rt,Fb:Ft] + K*MeanOfMean
            StdMeanGlobal[Rb:Rt,Fb:Ft] = StdMeanGlobal[Rb:Rt,Fb:Ft] + K*StdOfMean
            
            MeanOfMeanVector.append(MeanOfMean)
            StdOfMeanVector.append(StdOfMean)
            
            MeanStdEr = CreatePlot(FigStatError, k, j, Rb, Rt, Fb, Ft, StatError, 1)
            MeanEr.append(MeanStdEr[0])
            StdEr.append(MeanStdEr[1])
            MeanStdFun = CreatePlot(FigFunct, k, j, Rb, Rt, Fb, Ft, StatFunct, 0)
            MeanF.append(MeanStdFun[0])
            StdF.append(MeanStdFun[1])
            CreatePlot(FigTotal, k, j, Rb, Rt, Fb, Ft, StatTotal,0)
            
    return MeanOfMeanVector,StdOfMeanVector,MeanMeanGlobal, StdMeanGlobal

def Method22(discx, discy):
    MeanMeanGlobal = np.zeros((np.size(VF.Zerwth,0),np.size(VF.Zerwth,1)))
    MeanOfMeanVector = [] 
    for k in range(discy):
        for j in range(discx):
            Rb = ma.ceil(k*(VF.Ly/discy)/dy)
            Rt = ma.floor((k+1)*(VF.Ly/discy)/dy)+1
            Fb = ma.ceil(j*(VF.Lx/discx)/dx)
            Ft = ma.floor((j+1)*(VF.Lx/discx)/dx)+1
            
            MeanCell = VF.Mean[Rb:Rt,Fb:Ft]
            K = np.ones(np.shape(MeanCell))
            MeanCell = MeanCell.reshape(np.size(MeanCell))
            MeanOfMean = np.mean(MeanCell)
            
            MeanMeanGlobal[Rb:Rt,Fb:Ft] = MeanMeanGlobal[Rb:Rt,Fb:Ft] + K*MeanOfMean
            
            MeanOfMeanVector.append(MeanOfMean)
            
    return MeanOfMeanVector,MeanMeanGlobal

def Method23(discx, discy):
    StdMeanGlobal = np.zeros((np.size(VF.Zerwth,0),np.size(VF.Zerwth,1)))
    StdOfMeanVector  = []
    for k in range(discy):
        for j in range(discx):
            Rb = ma.ceil(k*(VF.Ly/discy)/dy)
            Rt = ma.floor((k+1)*(VF.Ly/discy)/dy)+1
            Fb = ma.ceil(j*(VF.Lx/discx)/dx)
            Ft = ma.floor((j+1)*(VF.Lx/discx)/dx)+1
            
            MeanCell = VF.Mean[Rb:Rt,Fb:Ft]
            K = np.ones(np.shape(MeanCell))
            MeanCell = MeanCell.reshape(np.size(MeanCell))
            StdOfMean = np.std(MeanCell)
            
            StdMeanGlobal[Rb:Rt,Fb:Ft] = StdMeanGlobal[Rb:Rt,Fb:Ft] + K*StdOfMean
            
            StdOfMeanVector.append(StdOfMean)
            
    return StdOfMeanVector, StdMeanGlobal

def Method24(discx, discy):
    MeanErrorGlobal = np.zeros((np.size(VF.Zerwth,0),np.size(VF.Zerwth,1)))
    MeanOfErrorVector = [] 
    for k in range(discy):
        for j in range(discx):
            Rb = ma.ceil(k*(VF.Ly/discy)/dy)
            Rt = ma.floor((k+1)*(VF.Ly/discy)/dy)+1
            Fb = ma.ceil(j*(VF.Lx/discx)/dx)
            Ft = ma.floor((j+1)*(VF.Lx/discx)/dx)+1
            L = VF.L
            ResidualWth = np.random.normal(0,1, n*m)
            ResidualWth = np.matmul(L,ResidualWth)
            
            
            ErrorCell = ResidualWth[Rb:Rt,Fb:Ft]
            K = np.ones(np.shape(ErrorCell))
            ErrorCell = ErrorCell.reshape(np.size(ErrorCell))
            MeanOfError = np.mean(ErrorCell)
            
            MeanErrorGlobal[Rb:Rt,Fb:Ft] = MeanErrorGlobal[Rb:Rt,Fb:Ft] + K*MeanOfError
            
            MeanOfErrorVector.append(MeanOfError)
            
    return MeanOfErrorVector,MeanErrorGlobal

def Method25(discx, discy):
    StdErrorGlobal = np.zeros((np.size(VF.Zerwth,0),np.size(VF.Zerwth,1)))
    StdOfErrorVector  = []
    for k in range(discy):
        for j in range(discx):
            Rb = ma.ceil(k*(VF.Ly/discy)/dy)
            Rt = ma.floor((k+1)*(VF.Ly/discy)/dy)+1
            Fb = ma.ceil(j*(VF.Lx/discx)/dx)
            Ft = ma.floor((j+1)*(VF.Lx/discx)/dx)+1
            
            MeanCell = VF.Mean[Rb:Rt,Fb:Ft]
            K = np.ones(np.shape(MeanCell))
            MeanCell = MeanCell.reshape(np.size(MeanCell))
            StdOfMean = np.std(MeanCell)
            
            StdErrorGlobal[Rb:Rt,Fb:Ft] = StdErrorGlobal[Rb:Rt,Fb:Ft] + K*StdOfMean
            
            StdOfErrorVector.append(StdOfMean)
            
    return StdOfMeanVector, StdMeanGlobal





MeanC, StandardC, MeanMethodC, StdMethodC = Method1(discx, discy)
MeanOfMeanVector,StdOfMeanVector,MeanMeanGlobal, StdMeanGlobal = Method2(discx, discy)

GD.ax.plot_surface(VF.X, VF.Y, VF.Mean)
#GD.ax.plot_wireframe(VF.X, VF.Y, np.reshape(MeanMeanGlobal, (Ch.n,Ch.m)), color="green")
GD.ax.plot_wireframe(VF.X, VF.Y,  np.reshape(MeanMethodC, (Ch.n,Ch.m)), color="red")

fig = plt.figure(6)
ax1 = fig.add_subplot(121)
ax1.contourf(VF.X, VF.Y, VF.Mean, levels = discx*discy)
ax2 = fig.add_subplot(222)
ax2.contourf(VF.X, VF.Y, np.reshape(MeanMethodC, (Ch.n,Ch.m)), levels = MeanC.sort())
ax2.contour(VF.X, VF.Y, np.reshape(MeanMethodC, (Ch.n,Ch.m)), levels = MeanC.sort())
ax3 = fig.add_subplot(224)
ax3.contourf(VF.X, VF.Y, np.reshape(MeanMeanGlobal, (Ch.n,Ch.m)), levels = MeanOfMeanVector.sort())
ax3.contour(VF.X, VF.Y, np.reshape(MeanMeanGlobal, (Ch.n,Ch.m)), levels = MeanOfMeanVector.sort())