# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 14:05:25 2020

@author: StoyanBoyukliyski
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as ma
import ChiouandYoung2014 as CY14
import Cholesky as Ch

'''
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

Lower = Ch.MatrixBuilder(select, m, n, dx, dy)
ResidualWth = np.reshape(VF.ResidualWth, (Ch.n,Ch.m))


StdGlobalwth = np.zeros((np.size(VF.Zerwth,0),np.size(VF.Zerwth,1)))
StdGlobalbtw = np.zeros((np.size(VF.Zerwth,0),np.size(VF.Zerwth,1)))
'''

def CreatePlot(figure, k,j, Rb, Rt, Fb, Ft, totalstats, discx, discy, z=1):
    stats = totalstats[Fb:Ft, Rb:Rt]
    mean = np.mean(stats)
    std = np.std(stats)
    axnu = figure.add_subplot(discy, discx, k*(discx) +j +1)
    stats = np.reshape(stats, np.size(stats))
    axnu.hist(stats, bins = 100)
    axnu.plot([mean,mean],[0,20], "r-", linewidth = 2)
    if z == 1:
        axnu.plot([0,0],[0,20], "k-", linewidth = 2)
    else:
        pass
    axnu.text(mean, 0, "Mean = " + str("{:.2f}".format(mean)) + "\n" + "Std = " + str("{:.2f}".format(std)), fontsize = 7)
    axnu.axis("on")
    return mean, std

'''
Method 1:
    Calculate the mean and standard deviation at central point location
'''



def Method1(discx,discy, Lx, Ly, slc, initx, inity, n, m, dx, dy, FRV, FNM, dZTOR, M, b, dDPP, ZTOR, VS30, dZ10, ni, Finferred, Fmeasured, Tfita, T, xl, xw, L, W, Wpr):
    MeanC = []
    StandardC = []
    MeanMethod1 = np.zeros((n*m,n*m))
    StdMethod1 = np.zeros((n*m,n*m))
    for k in range(discy):
        for j in range(discx):
            distx = (j+1/2)*(Lx/discx) + initx
            disty = (k+1/2)*(Ly/discy) + inity
            Rb = ma.ceil(k*(Ly/discy)/dy)
            Rt = ma.floor((k+1)*(Ly/discy)/dy)+1
            Fb = ma.ceil(j*(Lx/discx)/dx)
            Ft = ma.floor((j+1)*(Lx/discx)/dx)+1
            K = np.ones(np.shape(MeanMethod1[Fb:Ft, Rb:Rt]))
            LognFunct = CY14.LognormalFunct(slc, distx, disty,  FRV, FNM, dZTOR, M, b, dDPP, ZTOR, VS30, dZ10, ni, Finferred, Fmeasured, Tfita, T, xl, xw, L, W, Wpr)
            meanm1 = np.float(LognFunct[0])
            stdm1 = np.float(LognFunct[3])
            
            MeanMethod1[Fb:Ft, Rb:Rt] = MeanMethod1[Fb:Ft, Rb:Rt] + K*meanm1
            StdMethod1[Fb:Ft, Rb:Rt] = StdMethod1[Fb:Ft, Rb:Rt] + K*stdm1
            
            MeanC.append(meanm1)
            StandardC.append(stdm1)
            
    return MeanC, StandardC, MeanMethod1, StdMethod1


def Method2(discx, discy, n, m, dx, dy, initx, inity, Lx, Ly, slc, FRV, FNM, dZTOR, M, b, dDPP, ZTOR, VS30, dZ10, ni, Finferred, Fmeasured, Tfita, T, xl, xw, L, W, Wpr):
    FigFunct = plt.figure()
    x = np.linspace(initx,initx + Lx, n)
    y = np.linspace(inity,inity + Ly, m)
    lnSa, sigmaT, NLo, sigmaNLo, tau = CY14.LognormalFunct(slc, x, y, FRV, FNM, dZTOR, M, b, dDPP, ZTOR, VS30, dZ10, ni, Finferred, Fmeasured, Tfita, T, xl, xw, L, W, Wpr)
    ResidualBtw = np.random.normal(0,1,1)
    Zerbtw = lnSa + ResidualBtw*tau*(1+NLo)
    MeanF = []
    StdF = []
    MeanMeanGlobal = np.zeros((n*m,n*m))
    StdMeanGlobal = np.zeros((n*m,n*m))
    MeanOfMeanVector = [] 
    StdOfMeanVector  = []
    StatFunct = np.reshape(Zerbtw, (n,m))
    for k in range(discy):
        for j in range(discx):
            Rb = ma.ceil(k*(Ly/discy)/dy)
            Rt = ma.floor((k+1)*(Ly/discy)/dy)+1
            Fb = ma.ceil(j*(Lx/discx)/dx)
            Ft = ma.floor((j+1)*(Lx/discx)/dx)+1
            
            MeanCell = StatFunct[Fb:Ft, Rb:Rt]
            K = np.ones(np.shape(MeanCell))
            MeanCell = MeanCell.reshape(np.size(MeanCell))
            MeanOfMean = np.mean(MeanCell)
            StdOfMean = np.std(MeanCell)
            
            MeanMeanGlobal[Fb:Ft, Rb:Rt] = MeanMeanGlobal[Fb:Ft, Rb:Rt] + K*MeanOfMean
            StdMeanGlobal[Fb:Ft, Rb:Rt] = StdMeanGlobal[Fb:Ft, Rb:Rt] + K*StdOfMean
            
            MeanOfMeanVector.append(MeanOfMean)
            StdOfMeanVector.append(StdOfMean)
            
            MeanStdFun = CreatePlot(FigFunct, k, j, Rb, Rt, Fb, Ft, StatFunct, discx, discy, 0 )
            MeanF.append(MeanStdFun[0])
            StdF.append(MeanStdFun[1])
            
    return MeanOfMeanVector,StdOfMeanVector,MeanMeanGlobal, StdMeanGlobal

def MonteCarloError(ErrorFigure, discx, discy, Lx, Ly, n, m, dx, dy, Lower, slc, initx, inity, FRV, FNM, dZTOR, M, b, dDPP, ZTOR, VS30, dZ10, ni, Finferred, Fmeasured, Tfita, T, xl, xw, W, L, Wpr, iterations):
    x = np.linspace(initx,initx + Lx, n)
    y = np.linspace(inity,inity + Ly, m)
    lnSa, sigmaT, NLo, sigmaNLo, tau = CY14.LognormalFunct(slc, x, y, FRV, FNM, dZTOR, M, b, dDPP, ZTOR, VS30, dZ10, ni, Finferred, Fmeasured, Tfita, T, xl, xw, L, W, Wpr)
    AllCellsErrors = [ [] for _ in range(discy*discx)]
    AllMeanErrors = [ [] for _ in range(discy*discx)]
    AllStdErrors = [ [] for _ in range(discy*discx)]
    for num in range(iterations):
        ErrorSimB = np.random.normal(0,1, n*m)
        ErrorSim = np.reshape(np.matmul(Lower, ErrorSimB)*sigmaNLo, (n,m))
        for k in range(discy):
            for j in range(discx):
                Rb = ma.ceil(k*(Ly/discy)/dy)
                Rt = ma.floor((k+1)*(Ly/discy)/dy)+1
                Fb = ma.ceil(j*(Lx/discx)/dx)
                Ft = ma.floor((j+1)*(Lx/discx)/dx)+1
                
                ErrorCell = ErrorSim[Fb:Ft, Rb:Rt]
                AllMeanErrors[k*(discx) +j].append(np.mean(np.reshape(ErrorCell, np.size(ErrorCell))))
                AllStdErrors[k*(discx) +j].append(np.std(np.reshape(ErrorCell, np.size(ErrorCell))))
                AllCellsErrors[k*(discx) +j].extend(np.reshape(ErrorCell, np.size(ErrorCell)))
                if num == iterations-1:
                    axnu = ErrorFigure.add_subplot(discy, discx, k*(discx) +j +1)
                    axnu.hist(AllCellsErrors[k*(discx) +j], bins= 20)
                    mean = np.matmul(np.transpose(AllMeanErrors[k*(discx) +j]),np.ones(np.size(AllMeanErrors[k*(discx) +j])))/np.size(AllMeanErrors[k*(discx) +j])
                    std = np.matmul(np.transpose(AllStdErrors[k*(discx) +j]),np.ones(np.size(AllMeanErrors[k*(discx) +j])))/np.size(AllMeanErrors[k*(discx) +j])
                    axnu.plot([mean,mean],[0,20], "r-", linewidth = 2)
                    axnu.plot([0,0],[0,20], "k-", linewidth = 2)
                    axnu.text(mean, 0, "Mean = " + str("{:.2f}".format(mean)) + "\n" + "Std = " + str("{:.2f}".format(std)), fontsize = 7)
                    axnu.axis("on")
                else:
                    pass
    return mean, std


def Method1new(discx,discy, Lx, Ly, slc, initx, inity, FRV, FNM, dZTOR, M, b, dDPP, ZTOR, VS30, dZ10, ni, Finferred, Fmeasured, Tfita, T, xl, xw, L, W, Wpr):
    MeanC = []
    StandardC = []
    for k in range(discy):
        for j in range(discx):
            distx = (j+1/2)*(Lx/discx) + initx
            disty = (k+1/2)*(Ly/discy) + inity
            lnSa, sigmaT, NLo, sigmaNLo, tau = CY14.NewLogFunction(slc, distx, disty, FRV, FNM, dZTOR, M, b, dDPP, ZTOR, VS30, dZ10, ni, Finferred, Fmeasured, Tfita, T, xl, xw, L, W, Wpr)
            meanm1 = np.float(lnSa)
            stdm1 = np.float(sigmaNLo)
            
            MeanC.append(meanm1)
            StandardC.append(stdm1)
            
    return MeanC, StandardC

def MonteCarloErrornew(ErrorFigure, discx, discy, Lx, Ly, slc, initx, inity, FRV, FNM, dZTOR, M, b, dDPP, ZTOR, VS30, dZ10, ni, Finferred, Fmeasured, Tfita, T, xl, xw, W, L, Wpr, iterations, select, numpoints):
    AllCellsErrors = [ [] for _ in range(discy*discx)]
    AllMeanErrors = [ [] for _ in range(discy*discx)]
    AllStdErrors = [ [] for _ in range(discy*discx)]
    for num in range(iterations):
        for k in range(discy):
            for j in range(discx):
                x = np.random.uniform(initx, initx+ Lx, numpoints)
                y = np.random.uniform(inity, inity+ Ly, numpoints)
                lnSa, sigmaT, NLo, sigmaNLo, tau = CY14.NewLogFunction(slc, x, y, FRV, FNM, dZTOR, M, b, dDPP, ZTOR, VS30, dZ10, ni, Finferred, Fmeasured, Tfita, T, xl, xw, L, W, Wpr)
                Lower, Cg = Ch.FastMatrixBuilder(select, x,y)
                ErrorSimB = np.random.normal(0,1, np.size(x))
                ErrorSim = np.reshape(np.matmul(Lower, ErrorSimB)*sigmaNLo, (np.size(x)))
                ErrorCell = []
                for i in range(len(x)):
                    if initx + j*Lx/discx < x[i] < initx+ (j + 1)*Lx/discx:
                        if inity + k*Ly/discy < y[i] < inity + (k + 1)*Ly/discy:
                            ErrorCell.append(ErrorSim[i])
                        
                        else:
                            pass
                    else:
                        pass
                AllMeanErrors[k*(discx) +j].append(np.mean(np.reshape(ErrorCell, np.size(ErrorCell))))
                AllStdErrors[k*(discx) +j].append(np.std(np.reshape(ErrorCell, np.size(ErrorCell))))
                AllCellsErrors[k*(discx) +j].extend(np.reshape(ErrorCell, np.size(ErrorCell)))
                if num == iterations-1:
                    axnu = ErrorFigure.add_subplot(discy, discx, k*(discx) +j +1)
                    axnu.hist(AllCellsErrors[k*(discx) +j], bins= 20)
                    mean = np.matmul(np.transpose(AllMeanErrors[k*(discx) +j]),np.ones(np.size(AllMeanErrors[k*(discx) +j])))/np.size(AllMeanErrors[k*(discx) +j])
                    stddev = np.matmul(np.transpose(AllStdErrors[k*(discx) +j]),np.ones(np.size(AllMeanErrors[k*(discx) +j])))/np.size(AllMeanErrors[k*(discx) +j])
                    axnu.plot([mean,mean],[0,20], "r-", linewidth = 2)
                    axnu.plot([0,0],[0,20], "k-", linewidth = 2)
                    axnu.text(mean, 0, "Mean = " + str("{:.2f}".format(mean)) + "\n" + "Std = " + str("{:.2f}".format(stddev)), fontsize = 7)
                    axnu.axis("on")
                else:
                    pass
    return mean, stddev

