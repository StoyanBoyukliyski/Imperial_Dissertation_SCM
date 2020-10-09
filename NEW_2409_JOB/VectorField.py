# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 14:46:56 2020

@author: StoyanBoyukliyski
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Cholesky as Ch
import ChiouandYoung2014 as CY14
import PlotGeometry as PG

def CreateVectorField(Lx, Ly, n, m, initx, inity, dx, dy, Lower, T, Tfita, beta, W, L, xl, xw, slc, select, Wpr, ZTOR, lambangle,FRV, FNM, dZTOR, M, dDPP, VS30, dZ10, ni, Finferred, Fmeasured):
    x = np.linspace(initx,initx + Lx, n)
    y = np.linspace(inity,inity + Ly, m)

    scale = 3
    
    lnSa, sigmaT, NLo, sigmaNLo, tau = CY14.LognormalFunct(slc, x, y, FRV, FNM, dZTOR, M, beta, dDPP, ZTOR, VS30, dZ10, ni, Finferred, Fmeasured, Tfita, T, xl, xw, L, W, Wpr)
    ResidualWth = np.random.normal(0,1, n*m)
    ResidualBtw = np.random.normal(0,1,1)

    ResidualWth = np.matmul(Lower,ResidualWth)
    
    Res = ResidualWth*sigmaNLo
    Zerbtw = lnSa + ResidualBtw*tau*(1+NLo)
    Zerwth = Zerbtw + Res
    
    ax = PG.PlotGeometryF(L,W,ZTOR, beta, lambangle, T, Tfita, xw, xl, initx, inity, Lx, Ly, n, m, Wpr)
    ax.set_zlim([-W,np.max(Zerwth)*scale])
    X,Y = np.meshgrid(x,y)
    Mean = np.reshape(lnSa, np.shape(X))
    Zerbtw = np.reshape(Zerbtw, np.shape(X))
    Zerwth = np.reshape(Zerwth, np.shape(X))
    Res= np.reshape(Res, np.shape(X))
    
    if m == 1 or n == 1:
        pass
    else:
        ax.plot_surface(X, Y, Zerbtw*scale, cmap = "viridis")
#        ax.plot_wireframe(X, Y, Zerwth , cmap = "viridis")
        ax.legend(loc = "upper right", prop = {"size": 7})
        ax.grid(linestyle = "--")
        ax.set_title("3D plot of Intensity Measures", {'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'})
        ax.set_zlabel("Log of Intesity Measure (log(g))",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)
        ax.set_ylabel("Distance in North (km)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad=20)
        ax.set_xlabel("Distance in East (km)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)
    
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(X, Y, Zerbtw, cmap = "viridis", linewidth = 0.6)
#    ax.plot_wireframe(X, Y, Zerwth, color = "red", linewidth = 1, label = "Mean + Btw + Wth")
    
    ax.legend(loc = "upper right", prop = {"size": 7})
    ax.grid(linestyle = "--")
    ax.set_title("3D plot of Intensity Measures", {'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'})
    ax.set_zlabel("Log of Intesity Measure (log(g))",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)
    ax.set_ylabel("Distance in North (km)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad=20)
    ax.set_xlabel("Distance in East (km)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)
    
    ax1 = fig.add_subplot(2, 2, 2)
    ax1.plot(X[0, :n],np.exp(Mean[0, :n]), color = "black", linewidth = 0.6, label = "Mean Value")
    ax1.plot(X[0, :n],np.exp(Zerbtw[0, :n]), color = "green",  linewidth = 0.6, label = "Mean + Btw Residual")
    ax1.plot(X[0, :n],np.exp(Zerwth[0, :n]), color = "red",linewidth = 1, label = "Mean + Wth + Btw Residual")
    ax1.grid(linestyle = "--")
    ax1.set_yscale("log")
    ax1.set_title("Intensity Measures 2D", {'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'})
    ax1.set_xlabel("Distance in North (km)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad=20)
    ax1.set_ylabel("Intesity Measure (g)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)
    ax1.legend(loc = "upper right", prop = {"size": 7})
    ax2 = fig.add_subplot(2, 2, 4)
    
    if m == 1 or n == 1:
        pass
    else:
        CS = ax2.contourf(X,Y,Res, cmap = "viridis")
        ax2.contour(X, Y, Res, cmap = "viridis")
        fig.colorbar(CS)
        ax2.set_title("Contour Plot of Residuals", {'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'})
        ax2.set_ylabel("Distance in North (km)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad=20)
        ax2.set_xlabel("Distance in East (km)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)
    plt.tight_layout()
    
    lnSa =  np.reshape(lnSa, np.shape(X))
    sigmaNLo = np.reshape(sigmaNLo, np.shape(X))
    tauNLO = tau*(1+NLo)
    tauNLO =  np.reshape(tauNLO, np.shape(X))
    fignew = plt.figure()
    ax = fignew.add_subplot(111)
    ax.plot(X[int(m/2), :n], lnSa[int(m/2), :n], "b-", label = "Mean")
    ax.plot(X[int(m/2), :n], sigmaNLo[int(m/2), :n], "g-", label = "Standard Dev Within")
    ax.plot(X[int(m/2), :n], tauNLO[int(m/2), :n], "k-", label = "Standard Dev Between")
    plt.tight_layout()
    return Zerwth, Res

def CorrelationRepresentation(Lx, Ly, n, m, initx, inity, dx, dy, Lower, T, Tfita, beta, W, L, xl, xw, slc, select, Wpr, ZTOR, lambangle,FRV, FNM, dZTOR, M, dDPP, VS30, dZ10, ni, Finferred, Fmeasured):
    x = np.linspace(initx,initx + Lx, n)
    y = np.linspace(inity,inity + Ly, m)
    lnSa, sigmaT, NLo, sigmaNLo, tau = CY14.LognormalFunct(slc, x, y, FRV, FNM, dZTOR, M, beta, dDPP, ZTOR, VS30, dZ10, ni, Finferred, Fmeasured, Tfita, T, xl, xw, L, W, Wpr)
    fig = plt.figure()
    listoflist = [[] for _ in range(n*m)]
    f = 0
    num = 1000
    while f <num:
        ErrorX = np.random.normal(0,1, n*m)
        R = np.random.normal(0,1,1)
        ErrorY = np.matmul(Lower,ErrorX)
        lnSaf = lnSa + R*tau*(1+NLo) + ErrorY*sigmaNLo
        lnSaf = np.reshape(lnSaf, np.size(lnSaf))
        for i in range(n*m):
            listoflist[i].append(lnSaf[i])
            if f == num-1:
                for j in range(n*m):
                    ax = fig.add_subplot(n*m,n*m, i*n*m +j + 1)
                    if i == j:
                        ax.hist(listoflist[i][:(num-1)], color = "blue")
                        ax.set_xlabel("Variable " + str(i))
                    else:
                        ax.scatter(listoflist[i][:(num-1)], listoflist[j][:(num-1)], s  = 0.5, color = "blue")
                        ax.set_aspect('equal')
                        ax.set_xlabel("Variable " + str(i))
                        ax.set_ylabel("Variable " + str(j))
            else:
                pass
        plt.tight_layout()
        f = f + 1
        
def NewVectorField(x, y, initx, inity, Lower, T, Tfita, beta, W, L, xl, xw, slc, select, Wpr, ZTOR, lambangle,FRV, FNM, dZTOR, M, dDPP, VS30, dZ10, ni, Finferred, Fmeasured):
    scale = 5
    lnSa, sigmaT, NLo, sigmaNLo, tau = CY14.NewLogFunction(slc, x, y, FRV, FNM, dZTOR, M, beta, dDPP, ZTOR, VS30, dZ10, ni, Finferred, Fmeasured, Tfita, T, xl, xw, L, W, Wpr)
    ResidualBtw = np.random.normal(0,1,1)
    ResidualWth = np.random.normal(0,1, np.size(x))
    ResidualWth = np.matmul(Lower,ResidualWth)
    Res = ResidualWth*sigmaNLo
    Zerbtw = lnSa + ResidualBtw*tau*(1+NLo)
    Zerwth = Zerbtw + Res
    ax = PG.NewPlotGeometry(L,W,ZTOR, beta, lambangle, T, Tfita, xw, xl, initx, inity, x, y, Wpr)
    ax.plot_trisurf(x, y, Zerwth, cmap='viridis', linewidth=0.5)
    ax.legend(loc = "upper right", prop = {"size": 7})
    ax.grid(linestyle = "--")
    ax.set_title("3D plot of Intensity Measures", {'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'})
    ax.set_zlabel("Log of Intesity Measure log(g)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)
    ax.set_ylabel("Distance in North (km)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad=20)
    ax.set_xlabel("Distance in East (km)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)
    ax.set_zlim([-W,np.max(Zerwth)*scale])
    if len(x) > 2:
        if np.max(x) == 0 or np.max(y) == 0:
            pass
        else:
            for j in range(100):
                ResidualWth = np.random.normal(0,1, np.size(x))
                ResidualWth = np.matmul(Lower,ResidualWth)
                Res = ResidualWth*sigmaNLo
                Zerbtw = lnSa + ResidualBtw*tau*(1+NLo)
                Zerwth = Zerbtw + Res
                plt.cla()
                ax = PG.NewPlotGeometry(L,W,ZTOR, beta, lambangle, T, Tfita, xw, xl, initx, inity, x, y, Wpr)
                ax.plot_trisurf(x, y, Zerwth, cmap='viridis', linewidth=0.5)
                plt.pause(0.0005)
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        ax1 = fig.add_subplot(222, projection='3d')
        ax2 = fig.add_subplot(224)
        CS =ax2.scatter(x,y,c = Res, cmap = 'viridis', s = 0.5)
        ax2.set_title("Contour Plot of Residuals", {'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'})
        ax2.set_ylabel("Distance in North (km)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad=20)
        ax2.set_xlabel("Distance in East (km)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)
        
        fig.colorbar(CS)
        ax.plot_trisurf(x, y, Zerwth, cmap='viridis', linewidth=0.5)
        ax1.plot_trisurf(x, y, Zerbtw, cmap='viridis', linewidth = 1)
        ax.grid(linestyle = "--")
        ax.set_title("3D plot of Intensity Measures", {'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'})
        ax.set_zlabel("Log of Intesity Measure log(g)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)
        ax.set_ylabel("Distance in North (km)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad=20)
        ax.set_xlabel("Distance in East (km)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)
        ax1.grid(linestyle = "--")
        ax1.set_title("3D plot of Intensity Measures", {'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'})
        ax1.set_zlabel("Log of Intesity Measure log(g)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)
        ax1.set_ylabel("Distance in North (km)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad=20)
        ax1.set_xlabel("Distance in East (km)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)
        plt.tight_layout()
        
        fignew = plt.figure()
        ax = fignew.add_subplot(111, projection='3d')
        ax.plot_trisurf(x, y, lnSa, cmap='viridis', linewidth=0.5, label = "Mean")
        ax.plot_trisurf(x, y, sigmaNLo, cmap='viridis', linewidth=0.5, label = "Standard Dev Within")
        ax.plot_trisurf(x, y, tau*(1+NLo), cmap='viridis', label = "Standard Dev Between")
        plt.tight_layout()
    else:
        pass
    
def CorrelationRepresentationScatter(x, y, initx, inity, Lower, T, Tfita, beta, W, L, xl, xw, slc, select, Wpr, ZTOR, lambangle,FRV, FNM, dZTOR, M, dDPP, VS30, dZ10, ni, Finferred, Fmeasured):
    lnSa, sigmaT, NLo, sigmaNLo, tau = CY14.NewLogFunction(slc, x, y, FRV, FNM, dZTOR, M, beta, dDPP, ZTOR, VS30, dZ10, ni, Finferred, Fmeasured, Tfita, T, xl, xw, L, W, Wpr)
    fig = plt.figure()
    listoflist = [[] for _ in range(np.size(x))]
    f = 0
    num = 1000
    while f <num:
        ErrorX = np.random.normal(0,1, np.size(x))
        R = np.random.normal(0,1,1)
        ErrorY = np.matmul(Lower,ErrorX)
        lnSaf = lnSa + R*tau*(1+NLo) + ErrorY*sigmaNLo
        lnSaf = np.reshape(lnSaf, np.size(lnSaf))
        for i in range(np.size(x)):
            listoflist[i].append(lnSaf[i])
            if f == num-1:
                for j in range(np.size(x)):
                    ax = fig.add_subplot(np.size(x),np.size(x), i*np.size(x) +j + 1)
                    if i == j:
                        ax.hist(listoflist[i][:(num-1)], color = "blue")
                        ax.set_xlabel("Variable " + str(i))
                    else:
                        ax.scatter(listoflist[i][:(num-1)], listoflist[j][:(num-1)], s  = 0.5, color = "blue")
                        ax.set_aspect('equal')
                        ax.set_xlabel("Variable " + str(i))
                        ax.set_ylabel("Variable " + str(j))
            else:
                pass
        plt.tight_layout()
        f = f + 1
    