# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:00:19 2020

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
import tkinter as tk
import SourceGeometry as SG

'''
Period Independent regression coefficients:
    c2, c4, c4a, crb, c8a, c11 
'''

c2 = 1.06
c4 = -2.1
c4a= -0.5
crb = 50
c8a = 0.2695
c11 = 0

'''
Period and distance independent inputs: 
    This can be changed to include site-specific parameter changes
    FRV, FNM, ZTOR, M, beta, DPP, FHW, ZTOR, VS30, Z10, W, ni, Fmeasured and Finferred
'''

FRV = SG.FRV
FNM = SG.FNM
dZTOR = SG.dZTOR
M = SG.M
b = SG.beta
dDPP = SG.dDPP
ZTOR = SG.ZTOR
VS30 = SG.VS30
dZ10 = SG.dZ10
W = SG.W
ni = 0
Finferred = 1
Fmeasured = 0

def LognormalFunct(slc, x, y):
    Tfita = SG.Tfita
    T = SG.T
    xl = SG.xl
    xw = SG.xw
    L = SG.L
    W = SG.W
    Wpr = SG.Wpr
    X,Y = np.meshgrid(x,y)

    X = np.reshape(X, np.size(X))
    Y = np.reshape(Y, np.size(X))
    Z = np.zeros(np.size(X))
    XYZ = np.array([X,Y,Z])
    UVW = np.matmul(T, XYZ)
    UVprW = np.matmul(Tfita, XYZ)
    
    uc = [max(min(u, (1-xl)*L), -xl*L) for u in UVW[0]]
    vc = [max(min(v, (1-xw)*W), -xw*W) for v in UVW[0]]
    vcpr = [max(min(v, (1-xw)*Wpr), -xw*Wpr) for v in UVW[0]]
    RRUP = np.sqrt((UVW[0]-uc)**2 + (UVW[1]-vc)**2 + UVW[2]**2)
    RJB = np.sqrt((UVW[0]-uc)**2 + (UVW[1]-vcpr)**2)
    RX = UVprW[1] + xw*Wpr
    FHW = np.array([(1 if r > 0 else 0) for r in RX])
        
    P1 = slc.loc["c1"] + (slc.loc["c1a"]+(slc.loc["c1c"]/(np.cosh(2*max(M-4.5,0)))))*FRV
    P2 = slc.loc["c1b"] + slc.loc["c1d"]/np.cosh(2*max(M-4.5,0))*FNM
    P3 = slc.loc["c7"] + slc.loc["c7b"]/np.cosh(2*max(M-4.5,0))*dZTOR
    P4 = c11 + slc.loc["c11b"]/np.cosh(2*max(M-4.5,0))*(np.cos(b))**2
    StyleOfFaulting = P1 + P2 + P3 + P4
    
    P5 = c2*(M-6)+ (c2-slc.loc["c3"])*np.log(1+np.exp(slc.loc["cn"]*(slc.loc["cM"]-M)))/slc.loc["cn"]
    MagnitudeScaling = P5
    
    P6 = c4*np.log(RRUP+slc.loc["c5"]*np.cosh(slc.loc["c6"]*max(M-slc.loc["cHM"],0)))
    P7 = (c4a - c4)*np.log(np.sqrt(RRUP**2+crb**2))
    DistanceScaling = P6 + P7
    
    P8 = (slc.loc["cg1"] + slc.loc["cg2"]/np.cosh(max(M-slc.loc["cg3"],0)))*RRUP
    AnelasticAttenuation = P8
    
    P9 = np.array([slc.loc["c8"]*max(1-max(rup-40,0)/30,0) for rup in RRUP])
    P10 = min(max(M-5.5,0)/0.8,1)*np.exp(-c8a*(M-slc.loc["c8b"])**2)*dDPP
    DirectivityEffects = P9*P10
    
    P11 = slc.loc["c9"]*FHW*np.cos(b)*(slc.loc["c9a"] + (1- slc.loc["c9a"])*np.tanh(RX/slc.loc["c9b"]))*(1-np.sqrt(RJB**2+ZTOR**2)/(RRUP+1))
    HangingWallEffect = P11
    
    yref = np.exp(StyleOfFaulting+ MagnitudeScaling+ DistanceScaling+ AnelasticAttenuation + DirectivityEffects + HangingWallEffect)
    
#   ------------- Calculate the mean of the distribution ----------------
    F1 = slc.loc["f1"]*min(np.log(VS30/1130),0)
    LinearSiteResponse = F1
    
    F2 = slc.loc["f2"]*(np.exp(slc.loc["f3"]*min(VS30,1130)-360)-np.exp(slc.loc["f3"]*(1130-360)))*np.log((yref*np.exp(ni)+slc.loc["f4"])/slc.loc["f4"])
    NonlinearSiteResponse = F2
    
    F3 = slc.loc["f5"]*(1- np.exp(-dZ10/slc.loc["f6"]))
    SedimentDepthScaling = F3
    
    y = np.exp(np.log(yref)+ ni + LinearSiteResponse + NonlinearSiteResponse + SedimentDepthScaling)
    
#   ------------- Calculate the within event standard deviation of the distribution ----------------
    NLo = slc.loc["f2"]*(np.exp(slc.loc["f3"]*(min(VS30,1130)-360))-np.exp(slc.loc["f3"]*(1130-360)))*(yref/(yref+slc.loc["f4"]))
    sigmaNLo = (slc.loc["s1"] + (slc.loc["s2"]-slc.loc["s1"])*(min(max(M,5),6.5)-5)/1.5)*np.sqrt(slc.loc["s3"]*Finferred + 0.7*Fmeasured + (1+ NLo)**2)
    
    
#   ------------- Calculate the between event standard deviation of the distribution ----------------    
    tau = slc.loc["t1"] + (slc.loc["t2"]-slc.loc["t1"])*(min(max(M,5),6.5)-5)/1.5
    
    
#   ------------- Calculate the total standard deviation of the distribution ----------------    
    sigmaT = np.sqrt((1+NLo)**2*tau**2+sigmaNLo**2)
    return y, sigmaT, NLo, sigmaNLo, tau
