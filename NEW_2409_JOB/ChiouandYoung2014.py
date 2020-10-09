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

def LognormalFunct(slc, x, y, FRV, FNM, dZTOR, M, b, dDPP, ZTOR, VS30, dZ10, ni, Finferred, Fmeasured, Tfita, T, xl, xw, L, W, Wpr):
    X,Y = np.meshgrid(x,y)

    X = np.reshape(X, np.size(X))
    Y = np.reshape(Y, np.size(X))
    Z = np.zeros(np.size(X))
    XYZ = np.array([X,Y,Z])
    UVW = np.matmul(T, XYZ)
    UVprW = np.matmul(Tfita, XYZ)
    uc = [max(min(u, (1-xl)*L), -xl*L) for u in UVW[0]]
    vc = [max(min(v, (1-xw)*W), -xw*W) for v in UVW[1]]
    vcpr = [max(min(v, (1-xw)*Wpr), -xw*Wpr) for v in UVprW[1]]
    RRUP = np.sqrt((UVW[0]-uc)**2 + (UVW[1]-vc)**2 + UVW[2]**2)
    RJB = np.sqrt((UVW[0]-uc)**2 + (UVprW[1]-vcpr)**2)
    RX = UVprW[1] + xw*Wpr
    FHW = np.array([(1 if r > 0 else 0) for r in RX])
    
#   ------------- Parameters independent of period ----------------
    c2 = 1.06
    c4 = -2.1
    c4a= -0.5
    crb = 50
    c8a = 0.2695
    c11 = 0

    '''
    Calculation of the reference mean of the lognormally distributed intensity measures
    The parameteric regression model is developed using 6 different families of parameters
    Style of Faulting
    Magnitude Scaling
    Distance Scaling
    Anelastic Attenuation
    Directivity Effects
    Hanging Wall effects
    '''
    
#   ------------- Parameters reflecting Style of Faulting ----------------
        
    P1 = slc.loc["c1"] + (slc.loc["c1a"]+(slc.loc["c1c"]/(np.cosh(2*max(M-4.5,0)))))*FRV
    P2 = slc.loc["c1b"] + slc.loc["c1d"]/np.cosh(2*max(M-4.5,0))*FNM
    P3 = slc.loc["c7"] + slc.loc["c7b"]/np.cosh(2*max(M-4.5,0))*dZTOR
    P4 = c11 + slc.loc["c11b"]/np.cosh(2*max(M-4.5,0))*(np.cos(b))**2
    StyleOfFaulting = P1 + P2 + P3 + P4
    
#   ------------- Parameters reflecting Magnitude Scaling ----------------
    P5 = c2*(M-6)+ (c2-slc.loc["c3"])*np.log(1+np.exp(slc.loc["cn"]*(slc.loc["cM"]-M)))/slc.loc["cn"]
    MagnitudeScaling = P5
    
#   ------------- Parameters reflecting Distance Scaling ----------------
    P6 = c4*np.log(RRUP+slc.loc["c5"]*np.cosh(slc.loc["c6"]*max(M-slc.loc["cHM"],0)))
    P7 = (c4a - c4)*np.log(np.sqrt(RRUP**2+crb**2))
    DistanceScaling = P6 + P7
    
#   ------------- Parameters reflecting Inelastic Attenuation ----------------
    P8 = (slc.loc["cg1"] + slc.loc["cg2"]/np.cosh(max(M-slc.loc["cg3"],0)))*RRUP
    AnelasticAttenuation = P8
    
#   ------------- Parameters reflecting Directivity Effects ----------------
    P9 = np.array([slc.loc["c8"]*max(1-max(rup-40,0)/30,0) for rup in RRUP])
    P10 = min(max(M-5.5,0)/0.8,1)*np.exp(-c8a*(M-slc.loc["c8b"])**2)*dDPP
    DirectivityEffects = P9*P10
    
#   ------------- Parameters reflecting Hanging Wall Effects ----------------
    P11 = slc.loc["c9"]*FHW*np.cos(b)*(slc.loc["c9a"] + (1- slc.loc["c9a"])*np.tanh(RX/slc.loc["c9b"]))*(1-np.sqrt(RJB**2+ZTOR**2)/(RRUP+1))
    HangingWallEffect = P11
    
    yref = np.exp(StyleOfFaulting+ MagnitudeScaling+ DistanceScaling+ AnelasticAttenuation + DirectivityEffects + HangingWallEffect)
    
    '''
    In order to derive the actual mean of the field the reference mean is
    modified to accomodate Site response parameters from 3 different groups
    Linear Site Response parameters
    Nonlinear Site Response parameters
    Sediment Depth Scaling
    '''
    
#   ------------- Parameters reflecting Linear Site Response Effects ----------------
    F1 = slc.loc["f1"]*min(np.log(VS30/1130),0)
    LinearSiteResponse = F1
    
#   ------------- Parameters reflecting Linear Site Response Effects ----------------
    F2 = slc.loc["f2"]*(np.exp(slc.loc["f3"]*min(VS30,1130)-360)-np.exp(slc.loc["f3"]*(1130-360)))*np.log((yref*np.exp(ni)+slc.loc["f4"])/slc.loc["f4"])
    NonlinearSiteResponse = F2
    
#   ------------- Parameters reflecting Sediment Depth Scaling ----------------    
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
    return np.log(y), sigmaT, NLo, sigmaNLo, tau


def NewLogFunction(slc, x, y, FRV, FNM, dZTOR, M, b, dDPP, ZTOR, VS30, dZ10, ni, Finferred, Fmeasured, Tfita, T, xl, xw, L, W, Wpr):
    if np.size(x) > 1:
        z = np.zeros(np.size(x))
        XYZ = [x,y,z]
        UVW = np.matmul(T, XYZ)
        UVprW = np.matmul(Tfita, XYZ)
        uc = [max(min(u, (1-xl)*L), -xl*L) for u in UVW[0]]
        vc = [max(min(v, (1-xw)*W), -xw*W) for v in UVW[1]]
        vcpr = [max(min(v, (1-xw)*Wpr), -xw*Wpr) for v in UVprW[1]]
        RRUP = np.sqrt((UVW[0]-uc)**2 + (UVW[1]-vc)**2 + UVW[2]**2)
        RJB = np.sqrt((UVW[0]-uc)**2 + (UVprW[1]-vcpr)**2)
        RX = UVprW[1] + xw*Wpr
        FHW = np.array([(1 if r > 0 else 0) for r in RX])
    else:
        z = 0
        XYZ = [x,y,z]
        UVW = np.matmul(T, XYZ)
        UVprW = np.matmul(Tfita, XYZ)
        uc = max(min(UVW[0], (1-xl)*L), -xl*L)
        vc = max(min(UVW[1], (1-xw)*W), -xw*W)
        vcpr = max(min(UVprW[1], (1-xw)*Wpr), -xw*Wpr)
        RRUP = np.sqrt((UVW[0]-uc)**2 + (UVW[1]-vc)**2 + UVW[2]**2)
        RJB = np.sqrt((UVW[0]-uc)**2 + (UVprW[1]-vcpr)**2)
        RX = UVprW[1] + xw*Wpr
        if RX >0:
            FHW = 1
        else:
            FHW = 0
    
#   ------------- Parameters independent of period ----------------
    c2 = 1.06
    c4 = -2.1
    c4a= -0.5
    crb = 50
    c8a = 0.2695
    c11 = 0

    '''
    Calculation of the reference mean of the lognormally distributed intensity measures
    The parameteric regression model is developed using 6 different families of parameters
    Style of Faulting
    Magnitude Scaling
    Distance Scaling
    Anelastic Attenuation
    Directivity Effects
    Hanging Wall effects
    '''
    
#   ------------- Parameters reflecting Style of Faulting ----------------
        
    P1 = slc.loc["c1"] + (slc.loc["c1a"]+(slc.loc["c1c"]/(np.cosh(2*max(M-4.5,0)))))*FRV
    P2 = slc.loc["c1b"] + slc.loc["c1d"]/np.cosh(2*max(M-4.5,0))*FNM
    P3 = slc.loc["c7"] + slc.loc["c7b"]/np.cosh(2*max(M-4.5,0))*dZTOR
    P4 = c11 + slc.loc["c11b"]/np.cosh(2*max(M-4.5,0))*(np.cos(b))**2
    StyleOfFaulting = P1 + P2 + P3 + P4
    
#   ------------- Parameters reflecting Magnitude Scaling ----------------
    P5 = c2*(M-6)+ (c2-slc.loc["c3"])*np.log(1+np.exp(slc.loc["cn"]*(slc.loc["cM"]-M)))/slc.loc["cn"]
    MagnitudeScaling = P5
    
#   ------------- Parameters reflecting Distance Scaling ----------------
    P6 = c4*np.log(RRUP+slc.loc["c5"]*np.cosh(slc.loc["c6"]*max(M-slc.loc["cHM"],0)))
    P7 = (c4a - c4)*np.log(np.sqrt(RRUP**2+crb**2))
    DistanceScaling = P6 + P7
    
#   ------------- Parameters reflecting Inelastic Attenuation ----------------
    P8 = (slc.loc["cg1"] + slc.loc["cg2"]/np.cosh(max(M-slc.loc["cg3"],0)))*RRUP
    AnelasticAttenuation = P8
    
#   ------------- Parameters reflecting Directivity Effects ----------------
    if np.size(x) > 1:
        P9 = np.array([slc.loc["c8"]*max(1-max(rup-40,0)/30,0) for rup in RRUP])
    else:
        P9 = slc.loc["c8"]*max(1-max(RRUP-40,0)/30,0)
    P10 = min(max(M-5.5,0)/0.8,1)*np.exp(-c8a*(M-slc.loc["c8b"])**2)*dDPP
    DirectivityEffects = P9*P10
    
#   ------------- Parameters reflecting Hanging Wall Effects ----------------
    P11 = slc.loc["c9"]*FHW*np.cos(b)*(slc.loc["c9a"] + (1- slc.loc["c9a"])*np.tanh(RX/slc.loc["c9b"]))*(1-np.sqrt(RJB**2+ZTOR**2)/(RRUP+1))
    HangingWallEffect = P11
    
    yref = np.exp(StyleOfFaulting+ MagnitudeScaling+ DistanceScaling+ AnelasticAttenuation + DirectivityEffects + HangingWallEffect)
    
    '''
    In order to derive the actual mean of the field the reference mean is
    modified to accomodate Site response parameters from 3 different groups
    Linear Site Response parameters
    Nonlinear Site Response parameters
    Sediment Depth Scaling
    '''
    
#   ------------- Parameters reflecting Linear Site Response Effects ----------------
    F1 = slc.loc["f1"]*min(np.log(VS30/1130),0)
    LinearSiteResponse = F1
    
#   ------------- Parameters reflecting Linear Site Response Effects ----------------
    F2 = slc.loc["f2"]*(np.exp(slc.loc["f3"]*min(VS30,1130)-360)-np.exp(slc.loc["f3"]*(1130-360)))*np.log((yref*np.exp(ni)+slc.loc["f4"])/slc.loc["f4"])
    NonlinearSiteResponse = F2
    
#   ------------- Parameters reflecting Sediment Depth Scaling ----------------    
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
    return np.log(y), sigmaT, NLo, sigmaNLo, tau
