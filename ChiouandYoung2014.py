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
import time
from itertools import combinations
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

import matplotlib.animation as animation
import tkinter as tk


start_time = time.time()



'''
System Paramters:
    M = Moment magnitude.
    RRUP = Closest distance (km) to the ruptured plane.
    RJB = Closest distance (km) to the surface projection of ruptured plane.
    RX = Site coordinate (km) measured perpendicular to the fault strike from the
    fault line, with the down-dip direction being positive.
    FHW = Hanging-wall flag: 1 for RX ≥ 0 and 0 for RX < 0.
    b = Fault dip angle.
    ZTOR = Depth (km) to the top of ruptured plane.
    dZTOR = ZTOR centered on the M-dependent average ZTOR (km).
    FRV = Reverse-faulting flag: 1 for 30° ≤ λ ≤ 150° (combined reverse and
    reverse-oblique), 0 otherwise; λ is the rake angle.
    FNM = Normal faulting flag: 1 for −120° ≤ λ ≤ −60° (excludes normal-oblique),
    0 otherwise.
    VS30 = Travel-time averaged shear-wave velocity (m∕s) of the top 30 m of soil.
    Z10 = Depth (m) to shear-wave velocity of 1.0 km∕s.
    dZ10 = Z1.0 centered on the VS30-dependent average Z1.0 (m).
    DPP = Direct point parameter for directivity effect.
    dDPP = DPP centered on the site- and earthquake-specific average DPP.
'''



region = "US"
M = 6
lambangle = 120

data = pd.read_csv("C:\\Users\\StoyanBoyukliyski\\OneDrive\\Desktop\\MScDissertation\PythonFiles\\RegressionCoefficients.csv")
data.head()
data = data.set_index("Period(s)")

if lambangle >= 30 and lambangle <= 150:
    Fault = "Reverse"
    FNM = 0
    FRV = 1
    a = -1.61
    b = 0.41
    eZTOR = np.max(2.704 - 1.226*np.max(M-5.849,0))**2
    if M<3.5 or M>8.5:
        raise(ValueError("The Magnitude is out of the range of applicability"))
    else:
        pass
    
elif lambangle >= 240 and lambangle <= 300:
    Fault = "Normal"
    FRV = 1
    FNM = 0
    a = -1.14
    b = 0.35
    eZTOR = np.max(2.673-1.136*np.max(M-4.970,0))**2
    if M<3.5 and M>8.0:
        raise(ValueError("The Magnitude is out of the range of applicability"))
    else:
        pass
    
    
elif (lambangle >= 330 and lambangle <= 30) or (lambangle >=150 and lambangle <= 210):
    Fault = "Strike-Slip"
    a = -0.76
    b = 0.27
    FRV = 0
    FNM = 0
    eZTOR = np.max(2.673-1.136*np.max(M-4.970,0))**2
    if M<3.5 and M>8.0:
        raise(ValueError("The Magnitude is out of the range of applicability"))
    else:
        pass
    
elif (lambangle >=210 and lambangle <= 240) or (lambangle >=300 and lambangle <= 330):
    Fault = "Normal-Oblique"
    a = -1.14
    b = 0.35
    FRV = 0
    FNM = 0
    eZTOR = np.max(2.673-1.136*np.max(M-4.970,0))**2
    if M<3.5 and M>8.0:
        raise(ValueError("The Magnitude is out of the range of applicability"))
    else:
        pass
else:
    ValueError("Lambda should be >=0 and <=360")
    
dZTOR = 0    
ZTOR = eZTOR + dZTOR
W = np.exp(a + b*M)

b = 45
b = np.pi*b/180

VS30 = 760

DPP = 0
dDPP = 0
Finferred = 0
Fmeasured = 1


if VS30 < 180 or VS30 > 1500:
    raise(ValueError("The Shear Wave Velocity is out of the range of applicability"))
else:
    pass

if region == "Japan":
    eZ10 = np.exp((-5.23/2)*np.log((VS30**2+412**2)/(1360**2+412**2)))
else:
    eZ10 = np.exp((-7.15/4)*np.log((VS30**4 + 571**4)/(1360**4+571**4)))
    
dZ10 = 0
Z10 = dZ10 + eZ10
ni = 0

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

data = pd.read_csv("C:\\Users\\StoyanBoyukliyski\\OneDrive\\Desktop\\MScDissertation\PythonFiles\\RegressionCoefficients.csv")
data.head()
data = data.set_index("Period(s)")


'''

Calculation of the mean value of the PGA or PGV 
or other output parameter chosen!

'''


def annotate_dim(ax,xyfrom,xyto,text=None):

    if text is None:
        text = str(np.sqrt( (xyfrom[0]-xyto[0])**2 + (xyfrom[1]-xyto[1])**2 ))

    ax.annotate("",xyfrom,xyto,arrowprops=dict(arrowstyle='<->'))
    ax.text((xyto[0]+xyfrom[0])/2,(xyto[1]+xyfrom[1])/2,text,fontsize=10, horizontalalignment='center', verticalalignment='bottom')

def Part1(c1, c1a, c1c):
    P1 = c1 + (c1a+(c1c/(np.cosh(2*max(M-4.5,0)))))*FRV
    return P1

def Part2(c1b, c1d):
    P2 = c1b + c1d/np.cosh(2*max(M-4.5,0))*FNM
    return P2

def Part3(c7,c7b):
    P3 = c7 + c7b/np.cosh(2*max(M-4.5,0))*dZTOR
    return P3

def Part4(c11, c11b):
    P4 = c11 + c11b/np.cosh(2*max(M-4.5,0))*(np.cos(b))**2
    return P4

def Part5(c2,c3,cn,cm):
    P5 = c2*(M-6)+ (c2-c3)*np.log(1+np.exp(cn*(cm-M)))/cn
    return P5

def Part6(c4, c5, c6, chm):
    P6 = c4*np.log(RRUP+c5*np.cosh(c6*max(M-chm,0)))
    return P6

def Part7(c4a, c4, crb):
    P7 = (c4a - c4)*np.log(np.sqrt(RRUP**2+crb**2))
    return P7

def Part8(cy1, cy2, cy3):
    P8 = (cy1 + cy2/np.cosh(max(M-cy3,0)))*RRUP
    return P8

def Part9(c8):
    P9 = c8*max(1-max(RRUP-40,0)/30,0)
    return P9

def Part10(c8a, c8b):
    P10 = min(max(M-5.5,0)/0.8,1)*np.exp(-c8a*(M-c8b)**2)*dDPP
    return P10

def Part11(c9, c9a, c9b):
    P11 = c9*FHW*np.cos(b)*(c9a + (1- c9a)*np.tanh(RX/c9b))*(1-np.sqrt(RJB**2+ZTOR**2)/(RRUP+1))
    return P11

def PartF1(fi):
        F1 = fi*min(np.log(VS30/1130),0)
        return F1
    
def PartF2(f2, f3, f4,yref):
    F2 = f2*(np.exp(f3*min(VS30,1130)-360)-np.exp(f3*(1130-360)))*np.log((yref*np.exp(ni)+f4)/f4)
    return F2

def PartF3(f5, f6):
    F3 = f5*(1- np.exp(-dZ10/f6))
    return F3

def CalculateYref(slc):
    StyleOfFaulting = Part1(slc.loc["c1"], slc.loc["c1a"], slc.loc["c1c"]) + Part2(slc.loc["c1b"], slc.loc["c1d"]) + Part3(slc.loc["c7"],slc.loc["c7b"]) + Part4(c11, slc.loc["c11b"])
    MagnitudeScaling = Part5(c2,slc.loc["c3"],slc.loc["cn"],slc.loc["cM"])
    DistanceScaling = Part6(c4, slc.loc["c5"], slc.loc["c6"], slc.loc["cHM"]) + Part7(c4a, c4, crb)
    AnelasticAttenuation = Part8(slc.loc["cg1"], slc.loc["cg2"], slc.loc["cg3"])
    DirectivityEffects = Part9(slc.loc["c8"])*Part10(c8a, slc.loc["c8b"])
    HangingWallEffect = Part11(slc.loc["c9"], slc.loc["c9a"], slc.loc["c9b"])
    
    yref = np.exp(StyleOfFaulting+ MagnitudeScaling+ DistanceScaling+ AnelasticAttenuation + DirectivityEffects + HangingWallEffect)
    return yref

def CalculateY(slc, R):
    global RX
    global RJB
    global FHW
    global RRUP
    RX = R
    if RX >=0:
        if RX < W*np.cos(b):
            RJB = 0
        else:
            RJB = RX - W*np.cos(b)
        FHW = 1
        RRUP = RX*np.sin(b) + ZTOR*np.cos(b)
        if RRUP < 0 or RRUP > 300:
            raise(ValueError("The Distance is out of the range of applicability"))
        else:
            pass
        
        def plotter():
            figure = plt.figure(1,figsize = (15,10))
            ax1 = figure.add_subplot(121)
            ax1.plot([-RX*0.05,RX*1.05], [0,0], "g-", linewidth = 0.5)
            annotate_dim(plt.gca(),[0,0.5],[RX,0.5],"Rx = " + str(RX))
            annotate_dim(plt.gca(),[0-0.5,-ZTOR-0.5],[W*np.cos(b)-0.5,-ZTOR-W*np.sin(b)-0.5],"W = " + str("{:.2f}".format(W)))
            annotate_dim(plt.gca(),[RX,0],[-ZTOR*np.sin(b)*np.cos(b)+RX*np.cos(b)**2,-ZTOR*np.cos(b)**2 - RX*np.sin(b)*np.cos(b)],"Rrup = " + str("{:.2f}".format(RRUP)))
            annotate_dim(plt.gca(),[-1,-ZTOR],[-1,0],"Ztor = " + str("{:.2f}".format(ZTOR)))
            plt.axis("equal")
            '''
            ax1.text(RX+3,-(ZTOR + W)/2, "Region= " + region + "\n" + "Fault Type = " + 
                     Fault + " Type" + "\n" + "Rx = " + str(RX) + "m" + "\n" +
                     "Rrup = " + str("{:.2f}".format(RRUP)) + "m" + "\n" +
                     "Rjb = " + str(RJB) + "m" + "\n" + "M = " + str(M) +
                     "\n" +"Z1,0 = " + str("{:.2f}".format(Z10)) + "m" + "\n" + "Ztor = " +
                     str("{:.2f}".format(ZTOR)) + "m" +  "\n" + "VS30 = " + str("{:.2f}".format(VS30)) +
                     "m/s" + "\n" +  "DIP = " + str("{:.2f}".format(b*180/np.pi)) +
                     "deg" ,bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5})
            '''
            annotate_dim(plt.gca(),[W*np.cos(b),0],[RX,0],"Rjb = " + str(RJB))
        
            
            ax1.plot([0,W*np.cos(b)],[0,0], "g-" ,linewidth = 6)
            ax1.plot([0,W*np.cos(b)],[-ZTOR, -ZTOR-W*np.sin(b)], "g-", linewidth = 10)
            ax1.plot(RX,0, "bo", linewidth = 15)
        
    elif RX < 0:
        RJB = -RX
        FHW = 0
        RRUP = np.sqrt(ZTOR**2 + RJB**2)
        if RRUP < 0 or RRUP > 300:
            raise(ValueError("The Distance is out of the range of applicability"))
        else:
            pass
        def plotter():
            figure = plt.figure(1,figsize = (15,10))
            ax1 = figure.add_subplot(121)
            ax1.plot([-RX*0.05,RX*1.05], [0,0], "g-", linewidth = 0.5)
            annotate_dim(plt.gca(),[0,0.5],[RX,0.5],"Rx = Rjb = " + str(RX))
            annotate_dim(plt.gca(),[0-0.5,-ZTOR-0.5],[W*np.cos(b)-0.5,-ZTOR-W*np.sin(b)-0.5],"W = " + str("{:.2f}".format(W)))
            annotate_dim(plt.gca(),[RX,0],[0, -ZTOR],"Rrup = " + str("{:.2f}".format(RRUP)))
            annotate_dim(plt.gca(),[-1,-ZTOR],[-1,0],"Ztor = " + str("{:.2f}".format(ZTOR)))
            plt.axis("equal")
            '''
            ax1.text(RX+3,-(ZTOR + W)/1.3, "Region= " + region + "\n" + "Fault Type = " + 
                     Fault + " Type" + "\n" + "Rx = " + str(RX) + "m" + "\n" +
                     "Rrup = " + str("{:.2f}".format(RRUP)) + "m" + "\n" +
                     "Rjb = " + str(RJB) + "m" + "\n" + "M = " + str(M) +
                     "\n" +"Z1,0 = " + str("{:.2f}".format(Z10)) + "m" + "\n" + "Ztor = " +
                     str("{:.2f}".format(ZTOR)) + "m" +  "\n" + "VS30 = " + str("{:.2f}".format(VS30)) +
                     "m/s" + "\n" +  "DIP = " + str("{:.2f}".format(b)) +
                     "deg" ,bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5})
            '''
            
            ax1.plot([0,W*np.cos(b)],[0,0], "g-" ,linewidth = 6)
            ax1.plot([0,W*np.cos(b)],[-ZTOR, -ZTOR-W*np.sin(b)], "g-", linewidth = 10)
            ax1.plot(RX,0, "bo", linewidth = 15)
            

    yref = CalculateYref(slc)
    LinearSiteResponse = PartF1(slc.loc["f1"])
    NonlinearSiteResponse = PartF2(slc.loc["f2"], slc.loc["f3"], slc.loc["f4"],yref)
    SedimentDepthScaling = PartF3(slc.loc["f5"], slc.loc["f6"])
    
    y = np.exp(np.log(yref)+ ni + LinearSiteResponse + NonlinearSiteResponse + SedimentDepthScaling)
    return y

def CalcStdWithin(slc):
    yref = CalculateYref(slc)
    NLo = slc.loc["f2"]*(np.exp(slc.loc["f3"]*(min(VS30,1130)-360))-np.exp(slc.loc["f3"]*(1130-360)))*(yref/(yref+slc.loc["f4"]))
    sigmaNLo = (slc.loc["s1"] + (slc.loc["s2"]-slc.loc["s1"])*(min(max(M,5),6.5)-5)/1.5)*np.sqrt(slc.loc["s3"]*Finferred + 0.7*Fmeasured + (1+ NLo)**2)
    return NLo, sigmaNLo

def CalcStdBetween(slc):
    tau = slc.loc["t1"] + (slc.loc["t2"]-slc.loc["t1"])*(min(max(M,5),6.5)-5)/1.5
    return tau

def CalculateStd():
    data = pd.read_csv("C:\\Users\\StoyanBoyukliyski\\OneDrive\\Desktop\\MScDissertation\PythonFiles\\RegressionCoefficients.csv")
    data.head()
    data = data.set_index("Period(s)")
    
    select = str(0.1)
    slc = data.loc[select]
    (NLo, sigmaNLo) = CalcStdWithin(slc)
    tau = CalcStdBetween(slc)
    sigmaT = np.sqrt((1+NLo)**2*tau**2+sigmaNLo**2)
    return sigmaT

