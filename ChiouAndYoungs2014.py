# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
M = 7.5
FRV = 0
FNM = 0
RJB = 10

if FRV == 1:
    eZTOR = np.max(2.704 - 1.226*np.max(M-5.849,0))**2
else:
    eZTOR = np.max(2.673-1.136*np.max(M-4.970,0))**2
    
dZTOR = 0
ZTOR = eZTOR + dZTOR
RRUP = np.sqrt(RJB**2 + ZTOR**2)
RX = RJB
b = 90
VS30 = 760
DPP = 0
dDPP = 0
Finferred = 0
Fmeasured = 1

if RRUP < 0 or RRUP > 300:
    raise(ValueError("The Distance is out of the range of applicability"))
else:
    pass

if FRV == 1:
    FNM = 0
    FHW = 0
else:
    FNM = 1
    FHW = 1
    

if FRV == 1:
    if M<3.5 or M>8.5:
        raise(ValueError("The Magnitude is out of the range of applicability"))
    else:
        pass
else:
    if M<3.5 and M>8.0:
        raise(ValueError("The Magnitude is out of the range of applicability"))
    else:
        pass

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


data = pd.read_csv("RegressionCoefficients.csv")
data.head()
data = data.set_index("Period(s)")


'''

Calculation of the mean value of the PGA or PGV 
or other output parameter chosen!

'''

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

def CalculateYref(slc,R):
    global RJB
    RJB = R
    global RX 
    RX = RJB
    global RRUP
    RRUP = np.sqrt(R**2 + ZTOR**2)
    StyleOfFaulting = Part1(slc.loc["c1"], slc.loc["c1a"], slc.loc["c1c"]) + Part2(slc.loc["c1b"], slc.loc["c1d"]) + Part3(slc.loc["c7"],slc.loc["c7b"]) + Part4(c11, slc.loc["c11b"])
    MagnitudeScaling = Part5(c2,slc.loc["c3"],slc.loc["cn"],slc.loc["cM"])
    DistanceScaling = Part6(c4, slc.loc["c5"], slc.loc["c6"], slc.loc["cHM"]) + Part7(c4a, c4, crb)
    AnelasticAttenuation = Part8(slc.loc["cg1"], slc.loc["cg2"], slc.loc["cg3"])
    DirectivityEffects = Part9(slc.loc["c8"])*Part10(c8a, slc.loc["c8b"])
    HangingWallEffect = Part11(slc.loc["c9"], slc.loc["c9a"], slc.loc["c9b"])
    
    yref = np.exp(StyleOfFaulting+ MagnitudeScaling+ DistanceScaling+ AnelasticAttenuation + DirectivityEffects + HangingWallEffect)
    return yref

def CalculateY(yref):
    LinearSiteResponse = PartF1(slc.loc["f1"])
    NonlinearSiteResponse = PartF2(slc.loc["f2"], slc.loc["f3"], slc.loc["f4"],yref)
    SedimentDepthScaling = PartF3(slc.loc["f5"], slc.loc["f6"])
    
    y = np.exp(np.log(yref)+ ni + LinearSiteResponse + NonlinearSiteResponse + SedimentDepthScaling)
    return y

def CalcStdWithin(yref):
    NLo = slc.loc["f2"]*(np.exp(slc.loc["f3"]*(min(VS30,1130)-360))-np.exp(slc.loc["f3"]*(1130-360)))*(yref/(yref+slc.loc["f4"]))
    sigmaNLo = (slc.loc["s1"] + (slc.loc["s2"]-slc.loc["s1"])*(min(max(M,5),6.5)-5)/1.5)*np.sqrt(slc.loc["s3"]*Finferred + 0.7*Fmeasured + (1+ NLo)**2)
    return NLo, sigmaNLo

def CalcStdBetween():
    tau = slc.loc["t1"] + (slc.loc["t2"]-slc.loc["t1"])*(min(max(M,5),6.5)-5)/1.5
    return tau

def CalculateStd():
    (NLo, sigmaNLo) = CalcStdWithin()
    tau = CalcStdBetween()
    sigmaT = np.sqrt((1+NLo)**2*tau**2+sigmaNLo**2)
    return sigmaT

'''

The exercise is the following:
    We have an area of Lx km x Ly km
    It is divided into n segments in Lx
    and m segments into Ly 
    then for each node the IM is calculated
    along with the standard deviation.
    Also, the correlation matrix is assembled.

'''


Lx = 10
Ly = 10
n= 50    
m = 50
initdist = 12
dx = Lx/(n-1)
dy = Ly/(m-1)

select = str(0.1)
slc = data.loc[select]
x = np.linspace(initdist, Lx + initdist, n)
y = np.linspace(0, Ly, m)

if float(select) < 1:
    parb = 8.5 + 17.2*float(select)
else:
    parb = 22 + 3.7*float(select)

def Rho(h):
    rho = np.exp(-3*h/parb)
    return rho

correlation = plt.figure(1)
plt.plot(np.linspace(0,parb,1000), [Rho(i) for i in np.linspace(0,parb,1000)])



Cg = np.zeros((n*m,n*m))
C= np.zeros((n,n))
matrices = []

time_one = time.time()
print("--- %s seconds -- For Problem definition ---" % (time_one - start_time))

for z in range(m):
        for i in range(n):
            for j in range(i,n):
                C[i,j] = Rho(np.sqrt(((z)*dy)**2 + (dx*(j-i))**2))       
        matrices.append(np.array(C))
        
time_two = time.time()
print("--- %s seconds -- For Creation of small matrices ---" % (time_two - time_one))

for k in range(m):
    Cg[(k)*n:(k+1)*n, (k)*n:(k+1)*n] = matrices[0]
    for z in range(k+1,m):
        C =  matrices[z-k]
        Cr = C + np.transpose(C) - np.diag(np.diag(C))
        Cg[(k)*n:(k+1)*n, (z)*n:(z+1)*n] = Cr
        
del matrices
Cg = Cg + np.transpose(Cg) - np.identity(n*m)  
time_three = time.time()
print("--- %s seconds -- For Assembly into the big matrix ---" % (time_three - time_two))

X1 = np.random.normal(0,1, n*m)
fig2 = plt.figure(2)
ax1 = fig2.add_subplot(111)
ax1.hist(X1, bins = 40)
ax1.set_title("Uncorellated Residual Sampling", {'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'})
ax1.set_ylabel("Number of samples in Bin",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad=20)
ax1.set_xlabel("Residual Value",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)
ResidualBtw = np.random.normal(0,1,1)

time_four = time.time()
print("--- %s seconds -- For Generation of random variables ---" % (time_four - time_three))

L = np.linalg.cholesky(Cg)

time_five = time.time()
print("--- %s seconds -- For Cholesky Decomposition ---" % (time_five - time_four))
Mean = []
StdWth = []
StdBtw = []
between = CalcStdBetween()
for j in x:
    yref = CalculateYref(slc,j)
    meanval = CalculateY(yref)
    (NLo, within) = CalcStdWithin(yref)
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
'''
fig3 = plt.figure(4)
ax5 = fig3.add_subplot(111)
ax5.hist(Yer, bins = 40)
ax5.set_title("Correlated Residual Sampling", {'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'})
ax5.set_ylabel("Number of samples in Bin",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad=20)
ax5.set_xlabel("Residual Value",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)
'''
time_six = time.time()
print("--- %s seconds -- For development of Mean and Standard Deviations ---" % (time_six - time_five))

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

time_seven = time.time()
print("--- %s seconds -- For Reshaping of vectors ---" % (time_seven - time_six))

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

time_eight = time.time()
print("--- %s seconds -- For Development of vectors for field plotting ---" % (time_eight - time_seven))

print("--- %s seconds -- Total Time passed ---" % (time_eight - start_time))