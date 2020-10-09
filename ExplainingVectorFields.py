# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 15:11:30 2020

@author: StoyanBoyukliyski
"""
import VectorField as VF
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Cholesky as Ch
#--- The region is used to calculate the z10 value which would be altered if related to Japan ---
region = "US"

#--- M refers to magnitude for the moment magnitude scale ---
M = 7

#--- This is supposed to locate the regressions coefficients files in their directory ---
data = pd.read_csv("C:\\Users\\StoyanBoyukliyski\\OneDrive\\Desktop\\MScDissertation\PythonFiles\\RegressionCoefficients.csv")

'''
The next three parameters are to do with the geometry of the fault
'''
#--- Beta - dip of the earthquake faulting (converted to radians in next line) ---
beta = 30
beta = beta*np.pi/180

#--- rake - this is the orientation of the fault relative to true north (converted to radians in next line) ---
rake = 90
rake = rake*np.pi/180

#--- lambangle - this is the direction of slip (used in degrees) ---
lambangle = 90


#--- According to PJS calculations these are the transformation matrices used to calculate distances Rjb, Rx, Rrup ---
Tbeta = np.array([[1,0,0],[0, np.cos(beta),np.sin(beta)],[0,-np.sin(beta), np.cos(beta)]])
Tfita = np.array([[np.cos(rake), np.sin(rake),0],[-np.sin(rake),np.cos(rake),0],[0,0,1]])
T = np.matmul(Tbeta,Tfita)

    
#--- Lambda parameter for several key things:
#    Faulting type - depends on the angle and is bounded in specific intervals
#    Faulting indicators - depending on the type binary flags are turned off or on in the equation
#    Faulting regression coefficients - taken from Wells & Coppersmith (1994)
#    eZtor - expected value of the smallest distance from the ground surface to the rupture plane
#    Magnitude bounds - range of applicability of the model changes with respect to the faulting ---
if lambangle >= 30 and lambangle <= 150:
    Fault = "Reverse"
    FNM = 0
    FRV = 1
    
    al = -2.86
    bl = 0.63
    ad = -2.42
    bd = 0.58
    aw = -1.61
    bw = 0.41
    aa = -3.99
    ba = 0.98
    
    eZTOR = np.max(2.704 - 1.226*np.max(M-5.849,0))**2
    if M<3.5 or M>8.5:
        raise(ValueError("The Magnitude is out of the range of applicability"))
    else:
        pass
    
elif lambangle >= 240 and lambangle <= 300:
    Fault = "Normal"
    
    FRV = 1
    FNM = 0
    al = -2.01
    bl = 0.50
    ad = -1.88
    bd = 0.50
    aw = -1.14
    bw = 0.35
    aa = -2.87
    ba = 0.82
    
    eZTOR = np.max(2.673-1.136*np.max(M-4.970,0))**2
    if M<3.5 and M>8.0:
        raise(ValueError("The Magnitude is out of the range of applicability"))
    else:
        pass
    
    
elif (lambangle >= 330 and lambangle <= 30) or (lambangle >=150 and lambangle <= 210):
    Fault = "Strike-Slip"
    
    al = -3.55
    bl = 0.74
    ad = -2.57
    bd = 0.62
    aw = -0.76
    bw = 0.27
    aa = -3.42
    ba = 0.90
    
    FRV = 0
    FNM = 0
    eZTOR = np.max(2.673-1.136*np.max(M-4.970,0))**2
    if M<3.5 and M>8.0:
        raise(ValueError("The Magnitude is out of the range of applicability"))
    else:
        pass
    
elif (lambangle >=210 and lambangle <= 240) or (lambangle >=300 and lambangle <= 330):
    Fault = "Normal-Oblique"
    al = -2.01
    bl = 0.50
    ad = -1.88
    bd = 0.50
    aw = -1.14
    bw = 0.35
    aa = -2.87
    ba = 0.82
    
    FRV = 0
    FNM = 0
    eZTOR = np.max(2.673-1.136*np.max(M-4.970,0))**2
    if M<3.5 and M>8.0:
        raise(ValueError("The Magnitude is out of the range of applicability"))
    else:
        pass
else:
    ValueError("Lambda should be >=0 and <=360")
    
#Ztor is the actual smallest distance from the ground surface to the fault surface
#the dZtor terms is related to a specific dataset for a specific earthquake
dZTOR = 0    
ZTOR = eZTOR + dZTOR


#Vs30 - The average velocity of the shear wave velocity from the top to 30 meters below the ground surface
VS30 = 760

#Directivity parameters left at 0
DPP = 0
dDPP = 0

#Type of measurements produced (I assume its to do with weather there was actual dataset)
Finferred = 0
Fmeasured = 1

#Dummy parameter
ni = 0

#Rabge of applicability of the model, forbids you to put too small or too large values of velocity
if VS30 < 180 or VS30 > 1500:
    raise(ValueError("The Shear Wave Velocity is out of the range of applicability"))
else:
    pass

#Calculating the eZ10 as a function of VS30 and region location
if region == "Japan":
    eZ10 = np.exp((-5.23/2)*np.log((VS30**2+412**2)/(1360**2+412**2)))
else:
    eZ10 = np.exp((-7.15/4)*np.log((VS30**4 + 571**4)/(1360**4+571**4)))

#Z10 is the depth at which the shear wave velocity becomes equal to 1.0km/s
#the dZ10 terms is related to a specific dataset for a specific earthquake
dZ10 = 0
Z10 = dZ10 + eZ10

#Calculating the width, depth, length and area of the earthquake using Wells & Coppersmith (1994)
W = 10**(aw + bw*M)
L = 10**(al + bl*M)
D = 10**(ad + bd*M)
A = 10**(aa + ba*M)

#Width of the surface projection of the fault
Wpr = W*np.cos(beta)

#The relative location of the hypocenter on the fault
xw = 1/2
xl = 1/2

#the select corresponds to the period of vibration in the response spectrum
select = str(0.1)
data = pd.read_csv("C:\\Users\\StoyanBoyukliyski\\OneDrive\\Desktop\\MScDissertation\PythonFiles\\RegressionCoefficients.csv")
data = data.set_index("Period(s)")
slc = data.loc[select]
    
#The distance between points is fixed in between iterations, so that the random fields have similar 
dx = 2
dy = 1
Standard = []
CgMatrix = []

#Position of the site relative to the fault (0,0) being the bottom left corner coinciding with the rupture epicenter
initx = 10
inity = 10
iterations = 1000
time_mcs = []
time_analytic = []
ComputationalStandardVector = []
CentralStandardVector = []
RevisedCentralVector = []

Lx = 100
Ly = 100
n = int(Lx/dx) + 1
m = int(Ly/dy) + 1
Lower1, Cg  = Ch.MatrixBuilder(select, n, m, dx, dy)

Zer, Er = VF.CreateVectorField(Lx, Ly, n, m, initx, inity, dx, dy, Lower1, T, Tfita, beta, W, L, xl, xw, slc, select, Wpr, ZTOR, lambangle,FRV, FNM, dZTOR, M, dDPP, VS30, dZ10, ni, Finferred, Fmeasured)
if n*m < 8:
    VF.CorrelationRepresentation(Lx, Ly, n, m, initx, inity, dx, dy, Lower1, T, Tfita, beta, W, L, xl, xw, slc, select, Wpr, ZTOR, lambangle,FRV, FNM, dZTOR, M, dDPP, VS30, dZ10, ni, Finferred, Fmeasured)
else:
    pass

x = np.random.uniform(initx,initx + Lx, n*m)
y = np.random.uniform(inity,inity + Ly, n*m)

Lower2, Cg = Ch.FastMatrixBuilder(select, x,y)
VF.NewVectorField(x, y, initx, inity, Lower2, T, Tfita, beta, W, L, xl, xw, slc, select, Wpr, ZTOR, lambangle,FRV, FNM, dZTOR, M, dDPP, VS30, dZ10, ni, Finferred, Fmeasured)
if np.size(x) <8:
    VF.CorrelationRepresentationScatter(x, y, initx, inity, Lower2, T, Tfita, beta, W, L, xl, xw, slc, select, Wpr, ZTOR, lambangle,FRV, FNM, dZTOR, M, dDPP, VS30, dZ10, ni, Finferred, Fmeasured)
else:
    pass
