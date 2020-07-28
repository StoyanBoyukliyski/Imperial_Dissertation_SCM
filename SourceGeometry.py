# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 13:29:33 2020

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
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

region = "US"
M = 7.5
data = pd.read_csv("C:\\Users\\StoyanBoyukliyski\\OneDrive\\Desktop\\MScDissertation\PythonFiles\\RegressionCoefficients.csv")


beta = 90
beta = beta*np.pi/180
rake = 120
rake = rake*np.pi/180
lambangle = 180

Tbeta = np.array([[1,0,0],[0, np.cos(beta),np.sin(beta)],[0,-np.sin(beta), np.cos(beta)]])
Tfita = np.array([[np.cos(rake), np.sin(rake),0],[-np.sin(rake),np.cos(rake),0],[0,0,1]])
T = np.matmul(Tbeta,Tfita)

    

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
    

dZTOR = 0    
ZTOR = eZTOR + dZTOR

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

W = 10**(aw + bw*M)
L = 10**(al + bl*M)
D = 10**(ad + bd*M)
A = 10**(aa + ba*M)

xw = 1/2
xl = 1/2

Wpr = W*np.cos(beta)