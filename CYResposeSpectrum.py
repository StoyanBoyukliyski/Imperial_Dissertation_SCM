# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 16:32:09 2020

@author: StoyanBoyukliyski
"""

import ChiouandYoung2014 as CY
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


data = pd.read_csv("C:\\Users\\StoyanBoyukliyski\\OneDrive\\Desktop\\MScDissertation\PythonFiles\\RegressionCoefficients.csv")
data.head()
data = data.set_index("Period(s)")
        
Intensity = []
StdWth = []
StdBtw = []
usablespec = data.index[2:]

for j in usablespec:
    slc = data.loc[j]
    Intensity.append(float(CY.LognormalFunct(slc, 0,0)[0]))
    StdWth.append(float(CY.LognormalFunct(slc, 0,0)[3]))
    StdBtw.append(float(CY.LognormalFunct(slc, 0,0)[4]))

figure = plt.figure(1)
ax2 = figure.add_subplot(111)
C = np.zeros((len(usablespec),len(usablespec)))

def I(Tmin):
    if Tmin < 0.189:
        I = 1
    else:
        I = 0
    return I

parb = 100
for j in range(len(usablespec)):
    for k in range(j, len(usablespec)):
        Tmax = max(float(usablespec[k]), float(usablespec[j]))
        Tmin = min(float(usablespec[k]), float(usablespec[j]))
        h = Tmax-Tmin
        rho = np.exp(-3*h/parb)
#        rho = 1 - np.cos(np.pi/2 - (0.359 + 0.163*I(Tmin)*np.log(Tmin/0.189))*np.log(Tmax/Tmin))
        
        C[j,k] = C[j,k] + rho

C = C + np.transpose(C) - np.diag(np.diag(C))

ax2.plot([float(u) for u in usablespec], Intensity, "k-")
for j in range(5):
    ResidualBtw = np.random.normal(0,1,np.size(StdBtw))
    ResidualWth = np.random.normal(0,1,np.size(StdWth))
    L = np.linalg.cholesky(C)
    ResidualWth = np.matmul(L,ResidualWth)
    ax2.plot([float(u) for u in usablespec], Intensity*np.exp(ResidualBtw*StdBtw + ResidualWth*StdWth), "b-", linewidth = 0.6)
    
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_title("Response Spectrum using C&Y(2014)")
ax2.set_xlabel("Period T (sec)")
ax2.set_ylabel("Acceleration (g)")
ax2.grid(True, which = "both", axis = "both", linestyle = "--")
plt.show()

IntensityDict = pd.DataFrame(usablespec, Intensity)
ax2.text(1, max(Intensity)-0.5, "Max Intenisty = " + str("{:.2f}".format(max(Intensity))) + "g" + "\n" + "at T =" + IntensityDict.loc[max(Intensity)][0] + " sec")