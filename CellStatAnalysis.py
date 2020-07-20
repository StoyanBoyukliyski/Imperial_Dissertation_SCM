# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 14:05:25 2020

@author: StoyanBoyukliyski
"""

import RandomFields as RF
import math as m
import numpy as np
import matplotlib.pyplot as plt
import ChiouandYoung2014 as CY
import pandas as pd

discx = 10
discy = 10
dx = RF.dx
dy = RF.dy
Lx = RF.Lx
Ly = RF.Ly
lensegx = Lx/discx
lensegy = Ly/discy
matrixofmatrix = []


data = pd.read_csv("C:\\Users\\StoyanBoyukliyski\\OneDrive\\Desktop\\MScDissertation\PythonFiles\\RegressionCoefficients.csv")
data.head()
data = data.set_index("Period(s)")

select = str(0.1)
slc = data.loc[select]

IM = RF.Zerwth
MeansIM = []
StdSIM = []
StandardDevC = []
MeanC = []
print(IM)
for k in range(discy):
    for j in range(discx):
        dist = (j+1/2)*(Lx/discx)
        meanval = CY.CalculateY(slc,dist)
        StandardDevC.append(CY.CalculateStd())
        MeanC.append(meanval)
        
        IMval = IM[m.ceil(k*(Ly/discy)/dy):m.floor((k+1)*(Ly/discy)/dy)+1,m.ceil(j*(Lx/discx)/dx):m.floor((j+1)*(Lx/discx)/dx)+1]
        matrixofmatrix.append(np.array(IMval))
        IMval = IMval.reshape(np.size(IMval))
        MeanIM = np.mean(IMval)
        StdIM = np.std(IMval)
        MeansIM.append(MeanIM)
        StdSIM.append(StdIM)
        
        
        
        plt.figure()
        plt.hist(IMval, bins = 10)
        plt.text(0, 1, "Mean of population = " + str(MeanIM) + "m/s^2" + "\n" + "Standard Deviation = " + str(StdIM))
        

        
