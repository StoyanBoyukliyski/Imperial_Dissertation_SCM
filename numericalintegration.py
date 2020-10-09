# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 11:18:05 2020

@author: StoyanBoyukliyski
"""

import scipy.integrate as integrate
import scipy.special as special
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def NumericalIntegration(xb0, xb1, yb0, yb1, T):
    if float(T) < 1:
        parb = 8.5 + 17.2*float(T)
    else:
        parb = 22 + 3.7*float(T)
    def rho(x1,x2,y1,y2, parb):
        return np.exp(-3*np.sqrt((x2-x1)**2+(y2-y1)**2)/parb)/((xb1-xb0)**2*(yb1-yb0)**2)
    
    integral = integrate.nquad(rho, [[xb0,xb1], [xb0,xb1], [yb0,yb1], [yb0,yb1]], args = [parb])
    return integral[0]
