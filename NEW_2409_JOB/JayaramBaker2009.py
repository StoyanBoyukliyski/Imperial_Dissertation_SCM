# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 18:11:49 2020

@author: StoyanBoyukliyski
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
    
def Rho(h,T):
    if float(T) < 1:
        parb = 8.5 + 17.2*float(T)
    else:
        parb = 22 + 3.7*float(T)
        
    return np.exp(-3*h/parb)