# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 18:15:25 2020

@author: StoyanBoyukliyski
"""

import JayaramBaker2009 as JB
import numpy as np
import pandas as pd

Lx = 150
Ly = 150
n= 100
m = 100
initx = -75
inity = -75
dx = Lx/(n-1)
dy = Ly/(m-1)

data = pd.read_csv("C:\\Users\\StoyanBoyukliyski\\OneDrive\\Desktop\\MScDissertation\PythonFiles\\RegressionCoefficients.csv")
data = data.set_index("Period(s)")
select = str(0.1)
slc = data.loc[select]

def MatrixBuilder(select, m, n, dx, dy):
    Cg = np.zeros((n*m,n*m))
    C= np.zeros((n,n))
    matrices = []
    for z in range(m):
            for i in range(n):
                for j in range(i,n):
                    C[i,j] = JB.Rho(np.sqrt(((z)*dy)**2 + (dx*(j-i))**2),select)
            matrices.append(np.array(C))
    
    for k in range(m):
        Cg[(k)*n:(k+1)*n, (k)*n:(k+1)*n] = matrices[0]
        for z in range(k+1,m):
            C =  matrices[z-k]
            Cr = C + np.transpose(C) - np.diag(np.diag(C))
            Cg[(k)*n:(k+1)*n, (z)*n:(z+1)*n] = Cr
            
    del matrices
    Cg = Cg + np.transpose(Cg) - np.identity(n*m)
    return Cg