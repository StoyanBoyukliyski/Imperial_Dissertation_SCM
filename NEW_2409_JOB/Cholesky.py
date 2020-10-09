# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 18:15:25 2020

@author: StoyanBoyukliyski
"""

import JayaramBaker2009 as JB
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
def MatrixBuilder(select, n, m, dx, dy):
    time1 = time.time()
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
    time2 = time.time()
    print("Time to construct matrix for normal grid: ", time2-time1)
    L = np.linalg.cholesky(Cg)
    time3 = time.time()
    print("Time to perform Cholesky for normal grid: ", time3-time2)
    print("Total time for normal grid: ", time3-time1)
    return L, Cg

def FastMatrixBuilder(select, x,y):
    time1 = time.time()
    X = np.tile(x,(np.size(x),1))
    Y = np.tile(y,(np.size(y),1))
    D = np.sqrt((X-np.transpose(X))**2 + (Y - np.transpose(Y))**2)
    Cg = JB.Rho(D,select)
    time2 = time.time()
#    print("Time to construct matrix for randomly generated points: ", time2-time1)
    L = np.linalg.cholesky(Cg)
    time3 = time.time()
#    print("Time to perform Cholesky for randomly generated points: ", time3-time2)
#    print("Total time for randomly generated points: ", time3-time1)
    time_for_assembly = time2-time1
    time_for_cholesky = time3-time2
    return L, Cg
'''
times_ch = []
times_as = [] 
for f in np.linspace(100,10000, 20):
    x = np.random.uniform(0,1,int(f))
    y = np.random.uniform(0,1,int(f))
    times_ch.append(FastMatrixBuilder(0.1, x,y)[2])
    times_as.append(FastMatrixBuilder(0.1, x,y)[3])

fig, (ax1,ax2) = plt.subplots(1,2)
ax1.plot(np.linspace(100,10000, 20), times_as)
ax1.legend(loc = "upper right", prop = {"size": 7})
ax1.grid(linestyle = "--")
ax1.set_title("Time for Assembly of the Correlation Matrix", {'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'})
ax1.set_ylabel("Time required to assemble",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad=20)
ax1.set_xlabel("Number of points chosen",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)
ax2.plot(np.linspace(100,10000, 20), times_ch)
ax2.legend(loc = "upper right", prop = {"size": 7})
ax2.grid(linestyle = "--")
ax2.set_title("Time to perform Cholesky", {'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'})
ax2.set_ylabel("Time required to decompose",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad=20)
ax2.set_xlabel("Number of points chosen",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)
'''