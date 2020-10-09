# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 15:24:46 2020

@author: StoyanBoyukliyski
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def Default(rho12, rho2j, rho1j):
    return (rho2j-rho12*rho1j)/np.sqrt(1-rho12**2)

x = np.linspace(0.6, 1, 10)
y = np.linspace(0.6, 1, 10)

Matrix = []
for j in y:
    different = [Default(0.97, i, j) for i in x]
    plt.plot(x, different, label = "y = " + str(j))
    Matrix.append(different)

plt.legend()
X, Y = np.meshgrid(x,y)
Matrix = np.array(Matrix)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X,Y, Matrix)