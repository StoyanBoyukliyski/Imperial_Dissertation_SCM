# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 12:56:47 2020

@author: StoyanBoyukliyski
"""
import numpy as np
import matplotlib.pyplot as plt


select = 0.1

if float(select) < 1:
    parb = 8.5 + 17.2*float(select)
else:
    parb = 22 + 3.7*float(select)


def Rho(h):
    rho = np.exp(-3*h/parb)
    return rho
        
        
dist  = 10

x1 = 0
x2 = 0.3
x3 = 0
x4 = 0.3


y1 = 0
y2 = 0.3
y3 = 0
y4 = 0.3

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.set_title("Visual Presentation of the integral", {'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'})
ax1.set_ylabel("North position relative to bottom-left edge (km)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)
ax1.set_xlabel("East position relative to bottom-left edge (km)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)

ax2.grid(linestyle = "--")
ax2.set_title("Convergence study", {'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'})
ax2.set_ylabel("Effective correlation (/)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)
ax2.set_xlabel("Number of points in Simulation (/)",{'fontsize': 12, 'fontweight' : 12, 'verticalalignment': 'baseline'},labelpad = 10)
ax1.plot((x2+x1)/2,(y2+y1)/2, marker = "o", color = "black")
ax1.plot((x4+x3)/2,(y4+y3)/2, marker = "o", color = "black")


ax1.plot([x1,x1],[y1,y2],"black")
ax1.plot([x1,x2],[y1,y1],"black")
ax1.plot([x2,x2],[y1,y2],"black")
ax1.plot([x1,x2],[y2,y2],"black")

ax1.plot([x3,x3],[y3,y4],"black")
ax1.plot([x3,x4],[y3,y3],"black")
ax1.plot([x4,x4],[y3,y4],"black")
ax1.plot([x3,x4],[y4,y4],"black")

rhos = 0
n = 0
x1avg = 0
y1avg = 0
x2avg = 0
y2avg = 0
XAVG1 = 0
YAVG1 = 0
XAVG2 = 0
YAVG2 = 0
for j in range(1000):
    x = np.random.uniform(x1,x2,1)
    y = np.random.uniform(y1,y2,1)
    r = np.random.uniform(x3,x4,1)
    z = np.random.uniform(y3,y4,1)
    ax1.plot(x,y,"r.")
    ax1.plot(r,z,"k.")
    dist = np.sqrt((r-x)**2+(z-y)**2)
    Expected = Rho(np.sqrt(((x4+x3-x2-x1)/2)**2+((y4+y3-y2-y1)/2)**2))
    A = ax1.plot(XAVG1,YAVG1, marker = "o", color = "white")
    B = ax1.plot(XAVG2,YAVG2, marker = "o", color = "white")
    rhos = rhos + Rho(dist)
    n = n + 1
    ax2.plot(n,Expected,"k.")
    x1avg = (x1avg + x)
    y1avg = (y1avg + y)
    x2avg = (x2avg + r)
    y2avg = (y2avg + z)
    XAVG1 = x1avg/n
    YAVG1 = y1avg/n
    XAVG2 = x2avg/n
    YAVG2 = y2avg/n
    A = ax1.plot(XAVG1,YAVG1, marker = "o", color = "b")
    B = ax1.plot(XAVG2,YAVG2, marker = "o", color = "b")
    Rhoavg = rhos/n
    ax2.plot(n,Rhoavg, "b.")
    plt.pause(0.005)
    
A = ax1.plot(XAVG1,YAVG1, marker = "o", color = "b")
B = ax1.plot(XAVG2,YAVG2, marker = "o", color = "b")
Rhoavg = rhos/n
print("The effective correlation is :", str(Rho))