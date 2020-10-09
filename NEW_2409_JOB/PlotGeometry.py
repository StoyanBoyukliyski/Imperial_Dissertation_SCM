# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 14:39:34 2020

@author: StoyanBoyukliyski
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import lognorm
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
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

def PlotGeometryF(L,W,ZTOR, beta, lambangle, T, Tfita, xw, xl, initx, inity, Lx, Ly, n, m, Wpr):
    x1 = [(1-xl)*L,-xw*W,0]
    x2 = [(1-xl)*L,(1-xw)*W,0]
    x3 = [-xl*L,-xw*W,0]
    x4 = [-xl*L,(1-xw)*W,0]
    
    origin = [0,0,-ZTOR - xw*W*np.sin(beta)]
    truenorth = [W/2,0, 0]
    
    r1 = L/2.5
    r2 = W/2.5
    
    indicator = [np.cos(lambangle*np.pi/180)*W/2, np.sin(lambangle*np.pi/180)*W/2, 0]
    indicatoro= [np.cos(30*np.pi/180)*r1, np.sin(30*np.pi/180)*r2, 0]
    indicator1= [np.cos(150*np.pi/180)*r1, np.sin(150*np.pi/180)*r2, 0]
    indicator2= [np.cos(240*np.pi/180)*r1, np.sin(240*np.pi/180)*r2, 0]
    indicator3= [np.cos(300*np.pi/180)*r1, np.sin(300*np.pi/180)*r2, 0]
    indicator4= [np.cos(330*np.pi/180)*r1, np.sin(330*np.pi/180)*r2, 0]
    indicator5= [np.cos(210*np.pi/180)*r1, np.sin(210*np.pi/180)*r2, 0]
    indicator6= [np.cos(240*np.pi/180)*r1, np.sin(240*np.pi/180)*r2, 0]
    indicatorz= [np.cos(0*np.pi/180)*r1, np.sin(0*np.pi/180)*r2, 0]
        
    def circle(fita):
        dist = [np.cos(fita*np.pi/180)*r1*0.9, np.sin(fita*np.pi/180)*r2*0.99, 0]
        dist = np.matmul(np.linalg.inv(T), dist) + origin
        crcx = dist[0]
        crcy = dist[1]
        crcz = dist[2]
        return crcx,crcy,crcz
    
    xc = [circle(x)[0] for x in np.linspace(0,360,1000)]
    yc = [circle(x)[1] for x in np.linspace(0,360,1000)]
    zc = [circle(x)[2] for x in np.linspace(0,360,1000)]
    
    x13d = np.matmul(np.linalg.inv(T), x1) + origin
    x23d = np.matmul(np.linalg.inv(T), x2) + origin
    x33d = np.matmul(np.linalg.inv(T), x3) + origin
    x43d = np.matmul(np.linalg.inv(T), x4) + origin
    indicator = np.matmul(np.linalg.inv(T), indicator) + origin
    indicatoro = np.matmul(np.linalg.inv(T), indicatoro) + origin
    indicator1 = np.matmul(np.linalg.inv(T), indicator1) + origin
    indicator2 = np.matmul(np.linalg.inv(T), indicator2) + origin
    indicator3 = np.matmul(np.linalg.inv(T), indicator3) + origin
    indicator4 = np.matmul(np.linalg.inv(T), indicator4) + origin
    indicator5 = np.matmul(np.linalg.inv(T), indicator5) + origin
    indicator6 = np.matmul(np.linalg.inv(T), indicator6) + origin
    indicatorz = np.matmul(np.linalg.inv(T), indicatorz) + origin
    
    
    x1t = initx
    x2t = initx + Lx
    y1t = inity
    y2t = inity + Ly
    
    countx = n
    county = m
    
    x = np.linspace(x1t,x2t, countx)
    y = np.linspace(y1t,y2t, county)
    
    X,Y = np.meshgrid(x,y)
    x = np.array([[x13d[0], x23d[0]], [x33d[0], x43d[0]]])
    y = np.array([[x13d[1], x23d[1]], [x33d[1], x43d[1]]])
    z = np.array([[x13d[2], x23d[2]], [x33d[2], x43d[2]]])
    bx = np.array([[-(x2t+ 20),-(x2t+ 20)], [x2t+ 20,x2t+ 20]])
    by = np.array([[-(x2t+ 20),x2t+ 20], [-(x2t+ 20),x2t+ 20]])
    zs = np.array([[0,0], [0,0]])
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    
    a = Arrow3D([origin[0], indicator[0]], [origin[1], indicator[1]], [origin[2], indicator[2]], mutation_scale=7, lw=3, arrowstyle="-|>", color="black")
    b = Arrow3D([origin[0], indicatoro[0]], [origin[1], indicatoro[1]], [origin[2], indicatoro[2]], mutation_scale=0.2, lw=2, arrowstyle="-|>", color="r")
    b1 = Arrow3D([origin[0], indicator1[0]], [origin[1], indicator1[1]], [origin[2], indicator1[2]], mutation_scale=0.2, lw=2, arrowstyle="-|>", color="r")
    b2 = Arrow3D([origin[0], indicator2[0]], [origin[1], indicator2[1]], [origin[2], indicator2[2]], mutation_scale=0.2, lw=2, arrowstyle="-|>", color="r")
    b3 = Arrow3D([origin[0], indicator3[0]], [origin[1], indicator3[1]], [origin[2], indicator3[2]], mutation_scale=0.2, lw=2, arrowstyle="-|>", color="r")
    b4 = Arrow3D([origin[0], indicator4[0]], [origin[1], indicator4[1]], [origin[2], indicator4[2]], mutation_scale=0.2, lw=2, arrowstyle="-|>", color="r")
    b5 = Arrow3D([origin[0], indicator5[0]], [origin[1], indicator5[1]], [origin[2], indicator5[2]], mutation_scale=0.2, lw=2, arrowstyle="-|>", color="r")
    b6 = Arrow3D([origin[0], indicator6[0]], [origin[1], indicator6[1]], [origin[2], indicator6[2]], mutation_scale=0.2, lw=2, arrowstyle="-|>", color="r")
    c = Arrow3D([origin[0], indicatorz[0]], [origin[1], indicatorz[1]], [0, 0], mutation_scale=7, lw=3, arrowstyle="-|>", color="black")
    d = Arrow3D([origin[0], truenorth[0]], [origin[1], truenorth[1]], [0, truenorth[2]], mutation_scale=7, lw=2, arrowstyle="-|>", color="red")
    
    ax.plot_surface(x,y,z, color = "white", alpha = 0.6)
#    ax.plot_surface(x,y,zs, color = "white",alpha = 0.8)
    ax.plot_surface(bx,by,zs, color = "green",alpha = 0.2)
    ax.scatter(xc,yc,zc, color = "red", marker = ".",linewidths = 0.3)
    ax.scatter(origin[0],origin[1],origin[2], cmap = "Blacks", marker = "o", linewidths =1)
    ax.scatter(origin[0],origin[1],0, cmap = "Blacks", marker = "o", linewidths =1)
    ax.add_artist(a)
    ax.add_artist(b)
    ax.add_artist(b1)
    ax.add_artist(b2)
    ax.add_artist(b3)
    ax.add_artist(b4)
    ax.add_artist(b5)
    ax.add_artist(b6)
    ax.add_artist(c)
    ax.add_artist(d)
    
    ax.set_xlim([(x1t- 20),(x2t+ 20)])
    ax.set_ylim([(y1t- 20),(y2t+ 20)])
    return ax


def NewPlotGeometry(L,W,ZTOR, beta, lambangle, T, Tfita, xw, xl, initx, inity, x, y, Wpr):
    x1 = [(1-xl)*L,-xw*W,0]
    x2 = [(1-xl)*L,(1-xw)*W,0]
    x3 = [-xl*L,-xw*W,0]
    x4 = [-xl*L,(1-xw)*W,0]
    
    origin = [0,0,-ZTOR - xw*W*np.sin(beta)]
    truenorth = [W/2,0, 0]
    
    r1 = L/2.5
    r2 = W/2.5
    
    indicator = [np.cos(lambangle*np.pi/180)*W/2, np.sin(lambangle*np.pi/180)*W/2, 0]
    indicatoro= [np.cos(30*np.pi/180)*r1, np.sin(30*np.pi/180)*r2, 0]
    indicator1= [np.cos(150*np.pi/180)*r1, np.sin(150*np.pi/180)*r2, 0]
    indicator2= [np.cos(240*np.pi/180)*r1, np.sin(240*np.pi/180)*r2, 0]
    indicator3= [np.cos(300*np.pi/180)*r1, np.sin(300*np.pi/180)*r2, 0]
    indicator4= [np.cos(330*np.pi/180)*r1, np.sin(330*np.pi/180)*r2, 0]
    indicator5= [np.cos(210*np.pi/180)*r1, np.sin(210*np.pi/180)*r2, 0]
    indicator6= [np.cos(240*np.pi/180)*r1, np.sin(240*np.pi/180)*r2, 0]
    indicatorz= [np.cos(0*np.pi/180)*r1, np.sin(0*np.pi/180)*r2, 0]
        
    def circle(fita):
        dist = [np.cos(fita*np.pi/180)*r1*0.9, np.sin(fita*np.pi/180)*r2*0.99, 0]
        dist = np.matmul(np.linalg.inv(T), dist) + origin
        crcx = dist[0]
        crcy = dist[1]
        crcz = dist[2]
        return crcx,crcy,crcz
    
    xc = [circle(x)[0] for x in np.linspace(0,360,1000)]
    yc = [circle(x)[1] for x in np.linspace(0,360,1000)]
    zc = [circle(x)[2] for x in np.linspace(0,360,1000)]
    
    x13d = np.matmul(np.linalg.inv(T), x1) + origin
    x23d = np.matmul(np.linalg.inv(T), x2) + origin
    x33d = np.matmul(np.linalg.inv(T), x3) + origin
    x43d = np.matmul(np.linalg.inv(T), x4) + origin
    indicator = np.matmul(np.linalg.inv(T), indicator) + origin
    indicatoro = np.matmul(np.linalg.inv(T), indicatoro) + origin
    indicator1 = np.matmul(np.linalg.inv(T), indicator1) + origin
    indicator2 = np.matmul(np.linalg.inv(T), indicator2) + origin
    indicator3 = np.matmul(np.linalg.inv(T), indicator3) + origin
    indicator4 = np.matmul(np.linalg.inv(T), indicator4) + origin
    indicator5 = np.matmul(np.linalg.inv(T), indicator5) + origin
    indicator6 = np.matmul(np.linalg.inv(T), indicator6) + origin
    indicatorz = np.matmul(np.linalg.inv(T), indicatorz) + origin
    
    x = np.array([[x13d[0], x23d[0]], [x33d[0], x43d[0]]])
    y = np.array([[x13d[1], x23d[1]], [x33d[1], x43d[1]]])
    z = np.array([[x13d[2], x23d[2]], [x33d[2], x43d[2]]])
    bx = np.array([[-(np.max(x)+ 20),-(np.max(x)+ 20)], [np.max(x)+ 20,np.max(x)+ 20]])
    by = np.array([[-(np.max(y)+ 20),np.max(y)+ 20], [-(np.max(y)+ 20),np.max(y)+ 20]])
    zs = np.array([[0,0], [0,0]])
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    
    a = Arrow3D([origin[0], indicator[0]], [origin[1], indicator[1]], [origin[2], indicator[2]], mutation_scale=7, lw=3, arrowstyle="-|>", color="black")
    b = Arrow3D([origin[0], indicatoro[0]], [origin[1], indicatoro[1]], [origin[2], indicatoro[2]], mutation_scale=0.2, lw=2, arrowstyle="-|>", color="r")
    b1 = Arrow3D([origin[0], indicator1[0]], [origin[1], indicator1[1]], [origin[2], indicator1[2]], mutation_scale=0.2, lw=2, arrowstyle="-|>", color="r")
    b2 = Arrow3D([origin[0], indicator2[0]], [origin[1], indicator2[1]], [origin[2], indicator2[2]], mutation_scale=0.2, lw=2, arrowstyle="-|>", color="r")
    b3 = Arrow3D([origin[0], indicator3[0]], [origin[1], indicator3[1]], [origin[2], indicator3[2]], mutation_scale=0.2, lw=2, arrowstyle="-|>", color="r")
    b4 = Arrow3D([origin[0], indicator4[0]], [origin[1], indicator4[1]], [origin[2], indicator4[2]], mutation_scale=0.2, lw=2, arrowstyle="-|>", color="r")
    b5 = Arrow3D([origin[0], indicator5[0]], [origin[1], indicator5[1]], [origin[2], indicator5[2]], mutation_scale=0.2, lw=2, arrowstyle="-|>", color="r")
    b6 = Arrow3D([origin[0], indicator6[0]], [origin[1], indicator6[1]], [origin[2], indicator6[2]], mutation_scale=0.2, lw=2, arrowstyle="-|>", color="r")
    c = Arrow3D([origin[0], indicatorz[0]], [origin[1], indicatorz[1]], [0, 0], mutation_scale=7, lw=3, arrowstyle="-|>", color="black")
    d = Arrow3D([origin[0], truenorth[0]], [origin[1], truenorth[1]], [0, truenorth[2]], mutation_scale=7, lw=2, arrowstyle="-|>", color="red")
    
    ax.plot_surface(x,y,z, color = "white", alpha = 0.6)
    ax.plot_surface(bx,by,zs, color = "green",alpha = 0.2)
    ax.scatter(xc,yc,zc, color = "red", marker = ".",linewidths = 0.3)
    ax.scatter(origin[0],origin[1],origin[2], cmap = "Blacks", marker = "o", linewidths =1)
    ax.scatter(origin[0],origin[1],0, cmap = "Blacks", marker = "o", linewidths =1)
    ax.add_artist(a)
    ax.add_artist(b)
    ax.add_artist(b1)
    ax.add_artist(b2)
    ax.add_artist(b3)
    ax.add_artist(b4)
    ax.add_artist(b5)
    ax.add_artist(b6)
    ax.add_artist(c)
    ax.add_artist(d)
    
    return ax
