# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 11:46:08 2020

@author: StoyanBoyukliyski
"""

import matplotlib.pyplot as plt
import numpy as np

def annotate_dim(ax,xyfrom,xyto,text=None):

    if text is None:
        text = str(np.sqrt( (xyfrom[0]-xyto[0])**2 + (xyfrom[1]-xyto[1])**2 ))

    ax.annotate("",xyfrom,xyto,arrowprops=dict(arrowstyle='<->'))
    ax.text((xyto[0]+xyfrom[0])/2,(xyto[1]+xyfrom[1])/2,text,fontsize=16)

x = np.linspace(0,2*np.pi,100)
plt.plot(x,np.sin(x))
annotate_dim(plt.gca(),[0,0],[np.pi,0],'$\pi$')

plt.show()