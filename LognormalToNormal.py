# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 11:13:47 2020

@author: StoyanBoyukliyski
"""
import numpy as np
import matplotlib.pyplot as plt

def lognormaldist(mu, sigma, im):
    return (1/(im*sigma*np.sqrt(2*np.pi)))*np.exp(-(np.log(im)-mu)**2/(2*sigma**2))
    
def normal(mu, sigma, im):
    return (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(im-mu/sigma)**2/2)

fig = plt.figure()
ax = fig.add_subplot(121)
ax1 = fig.add_subplot(122)

mux = 0.7
sigmax = 0.5

muy = np.log(mux**2/np.sqrt(mux**2 + sigmax**2))
sigmay = np.log(1 + sigmax**2/mux**2)
print(muy)
print(sigmay)
z = np.linspace(muy - sigmay*10,muy + sigmay*10, 1000)

ax.plot(z, [lognormaldist(muy, sigmay, im) for im in z])
ax1.plot(z, [normal(muy, sigmay, im) for im in z])
