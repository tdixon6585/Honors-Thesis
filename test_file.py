#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 19:24:18 2021

@author: thomasdixon
"""

import numpy as np
import RK4_numba as RK4
import matplotlib.pyplot as plt
#%%
r = RK4.Rocket()

angle = 0
throttle = 0
staged = 0
r.input_vars = np.array([angle, throttle, staged], dtype=np.float32)
r.ry = 6578000
r.vx = 6000#7784
dt = 0.1
#%%
RX = []
RY = []
for i in range(1000):
    pt_vars = np.array([r.rx, r.ry, r.vx, r.vy, r.fuelm], dtype=np.float32)
    r.RK4_step(pt_vars, dt, r.input_vars)
    RX.append(r.rx)
    RY.append(r.ry)
#%%

min_per = 6478000 #karman line
    
v = np.sqrt(r.vx**2 + r.vy**2)
rad = np.sqrt(r.rx**2 + r.ry**2)

phi = np.abs(np.arctan(r.rx/(r.ry+1e-9))) - np.abs(np.arctan(r.vy/(r.vx+1e-9)))
h = rad*v*np.cos(phi)

mu = r.G * r.M
semimajor = ((2/rad)-((v**2)/mu))**-1
ecc = np.sqrt(1-(h**2/(semimajor*mu)))

calc_per = semimajor*(1-ecc)
calc_ap = semimajor*(1+ecc)
esc_vel = np.sqrt(2*r.G*r.M/rad)

#%%
rad = 6378000
surface = plt.Circle((0, 0), rad, color='b', fill=False)
karman_line = plt.Circle((0, 0), rad+100000, color='b', fill=False)
ax = plt.gca()

plt.plot(RX, RY)

ax.add_patch(surface)
ax.add_patch(karman_line)
plt.axes().set_aspect('equal','datalim')
plt.show()





