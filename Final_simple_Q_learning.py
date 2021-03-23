#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 18:53:05 2021

@author: thomasdixon
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
import RK4_numba as RK4

r = RK4.Rocket()

dAngle = 0.1*np.pi
dThrottle = 0.25
dt = 0.5


def step(r, action):
    pt_vars = np.array([r.rx, r.ry, r.vx, r.vy, r.fuelm], dtype=np.float32)
    
    angle, throttle, staged = r.input_vars
    
    #React to the action
    if action == 0:
        angle += dAngle
    elif action == 1:
        angle -= dAngle
    elif action == 2:
        throttle += dThrottle
    elif action == 3:
        throttle -= dThrottle
    elif action == 4:
        angle = angle
        throttle = throttle
    else:
        print("NO ACTION SELECTED")
        
        
    #Maintain throttle and angle numbers
    if throttle > 1:
        throttle = 1
    elif throttle < 0:
        throttle = 0
        
    if angle > 2*np.pi:
        angle = angle - 2*np.pi
    elif angle < 0:
        angle = angle + 2*np.pi
    
        
        
        
    #Find periapsis
    min_per = 6478000 #karman line
    
    v = np.sqrt(r.vx**2 + r.vy**2)
    rad = np.sqrt(r.rx**2 + r.ry**2)
    
    phi = np.abs(np.arctan(r.rx/(r.ry+1e-9)) - np.arctan(r.vy/(r.vx+1e-9)))
    h = rad*v*np.cos(phi)
    
    mu = r.G * r.M
    semimajor = ((2/rad)-((v**2)/mu))**-1
    ecc = np.sqrt(1-(h**2/(semimajor*mu)))
    
    calc_per = semimajor*(1-ecc)
    calc_ap = semimajor*(1+ecc)
    esc_vel = np.sqrt(2*r.G*r.M/rad)

    status = ''
    
    #ACHIEVED ORBIT
    if (calc_per > min_per) and (v < esc_vel):
        print('ORBIT-------------------------------------')
        done = True
        status = 'in_orbit'
    #OUT OF FUEL
    elif r.fuelm <= 0:
        print("Out Of Fuel")
        done = True
        status = 'Out Of Fuel'
    #CRASHES
    elif np.sqrt(r.rx**2+r.ry**2) < r.sea:
        print("Crashed")
        done = True
        status = 'Crashed'
    #TIME CUTOFF
    elif r.t > 5000:
        print("Time Limit")
        status = 'Time Limit'
        done = True
    else:
        done = False
        
    r.input_vars = np.array([angle, throttle, staged], dtype=np.float32)
    pt_vars = r.RK4_step(pt_vars, dt, r.input_vars)
    r.rx, r.ry, r.vx, r.vy, r.fuelm = pt_vars
    
    return r, done, status, calc_per, calc_ap


def reward_function(state, status, calc_per, calc_ap):
    
    rx, ry, vx, vy, m, _, angle, throttle = state[0]
    
    #diff = np.abs(calc_per - calc_ap)
    
    #rew = 20*np.exp(-0.000001*(diff-r.sea))
    
    v = np.sqrt(r.vx**2 + r.vy**2)
    rad = np.sqrt(r.rx**2 + r.ry**2)
    
    phi = np.abs(np.arctan(r.rx/(r.ry+1e-9)) - np.arctan(r.vy/(r.vx+1e-9)))
    h = rad*v*np.cos(phi)
    
    mu = r.G * r.M
    semimajor = ((2/rad)-((v**2)/mu))**-1
    ecc = np.sqrt(1-(h**2/(semimajor*mu)))
    
    calc_new_per = semimajor*(1-ecc)
    #calc_new_ap = semimajor*(1+ecc)
    
    #new_diff = np.abs(calc_new_per - calc_new_ap)
    
    if calc_per < calc_new_per:
        rew = 1
    elif calc_per > calc_new_per:
        rew = -1
    else:
        rew = 0
    
    return rew

'''
def reward_function(state, status):
    '''''''
    if status == 'in_orbit':
        return 100000
    elif status == 'Out Of Fuel':
        return -10
    elif status == 'Crashed':
        return -100
    elif status == 'Time Limit':
        return 0
    '''''''
    
    rx, ry, vx, vy, m, _, angle, throttle = state[0]
    
    rad = np.sqrt(rx**2+ry**2)
    
    v_perfect = 7784.26
    
    vy_p = -v_perfect * rx/rad
    vx_p = v_perfect * ry/rad
    
    rw_vy = (np.exp(-1*(vy-vy_p)**2/((vy_p/2 + 2)**2)) + np.exp(-1*(vy-vy_p)**2/((vy_p/20 + 2)**2)) )/2
        
    rw_vx = (np.exp(-1*(vx-vx_p)**2/((vx_p/2 + 2)**2)) + np.exp(-1*(vx-vx_p)**2/((vx_p/20 + 2)**2)) )/2
    
    rw_x = np.exp(-1*(rad-r.sea-200000)**2/(2*40000**2))
    punish = 0.26*np.log(rad-r.sea+1100)-2.8
    
    rew = rw_x + punish + 1 + rw_vy + rw_vx
    return rew
'''



def scale(s):
    #a = []
    #a.append(s[:,0][0]/6478000)
    #a.append(s[:,1][0]/6478000)
    #a.append(s[:,2][0]/60000)
    #a.append(s[:,3][0]/60000)
    #a.append(s[:,4][0]/(2*np.pi))
    #a = np.array([a])
    return s#a



inputs = keras.Input(shape=(8))
x = keras.layers.Dense(64, activation="relu")(inputs)
x = keras.layers.Dense(32, activation="relu")(inputs)
outputs = keras.layers.Dense(5, activation="linear")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-7),
              loss="mean_squared_error")

model.set_weights(np.array(model.get_weights())/10000)

def test():
    y = RK4.Rocket()
    y_state = np.array([[y.rx, y.ry, y.vx, y.vy, y.m, int(y.has_staged), y.input_vars[0], y.input_vars[1]]])
    print(model.predict(scale(y_state)))


epochs = 1
greed = 1
greed_decay = 0.00001
discount_factor = 0.999

all_RX = []
all_RY = []
all_rewards =[]
all_loss = []

test_RX = []
test_RY = []
test_rewards =[]

w = 0
i = 0
#for i in range(epochs):
test()
while greed >= 0.01 or w < 20:
    if greed <= 0.01:
        w += 1
    i+=1

    r = RK4.Rocket()
    state = np.array([[r.rx, r.ry, r.vx, r.vy, r.m, int(r.has_staged), r.input_vars[0], r.input_vars[1]]])
    
    RX = []
    RY = []
    rewards = []

    done = False

    while not done:   
        if greed > 0.01:
            greed -= greed_decay
        if np.random.random() < greed:
            action = np.random.randint(0, 5)
        else:
            action = np.argmax(model.predict(scale(state)))
            
        RX.append(r.rx)
        RY.append(r.ry)
            
        r, done, status, calc_per, calc_ap = step(r, action)

        new_state = np.array([[r.rx, r.ry, r.vx, r.vy, r.m, int(r.has_staged), r.input_vars[0], r.input_vars[1]]])
        reward = reward_function(new_state, status, calc_per, calc_ap)
        rewards.append(reward)
        
        q_s_p = model.predict(scale(new_state))
    
        targets = model.predict(scale(state))
        
        if not done:  target = reward + discount_factor * np.max(q_s_p)
        else:         target = reward
        
        targets[0][action] = target
    
        h = model.fit(scale(state), targets, epochs=1, verbose=0)
        l = h.history['loss'][0]
        
        state = new_state
        all_loss.append(l)
            
        
    all_rewards.append(np.sum(rewards))
    print("EPOCH: ", i, 'Exploration: ', greed, 'Reward: ', np.sum(rewards))
    if i % 1 == 0:
        all_RX.append(RX)
        all_RY.append(RY)
    if i % 10 == 0: test()
        
#%%
path = '/Final-simple-inc'
        
model.save(f'.{path}/Model_Q_Learning.ms')
np.savetxt(f'.{path}/all_RX.txt', all_RX, fmt='%s')
np.savetxt(f'.{path}/all_RY.txt', all_RY, fmt='%s')
np.savetxt(f'.{path}/all_rewards.txt', all_rewards, fmt='%s')
np.savetxt(f'.{path}/all_loss.txt', all_loss, fmt='%s')



