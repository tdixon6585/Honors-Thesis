#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 18:53:05 2021

@author: thomasdixon
"""


import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import RK4_numba as RK4
import random
import time

r = RK4.Rocket()

dAngle = 0.01*np.pi
dThrottle = 0.05
dt = 0.1

#R = []
#RX = []
#RY = []
#V = []
#Fuel = []

def step(r, action):
    pt_vars = np.array([r.rx, r.ry, r.vx, r.vy, r.fuelm], dtype=np.float32)
    
    angle, throttle, staged = r.input_vars
    
    if action == 0:
        angle += dAngle
        throttle += dThrottle
    elif action == 1:
        angle -= dAngle
        throttle += dThrottle
    elif action == 2:
        angle += dAngle
        throttle -= dThrottle
    elif action == 3:
        angle -= dAngle
        throttle -= dThrottle
    elif action == 4:
        angle = angle
        throttle = throttle
    else:
        print("NO ACTION SELECTED")
        
    if throttle > 1:
        throttle = 1
    elif throttle < 0:
        throttle = 0
        
    if angle > 2*np.pi:
        angle = angle - 2*np.pi
    elif angle < 0:
        angle = angle + 2*np.pi

    
    #OUT OF FUEL
    if r.fuelm <= 0:
        print("Out Of Fuel")
        done = True
    #CRASHES
    elif np.sqrt(r.rx**2+r.ry**2) < r.sea:
        print("Crashed")
        done = True
    #TIME CUTOFF
    elif r.t > 100000:
        print("Time Limit")
        done = True
    #ACHIEVE ORBIT HERE DON"T FORGET TO INCLUDE THIS HERE 
    else:
        done = False
        
    r.input_vars = np.array([angle, throttle, staged], dtype=np.float32)
    r.RK4_step(pt_vars, dt, r.input_vars)
    
    return r, done

'''
actions = {
(+throttle, +angle),
(+throttle, -angle),
(-throttle, +angle),
(-throttle, -angle),
(do nothing)
}
'''
k = 10000


def reward_1(state):
    rx, ry, vx, vy, m, has_staged, angle, throttle = state[0]
    
    x = np.sqrt(rx**2+ry**2)
    v = np.sqrt(vx**2+vy**2)
    
    rw_x = 1/(1+np.exp(-0.00004*(x-r.karman)))
    rw_v = 1/(1+np.exp(-0.001*(v-7790/2)))
    rw_m = 0  #0.1*m/r.mi
    
    punish = 0
    if np.sqrt(r.rx**2+r.ry**2) < r.sea:
        punish = -2
    
    return rw_x + rw_v + rw_m + punish

def reward_2(state):
    rx, ry, vx, vy, m, has_staged, angle, throttle = state[0]
    
    x = np.sqrt(rx**2+ry**2)
    
    rw_x = np.exp(-1*(x-r.sea-200000)**2/(2*40000**2))
    punish = -np.exp(-1*(x-r.sea)**2/(2*30000**2))
    
    return k*rw_x + k*punish

def reward_3(state):
    rx, ry, vx, vy, m, has_staged, angle, throttle = state[0]
    
    x = np.sqrt(rx**2+ry**2)
    
    rw_x = np.exp(-1*(x-r.sea-200000)**2/(5*40000**2))
    
    return rw_x


inputs = keras.Input(shape=(8))
x = keras.layers.Dense(64, activation="relu")(inputs)
x = keras.layers.Dense(32, activation="relu")(x)
outputs = keras.layers.Dense(5, activation="linear")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss="mean_squared_error")




def train(replay_memory, batch_size):
    batch = np.random.choice(replay_memory, batch_size, replace=True)
    
    s_p = np.array(list(map(lambda x: x['s_p'], batch)))
    s = np.array(list(map(lambda x: x['s'], batch)))
    
    q_s_p = model.predict(s_p)

    targets = model.predict(s)

    
    for i,m in enumerate(batch): 
        a = m['a']
        r = m['r']
        done = m['done']
        if not done:  target = r + discount_factor * np.max(q_s_p)
        else:         target = r
        targets[i][a] = target

    h = model.fit(s, targets, epochs=1, verbose=0)
    return model, h.history['loss'][0]



epochs = 300
greed = 1
greed_decay = 0.002
discount_factor = 0.999

replay_memory = []
max_mem_size = 100000
batch_size = 32

all_RX = []
all_RY = []
all_rewards =[]
all_loss = []

for i in range(epochs):
    print("EPOCH: ", i)
    print("replay_memory: ", len(replay_memory))
    r = RK4.Rocket()
    state = np.array([[r.rx, r.ry, r.vx, r.vy, r.m, int(r.has_staged), r.input_vars[0], r.input_vars[1]]])
    if greed > 0.01:
        greed -= greed_decay
    
    RX = []
    RY = []
    rewards = []

    done = False

    while not done:
    
        if np.random.random() < greed:
            action = np.random.randint(0, 5)
        else:
            action = np.argmax(model.predict(state))
            
        RX.append(r.rx)
        RY.append(r.ry)
            
        r, done = step(r, action)

        new_state = np.array([[r.rx, r.ry, r.vx, r.vy, r.m, int(r.has_staged), r.input_vars[0], r.input_vars[1]]])
        reward = reward_2(state)
        rewards.append(reward)
        
        if len(replay_memory) > max_mem_size:
            replay_memory.pop(0)
        replay_memory.append({'s': state[0], 'a': action, 'r': reward, "s_p": new_state[0], 'done': done})
        
        model, l = train(replay_memory, batch_size)
        state = new_state
        all_loss.append(l)
            
        
    all_rewards.append(sum(rewards))
    print('Reward: ', sum(rewards))
    if i % 10 == 0:
        all_RX.append(RX)
        all_RY.append(RY)
        

        
model.save('./Model_Q_Learning.ms')
np.savetxt('all_RX.txt', all_RX, fmt='%s')
np.savetxt('all_RY.txt', all_RY, fmt='%s')
np.savetxt('all_rewards.txt', all_rewards, fmt='%s')
np.savetxt('all_loss.txt', all_loss, fmt='%s')



