#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 20:30:19 2021

@author: thomasdixon
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
import RK4_numba as RK4

r = RK4.Rocket()

dAngle = 0.25*np.pi
dThrottle = 0.05
dt = 0.1


def step(r, action):
    pt_vars = np.array([r.rx, r.ry, r.vx, r.vy, r.fuelm], dtype=np.float32)
    
    angle, throttle, staged = r.input_vars
    
    if action == 0:
        angle += dAngle
    elif action == 1:
        angle -= dAngle
    elif action == 2:
        angle = angle
    else:
        print("NO ACTION SELECTED")
        
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



def reward_function(state):
    rx, ry, vx, vy, angle = state[0]
    
    x = np.sqrt(rx**2+ry**2)
    
    rw_x = 0.00001*(x-r.sea)
    
    return rw_x


inputs = keras.Input(shape=(5))
x = keras.layers.Dense(32, activation="relu")(inputs)
outputs = keras.layers.Dense(3, activation="linear")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss="mean_squared_error")


def test(model):
    r = RK4.Rocket()
    state = np.array([[r.rx, r.ry, r.vx, r.vy, r.input_vars[0]]])
    
    RX = []
    RY = []
    rewards = []

    done = False

    while not done:
        action = np.argmax(model.predict(state))
            
        RX.append(r.rx)
        RY.append(r.ry)
            
        r, done = step(r, action)

        new_state = np.array([[r.rx, r.ry, r.vx, r.vy, r.input_vars[0]]])
        reward = reward_function(state)
        rewards.append(reward)
        
        state = new_state            
    
    print('Test Reward: ', np.mean(rewards))
    return RX, RY, np.mean(rewards)


epochs = 100
greed = 1
greed_decay = 0 #0.001
discount_factor = 0.99

all_RX = []
all_RY = []
all_rewards =[]
all_loss = []

test_RX = []
test_RY = []
test_rewards =[]

for i in range(epochs):
    print("EPOCH: ", i)
    r = RK4.Rocket()
    state = np.array([[r.rx, r.ry, r.vx, r.vy, r.input_vars[0]]])
    
    if greed > 0.01:
        greed -= greed_decay
    
    RX = []
    RY = []
    rewards = []

    done = False

    while not done:  
        if np.random.random() < greed:
            action = np.random.randint(0, 3)
        else:
            action = np.argmax(model.predict(state))
            
        RX.append(r.rx)
        RY.append(r.ry)
            
        r, done = step(r, action)

        new_state = np.array([[r.rx, r.ry, r.vx, r.vy, r.input_vars[0]]])
        reward = reward_function(state)
        rewards.append(reward)
        
        q_s_p = model.predict(new_state)
    
        targets = model.predict(state)
    
        if not done:  target = reward + discount_factor * np.max(q_s_p[0])
        else:         target = reward
        targets[0][action] = target
        
        #print(target)
        #print(targets)
        h = model.fit(state, targets, epochs=1, verbose=0)
        l = h.history['loss'][0]
    
        state = new_state
        all_loss.append(l)
        
    all_rewards.append(np.mean(rewards))
    print('Reward: ', np.mean(rewards))
    if i % 1 == 0:
        tRX, tRY, treward = test(model)
        test_RX.append(tRX)
        test_RY.append(tRY)
        test_rewards.append(treward)
        
        all_RX.append(RX)
        all_RY.append(RY)
        

path = '/basic'
        
model.save(f'.{path}/Model_Q_Learning.ms')
np.savetxt(f'.{path}/all_RX.txt', all_RX, fmt='%s')
np.savetxt(f'.{path}/all_RY.txt', all_RY, fmt='%s')
np.savetxt(f'.{path}/all_rewards.txt', all_rewards, fmt='%s')
np.savetxt(f'.{path}/all_loss.txt', all_loss, fmt='%s')

np.savetxt(f'.{path}/test_RX.txt', test_RX, fmt='%s')
np.savetxt(f'.{path}/test_RY.txt', test_RY, fmt='%s')
np.savetxt(f'.{path}/test_rewards.txt', test_rewards, fmt='%s')



