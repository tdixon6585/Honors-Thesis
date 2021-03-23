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

dAngle = 1*np.pi
dThrottle = 0.05
dt = 6


def step(r, action):
    pt_vars = np.array([r.rx, r.ry, r.vx, r.vy, r.fuelm], dtype=np.float32)
    
    angle, throttle, staged = r.input_vars
    
    if action == 0:
        angle += dAngle
    elif action == 1:
        angle = angle
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

    
    if r.fuelm <= 0:
        #print("Out Of Fuel")
        done = True
    #CRASHES
    elif np.sqrt(r.rx**2+r.ry**2) < r.sea:
        #print("Crashed")
        done = True
    #TIME CUTOFF
    elif r.t > 100000:
        #print("Time Limit")
        done = True
    #ACHIEVE ORBIT HERE DON"T FORGET TO INCLUDE THIS HERE 
    else:
        done = False
        
    r.input_vars = np.array([angle, throttle, staged], dtype=np.float32)
    r.RK4_step(pt_vars, dt, r.input_vars)
    
    return r, done


def reward_function(state, new_state):
    rx, ry, vx, vy, angle = state[0]
    rx_, ry_, vx_, vy_, angle_ = new_state[0]
    
    if ry_ > ry:
        return 1
    elif ry_< ry:
        return -1
    else:
        return 0


'''
def reward_function(state, new_state):
    rx, ry, vx, vy, angle = state[0]
    
    rw_x = 0.001*(ry-r.sea)
    if ry < r.sea:
        return 0
    
    return rw_x
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



inputs = keras.Input(shape=(5))
x = keras.layers.Dense(64, activation="relu")(inputs)
x = keras.layers.Dense(32, activation="relu")(inputs)
outputs = keras.layers.Dense(2, activation="linear")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-7),
              loss="mean_squared_error")

model.set_weights(np.array(model.get_weights())/100)


def test():
    y = RK4.Rocket()
    y_state = np.array([[y.rx, y.ry, y.vx, y.vy, y.input_vars[0]]])
    print(model.predict(scale(y_state)))


epochs = 100
greed = 1
greed_decay = 0.00005
discount_factor = 0.995

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
while greed > 0.01 or w < 300:
    if greed <= 0.01:
        w += 1
    i+=1

    r = RK4.Rocket()
    state = np.array([[r.rx, r.ry, r.vx, r.vy, r.input_vars[0]]])
    
    RX = []
    RY = []
    rewards = []

    done = False

    while not done:   
        if greed > 0.01:
            greed -= greed_decay
        if np.random.random() < greed:
            action = np.random.randint(0, 2)
        else:
            action = np.argmax(model.predict(scale(state)))
            
        RX.append(r.rx)
        RY.append(r.ry)
            
        r, done = step(r, action)

        new_state = np.array([[r.rx, r.ry, r.vx, r.vy, r.input_vars[0]]])
        reward = reward_function(state, new_state)
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
    if i % 10 ==0: test()
        
#%%
path = '/1D5'
        
model.save(f'.{path}/Model_Q_Learning.ms')
np.savetxt(f'.{path}/all_RX.txt', all_RX, fmt='%s')
np.savetxt(f'.{path}/all_RY.txt', all_RY, fmt='%s')
np.savetxt(f'.{path}/all_rewards.txt', all_rewards, fmt='%s')
np.savetxt(f'.{path}/all_loss.txt', all_loss, fmt='%s')



