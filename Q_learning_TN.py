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

dAngle = 0.01*np.pi
dThrottle = 0.05
dt = 0.1


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



def reward_function(state):
    rx, ry, vx, vy, m, has_staged, angle, throttle = state[0]
    
    x = np.sqrt(rx**2+ry**2)
    
    rw_x = np.exp(-1*(x-r.sea-200000)**2/(2*40000**2))
    punish = -np.exp(-1*(x-r.sea)**2/(2*30000**2))
    
    return rw_x + punish + 1

'''
inputs = keras.Input(shape=(8))
x = keras.layers.Dense(64, activation="relu")(inputs)
x = keras.layers.Dense(32, activation="relu")(x)
outputs = keras.layers.Dense(5, activation="linear")(x)
'''


model = keras.Sequential()
model.add(keras.layers.Dense(64, activation="relu", input_shape=(8,)))
model.add(keras.layers.Dense(32, activation="relu"))
model.add(keras.layers.Dense(5, activation="linear"))

target_model = keras.Sequential()
target_model.add(keras.layers.Dense(64, activation="relu", input_shape=(8,)))
target_model.add(keras.layers.Dense(32, activation="relu"))
target_model.add(keras.layers.Dense(5, activation="linear"))


#model = keras.Model(inputs=inputs, outputs=outputs)

#target_model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss="mean_squared_error")

target_model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss="mean_squared_error")


target_model.set_weights(model.get_weights())


def train(replay_memory, batch_size):
    if len(replay_memory) < batch_size:
        batch = np.random.choice(replay_memory, len(replay_memory), replace=False)
    else:
        batch = np.random.choice(replay_memory, batch_size, replace=False)
    
    s_p = np.array(list(map(lambda x: x['s_p'], batch)))
    s = np.array(list(map(lambda x: x['s'], batch)))
    
    s[:,0] = s[:,0]/6478000
    s[:,1] = s[:,1]/6478000
    s[:,2] = s[:,2]/60000
    s[:,3] = s[:,3]/60000
    s[:,4] = s[:,4]/1420788
    s[:,5] = s[:,5]
    s[:,6] = s[:,6]/(2*np.pi)
    s[:,7] = s[:,7]
    
    s_p[:,0] = s_p[:,0]/6478000
    s_p[:,1] = s_p[:,1]/6478000
    s_p[:,2] = s_p[:,2]/60000
    s_p[:,3] = s_p[:,3]/60000
    s_p[:,4] = s_p[:,4]/1420788
    s_p[:,5] = s_p[:,5]
    s_p[:,6] = s_p[:,6]/(2*np.pi)
    s_p[:,7] = s_p[:,7]
    
    
    #Potentially impliment Double DQN
    # Use the main model to make the future choice 
    #  best_actions = model.predict(s_p)
    #  best_actions.argmax() or something like this
    # Use the target model to estimate Q-Value of that choice
    #  q_s_p = target_model.predict(s_p)
    #  make q_s_p only contains the actions chosen from the main model
    #  np.max is no longer necessary

    q_s_p = target_model.predict(s_p)
    best_actions = model.predict(s_p)
    ba_i = np.argmax(best_actions, 1)

    

    
    q_s_p_ba = [q_s_p[i][ba_i[i]] for i in range(len(q_s_p))]

    targets = model.predict(s)


    for i,m in enumerate(batch): 
        a = m['a']
        r = m['r']
        done = m['done']
        
        if not done:  target = r + discount_factor * q_s_p_ba[i]
        else:         target = r
        targets[i][a] = target


    h = model.fit(s, targets, epochs=1, verbose=0)
    

    return model, h.history['loss'][0]

def test(model):
    r = RK4.Rocket()
    state = np.array([[r.rx, r.ry, r.vx, r.vy, r.m, int(r.has_staged), r.input_vars[0], r.input_vars[1]]])
    
    RX = []
    RY = []
    rewards = []

    done = False

    while not done:
        action = np.argmax(model.predict(state))
            
        RX.append(r.rx)
        RY.append(r.ry)
            
        r, done = step(r, action)

        new_state = np.array([[r.rx, r.ry, r.vx, r.vy, r.m, int(r.has_staged), r.input_vars[0], r.input_vars[1]]])
        reward = reward_function(state)
        rewards.append(reward)
        
        state = new_state            
    
    print('Test Reward: ', np.mean(rewards))
    return RX, RY, np.mean(rewards)

epochs = 1000
greed = 1
greed_decay = 0.0001
discount_factor = 0.99

replay_memory = []
max_mem_size = 100000
batch_size = 64
min_mem_size = batch_size

sync_target_steps = 1 #simulations

all_RX = []
all_RY = []
all_rewards =[]
all_loss = []

test_RX = []
test_RY = []
test_rewards =[]

c = 0

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
        reward = reward_function(state)
        rewards.append(reward)
        
        if len(replay_memory) >= max_mem_size:
            replay_memory.pop(0)  
        replay_memory.append({'s': state[0], 'a': action, 'r': reward, "s_p": new_state[0], 'done': done})
        
        if len(replay_memory) > min_mem_size:
            model, l = train(replay_memory, batch_size)
            all_loss.append(l)
        
        state = new_state
        
    c+=1
        
    if c % sync_target_steps == 0:
        target_model.set_weights(model.get_weights()) 
    
    all_rewards.append(np.mean(rewards))
    print('Reward: ', np.mean(rewards))
    if i % 1 == 0:
        tRX, tRY, treward = test(model)

        test_RX.append(tRX)
        test_RY.append(tRY)
        test_rewards.append(treward)
        
        all_RX.append(RX)
        all_RY.append(RY)
        
path = '/T'
        
model.save(f'.{path}/Model_Q_Learning.ms')
np.savetxt(f'.{path}/all_RX.txt', all_RX, fmt='%s')
np.savetxt(f'.{path}/all_RY.txt', all_RY, fmt='%s')
np.savetxt(f'.{path}/all_rewards.txt', all_rewards, fmt='%s')
np.savetxt(f'.{path}/all_loss.txt', all_loss, fmt='%s')

np.savetxt(f'.{path}/test_RX.txt', test_RX, fmt='%s')
np.savetxt(f'.{path}/test_RY.txt', test_RY, fmt='%s')
np.savetxt(f'.{path}/test_rewards.txt', test_rewards, fmt='%s')



