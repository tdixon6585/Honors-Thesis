#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 20:30:19 2021

@author: thomasdixon
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import RK4_numba as RK4

r = RK4.Rocket()

dAngle = 0.10*np.pi
dThrottle = 0.25
dt = 0.5


def step(r, action):
    pt_vars = np.array([r.rx, r.ry, r.vx, r.vy, r.fuelm], dtype=np.float32)
    
    angle, throttle, staged = r.input_vars
    
    '''
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
    '''
    
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
    elif r.t > 10000:
        print("Time Limit")
        done = True
    #ACHIEVE ORBIT HERE DON"T FORGET TO INCLUDE THIS HERE 
    else:
        done = False
        
    r.input_vars = np.array([angle, throttle, staged], dtype=np.float32)
    r.RK4_step(pt_vars, dt, r.input_vars)
    
    return r, done

'''
def reward_function(state):
    rew = 1
    return rew
'''

'''
def reward_function(state):
    rx, ry, vx, vy, angle = state[0]
    
    x = np.sqrt(rx**2+ry**2)
    
    
    if x <= r.sea:
        return 0
    
    
    rw_x = np.exp(-1*(x-r.sea-200000)**2/(2*40000**2))
    #punish = -np.exp(-1*(x-r.sea)**2/(2*30000**2))
    punish = 0.26*np.log(x-r.sea+1100)-2.8
    rew = rw_x + punish + 1
    
    return rew
'''

'''

perfect:
    orbit = 200km from Earth surface
    v_perfect = 7784.26
    
    vy = - v_perfect * rx/r
    vx = v_perfect * ry/r


'''

def reward_function(state):
    rx, ry, vx, vy, m, _, angle, throttle = state[0]
    
    rad = np.sqrt(rx**2+ry**2)
    
    v_perfect = 7784.26
    
    vy_p = -v_perfect * rx/rad
    vx_p = v_perfect * ry/rad
    
    rw_vy = (np.exp(-1*(vy-vy_p)**2/((vy_p/2 + 2)**2)) + np.exp(-1*(vy-vy_p)**2/((vy_p/20 + 2)**2)) )/2
        
    rw_vx = (np.exp(-1*(vx-vx_p)**2/((vx_p/2 + 2)**2)) + np.exp(-1*(vx-vx_p)**2/((vx_p/20 + 2)**2)) )/2
        
    if rad <= r.sea:
        return 0
    
    rw_x = np.exp(-1*(rad-r.sea-200000)**2/(2*40000**2))
    punish = 0.26*np.log(rad-r.sea+1100)-2.8
    
    rew = rw_x + punish + 1 + rw_vy + rw_vx
    return rew

action_space = 5
state_space = 8

def make_Network():
    inputs = keras.Input(shape=(state_space))
    X = keras.layers.Dense(128, activation="relu")(inputs)
    X = keras.layers.Dense(64, activation="relu")(X)

    state_value = keras.layers.Dense(32)(X)
    state_value = keras.layers.Dense(1)(state_value)
    state_value = keras.layers.Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(action_space,))(state_value)
    
    action_advantage = keras.layers.Dense(32)(X)
    action_advantage = keras.layers.Dense(action_space)(action_advantage)
    action_advantage = keras.layers.Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_space,))(action_advantage)
    
    X = keras.layers.Add()([state_value, action_advantage])
    
    
    model = keras.Model(inputs=inputs, outputs=X)
    return model

model = make_Network()
target_model = make_Network()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-7),
              loss="mean_squared_error")

target_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-7),
              loss="mean_squared_error")

model.set_weights(np.array(model.get_weights())/100)

target_model.set_weights(model.get_weights())

def scale(s):
    #a = []
    #a.append(s[:,0][0]/6478000)
    #a.append(s[:,1][0]/6478000)
    #a.append(s[:,2][0]/60000)
    #a.append(s[:,3][0]/60000)
    #a.append(s[:,4][0]/(2*np.pi))
    #a = np.array([a])
    return s#a

def train(replay_memory, batch_size):
    batch = np.random.choice(replay_memory, batch_size, replace=False)
    
    s_p = np.array(list(map(lambda x: x['s_p'], batch)))
    s = np.array(list(map(lambda x: x['s'], batch)))
    
    s = scale(s)
    s_p = scale(s_p)

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

def test():
    y = RK4.Rocket()
    y_state = np.array([[y.rx, y.ry, y.vx, y.vy, y.m, int(y.has_staged), y.input_vars[0], y.input_vars[1]]])
    y_state_scaled = scale(y_state)
    print(model.predict(y_state_scaled), target_model.predict(y_state_scaled))

epochs = 1
greed = 1 
greed_decay = 0.00001
#greed_decay = 0.0003
#discount_factor = 0.999
discount_factor = 0.995

replay_memory = []
max_mem_size = 1000000
batch_size = 128
min_mem_size = batch_size

#sync_target_steps = 50000
sync_target_steps = 5000

all_RX = []
all_RY = []
all_rewards =[]
all_loss = []


test_RX = []
test_RY = []
test_rewards =[]

c = 0
i = 0
w = 0
#for i in range(epochs):
test()
while greed >= 0.01 or w < 50:
    if greed <= 0.01:
        w += 1
    i += 1
    r = RK4.Rocket()
    state = np.array([[r.rx, r.ry, r.vx, r.vy, r.m, int(r.has_staged), r.input_vars[0], r.input_vars[1]]])
    
    RX = []
    RY = []
    rewards = []

    done = False

    while not done: 
    #for ww in range(1):
        c+=1
        
        if greed > 0.01:
            greed -= greed_decay

        if np.random.random() < greed:
            action = np.random.randint(0, action_space)
        else:
            state_scaled = scale(state)
            action = np.argmax(model.predict(state_scaled))
            
        RX.append(r.rx)
        RY.append(r.ry)
            
        r, done = step(r, action)

        new_state = np.array([[r.rx, r.ry, r.vx, r.vy, r.m, int(r.has_staged), r.input_vars[0], r.input_vars[1]]])
        reward = reward_function(state)
        rewards.append(reward)
        
        if len(replay_memory) >= max_mem_size:
            replay_memory.pop(0)  
            
        #Prioritized replay memory
        #USE MODEL.EVALUATE TO CALCULATE LOSS OF NEW MEMORY
        #CALCULATE PROBABILITY
        #  SUM ALL LOSS, DIVIDE EACH BY TOTAL
        #ADD TO REPLAY MEMORY
        #PROBLEM: this loss changes when target is updated
        #         Each probability changes every step: inefficient
        replay_memory.append({'s': state[0], 'a': action, 'r': reward, "s_p": \
                              new_state[0], 'done': done})
        
        if len(replay_memory) >= min_mem_size:
            model, l = train(replay_memory, batch_size)
            all_loss.append(l)
    
        state = new_state
        
        if c % sync_target_steps == 0:
            #print('UPDATED')
            target_model.set_weights(model.get_weights())
        
    print("EPOCH: ", i, 'Exploration: ', greed, 'Reward: ', np.sum(rewards))
    all_rewards.append(np.sum(rewards))
    if i % 1 == 0:
        all_RX.append(RX)
        all_RY.append(RY)
    if i % 10 == 0: test()
    
        
#%%

path = '/Final1'
        
model.save(f'.{path}/Model_Q_Learning.ms')
np.savetxt(f'.{path}/all_RX.txt', all_RX, fmt='%s')
np.savetxt(f'.{path}/all_RY.txt', all_RY, fmt='%s')
np.savetxt(f'.{path}/all_rewards.txt', all_rewards, fmt='%s')
np.savetxt(f'.{path}/all_loss.txt', all_loss, fmt='%s')

np.savetxt(f'.{path}/test_RX.txt', test_RX, fmt='%s')
np.savetxt(f'.{path}/test_RY.txt', test_RY, fmt='%s')
np.savetxt(f'.{path}/test_rewards.txt', test_rewards, fmt='%s')



