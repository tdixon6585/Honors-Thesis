#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 16:46:49 2021

@author: thomasdixon
"""


import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


epochs = 40
greed = 1
greed_decay = 0.001
#greed_decay = 0.00001
discount_factor = 0.99
sync_target_steps = 1

replay_memory = []
max_mem_size = 100000
batch_size = 64
min_mem_size = batch_size


env = gym.make("CartPole-v1")
#env = gym.make("MountainCar-v0")
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

def make_Network():
    state_space = observation_space
    inputs = keras.Input(shape=(state_space))
    X = keras.layers.Dense(64, activation="relu")(inputs)
    X = keras.layers.Dense(32, activation="relu")(X)

    state_value = keras.layers.Dense(16)(X)
    state_value = keras.layers.Dense(1)(state_value)
    state_value = keras.layers.Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(action_space,))(state_value)
    
    action_advantage = keras.layers.Dense(16)(X)
    action_advantage = keras.layers.Dense(action_space)(action_advantage)
    action_advantage = keras.layers.Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_space,))(action_advantage)
    
    X = keras.layers.Add()([state_value, action_advantage])
    
    
    model = keras.Model(inputs=inputs, outputs=X)
    return model

'''
model = keras.Sequential()
model.add(keras.layers.Dense(64, activation="relu", input_shape=(observation_space,)))
model.add(keras.layers.Dense(32, activation="relu"))
model.add(keras.layers.Dense(action_space, activation="linear"))

target_model = keras.Sequential()
target_model.add(keras.layers.Dense(64, activation="relu", input_shape=(observation_space,)))
target_model.add(keras.layers.Dense(32, activation="relu"))
target_model.add(keras.layers.Dense(action_space, activation="linear"))
'''
model = make_Network()
target_model = make_Network()

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
    y = gym.make("CartPole-v1")
    y_state = y.reset()
    print(model.predict(np.array([y_state])), target_model.predict(np.array([y_state])))





run = 0
for i in range(epochs):
    run += 1
    state = env.reset()
    step = 0
    
    #if i % 3 == 0:  test()
        
    if run % sync_target_steps == 0:
        target_model.set_weights(model.get_weights()) 
    
    while True:
        step += 1
        #env.render()
        if greed > 0.01:
            greed -= greed_decay
        
        if np.random.random() < greed:
            action = np.random.randint(0, action_space)
        else:
            action = np.argmax(model.predict(np.array([state])))
        
        state_next, reward, terminal, info = env.step(action)
        
        if len(replay_memory) >= max_mem_size:
            replay_memory.pop(0)  
        replay_memory.append({'s': state, 'a': action, 'r': reward, "s_p": state_next, 'done': terminal})
        
        
        if len(replay_memory) > min_mem_size:
            model, l = train(replay_memory, batch_size)
        
        state = state_next
        if terminal:
            print("Run: " + str(run) + ", exploration: " + str(greed) \
                  + ", score: " + str(step))
            break
        
        
        
        
        
        
        
        
        
        
        