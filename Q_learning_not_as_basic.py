
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
from tensorflow.keras import backend as K

r = RK4.Rocket()

dAngle = 0.10*np.pi
dThrottle = 0.05
dt = 0.5


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
    if v > esc_vel:
        print("Escaped")
        done = True
        status = 'escaped'
    elif (calc_per > min_per) and (v < esc_vel):
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

'''
def reward_function(state):
    rew = 1
    return rew
'''
'''
def reward_function(state):
    rx, ry, vx, vy, angle = state[0]
    
    x = np.sqrt(rx**2+ry**2)
    
    rw_x = np.exp(-1*(x-r.sea-200000)**2/(2*40000**2))
    #punish = -np.exp(-1*(x-r.sea)**2/(2*30000**2))
    punish = 0.26*np.log(x-r.sea+1100)-2.8
    rew = rw_x + punish + 1
    if x <= r.sea:
        rew = 0
    #USE NATURAL LOG NEAR THE SURFACE
    # STEEP INCLINE AWAY FROM SURFACE
    
    
    return rew
'''
def reward_function(state, status, calc_per, calc_ap):
    
    rx, ry, vx, vy, angle = state[0]
    
    
    v = np.sqrt(r.vx**2 + r.vy**2)
    rad = np.sqrt(r.rx**2 + r.ry**2)
    
    phi = np.abs(np.arctan(r.rx/(r.ry+1e-9)) - np.arctan(r.vy/(r.vx+1e-9)))
    h = rad*v*np.cos(phi)
    
    mu = r.G * r.M
    semimajor = ((2/rad)-((v**2)/mu))**-1
    ecc = np.sqrt(1-(h**2/(semimajor*mu)))
    
    calc_new_per = semimajor*(1-ecc)

    per_diff = calc_new_per - calc_per
    rew = per_diff/1e3
    
    return rew

'''
model = keras.Sequential()
model.add(keras.layers.Dense(64, activation="relu", input_shape=(5,)))
model.add(keras.layers.Dense(32, activation="relu"))
model.add(keras.layers.Dense(3, activation="linear"))

target_model = keras.Sequential()
target_model.add(keras.layers.Dense(64, activation="relu", input_shape=(5,)))
target_model.add(keras.layers.Dense(32, activation="relu"))
target_model.add(keras.layers.Dense(3, activation="linear"))

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss="mean_squared_error")

target_model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss="mean_squared_error")


target_model.set_weights(model.get_weights())
'''
action_space = 3
state_space = 5

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
    y = RK4.Rocket()
    y_state = np.array([[y.rx, y.ry, y.vx, y.vy, y.input_vars[0]]])
    y_state_scaled = y_state
    print(model.predict(y_state_scaled), target_model.predict(y_state_scaled))


epochs = 1
greed = 1 # 0.01
greed_decay = 0.00005
discount_factor = 0.999

replay_memory = []
max_mem_size = 1000000
batch_size = 128
min_mem_size = batch_size

sync_target_steps = 1

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
while greed >= 0.01 or w < 1000:
    if greed <= 0.01:
        w += 1
    i += 1
    r = RK4.Rocket()
    state = np.array([[r.rx, r.ry, r.vx, r.vy, r.input_vars[0]]])
    
    RX = []
    RY = []
    rewards = []

    done = False

    while not done:  
        c+=1

        if greed > 0.01:
            greed -= greed_decay

        if np.random.random() < greed:
            action = np.random.randint(0, 3)
        else:
            action = np.argmax(model.predict(state))
            
        RX.append(r.rx)
        RY.append(r.ry)
            
        r, done, status, calc_per, calc_ap = step(r, action)

        new_state = np.array([[r.rx, r.ry, r.vx, r.vy, r.input_vars[0]]])
        reward = reward_function(state, status, calc_per, calc_ap)
        rewards.append(reward)
        
        if len(replay_memory) >= max_mem_size:
            replay_memory.pop(0)  
        replay_memory.append({'s': state[0], 'a': action, 'r': reward, "s_p": \
                              new_state[0], 'done': done})
        
        if len(replay_memory) > min_mem_size:
            model, l = train(replay_memory, batch_size)
            all_loss.append(l)
    
        state = new_state
        
        if c % sync_target_steps == 0:
            target_model.set_weights(model.get_weights())
        
    print("EPOCH: ", i, 'Exploration: ', greed, 'Reward: ', np.sum(rewards))
    all_rewards.append(np.sum(rewards))
    if i % 1 == 0:
        all_RX.append(RX)
        all_RY.append(RY)
    if i % 5 == 0: test()
    
        
#%%

path = '/notbasic-per2'
        
model.save(f'.{path}/Model_Q_Learning.ms')
np.savetxt(f'.{path}/all_RX.txt', all_RX, fmt='%s')
np.savetxt(f'.{path}/all_RY.txt', all_RY, fmt='%s')
np.savetxt(f'.{path}/all_rewards.txt', all_rewards, fmt='%s')
np.savetxt(f'.{path}/all_loss.txt', all_loss, fmt='%s')

np.savetxt(f'.{path}/test_RX.txt', test_RX, fmt='%s')
np.savetxt(f'.{path}/test_RY.txt', test_RY, fmt='%s')
np.savetxt(f'.{path}/test_rewards.txt', test_rewards, fmt='%s')



