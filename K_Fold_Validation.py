#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 10:08:54 2021

@author: thomasdixon
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import KFold
import NN_functions as NN_p

print("CLEAR 1")

from tensorflow import keras

print("CLEAR 2")

num_data_points = 100
Z = NN_p.create_data(num_data_points)

X = Z[:,:-1]
y = Z[:,-1]
y = y.reshape(-1,1)

def KFold_iteration_Basic(Z, layers, n_splits):
    cv = KFold(n_splits=n_splits, shuffle=False)

    all_accuracies = []
    test_accuracies = []
    for i, (train_index, test_index) in enumerate(cv.split(Z)):
        #print("Train Index: ", train_index, "\n")
        #print("Test Index: ", test_index)

        print('Split ', i)
        #SPLIT DATA
        Z_train, Z_test = Z[train_index], Z[test_index]

        #RUN NN
        nn = NN_p.NN(layers, learning_rate = 0.1)

        accuracies = nn.train(Z_train,500)
        all_accuracies.append(accuracies)

        #TEST NN
        X_test, y_test = Z_test[:,:-1], Z_test[:,-1]
        y_test = y_test.reshape(-1,1)

        predictions = nn.predict(X_test)
        acc = nn.evaluate(predictions,y_test)

        test_accuracies.append(acc)
    
    return all_accuracies, test_accuracies

all_layers = [
    [2,2,1]#, [2,3,1], [2,4,1], [2,5,1], [2,6,1], [2,7,1], [2,8,1],
    #[2,2,2,1], [2,2,4,1], [2,2,6,1]
]

n_splits = 10

print("Basic NN")
for i in range(len(all_layers)):
    layers = all_layers[i]
    
    
    all_accs, test_accs = KFold_iteration_Basic(Z, layers, n_splits)
    print(layers)
    print("Testing accuracies: ", np.array(test_accs).mean())
    print("Training accuracies: ", np.array(all_accs)[:,-1].mean())
    
def KFold_iteration_Keras(Z, layers, n_splits):
    cv = KFold(n_splits=n_splits, shuffle=False)

    all_accuracies = []
    test_accuracies = []
    for i, (train_index, test_index) in enumerate(cv.split(Z)):
        #print("Train Index: ", train_index, "\n")
        #print("Test Index: ", test_index)

        print('Split ', i)
        #SPLIT DATA
        Z_train, Z_test = Z[train_index], Z[test_index]

        #Build Layers
        for j, l in enumerate(layers):
            if j == 0:
                inputs = keras.Input(shape=(l))
                x = inputs
                continue
            x = keras.layers.Dense(l, activation="sigmoid")(x)
        
        outputs = x

        
        #RUN NN
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-4, momentum=0.0),
              loss=keras.losses.MeanSquaredError(),
             metrics=[keras.metrics.BinaryAccuracy()])
        
        X_train, y_train = Z_train[:,:-1], Z_train[:,-1]
        y_train = y_train.reshape(-1,1)
        
        history = model.fit(X_train, y_train, batch_size=1, epochs=500, verbose=0)
        
        all_accuracies.append(history)
        
        #TEST NN
        X_test, y_test = Z_test[:,:-1], Z_test[:,-1]
        y_test = y_test.reshape(-1,1)
        
        results = model.evaluate(X_test, y_test, batch_size=1, verbose=0)

        test_accuracies.append(results)
    
    return all_accuracies, test_accuracies

all_layers = [
    [2,2,1]#, [2,3,1], [2,4,1], [2,5,1], [2,6,1], [2,7,1], [2,8,1],
    #[2,2,2,1], [2,2,4,1], [2,2,6,1]
]

n_splits = 10

print("Keras NN")
for i in range(len(all_layers)):
    layers = all_layers[i]
    
    
    all_accs, test_accs = KFold_iteration_Keras(Z, layers, n_splits)
    print(layers)
    print("Testing accuracies: ", np.array(test_accs)[:,1].mean())
    print("Training accuracies: ", np.mean([i.history['binary_accuracy'][-1] for i in all_accs]))