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
from tensorflow import keras
import multiprocessing
import os

#%%


num_data_points = 100
Z = NN_p.create_data(num_data_points)

X = Z[:,:-1]
y = Z[:,-1]
y = y.reshape(-1,1)

def KFold_iteration_Basic(Z, layers, n_splits, learning_rate):
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
        nn = NN_p.NN(layers, learning_rate = learning_rate)

        accuracies = nn.train(Z_train,500)
        all_accuracies.append(accuracies)

        #TEST NN
        X_test, y_test = Z_test[:,:-1], Z_test[:,-1]
        y_test = y_test.reshape(-1,1)

        predictions = nn.predict(X_test)
        acc = nn.evaluate(predictions,y_test)

        test_accuracies.append(acc)
    
    return all_accuracies, test_accuracies
    
    
def KFold_iteration_Keras(Z, layers, n_splits, learning_rate):
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
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.0),
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


def run_Network(tup):   
    layers, learning_rate = tup
    
    Ball_accs, Btest_accs = KFold_iteration_Basic(Z, layers, n_splits, learning_rate)
    Kall_accs, Ktest_accs = KFold_iteration_Keras(Z, layers, n_splits, learning_rate)
    
    Btest_acc = np.array(Btest_accs).mean()
    Btrain_acc = np.array(Ball_accs)[:,-1].mean()
    
    Ktest_acc = np.array(Ktest_accs)[:,1].mean()
    Ktrain_acc = np.mean([i.history['binary_accuracy'][-1] for i in Kall_accs])
    
    print(os.getpid())
    print(layers, learning_rate)
    print("Basic Testing accuracies: ", Btest_acc)
    print("Basic Training accuracies: ", Btrain_acc)
    print("Keras Testing accuracies: ", Ktest_acc)
    print("Keras Training accuracies: ", Ktrain_acc)
    
    return (Btest_acc, Btrain_acc, Ktest_acc, Ktrain_acc)
    
    #fB.write(f"\n{layers}\t{round(test_acc,3)}\t{round(train_acc,3)}")
    #fB.flush()


#%%

all_layers = [
    [2,2,1], [2,3,1]#, [2,4,1], [2,5,1], [2,6,1], [2,7,1], [2,8,1],
    #[2,2,2,1], [2,2,4,1], [2,2,6,1]
]

learning_rates = [1e-1,1e-2]#,1e-3,1e-4,1e-5,1e-6]

input_array = [(x,y) for x in all_layers for y in learning_rates]

#%%

n_splits = 2



p = multiprocessing.Pool(4)

results = p.map(run_Network,input_array)

print("DONE!")
print(results)

#f = open("K_Fold_Accuracies.txt", "w")
#fB.write(f"\nLayers\tTesting Accuracy\tTraining Accuracy")

    
#fB.close()
#fK.close()