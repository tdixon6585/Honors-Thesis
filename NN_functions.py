#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 11:37:22 2021

@author: thomasdixon
"""


import numpy as np
import matplotlib.pyplot as plt


def create_data(num_data_points):
    x = np.random.rand(num_data_points)
    y_p = np.random.rand(num_data_points)

    X = np.array([(x[i], y_p[i]) for i in range(num_data_points)])
    y = []
    for index in range(num_data_points):
        if y_p[index] > 3*(x[index]-0.5)**2 + 0.2:# and y_p[index] < 3*(x[index]-0.5)**2 + 0.55:
            y.append([1])
        else:
            y.append([0])

    y=np.array(y)
    
    Z = np.concatenate((X,y),1)
    return Z

class node_class(object):
    def __init__(self, input_dim, learning_rate=0.1):
        #An array of weights, an additional weight is added for the dummy input `a_o = 1`
        self.W = np.random.rand(int(input_dim+1))
        self.learning_rate = learning_rate
        
    def sigmoid_activation(self, t):
        self.final_output = 1 / (1 + np.exp(-t))
        return self.final_output
    
    def input_function(self, input_array):
        self.input_array = np.append(input_array,1)
        self.input_sum = np.sum(self.input_array * self.W)
        return self.input_sum
    
    def output(self, input_array):
        input_sum = self.input_function(input_array)
        return self.sigmoid_activation(input_sum)
    
    def sigmoid_der(self,t):
        return self.sigmoid_activation(t) * (1-self.sigmoid_activation(t))
        
    def calc_delta(self, actual_y=None, delta_j=None, final_layer=False):
        if final_layer:
            self.delta = self.sigmoid_der(self.input_sum)*(actual_y-self.final_output)
            return self.delta
        else:
            self.delta = self.sigmoid_der(self.input_sum) * np.sum(self.W * delta_j)
            return self.delta
            
    def update_weights(self):
        self.W = self.W + (self.learning_rate * self.input_array * self.delta)
        
        
class NN(object):
    def __init__(self, layers, learning_rate=0.1):
        self.layers_nodes = []
        for i, num_nodes in enumerate(layers):
            if i==0:
                input_dim = num_nodes
                continue
                
            dim = input_dim
            self.layers_nodes.append([])
            for j in range(num_nodes):
                n = node_class(dim, learning_rate=learning_rate)
                self.layers_nodes[i-1].append(n)
                
            input_dim = num_nodes
            
    def evaluate(self, predictions, y, categorical=True):
        if categorical:
            evals = []
            preds = np.array(predictions)
            preds = np.round(preds)
            
            for i in range(len(y)):
                if np.array_equal(preds[i], y[i]):
                    evals.append(1)
                else:
                    evals.append(0)
                    
            evals = np.array(evals)
            return evals.mean()
            
    def predict(self, X):
        preds = []
        for index, i in enumerate(X):
            outputs = [i]
            for h, layer in enumerate(self.layers_nodes):
                outputs.append([])
                for node in layer:
                    output = node.output(outputs[h])
                    outputs[h+1].append(output)
                    
            preds.append(outputs[-1])
                        
        return preds
            
    def train(self, Z, epochs):
        accs = []
        for j in range(epochs):
            np.random.shuffle(Z)
            
            X = Z[:,:-1]
            y = Z[:,-1]
            y = y.reshape(-1,1)
            
            predictions = self.predict(X)
            acc = self.evaluate(predictions,y)
            accs.append(acc)
            for index, i in enumerate(X):
                outputs = [i]
                for h, layer in enumerate(self.layers_nodes):
                    outputs.append([])
                    for node in layer:
                        output = node.output(outputs[h])
                        outputs[h+1].append(output)
                            
                for h, layer in enumerate(reversed(self.layers_nodes)):
                    for g, node in enumerate(layer):
                        if h == 0:
                            delta_j = node.calc_delta(actual_y=y[index][g], final_layer=True)
                        else:
                            delta_j = node.calc_delta(delta_j=delta_j)
                        node.update_weights()
        return accs
    
