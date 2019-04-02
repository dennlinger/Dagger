#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 11:17:26 2019

@author: dennis
"""

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    
    x = list(range(25,301, 25))    
    fig, ax = plt.subplots()
    plt.title("Task Performance on Penn Treebank PoS Tagging Task")
    plt.xlabel("Dataset size in sentences")
    plt.ylabel("Weighted F1")
    
    with open("baseline_results.csv") as f:
        lines = f.readlines()
        
    y = {k: list() for k in x}
    for line in lines:
        processed = line.rstrip("\n").split(" ")
        ind = int(processed[0])
        value = float(processed[-1])
        
        y[ind].append(value)
    
    y = [np.mean(v) for k,v in y.items()]

    plt.plot(x,y, 'o-', label="Baseline")
    
    
    with open("dagger_results.csv") as f:
        lines = f.readlines()
        
    y = {k: list() for k in x}
    for line in lines:
        processed = line.rstrip("\n").split(" ")
        ind = int(processed[0])
        value = float(processed[-1])
        
        y[ind].append(value)
    
    y = [np.mean(v) for k,v in y.items()]

    plt.plot(x,y, 'o-', label="DAgger")
    
    
    with open("nodagger_results.csv") as f:
        lines = f.readlines()
        
    y = {k: list() for k in x}
    for line in lines:
        processed = line.rstrip("\n").split(" ")
        ind = int(processed[0])
        value = float(processed[-1])
        
        y[ind].append(value)
    
    y = [np.mean(v) for k,v in y.items()]

    plt.plot(x,y, 'o-', label="NoDAgger")
    
    plt.legend()
    
    plt.savefig("pos_perf.png")
        