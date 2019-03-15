#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 09:36:34 2019

@author: Dennis Aumiller
"""

import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    
    x = list(range(25,301, 25))    
    fig, ax = plt.subplots()
    plt.title("Task Performance on CONLL 2000 Chunking Shared Task")
    plt.xlabel("Dataset size in sentences")
    plt.ylabel("Test F1")
    
    with open("baseline_accuracy.txt") as f:
        lines = f.readlines()
    y = {k: list() for k in x}
    
    for i, line in enumerate(lines):
        f1 = float(line.rstrip("\n").split(" ")[-1])
        y[x[i%12]].append(f1)
        
    y = [np.mean(v) for k,v in y.items()]

    plt.plot(x,y, 'o-', label="Baseline")
    
    
    with open("dagger_accuracy.txt") as f:
        lines = f.readlines()
    y = {k: list() for k in x}
    
    for i, line in enumerate(lines):
        f1 = float(line.rstrip("\n").split(" ")[-1])
        y[x[i%12]].append(f1)
        
    y = [np.mean(v) for k,v in y.items()]
    plt.plot(x,y, 'o-', label="DAgger")
    
    
    with open("nodagger_accuracy.txt") as f:
        lines = f.readlines()
    y = {k: list() for k in x}
    
    for i, line in enumerate(lines):
        f1 = float(line.rstrip("\n").split(" ")[-1])
        y[x[i%12]].append(f1)
        
    y = [np.mean(v) for k,v in y.items()]
    plt.plot(x,y, 'o-', label="NoDAgger")
    
    plt.legend()
    
    plt.savefig("chunking.png")
    
    

    
