#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:12:29 2019

@author: dennis
"""

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    x = list(range(25,301, 25))    
    fig, ax = plt.subplots()
    plt.title("Training Time on CONLL 2000 Chunking Shared Task")
    plt.xlabel("Dataset size in sentences")
    plt.ylabel("Measured real execution time in s")
    
    with open("baseline_time.txt") as f:
        lines = f.readlines()
    y = {k: list() for k in x}
        
    counter = 0
    for line in lines:
        if line.split("\t")[0] == "real":
            time_string = line.rstrip("\n").split("\t")[-1]
            time_split = time_string.split("m")
            actual_time = int(time_split[0])*60 + int(time_split[1].split(",")[0])
            
            y[x[counter%12]].append(actual_time)
            
            counter += 1
            
    y = [np.mean(v) for k,v in y.items()]

    plt.plot(x,y, 'o-', label="Baseline")
    
    
    with open("dagger_time.txt") as f:
        lines = f.readlines()
    y = {k: list() for k in x}
        
    counter = 0
    for line in lines:
        if line.split("\t")[0] == "real":
            time_string = line.rstrip("\n").split("\t")[-1]
            time_split = time_string.split("m")
            actual_time = int(time_split[0])*60 + int(time_split[1].split(",")[0])
            
            y[x[counter%12]].append(actual_time)
            
            counter += 1
            
    y = [np.mean(v) for k,v in y.items()]

    plt.plot(x,y, 'o-', label="DAgger")
    
    
    with open("nodagger_time.txt") as f:
        lines = f.readlines()
    y = {k: list() for k in x}
        
    counter = 0
    for line in lines:
        if line.split("\t")[0] == "real":
            time_string = line.rstrip("\n").split("\t")[-1]
            time_split = time_string.split("m")
            actual_time = int(time_split[0])*60 + int(time_split[1].split(",")[0])
            
            y[x[counter%12]].append(actual_time)
            
            counter += 1
            
    y = [np.mean(v) for k,v in y.items()]

    plt.plot(x,y, 'o-', label="NoDAgger")
    
    plt.legend()
    
    plt.savefig("chunking_time.png")
