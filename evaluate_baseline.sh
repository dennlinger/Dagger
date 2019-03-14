#!/bin/bash

for i in `seq 1500 500 4000`;
do python3 baseline.py PTB_PoS_pairs.csv $i;
done
