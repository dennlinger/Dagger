#!/bin/bash

for i in `seq 200 200 4000`;
do python3 dagger.py PTB_PoS_pairs.csv $i;
done
