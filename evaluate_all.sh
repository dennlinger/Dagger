#!/bin/bash

randoms=(12345 54321 2341 4444)
for i in ${randoms[@]}; do
    for nums in `seq 25 25 300`; do
        (time python3 baseline.py PTB_PoS_pairs.csv $nums $i) &>> baseline_time.txt
        (time python3 dagger.py PTB_PoS_pairs.csv $nums $i) &>> dagger_time.txt
        (time python3 nodagger.py PTB_PoS_pairs.csv $nums $i) &>> nodagger_time.txt;
    done;
done
