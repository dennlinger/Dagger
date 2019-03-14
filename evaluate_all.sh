#!/bin/bash

randoms=(12345 54321 2341 4444)
for i in ${randoms[@]}; do
    for nums in `seq 25 25 300`; do
        (time python3 baseline.py ./NER/train.txt $nums $i) &>> baseline_time.txt
        ./NER/conlleval < ./NER/baseline_eval | grep accuracy >> baseline_accuracy.txt
        (time python3 dagger.py ./NER/train.txt $nums $i) &>> dagger_time.txt
        ./NER/conlleval < ./NER/dagger_eval | grep accuracy >> dagger_accuracy.txt
        (time python3 nodagger.py ./NER/train.txt $nums $i) &>> nodagger_time.txt
        ./NER/conlleval < ./NER/nodagger_eval | grep accuracy >> nodagger_accuracy.txt;
    done;
done
