#!/bin/bash

#Script to run dates in background on server, with log files

dates=(220608 220609 220610 220715 220716 220717 220718 220719)
animal="Pancho"

for i in "${dates[@]}"
do
    taskset -a -c 0-9 nohup ../notrain_run_script.sh -e ${i} -b -a ${animal} --loop -g > ./logs/${animal}/${i}.log 2>&1 &
    wait
    # ../notrain_run_script.sh -e ${i} -b -a ${animal} --loop -g
done
