#!/bin/bash

#Script to run dates in background on server, with log files

# dates=(230913 231118)
# dates=(230726 230816 230817 230913 230914 231116 231118 240822 240827 230920 230921 230922 230924 230925 230723 230724 230727 230728 230730 230815 230915 250319 250320 250321)
# dates=(230726 230816 230817 230913 230914 231116 231118 240822 240827 230920)
dates=(231122 231128 231201 231205 231220 231120 231204 231206 231211 231213 231218)
# date=(231219)
# dates=(231120 231128)
animal="Diego"

for i in "${dates[@]}"
do
    if [ -f ./logs/${animal}/${i}.log ]; then
        rm ./logs/${animal}/${i}.log
    fi
    if [[ "$i" < "231021" ]]; then
        dir_type="g"
    else
        dir_type="l"
    fi
    taskset -a -c 10-19 nohup ../notrain_run_script.sh -e ${i} -b -a ${animal} --loop -${dir_type} > ./logs/${animal}/${i}.log 2>&1 &
    wait
    echo "$(tail -1000 ./logs/${animal}/${i}.log)" > ./logs/${animal}/${i}.log
done

