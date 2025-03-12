#!/bin/bash

#Script to run dates in background on server, with log files

# dates=(230616 220716 240508 240524 230622 230623 230626)
#220604 -- has some trial mismatch in cams, if crips up more then handle otherwise will just ignore
dates=(231114 231116 220902 220906 220907 220908 220909 230920 230921 230923 231019 231020 240828 240829)
animal="Pancho"

for i in "${dates[@]}"
do
    if [ -f ./logs/${animal}/${i}.log ]; then
        rm ./logs/${animal}/${i}.log
    fi
    if [[ "$i" < "231231" ]]; then
        dir_type="g"
    else
        dir_type="l"
    fi
    taskset -a -c 0-9 nohup ../notrain_run_script.sh -e ${i} -b -a ${animal} --loop -${dir_type} > ./logs/${animal}/${i}.log 2>&1 &
    wait
    echo "$(tail -1000 ./logs/${animal}/${i}.log)" > ./logs/${animal}/${i}.log
done