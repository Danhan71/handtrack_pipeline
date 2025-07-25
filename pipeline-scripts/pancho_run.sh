#!/bin/bash

#Script to run dates in background on server, with log files

# dates=(230616 220716 240508 240524 230622 230623 230626)
#220604 -- has some trial mismatch in cams, if crips up more then handle otherwise will just ignore
# dates=(231114 231116 220902 220906 220907 220908 220909 230920 230921 230923 231019 231020 240828 240829 220829 220830 220831 220901 230810 230811 230824 230826 230829 250321 250322 250324 250325)
# dates=(231114)
# dates=(220618 220626 220630 230125 230119 230120 230126 230127 220616 220621 220622 220624 220614 231114 231116 220902 220906 220907 220908)
# dates=(220715 220716 220717 220724 240530)
dates=(220724 240530)
# dates=(220618)
# dates=(230125 230120)
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
    tail -1000 "./logs/${animal}/${i}.log" > "./logs/${animal}/${i}.log.tmp" && \
    mv "./logs/${animal}/${i}.log.tmp" "./logs/${animal}/${i}.log"
done