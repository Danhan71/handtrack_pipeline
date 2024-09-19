#!/bin/bash

datelist=(230616 230615 230618 230619) #
animal=Diego

for date1 in "${datelist[@]}"
do
  logfile="~/code/drawmonkey/logs/log_substrokes-${date1}_${animal}.txt"
  touch ${logfile}
  echo ${logfile}
  python substrokes_extract.py ${animal} ${date1} 2>&1 | tee ${logfile} &
  sleep 2s
done
