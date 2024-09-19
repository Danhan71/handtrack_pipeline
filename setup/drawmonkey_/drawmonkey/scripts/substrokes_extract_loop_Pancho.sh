#!/bin/bash

#datelist=(220715 220716 220718 220719 220918 221217) # All
datelist=(220918) # Geetting missed ones
animal=Pancho

for date1 in "${datelist[@]}"
do
  logfile="~/code/drawmonkey/logs/log_substrokes-${date1}_${animal}.txt"
  touch ${logfile}
  echo ${logfile}
  python substrokes_extract.py ${animal} ${date1} 2>&1 | tee ${logfile} &
  sleep 2s
done
