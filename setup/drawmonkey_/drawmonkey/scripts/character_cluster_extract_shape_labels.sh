#!/bin/bash -e

animal=$1

if [[ $animal == Diego ]]; then
#  datelist=(231201 231204 231219)
#  datelist=(231130)
  datelist=(231205 231207 231211 231220)
elif [[ $animal == Pancho ]]; then
  #datelist=(220618 220627 220630 230119 230122 230126)
  datelist=(220618 220627 220630 230119 230122)
else
  echo $animal
  echo "Error! Inputed non-existing animal" 1>&2
  exit 1
fi

for date1 in "${datelist[@]}"
do
  logfile="../logs/character_cluster_extract_${animal}_${date1}.txt"
  touch ${logfile}
  echo ${logfile}
  taskset --cpu-list 0,1,2,3,4,5 python character_cluster_extract_shape_labels.py ${animal} ${date1} 2>&1 | tee ${logfile} &
  sleep 10s
done

