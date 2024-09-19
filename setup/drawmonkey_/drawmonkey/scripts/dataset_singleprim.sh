#!/bin/bash

# datelist=(230525 230527 230601 230613) #
# animal=Diego

datelist=(230613) # missed
animal=Diego

for date1 in "${datelist[@]}"
do
    logfile="log_dataset_singleprim-${date1}_${animal}.txt"
	touch ${logfile}
    echo ${logfile}
    # python dataset.py plot_primsingrid null ${date} ${date} ${animal} daily null True 2>&1 | tee ${logfile} &
    python -m scripts.dataset plot_singleprim null ${date1} ${date1} ${animal} daily null True 2>&1 | tee ${logfile} &
    sleep 3s
done

