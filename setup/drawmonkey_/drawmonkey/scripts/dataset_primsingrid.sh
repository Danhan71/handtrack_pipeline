
#!/bin/bash

# datelist=( 220709 220711 220714 220727 220731 220805 230105 230106 230108 230109 230615 230616 230620)
# datelist=(221129 221130 221201 230111 230602 230608 230609) # pigrand
# animal=Pancho

# datelist=(230605 230606) #
datelist=(230605 230606 230607 230608 230609 230610 230612) #
animal=Diego

for date1 in "${datelist[@]}"
do
    logfile="logs/log_dataset_primsingrid-${date1}_${animal}.txt"
	touch ${logfile}
    echo ${logfile}
    python -m scripts.dataset plot_primsingrid null ${date1} ${date1} ${animal} daily null True 2>&1 | tee ${logfile} &
    sleep 3s
done
