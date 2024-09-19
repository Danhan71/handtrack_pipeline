#!/bin/bash

# animals=[ "Pancho" "Red" "Diego" ]
# echo ${animals}
for a in "Pancho" "Red" "Diego" "Taz" "Barbossa" "Lucas"
# for a in "Red"
do
	taskset --cpu-list 0,1,2,3 python -m tools.preprocess ${a} 2>&1 | tee ${a}_preprocess_log.txt & 
done