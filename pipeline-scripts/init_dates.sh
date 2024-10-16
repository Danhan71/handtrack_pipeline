#!/bin/bash
# dates_list1=("220704_charparts2" "230908_dirgrammarpancho6b" "231219_gramdirstimpancho5c")
# for date1 in "${dates_list1[@]}"; do
# 	yes | pipeline init -e ${date1} -a Pancho -d early -c behavior
# done
dates_list2=("230403_gridlinecircle4" "230730_dirgrammardiego3c" "231201_chardiego2c")
for date2 in "${date_list2[@]}"; do
	yes | pipeline init -e ${date2} -a Diego -d early -c behavior
done
