#!/bin/bash

##
#Script to check dates that have gone through to certain checkpoint
##

help_message () {
	echo ""
	echo "Make sure config file is filled out with the proper information/dirs"
    echo ""
    echo " >>>> Usage : ./check_checkpoints {animal}"
    echo ""
}

comm () { ${scripts}/print_comment.py "$1" "-"; }
error () { ${scripts}/print_comment.py "$1" "*"; exit 1; }
warning () { ${scripts}/print_comment.py "$1" "*"; }
announcement () { ${scripts}/print_comment.py "$1" "#"; }

this_dir="$(dirname "$0")"
config_file=${this_dir}/../config
source $config_file
if [[ $? -ne 0 ]]; then 
	echo "no config file."
	exit 1
fi

animal="$1"


mapfile -t dirs < <(find "${data_dir}/${animal}" -maxdepth 1 | sort -u)

for dir in "${dirs[@]}"
do 
    dep_count=$(grep -o "/" <<< "$dir" | wc -l)
    dep_count=$((dep_count+1))
    expt=$(echo "$dir" | cut -d'/' -f${dep_count})

    if [ -f "${dir}/checkpoints/done" ] || [ -f "${dir}/wand5" ]; then
        echo "${expt},done"
    elif [ -f "${dir}/checkpoints/wand4" ]; then
        echo "${expt},wand4"
    elif [ -f "${dir}/checkpoints/analyze" ]; then
        echo "${expt},analyze"
    elif [ -f "${dir}/checkpoints/dlc_setup" ]; then
        echo "${expt},dlc_setup"
    elif [ -f "${dir}/checkpoints/init" ]; then
        echo "${expt},init"
    fi
done > ${scripts}/logs/checkpoints/${animal}_checkpoints.csv
sort ${scripts}/logs/checkpoints/${animal}_checkpoints.csv -o ${scripts}/logs/checkpoints/${animal}_checkpoints.csv