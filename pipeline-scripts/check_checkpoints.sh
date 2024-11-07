#!/bin/bash

##
#Script to check dates that have gone through to certain checkpoint
##

help_message () {
	echo ""
	echo "Make sure config file is filled out with the proper information/dirs"
    echo ""
    echo " >>>> Usage : ./check_dates_checkpoints {animal} {checkpoint_name}"
    echo ""
    echo "  *Checkpoint names are the same as the various pipeline modules"
    echo "  * If multiple steps, use [name][step#]"
    echo "  E.G: analyze, wand1, wand2,...,wand5,done"
    echo "  Use done as checkpoinbt to check dates fully run"
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
checkp="$2"

if [ "$checkp" = "done" ]; then
    checkp="wand5"
fi

mapfile -t dirs < <(find "${data_dir}/${animal}" -name "${checkp}" -exec dirname {} \; | sort -u)

for dir in "${dirs[@]}"
do 
    dep_count=$(grep -o "/" <<< "$dir" | wc -l)
    # echo $dep_count
    # echo $dir
    expt=$(echo "$dir" | cut -d'/' -f${dep_count})
    echo $expt
done