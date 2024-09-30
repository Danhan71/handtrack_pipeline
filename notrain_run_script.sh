#!/usr/bin/env bash
set -e

#######################################################
#
#General script for a normal run day. Includes option for inputting condition
#list, for days that you want to run the wand calibration on.
#
#######################################################

help_message () {
	echo ""
	echo "Usage: notrain_run_script.sh"
	echo "MAKE SURE YOU HAVE METADATA TEMPLATE FILLED OUT AND SAVED WITH EXPT NAME"
	echo "AND GOOD CHECKERBOARD FRAMES"
	echo "MAKE SURE YOU ARE IN THE p_drawcamera ENV (or equivalent)"
	echo ""
	echo "Options:"
	echo "		-e STR			Enter the experiment name (e.g. 123456_expt_sess# or just 123456 for loop condition)"
	echo "		-w/b/3			Conditions for model labelling (-w = wand, -b = behavior, -a = both, -3 3dgrid)"
	echo "		-a				Animal"
	echo "		-l/g			-l for late 2023 dir -g for gorilla dir"
	echo "		--loop			Loop through a date folder (with only behavior conditions)"
	echo "						If this is selected enter the date for the name"
	echo ""
}

comm () { ${scripts}/print_comment.py "$1" "-"; }
error () { ${scripts}/print_comment.py "$1" "*"; exit 1; }
warning () { ${scripts}/print_comment.py "$1" "*"; }
announcement () { ${scripts}/print_comment.py "$1" "#"; }
loop=false;
while true; do
	case "$1" in
		-w) cond='wand'; shift 1;;
		-b) cond='behavior'; shift 1;;
		-e) name="$2"; shift 2;;
		-3) cond='3dgrid'; shift 1;;
		-a) animal=$2; shift 2;;
		-h | --help) help_message; exit 1; shift 1;;
		-l) dir="late"; loop_dir="/home/danhan/freiwaldDrive/ltian/camera_late_2023_onwards/${animal}/${name}"; shift 1;;
		-g) dir="early"; loop_dir="/home/danhan/freiwaldDrive/ltian/backup/gorilla/gorilla2/camera/${animal}/${name}"; shift 1;;
		--loop) loop=true; shift 1;;
		--) help_message; exit 1; shift; break;;
		*) break;;
	esac		
done

while true; do
	read -p "Is the expt name/date correct ${name} (y/n)" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) echo "Do not use loop conition or else move this folder"; exit;;
        * ) echo "Please answer y or n.";;
    esac
done


if [ ${loop} = true ]; then
	
	for i in ${loop_dir}/*; do
		# Find dir depth to properly caluclate where the expt name is
		dep_count=$(grep -o "/" <<< "$loop_dir" | wc -l)
		((dep_count+=2))
		echo ${dep_count}
		expt=$(echo "$i" | cut -d'/' -"f${dep_count}")

		this_dir="$(dirname "$0")"
		pipe="${this_dir}/pipeline"

		yes | pipeline init -e ${expt} -a ${animal} -c ${cond} -d ${dir} &&

		yes | pipeline dlc-setup -e ${expt} --${cond} --skiplabel --skipext &&

		yes | pipeline analyze -e ${expt} --cond ${cond}

		yes | pipeline wand -e ${expt} --step 4 --cond ${cond} -a ${animal}

		yes | pipeline wand -e ${expt} --step 5 --reg -a ${animal}

	done
else
	this_dir="$(dirname "$0")"
	pipe="${this_dir}/pipeline"

	# yes | $pipe init -e ${name} -a ${animal} -c ${cond} -d ${dir}

	# yes | pipeline dlc-setup -e ${name} --${cond} --skiplabel --skipext -a ${animal}

	yes | pipeline analyze -e ${name} --cond ${cond} -a ${animal}

	yes | pipeline wand -e ${name} -a ${animal} --step 4 --cond ${cond}

	yes | pipeline wand -e ${name} --step 5 --reg -a ${animal}

fi