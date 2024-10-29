#!/usr/bin/env bash
set -e

#######################################################
#
#Module to complete the checkerboard camera config step
#requires you to run the -0 flag first, then add good frames
#to the metadata file. Finally, run the -1 flag to calibrate.
#
#######################################################

help_message () {
	echo ""
	echo "MAKE SURE YOU HAVE METADATA TEMPLATE FILLED OUT AND SAVED WITH EXPT NAME"
	echo "AND GOOD CHECKERBOARD FRAMES"
	echo "MAKE SURE YOU ARE IN THE p_drawcamera ENV (or equivalent)"
	echo ""
	echo ">>>> Usage: pipeline checkerb [options] -e [expt_name]"
	echo "	Options:"
	echo "		-e STR				Enter the experiment name used in the metadata file"
	echo "		-0				Only run extract frames step (do this if you still need to pick good frames for metadata"
	echo "		-1				Only run calibrate step (frames must already be extracted)"
	echo "		--checkerbdim INT INT		ONLY if you need to enter the dimensions of the checkerboard."
	echo "						(If they are different than 10,7)(nrow vert, n cols). Unlikely you need this."
	echo "		-a			Animal name (If empty, Pancho, really only matters for which dir to look in)"		
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


#default params	
calibrate=false;extract=false;dim1=10;dim2=7;animal="Pancho";

while true; do
	case "$1" in
		--checkerbdim) dim1=$2; dim2=$3; shift 3;;
		-e) name="$2"; shift 2;;
		-h | --help) help_message; exit 1; shift 1;;
		-0) extract=true; shift 1;;
		-1) calibrate=true; shift 1;;
		-a) animal="$2"; shift 2;;
		--) help_message; exit 1; shift; break;;
		*) break;;
	esac		
done

while true; do
    read -p "Is the expt name correct ${name} (y/n)" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer y or n.";;
    esac
done

echo $calibrate

if [ $extract == true -a $calibrate == true ]
then
	echo "Please only pick one option out of extract and calibrate" &&
	help_message &&
	exit 1
elif [ $extract == true ]
then
	echo "Extracting frames..."
	python3 ${pyvm}/run/checkerboard.py ${name} --runtype 0 --dim1 10 --dim2 7 --animal ${animal}
elif [ $calibrate == true ]
then
	echo "Calibrating cameras..."
	python3 ${pyvm}/run/checkerboard.py ${name} --runtype 1 --dim1 10 --dim2 7 --animal ${animal}
	echo "Checkerboard calibration complete, you may now move on to the next step in the procedure."
else
	echo "Please pick a correct option"
	help_message
	exit 1
fi


