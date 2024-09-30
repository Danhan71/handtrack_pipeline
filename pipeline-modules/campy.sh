#!/usr/bin/env bash
set -e

#######################################################
#
#Module for campy extraction, make sure you have completed training and data extraction. This module will extract frame
#times and check alignment.
#
#######################################################

help_message () {
	echo ""
	echo "MAKE SURE YOU HAVE METADATA TEMPLATE FILLED OUT AND SAVED WITH EXPT NAME"
	echo "MAKE SURE YOU ARE IN THE p_dlc ENV (or equivalent)"
	echo ""
	echo ">>>> Usage: pipeline campy -e [expt_name]"
	echo "	Options:"
	echo ""
	echo "		-e STR		Enter the experiment name used in the metadata file"
	echo "		-a STR		Animal name (e.g. Pancho or Diego)"
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

#Default values
expt="chunkbyshape4"; sess=1;

while true; do
	case "$1" in
		-e) name="$2"; shift 2;;
		-a) animal="$2"; shift 2;;
		-x) expt="$2"; shift 2;;
		-s) sess=$2; shift 2;;
		-h | --help) help_message; exit 1; shift 1;;
		--) help_message; exit 1; shift; break;;
		*) break;;
	esac		
done


# python3 ${pyvm}/run/campy_extraction.py ${name}

cd ${pyvm}
campy=$(< ${pyvm}/run/campy_inspect.py)
echo "name = '${name}'; animal = '${animal}'; ${campy}" | ipython