#!/usr/bin/env bash
set -e

#######################################################
#
#This module will train the DLC model, make sure you have labelled the frames extracted from the previous step 
#before training! Default training iterations is set for 100k, but if you use the test flag it will be set to
#10k training iterations. 100k is probably good for a full model. There are a lot of required options here
#so please read the notes carefully.
#
#######################################################

help_message () {
	echo ""
	echo "Note: All options are required!!"
	echo "MAKE SURE YOU HAVE METADATA TEMPLATE FILLED OUT AND SAVED WITH EXPT NAME"
	echo "MAKE SURE YOU ARE IN THE p_dlc ENV (or equivalent)"
	echo ""
	echo ">>>> Usage: pipeline train -e [expt_name] [options]"
	echo "	Options:"
	echo ""
	echo "		-e		STR		Enter the experiment name used in the metadata file"
	echo "		--iters	|-i	INT		(Default) Use Uniform algo to extract frames"
	echo "		--gpufrac | -g	INT		Percentage of GPU to use (default 60)"
	echo ""
	echo "		Run steps:"
	echo ""
	echo "		--step INT 	See options below"
	echo "		0------------Run training step"				
	echo "		1------------Run evaluation step (evaluate, analyze, filter/plot, and label videos)"
	echo ""
	echo " 		The below are optional but must be run in succession"
	echo "		2------------Extract outlier frames for retrain"
	echo "		3------------Refine labels on extracted outlier frames"
	echo "		4------------Retrain model refined labels"
	echo "		Run step 1 again to evaliate after 4"
	echo "		** It is best to just retrain then model with the refined labels instead of just continuing (i.e. use --trainver 'new')"
	echo ""
	echo "		The below options should be a string formatted like a python list, with one entry for each condition. DO NOT ADD SPACES!!"
	echo ""
	echo "		--condition		STR		List of conditions (e.g. wand, behavior...)"
	echo "		--checkpath		STR		List of paths for stored checkpoints (usually None)"
	echo "		--trainver		STR		List of train version options (e.g. new, continue)"
	echo ""
	echo "		Example	--condition 	'wand','behavior'"
	echo "		Example --checkpath 	'None','None'"
	echo "		Example --trainver	'new','new'"
	echo "!## Read the documentation closely to determine which options are what you need. ##!"
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

checkpath="None";trainver="new";

while true; do
	case "$1" in
		--iters | -i) iters=$2; shift 2;;
		--gpufrac | -g) gpu_frac=$2; shift 2;;
		--condition) conditions="$2"; shift 2;;
		--checkpath) checkpath="$2"; shift 2;;
		--trainver) trainver="$2"; shift 2;;
		--step) step=$2; shift 2;;
		-e) name="$2"; shift 2;;
		-h | --help) help_message; exit 1; shift 1;;
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

if [ $step -lt 3 ]; then
	python ${pyvm}/dlc/run.py ${name} --step ${step} --iters ${iters} --cond ${conditions} --checkp ${checkpath} --tver 'new' --frac ${gpu_frac}
	checkpoints="${data_dir}/${animal}/${name}/checkpoints"
	touch "${checkpoints}/train${step}"
elif [ $step == 3 ]; then
	cd ${pyvm}/dlc
	labels=$(< ${pyvm}/dlc/refine_labels.py)
	echo "name = '${name}';condition='${conditions}';${labels}" | ipython
	checkpoints="${data_dir}/${animal}/${name}/checkpoints"
	touch "${checkpoints}/train${step}"
elif [ $step == 4 ]; then
	python ${pyvm}/dlc/run.py ${name} --step ${step} --iters ${iters} --cond ${conditions} --checkp ${checkpath} --tver ${trainver} --frac ${gpu_frac}
	checkpoints="${data_dir}/${animal}/${name}/checkpoints"
	touch "${checkpoints}/train${step}"
fi

if [ $step == 1 ]; then
	echo "Please take a look at the labelled videos to ensure quality. You can use a command like mpv --fs --pause *.mp4 to play all videos in a folder frame by frame (use the . button to advance frames and the , button to move back frames)"
fi

