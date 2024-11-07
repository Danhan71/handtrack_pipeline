#!/usr/bin/env bash
set -e

#######################################################
#
#All the neccessary setup before training the DLC model. This will extract frames and stuff for hand labelling
#then open a GUI where you will have to hgand label the frames. This step is only necessary if you are training a new model. 
#It is not necessary if you are retraining or not training.
#
#######################################################

help_message () {
	echo ""
	echo "MAKE SURE YOU HAVE METADATA TEMPLATE FILLED OUT AND SAVED WITH EXPT NAME"
	echo "MAKE SURE YOU ARE IN THE p_dlc ENV (or equivalent)"
	echo ""
	echo ">>>> Usage: pipeline train-setup -e [expt_name] [options]"
	echo "	Options:"
	echo ""
	echo "		-e STR		Enter the experiment name used in the metadata file"
	echo "		-u 		(Default) Use Uniform algo to extract frames"
	echo "		-k		Use kmeans algo to extract frames"
	echo "		-a		Animal name"
	echo "		--wand		Only do wand data (same as --debug)"
	echo '		--behavior	Only do behavior data'
	echo "		--debug		Use debug mode (will only setup train on wand data since there is less content there)"
	echo "		--skipext	Skip frame extraction (when not training)"
	echo "		--label		Only do frame labelling"
	echo "		--skiplabel	Skip labelling (for when you do not need to train a model)"
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
algo="uniform";no_skip=1;data_selection="behavior";label_only=false; skiplabel=false;

while true; do
	case "$1" in
		--label) label_only=true; shift 1;;
		--debug | --wand) data_selection="wand"; shift 1;;
		--behavior) data_selection="behavior"; shift 1;;
		--skipext) no_skip=0; shift 1;;
		--skiplabel) skiplabel=true; shift 1;;
		-e) name="$2"; shift 2;;
		-a) animal="$2"; shift 2;;
		-k) algo="kmeans"; shift 1;;
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

if [ $label_only == false ]
then
	echo "Initializing ${name} mode..."
	python3 ${pyvm}/dlc/initialize.py ${name} ${algo} ${animal} --data ${data_selection} --skip ${no_skip}
fi

if [  $skiplabel == false ]; then
	cd ${pyvm}/dlc
	labels=$(< ${pyvm}/dlc/labels_${data_selection}.py)
	echo "name = '${name}'; ${labels}" | ipython
fi

checkpoints="${data_dir}/${animal}/${name}/checkpoints"
touch "${checkpoints}/dlc-setup"