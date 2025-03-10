#!/usr/bin/env bash
set -e

##############################################################################################################################################################
#
#Script to set up the envs needed for this module
#Incorporates some random fixes that works along the way. If these envs dont work you will
#have to figure that out yourself
# 
##############################################################################################################################################################

help_message () {
	echo ""
	echo "At the time of writing this, these worked. If they dont work for you, you will need to figure that out yourself."
	echo "A few python libraries will ned to be installed manually, but this handles the bulk, the others ones will be obvious by the errors throw (and a quick pip)."
	echo ""
	echo ">>>> Usage: pipeline env-setup [options]"
	echo "	Options:"
	echo "		--dlc		Only make dlc_env"
	echo "		--draw		Only draw env"
	echo "		--both		Both envs (Default)"
	echo ""
}

#Default params
dlc=true;draw=true;

echo $1
while true; do
	case "$1" in
		--dlc) draw=false; shift 1;;
		--draw) dlc=false; shift 1;;
		-h | --help) help_message; exit 1; shift 1;;
		--) help_message; exit 1; shift; break;;
		*) break;;
	esac		
done
echo $draw

this_dir="$(dirname "$0")"
config_file=${this_dir}/../config
source $config_file
if [[ $? -ne 0 ]]; then 
	echo "no config file."
	exit 1
fi

#Check if envs already exist
if [ `echo $envs | grep -c "p_drawcamera" ` -gt 0 -a $draw == true ]
then
	echo "p_drawcamera conda env found!" && exit 1
fi 

if [ `echo $envs | grep -c "p_dlc" ` -gt 0 -a $dlc == true ]
then
	echo "p_dlc conda env found, cancelling! If you just need 1 env sorry, no edge cases :)" && exit 1
fi

if [ $draw == true ]; then
	conda env create -n p_drawcamera --file=${pipe_path}/setup/envs/drawcamera_env.yaml
	source activate p_drawcamera
	conda install jupyterlab ipykernel
	pip install -e ${pyvm}
	pip install -e ${pythonlib}
fi

if [ $dlc == true ]; then
	conda env create -n p_dlc --file=${pipe_path}/setup/envs/dlc_env.yaml
	source activate p_dlc
	conda install jupyterlab ipykernel
	pip install -e ${pythonlib}
	pip install -e ${pyvm}
	pip install -e ${pipe_path}/setup/deeplabcut
	pip install torch
	pip uninstall opencv-python
	pip install opencv-python-headless
fi

ln -s ${pyvm}/pyvm/metadata ${pipe_path}/metadata
