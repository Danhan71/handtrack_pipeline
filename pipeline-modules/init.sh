#!/usr/bin/env bash
set -e

#######################################################
#
#Initilaizes the basic structure of directory for the experiment at hand
#make sure you fill out the config and metadata file properly.
#
#######################################################

help_message () {
	echo ""
	echo "Usage: pipeline init -e [expt_name]"
	echo "MAKE SURE YOU HAVE METADATA TEMPLATE FILLED OUT AND SAVED WITH EXPT NAME"
	echo ""
	echo "Options:"
	echo ""
	echo "		-e STR		Enter the experiment name used in the metadata file"
	echo "		-a STR		Animal name"
	echo "		-d STR		Directory type ('late' for late_2023... 'early' for gorilla..."
	echo "		-c STR		Enter condition type for this expt (behavior, wand, 3dgrid). Only enter 1!!!"
	echo "		--skiplink	Skip linking data from server"
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

echo "### Checking for environment setup (if you have not setup environemnts, I reccomend using the env-setup module to do it for you). ###"
envs=$(conda env list)

if [ `echo $envs | grep -c "p_drawcamera" ` -gt 0 ]
then
	echo "p_drawcamera conda env found!"
else
	printf "p_drawcamera env not found, ensure you have the env and it is named properly...Quitting" && exit 1
fi 

if [ `echo $envs | grep -c "p_dlc" ` -gt 0 ]
then
	echo "p_dlc conda env found!"
else
	printf "p_dlc env not found, ensure you have the env and it is named properly...Quitting" && exit 1
fi 

echo "################################################################"
echo "################################################################"

#Default param
skiplink=false

while true; do
	case "$1" in
		-e) name="$2"; shift 2;;
		-a) animal="$2"; shift 2;;
		-d) dir_type="$2"; shift 2;;
		-c) cond="$2"; shift 2;;
		--skiplink) skiplink=true; shift 1;;
		-h | --help) help_message; exit 1; shift 1;;
		--) help_message; exit 1; shift; break;;
		*) break;;
	esac		
done

while true; do
    read -p "Is the expt name correct? ${name} (y/n)" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer y or n.";;
    esac
done

if [ -f "${pyvm}/globals.py" ]; then
	rm ${pyvm}/globals.py
fi

if [ ! -d "${data_dir}/${animal}" ]; then
	echo "###### ERROR"
	echo "Base data directory ${data_dir}/${animal} is not found please update the config file with the proper directory"
	exit 1
fi
if [ ! -d "${pyvm}" ]; then
	echo "###### ERROR"
	echo "pyvm directory ${pyvm} is not found. Please update the config file with the proper pyvm directory (i.e. update pipe_path with the current directory of this pipeline)"
	exit 1
fi

echo -e "from pythonlib.globals import MACHINE\nBASEDIR = '${data_dir}'\nPYVM_DIR = '${pyvm}'\nWINDOWS = False\nNCAMS=${NCAMS}\nCB_DIR='${checkb_dir}'" >> ${pyvm}/globals.py

if [ ${skiplink} == false ]; then
	${scripts}/link.sh -e ${name} -a ${animal} -d ${dir_type}
fi

# echo "Please verify BASEDIR in the file that is about to open"
# sleep 3
# nano ${pyvm}/globals.py

# while true; do
#     read -p "BASDIR is set as the directory that your data folders are stored in? (y/n)" yn
#     case $yn in
#         [Yy]* ) break;;
#         [Nn]* ) nano ${pyvm}/globals.py;;
#         * ) echo "Please answer y or n.";;
#     esac
# done
if [ -f "${pipe_path}/${name}.yaml" ]; then
	rm "${pipe_path}/${name}.yaml"
fi

python3 ${scripts}/make_dirs.py ${name} ${animal} --pipepath ${pipe_path} --datadir "${data_dir}/${animal}" --cond ${cond}

echo "Directories successfully generated! You will need to manually move videos from the raw checkerboard folder to the new checkerboard folder (arranging by camera). I was too lazy to automate this. You may then move on to the next step."
