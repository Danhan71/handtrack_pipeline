#!/usr/bin/env bash

#Shell script to link expt videos out of server

help_message () {
	echo ""
	echo "Usage: pipeline init -e [expt_name]"
	echo "MAKE SURE YOU HAVE METADATA TEMPLATE FILLED OUT AND SAVED WITH EXPT NAME"
	echo ""
	echo "Options:"
	echo ""
	echo "		-e STR		Enter the experiment name used in the metadata file"
	echo "		-d STR		Which dir stored in ('late'=camera_late_2023_onwards, 'early='backup/gorilla...'"
	echo "		-a STR		Animal name (Pancho/Diego)"
	echo ""
}


this_dir="$(dirname "$0")"
config_file=${this_dir}/../config
source $config_file
if [[ $? -ne 0 ]]; then 
	echo "no config file."
	exit 1
fi

while true; do
	case "$1" in
		-d) dir_type="$2"; shift 2;;
		-a) ANIMAL="$2"; shift 2;;
		-e) name="$2"; shift 2;;
		-h | --help) help_message; exit 1; shift 1;;
		--) help_message; exit 1; shift; break;;
		*) break;;
	esac		
done


#Pick one
# ANIMAL="Pancho"
# ANIMAL="Diego"
# name="240510_primpancho1d_2"

date=$(echo ${name} | cut -c1-6)

#Pick one
# dir_type="/home/danhan/freiwaldDrive/ltian/camera_late_2023_onwards"
# dir_type="/home/danhan/freiwaldDrive/ltian/backup/gorilla/gorilla2/camera"

base_local="${data_dir}/${ANIMAL}/${name}"

if [ ! -d ${base_local} ]; then
	mkdir "${base_local}"
else
	while true; do
    read -p "Directory for this expt already exists, would you like to replace it? (y/n)" yn
    case $yn in
        [Yy]* ) rm -r "${base_local}"; mkdir "${base_local}";break;;
        [Nn]* ) echo "Ensure directory is properly linked to the data you want, if not delete and run again"; exit 0;;
        * ) echo "Please answer y or n.";;
    esac
done
fi


if [ $dir_type == "late" ]; then
	dir="${server_dir}/ltian/camera_late_2023_onwards"
	for cam_dir in ${dir}/${ANIMAL}/${date}/${name}/*/; do
		echo "Making directory:"
		echo ${cam_dir}
		cam_name=$(echo "${cam_dir}" | cut -d'/' -f10)
		echo "For ${cam_name}"
		ln -s ${cam_dir} ${base_local}/${cam_name}
	done
elif [ $dir_type == "early" ]; then
	dir="${server_dir}/ltian/backup/gorilla/gorilla2/camera"
	for cam_dir in ${dir}/${ANIMAL}/${date}/${name}/*/; do
		echo "Making directory:"
		echo ${cam_dir}
		cam_name=$(echo "${cam_dir}" | cut -d'/' -f13)
		echo "For ${cam_name}"
		ln -s ${cam_dir} ${base_local}/${cam_name} 
	done
fi
