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
	echo "		--loop			Loop through a date folder will skip wandgrids"
	echo "						If this is selected enter the date for the name"
	echo "						loop mode also tracks checkpoints"
	echo ""
}

comm () { ${scripts}/print_comment.py "$1" "-"; }
error () { ${scripts}/print_comment.py "$1" "*"; exit 1; }
warning () { ${scripts}/print_comment.py "$1" "*"; }
announcement () { ${scripts}/print_comment.py "$1" "#"; }

this_dir="$(dirname "$0")"
config_file=${this_dir}/config
source $config_file
if [[ $? -ne 0 ]]; then 
	echo "no config file."
	exit 1
fi

loop=false;
while true; do
	case "$1" in
		-w) cond='wand'; shift 1;;
		-b) cond='behavior'; shift 1;;
		-e) name="$2"; shift 2;;
		-3) cond='3dgrid'; shift 1;;
		-a) animal=$2; shift 2;;
		-h | --help) help_message; exit 1; shift 1;;
		-l) dir="late"; loop_dir="${server_dir}/ltian/camera_late_2023_onwards/${animal}/${name}"; shift 1;;
		-g) dir="early"; loop_dir="${server_dir}/ltian/backup/gorilla/gorilla2/camera/${animal}/${name}"; shift 1;;
		--loop) loop=true; shift 1;;
		--) help_message; exit 1; shift; break;;
		*) break;;
	esac
done

# while true; do
# 	read -p "Is the expt name/date correct ${name} (y/n)" yn
#     case $yn in
#         [Yy]* ) break;;
#         [Nn]* ) echo "Do not use loop conition or else move this folder"; exit;;
#         * ) echo "Please answer y or n.";;
#     esac
# done




if [ ${loop} = true ]; then
	
	for i in ${loop_dir}/*/; do
		#Will only access dirs that are not wandgrid or test, if want to do that do it indivudally 
		if [[ "${i}" != *wand* && "${i}" != *test* ]]; then
			# Find dir depth to properly caluclate where the expt name is
			dep_count=$(grep -o "/" <<< "$loop_dir" | wc -l)
			((dep_count+=2))

			expt=$(echo "$i" | cut -d'/' -"f${dep_count}")
			checkpoints="${data_dir}/${animal}/${expt}/checkpoints"

			# # Temp hack bc some days had specific trials not ds/label
			# if [ ! -f "${checkpoints}/done" ] && [ -f "${checkpoints}/analyze" ]; then
			# 	rm -r "${checkpoints}"
			# fi


			echo "Doing ${expt}"

			if [ ! -f "${checkpoints}/init" ]; then
				yes | pipeline init -e ${expt} -a ${animal} -c ${cond} -d ${dir}
			elif  [ ! -f "${expt_files}/${animal}/${expt}" ]; then
				yes | pipeline init -e ${expt} -a ${animal} -c ${cond} -d ${dir} --skiplink
			else
				echo "Skipping pipeline init, checkpoint found"
			fi

			# mkdir -p $checkpoints
			# touch "${checkpoints}/init"
			

			if [ ! -f "${checkpoints}/dlc-setup" ]; then
				yes | pipeline dlc-setup -e ${expt} --${cond} --skiplabel --skipext -a ${animal}
			else
				echo "Skipping pipeline dlc-setup, checkpoint found"
			fi

			if [ ! -f "${checkpoints}/analyze" ]; then
				yes | pipeline analyze -e ${expt} --cond ${cond} -a ${animal}
			else
				echo "Skipping pipeline analyze, checkpoint found"
			fi

			if [ ! -f "${checkpoints}/wand4" ]; then
				yes | pipeline wand -e ${expt} --step 4 --cond ${cond} -a ${animal}
			else
				echo "Skipping pipeline wand step 4, checkpoint exists"
			fi

			if [ ! -f "${checkpoints}/wand5" ]; then
				yes | pipeline wand -e ${expt} --step 5 -a ${animal}
			else
				echo "Day is fully run, skipping. If you want to rerun the day from certain step\
				 delete files and checkpoints"
			fi
		fi

	done
	echo "For our struggle is not against flesh and blood, but against the rulers,\
	against the authorities, against the powers of this dark world\
	  and against the spiritual forces of evil in the heavenly realms. 
	  — Ephesians 6:12 (NIV)"

else
	checkpoints="${data_dir}/${animal}/${name}/checkpoints"
	mkdir -p "${checkpoints}"

	# yes | pipeline init -e ${name} -a ${animal} -c ${cond} -d ${dir}

	yes | pipeline dlc-setup -e ${name} --${cond} --skiplabel --skipext -a ${animal}

	yes | pipeline analyze -e ${name} --cond ${cond} -a ${animal}

	# yes | pipeline wand -e ${name} -a ${animal} --step 4 --cond ${cond}

	# yes | pipeline wand -e ${name} --step 5 --reg -a ${animal}

fi
