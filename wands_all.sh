#!/usr/bin/env bash
set -e
dates_list=("221015_dircolor2_wandgrid" "221015_dirshapecolor1b_wandgrid" "221015_dircolor3b_wandgrid")

this_dir="$(dirname "$0")"
config_file=${this_dir}/config
source $config_file
if [[ $? -ne 0 ]]; then 
	echo "no config file."
	exit 1
fi


for date in "${dates_list[@]}"; do
	echo ${date}
	# yes | ./pipeline dlc-setup -e ${date} --wand --skipext --skiplabel &&
	# yes | ./pipeline analyze -e ${date} --cond wand &&
	yes | ./pipeline wand -e ${date} -a Pancho --step 0 &&
	yes | ./pipeline wand -e ${date} -a Pancho --step 1
done

# python3 ${pipe_path}/pipeline-scripts/align_wand.py
