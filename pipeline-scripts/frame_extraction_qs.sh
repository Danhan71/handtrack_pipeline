#!/bin/bash

this_dir="$(dirname "$0")"
config_file=${this_dir}/../config
source $config_file
if [[ $? -ne 0 ]]; then 
	echo "no config file."
	exit 1
fi

name="$1"
animal="$2"
camera="$3"
dir="${data_dir}/${animal}/${name}/behavior/${camera}"
trials=(10 11 12 13 14 16 17 18 19 20)
for trial in "${trials[@]}"; do
	mkdir "${dir}/t${trial}_frames"
	ffmpeg -i ${dir}/vid-t${trial}.mp4 -c:v png ${dir}/t${trial}_frames/frame-%04d.png
done
