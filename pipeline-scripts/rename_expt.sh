#!/bin/bash
set -e

# Quick script for renaming experiment

help_message () {
	echo ""
	echo "Usage: pipeline rename [animal] [old_name] [new_name]"
    echo "Enter full experiment name (date_expt) for both old_name and new_name"
	echo ""
}

this_dir="$(dirname "$0")"
config_file=${this_dir}/../config
source $config_file
if [[ $? -ne 0 ]]; then 
	echo "no config file."
	exit 1
fi

a="$1"
old="$2"
new="$3"

echo "Changing expt ${old} to ${new}"

if [ -d "${data_dir}/${a}/${old}" ]; then
    mv "${data_dir}/${a}/${old}" "${data_dir}/${a}/${new}"
else
    while true; do
    read -p "Cam data dir for ${old} not found, would you like to still rename any existing metadata files? (y/n)" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer y or n.";;
    esac
    done
fi

if [ -f "${data_dir}/${a}/${new}/metadat.yaml" ]; then
    sed -i -e "s/${old}/${new}/g" "${data_dir}/${a}/${new}/metadat.yaml"
else
    echo "metadat file for ${old} not found, skipping..."
fi


if [ -f "${pipe_path}/metadata/${a}/${old}.yaml" ]; then
    mv "${pipe_path}/metadata/${a}/${old}.yaml" "${pipe_path}/metadata/${a}/${new}.yaml"
else
    echo "DLC metadata not found, skipping..."
fi

echo "Expt ${old} successfully renamed to ${new}"

