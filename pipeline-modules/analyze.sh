#!/usr/bin/env bash
set -e

#######################################################
#
#This module is to analyze videos ONLY when you do not want to train a new model. 
#Can also be done if you want to re-analyze a day with trained model
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
	echo "		-e		STR	Enter the experiment name used in the metadata file"
	echo "		--cond		STR	Enter a list of conditions you want the model to analyze in a comma separated list (no spaces)"
	echo "		-a		STR	Amnimal name"
	echo "						e.g. wand,behavior"
	echo "		--skiplink 	STR	Skip linking model"
	echo "		--framerev		Extract and review labelled frames."
	echo "		--ext			Only extract labelled frames for review"
	echo "		--check			Only frame checking"
	echo ""
	echo "		Additional Options:"
	echo "		--nv	INT	Number of vids to extract frames from (df 20)"
	echo "		--nf	INT	Number of frames to extract in total (from allv ids)"
	echo "If your naming conventions hav changed since the writing of this you will need to sym link the model you are using manually"
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

skiplink=false; skipdlc=false; skipreview=true; nv=20; nf=200;

while true; do
	case "$1" in
		--cond) cond="$2"; shift 2;;
		--skiplink) skiplink=true; shift 1;;
		--framerev) mode='both'; skipdlc=true; skipreview=false; shift 1;;
		--ext) mode='extract'; skipdlc=true; skipreview=false; shift 1;;
		--check) mode='review'; skipdlc=true; skipreview=false; shift 1;;
		--nv) nv=$2; shift 2;;
		--nf) nf=$2; shift 2;;
		-a) animal="$2"; shift 2;;
		-e) name="$2"; shift 2;;
		-h | --help) help_message; exit 1; shift 1;;
		--) help_message; exit 1; shift; break;;
		*) break;;
	esac		
done

if [ $skipdlc == false ]; then

	if [ $skiplink == false ]; then
		python3 ${scripts}/link_old_model.py ${name} --cond ${cond} --behmodeldir ${behavior_model_dir} --wandmodeldir ${wand_model_dir} --animal ${animal}
	fi

	python ${pyvm}/dlc/run.py ${name} --step 5 --cond ${cond} --animal ${animal}
fi

if [ $skipreview == false ]; then
python ${pyvm}/dlc/metrics.py ${name} ${animal} --numvids ${nv} --numframes ${nf} --do ${mode}
fi

# STARTED AT 1824



