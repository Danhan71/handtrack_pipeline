#!/usr/bin/env bash
set -e

#######################################################
#
# Calculates and extracts lag times between touch screen data and cam data.
# Also generates pltos for reviewing good trials to use lags for and final
# summary plot of lag time distribution.
#
#######################################################

help_message () {
	echo ""
	echo "MAKE SURE YOU HAVE METADATA TEMPLATE FILLED OUT AND SAVED WITH EXPT NAME"
	echo "MAKE SURE YOU ARE IN THE p_dlc ENV (or equivalent)"
	echo ""
	echo ">>>> Usage: pipeline lags -e [expt_name] -a [animal] --step [1 or 2] [options]"
	echo "	Options:"
	echo "      Rq'd:"
	echo "		-e STR			Enter the experiment name used in the metadata file (date_exptname)"
    echo "      -a STR          Animal name"
    echo "      --step INT      Step 1 is calc lags and plot. Step 2 is plot final unfilt/filt lags. More detials below.**"
    echo "      -o STR          Output dir for processes data and figs, subdirs auto-generated from this"
    echo "      Opt:"
    echo "      --ts            Trial num to start data extraction from (df: 10)"
    echo "      --te            Trial num to end data extraction from (df: 110)" 
    echo "      ** Step one requires no user action, will plot the best correlation match. User should then review the figures and pick ones that look"
    echo "          good (i.e. not straight lines, and not absurdly bad fits. Then enter these in a comma separated list in the 'good_inds.txt' file)" 
    echo "          formate should be 'trial-stroke','trial-stroke'... (matching name formatting of lag plot names). Then run step two to generate plots for filt and unfilt data."         
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
ts=10; te=110;

while true; do
	case "$1" in
		--ts) ts=$2; shift 2;;
		--te) te=$2; shift 2;;
        --step) step=$2; shift 2;;
		-e) name="$2"; shift 2;;
		-a) animal="$2"; shift 2;;
		-h | --help) help_message; exit 1; shift 1;;
		--) help_message; exit 1; shift; break;;
		*) break;;
	esac		
done


if [[ $step -eq 1 ]]; then
    python ${scripts}/alignment_check.py ${name} --animal ${animal} --tstart ${ts} --tend ${te}
    echo "Initial plotting complete, please review plots and enter good trial-strokes into the file good_inds.txt as a comma seprated list."
elif [[ $step -eq 2 ]]; then
    python ${scripts}/alignment_check.py ${name} --animal ${animal} --tstart ${ts} --tend ${te} --plot
    echo "All done, summary plot saved."
    echo "Proverbs 6:6-8 ~ Go to the ant, O sluggard; consider her ways, and be wise. Without having any chief, officer, or ruler, she prepares her bread in summer and gathers her food in harvest."
fi