#!/usr/bin/env bash
set -e

#######################################################
#
#Does all the final analysis and plotting of the data, also the module for doing
#wand calibration if necessary (steps 0-3 do this).
#
#######################################################

help_message () {
	echo ""
	echo "MAKE SURE YOU HAVE METADATA TEMPLATE FILLED OUT AND SAVED WITH EXPT NAME"
	echo "MAKE SURE YOU ARE IN THE p_dlc ENV (or equivalent)"
	echo ""
	echo ">>>> Usage: pipeline train-setup -e [expt_name]"
	echo "	Options:"
	echo ""
	echo "		-e STR			Enter the experiment name used in the metadata file"
	echo "		--setup			Set up the programs to run this module (only necessary if you have not done so already)"
	echo "		-a				Animal (Pancho, Diego)"
	echo ""
	echo "		--step		Step (see options below, run all steps in succession)"
	echo ""
	echo "	#### SKIP STEPS 0-3 IF YOU ARE USING PRECALCULATED WAND COEFSS"
	echo "		cb----------Extract and save the cb calibration params to use with easywand (run checkerb module first)"
	echo "		0-----------Extract wand frames good accross all cameras"
	echo "		1-----------The Lord, God, Watches Us All"
	echo "		2-----------Open DLTdv8 to extract axis pts from 3d grid photos. Refer to documentation for labelling instructions"
	echo "		MAKE SURE TO DELETE THE NaN ROWS IN CSV data_xypts.csv FILE AFTER DONE"
	echo "		3-----------Runs the easyWand calibration step, follow the instructions closely."
	echo "	<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"
	echo "		4-----------Once you are happy with stuff from step 3 (may need 2-3 runs if you aren't careful the first time)"
	echo "		 -----------This step will finalize the data and extract it."
	echo "		5-----------This step does the final analysis of beh and campy data, also generate plots."
	echo "			Extra Options for step 4 and 5:"
	echo "			(4) --cond	STR	Condition type (default = behavior)"
	echo "			(5) --noreg	Do not do linear regression on cam pts (one regression for whole day applied to each trial) "
	echo "			Default:	True, do regression (also outputs non-regressed data)"
	echo "			(5) --ploton	Generate plots (default is false)"
	echo "			(5) --force_redo_regression	Force redo of regression even if it exists"
	echo ""
	echo "			ALSO make sure to update config with current dlt_coeffs"
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
setup=false; cond='behavior'; reg=1;

while true; do
	case "$1" in
		--setup) setup=true; shift 1; break;;
		--step) step=$2; shift 2;;
		--cond) cond="$2"; shift 2;;
		# --rmcam) rmcam="$2"; shift 2;;
		--noreg) reg=0; shift 1;;
		--ploton) plot_flag='--ploton'; shift 1;;
		--force_redo_regression) force_reg_flag='--force_redo_regression'; shift 1;;
		-e) name="$2"; shift 2;;
		-a) animal="$2"; shift 2;;
		-h | --help) help_message; exit 1; shift 1;;
		--) help_message; exit 1; shift; break;;
		*) break;;
	esac		
done

if [ $setup == true ]; then
	${pipe_path}/setup/DLTdv8a_v8p2p9_linux.install
	exit 0
fi

while true; do
    read -p "Is the expt name correct ${name} (y/n)" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer y or n.";;
    esac
done
if [ "$step" == "cb" ]; then
	if [ ! -d "${checkb_dir}" ]; then
		echo "###### ERROR"
		echo "Checkerboard dir ${checkb_dir} is not found. Please update the config file with the proper pyvm directory (i.e. update pipe_path with the current directory of this pipeline)"
		exit 1
	else
		python3 ${pyvm}/run/wand_pts_extraction.py ${name} ${animal} --cb ${checkb_dir}
	fi
elif [ $step == 0 ]; then
	python3 ${pyvm}/run/wand_pts_extraction.py ${name} ${animal}
	checkpoints="${data_dir}/${animal}/${name}/checkpoints"
	touch "${checkpoints}/wand0"
elif [ $step == 1 ]; then
	echo "2 Corinthians 4:16-18 ~ So we do not lose heart. Though our outer self is wasting away, our inner self is being renewed day by day. For this light momentary affliction is preparing for us an eternal weight of glory beyond all comparison, as we look not to the things that are seen but to the things that are unseen. For the things that are seen are transient, but the things that are unseen are eternal."
elif [ $step == 2 ]; then
	if [ ! -f "${dlt_dir}" -o ! -d "${matrun_dir}" ]; then
    	echo "Error: DLT or MATLAB run time not found. Ensure that you have run the DLT install script, or install it via your preferred method. You must also update the config file with the proper dirs for MATLAB runtime and DLT."
    	exit 1
	fi
	${dlt_dir} ${matrun_dir}
	checkpoints="${data_dir}/${animal}/${name}/checkpoints"
	touch "${checkpoints}/wand2"
elif [ $step == 3 ]; then
	if [ ! -f "${matfull_dir}/bin/matlab" ]; then
		echo "MATLAB install not found make sure you have MATLAB installed and the config file has the correct directory"
		exit 1
	fi
	echo ""
	echo "######################## USAGE INSTRUCTIONS ##############################"
	echo "STEP 1: Load wand points file in /wand/wand_calibration/wandPoints.csv"
	echo "STEP 2: Load camera profiles in /wand/wand_calibration/camera_profiles_checkerboard.txt"
	echo "STEP 3: Set the following params"
	echo "		Optim. Mode: No camera intrinsic"
	echo "		Wand Span (m): 0.0645"
	echo "		Distortion Coeffs: No distortion (we already calibrated this)"
	echo "		2ndary camera mode: Default (bundle adjust.)"
	echo "STEP 4: Load axis points saved from last step (wherever you saved them)"
	echo "		Should be at /savedir/DLTdv8_data_xypts.csv or whatever you saved it as."
	echo "		Optionally: Load background points"
	echo "STEP 5: Click compute calibration"
	echo "	yes, yes, yes, okay, yes, okay, yes, etc."
	echo "STEP 6: Prune outliers. If you have really bad outliers first make sure there are no upstream errors."
	echo "			if there are not upstream errors then remove what you feel is right. Wand sd should be 0.002 +/- some"
	echo "STEP 7: Save results, type results into box so final name is results_dltCoefs"
	echo "######################## DONE ################################"
	echo "	CLOSE MATLAB TO PROCEED TO NEXT STEP"
	echo ""
	echo ""

	${matfull_dir}/bin/matlab -sd "${pipe_path}/setup/easyWand/easyWand5" -r easyWand5

	

	echo ""
	echo ""
	echo "######################### USAGE INSTRUCTIONS #################################"
	echo "Load project from before and add the coeffs from the previous step by clicking 3D > load new DLT coefs."
	echo "Do some tests by clicking on a point and checking if the blue line/green diamond intersects that point on other camera."
	echo "****2-4 pixel DLT error is the goal****"
	echo "################################ DONE ########################################"
	echo ""

	${dlt_dir} ${matrun_dir}

	checkpoints="${data_dir}/${animal}/${name}/checkpoints"
	touch "${checkpoints}/wand3"
elif [ $step == 4 ]; then
	#  python3 ${pyvm}/run/dlc_xyz_extraction.py ${name} ${animal} --cond ${cond} --pipe ${pipe_path} --coeff ${calib_prefix} --step 1
	# Uncomment the following line and remove the previous line if you want to use auto coeff names
	python3 ${pyvm}/run/dlc_xyz_extraction.py ${name} ${animal} --cond ${cond} --pipe ${pipe_path} --step 1
	dir="${pipe_path}/temp_matlab_files/${animal}/${name}"
	for coeff_dir in "${dir}"/*; do
		if [ -d "${coeff_dir}" ]; then
			matlab -nodisplay -r "cd('${pipe_path}/setup/easyWand/easyWand5');reconstruct_middleman('${coeff_dir}');exit"
		fi
	done
	#  python3 ${pyvm}/run/dlc_xyz_extraction.py ${name} ${animal} --cond ${cond} --pipe ${pipe_path} --step 2 --coeff ${calib_prefix}
	# Uncomment the following line and remove the previous line if you want to use auto coeff names
	python3 ${pyvm}/run/dlc_xyz_extraction.py ${name} ${animal} --cond ${cond} --pipe ${pipe_path} --step 2
	
	checkpoints="${data_dir}/${animal}/${name}/checkpoints"
		touch "${checkpoints}/wand4"
	#Remove regression files if they exist
	trans_path="${data_dir}/${animal}/${name}/transforms" 
	if [ -d $trans_path ]; then
		echo "Removing old regression and transformation matrices"
		rm -r $trans_path
	fi
elif [ $step == 5 ]; then
	python3 ${draw_monk}/final_analysis.py ${name} --animal ${animal} --reg ${reg} --pyvm ${pyvm} --data ${data_dir} $plot_flag $force_reg_flag
	checkpoints="${data_dir}/${animal}/${name}/checkpoints"
	touch "${checkpoints}/wand5"
	touch "${checkpoints}/done"
fi





	#Graveyard
	# matlab -nodisplay -r "cd('${pipe_path}/setup/easyWand/easyWand5');triangulate_middle_man('${pipe_path}/wand_data/${calib_dir}','${pipe_path}/temp_matlab_files/${animal}/${name}');exit"

