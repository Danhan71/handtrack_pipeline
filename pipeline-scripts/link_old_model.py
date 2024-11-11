import deeplabcut
from pyvm.utils.experiments import get_params
from pyvm.dlc.utils import find_expt_config_paths
from pyvm.globals import BASEDIR, WINDOWS
import os, os.path
import argparse


from initialize import find_expt_config_paths
from pythonlib.tools.expttools import load_yaml_config


def get_video_list(path_config_file):
    """ return list of videos"""
    # import yaml
    config = load_yaml_config(path_config_file)
    vidlist = list(config["video_sets"].keys())
    return vidlist
def get_train_frac(path_config_file):
    # import yaml
    config = load_yaml_config(path_config_file)
    train_frac = list(config["TrainingFraction"])
    train_frac = int(train_frac[0]*100)
    return train_frac
def get_dlc_date(path_config_file):
	config = load_yaml_config(path_config_file)
	date = config["date"]
	return date
def load_config_file(path_config_file):
    """ return dict"""
    # import yaml
    config = load_yaml_config(path_config_file)
    return config

if __name__=="__main__":

	import shutil

	parser = argparse.ArgumentParser(description="Description of your script.")
	parser.add_argument("name", type=str, help="Expt name")
	parser.add_argument("--cond", type=str, help="List of conditions")
	parser.add_argument("--wandmodeldir", type=str, help="Full directory path to the already trained wand model being used")
	parser.add_argument("--behmodeldir", type=str, help="Full directory path to the already trained beh model being used")
	parser.add_argument("--animal", type=str, help="Animal name")


	args = parser.parse_args()
	name = args.name
	animal = args.animal
	conditionlist = args.cond.split(",") 
	modeldirlist = [args.behmodeldir, args.wandmodeldir]

	print(name,conditionlist)

	for cond in conditionlist:
		dict_paths, base_paths = find_expt_config_paths(exptname=name, condition=cond, animal=animal)
		pcflist = dict_paths.values()
		base_paths = base_paths.values()
		if cond == "behavior":
			modeldir = modeldirlist[0]
		elif cond == "wand":
			modeldir = modeldirlist[1]
		else:
			assert False, "Unrecognized condition. If your condition is beyond the bounds of this script you will have to sym link the model you want to use manually. Good luck."

		for pcf, bp in zip(pcflist, base_paths):
			if os.path.isdir(modeldir):
				train_frac=get_train_frac(pcf)
				date=get_dlc_date(pcf)
				params = get_params(name,animal)
				list_camnames = params["list_camnames"]
				task="combined-" + "_".join(params['list_camnames'])
				if os.path.isdir(f"{bp}/dlc-models"):
					try:
						shutil.rmtree(f"{bp}/dlc-models")
					except OSError as error:
						print(error, "Umnable to delete old model dir")
				os.mkdir(f"{bp}/dlc-models")
				os.mkdir(f"{bp}/dlc-models/iteration-0")
				os.symlink(modeldir, f"{bp}/dlc-models/iteration-0/{task}{date}-trainset{train_frac}shuffle1")
				print(f"Model for {cond} symlink successful!")
				# print(f"{pcf}/dlc-models/iteration-0")
				# print(modeldir, f"  {pcf}/dlc-models/iteration-0/{task}-{train_frac}shuffle1")
			else:
				assert False, f"Model directory for {cond} not found!!"