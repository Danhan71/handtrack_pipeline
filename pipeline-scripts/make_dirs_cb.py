import os, os.path
import sys
from pyvm.utils.directories import get_metadata
from pythonlib.tools.expttools import load_yaml_config
from pyvm.globals import BASEDIR
import argparse
import yaml


def generate_expt_yaml (expt_name, pipe_path, data_dir, condition):

	#Generate condition dependent parameters
	if condition == "behavior":
		this_datadir=f"{data_dir}/{expt_name}/Camera1"
		vidmax = len([name for name in os.listdir(this_datadir) if name.endswith(".mp4")])
		list_conditions = ["behavior"]
		list_bodyparts = [["fingertip"]]
		list_skeletons = [[]]
		list_start = [0.15]
		list_stop = [0.85]
		list_combinecams = [True]
		list_vidnums = [[0,vidmax-2]]
		if vidmax > 300:
			list_numframes2pick = [10]
		elif vidmax > 100:
			list_numframes2pick = [20]
		else:
			list_numframes2pick = [100]
		list_dlciternum = [100000]

	elif condition == "wand":
		this_datadir=f"{data_dir}/{expt_name}/Camera1"
		vidmax = len([name for name in os.listdir(this_datadir) if name.endswith(".mp4")])
		list_conditions = ["wand"]
		list_bodyparts = ["green", "red"]
		list_skeletons = []
		list_start = [0.]
		list_stop = [1.]
		list_combinecams = [True]
		list_vidnums = ["null"]
		list_numframes2pick = [100]
		list_dlciternum = [100000]

	elif condition == "checkerboard":
		this_datadir=f"{data_dir}/{expt_name}/Camera1"
		vidmax = len([name for name in os.listdir(this_datadir) if name.endswith(".mp4")])
		list_conditions = ["checkerboard"]
		list_bodyparts = ["green", "red"]
		list_skeletons = []
		list_start = [0.]
		list_stop = [1.]
		list_combinecams = [False]
		list_vidnums = ["null"]
		list_numframes2pick = ["null"]
		list_dlciternum = [None]
	else:
		assert False, f"Condition {condition} is invalid.Please pick a valid condition (behavior, wand, or checkerboard)"


	TEMP = load_yaml_config(f"{pipe_path}/metadata/TEMPLATE.yaml")
	varlist = [list_conditions, list_bodyparts, list_skeletons, list_start, list_stop, list_combinecams, list_vidnums, list_numframes2pick, list_dlciternum]
	str_varlist = ["list_conditions", "list_bodyparts", "list_skeletons", "list_start", "list_stop", "list_combinecams", "list_vidnums", "list_numframes2pick", "list_dlciternum"]

	for varname, var in zip(str_varlist, varlist):
		TEMP[varname] = var 
	with open(f"{pipe_path}/metadata/{name}.yaml", "w") as f:
		yaml.dump(TEMP,f, default_flow_style=False)

	if condition == "checkerboard":
		print (f"Since you will be calibrating the checkerboard, you must go into the metadata file at {pipe_path}/metadata/{name}.yaml and enter good frames for each camera. Pay attention to the camera order when doing this (can be found in the metadat.yaml file in the expt directory).")
		print ("Enter the information as a list fo lists, with one list of frames for each camera")

# BASEDIR = '/data2/camera'
# BASEDIR = 'Y:\hopfield_data01\ltian\camera\Pancho\220714'
# BASEDIR = 'Y:/hopfield_data01/ltian/camera/Pancho'
if __name__ == "__main__":

	parser = argparse.ArgumentParser(description = "Description of your script.")
	parser.add_argument("name", type = str, help = "Experiment name/date")
	parser.add_argument("--pipepath", type = str, help = "Path to pipeline dir")
	parser.add_argument("--cond", type = str, help = "Conditon for this expt")
	parser.add_argument("--datadir", type = str, help = "Directory for data")
	args = parser.parse_args()

	name  =  args.name
	pipe_path = args.pipepath
	data_dir = args.datadir
	condition = args.cond
	# condition = args.cond
	# name = "240510_2"
	# pipe_path = "/home/danhan/Documents/pipeline"
	# data_dir = f"{BASEDIR}/{name}/Camera1"

	generate_expt_yaml(expt_name=name, pipe_path=pipe_path, data_dir=data_dir, condition=condition)
	METADAT = get_metadata(name, condition=condition)

	# basedir = f"{BASEDIR}/{DATE}"

	# list_camnames = ["bfs", "flea", "ffly"]
	# list_camnames = ["bfs", "flea", "ffly"]

	# Get metadata about this data
	# camera_mapping = METADAT["camera_mapping"]
	# camera_mapping = {
	# 	1: "flea",
	# 	2: "ffly", # Blackfly S BFS-U3-63S4C
	# 	3: "bfs1", # Blackfly S BFS-U3-16S2C
	# 	4: "bfs2"
	# 	}

	map_condition_cam_to_dir = METADAT["map_condition_cam_to_dir"]


	if condition != "checkerboard":

		for condcam, path in map_condition_cam_to_dir.items():
			print("made dir: ", path, "for cond/cam: ", condcam)
			os.makedirs(f"{path}/", exist_ok = True)
			print("[NOTE!] now you have to move the correct videos into their directories")
		# Now move all videos
		metadat = load_yaml_config(f"{data_dir}/{name}/metadat.yaml")
		list_camnames = metadat["list_camnames"]

		config = load_yaml_config(f"{pipe_path}/metadata/{name}.yaml")
		vid_inds = config["list_vidnums"][0]

		cam_ind = 1
		for cam in list_camnames:
			file_list = [file for file in os.listdir(f"{data_dir}/{name}/Camera{cam_ind}") if not file.endswith(".mp4")]
			vid_list = [vid for vid in os.listdir(f"{data_dir}/{name}/Camera{cam_ind}") if vid.endswith(".mp4")]
			vid_ind = 0
			for file, vid in zip(file_list, vid_list):
				file_dir = os.path.join(f"{data_dir}/{name}/Camera{cam_ind}", file)
				vid_dir = os.path.join(f"{data_dir}/{name}/Camera{cam_ind}", vid)
				os.symlink(file_dir, f"{data_dir}/{name}/{condition}/{cam}/{file}")
				if vid_ind <= vid_inds[1]:
					os.symlink(vid_dir, f"{data_dir}/{name}/{condition}/{cam}/{vid}")
				vid_ind = vid_ind + 1
			cam_ind = cam_ind + 1


		# Print summary of vidoes
		# print_summary_videos(METADAT)







