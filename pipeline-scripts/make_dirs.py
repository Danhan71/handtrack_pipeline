import os, os.path
import sys
from pyvm.utils.directories import get_metadata, get_cam_list
from pythonlib.tools.expttools import load_yaml_config
from pyvm.globals import BASEDIR
import argparse
import yaml


def generate_expt_yaml (expt_name, pipe_path, data_dir, condition, animal):

	list_camnames = get_cam_list(expt_name, animal)
	#Brazenly discriminate against bfs2, if there are more than 4 cameras (4 cam setup bfs2 is okay)
	if len(list_camnames) > 4:
		list_camnames = [cam for cam in list_camnames if cam != 'bfs2']

	#Generate condition dependent parameters
	if condition == "behavior":
		this_datadir=f"{data_dir}/{expt_name}/Camera1"
		vidmax = len([name for name in os.listdir(this_datadir) if name.endswith(".mp4")])
		# vidmax = min(20,vidmax)
		list_conditions = ["behavior"]
		list_bodyparts = [["fingertip"]]
		list_skeletons = [[]]
		list_start = [0.15]
		list_stop = [0.85]
		list_combinecams = [True]
		list_vidnums = [[2,vidmax-2]]
		if vidmax > 300:
			list_numframes2pick = [10]
		elif vidmax > 100:
			list_numframes2pick = [20]
		else:
			list_numframes2pick = [100]
		list_dlciternum = [100000]

	elif condition == "wand":
		list_conditions = ["wand"]
		list_bodyparts = [["green", "red"]]
		list_skeletons = [[]]
		list_start = [0.]
		list_stop = [1.]
		list_combinecams = [True]
		print("Default vid numbers for wand is [1,1]. Edit in metadata file for correct vid number if diff (do list with [x,x] where x is vid number with wand)")
		list_vidnums = [[1,1]]
		list_numframes2pick = [100]
		list_dlciternum = [10000]

	elif condition == "checkerboard":
		# this_datadir=f"{data_dir}/{expt_name}/Camera1"
		# vidmax = len([name for name in os.listdir(this_datadir) if name.endswith(".mp4")])
		list_conditions = ["checkerboard"]
		list_bodyparts = ["null"]
		list_skeletons = [[]]
		list_start = [0.]
		list_stop = [1.]
		list_combinecams = [False]
		list_vidnums = [None]
		list_numframes2pick = ["null"]
		list_dlciternum = [None]
	else:
		assert False, f"Condition {condition} is invalid.Please pick a valid condition (behavior, wand, or checkerboard)"


	TEMP = load_yaml_config(f"{pipe_path}/metadata/TEMPLATE.yaml")
	varlist = [animal, list_camnames, list_conditions, list_bodyparts, list_skeletons, list_start, list_stop, list_combinecams, list_vidnums, list_numframes2pick, list_dlciternum]
	str_varlist = ["animal", "list_camnames", "list_conditions", "list_bodyparts", "list_skeletons", "list_start", "list_stop", "list_combinecams", "list_vidnums", "list_numframes2pick", "list_dlciternum"]

	for varname, var in zip(str_varlist, varlist):
		TEMP[varname] = var 
	with open(f"{pipe_path}/metadata/{animal}/{name}.yaml", "w") as f:
		yaml.dump(TEMP,f, default_flow_style=False)

	if condition == "checkerboard":
		print (f"Since you will be calibrating the checkerboard, you must go into the metadata file at {pipe_path}/metadata/{name}.yaml and enter good frames for each camera. Pay attention to the camera order when doing this (can be found in the metadat.yaml file in the expt directory).")
		print ("Enter the information as a list fo lists, with one list of frames for each camera")

# BASEDIR = '/data2/camera'
# BASEDIR = 'Y:\hopfield_data01\ltian\camera\Pancho\220714'
# BASEDIR = 'Y:/hopfield_data01/ltian/camera/Pancho'

def check_vid_nums(data_dir,animal,name):
	'''
	Function to check num vids for each camera, any cam with less than max vids will have last vid
	deleted, to avoid issues with vid corruption. Rest of pipeline can handled mismatched lengths methinks.
	'''
	list_camnames = get_cam_list(name, animal)
	#Brazenly discriminate against bfs2, if there are more than 4 cameras (4 cam setup bfs2 is okay)
	if len(list_camnames) > 4:
		list_camnames = [cam for cam in list_camnames if cam != 'bfs2']
	vid_dir = f'{data_dir}/{name}/behavior'
	dir_lengths = {}
	for cam in list_camnames:
		this_dir = f'{vid_dir}/{cam}'
		vids = [f for f in os.listdir(this_dir) if f.endswith('.mp4')]
		dir_lengths[cam] = len(vids)
	vids_max = max(dir_lengths.values())
	cut_dirs = [k for k,v in dir_lengths.items() if v < vids_max]
	for cam in list_camnames:
		this_dir = f'{vid_dir}/{cam}'
		if cam in cut_dirs:
			vids = [f for f in os.listdir(this_dir) if f.endswith('.mp4')]
			trial_cutoff = max([vid.split('vid-t')[1].split('.mp4')[0] for vid in vids])
			os.remove(f'{this_dir}/vid-t{trial_cutoff}.mp4')





if __name__ == "__main__":

	parser = argparse.ArgumentParser(description = "Description of your script.")
	parser.add_argument("name", type = str, help = "Experiment name/date")
	parser.add_argument("animal", type=str, help="Alright then")
	parser.add_argument("--pipepath", type = str, help = "Path to pipeline dir")
	parser.add_argument("--cond", type = str, help = "Conditon for this expt")
	parser.add_argument("--datadir", type = str, help = "Directory for data")
	parser.add_argument("--skiplink", type = str, help = "skiplinking vids", default=False)
	args = parser.parse_args()

	name  =  args.name
	pipe_path = args.pipepath
	data_dir = args.datadir
	condition = args.cond
	animal = args.animal
	skiplink = args.skiplink
	if skiplink == 'true':
		skiplink = True
	else:
		skiplink = False
	# condition = args.cond
	# name = "240510_2"
	# pipe_path = "/home/danhan/Documents/pipeline"
	# data_dir = f"{BASEDIR}/{name}/Camera1"

	
	generate_expt_yaml(expt_name=name, pipe_path=pipe_path, data_dir=data_dir, condition=condition, animal=animal)
	METADAT = get_metadata(name, animal=animal, condition=condition)


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
		cam_dirs = metadat["conditions_dict"][condition]["map_camname_to_path"]

		config = load_yaml_config(f"{pipe_path}/metadata/{animal}/{name}.yaml")
		vid_inds = config["list_vidnums"][0]

		if not skiplink:
			for cam_name, cam_path in zip(cam_dirs.keys(), cam_dirs.values()):
				file_list = [file for file in os.listdir(cam_path) if not file.endswith(".mp4")]
				vid_list = [vid for vid in os.listdir(cam_path) if vid.endswith(".mp4")]
				for file in file_list:
					file_dir = os.path.join(cam_path, file)
					sdir = f"{data_dir}/{name}/{condition}/{cam_name}/{file}"
					if os.path.exists(sdir):
						print(f"File {sdir} exists, deleting :)")
						os.remove(sdir)
					os.symlink(file_dir, sdir)
				for vid in vid_list:
					vid_dir = os.path.join(cam_path, vid)
					sdir = f"{data_dir}/{name}/{condition}/{cam_name}/{vid}"
					if os.path.exists(sdir):
						print(f"Vid {sdir} exists, deleting >:(")
						os.remove(sdir)
					os.symlink(vid_dir, sdir)

	if condition == 'behavior':
		check_vid_nums(data_dir,animal,name)



		# Print summary of vidoes
		# print_summary_videos(METADAT)







