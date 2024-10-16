import os
import random
from pyvm.utils.directories import get_metadata

data_dir = "/data3/hand_track/Pancho"

cam_dirs = []
vids_added = {
	"fly1" : 0,
	"fly2" : 0,
	"bfs1" : 0,
	"bfs2" : 0,
	"flea" : 0
	}

data_dirs = [d for d in os.listdir(data_dir) if d[:6].isdigit()]
for d in data_dirs:
	cur = os.path.join(data_dir,d)
	# cams = [di for di in os.listdir(cur) if "Camera" in di]
	print(cur)
	meta = get_metadata(d)
	list_cammap = meta["conditions_dict"]["behavior"]["map_camname_to_path"]
	# for c in cams:
	# 	cam_dirs.append(os.path.join(cur,c))

	dir_size = int((len(os.listdir(list_cammap["flea"]))-1)/3)
	for name in list_cammap.keys():
		vid_dirs = []
		csv_dirs = []
		npy_dirs = []
		cam_dir = list_cammap[name]
		vid_list = [vid for vid in os.listdir(cam_dir) if vid.endswith(".mp4")]
		csv_list = [csv for csv in os.listdir(cam_dir) if csv.endswith(".csv")]
		npy_list = [npy for npy in os.listdir(cam_dir) if npy.endswith(".npy")]
		dir_size = int((len(os.listdir(cam_dir))-1)/3)

		for vid, csv, npy in zip(vid_list, csv_list, npy_list):
			vid_dirs.append(os.path.join(cam_dir, vid))
			csv_dirs.append(os.path.join(cam_dir, csv))
			npy_dirs.append(os.path.join(cam_dir, npy))

		dir_tuple = list(zip(vid_dirs, csv_dirs, npy_dirs))
		move_list = random.sample(dir_tuple, 5)

		for files in move_list:
			vid, csv, npy = files
			index = vids_added[f"{name}"]
			print(name)
			os.symlink(vid, f'/data3/hand_track/Pancho/meow_240725_trainset/{name}/vid-t{index}.mp4')
			os.symlink(csv, f"/data3/hand_track/Pancho/meow_240725_trainset/{name}/metadata-t{index}.csv")
			os.symlink(npy, f"/data3/hand_track/Pancho/meow_240725_trainset/{name}/frametimes-t{index}.npy")
			vids_added[f"{name}"] += 1

data_dir = "/data3/hand_track/Pancho"

cam_dirs = []
vids_added = {
	"Camera1" : 0,
	"Camera2" : 35,
	"Camera3" : 0,
	"Camera4" : 0,
	"Camera5" : 0
	}

cam_dir = "/data3/hand_track/Pancho/meow_240509_primpancho1c/Camera2"

vid_list = [vid for vid in os.listdir(cam_dir) if vid.endswith(".mp4")]
csv_list = [csv for csv in os.listdir(cam_dir) if csv.endswith(".csv")]
npy_list = [npy for npy in os.listdir(cam_dir) if npy.endswith(".npy")]
dir_size = int((len(os.listdir(cam_dir))-1)/3)
vid_dirs = []
csv_dirs = []
npy_dirs = []
cam = "Camera2"
for vid, csv, npy in zip(vid_list, csv_list, npy_list):
	vid_dirs.append(os.path.join(cam_dir, vid))
	csv_dirs.append(os.path.join(cam_dir, csv))
	npy_dirs.append(os.path.join(cam_dir, npy))

dir_tuple = list(zip(vid_dirs, csv_dirs, npy_dirs))
move_list = random.sample(dir_tuple, 15)

for files in move_list:
	vid, csv, npy = files
	print(vid)
	index = vids_added[f"{cam}"]
	os.symlink(vid, f'/data3/hand_track/Pancho/meow_240725_trainset/fly2/vid-t{index}.mp4')
	os.symlink(csv, f"/data3/hand_track/Pancho/meow_240725_trainset/fly2/metadata-t{index}.csv")
	os.symlink(npy, f"/data3/hand_track/Pancho/meow_240725_trainset/fly2/frametimes-t{index}.npy")
	vids_added[f"{cam}"] += 1


