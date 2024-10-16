import os
import random

data_dir = "/data3/hand_track/Panchego"

cam_dirs = []
vids_added = {
	"Camera1" : 0,
	"Camera2" : 40,
	"Camera3" : 0,
	"Camera4" : 0,
	"Camera5" : 45
	}

cam_dir = "/data3/hand_track/Panchego/meow_240509_primpancho1c/Camera2"

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
move_list = random.sample(dir_tuple, 10)

for files in move_list:
	vid, csv, npy = files
	print(vid)
	index = vids_added[f"{cam}"]
	os.symlink(vid, f'/data3/hand_track/Panchego/train_set/fly2/vid-t{index}.mp4')
	os.symlink(csv, f"/data3/hand_track/Panchego/train_set/fly2/metadata-t{index}.csv")
	os.symlink(npy, f"/data3/hand_track/Panchego/train_set/fly2/frametimes-t{index}.npy")
	vids_added[f"{cam}"] += 1


