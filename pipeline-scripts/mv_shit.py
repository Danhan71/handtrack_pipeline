import shutil
import numpy as np

ddir = "/data3/hand_track/Pancho"
cdir=f"{ddir}/220914_wandall/wand"

counts = {
	"bfs1" : 0,
	"bfs2" : 0,
	"flea" : 0,
	"fly1" : 0,
	"fly2" : 0
}
expts_list = {
"220922_grammardircolor1_wandgrid" : ["t0"], 
"220924_grammardircolor1b_wandgrid": ["t1"], 
"220926_grammardircolor2_wandgrid" : ["t2"], 
"221015_dircolor2_wandgrid" : ["t2"], 
"221015_dircolor3b_wandgrid" : ["t1"], 
"221015_dirshapecolor1b_wandgrid": ["t1"]}
cams_list = ["bfs1", "bfs2", "flea", "fly1", "fly2"]

for expt in expts_list:
	expt_dir = f"{ddir}/{expt}/wand"
	for cam in cams_list:
		camdir = f"{expt_dir}/{cam}"
		for t in expts_list[expt]:
			vidname = f"{camdir}/vid-{t}.mp4"
			ftname = f"{camdir}/frametimes-{t}.npy"
			this_cdir = f"{cdir}/{cam}"
			shutil.copy(vidname,f"{cdir}/{cam}/vid-t{counts[cam]}.mp4")
			shutil.copy(ftname,f"{cdir}/{cam}/frametimes{counts[cam]}.npy")
			counts[cam] = counts[cam] + 1

