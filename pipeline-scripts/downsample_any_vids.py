from pyvm.dlc.initialize import downsample_all_videos

cam_list = ["bfs1","flea","fly1","fly2"]
vid_path_list = []

for cam in cam_list:
	vid_path_list.append(f"/data3/hand_track/Pancho/221015_dircolor2_wandgrid/3dgrid/{cam}/vid-t0.mp4")

downsample_all_videos(vid_path_list, cam_list)