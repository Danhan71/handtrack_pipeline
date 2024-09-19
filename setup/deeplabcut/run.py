# LT, obsolete (use videomonkey)

import deeplabcut
import tensorflow as tf

def get_video_list(path_config_file):
	""" return list of videos"""
	import yaml
	with open(path_config_file) as f:
		config = yaml.load(f)
		vidlist = list(config["video_sets"].keys())
	return vidlist

def get_dlc_video_list(path_config_file):
	""" return list of videos"""
	dlc_vid_list = []
	import yaml
	with open(path_config_file) as f:
		config = yaml.load(f)
		vidlist = list(config["video_sets"].keys())
	for vid in vidlist
		vid = vid.split('.mp4')[0]
		with open(path_config_file) as f:
			config = yaml.load(f)
			vid_suffix = str(config["Task"].values())
			vid = vid + vid_suffix + ".mp4"
			dlc_vid_list.append(vid)
	return dlc_vid_list


def main(path_config_file, ver):

	if ver=="train":
		deeplabcut.train_network(path_config_file, allow_growth=True)
	elif ver=="evaluate":
		deeplabcut.evaluate_network(path_config_file, plotting=True)
	elif ver =="analyze":
		vidlist = get_video_list(path_config_file)
		deeplabcut.analyze_videos(path_config_file, vidlist, videotype='.mp4', save_as_csv=True)
	elif ver=="create_labeled_video":
		vidlist = get_video_list(path_config_file)
		deeplabcut.create_labeled_video(path_config_file, vidlist)





if __name__=="__main__":
	# pcflist = [
	# 	'/data1/code/python/DeepLabCut/examples/test_flea3-Lucas-2021-03-25/config.yaml', 
	# 	'/data1/code/python/DeepLabCut/examples/test_blackfly-Lucas-2021-03-25/config.yaml'
	# ]
	pcflist = [
		'/data1/code/python/DeepLabCut/examples/camtest3_wand2_cam1flea-Lucas-2021-04-19/config.yaml', 
		'/data1/code/python/DeepLabCut/examples/camtest3_wand2_cam2blackfly-Lucas-2021-04-19/config.yaml', 
	]
	for path_config_file in pcflist:
		ver = "create_labeled_video"

		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

		main(path_config_file, ver)