
"""
Various functions for computing/processing and looking at metrics
	extract_frames: Extract frames for review
	review_frames: Review extracted frames for correctness of label
	extract_review: Do both
	calculate: Calcultae various metrics for this trial, incorporates frame review results, quantifies jumps, and pulls out
	cameras with bad performance at certain points. 
"""

import os
import sys
import shutil
import re
import glob
import numpy as np
import random
from pathlib import Path
from skimage import io
from deeplabcut.utils import auxiliaryfunctions


def extract_frames(
	config,
	numvids,
	numframestot,
	videos_list,
	algo="uniform",
	crop=False,
	userfeedback=False,
	cluster_step=1,
	cluster_resizewidth=30,
	cluster_color=False,
	opencv=True,
	slider_width=25,
	config3d=None,
	extracted_cam=0,
):
	"""
	Extracts random sample of labelled frames from DLC 
	"""
	from skimage.util import img_as_ubyte
	from deeplabcut.utils import frameselectiontools
	from deeplabcut.utils import auxiliaryfunctions

	config_file = Path(config).resolve()
	cfg = auxiliaryfunctions.read_config(config_file)
	print("Config file read successfully.")

	start = cfg["start"]
	stop = cfg["stop"]

	# Check for variable correctness
	if start > 1 or stop > 1 or start < 0 or stop < 0 or start >= stop:
		raise Exception(
			"Erroneous start or stop values. Please correct it in the config file."
		)
	if numframestot < 1 and not int(numframestot):
		raise Exception(
			"Perhaps consider extracting more, or a natural number of frames."
		)
	videos_all = videos_list

	if numvids > len(videos_all):
		print("######### Actual number of videos is less than the number of videos this function was told to pick, defaulting to actual...")
		numvids = len(videos_all)


	numframes2pick = int(numframestot/numvids)
	#This ensures we always a number of frames equal to the indictated total
	assert numframestot == numvids * numframes2pick

	#Sample subset of videos to extaact frames from
	videos = random.sample(videos_all, numvids)

		
	if opencv:
		from deeplabcut.utils.auxfun_videos import VideoReader
	else:
		from moviepy.editor import VideoFileClip

	output_path = Path(config).parents[0] / "check-labeled-data"
	if output_path.exists() and os.listdir(output_path):
		while True:
				user_input = input(f"It appears that {len(os.listdir(output_path))} frames have been extracted for this day, would you like to extract {numframestot} more? (y/n)")
				if user_input == 'y' or user_input == 'Y':
					break
				else:
					while True:
						user_input = input(f"Would you like to delete the already extracted frames?")
						if user_input == 'y' or user_input == 'Y':
							shutil.rmtree(output_path)
							extract_frames(videos_list=videos_list, config=config, numvids=numvids, numframestot=numframes)
						else:
							print("Okay, moving on to frame review step")
							return
	if not output_path.exists():
		os.mkdir(output_path)
	has_failed = []
	for video in videos:
		 
		cap = VideoReader(video)
		nframes = len(cap)
		if not nframes:
			print("Video could not be opened. Skipping...")
			continue


		indexlength = int(np.ceil(np.log10(nframes)))

		fname = Path(video)
		


		if crop and not opencv:
			clip = clip.crop(
				y1=int(coords[2]),
				y2=int(coords[3]),
				x1=int(coords[0]),
				x2=int(coords[1]),
			)
		elif not crop:
			coords = None

		print("Extracting frames based on %s ..." % algo)
		if algo == "uniform":
			if opencv:
				frames2pick = frameselectiontools.UniformFramescv2(
					cap, numframes2pick, start, stop
				)
			else:
				frames2pick = frameselectiontools.UniformFrames(
					clip, numframes2pick, start, stop
				)
		elif algo == "kmeans":
			if opencv:
				frames2pick = frameselectiontools.KmeansbasedFrameselectioncv2(
					cap,
					numframes2pick,
					start,
					stop,
					crop,
					coords,
					step=cluster_step,
					resizewidth=cluster_resizewidth,
					color=cluster_color,
				)
			else:
				frames2pick = frameselectiontools.KmeansbasedFrameselection(
					clip,
					numframes2pick,
					start,
					stop,
					step=cluster_step,
					resizewidth=cluster_resizewidth,
					color=cluster_color,
				)

		if not len(frames2pick):
			print("Frame selection failed...")
			return

		is_valid = []
		if opencv:
			for index in frames2pick:
				cap.set_to_frame(index)  # extract a particular frame
				frame = cap.read_frame()
				if frame is not None:
					image = img_as_ubyte(frame)
					img_name = (
						str(output_path)
						+ "/img"
						+ str(index).zfill(indexlength)
						+ ".png"
					)
					if crop:
						io.imsave(
							img_name,
							image[
								int(coords[2]) : int(coords[3]),
								int(coords[0]) : int(coords[1]),
								:,
							],
						)  # y1 = int(coords[2]),y2 = int(coords[3]),x1 = int(coords[0]), x2 = int(coords[1]
					else:
						io.imsave(img_name, image)
					is_valid.append(True)
				else:
					print("Frame", index, " not found!")
					is_valid.append(False)
			cap.close()
		else:
			for index in frames2pick:
				try:
					image = img_as_ubyte(clip.get_frame(index * 1.0 / clip.fps))
					img_name = (
						str(output_path)
						+ "/img"
						+ str(index).zfill(indexlength)
						+ ".png"
					)
					io.imsave(img_name, image)
					if np.var(image) == 0:  # constant image
						print(
							"Seems like black/constant images are extracted from your video. Perhaps consider using opencv under the hood, by setting: opencv=True"
						)
					is_valid.append(True)
				except FileNotFoundError:
					print("Frame # ", index, " does not exist.")
					is_valid.append(False)
			clip.close()
			del clip

		if not any(is_valid):
			has_failed.append(True)
		else:
			has_failed.append(False)

	if all(has_failed):
		print("Frame extraction failed. Video files must be corrupted.")
		return
	elif any(has_failed):
		print("Although most frames were extracted, some were invalid.")
	else:
		print(
			"Frames were successfully extracted for a random sample of the labelled videos."
			)

def get_key():
	"""
	Function to get user input for the "GUI"
	"""
	from pynput import keyboard
	key_pressed = []
	def on_press(key):
		try:
			# Check if the key is a number between 1 and 5
			if key.char in ['1', '2', '3', '4', '5','0','9']:
				# Convert the key to an integer and store it
				key_pressed.append(int(key.char))
				return key_pressed
		except AttributeError:
			pass
			print("Enter valid key")

	def on_release(key):
		# Stop the listener once a valid key is pressed
		try:
			# Check if the key is a number between 1 and 5
			if key.char in ['1', '2', '3', '4', '5','0','9']:
				# Convert the key to an integer and store it
				return False
		except AttributeError:
			pass
			print("Enter valid key")

	with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
		listener.join()
	if key_pressed:
		return key_pressed[0]  
	else:
		print("Label issue please repeat key stroke")
		return get_key()
	
	# Return the stored key

def review_frames(config):
	"""
	Main function for frame review+
	"""
	from PIL import Image, ImageTk
	import csv
	import psutil
	import tkinter as tk
	import matplotlib.pyplot as plt

	config_file = Path(config).resolve()
	cfg = auxiliaryfunctions.read_config(config_file)
	print("Config file read successfully.")

	frame_path = Path(config).parents[0] / "check-labeled-data/"

	if os.path.exists(f"{frame_path}/metrics.csv"):
		print()
		while True:
				user_input = input("Metrics file already exists, would you like to overwrite it? (y/n)")
				if user_input == 'y' or user_input == 'Y':
					os.remove(f"{frame_path}/metrics.csv")
					break
				else:
					assert False, "Fix your issue then come back to me"

	# review_displays("instructions", metrics=None)

	frames_dict = {}
	frames_list = os.listdir(frame_path)
	# print(frame_path)
	# print(dirs_list)
	# assert False
	import time
	i = 0
	while i < len(frames_list):
		frame = frames_list[i]
		this_path = os.path.join(frame_path,frame)
		with Image.open(this_path) as im:
			plt.figure(figsize=(10,10))
			plt.tight_layout()
			plt.imshow(im)
			plt.axis('off')
			plt.show(block=False)
			plt.draw()
			plt.pause(0.5)
			print("""Hit number to classify image (1-5) or mv fwd/bkwd(9/0)\n
			1: Label, good\n
			2: Label, bad\n
			3: No label, good\n
			4: No label, bad\n
			5: Don't label""")
			print(f"Doing image: {i}")
			response = get_key()
			print(f"You entered: {response}")
			if response == 9:
				i = i-1
			elif response == 0:
				i = i+1
			else:
				frames_dict[f"{frame}"] = response
				i = i+1
			plt.close()
	labels = list(frames_dict.values())
	good_labels = labels.count(1)
	good_misses = labels.count(3)
	bad_labels = labels.count(2)
	bad_misses = labels.count(4)
	total_labels = len(labels) - labels.count(5)
	extracted_frames = len(labels)

	metrics = {
	"good_labels": good_labels,
	"good_misses": good_misses,
	"bad_labels": bad_labels,
	"bad_misses": bad_misses,
	"total_labels": total_labels,
	"extracted_frames": extracted_frames
	}


	with open(f"{frame_path}/metrics.csv", 'w') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerows(metrics.items())

	
	try:
		print(f"""Result Name : (proportion)\n
Good: {(metrics["good_labels"] + metrics["good_misses"])/metrics["total_labels"]}
  Good Labels: {metrics["good_labels"]/metrics["total_labels"]}
  Good Misses: {metrics["good_misses"]/metrics["total_labels"]}
Bad: {(metrics["bad_labels"] + metrics["bad_misses"])/metrics["total_labels"]}
  Bad Misses : {metrics["bad_misses"]/metrics["total_labels"]}
  Bad Labels: {metrics["bad_labels"]/metrics["total_labels"]}
Total Frames : {metrics["extracted_frames"]}
  Total Counted : {metrics["total_labels"]}""")
	except ZeroDivisionError:
		assert False, "Look slike there is no data here"



def  main(name, do_list, numvids=None, numframes=None):
	from pyvm.globals import BASEDIR
	for task in do_list:
		dict_paths, _ = find_expt_config_paths(name, "behavior")
		pcf = list(dict_paths.values())[0]
		if task == "extract":
			videos_list = findPath(f"{BASEDIR}/{animal}/{name}/behavior/DLC", [["combined"],["allvideos"]], ext = ".mp4")
			extract_frames(videos_list=videos_list, config=pcf, numvids=numvids, numframestot=numframes)
		if task == "review":
			review_frames(config=pcf)

if __name__=="__main__":
	import argparse
	from pythonlib.tools.expttools import findPath
	from pyvm.globals import BASEDIR
	from initialize import find_expt_config_paths

	parser = argparse.ArgumentParser(description="Description of your script.")
	parser.add_argument("name", type=str, help="Experiment name/date")
	parser.add_argument("animal", type=str, help="Help yourself")
	parser.add_argument("--numvids", type=str, help="Number of videos to pick", default=20, required=False)
	parser.add_argument("--numframes", type=str, default=200, required=False, help="Number of frames to pick TOTAL (i.e. numvids*numframes/vid this is to ensure that this many frames are selected if there are fewer videos than indicated as this is mos timportat)")
	parser.add_argument("--do", type=str, help="Do extract/review/both steps")
	args = parser.parse_args()
	name = args.name
	animal = args.animal
	numvids = int(args.numvids)
	numframes = int(args.numframes)
	do = args.do

	if do == "extract":
		print("Doing frame extraction...")
		main(name=name, do_list=["extract"], numvids=numvids, numframes=numframes)
	elif do == "review":
		print("Doing frame review, please see popups for instructions...")
		main(name=name, do_list=["review"])
	elif do == "both":
		print("Doing both steps...")
		main(name=name, do_list=["extract","review"], numvids=numvids, numframes=numframes)
		

