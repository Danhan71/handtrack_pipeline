import os
import numpy as np
from pathlib import Path
from pyvm.globals import BASEDIR

def get_key():
	"""
	Function to get user input for the "GUI"
	"""
	from pynput import keyboard
	key_pressed = []
	def on_press(key):
		try:
			# Check key
			if key.char in ['1', '2','0','9','5']:
				# Convert the key to an integer and store it
				key_pressed.append(int(key.char))
				return key_pressed
		except AttributeError:
			pass
			print("Enter valid key")

	def on_release(key):
		# Stop the listener once a valid key is pressed
		try:
			# Check key
			if key.char in ['1', '2','0','9','5']:
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

def review_strokes(name,animal):
    """
    Main function for stroke review+
    """
    from PIL import Image, ImageTk
    import csv
    import psutil
    import tkinter as tk
    import matplotlib.pyplot as plt
    import re

    date = int(name.split('_')[0])
    if date < 220914:
        prefix = '220412_no_f1bf2'
    else:
        prefix = 'merged'

    lag_path = f'{BASEDIR}/{animal}/{name}/lag/{prefix}'

    with open(f"{lag_path}/good_inds.txt",'r') as f:
        good_inds_read = f.read()
        if len(good_inds_read) > 0:
            print()
            while True:
                    user_input = input(f"Good inds already exist for {name}, would you like to overwrite them? (y/n)")
                    if user_input == 'y' or user_input == 'Y':
                        break
                    else:
                        print("Skipping")
                        return
    stroke_path = f"{lag_path}/corr_figs"

    # review_displays("instructions", metrics=None)

    good_inds = []
    strokes_list = os.listdir(stroke_path)
    if len(strokes_list) < 40:
          print("Not enough trials to extract strokes from. Skipping...")
    # print(stroke_path)
    # print(dirs_list)
    # assert False
    import time
    i = 0
    good_count = 0
    while i < len(strokes_list) and good_count < 20:
        stroke = strokes_list[i]
        this_path = os.path.join(stroke_path,stroke)
        with Image.open(this_path) as im:
            plt.figure(figsize=(40,40))
            plt.tight_layout()
            plt.imshow(im)
            plt.axis('off')
            plt.show(block=False)
            plt.draw()
            plt.pause(0.5)
            print("""Yes/no (1/2) or mv fwd/bkwd(9/0)\n
            1: Good\n
            2: Bad\n
            0: forward\n
            9: backward\n""")
            print(f"Doing image: {stroke}")
            response = get_key()
            print(f"You entered: {response}")
            if response == 9:
                i = i-1
                good_count -= 1
            elif response == 0:
                i = i+1
            elif response == 2:
                good_count += 1
                i += 1
                match = re.search(r'(\d+-\d+)', stroke)
                trial_stroke_ind = match.group(1) if match else None
                good_inds.append(trial_stroke_ind)
            elif response == 5:
                print('Manual skip all day')
                plt.close()
                return
            else:
                  i += 1
            plt.close()
    with open(f"{lag_path}/good_inds.txt","w") as f:
        f.write(','.join(good_inds))


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Do lag stuff")
    parser.add_argument("animal", type=str, help="amnoimal")

    animal = parser.parse_args().animal
    
    names_list = os.listdir(f"{BASEDIR}/{animal}")

    name_list_good = []
    for name in names_list:
        try:
            date = int(name.split('_')[0])
        except ValueError:
            print(f'skipping {name} not real data folder')
            continue
        if date < 220914:
            prefix = '220412_no_f1bf2'
        else:
            prefix = 'merged'
        if os.path.exists(f"{BASEDIR}/{animal}/{name}/lag/{prefix}/corr_figs"):
            name_list_good.append(name)

    import random
    names_use = random.sample(name_list_good, 5)

    for name in names_use:
        review_strokes(name,animal)