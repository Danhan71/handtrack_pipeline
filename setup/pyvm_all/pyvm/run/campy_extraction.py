""" 
Extract campy frametimes --> for use in ml2
Also extracts each video's frames.
"""
from pyvm.classes.videoclass import Videos
import os
import numpy as np
from pythonlib.tools.expttools import writeStringsToFile
import argparse


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Description of your script.")
    parser.add_argument("name", type=str, help="Experiment name/date")
    parser.add_argument("animal", type=str, help="meow")
    args = parser.parse_args()    

    name = args.name
    date = name.split("_")[0]
    expt = name.split("_")[1]
    condition = "behavior"
    animal = args.animal

    V = Videos()
    V.load_data_wrapper(date=date, expt=expt, condition=condition, animal=animal)
    # assert False
    ##### Campy - loading and checking
    V.campy_preprocess_check_frametimes()

    ########### camy plots - USEFUL PLOTTING STUFF
    if False:
        # Plots frametimes, etc.
        trial = list_trials[0]
        trial = 141
        for indgrp in range(3):
            V.campy_plot_frametimes((indgrp, trial))

    if False:
        # if want to keep all videos, then dont run this. 
        # should keep.
        V.clean_videos()

    ##### Export data (campy, dlc), so can load into ml2
    V.campy_export_to_ml2()
