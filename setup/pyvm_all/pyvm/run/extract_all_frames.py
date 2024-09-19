""" 
OTIONAL - to extract all frames and save, for all vidoes. Don't need to run this for 
normal analysis.
"""
from pyvm.classes.videoclass import Videos
import os
import numpy as np
from pythonlib.tools.expttools import writeStringsToFile


if __name__=="__main__":
    # === load: cagetest2 - checkerboard
    # expt = "chunkbyshape4"
    expt = "220811"
    condition = "behavior"

    V = Videos()
    V.load_data_wrapper(expt, condition)
    V.extract_all_frames_from_videos("orig")
