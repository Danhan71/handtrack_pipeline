""" related to taskmodel - analysis functions"""

from pythonlib.tools.stroketools import *
import matplotlib.pyplot as plt
import numpy as np
from .taskmodel import Model, Dataset 
        

################### HELPER FUNCTIONS


def makeDataset(filedata, trials_list):
    """ helper to maek a Dataset object with the trials included here"""
    # 1) behavior
    strokes_all = getMultTrialsStrokes(filedata, trials_list)

    # 2) task
    tasks = [getTrialsTask(filedata, t) for t in trials_list]
    fix_all_task = [getTrialsFix(filedata, t)["fixpos_pixels"] for t in trials_list]
    strokes_all_task = [getTrialsTaskAsStrokes(filedata, t, fake_timesteps="from_orig") for t in trials_list]
    for strokes, t, f in zip(strokes_all_task, tasks, fix_all_task):
        t["strokes"] = strokes
        t["fixpos"] = f
        
    dset = Dataset(strokes_all, tasks)
    return dset

def makeModel(PARAMS_MODEL):
    mod = Model(PARAMS_MODEL["modelname"], priorFunction, likeliFunction,
                parse_ver=PARAMS_MODEL["parse_ver"], chunkmodel=PARAMS_MODEL["chunkmodel"])
    return mod

############# (11/2/20 ADDED)


