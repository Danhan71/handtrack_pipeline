""" Put NN behavior into a Dataset object, to then be able to use all the Dataset functioanlity.

"""

import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pythonlib.tools.pandastools import applyFunctionToAllRows
# import torch
# import os
from pythonlib.tools.expttools import makeTimeStamp, findPath


def make_dataset_from_model_and_tasks(Model, Tasks, ver="drawnn", animal=None):
    """ given Model (loaded pre-saved Model), and Tasks,
    list of Tasks (which were pre-saved, see .tasks.py, then 
    tests Model on tasks, and saves the resulting behavior in a 
    Dataset.
    INPUT:
    - Model
    - Tasks
    RETURNS
    - Dataset object
    - modifies Model.
    """

    if ver=="drawnn":
        from drawnn.tools.taskclass import convertToOldTasks
        from pythonlib.dataset.dataset import Dataset
        from pythonlib.drawmodel.taskgeneral import getSketchpad

        # Convert to old tasks for running in drawnn
        # Convert all tasks to old DrawNN tasks
        Taskslist = [T["Task"] for T in Tasks]
        _, TasksNN, params_tasks = convertToOldTasks(Taskslist)

        # Test NN on these new tasks
        Model.tasksReplace(TasksNN)
        Model.activationsGenerate()

        ### [GOOD] Put model behavior into Dataset, then run same analyses as for monkey
        D = Dataset([])

        params={}
        params["strokes_list"] = Model.A.strokes_all
        params["task_list"] = Taskslist

        D.initialize_dataset(ver="generic", params=params)

        # add task information, based on loaded inforamtion in task
        assert(len(Tasks)==len(D.Dat))
        dftasks = pd.DataFrame(Tasks)

        for col in dftasks.columns:
            if col !="Task":
                D.Dat[col] = dftasks[col]

        if animal is not None:
            D.Dat["animal"] = animal

        # pull out strokes from Task
        def F(x):
            return [S[:,:2] for S in x["Task"].Strokes]

        D.Dat = applyFunctionToAllRows(D.Dat, F, "strokes_task")

        # Check that coords are same for abstract and drawnn. oterhwise need to convert
        a = getSketchpad(Taskslist, "abstract")
        b = getSketchpad(Taskslist, "drawnn")
        assert np.all(a==b), "need to convert to drwann coord"

    else:
        print(ver)
        assert False, "not coded"

    return D


def align_behavior_and_nn_datasets(Dnn, Dbeh, Tasks, ploton=False):
    """ makes sure they have identical rows, same size,
    with rows defined by trial-code-animal combination.
    NOTE: both datsets must have been constried from same
    animal and expt. where Dnn build from 
    make_dataset_from_model_and_tasks, where it took in tasks
    that were from the behaviora experiment in Dbeh.
    INPUTS:
    - Tasks, the same list of tasks that were passed into Dnn for
    making it. THis is only used for converting sketchpad coords 
    between Dnn and Dbeh
    RETURNS:
    - None
    - modifies Dnn in place. Dbeh is modified to match rows of Dnn
    """
    from pythonlib.dataset.dataset import matchTwoDatasets, mergeTwoDatasets
    import random

    # Assign each row a unique id.
    def F(x):
        return (x["animal"], x["trialcode"])
    Dnn.Dat = applyFunctionToAllRows(Dnn.Dat, F, "rowid")
    Dbeh.Dat = applyFunctionToAllRows(Dbeh.Dat, F, "rowid")

    # for reducing size datasets to match each other exacltly.
    # Then merge.
    matchTwoDatasets(Dnn, Dbeh)
    matchTwoDatasets(Dbeh, Dnn)
    Dnn.Dat = mergeTwoDatasets(Dnn, Dbeh)

    if ploton:
        inds = random.sample(range(len(Dnn.Dat)), 10)
        Dnn.plotMultTrials(inds)
        Dbeh.plotMultTrials(inds)

        Dnn.plotMultTrials(inds, "strokes_task")
        Dbeh.plotMultTrials(inds, "strokes_task")

    # # put all in Dnn, by adding column taken from Dbeh
    # Dnn["strokes_nn"] = Dnn

    return Dnn, Dbeh


def align_coord_behavior_and_nn(Dbeh, Tasks):
    """ modify coord for Dbeh so mnatches those of Dnn,
    based on coords saved in Tasks. Tasks is waht was used
    for constructing Dnn, and holds sketchpad edges for
    nn and beh
    RETURNS
    - modifies Dbeh in place
    """
    from pythonlib.drawmodel.taskgeneral import getSketchpad

    # Convert coord from monkey to drawnn
    Taskslist = [T["Task"] for T in Tasks]
    edges_in = getSketchpad(Taskslist, "ml2")
    edges_out = getSketchpad(Taskslist, "drawnn")
    Dbeh.convertCoord(edges_out=edges_out, edges_in = edges_in)

