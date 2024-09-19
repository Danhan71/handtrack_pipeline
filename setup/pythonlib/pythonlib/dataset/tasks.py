""" For loading/manipulating datasetes of tasks.
Here tasks are saved task class objects 
"""

import pickle
from pythonlib.globals import PATH_DATASET_BEH

def load_presaved_tasks(animal, expt):
    """ helper, to load pre-saved tasks for animal/expt
    RETURNS:
    - Tasks
    """
    from pythonlib.tools.expttools import findPath

    #  LOAD PRESAVED MONKEY TASKS
    sdir = f"{PATH_DATASET_BEH}/TASKS_GENERAL/{animal}-{expt}-all"
    pathlist = findPath(sdir, [], "Tasks", "pkl")
    assert len(pathlist)==1
    with open(pathlist[0], "rb") as f:
        Tasks = pickle.load(f)

    return Tasks

