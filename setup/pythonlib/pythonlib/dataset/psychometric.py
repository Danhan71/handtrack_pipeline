""" Need combines task and beh in Dataset. Does manipulations of Dataset to allow analysis 
of psychometric tasks.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from ..drawmodel.strokePlots import plotDatStrokes
from pythonlib.tools.pandastools import applyFunctionToAllRows
from pythonlib.tools.expttools import makeTimeStamp, findPath



def _check_dataset(D):
    """ check that it is compatible with these anslyes
    """
    assert "Task" in D.Dat.columns, "need to first extract Tasks (general class)"

def _get_row_col_map(taskcategory):
    """ Stores row and column for plotting each task, as function of
    taskcategory (i.e. taskcategory) and task num, i.e., in task str, the 
    last number, e.g,, 3line_psych2_3 is num 3.
    """

    if taskcategory=="3line_psych2":
        rowcolmappings = {
            1:{"row":0, "col":0},
            2:{"row":0, "col":1},
            3:{"row":0, "col":2},
            4:{"row":0, "col":3},
            5:{"row":0, "col":4},
            6:{"row":1, "col":0},
            7:{"row":1, "col":1},
            8:{"row":1, "col":2},
            9:{"row":1, "col":3},
            10:{"row":1, "col":4},
            11:{"row":2, "col":0},
            12:{"row":2, "col":1},
            13:{"row":2, "col":2},
            14:{"row":2, "col":3},
            15:{"row":2, "col":4},
            16:{"row":3, "col":2},
            17:{"row":3, "col":0},
            18:{"row":3, "col":1},
            19:{"row":3, "col":3},
            20:{"row":3, "col":4},
        }
    elif taskcategory=="3line_psych1":
        rowcolmappings = {
            1:{"row":0, "col":0},
            2:{"row":0, "col":1},
            3:{"row":0, "col":2},
            4:{"row":0, "col":3},
            5:{"row":0, "col":4},
            6:{"row":1, "col":0},
            7:{"row":1, "col":1},
            8:{"row":1, "col":2},
            9:{"row":1, "col":3},
            10:{"row":1, "col":4},
            11:{"row":2, "col":0},
            12:{"row":2, "col":1},
            13:{"row":2, "col":2},
            14:{"row":2, "col":3},
            15:{"row":2, "col":4},
        }
    elif taskcategory=="3line_psych6":
            rowcolmappings = {
            1:{"row":0, "col":0},
            2:{"row":0, "col":1},
            3:{"row":0, "col":2},
            4:{"row":1, "col":0},
            5:{"row":1, "col":1},
            6:{"row":1, "col":2},
            7:{"row":2, "col":0},
            8:{"row":2, "col":1},
            9:{"row":2, "col":2},
            10:{"row":3, "col":0},
            11:{"row":3, "col":2},
        }
    elif taskcategory=="3line_psych7":
            rowcolmappings = {
            1:{"row":0, "col":0},
            2:{"row":0, "col":1},
            3:{"row":0, "col":2},
            4:{"row":0, "col":3},
            5:{"row":0, "col":4},
            6:{"row":1, "col":1},
            7:{"row":1, "col":2},
            8:{"row":1, "col":3},
            9:{"row":1, "col":4},
            10:{"row":2, "col":1},
            11:{"row":2, "col":2},
            12:{"row":2, "col":3},
            13:{"row":2, "col":4},
        }        
    else:
        rowcolmappings = {i+1:{"row":0, "col":i} for i in range(30)}
    # else:
    #     print(taskcategory)
    #     assert False, "have not coded for this set"
    return rowcolmappings


def extractPsychometricDF(D, taskcategory):
    """ Gets subset of D, which contains just tasks for this
    taskcategory, and also assigns rows and cols, for plotting
    these tasks.
    """
    
    _check_dataset(D)

    # get only this dataset
    dfthis = D.Dat[D.Dat["task_stagecategory"]==taskcategory]
    dfthis = dfthis.reset_index(drop=True)

    map_row_col = _get_row_col_map(taskcategory)

    # for each row get its data
    def F(x, rowcol):
        # get its id
        num = x["Task"].get_task_id()[1]
        return map_row_col[num][rowcol]
        
    for rowcol in ["row", "col"]:
        dfthis = applyFunctionToAllRows(dfthis, lambda x: F(x, rowcol), rowcol)

    return dfthis


def plotMultTaskcategories(D, sdir=None, max_n_per_grid=None):
    """ wrapper, to plot all task categories that have name "psych" as 
    part of name.
    - sdir, leave None to not save. otherwise give me directory.
        sdir = f"/data2/analyses/main/psychometric/{animal}-{expt}/plots"
        os.makedirs(sdir, exist_ok=True)
        print(sdir)
    """
    # extract data to plot
    from pythonlib.dataset.plots import plot_dat_grid_inputrowscols
    from pythonlib.dataset.psychometric import extractPsychometricDF

    # Get all psychometric tasks.
    sets_ = set(D.Dat["task_stagecategory"])
    sets_ = sorted([s for s in sets_ if "psych" in s])
    print("Plotting these cats:")
    print(sets_)

    for settoget in sets_:
        dfthis = extractPsychometricDF(D, settoget)
        figbeh, figtask = plot_dat_grid_inputrowscols(dfthis, max_n_per_grid=max_n_per_grid)

        if sdir is not None:
            figbeh.savefig(f"{sdir}/{settoget}-beh.pdf")
            figtask.savefig(f"{sdir}/{settoget}-task.pdf")
            print(f"Saved {settoget} to {sdir}")
