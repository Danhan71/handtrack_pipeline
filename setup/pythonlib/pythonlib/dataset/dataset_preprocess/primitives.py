from pythonlib.tools.pandastools import applyFunctionToAllRows
import os
import numpy as np
import matplotlib.pyplot as plt

def preprocess(D, animal, expt, recenter=True):
    """ for preparaing for prim anslsyus.
    Extracts spatial coordinates and shapes, etc.
    """

    D = D.copy()

    #### First preprocess dataset so that strokes are all centerized
    list_sb, list_st, D = D.extractStrokeLists(recenter=recenter)
    D.Dat["strokes_beh"] = list_sb

    # Expose each tasks params
    def F(x, ver):
        """ expose the task params"""
        T = x["Task"]
        
        if "circle" in x["character"] and ver=="theta":
            # circles are symmetric circluar.
            return 0.
        else:
            return T.ShapesOldCoords[0][1][ver] 
        
    for ver in ["sx", "sy", "x", "y", "theta"]:
        D.Dat = applyFunctionToAllRows(D.Dat, lambda x:F(x, ver), ver)


        # === Remove short strokes
    if False:
        # To plot distribution of stroke distances
        from pythonlib.drawmodel.features import strokeDistances

        strokeDistances(strokes)

        listns = []
        listdists = []
        for i in range(len(D.Dat)):
            strokes = D.Dat.iloc[i]["strokes_beh"]
            for s in strokes:
                listns.append(len(s))
            listdists.extend(strokeDistances(strokes))
        #         if len(s)<5:
        #             print(i)
        #             print(s)
        #             assert False

        plt.figure()
        plt.hist(listns, 100)
        plt.figure()
        plt.hist(listdists, 100)
        plt.figure()
        plt.plot(listns, listdists, "xk")

    THRESH = 10 # min num pts. Remove any trial for which all traj in strokes are shorter than this.
    def F(x):
        strokes = x["strokes_beh"]
        tmp = [len(s)<THRESH for s in strokes]
        if all(tmp):
            return True
        else:
            return False
    D.Dat = applyFunctionToAllRows(D.Dat, F, "tmp")

    # Remove all that are true
    D = D.filterPandas({"tmp":[False]}, "dataset")
    del D.Dat["tmp"]


    # train or test defined by block
    if expt=="primcat12":
        # I made this just by looking at the primcat blocksequence code.
        list_train = [1, 2, 3, 4, 5, 6, 7, 8, 9, 22, 23, 24, 25, 11, 26, 13, 15]
        list_test = list(range(1, 36+1))
        list_test = [x for x in list_test if x not in list_train]
        list_other = [37, 38, 39, 40]
    D.analy_reassign_monkeytraintest("block", list_train, list_test, list_other)


    # Grouping, based on primtives kind
    # In general, assign a name based on aspects of Objects (shapes and parameters)
    groupby = ["character", "sx", "sy", "theta"]
    D.grouping_append_col(groupby, "primtuple")

    # Dates
    return D

def clean_and_group_data(D, TRAIN, animal, expt, summary_date_epoch = None, SAVEDIR_THIS=None):
    """ 
    Extract subset of D, based on various criteria. Will note whether each trial
    is in early or late (or undefined) stage, based on date. Only keep prims that have
    data for both early and late.
    prims that have data for both early and late dates.
    - TRAIN, 'train' or 'test'.
    - animal, expt...
    - SAVEDIR_THIS, will save the dates dicts. None, to not save.
    """
    from pythonlib.dataset.analy import preprocess_dates

    Dthis = D.filterPandas({"monkey_train_or_test":[TRAIN]}, "dataset")

    ### ASSIGN UNIQUE PRIMTUPLES AS A NEW COLUMN
    DictDateEarlyLate, DictDay, DictDateInds, Dthis = preprocess_dates(Dthis, 
        "primtuple", animal, expt, return_filtered_dataset=True, SAVEDIR=SAVEDIR_THIS)

    # For each row of dataset, figure out if it is in ealry or late, or neither
    def F(x):
        # doesnt have dates?
        if x["primtuple"] not in DictDateEarlyLate:
            print(x["primtuple"])
            assert False

        # check whether this row's data is in this date
        datethis = x["date"]
        dates_earlylate = DictDateEarlyLate[x["primtuple"]]
        if datethis == dates_earlylate[0]:
            return "early"
        elif datethis == dates_earlylate[1]:
            return "late"
        else:
            return "not_assigned"

    Dthis.Dat = applyFunctionToAllRows(Dthis.Dat, F, "summary_date_epoch")

    if summary_date_epoch is not None:
        Dthis = Dthis.filterPandas({"summary_date_epoch":[summary_date_epoch]}, "dataset")
    return Dthis, DictDateEarlyLate, DictDay, DictDateInds

def extract_primlist(D):
    """ helper to pull out list of prim names (strings)
    """
    prim_list = D.Dat["character"].unique()
    return prim_list

def extract_list_features(D, PRIM):
    list_sx = np.unique(D.Dat[D.Dat["character"]==PRIM]["sx"].to_list())
    list_sy = np.unique(D.Dat[D.Dat["character"]==PRIM]["sy"].to_list())
    list_theta = np.unique(D.Dat[D.Dat["character"]==PRIM]["theta"].to_list())
    return list_sx, list_sy, list_theta


def extract_primtuple_list(D, Nmin = 8, list_prims_exclude = ("dot", "line",)):
    """ 
    GET list of primtuples that have more than N trials over all days.
    primtuple is (prim, sx, sy, theta)
    """
    
    from itertools import product

    prim_list = extract_primlist(D)

    # 1) collect list of primitives
    list_primtuple = []
    for PRIM in prim_list:
        list_sx, list_sy, list_theta = extract_list_features(D, PRIM)

        # Dprim = D.filterPandas({"character":[PRIM]}, "dataset")
        # list_sx = np.unique(Dprim.Dat["sx"])
        # list_sy = np.unique(Dprim.Dat["sy"])
        # list_theta = np.unique(Dprim.Dat["theta"])

        # For each combo of params, plot sample size
        for sx, sy, th in product(list_sx, list_sy, list_theta):
            DprimThis = D.filterPandas({"character":[PRIM], "sx":[sx], "sy":[sy], "theta":[th]}, "dataset")

            n =  len(DprimThis.Dat)
            if n>Nmin:
                primtuple = (PRIM, sx, sy, th)
                list_primtuple.append(primtuple)
                
    for prim_exclude in list_prims_exclude:
        list_primtuple = [this for this in list_primtuple if f"{prim_exclude}-" not in this[0]]

    # Collect list of days
    list_days = D.Dat["date"].unique()

    return list_primtuple, list_days
