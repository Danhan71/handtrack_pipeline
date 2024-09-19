""" Plotting raw behavior, organized by experiemnt structure, related to 
plotsTaskScore, but there is summary stats. Here is thinks like example trials, etc.
This taken from analysis_lines5redo_020421 on 2/25/21.
"""

debug=False
from .modelexpt import *
#from .modelexpt import loadMultDataForExpt, loadProbeDatWrapper
from .line2 import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tools.plots import plotTrial2dGrid
from .probedatTaskmodel import ProbedatTaskmodel

def plotalltrials(PROBEDATthis, fdlist, stage, tasklist, SAVEDIR, datelist=None, blocklist=None, ver="bydate"):
    tasklist = sorted(set([P["unique_task_name"] for P in PROBEDATthis if P["task_stagecategory"]==stage]))
    if len(tasklist)>100:
        assert False, "why so many tasks for this stage?"

    # -- Plot 2d grid, separated by days
    if ver=="bydate":
        assert datelist is not None
        CAT1 = ["date", datelist]
    elif ver=="byblock":
        assert blocklist is not None
        CAT1 = ["block", blocklist]
    else:
        print(ver)
        assert False, "not coded"
    CAT2 = ["unique_task_name", tasklist]
    fdlist = None

    # == 1) All behavior trials, all strokes overlaid
    plotargs = {"zoom":True, "plotver":"order", "markersize":4, "alpha":0.25}
    # fdlist = [P["filedata"][0]() for P in PROBEDATthis]
    # fdlist = [FD[P["ii"]]["fd"] for P in PROBEDATthis]
    
    fig = plotTrial2dGrid(PROBEDATthis, fdlist = fdlist, cat1 = CAT1, cat2 = CAT2, ver="beh", plotargs=plotargs);
    fig.savefig(f"{SAVEDIR}/datebycategory_beh-{stage}-{ver}.pdf")
    fig = plotTrial2dGrid(PROBEDATthis, fdlist = fdlist, cat1 = CAT1, cat2 = CAT2, ver="task", plotargs=plotargs);
    fig.savefig(f"{SAVEDIR}/datebycategory_task-{stage}-{ver}.pdf")

    # == 2) Split by stroke number (one plot for each stroke)
    if False:
        plotargs = {"zoom":True, "plotver":"order", "markersize":3, "alpha":0.2}
        strokenums_to_plot_alone=list(range(max_strokenum+1))
        overlay_stroke_mean=False

        fig = plotTrial2dGrid(PROBEDATthis, fdlist = fdlist, cat1 = CAT1, cat2 = CAT2, ver="behtask", 
                              plotargs=plotargs, strokenums_to_plot_alone=strokenums_to_plot_alone, 
                             overlay_stroke_mean = overlay_stroke_mean);
        fig.savefig(f"{SAVEDIR}/alltrials_datebycategory_eachstroke_beh_{stage}.pdf")


    # == 3) All strokes (faint) and overlay average
    if False:
        plotargs = {"zoom":True, "plotver":"order", "markersize":2, "alpha":0.15}

        overlay_stroke_mean=True
        fig = plotTrial2dGrid(PROBEDATthis, fdlist = fdlist, cat1 = CAT1, cat2 = CAT2, ver="beh", 
                              plotargs=plotargs, overlay_stroke_mean=overlay_stroke_mean);
        fig.savefig(f"{SAVEDIR}/alltrials_datebycategory_strokemeans_{stage}.pdf")



def plotalltrials_good(Probedat, SAVEDIR, ver="bydate", 
    taskgrouplist = ["test_fixed", "G2", "G3", "G4"], random_tasklist = [False]):
    """
    - Probedat, class object.
    NOTE: updates rleative to old:
    - iterates over all stages automatically.
    - flexible filtering.
    - finds tasklist here automatilcaly. 
    """
    P = Probedat
    stagelist = P.pandas()["task_stagecategory"].unique()
    datelist = P.pandas()["date"].unique()
    blocklist = P.pandas()["block"].unique()

    if len(taskgrouplist)<3:
        taskgroupsave = "_".join(taskgrouplist)
    elif "G2" or "G3" or "test_fixed" in taskgrouplist:
        taskgroupsave = "test"
    elif "train_fixed" in taskgrouplist:
        taskgroupsave = "train"
    else:
        print(taskgrouplist)
        assert False, "what task groups?"


    for stage in stagelist:
        filtdict = {
            "random_task":random_tasklist,
            "taskgroup":taskgrouplist,
            "task_stagecategory":[stage]
        }

        idxs = P.filterPandas(filtdict, return_indices=True)
        Pthis = P.subsampleProbedat(idxs)

        if len(Pthis.Probedat)==0:
            continue

        # - get unique tasks
        tasklist = Pthis.pandas()["unique_task_name"].unique()

        # -- Plot 2d grid, separated by days
        if ver=="bydate":
            assert datelist is not None
            CAT1 = ["date", datelist]
        elif ver=="byblock":
            assert blocklist is not None
            CAT1 = ["block", blocklist]
        else:
            print(ver)
            assert False, "not coded"
        CAT2 = ["unique_task_name", tasklist]

        # == 1) All behavior trials, all strokes overlaid
        plotargs = {"zoom":True, "plotver":"order", "markersize":4, "alpha":0.25}

        fig = plotTrial2dGrid(Pthis.Probedat, fdlist = None, cat1 = CAT1, cat2 = CAT2, ver="beh", plotargs=plotargs);
        fig.savefig(f"{SAVEDIR}/datebycategory_beh_good-{stage}-{ver}-{taskgroupsave}.pdf")
        fig = plotTrial2dGrid(Pthis.Probedat, fdlist = None, cat1 = CAT1, cat2 = CAT2, ver="task", plotargs=plotargs);
        fig.savefig(f"{SAVEDIR}/datebycategory_task_good-{stage}-{ver}-{taskgroupsave}.pdf")


def plotBehaviors(expt, animal, thingstoplot, debug, whichdates="all"):
    FD, MD = loadMultDataForExpt(expt, animal, whichdates=whichdates, metadatonly=False)
    PROBEDAT = loadProbeDatWrapper(FD, MD, getnumstrokes=True)

    # === NEW, MORE FLEXIBLE, so can choose just test tasks, for example.
    if "alltrials_good" in thingstoplot:
        # saving dir
        sdate = MD["sdate"]
        edate = MD["edate"]
        SAVEDIR = f"{FD[0]['fd']['params']['figuredir_notebook']}/analysis_modelexpt_multsession/{expt}/multday-{animal}_{sdate}_to_{edate}/alltrials"
        import os
        os.makedirs(SAVEDIR, exist_ok=True)
        print(f"saving at {SAVEDIR}")

        P = ProbedatTaskmodel(PROBEDAT)
        plotalltrials_good(P, SAVEDIR, ver="bydate", 
            taskgrouplist = ["test_fixed", "G1", "G2", "G3", "G4"], random_tasklist = [False])


    # ==== OLD, 

    # get all fixed tasks of a particular kind
    kindlist = set([P["kind"] for P in PROBEDAT if P["random_task"]==False])
    task_per_kind = {}
    for kind in kindlist:
        tasklist = set([P["unique_task_name"] for P in PROBEDAT if P["kind"]==kind])
        task_per_kind[kind]=sorted(tasklist)

    if len(task_per_kind.keys())==0:
        print("*** DID NOT FIND ANY TASKS - likely they are all random... Only coded for fixed tasks")
        print(f"Skipping {animal},  {expt}")
        return
    else:
        print("tasks per kind found")
        for k, v in task_per_kind.items():
            print("----")
            print(f"=={k}")
            [print(vv) for vv in v]

    ## PLOT - all trials, 2d grid sorted by date and task category
    # only keep data for fixed tasks.
    PROBEDATthis = [P for P in PROBEDAT if P["random_task"]==False]
    datelist = sorted(set([P["date"] for P in PROBEDATthis]))
    blocklist = sorted(set([P["block"] for P in PROBEDATthis]))
    fdlist = None
    
    # for each stage, make a 2d grid plot (date x task)
    if "alltrials" in thingstoplot:
        # saving dir
        sdate = MD["sdate"]
        edate = MD["edate"]
        SAVEDIR = f"{FD[0]['fd']['params']['figuredir_notebook']}/analysis_modelexpt_multsession/{expt}/multday-{animal}_{sdate}_to_{edate}/alltrials"
        import os
        os.makedirs(SAVEDIR, exist_ok=True)
        print(f"saving at {SAVEDIR}")
    
        # all categories that have fixed tasks
        stagelist = set([P["task_stagecategory"] for P in PROBEDATthis if P["random_task"]==False])
        for stage in stagelist:
            plotalltrials(PROBEDATthis, fdlist, stage, tasklist, SAVEDIR, datelist=datelist, ver="bydate")
            plotalltrials(PROBEDATthis, fdlist, stage, tasklist, SAVEDIR, blocklist=blocklist, ver="byblock")
        

    # ==== PLOT ALL TRIALS
    if "egtrials" in thingstoplot:
        
        # saving dir
        sdate = MD["sdate"]
        edate = MD["edate"]
        SAVEDIR = f"{FD[0]['fd']['params']['figuredir_notebook']}/analysis_modelexpt_multsession/{expt}/multday_{animal}_{sdate}_to_{edate}/egtrials"
        import os
        os.makedirs(SAVEDIR, exist_ok=True)
        print(f"saving at {SAVEDIR}")

        if debug:
            # making plots for lab meeting..
            tasklist = set([P["unique_task_name"] for P in PROBEDATthis if P["random_task"]==False
               and (P["task_stagecategory"] in ["LplusL", "2linePlusL", "3linePlusL"] or "linePlusLv2_51" in P["unique_task_name"])])
        else:
            tasklist = set([P["unique_task_name"] for P in PROBEDATthis if P["random_task"]==False])
        NMAX = 20 # trials to plot, starting from 1st trial int he day


        for task in tasklist:

            PD = [P for P in PROBEDATthis if P["random_task"]==False and P["unique_task_name"]==task]
            
            # only do reverse if more trials than will plot. otherwise is redundant.
            PD, countlist = probeDatIndexWithinDay(PD, task, reverse_order=False)
            if max(countlist)>NMAX:
                rev = [False, True]
            else:
                rev= [False]
            
            for reverse in rev:
                # -- task presentation num as column
                for P in PD:
                    P["idx_today_uniquetask"] = None
                PD, countlist = probeDatIndexWithinDay(PD, task, reverse_order=reverse);

                # -- how many examples to plot?
                ntoplot = min((max(countlist), NMAX))

                # -- Plot 2d grid, separated by days
                CAT1 = ["date", datelist]
                CAT2 = ["idx_today_uniquetask", range(ntoplot)]
                fdlist = None

                # == 1) All behavior trials, all strokes overlaid
                plotargs = {"zoom":True, "plotver":"order", "markersize":8, "alpha":0.7}
                plot_task_last_col = True
                ver = "beh"

                fig = plotTrial2dGrid(PD, fdlist = fdlist, cat1 = CAT1, cat2 = CAT2, ver=ver, 
                                      plotargs=plotargs, plot_task_last_col=plot_task_last_col);
                if reverse:
                    fig.savefig(f"{SAVEDIR}/datebyexample_revchronorder_{task}.pdf")
                else:
                    fig.savefig(f"{SAVEDIR}/datebyexample_chronorder_{task}.pdf")

        plt.close("all")

    # ==== PLOT ALL TRIALS
    if "egtrials_byblock" in thingstoplot:
        tasklist = set([P["unique_task_name"] for P in PROBEDATthis if P["random_task"]==False])
        datelist = set([P["date"] for P in PROBEDATthis if P["random_task"]==False])
        NMAX = 20 # trials to plot, starting from 1st trial int he day
        reverse = False

        # saving dir
        sdate = MD["sdate"]
        edate = MD["edate"]
        SAVEDIR = f"{FD[0]['fd']['params']['figuredir_notebook']}/analysis_modelexpt_multsession/{expt}/multday_{animal}_{sdate}_to_{edate}/egtrials_byblock"
        import os
        os.makedirs(SAVEDIR, exist_ok=True)
        print(f"saving at {SAVEDIR}")

#         OLD VERSION - one plot per day.
#         for task in tasklist:
#             for date in datelist:
#                 PD = [P for P in PROBEDATthis if P["random_task"]==False and P["unique_task_name"]==task
#                      and P["date"]==date]
                
#                 # -- task presentation num as column
#                 for P in PD:
#                     P["idx_today_uniquetask"] = None
#                 PD, countlist = probeDatIndexWithinDay(PD, task, reverse_order=reverse);

#                 # -- new column, which is date-block as string
#                 for P in PD:
#                     P["date-block"] = [f"{P['date']}-{P['block']}" for P in PD]
                
#                 # -- how many examples to plot?
#                 ntoplot = min((max(countlist), NMAX))

#                 # -- Plot 2d grid, separated by days
#                 blocklist = sorted(set([P["block"] for P in PD]))
#                 CAT1 = ["block", blocklist]
#                 CAT2 = ["idx_today_uniquetask", range(ntoplot)]
#                 fdlist = None

#                 # == 1) All behavior trials, all strokes overlaid
#                 plotargs = {"zoom":True, "plotver":"order", "markersize":8, "alpha":0.7}
#                 plot_task_last_col = True
#                 ver = "beh"

#                 fig = plotTrial2dGrid(PD, fdlist = fdlist, cat1 = CAT1, cat2 = CAT2, ver=ver, 
#                                       plotargs=plotargs, plot_task_last_col=plot_task_last_col);
#                 fig.savefig(f"{SAVEDIR}/egtrials_datebyexample_chronorder-{task}-{date}.pdf")
#         assert False

#         plt.close("all")

        for task in tasklist:
            PD = [P for P in PROBEDATthis if P["random_task"]==False and P["unique_task_name"]==task]

            # -- task presentation num as column
            for P in PD:
                P["idx_today_uniquetask"] = None
            PD, countlist = probeDatIndexWithinDay(PD, task, reverse_order=reverse);

            # -- new column, which is date-block as string
            for P in PD:
                P["date_block"] = f"{P['date']}-{P['block']}"
            date_block_list = sorted(set([P["date_block"] for P in PD]))

            # -- how many examples to plot?
            ntoplot = min((max(countlist), NMAX))

            # -- Plot 2d grid, separated by days
            blocklist = sorted(set([P["block"] for P in PD]))
            CAT1 = ["date_block", date_block_list]
            CAT2 = ["idx_today_uniquetask", range(ntoplot)]
            fdlist = None

            # == 1) All behavior trials, all strokes overlaid
            plotargs = {"zoom":True, "plotver":"order", "markersize":8, "alpha":0.7}
            plot_task_last_col = True
            ver = "beh"

            fig = plotTrial2dGrid(PD, fdlist = fdlist, cat1 = CAT1, cat2 = CAT2, ver=ver, 
                                  plotargs=plotargs, plot_task_last_col=plot_task_last_col, 
                                 clean=True);
            fig.savefig(f"{SAVEDIR}/datebyexample_chronorder-{task}.pdf")

            plt.close("all")    