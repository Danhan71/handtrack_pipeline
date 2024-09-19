""" e..g, each datapoint is a trial, separate by task kind, etc. 
Also plot example trial behaviors.
Also integrate with task model.
e..g, see notebook analysis_mem2
"""

# import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plotEachUniqueTask(DF, PROBEDAT, valstoplot, SAVEDIR):
    # ***************************************
    # [one plot for each unqiue task]
    # === FIRST: THROW OUT RANDOM TASKS
    DF = DF[DF["random_task"]==False]
    if len(DF)<0:
        print("SKIPPING plotEachUniqueTask, since no non-random tasks")
        return
    
    # ===== get dict of {taskgroup:[list of tasks]}
    tasklist = set(DF["task_stagecategory"])
    taskgrouplist = set(DF["taskgroup"])

    taskgroupdict = {k:[] for k in set([P["taskgroup"] for P in PROBEDAT])}
    for P in PROBEDAT:
        taskgroupdict[P["taskgroup"]].append(P["task_stagecategory"])

    for k, v in taskgroupdict.items():
        taskgroupdict[k] = sorted(set(v))


    # === one figure per taskgroup.
    for taskgroup, tasklist in taskgroupdict.items():
        for task in tasklist:
            DFthis = DF[(DF["taskgroup"]==taskgroup) & (DF["task_stagecategory"]==task)]
            if len(DFthis)>0:
                for y in valstoplot:
                    ALPHA=0.4
                    if y in ["modelcomp", "modelcompflipped"]:
                        YLIM=(-0.7, 0.7)
                    elif y=="modelcomp_offline":
                        YLIM=(-1,1)
                    elif y=="hausdorff_positive":
                        YLIM=np.percentile(DF["hausdorff_positive"], q=[0.5, 99.5])
                    elif y=="nstrokesactual":
                        YLIM=np.percentile(DF["nstrokesactual"], q=[0.5, 99.5])
                        YLIM = [YLIM[0]-0.5, YLIM[1]+0.5]
                        ALPHA=0.15
                    else:
                        YLIM = None

                    g = sns.FacetGrid(DFthis, col="unique_task_name", col_wrap=5, height=4, aspect=2, 
                                      sharex=True, sharey=True, ylim=YLIM)
                    g.map(sns.lineplot, "tvalday", y, **{"err_style":"bars", "ci":68, "color":"k", "linewidth":2})
                    g.map(sns.scatterplot, "tval", y, "epoch", **{"marker":"x", 
                                                                  "alpha":ALPHA,
                                                                          "s":40, 
                                                                          "palette":{1:"r", 2:"b"}})
                    g.map(plt.axhline, **{"color":[0.7, 0.7, 0.7]})
                    g.savefig(f"{SAVEDIR}/timecourses-{taskgroup}-{task}-{y}-1.pdf")

                    # === GROUP BY BLOKK
                    if False:
                        g = sns.FacetGrid(DFthis, col="unique_task_name", col_wrap=5, height=4, aspect=2, 
                                          sharex=True, sharey=True, ylim=YLIM)
                        g.map(sns.lineplot, "blokk_across_days", y, "block", **{"err_style":"bars", "ci":68, "color":"k", "linewidth":2})
                        g.map(sns.scatterplot, "blokk_across_days", y, "block", **{"marker":"x", 
                                                                      "alpha":ALPHA,
                                                                              "s":40})
                        g.map(plt.axhline, **{"color":[0.7, 0.7, 0.7]})
                        g.savefig(f"{SAVEDIR}/timecourses-{taskgroup}-{task}-{y}-groupbyblokk-1.pdf")
                    else:
                        ax = sns.relplot(data=DFthis, x = "blokk_across_days", y=y, hue="block", col="unique_task_name", 
                            col_wrap=5, height=4, aspect=2, kind="scatter", **{"marker":"x", 
                                                                      "alpha":ALPHA,
                                                                              "s":40})
                        ax.savefig(f"{SAVEDIR}/timecourses-{taskgroup}-{task}-{y}-groupbyblokk-1.pdf")

                    # === summary
                    DFsummary = DFthis[DFthis["keepforsummary"]==True]
                    if len(DFsummary)>0:
                        g = sns.FacetGrid(DFsummary, col="unique_task_name", col_wrap=5, height=4, aspect=1.2, 
                          sharey=True, ylim=YLIM)
                        g.map(sns.swarmplot, "epoch", y, "epoch", **{"alpha":ALPHA,
                                                                              "s":4, 
                                                                              "palette":{1:"r", 2:"b"}})
                        g.map(sns.pointplot, "epoch", y, **{"err_style":"bars", "ci":68, "color":"k", "linewidth":1})
                        g.map(plt.axhline, **{"color":[0.7, 0.7, 0.7]})
                        g.savefig(f"{SAVEDIR}/timecourses-{taskgroup}-{task}-{y}-summary.pdf")

                        # ==== GROUP BY BLOCK
                        if True:
                            g = sns.FacetGrid(DFsummary, col="unique_task_name", col_wrap=5, height=4, aspect=1.2, 
                              sharey=True, ylim=YLIM)
                            g.map(sns.swarmplot, "block", y, "epoch", **{"alpha":ALPHA,
                                                                                  "s":4, 
                                                                                  "palette":{1:"r", 2:"b"}})
                            g.map(sns.pointplot, "block", y, **{"err_style":"bars", "ci":68, "color":"k", "linewidth":1})
                            g.map(plt.axhline, **{"color":[0.7, 0.7, 0.7]})
                            g.savefig(f"{SAVEDIR}/timecourses-{taskgroup}-{task}-{y}-summary-groupbyblokk.pdf")
                        else:
                            ax = sns.catplot(data = DFsummary, x="block", y=y, hue="epoch", col="unique_task_name", 
                                col_wrap=5, height=4, aspect=1.2, sharey=True, ylim=YLIM, kind="point", **{"alpha":ALPHA,
                                                                                  "s":4})
                            ax.savefig(f"{SAVEDIR}/timecourses-{taskgroup}-{task}-{y}-summary-groupbyblokk.pdf")
            
def plotEachUniqueTaskGrouped(DF,  valstoplot, SAVEDIR):
    # === ONE LINE PER UNIQUE TASK
    from pythonlib.tools.pandastools import aggregGeneral
    from pythonlib.tools.pandastools import filterGroupsSoNoGapsInData
    
    DF = DF[DF["keepforsummary"]==True]

    # aggregate over unique tasks
    values = valstoplot
    # DFsummary = aggregGeneral(DF, ["unique_task_name", "epoch", "taskgroup"], values, nonnumercols=["task_stagecategory"], aggmethod=["median"])
    # DFsummaryBlock = aggregGeneral(DF, ["unique_task_name", "epoch", "block", "taskgroup"], values, nonnumercols=["task_stagecategory"], aggmethod=["median"])
    DFsummary = aggregGeneral(DF, ["unique_task_name", "epoch", "taskgroup"], values, nonnumercols=["task_stagecategory"], aggmethod=["mean"])
    DFsummaryBlock = aggregGeneral(DF, ["unique_task_name", "epoch", "block", "taskgroup"], values, nonnumercols=["task_stagecategory"], aggmethod=["mean"])

    # == only keep cases that have data for all epochs.
    values_to_check = list(set(DF["epoch"].values))
    # values_to_check = [1,2] old version, more general is to do the above.
    colname = "epoch"
    group = "unique_task_name"
    DFsummary = filterGroupsSoNoGapsInData(DFsummary, group, colname, values_to_check)
    DFsummaryBlock = filterGroupsSoNoGapsInData(DFsummaryBlock, group, colname, values_to_check)

    if len(DFsummaryBlock)>0:
        # === PLOT
        for y in values:
            from pythonlib.tools.snstools import relplotOverlaid
            fig = relplotOverlaid(DFsummaryBlock, "unique_task_name", "k",
                           data=DFsummaryBlock, x="block", y=y, col="taskgroup", row="task_stagecategory", 
                    hue="unique_task_name", kind="line")
            fig.savefig(f"{SAVEDIR}/summarypaired-{y}-groupbyblock-1.pdf")


            # === separate by blokks
            fig = sns.catplot(data=DFsummaryBlock, x="block", y=y, col="taskgroup", row="task_stagecategory",  
                kind="point")
            fig.savefig(f"{SAVEDIR}/summarypaired-{y}-groupbyblock-2.pdf")

    if len(DFsummary)>0:
        # === PLOT
        for y in values:
            from pythonlib.tools.snstools import relplotOverlaid
            fig = relplotOverlaid(DFsummary, "unique_task_name", "k",
                           data=DFsummary, x="epoch", y=y, col="taskgroup", row="task_stagecategory", 
                    hue="unique_task_name", kind="line")
            fig.savefig(f"{SAVEDIR}/summarypaired-{y}-1.pdf")

            fig = sns.catplot(data=DFsummary, x="epoch", y=y, col="taskgroup", row="task_stagecategory",  kind="point")
            fig.savefig(f"{SAVEDIR}/summarypaired-{y}-2.pdf")
    else:
        print("Skipping")



