""" single day analsysi - try to make it work for any day regardless of features for that day"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
from tools.plots import *
from analysis.probedatTaskmodel import *

sys.path.append("/home/lucast4")


def goodfig(ax, df, blockver="blokk"):
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    """
     gives trial where things change - i.e., bloque_onsets are 
    trials where first trial for a new bloque.
    FIXED on 9/29/20
    was using before index instead of trials, so plotting bloque onsets too early.
    
    """
    # overlay info
    # sns.lineplot(data=df, x="trial", y="bloque")
    # bloque_onsets = np.argwhere(np.diff(df["bloque"].values))+1
    # bloque_onsets = []
    # bloque_nums = []
    # for i in np.argwhere(DF["bloque"].diff()):
    #     bloque_onsets.append(DF.loc[i]["trial"].values[0])
    #     bloque_nums.append(DF.loc[i]["bloque"].values[0])
    # bloque_nums = df["bloque"].values[bloque_onsets]
    # bloque_onsets = np.insert(bloque_onsets, 0, 0)
    # bloque_nums = np.insert(bloque_nums, 0, 1)
    # block_nums = df["block"].values[bloque_onsets]
    idx_of_bloque_onsets = []
    for i in np.argwhere(df[blockver].diff().values):
        idx_of_bloque_onsets.append(i[0])
    bloque_onsets = df["trial"].values[idx_of_bloque_onsets]
    # bloque_nums = df["bloque"].values[idx_of_bloque_onsets]
    blokk_nums = df["blokk"].values[idx_of_bloque_onsets]
    block_nums = df["block"].values[idx_of_bloque_onsets]

    # blocks
    # block_nums = [getTrialsBlock(fd, t) for t in bloque_onsets]
    for b, x,y in zip(block_nums, bloque_onsets, blokk_nums):
        ax.axvline(x)
        ax.text(x, ax.get_ylim()[1], f"k{b}\nkk{y}\nt{x}", size=10)
    ax.grid(True)

def plotOverview_(df, smwin=20, featurestoplot=tuple(["hausdorff"])):
    """
    - smwin, in trials, for smoothing (after throwing out nan trials.)
    """

    # === get rolling mean df
    cols_keep = list(set(["trial", "reward", "behscore", "binaryscore", "biasscore", "num_replays", "hausdorff", "behscore"] + featurestoplot))
    dfrolling = df[~np.isnan(df["reward"].values)][cols_keep].rolling(window=smwin, center=True).median()

    # === STATS OVER TRIALS.
    # stats_to_plot = (hausdorff, behscore)
    NCOLS = 6+len(featurestoplot)
    NROWS = 1

    fig, axes = plt.subplots(NCOLS, NROWS, sharex=True, figsize=(NROWS*15, NCOLS*5), squeeze=False)

    # 1) Score over trials
    ax = axes[0,0]
    if "hausdorff" in featurestoplot:
        f = "hausdorff"
    else:
        f = "behscore"
    ax = sns.scatterplot(ax=ax, data=df, x="trial", y=f, hue="trial_end_method")
    try:
        ax = sns.lineplot(ax=ax, data=dfrolling, x="trial", y=f, color="k")
    except Exception as err:
        # Giving this error KeyError: 'y', 240503 -- just igonre..
        pass
    goodfig(ax, df)

    # ax = axes[1,0]
    ct = 0
    for f in featurestoplot:
        if f!="hausdorff":
            ct+=1
            ax = axes[ct, 0]
            try:
                ax = sns.scatterplot(ax=ax, data=df, x="trial", y=f, label=f)
                ax = sns.lineplot(ax=ax, data=dfrolling, x="trial", y=f, label=f)
                # ax = sns.lineplot(data=df, x="trial", y="hausdorff", hue="bloque", style="trial_end_method")
                goodfig(ax, df)
            except Exception as err:
                # Giving this error KeyError: 'y', 240503 -- just igonre..
                pass
                # print(dfrolling.columns)
                # print(dfrolling[:5])
                # print(f)
                # raise err

    # 2) Overlay reward
    ax = axes[ct+1,0]
    try:
        ax = sns.scatterplot(ax=ax, data=df, x="trial", y="reward", label="reward")
        ax = sns.lineplot(ax=ax, data=dfrolling, x="trial", y="reward", color="k")
        ax = sns.lineplot(ax=ax, data=df, x="trial", y="reward_max", label="reward_max")
        goodfig(ax, df)
    except Exception as err:
        # Giving this error KeyError: 'y', 240503 -- just igonre..
        pass    

    # 3) Overlay score factors
    ax = axes[ct+2,0]
    try:
        ax = sns.scatterplot(ax=ax, data=df, x="trial", y="behscore", label="behscore")
        ax = sns.scatterplot(ax=ax, data=df, x="trial", y="binaryscore", label="binaryscore")
        ax = sns.scatterplot(ax=ax, data=df, x="trial", y="biasscore", label="biasscore")
        ax = sns.lineplot(ax=ax, data=dfrolling, x="trial", y="behscore", label="behscore")
        ax = sns.lineplot(ax=ax, data=dfrolling, x="trial", y="binaryscore", label="binaryscore")
        ax = sns.lineplot(ax=ax, data=dfrolling, x="trial", y="biasscore", label="biasscore")
        # ax = sns.lineplot(data=df, x="trial", y="hausdorff", hue="bloque", style="trial_end_method")
        goodfig(ax, df)
    except Exception as err:
        # Giving this error KeyError: 'y', 240503 -- just igonre..
        pass

    # 3) Overlay nreplays factors
    ax = axes[ct+3,0]
    try:
        ax = sns.scatterplot(ax=ax, data=df, x="trial", y="num_replays", label="num_replays")
        ax = sns.lineplot(ax=ax, data=dfrolling, x="trial", y="num_replays")
        # ax = sns.scatterplot(ax=ax, data=df, x="trial", y="binaryscore", label="binaryscore")
        # ax = sns.scatterplot(ax=ax, data=df, x="trial", y="biasscore", label="biasscore")
        # ax = sns.lineplot(data=df, x="trial", y="hausdorff", hue="bloque", style="trial_end_method")
        goodfig(ax, df)
    except Exception as err:
        # Giving this error KeyError: 'y', 240503 -- just igonre..
        pass
    
    # fade values
    MIN_ALPHA_VISIBLE = 0.021 # any alpha less than this is invisible (empriically)
    ax = axes[ct+4,0]
    for y in ("fade_samp1", "fade_guide1task", "fade_guide1fix"):
        sns.scatterplot(ax=ax, x = "trial", y=y, data=df, label=y)
    # which trials fully faded?  
    goodfig(ax, df)
    
    dfthis = df[(df["fade_samp1"].values<MIN_ALPHA_VISIBLE) & (df["fade_guide1task"].values<MIN_ALPHA_VISIBLE) & (df["fade_guide1fix"].values<MIN_ALPHA_VISIBLE)]
    plt.plot(dfthis["trial"].values, np.ones_like(dfthis["trial"].values)*(0.02), "xk", alpha=0.3)
    
    fig1 = fig
        


    # === STATS OVER BLOQUES
    NCOLS = 5+len(featurestoplot)
    NROWS = 1

    fig, axes = plt.subplots(NCOLS, NROWS, sharex=True, figsize=(NROWS*15, NCOLS*5), squeeze=False)

    try:
        # 2) Score over bloques, taking mean
        ax = axes[0,0]
        # ax = sns.lineplot(ax=ax, x = "bloque", y="hausdorff", data=df, err_style="bars")
        ax = sns.pointplot(ax=ax, x = "bloque", y="hausdorff", hue="block", data=df)

        # 2) Score over bloques, taking mean
        ax = axes[1,0]
        # ax = sns.lineplot(ax=ax, x = "bloque", y="reward", data=df, err_style="bars")
        ax = sns.pointplot(ax=ax, x = "bloque", y="reward", hue="block", data=df)

        # 2) Score over bloques, taking mean
        ax = axes[2,0]
        # ax = sns.lineplot(ax=ax, x = "bloque", y="behscore", data=df, err_style="bars", label="behscore")
        # ax = sns.lineplot(ax=ax, x = "bloque", y="binaryscore", data=df, err_style="bars", label="binaryscore")
        # ax = sns.lineplot(ax=ax, x = "bloque", y="biasscore", data=df, err_style="bars", label="biasscore")
        ax = sns.pointplot(ax=ax, x = "bloque", y="behscore", hue="block", data=df, err_style="bars", label="behscore")

        ax = axes[3,0]
        ax = sns.pointplot(ax=ax, x = "bloque", y="binaryscore", hue="block", data=df, err_style="bars", label="binaryscore")

        ax = axes[4,0]
        ax = sns.pointplot(ax=ax, x = "bloque", y="biasscore", hue="block", data=df, err_style="bars", label="biasscore")

        ct = 4
        try:
            for f in featurestoplot:
                if f!="hausdorff":
                    ct+=1
                    ax = axes[ct, 0]
                    ax = sns.pointplot(ax=ax, x = "bloque", y=f, hue="block", data=df)
        except:
            pass
        # ax = axes[3,0]
        # ax = sns.violinplot(ax=ax, x = "bloque", y="behscore", data=df)
        # ax = sns.lineplot(ax=ax, x = "bloque", y="binaryscore", data=df, err_style="bars", label="binaryscore")
        # ax = sns.lineplot(ax=ax, x = "bloque", y="biasscore", data=df, err_style="bars", label="biasscore")
        plt.xlabel("bloque")
    except ValueError:
        ax.text(0,0, "skip, sns error, too few trials, becuase of hue=block, they are confounded with bloque")
        # ax.text(0,0, "this becuase of hue=block, they are confounded with bloque, so sns throws error")
    except TypeError:
        # TypeError: pointplot() got an unexpected keyword argument 'err_style'
        pass
    fig2 = fig


    # === FOR EACH BLOCK, PROGRESSION OVER BLOQUES
    try:
        fig3 = sns.catplot(x = "bloque", y="hausdorff", row="block", data=df, kind="point", height=5, aspect=3)
        fig4 = sns.catplot(x = "bloque", y="behscore", row="block", data=df, kind="point", height=5, aspect=3)
    except ValueError:
        # ran into issue with Diego-230212, 1017 trials made figure too big...
        # 'ValueError: ImageSize is too large'
        fig3 = sns.catplot(x = "bloque", y="hausdorff", row="block", data=df[0:len(df)/2], kind="point", height=5, aspect=3)
        fig4 = sns.catplot(x = "bloque", y="behscore", row="block", data=df[0:len(df)/2], kind="point", height=5, aspect=3)
    
    return fig1, fig2, fig3, fig4



def plotReward(df, featurestoplot=None):
    """ look at factors contributing to reward magnitude"""
    figs = []

    if featurestoplot is None:
        featurestoplot = ["hausdorff"]

    axes = sns.pairplot(data=df, vars=["reward","behscore", "biasscore", "binaryscore"], height=5)
    figs.append(axes)

    for f in featurestoplot:
        try:
            axes = sns.pairplot(data=df, vars=["reward","behscore", f], height=5, hue="bloque")
            figs.append(axes)

            axes = sns.pairplot(data=df, vars=["reward","behscore", f], height=5)
            figs.append(axes)
        except:
            pass

    return figs


def plotBehSortedByScore(df, filedata, score_type="hausdorff", N=20, 
        list_ver = ("percentiles", "bottomN", "midN", "topN")):
    """ 
    """
    def _findValues(dfthis, ver, score_type_, N=20):
        """ gets array of values, which can then use to 
        index the dfthis
        - Returns None if all values are nan
        """
        values = dfthis[score_type_].values
        
        if all(np.isnan(values)):
            return None

        values = values[~np.isnan(values)]
        values = sorted(values)
        if ver=="percentiles":
            # get trials that match percentiles.
            p = np.linspace(0, 100, N) # percentiles
            values_selected = list(np.percentile(values, p, interpolation="nearest"))
        elif ver=="topN":
            # get top N
            values_selected = values[-N:]
        elif ver=="midN":
            # get top N
            n = len(values)
            n1 = round(n/2-N/2)
            n2 = n1+N
            values_selected = values[n1:n2]
        elif ver=="bottomN":
            # get top N
            values_selected = values[:N]

        return values_selected

    FIGS = {}
    for ver in list_ver:
        values_selected = _findValues(df, ver, score_type, N=N)
        if values_selected is None:
            print(f"Skipping {score_type}, {ver},  since all values are nan")
            continue

        # df_selected = df.loc[df[score_type].isin(values_selected)]
        df_selected = df[df[score_type].isin(values_selected)]

        # sort by desired score type
        df_selected = df_selected.sort_values(by=score_type)

        if len(df_selected)>N:
            df_selected = df_selected.sample(N)
            df_selected = df_selected.sort_values(by=score_type) # re-sort, since will be random

        # ==== PLOT
        trials_ordered = df_selected["trial"].values
        scores_ordered = df_selected[score_type].values
        rew_ordered = df_selected["reward"].values

        # trials_ordered = [d[1]["trial"] for d in df_selected.iterrows()]
        # scores_ordered = [d[1][score_type] for d in df_selected.iterrows()]
        # rew_ordered = [d[1]["reward"] for d in df_selected.iterrows()]

        titles = [f"{t}-{score_type}:{s:.2f}\nrew:{r}" for t, s, r in zip(trials_ordered, scores_ordered, rew_ordered)]

        # ==== MAKE FIGURES
        FIGS[ver] = []
        print("- plotBehSortedByScore, plotting mult trials")
        fig = plotMultTrialsSimple(filedata, trials_ordered, zoom=True, strokes_ver="peanuts", 
            plot_fix=False,plotver="strokes", titles=titles)
        FIGS[ver].append(fig)

        # =====
        fig = plt.figure()
        v = df[score_type].values
        plt.hist(v)
        for s in scores_ordered:
            plt.axvline(s, color="k")
        FIGS[ver].append(fig)

        # =====
        fig = plt.figure()
        v = df[score_type].values
        r = df["reward"].values
        plt.scatter(v, r)
        plt.xlabel(score_type)
        plt.ylabel("reward")
        plt.title("all trials")
        for s, r in zip(scores_ordered, rew_ordered):
            plt.plot(s, r, 'ok')
        FIGS[ver].append(fig)
    return FIGS


def plotTaskSchedules(df):
    """ for each bloque, plot tasks and how often played
    - only plots if succesfulyl pass fixation"""
    figs = []

    # 1) only keep trials where fixation was made
    dfthis = df[df["trial_end_method"]!="fixation_error"]

    ax = sns.catplot(x = "task_string", kind="count", data=dfthis, hue="bloque", aspect = 3)
    from pythonlib.tools.snstools import rotateLabel
    rotateLabel(ax)
    figs.append(ax)

    # == plot order of task presentations
    dfthis = df[df["trial_end_method"]!="fixation_error"]
    fig = plt.figure(figsize=(15,10))
    ax = sns.scatterplot(x = "trial", y="task_string", data=dfthis, hue="bloque")
    goodfig(ax, df)
    figs.append(fig)

    return figs


def plotProbesSummary(FD, SAVEDIR):
    """ all Probes plots (from dayanalysis_probes_083020 notebook)
    """
    # ==== Flatten all trials across days x animals
    # for each trial collect relevant information
    from analysis.line2 import PROBEDATfromFD
    probedat = PROBEDATfromFD(FD)

    # Make sure clean (no tasks with 0 strokes for example)
    P  = ProbedatTaskmodel(probedat)
    P = P.clean()
    probedat = P.Probedat

    # == for each category, list names of all tasks
    kindlist = set([p["kind"] for p in probedat])
    print("probe kinds -- trials");
    for kind in kindlist:
        print(" ")
        print(kind)
        trials = [p["trial"] for p in probedat if p["kind"]==kind]
        tasknames = [p["unique_task_name"] for p in probedat if p["kind"]==kind]
    #     print(trials)
        print(set(tasknames))
        plt.figure()
        plt.plot(trials, tasknames)

    import seaborn as sns
    import pandas as pd
    dframe = pd.DataFrame(probedat).drop(columns=['filedata'])

    # sns.scatterplot(x="trial", y="taskname", data=dframe, hue="kind")
    fig, axes = plt.subplots(5, 1, sharex=True, squeeze=False, figsize=(28, 10))

    sns.scatterplot(x="trial_day", y="kind", data=dframe, hue="kind", ax=axes[0][0])
    sns.scatterplot(x="trial_day", y="task_stagecategory", data=dframe, hue="kind", ax=axes[1][0])
    sns.scatterplot(x="trial_day", y="task_stagecategory", data=dframe, hue="random_task", ax=axes[2][0])
    sns.lineplot(x="trial_day", y="session", data=dframe, ax=axes[3][0])
    sns.lineplot(x="trial_day", y="block", data=dframe, ax=axes[4][0])

    fig.savefig(f"{SAVEDIR}/overviewAllTrials.pdf")

    stagelist = set([p["task_stagecategory"] for p in probedat])
    sesslist = set([p["session"] for p in probedat])    
    kindlist = set([p["kind"] for p in probedat])
    blocklist = set([p["block"] for p in probedat])
    
    ## PLOT PROBE TASKS OVER THE ENTIRE DAY
    maxtrialsplot = 100

    for kind in kindlist:
        for stage in stagelist:
            for s in sesslist:
                trialsthis = [p["trial"] for p in probedat if p["session"]==s and p["kind"]==kind and p["task_stagecategory"]==stage]
                fd = [F["fd"] for F in FD if F["session"]==s]
#                 print(len(fd))
                assert len(fd)==1
                fd = fd[0]
                
                print(f"[dayanalysis.py] -- for session {s}, kind {kind}, stage {stage}; trials:")
                print(trialsthis)

                if len(trialsthis)>0:

                    nplots = int(np.ceil(len(trialsthis)/maxtrialsplot))

                    for i in range(nplots):
                        inds = range(i*maxtrialsplot, (i+1)*maxtrialsplot)
                        inds = [i for i in inds if i in range(len(trialsthis))]
                        trialsthissub = [trialsthis[idx] for idx in inds]

                        # if len(trialsthis)>maxtrialsplot:
                        #     trialsthis = sorted(random.sample(trialsthis, maxtrialsplot))                        
                        r = None
                        fig1 = plotMultTrialsSimple(fd, trialsthissub, zoom=True, strokes_ver="peanuts", 
                                                    plot_fix=True, rand_subset = r)

                        # save
                        fig1.savefig(f"{SAVEDIR}/trialsByProbeKind_sess{s}-{kind}-{stage}-fig1-sub{i}.pdf")

                        plt.close('all')
                        if False:
                            # since i dont use, and takes too long for large tasks,.
                            scores = [getTrialsScoreRecomputed(fd, t, normalize=True) for t in trialsthis]
                            scores_compos = [getTrialsScoreRecomputed(fd, t, ver="DTW_min", normalize=True) for t in trialsthis]
                            scores_compos2 = [getTrialsScoreRecomputed(fd, t, ver="DTW_min_minus_max", normalize=True) for t in trialsthis]

                            # == plot quick scores
                            fig2 = plt.figure(figsize=(10, 12))

                            plt.subplot(311)
                            plt.title(f"session {s}, kind {kind}, stage {stage}")
                            plt.plot(trialsthis, scores, "ok", label="pts")
                            plt.ylabel("score (pts) (norm HD, high is good)")
                            plt.ylim([-0.2, 1])
                            plt.legend()

                            plt.subplot(312)
                            plt.title(f"session {s}, kind {kind}, stage {stage}")
                            plt.plot(trialsthis, scores_compos, "or", label="compositonal")
                            plt.ylabel("score (compositonal) (norm HD, high is good)")
                            plt.ylim([-0.5, 1])
                            plt.legend()

                            plt.subplot(313)
                            plt.title(f"session {s}, kind {kind}, stage {stage}")
                            plt.plot(trialsthis, scores_compos2, "og", label="compositonal_min_minus_max")
                            plt.ylabel("score (compositonal_min_minus_max) (norm HD, high is good)")
                            plt.legend()

                            fig2.savefig(f"{SAVEDIR}/trialsByProbeKind_sess{s}-{kind}-{stage}-fig2.pdf")
            #                 fig2.savefig(f"{SAVEDIR}/trialsByProbeKind_sess{s}-{kind}-{stage}-fig2.pdf")


                    # ========== separate plot for each block
                    for b in blocklist:

                        trialsthis = [p["trial"] for p in probedat if p["block"]==b and p["session"]==s and p["kind"]==kind and p["task_stagecategory"]==stage]
#                         fd = FD[s]
                        print(f"-- for session {s}, kind {kind}, stage {stage}, block {b}; trials:")
                        print(trialsthis)


                        if len(trialsthis)>0:

                            nplots = int(np.ceil(len(trialsthis)/maxtrialsplot))

                            for i in range(nplots):
                                inds = range(i*maxtrialsplot, (i+1)*maxtrialsplot)
                                inds = [i for i in inds if i in range(len(trialsthis))]
                                trialsthissub = [trialsthis[idx] for idx in inds]
                                r = None
                                # construct titles

                                fig1 = plotMultTrialsSimple(fd, trialsthissub, zoom=True, strokes_ver="peanuts", plot_fix=True,
                                                            rand_subset = r)
                                fig1.savefig(f"{SAVEDIR}/trialsByProbeKind_sess{s}-{kind}-{stage}_block{b}-fig1-sub{i}.pdf")
                            plt.close('all')

        plt.close('all');    