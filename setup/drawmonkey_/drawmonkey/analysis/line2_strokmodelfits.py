
from tools.utils import * 
from tools.plots import *
from tools.analy import *
from tools.calc import *
from tools.analyplot import *
from tools.preprocess import *
from tools.dayanalysis import *
from analysis.line2 import *

from pythonlib.drawmodel.analysis import *
from pythonlib.tools.stroketools import *

from pythonlib.drawmodel.motormodel import *
from pythonlib.drawmodel import primitives as P
from math import pi


def getShuffleBehDistances(filedata, fs, N=200, ploton=False):
    """ goal is to get a baseline of strok-strok distance scores by 
    randomly sampling pairs of trials, and random stroks within trials,
    doing that pairwise comparison, then tallying the scores. this is
    useful, for instance, or figuring out nioramlization if want to 
    weight diff parst of score differently (e.g, spatial vs. velocity score)
    - N, how many shuffles to do.
    """
    # Normalize spatial and velocity cost functions so that on same scale.
    # - get random pairs of strokes and compute distance - use this is scale estimate

    fd = filedata
    triallist = [t for t in getIndsTrials(fd) if getTrialsFixationSuccess(fd, t)]

    # for each round, pick 2 random trials
    # N = 100
    dist_all = []
    n_all =[]
    for _ in range(N):
        t1, t2 = random.sample(triallist, 2)
        
        #  get a random strok
        try:
            strok1 = random.sample(getTrialsStrokesByPeanuts(fd,t1), 1)[0]
            strok2 = random.sample(getTrialsStrokesByPeanuts(fd,t2), 1)[0]
        except ValueError:
            print("skipping getShuffleBehDistances - error")
            continue
        
        # shift strok so start at time 0 (checked, this only mods a copy)
        # and start at 0,0
        strok1 -= strok1[0,:]
        strok2 -= strok2[0,:]
        
        # match their length - prune longers strok
        n = min((len(strok1), len(strok2)))
        strok1 = strok1[:n]
        strok2 = strok2[:n]
        
        if n/fs < 0.2:
            print("skip, too short")
            continue
        # compute distance
        dist = distStrokTimeptsMatched(strok1, strok2, fs=fs, ploton=False, return_separate_scores=True)
        if np.any(np.isnan(dist)):
            dist = distStrokTimeptsMatched(strok1, strok2, fs=fs, ploton=True, return_separate_scores=True)
            print((t1, t2))
            print(n)
        dist_all.append(dist)
        n_all.append(n)
        
    if ploton:
        plt.figure()
        plt.hist(n_all, bins=20)
        plt.figure()
        plt.hist([d[0] for d in dist_all], bins=20)
        plt.figure()
        plt.hist([d[1] for d in dist_all], bins=20)

    vec_over_spatial_ratio = np.median([d[1] for d in dist_all])/np.median([d[0] for d in dist_all])

    print("spatial score should by multiplied by the following, so that it matches scale of vel score")
    print(vec_over_spatial_ratio)
    
    return vec_over_spatial_ratio


def getFitStuff(Nsub, T, fs, theta0=0, dist0=160):
    """ wrapper for model fitting, for returning 
    initialization params.
    - Nsub is how mnay substrokes in each stroke.
    - theta0 and dist0 and initial angles and lengths,
    to help intialization.
    - empirically tested that the parameteres are good 
    for 1, 2, 3, substroke tasks.
    """

    if Nsub==1:
        # 1) Single substrok
        def program_func(params):
            assert len(params)==4
            t1 = params[0]
            t2 = params[1]
            theta1 = params[2]
            l1 = params[3]
            program = {
                "substroks":[
                    (t1*T, t2*T, theta1, l1)],
                "totaltime":T,
                "fs":fs}
            return program
        params0 = (0, 1., theta0, dist0)
        bounds = [
            (0.02, 0.45),
            (0.55, 0.98),
            (0, 2*pi),
            (50, 500)]
    elif Nsub==2:
        def program_func(params):
            assert len(params)==6
            t1 = params[0]
            t2 = params[1]
            theta1 = params[2]
            theta2 = params[3]
            l1 = params[4]
            l2 = params[5]
            program = {
                "substroks":[
                    (0.02, t1*T, theta1, l1),    
                    (t2*T, 0.98*T, theta2, l2)],
                "totaltime":T,
                "fs":fs}
            return program
        params0 = (0.3, 0.6, theta0, theta0, dist0/2, dist0/2)
        bounds = [
            (0.2, 0.48),
            (0.52, 0.8),
            (0, 2*pi),
            (0, 2*pi),
            (50, 500),
            (50, 500)]
    elif Nsub==3:
        def program_func(params):
            assert len(params)==10
            t1 = params[0]
            t2 = params[1]
            t3 = params[2]
            t4 = params[3]
            theta1 = params[4]
            theta2 = params[5]
            theta3 = params[6]
            l1 = params[7]
            l2 = params[8]
            l3 = params[9]
            program = {
                "substroks":[
                    (0.02, t1*T, theta1, l1),    
                    (t2*T, t3*T, theta2, l2),
                    (t4*T, 0.98*T, theta3, l3)],
                "totaltime":T,
                "fs":fs}
            return program
        params0 = (0.2, 0.4, 0.6, 0.7, theta0, theta0, theta0, dist0/3, dist0/3, dist0/3)
        bounds = [
            (0.15, 0.33),
            (0.33, 0.45),
            (0.45, 0.66),
            (0.66, 0.85),
            (0, 2*pi),
            (0, 2*pi),
            (0, 2*pi),
            (50, 500),
            (50, 500),
            (50, 500)]
        
    return program_func, params0, bounds


########  post -processing code
from analysis.line2_strokmodelfits import *

def postProcess(a = "Pancho", s = 2, d = "200902", e = "lines2", 
                fit_tstamp = "200922_093340_lines2", model = "spatial", 
               ploton=False):
    """ code to load saved stroke model and extract relevant statsitics
    - ploton will also save automatically."""
    import pickle, os
    
    # ==== LOAD THINGS
    # --- 1) filedata
    fd = loadSingleData(a, d, e, s, load_resaved_data=True, resave_overwrite=False)

    # ---- 2) load model fits
    fname1 = f"strokmodelfits-{fit_tstamp}"
    fname2 = f"{a}-{e}-{d}-{s}-{model}"
    sdir = f"{fd['params']['figuredir_main']}/{fname1}/{fname2}.pkl"
    with open(sdir, "rb") as f:
        modelfits = pickle.load(f)
    with open(f"{fd['params']['figuredir_main']}/{fname1}/strokclass_{fname2}.pkl", "rb") as f:
        strokclass = pickle.load(f)

    print("found modelfits, length this (num strok):")
    print(len(modelfits))

    SAVEDIRTHIS = f"{fd['params']['figuredir_main']}/{fname1}/figures/{fname2}"
    os.makedirs(SAVEDIRTHIS, exist_ok=True)

    for m in modelfits:
        if "strok_num_0" not in m.keys():
            m["strok_num_0"] = m["strok_num"]-1 # beducase I stupidly went from 1+
            del m["strok_num"]

    ## COLLECT DISTRIBUTION OF params/scores across all single strok
    # i./e, collect all into flattened long-form data
    if False:
        vec_over_spatial_ratio = 1 # leave as 1 for now. forgot to save
        fs = fd["params"]["sample_rate"]
        strokclass = strokModel(fs, vec_over_spatial_ratio=vec_over_spatial_ratio)
    
    strokdat = []
    for m in modelfits:

        # extract strok
        t= m["trial"]
        snum = m["strok_num_0"]

        strokes = getTrialsStrokesByPeanuts(fd, t)
        strok_beh = strokes[snum]
        strok_beh -= strok_beh[0,:]
        strokdat.append({
            "strok_beh":strok_beh,
            "strok_dur":strok_beh[-1,2] - strok_beh[0,2]
        })

        # insert model things
        for k, v in m.items():
            if k=="res":
                strokdat[-1]["finalcost"]=v["fun"]
                strokdat[-1]["success"]=v["success"]
                strokdat[-1]["paramsfit"]=v["x"]
                strokdat[-1]["message"]=v["message"]
            elif k not in ["finalcost", "success", "paramsfit", "message"]:
                strokdat[-1][k]=v

        # syntehsize model stroke based on fit params
        S = strokdat[-1]

        Nsub = S["nsubstrokes"]
        params = S["paramsfit"]
        strok_beh = S["strok_beh"]
        T = strok_beh[-1,2] - strok_beh[0,2]

        program_func = getFitStuff(Nsub, fs=strokclass.fs, T=T)[0]
        program = program_func(params)
        strok_mod = strokclass.synthesize(program, ploton=False)

        strokdat[-1]["strok_mod"] = strok_mod

    # 1) plot distributions of scores (substrok =1 vs 2)
    import pandas as pd
    import seaborn as sns

    DF = pd.DataFrame(strokdat)

    DF2 = pd.pivot_table(DF, index=["trial", "strok_num_0"], values="finalcost", columns="nsubstrokes").reset_index()
    DFstrokdur = pd.pivot_table(DF, index=["trial", "strok_num_0"], values="strok_dur").reset_index()
    DF2 = pd.merge(DF2, DFstrokdur)
    
    # ===== compute ratios of scores
    from pythonlib.tools.pandastools import applyFunctionToAllRows

    fun = lambda x: x[1]/(x[1] + x[2])
    DF2 = applyFunctionToAllRows(DF2, fun, newcolname="0/(0+1)")

    fun = lambda x: x[1]/(x[2])
    DF2 = applyFunctionToAllRows(DF2, fun, newcolname="0/1")

    if ploton:
        # === plot relation between all variables
        #     fig = plt.figure(figsize=(15,15))
        fig = sns.pairplot(data=DF2, vars=[1,2,"strok_dur", "0/(0+1)", "0/1"], height=5, markers=["x"])
        plt.title(f"{a}-{d}-{s}")
        fig.savefig(f"{SAVEDIRTHIS}/overview_pairplot.pdf")

        ## ===== 2) Plot example trials along distribution of scores (grid, each element is one trial within
        # a given prctile bin)
        _plotGridByPercentiles(DF, DF2, strokclass.fs, SAVEDIRTHIS, plotver="both", Nresamples=5)
    return strokdat, DF, DF2, fd


def _plotGridByPercentiles(DF, DF2, fs, SAVEDIRTHIS, plotver="pos", Nresamples=10):
    """ 
    - Nresamples, num separate plots to amke, each time resamplinga
    again randomly.
    """
    from pythonlib.tools.plottools import annotate
    from pythonlib.tools.plottools import makeColors, colorGradient
    from pythonlib.drawmodel.strokePlots import plotDatStrokesVelSpeed, plotDatStrokes
    modellist = [1,2]
    modelcols = np.array([[0,0,1], [1,0,0]])
    p = [0, 1, 2.5, 5, 10, 20, 40, 60, 80, 90, 95, 97.5, 99, 100]

    # print("DF2 cols")
    # print(DF2.columns)
    # print("DF cols")
    # print(DF.columns)
    # assert False

    if plotver=="pos":
        plotfunc = lambda strokes, ax, pcol, alpha: plotDatStrokes(strokes, ax, pcol=pcol, alpha=alpha)
    elif plotver=="vel":
        plotfunc = lambda strokes, ax, pcol, alpha: plotDatStrokesVelSpeed(strokes, ax,  fs=fs, pcol=pcol, 
                                                                           alpha=alpha, plotver="vel", 
                                                                           nolegend=True)
    elif plotver=="both":
        plotfuncs = [
        lambda strokes, ax, pcol, alpha: plotDatStrokes(strokes, ax, pcol=pcol, alpha=alpha),
        lambda strokes, ax, pcol, alpha: plotDatStrokesVelSpeed(strokes, ax,  fs=fs, pcol=pcol, 
                                                                           alpha=alpha, plotver="vel", 
                                                                           nolegend=True)]
    for n in range(Nresamples):

        binedges = np.percentile(DF2[1], p)
        inds1 = np.digitize(DF2[1], binedges)

        binedges = np.percentile(DF2[2], p)
        inds2 = np.digitize(DF2[2], binedges)

        SIZE = 2.5
        if plotver=="both":
            figs =[]
            axess =[]
            for _ in range(2):
                f, a = plt.subplots(len(p), len(p), squeeze=False, sharex=True, sharey=True, figsize=(len(p)*SIZE,len(p)*SIZE))
                figs.append(f)
                axess.append(a)
        else:
            fig, axes = plt.subplots(len(p), len(p), squeeze=False, sharex=True, sharey=True, figsize=(len(p)*SIZE,len(p)*SIZE))

        for i in range(1, len(p)+1):
            print(i)
            for ii in range(1, len(p)+1):
                dfthis = DF2[(inds1==i) & (inds2==ii)]

                # pick a random one
                if len(dfthis)>0:
                    dfsingle = dfthis.sample(1)

        #             tlist = dfthis["trial"].values
        #             slist = dfthis["strok_num"].values
        #             tmp = random.sample([(t,s) for t, s in zip(tlist, slist)],1)[0]
                    t = dfsingle["trial"].values[0]
                    s = dfsingle["strok_num_0"].values[0]
                    strok_beh = DF[(DF["trial"]==t) & (DF["strok_num_0"]==s)]["strok_beh"].values[0]
                    # - get the strok
        #             strok_beh = [S["strok_beh"] for S in strokdat if S["trial"]==t and S["strok_num"]==s][0]

                    def goodfig(ax):
                        ax.set_ylabel(f"0-score {dfsingle[1].values[0]:.2f}")
                        ax.set_xlabel(f"1-score {dfsingle[2].values[0]:.2f}")
    #                 ax.set_title(f"1/(1+2) {dfsingle['1/(1+2)'].values[0]:.2f}", color="r")
                        ax.set_title(f"t{t}-s{s}")

                        MAX = 2.5 # upper boudn, for coloring by gradient
                        pos = dfsingle['0/1'].values[0]/MAX
                        col = colorGradient(pos, col1=[0,0,1], col2=[1,0,0])
                        annotate(f"1/2({dfsingle['0/1'].values[0]:.2f}), 1/1+2({dfsingle['0/(0+1)'].values[0]:.2f})", ax=ax,color=col)

                    if plotver=="both":
                        for k in range(2):
                            ax = axess[k][i-1, ii-1]
                            plotfuncs[k]([strok_beh], ax, pcol="k", alpha=0.5)
                            goodfig(ax)
                    else:
                        ax = axes[i-1, ii-1]
                        plotfunc([strok_beh], ax, pcol="k", alpha=0.5)
                        goodfig(ax)

                    # ======= OVERLAY MODEL FIT
                    # for all models
                    for m in modellist:
                        strok_mod = DF[(DF["trial"]==t) & (DF["strok_num_0"]==s) & (DF["nsubstrokes"]==m)]["strok_mod"].values[0]
                        pcol = colorGradient

                        if plotver=="both":
                            for k in range(2):
                                ax = axess[k][i-1, ii-1]
                                plotfuncs[k]([strok_mod], ax, pcol=modelcols[m-1,:], alpha=0.1)
                        else:
                            ax = axes[i-1, ii-1]
                            plotfunc([strok_mod], ax, pcol=modelcols[m-1,:], alpha=0.1)


        if plotver=="both":
            figs[0].savefig(f"{SAVEDIRTHIS}/egtrials_grid_pos_randsamp{n}.pdf")
            figs[1].savefig(f"{SAVEDIRTHIS}/egtrials_grid_vel_randsamp{n}.pdf")
        else:
            fig.savefig(f"{SAVEDIRTHIS}/egtrials_grid_{plotver}_randsamp{n}.pdf")


if __name__=="__main__":
    from math import pi
    from pythonlib.tools.modfittools import minimize    
    from pythonlib.tools.distfunctools import *
    
    ## === PARAMS
    expt = "lines2"

    if True:
        from analysis.modelexpt import loadMetadat
        MD = loadMetadat(expt)
        datelist = MD["dates_for_summary"]        
    else:
        # ---- DATES
        if False:
            sdate = 200902
            edate = 200907
            from pythonlib.tools.datetools import getDateList
            datelist = getDateList(sdate, edate)
        else:
            datelist = [200903]

    animal_list = ["Pancho", "Red"]

    # --- model classes to apply (defines how to weight different costs for fitting)
    # modelclass_list = ["spatial", "spatial_vec", "vec"]    
    modelclass_list = ["spatial"]    


    # --- mostly unchanged below.
    just_to_get_ratio = False
    vec_over_spatial_ratio = 5.0 # 5.0 seems empirically good based on Pancho and red
    # across 9/2 to 9/7. I ran "just_to_get_ratio", saved pkl files, but I looked at what
    # printed. Range is about 4.3 to 5.7, with Pancho being generally lower. 
    # If pass None here, then will do get autoamticlaly but will differ for each analysis...


    # make timestamp for saving
    from pythonlib.tools.expttools import makeTimeStamp
    ts = makeTimeStamp(expt)


    # === analtysis params
    MINTIME = 0.2
    MINDIST=50
    Nsub_to_run = [1,2]


    for animal in animal_list:
        
        # =================
        FD = loadMultData({"animal":animal, "expt":expt, "dates":datelist})

        # ====== FOR EACH FILEDATA, PERFORM STROK MODEL FITS
        for dat in FD:
            fd = dat["fd"]
            fs = fd["params"]["sample_rate"]
            SAVEDIR = f"{fd['params']['figuredir_main']}/strokmodelfits-{ts}"


            # ---------------- BEHAVIOR
            ## ============= ITERATE OVER A DAY OF STROKES, COLLECTING 
            # MODEL RESULTS ACROSS ALL STROKES

            # 1) Get list of strokes across tasks
            trialslist = [t for t in getIndsTrials(fd) if getTrialsFixationSuccess(fd,t)]
            strokeslist = [getTrialsStrokesByPeanuts(fd,t) for t in trialslist]

            # strokeslist = strokeslist[:5]

            # --------------- MODELS
            for model in modelclass_list:
                if model=="default":
                    if vec_over_spatial_ratio is None:
                        vosr = getShuffleBehDistances(fd, 
                            fs, N=100, ploton=False)
                    else:
                        vosr = vec_over_spatial_ratio

                    if just_to_get_ratio:
                        vosr = getShuffleBehDistances(fd, 
                            fs, N=100, ploton=False)
                        import pickle
                        with open(f"{SAVEDIR}/{dat['animal']}-{dat['expt']}-{dat['date']}-{dat['session']}_vosr.pkl", "wb") as f:
                            pickle.dump(vosr, f)
                        continue
                elif model=="spatial":
                    vosr = (vec_over_spatial_ratio, 0)
                elif model=="spatial_vec":
                    vosr = (vec_over_spatial_ratio, 1)
                elif model=="vec":
                    vosr = (0, 1)
                else:
                    print(model)
                    assert False, "not coded"

                # ----- COST FUNCTION
                # -- initailize model class, which will use for fitting
                # (i.e. cost function)
                strokclass = strokModel(fs, vec_over_spatial_ratio=vosr)

                modelfits = []
                for t, strokes in zip(trialslist, strokeslist):

                    strok_list = [strok for strok in strokes]
                    strok_num_list = list(range(1, len(strok_list)+1))
                    print(f"-- running trial {t}")

                    for i, (strok_beh, strok_num) in enumerate(zip(strok_list, strok_num_list)):

                        # - params to initialize for this strok.
                        T = strok_beh[-1,2] - strok_beh[0,2]
                        theta0 =stroke2angle([strok_beh])[0]
                        dist0 = strokeDistances([strok_beh])[0]
                        
                        if dist0>MINDIST and T>MINTIME:
                            
                            # ---- DIFFERENT STROK MODELS (E..G, 1, 2 STROKES..)
                            for Nsub in Nsub_to_run:
                                program_func, params0, bounds = getFitStuff(Nsub, fs=fs,
                                                                            theta0 = theta0,
                                                                           dist0 = dist0, T=T)

                                func = strokclass.getCostFunc(strok_beh, program_func)
                                # strokclass.synthesize(program_func(params0), ploton=True)

                                res = minimize(func, params0, bounds=bounds)

                                # == take optimization results and extract useful things
                                ploton=False
                                if ploton:
                                    func(params0, ploton=True)
                                    func(params_fit, ploton=True)
                                
                                modelfits.append({
                                    "trial":t,
                                    "strok_num_1":strok_num,
                                    "strok_num_0":strok_num-1,
                                    "nsubstrokes":Nsub,
                                    "model":model,
                                    "res":res})
                                                # === save for this dat
                # saving dir
                import os
                os.makedirs(SAVEDIR, exist_ok=True)
                print(f"saving at {SAVEDIR}")

                import pickle
                with open(f"{SAVEDIR}/{dat['animal']}-{dat['expt']}-{dat['date']}-{dat['session']}-{model}.pkl", "wb") as f:
                    pickle.dump(modelfits, f)
                with open(f"{SAVEDIR}/strokclass_{dat['animal']}-{dat['expt']}-{dat['date']}-{dat['session']}-{model}.pkl", "wb") as f:
                    pickle.dump(strokclass, f)
                


                                    
