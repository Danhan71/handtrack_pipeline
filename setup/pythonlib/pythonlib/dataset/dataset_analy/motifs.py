""" Given dataset, extract all motifs (ordered list of tokens, where the relevatn 
features that define
a token are user-defined) and some basic analysis, stats, and plotting. 

Main goal are tools for extraction, filtering the motifs.

1/31/23 
Builds on:
- notebook: 220707_analy_spatial_prims_motifs, which I used to set up tasks for
experiments to have common subsequence (motifs). [REPLACES THIS]
- diagnostic modeling, comparing the actual motifs (beh) to the expected motifs
given specific models. in progress.
"""

import numpy as np
import matplotlib.pyplot as plt


def preprocess_dat(D, only_prims_on_grid=True):
    """ Helper to generate MotifsClass given a dataset D
    """

    if only_prims_on_grid:
        # 1) Only use prims in grid
        # D.filterPandas({"task_kind":["prims_on_grid"]}, "modify")
        D.filterPandas({"task_kind":["prims_on_grid"]}, "modify")

    # 1) preprocess
    params = ["no_supervision", "remove_online_abort", "correct_sequencing"]
    D.preprocessGood(params =params)

    # Generate tokens
    D.behclass_preprocess_wrapper()


class MotifsClass(object):
    """
    NOTE: each instance should work with only one kind of motif. If you want multiple
    motifs, just make multiple instances.
    """
    def __init__(self, Dataset):

        self.Dataset = Dataset.copy(dfdeep=False) # 

    def preprocess_motifs_extract(self, features=["shape_oriented", "gridloc"],
        which_sequence="beh_firsttouch", nprims=2, shapes_to_ignore=None):
        """ Generate initial set of all motifs.
        RETURNS:
        - saves motifs in self.MotifsAllDict, where keys are motifs (tuples of tokens) and itmes are 
        list of indices, where and index is (trial, starting strokenum)
        e..g, (('line-8-4-0', (1, 0)), ('line-8-4-0', (1, 1))): [(0, 0),
          (20, 0),
          (45, 0),
          (64, 0),
          (65, 0)]
        - self.MotifsList, list of motifs, sorted.
        """

        # from pythonlib.dataset.dataset_analy.datseg_motifs import generate_dict_of_all_used_motifs
        # if e in ["primsingrid2", "primsingrid2sub", "shapedirsequence1"]:
        #     shapes_to_ignore = []
        # else:
        #     assert False
        #     # not sure what this means (shapes_to_ignore = ['squiggle3-2-0', 'V-2-0' ,'Lcentered-4-0'])
        #     shapes_to_ignore = ['squiggle3-2-0', 'V-2-0' ,'Lcentered-4-0']
        from pythonlib.tools.listtools import sort_mixed_type
        motifs_all_dict = generate_dict_of_all_used_motifs(self.Dataset, nprims, features, 
                                                            WHICH_DATSEGS = which_sequence, 
                                                           shapes_to_ignore=shapes_to_ignore)
        self._MotifsAllDict = motifs_all_dict
        self.MotifsList = sort_mixed_type(list(motifs_all_dict.keys()))

    def motifs_all_dict_extract(self):
        """ The original reprenstiaton, before moved here in to Class
        """
        # Restructure data to fit older code below.
        motifs_all_dict = {}
        for k, v in self._MotifsAllDict.items():
            motifs_all_dict[k] = [vv[1] for vv in v]
        return motifs_all_dict

    def preprocess_motifgroups_extract(self):
        # from pythonlib.dataset.dataset_analy.datseg_motifs import generate_motifgroup_data

        # Restructure data to fit older code below.
        motifs_all_dict = self.motifs_all_dict_extract()
        # for k, v in self._MotifsAllDict.items():
        #     motifs_all_dict[k] = [vv[1] for vv in v]
        self.MotifGroupsAllDict, self.MotifGroupsAllKinds = generate_motifgroup_data(motifs_all_dict)

    def _plot_single_motif_motifdat(self, motifdat, ax, which_overlay, underlay_single_color):
        trial = motifdat["trial"]
        # self.Dataset.plot_single_trial_highlighting_overlaid_strokes(md["trial"], 
        #     md["strokeinds"], ax, "beh", "beh", underlay_single_color="k")
        strokes = self._motifdat_strokes_extract(motifdat, which_overlay)
        self.Dataset._plot_single_trial_highlighting_overlaid_strokes(motifdat["trial"], 
            strokes, ax, "beh", underlay_single_color=underlay_single_color)

    def plot_single_motif_motifdats(self, list_motifdat, which_overlay="beh",
            underlay_single_color="k"):
        """ Plot each motifdat as a subplot, overlaying on the original trial
        """
        n= len(list_motifdat)
        ncols = 5
        nrows = int(np.ceil(n/ncols))
        SIZE = 2.5
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE),
            sharex=True, sharey=True)
        for md, ax in zip(list_motifdat, axes.flatten()):
            self._plot_single_motif_motifdat(md, ax, which_overlay, 
                    underlay_single_color=underlay_single_color)
        return fig, axes


    def plot_single_motif_trials(self, indmotif, trials=None, nrand=20, which_overlay="beh"):
        """ Plot trials for a single motif
        PARAMS:
        - trials,  if None, then plots all trials (nrand), otherwise a list of ints,
        trials into Dataset
        """
        assert trials is None, "not coded, this shoud index ibnto the trials"
        # if trials is None:
        #     import random
        #     trials = range(len(self.Dataset))


        # indmotif = random.sample(range(len(motifs_all_dict)), 1)[0]
        # motif = self.MotifsList[indmotif]

        list_motifdat = self.motifdat_extract_all(indmotif, strokes_extract=True)
        self.plot_single_motif_motifdats(list_motifdat, which_overlay="beh")


    def motifindex_to_str(self, indmotif):
        """
        Reutnrs string, input can be int or str
        """
        if isinstance(indmotif, int):
            motif = self.MotifsList[indmotif]
        else:
            motif = indmotif
        return motif

    def motifdat_extract_all(self, indmotif, strokes_extract=False):
        """ Extract all instances for this motif.
        RETURNS:
        - list of dicts, each an instance
        """ 
        motif = self.motifindex_to_str(indmotif)
        inds = range(len(self._MotifsAllDict[motif]))
        list_out = [self.motifdat_extract(indmotif, i, strokes_extract=strokes_extract) for i in inds]
        return list_out

    def motifdat_extract(self, indmotif, ind_withinmotif, strokes_extract=False):
        """ Helper to extract in useful terms
        PARAMS:
        - indmotif, index into self.MotifsList (if int) or self.MotifsAllDict,
        if str
        - ind_withinmotif, index into self.MotifsAllDict[motifstr]
        RETURNS:
        - dict, summarizing this motif instance
        """

        motif = self.motifindex_to_str(indmotif)
        trial_strok = self._MotifsAllDict[motif][ind_withinmotif][1]
        tokens = self._MotifsAllDict[motif][ind_withinmotif][0]
        if strokes_extract:
            strokes = [t["Prim"].Stroke() for t in tokens]
        else:
            strokes = None

        n = len(motif)
        out = {
            "motif":motif,
            "indmotif":indmotif,
            "trial":trial_strok[0],
            "strokeinds":list(range(trial_strok[1], trial_strok[1]+n)),
            "tokens":tokens,
            "strokes_task":strokes
        }
        return out

    def _motifdat_indsbeh_extract(self, motifdat):
        """ Returns the list of ints
        """

        # print(motifdat)
        inds_inds = [t["ind_behstrokes"] for t in motifdat["tokens"]]
        inds = [i for inds in inds_inds for i in inds] # flatten
        # print(inds)
        # assert False
        return inds

    def _motifdat_strokes_extract(self, motifdat, which="beh"):
        """ Return the strokes for this datmotif,
        """
        if which=="task":
            strokes = motifdat["strokes_task"]
        elif which=="beh":
            inds_beh = self._motifdat_indsbeh_extract(motifdat)
            trial = motifdat["trial"]
            strokes_beh = self.Dataset.Dat.iloc[trial]["strokes_beh"]
            strokes = [strokes_beh[i] for i in inds_beh]
        else:
            assert False
        return strokes

    def print_summary(self):

        for i, motif in enumerate(self.MotifsList):
            n = len(self._MotifsAllDict[motif])
            print(i, ' -- ', motif, ' -- ', n)

    def print_summary_thismotif(self, indmotif):

        list_md = self.motifdat_extract_all(indmotif)
        for md in list_md:
            print(md["trial"], ' -- ', md["strokeinds"])

    ################ MOTIFGROUP
    def print_motifgroup_summary(self):
        """ Print summary of all the motif groups
        """
        DatGroups = self.MotifGroupsAllDict
        list_group_kind = self.MotifGroupsAllKinds

        # Summarize, how many groups found and how many trials each
        do_print_shapes = True

        def _print(DatGroups, which_group, inds_motifgroup_ordered=None):
            if inds_motifgroup_ordered is None:
                inds_motifgroup_ordered = range(len(self.MotifGroupsAllDict[which_group]))

            for i in inds_motifgroup_ordered:
                # 1) info about the motifs/tokens/shapes actually found for this group
                list_motifs, motifgroup, list_tokens, list_shapes = self.motifgroup_extract_motifs_simple(which_group, i)
                        
                # 2) Info about the n distributions
                Dat = DatGroups[which_group][i]
                ntrials_per_unique_motif = sorted(Dat["n_trials_per_motif"], reverse=True)
                ntrials_per_unique_motif = [n for n in ntrials_per_unique_motif if n>0]
                if do_print_shapes:
                    print("#", i, "|n trials tot:", Dat["n_trials_summed_across_motifs"], "|n motifs:", Dat["n_unique_motifs_with_at_least_one_trial"], "|n trials/motif:", ntrials_per_unique_motif, "|shapes:", list_shapes)
                else:
                    print("#", i, "|n trials tot:", Dat["n_trials_summed_across_motifs"], "|n motifs:", Dat["n_unique_motifs_with_at_least_one_trial"], "|n trials/motif:", ntrials_per_unique_motif)
        
        for which_group in list_group_kind:
            print(" === ", which_group)
            _print(DatGroups, which_group, None)
    
    def motifgroup_count_num_motifkeys(self, which_group):
        """ Return int, the number of keys within this group
        NOTE: a key is a set of motifs which have some shared feature.
        """
        dat = self.MotifGroupsAllDict[which_group]
        return len(dat)

    # def motifgroup_extract_trials(self, which_group, ind_within_group, doprint=False,
    #         NMAX_PER_MOTIF=None):
    #     """ Get all trials (and stroke num) that are in this which_group (this indx within)
    #     # indtoplot = 1 # where 0 means the best (group with most cases)
    #     # which_group = "diff_sequence"
    #     # which_group = "diff_location"
    #     # which_group = "same"
    #     """
    #     assert False, "use motifgroup_extract_all_motifs instead"
    #     from pythonlib.dataset.dataset_analy.datseg_motifs import get_inds_this_motifgroup

    #     motifs_all_dict = self.motifs_all_dict_extract()

    #     # Extract all the specific trials that are part of this group
    #     dat = self.MotifGroupsAllDict[which_group]
    #     motifgroup = dat[ind_within_group]["key"]
    #     inds_idxs_all, list_motifs, _ = get_inds_this_motifgroup(self.MotifGroupsAllDict, which_group, motifs_all_dict, 
    #                                                              motifgroup, ntrials_per_motif=NMAX_PER_MOTIF)
    #     inds_trials_all = [idx[0] for idx in inds_idxs_all]
    #     inds_strokenums_all = [idx[1] for idx in inds_idxs_all]

    #     if doprint:
    #         print("Thiese trials and strokes and motifs:")
    #         for mot, inds in zip(list_motifs, inds_idxs_all):
    #             print(mot, ' --- ', inds)
    #         # print(inds_idxs_all)
    #         # print(list_motifs)
    #     return inds_trials_all, inds_strokenums_all, list_motifs

    def motifgroup_extract_motifs_thiskey(self, which_group, motifkey_index, doprint=False,
            NMAX_PER_MOTIF=None, return_as_list_motifdat=False):
        """ Get all motif instances that are in this which_group (this indx within)
        PARAMS
        - which_group, str, what group of motifs? 
        e.g., "same", "diff_sequence", "diff_location"
        - motifkey_index, index within the sets of motifs that are cases for,
        where 0 means the best (group with most cases)
        RETURNS:
        - list of motif instances, which are either dicts or motifdats, depending on 
        whether return_as_list_motifdat is True.
        """
        # from pythonlib.dataset.dataset_analy.datseg_motifs import get_inds_this_motifgroup

        # motifs_all_dict = self.motifs_all_dict_extract()

        # Extract all the specific trials that are part of this group
        dat = self.MotifGroupsAllDict[which_group]
        motifkey = self.MotifGroupsAllDict[which_group][motifkey_index]["key"]
        n_trials_per_motif = self.MotifGroupsAllDict[which_group][motifkey_index]["n_trials_per_motif"]
        
        list_list_md = []
        out = []
        for i, (motif, ntrials) in enumerate(zip(motifkey, n_trials_per_motif)):
            if ntrials>0:
                list_md = self.motifdat_extract_all(motif)
                for md in list_md:
                    out.append({
                        "motifdat":md,
                        "motif":motif,
                        "motif_index":i,
                        "motifkey":motifkey,
                        "motifkey_index":motifkey_index,
                        "which_group":which_group,
                        "trial":md["trial"],
                        "indstroke_on":md["strokeinds"][0]
                        })
        if return_as_list_motifdat:
            out = [d["motifdat"] for d in out]

        return out

    def motifgroup_extract_motifs_all_filtered(self, which_group, feature_vary=None, 
            feature_constant=None):
        """ Clever way to represent motifs, related to getting variation in character (for
        example). e.g,, if want to get motifs which exist across characters, holding epoch
        constant.
        PARAMS:
        - which_group, str, collects all sets of motifs (i.e.,, motif keys) in this group, e.g.
        "same", "diff_prims"
        - feature_vary, str, whether to filter, so that only get motif_keys which have trials across
        >1 level for this feature. e.g., if want motif_keys that have motifs present in at least 2
        characters, make this "character"
        - feature_constant, str, what to condition on, as you check feature_vary. e.g., if want to extract
        only cases which, for each epoch, have trials across levels of character, make this "epoch"
        E.G. motifgroup_extract_motifs_all_filtered("same", feature_vary=epoch, feature_constant=character)
        will extract only conjunction of motif_key and character which have cases of "same" motif across epochs
        RETURNS: 
        - OUT, list of dict, where each dict represents a single motif isntance on a specific trial. 
        """

        D = self.Dataset

        if feature_constant is not None:
            # get each level.
            list_levels_feature_constant = sorted(self.Dataset.Dat[feature_constant].unique().tolist())
        else:
            # then dont care about this
            list_levels_feature_constant = [None]

        # for each group, get its list of trials
        nkeys = self.motifgroup_count_num_motifkeys(which_group)

        OUT = []
        for i in range(nkeys):
            
            list_motifdict = self.motifgroup_extract_motifs_thiskey(which_group, i, return_as_list_motifdat=False)
            inds_trials_all = [x["trial"] for x in list_motifdict]
                
            for feat_con in list_levels_feature_constant:

                # Restrict just to trials within this feature_con
                if feat_con is not None:
                    trials_good = D.Dat[(D.Dat[feature_constant]==feat_con)].index.tolist()
                    inds_trials_good = [i_ for i_ in inds_trials_all if i_ in trials_good]
                    feat_con_save = feat_con
                else:
                    inds_trials_good = inds_trials_all
                    feat_con_save = "IGNORE"

                if len(inds_trials_good)==0:
                    print(i, feat_con, "failed inds_trials_good...")
                    continue

                # chekc if have multipel llevels of desired feature
                if feature_vary:
                    list_feat_unique = D.Dat.iloc[inds_trials_good][feature_vary].unique().tolist()
                    if len(list_feat_unique)<2:
                        print(i, feat_con, "failed list_feat_unique...")
                        continue

                # extract the motifdats 
                list_motifdict_this = [x for x in list_motifdict if x["trial"] in inds_trials_good]
                list_motifdat = [x["motifdat"] for x in list_motifdict_this]
                list_trialsthis = [x["trial"] for x in list_motifdict_this]
                # For each motif, get its level
                if feature_vary:
                    list_feature_vary_levels = D.Dat.iloc[list_trialsthis][feature_vary].tolist()
                else:
                    list_feature_vary_levels = None

                for motifdict in list_motifdict_this:
                    trial = motifdict["trial"]

                    OUT.append({
                        "trial": trial,
                        "motifdict":motifdict,
                        "which_group":which_group,
                        "motifkey":i,
                        "feature_constant":feat_con_save,
                        "feature_vary":feature_vary,
                        "feature_vary_level":D.Dat.iloc[trial][feature_vary] if feature_vary is not None else None
                        })
        return OUT
                        


    def motifgroup_extract_motifs_simple(self, which_group="same", index=0):
        """ Return the list of motifs found for this motifgrop
        e.g,: returns:
        ([(('line-8-4-0', (-1, 0)), ('line-8-4-0', (-1, 1)))],
             ((('line-8-4-0', (-1, 0)), ('line-8-4-0', (-1, 1))),),
             [('line-8-4-0', (-1, 1)), ('line-8-4-0', (-1, 0))],
             ['line-8-4-0'])
        """
        # from pythonlib.dataset.dataset_analy.datseg_motifs import get_all_motifs_found_for_this_motifgroup
        # if inds_motifgroup_ordered is None:
        inds_motifgroup_ordered = range(len(self.MotifGroupsAllDict[which_group]))
        return get_all_motifs_found_for_this_motifgroup(self.MotifGroupsAllDict,
            which_group, index) 

    def plot_motifgroup_singlegroup_trials(self, which_group, ind_within_group, 
        NMAX_PER_MOTIF = 30, underlay_single_color="k"):
        if False:
            inds_trials_all, inds_strokenums_all, list_motifs = self.motifgroup_extract_trials(
                which_group, ind_within_group, doprint=True, NMAX_PER_MOTIF=NMAX_PER_MOTIF)
        else:
            list_motifdat = self.motifgroup_extract_motifs_thiskey(which_group, ind_within_group, 
                NMAX_PER_MOTIF=NMAX_PER_MOTIF, return_as_list_motifdat=True)

        #### Plot each trial for this group
        if False:
            # for trial, snum in zip(inds_trials_all, inds_strokenums_all):
            self.Dataset.plotMultTrials2(inds_trials_all, titles = inds_strokenums_all);
            self.Dataset.plotMultTrials(inds_trials_all, "strokes_task", titles = inds_strokenums_all);
        else:
            fig, axes = self.plot_single_motif_motifdats(list_motifdat, 
                underlay_single_color=underlay_single_color)
            # self.plot_single_motif_motifdats([dat[0]["motifdat"]], ax)

        return fig, axes, list_motifdat


def plotwrapper_drawings_trials_spanning_feature(D, nprims, SDIR, which_group="same", feature_vary = None,
    feature_constant = None):
    """ Plot drawings, where each plot is all drawins across levels of a given feature
    (feature_vary, e.g., epoch), for a single level of the feature (feature_constant, e.g, char).
    both feature_vary and feature_constant can be None, in which case ignore it. All drawings depict
    instances of motifs within a motif_key, under the grouping which_group (e..g, same). 
    (SEE e.g.);
    PARAMS;
    - nprims, motifs with this many prims.
    EXAMPLE:
    e..g, for this, each plot is a single motif (shape-location) for all trials across all chars it 
    is found (and skipping this modif if doesnt find across chars), one plot for each epoch-motif
    conjunction.
    plotwrapper_drawings_trials_spanning_feature(D, 2, SDIR, which_group="same", feature_vary = "char",
        feature_constant = "epoch"):
    """

    # # Find groups that have trials spanning a levels of a vairble (ee..g epoch)
    # which_group="same"
    # # feature_vary = "epoch" # want trials across all levels of this feature
    # feature_vary = None # want trials across all levels of this feature
    # # feature_constant = "character" # perform analysis separately for each level of this feature
    # feature_constant = None # perform analysis separately for each level of this feature
    import pandas as pd
    from pythonlib.tools.expttools import writeStringsToFile

    sdir = f"{SDIR}/nprims_{nprims}-matchacrosstrials-grp_{which_group}-varyby_{feature_vary}-constby_{feature_constant}"
    import os
    os.makedirs(sdir, exist_ok=True)
    print("Saving at:", sdir)

    # 1) Generate MotifsClass and preprocess
    MC = MotifsClass(D)
    MC.preprocess_motifs_extract(nprims=nprims)
    MC.preprocess_motifgroups_extract()

    # 2) Extract motifs across features.
    OUT = MC.motifgroup_extract_motifs_all_filtered(which_group, feature_vary, feature_constant)
    dfout = pd.DataFrame(OUT)

    #### Write all trials to a text file.
    list_str = []
    list_str_trialcodes = []
    list_motifkey = dfout["motifkey"].unique().tolist()
    list_feature_constant = dfout["feature_constant"].unique().tolist()
    list_feature_level = dfout["feature_vary_level"].unique().tolist()
    for key in list_motifkey:
        for fc in list_feature_constant:
            n = sum((dfout["motifkey"]==key) & (dfout["feature_constant"]==fc))
            if n>0:
                list_str.append(f"#{key} -- {fc}")
                list_str_trialcodes.append(f"#{key} -- {fc}")
                for lev in list_feature_level:
                    dfthis = dfout[(dfout["motifkey"]==key) & (dfout["feature_constant"]==fc) & (dfout["feature_vary_level"]==lev)]
                    list_trial = dfthis["trial"].tolist()
                    list_motif = [x["motif"] for x in dfthis["motifdict"].tolist()]
                    list_str.append(f"  {lev} -- {len(dfthis)}")    
                    list_tc = D.Dat.iloc[list_trial]["trialcode"].tolist()
                    for tc, mot in zip(list_motif, list_tc):
                        list_str_trialcodes.append(f"  {lev}  |  {tc}  |  {mot}")
                    
    # how many trials across each level of feature vary
    fname = f"{sdir}/eachnum_trials.txt"
    writeStringsToFile(fname, list_str)
    fname = f"{sdir}/each_trial_motif.txt"
    writeStringsToFile(fname, list_str_trialcodes)

    #### Plot each drawing
    for key in list_motifkey:
        for fc in list_feature_constant:
            dfthis = dfout[(dfout["motifkey"]==key) & (dfout["feature_constant"]==fc)]
            if len(dfthis)>0:
                print(f"#{key} -- {fc}")
                list_mdict = dfthis["motifdict"].tolist()
                list_motifdat = [x["motifdat"] for x in list_mdict]
                list_featvar_levels = dfthis["feature_vary_level"].tolist()
                list_trials = dfthis["trial"].tolist()
                list_trialcode = D.Dat.iloc[list_trials]["trialcode"]
                # for each trial, get its value across desired levsls.
                fig, axes = MC.plot_single_motif_motifdats(list_motifdat, underlay_single_color=None)

                # show its feature
                titles = [f"{tcode}-{feat}" for tcode, feat in zip(list_trialcode, list_featvar_levels)]
    #             titles = D.Dat.iloc[inds_trials_good][feature_vary].tolist()
                
                for ax, tit in zip(axes.flatten(), titles):
                    ax.set_title(tit)

                # save
                fig.savefig(f"{sdir}/motifkey_{key}-featconst_{fc}.pdf")



""" 
7/7/22 - for extracting sequences from (e.g) n prims in grid, based on things like:
- diff prims, but same sequence and location.
Assumes clean tasks (prims in grid)

Works using datsegs (either aligned to beh or task strokes)

See notebook: analy_spatial_prims_motifs_220707

Is like "motifs in a bag"...
"""

VARIABLES_KEEP = ["shape_oriented", "gridloc"]

def generate_dict_of_all_used_motifs(D, nprims=2, 
    variables_keep = None,
    WHICH_DATSEGS = "beh_firsttouch", shapes_to_ignore=None):
    """ Generate dict holding all motifs used in this dataset.
    PARAMS:
    - D, Dataset
    - nprims, number of prims in motif. Currently only coded for 2
    - variables_keep, list of str keys into datsegs. order of this list
    defines the feature tuple for each token. e..g, ["shape_oriented", "gridloc"]
    - WHICH_DATSEGS, str, {'beh', 'beh_firsttouch'}, defines which datsegs to use, ether
    aligned to beh or beh_firsttouch.
    - shapes_to_ignore, list of str of shapes_oriented, ignores any motifs that inlcude this
    shape. None to not apply this
    RETURNS:
    - motifs_all_dict, dict, where keys are motifs (tuples of tokens) and itmes are 
    list of indices, where and index is (trial, starting strokenum)
    """

    assert WHICH_DATSEGS in {"beh_using_task_data", "beh_firsttouch"}, "these are the only ones that make sens..."
    # assert WHICH_DATSEGS == "beh", "this is the only one that maintains mappign between stroke indices across all dat. if use beh_firsttouch, plots will be weird."

    if variables_keep is None:
        variables_keep = VARIABLES_KEEP
    motifs_all_dict = {}
    def prune_token_variables(token, variables_keep):
        """
        keep only the variables you care about.
        token = datsegs[i]
        RETURNS:
        tuple of items, in order of variables_keep
        """
    #     token = {k:v for k, v in token.items() if k in variables_keep}
        token_tuple = tuple([token[k] for k in variables_keep])
        return token_tuple

    for ind in range(len(D.Dat)):
        datsegs = D.taskclass_tokens_extract_wrapper(ind, which_order=WHICH_DATSEGS)
        # Beh = D.Dat.iloc[ind]["BehClass"]
        # primlist, datsegs_behlength, datsegs_tasklength = D.behclass_extract_beh_and_task(ind)[:3]
        # if WHICH_DATSEGS=="task":
        #     # matching beh strokes first touch
        #     datsegs = datsegs_tasklength
        # elif WHICH_DATSEGS=="beh":
        #     # matching beh strokes,
        #     datsegs = datsegs_behlength
        # else:
        #     assert False
            
        # print(datsegs)
        # print(datsegs[0]["Prim"].Stroke())
        # assert False

        for i in range(len(datsegs)-(nprims-1)):
            motif = datsegs[i:i+nprims]
                
    #         # MOtifs that should obviously not keep
    #         if motif[0][1]
            
            # save this motif
            # use the motif as a key (hash it)
            # just keep the variables that you care about
            motif_features = tuple([prune_token_variables(token, variables_keep) for token in motif])
            
            # This is hashable. use as key in dict
            index = (motif, (ind, i))
            if motif_features in motifs_all_dict.keys():
                motifs_all_dict[motif_features].append(index)
            else:
                motifs_all_dict[motif_features] = [index]

            # # Save its tokens and strokes.
            # motif_tokens = motif

    # prune motifs
    if shapes_to_ignore is not None:
        # First, check you didnt make mistake netering shapes
        list_shapes = extract_list_shapes_loc(motifs_all_dict)[0]
        for sh in shapes_to_ignore:
            if sh not in list_shapes:
                print(sh)
                print(list_shapes)
                assert False

        # Second, clean up the motifs
        def _is_bad(motif):
            for token in motif_features:
                if token[0] in shapes_to_ignore:
                    return True
            return False

        motifs_all_dict_new = {}
        for motif_features, v in motifs_all_dict.items():
            if _is_bad(motif_features):
                print("removing motif: ", motif_features)
                continue
            else:
                motifs_all_dict_new[motif_features] = v
        motifs_all_dict = motifs_all_dict_new

    print("Found this many motifs: ", len(motifs_all_dict))
    # sorted(list(motifs_all_dict.keys()))
    print("This many instances per motif_features: ", [len(v) for k, v in motifs_all_dict.items()])
    
    return motifs_all_dict

def extract_shapes_in_motif(motif_features):
    """
    Get list of shapes (unique, sorted) in this motif_features
    """
    shapes = []
    for token in motif_features:
        # loc = token[1]
        shape = token[0]
        shapes.append(shape)
    return shapes


def extract_list_shapes_loc(motifs_all_dict):
    """ get the list of all unqiue shapes and locations
    """
    # 1) Get list of all unique shapes and locations in the dataset.
    list_shapes = []
    list_locs = []
    for motif in motifs_all_dict.keys():
        for token in motif:
            loc = token[1]
            shape = token[0]
            list_locs.append(loc)
            list_shapes.append(shape)
    list_locs = sorted(list(set(list_locs)))
    list_shapes = sorted(list(set(list_shapes)))
    return list_shapes, list_locs

def generate_motifgroup_data(motifs_all_dict, 
        list_group_kind= None):
    """ Main extraction of computed data. For each way of grouping (e.g., same prim, same seq, diff location),
    pull out each "motifgroup" (which is a set of all motifs which are related under this grouping), and
    for each motifgroup save the trials that have these motifs. 
    RETURNS:
    - DatGroups, dict where keys are each kind of grouping (e.g, diff_sequence), and items 
    are lists of dicts, where each dict is a motifgroup and its data.
    """
    if list_group_kind is None:
        list_group_kind = ["same", "diff_sequence", "diff_location", "diff_prims"]

    list_shapes, list_locs = extract_list_shapes_loc(motifs_all_dict)
    
    # 2) Extract data
    DatGroups = {}
    for group_kind in list_group_kind:

        def get_all_possible_motifs_for_this_motifs_group(motif):
            """
            RETURNS:
            - motif_group, tuple of tples, sorted, where one of the
            tuples if motif, and the rest are the other motifs in this 
            group for this motif. This works becauseof symmatry -i.e,
            each motif is only in one possible group
            """

            list_motif = []
            if group_kind=="diff_sequence":
                # Get motif and its reverse
                list_motif = [motif, motif[::-1]]
            elif group_kind=="diff_location":
                # get this sequence of prims, at all possibel locations
                shape1 = motif[0][0]
                shape2 = motif[1][0]
                for i in range(len(list_locs)):
                    for ii in range(len(list_locs)):
                        if i!=ii: # different locations
                            # This is a unique motif
                            loc1 = list_locs[i]
                            loc2 = list_locs[ii]
                            motif = ((shape1, loc1), (shape2, loc2))
                            list_motif.append(motif)
            elif group_kind=="diff_prims":
                # Get this sequence of locations, all possible prim location pairs.
                loc1 = motif[0][1]
                loc2 = motif[1][1]
                for shape1 in list_shapes:
                    for shape2 in list_shapes:
                        # This is a unique motif
                        motif = ((shape1, loc1), (shape2, loc2))
                        list_motif.append(motif)
            elif group_kind=="same":
                # simply: motifs should be identical
                list_motif = [motif]
            else:
                assert False

            # assert len(list_motif)>1, "even if you want same motif, just include it twice."
            # Convert to sorted tuple
            list_motif = sorted(list_motif)
            motif_group = tuple(list_motif)
            return motif_group

        def count_num_instances_this_motif(motif):
            # How often did this motif occur in dataset?
            if motif in motifs_all_dict.keys():
                return len(motifs_all_dict[motif])
            else:
                return 0

        # Got thru all motifs. check which group it is in
        dict_motifgroups_all = {}
        list_motifs_iter = list(motifs_all_dict.keys())
        for motif in list_motifs_iter:

            motif_group = get_all_possible_motifs_for_this_motifs_group(motif)

            if motif_group in dict_motifgroups_all.keys():
                # Then skip, since this already done
                continue
            else:
                # if group_kind=="same":
                #     print(motif_group)
                #     assert False
                # Then go thru each motif in this group, and see how many instances there are
                list_n = [count_num_instances_this_motif(motif_this) for motif_this in motif_group]    
                # if motif_group==((('V-4-0', (1, 1)), ('Lcentered-3-0', (0, 0))), ):
                #     print(list_n)
                #     assert False

                dict_motifgroups_all[motif_group] = list_n

        
        #  Convert to a list of dicts
        dat = []
        for k, v in dict_motifgroups_all.items():
            n_unique_motifs_with_at_least_one_trial = sum([vv>0 for vv in v])
            n_trials_per_motif = v
            n_trials_summed_across_motifs = sum(v)
            dat.append({
                "n_unique_motifs_with_at_least_one_trial":n_unique_motifs_with_at_least_one_trial,
                "key":k, # motifgroup
                "n_trials_per_motif":n_trials_per_motif,
                "n_trials_summed_across_motifs":n_trials_summed_across_motifs
            })
            # print({
            #     "n_unique_motifs_with_at_least_one_trial":n_unique_motifs_with_at_least_one_trial,
            #     "key":k, # motifgroup
            #     "n_trials_per_motif":n_trials_per_motif,
            #     "n_trials_summed_across_motifs":n_trials_summed_across_motifs
            # })
            # assert False

        del dict_motifgroups_all # inefficient code, went thru this intermediate
            
        # sort motifgroups in order from most to least cases 
        dat = sorted(dat, key=lambda x:(x["n_unique_motifs_with_at_least_one_trial"], x["n_trials_summed_across_motifs"]), reverse=True)
        DatGroups[group_kind] = dat

    # Prune, only keep motifgroups that have enough trials to actually use these for an 
    # experiment.
    for which_group in list_group_kind:
        # 1) only keep cases with at least two unique motifs.
        if which_group!="same":
            list_dat = []
            for Dat in DatGroups[which_group]:
                if Dat["n_unique_motifs_with_at_least_one_trial"]>1:
                    list_dat.append(Dat)
            DatGroups[which_group] = list_dat
        
        # 2) Only keep cases with >1 trial
        list_dat = []
        for Dat in DatGroups[which_group]:
            if Dat["n_trials_summed_across_motifs"]>1:
                list_dat.append(Dat)
        DatGroups[which_group] = list_dat



    return DatGroups, list_group_kind
        

def extract_motifgroup_dict(DatGroups, which_group, motifgroup):
    """ Help to extract a single data dict, for this motifgroup
    - DatGroups, dict, where each key is a way of grouping (e..g, "diff_sequence") and
    each item is DAT the data (list of dicts, each dict corresponding to one group of 
    motifs (motifgroup))
    - which_group, string, indexes DatGroups
    - motifgroup, either:
    --- int index, motifgroup sorted by num cases (trials) in which case 0 means get the motifgroup with the most identified cases 
    --- tuple (motif), tuple of motifs, where motif is a tuple of tokens, where token is
    a tuple of features (e..g, (circle, [0,0]) for (shape, location)
    This is a group that indexes DAT
    RETURNS:
    - datdict, a dict for this motifgroup
    Or: None if cant find it
    """

    dat = DatGroups[which_group]
    
    # Get DAT
    if isinstance(motifgroup, int):
        # Then get in order of mostcases --> leastcases
        # Assumes it is already sorted
        if motifgroup>=len(dat):
            return None, None
        datdict = dat[motifgroup]
    else:
        # Then search for this motifgroup
        datdict = [d for d in dat if d["key"]==motifgroup][0]

    motifgroup = datdict["key"]
    return datdict, motifgroup



def get_all_motifs_found_for_this_motifgroup(DatGroups, which_group, motifgroup):
    """
    PARAMS:
    - DatGroups, dict, where each key is a way of grouping (e..g, "diff_sequence") and
    each item is DAT the data (list of dicts, each dict corresponding to one group of 
    motifs)
    - which_group, string, indexes DatGroups
    - motifgroup, either:
    --- int index, motifgroup sorted by num cases (trials) in which case 0 means get the motifgroup with the most identified cases 
    --- tuple (motif), see above.tuple, a group that indexes DAT
    RETURNS:
    - list_motifs, list of motifs for which >0 cases found, 
    in this motif group
    - motifgroup, the tuple holding all possible motifs
    - list_tokens, list of unique tokens across used motifs
    - list_shapes, list of unique shapes across used motifs
    """
    
    datdict, motifgroup = extract_motifgroup_dict(DatGroups, which_group, motifgroup)

    if datdict is None:
        return None, None

    # Get motifs found in this group
    list_motifs = []
    for motif, ncases in zip(datdict["key"], datdict["n_trials_per_motif"]):
        if ncases>0:
            list_motifs.append(motif)

    # tokens and shapes
    list_tokens = list(set([token for motif in list_motifs for token in motif]))
    list_shapes = list(set([token[0] for token in list_tokens]))

    return list_motifs, motifgroup, list_tokens, list_shapes
    
def get_inds_this_motif(motifs_all_dict, motif, random_subset=None):
    """ REturn list of indices that have this motif
    PARAMS:
    - motifs_all_dict, dict where keys are motifs, each vals are list of index where 
    this is found in dataset, where index is (trialnum, strokenum start)
    - random_subset, either None or int, if the latter, then extracts maximum this many trials
    RETURNS:
    - list_idxs, each is (trial, strokenum)
    """
    assert motif in motifs_all_dict.keys(), "asking for motif that was never detected for this dataset..."
    list_idxs = motifs_all_dict[motif]

    # Take random subset
    if random_subset and len(list_idxs)>random_subset:
        import random
        list_idxs = random.sample(list_idxs, random_subset)

    # break out trials and strokenums
    return list_idxs
    
def get_inds_this_motifgroup(DatGroups, which_group, motifs_all_dict, motifgroup,
        ntrials_per_motif=None):
    """ For this motifgroup, find all indices across all motifs
    that were found within this group.
    PARAMS;
    - DatGroups, see avbove
    - which_group, str, see above
    - motifs_all_dict, see above
    - motifgroup, either:
    --- int index, motifgroup sorted by num cases (trials) in which case 0 means get the motifgroup with the most identified cases 
    --- tuple (motif), see above.
    - ntrials_per_motif, int, if not None, then only keeps this many trials per motif, i.e., if any
    motifs have > this many trials, picks random subset
    RETURNS:
    - list_trials, list of ints into D.Dat
    - list_strokenums, list of strokenums taht start each motif instance,
    paired with list_trials
    - list_motifs_all, aligned with the above, the motif for each.
    """
        
    list_motifs, motifgroup = get_all_motifs_found_for_this_motifgroup(DatGroups, which_group, motifgroup)[:2]
    # for each motif, extract trials

    if list_motifs is None:
        # Then this motifgroup doesnt exist?
        return None, None, None

    inds_idxs_all = []
    inds_strokenums_all =[]
    list_motifs_all = []
    for motif in list_motifs:
        inds_idxs = get_inds_this_motif(motifs_all_dict, motif, 
            random_subset=ntrials_per_motif)
        # print(inds_idxs)
        # print(ntrials_per_motif)
        # assert False
        inds_idxs_all.extend(inds_idxs)
        for _ in inds_idxs:
            list_motifs_all.append(motif)

    assert len(inds_idxs_all)==len(list_motifs_all)
    return inds_idxs_all, list_motifs_all, motifgroup




###################### GETTING SETS OF TRIALS, FOR GENERATING EXPT


def expt_extract_motifgroups_to_regenerate(which_group, DatGroups, motifs_all_dict):
    """ Determine what are trials to generate for this motifgroup, collected from best to worst, 
    and using criteria which loosen as you go thru and run out of groups to try.
    PARAMS:
    - which_group, string key into DatGroups, will get data only for this group.
    """
    import random
    n_trials_get = 50 # how many unqiue trials to pull out. will stop at or slightly higher than this
    ntrials_per_motif = 2 # for each motif, how many unique trials to get that have this motif
    ntrials_max_per_motifgroup = 6 # useful, since diff_prims can have too many motifs per motifgroup...
    # ntrials_per_motif = 2 
    # ntrials_max_per_motifgroup = 5 # useful, since diff_prims can have too many motifs per motifgroup...

    def list_intersect(list1, list2):
        """ Returns True if any items intersect.
        otehrwies False
        """
        for x in list1:
            if x in list2:
                return True
        return False

    ALREADY_USED_MOTIFGROUPS = []
    ALREADY_USED_SHAPES = []
    ALREADY_USED_MOTIFGROUPS_INDS = []
    def get_best_motifgroup(which_group, list_criteria_inorder_THIS):
        for motifgroup_ind in range(len(DatGroups[which_group])):
            inds_idxs_all, list_motifs_all, motifgroup = get_inds_this_motifgroup(DatGroups, which_group, 
                                                                        motifs_all_dict, motifgroup_ind, 
                                                                      ntrials_per_motif=ntrials_per_motif)
            
            ### Take subset here before doing tests below.
            if len(inds_idxs_all)>ntrials_max_per_motifgroup:
                # Take a random subset
                # NOTE: this must be taken in order, to make sure gets distribution of motifs
                assert ntrials_max_per_motifgroup > 2*ntrials_per_motif, "otherwise wil not necessaril;y get multiple motifs"
                if False:
                    n = len(inds_idxs_all)
                    inds_sub = random.sample(range(n), ntrials_max_per_motifgroup)
                    inds_idxs_all = [inds_idxs_all[i] for i in inds_sub]
                    list_motifs_all = [list_motifs_all[i] for i in inds_sub]
                else:
                    inds_idxs_all = inds_idxs_all[:ntrials_max_per_motifgroup]
                    list_motifs_all = list_motifs_all[:ntrials_max_per_motifgroup]
            # print(list_motifs_all)
            # assert False


            #### Check constraints
            # 1) already gotten this group?
            if motifgroup in ALREADY_USED_MOTIFGROUPS:
                continue
                
            # 1b) Don't take this if only one trial found. i.e., need at least two to make a set
            if len(list(set(list_motifs_all)))<2:
                continue
                
            # 1c) Dont take this if the motifs are same for all trials. ingore this if your
            # goal is _actually_ to get same across trials
            if which_group!='same':
                if len(list(set(list_motifs_all)))==1:
                    for m in list_motifs_all:
                        print(m)
                    print("Confirm that indeed these are identical motifs")
                    assert False

            # 2) Other criteria
            # -- some Criteria that prune the motifs/indices
            # -- others that fail outright
            FAIL = False
            for crit in list_criteria_inorder_THIS:
                if crit=="dont_reuse_shape_complete":
                    # then throw out =motifs if it resuses ANY shape
                    inds_sub = []
                    for i, motif in enumerate(list_motifs_all):
                        shapes_in_motif = extract_shapes_in_motif(motif)
                        if not list_intersect(shapes_in_motif, ALREADY_USED_SHAPES):
                            # then keep
                            inds_sub.append(i)
                    # print(motifgroup_ind, crit, len(list_motifs_all), inds_sub)
                    inds_idxs_all = [inds_idxs_all[i] for i in inds_sub]
                    list_motifs_all = [list_motifs_all[i] for i in inds_sub]
                    # print(len(inds_idxs_all))

                elif crit=="dont_reuse_shape":
                    # Then throw out motif if it uses only shapes that are all arleady used (weaker constraint that above).
                    inds_sub = []
                    for i, motif in enumerate(list_motifs_all):
                        shapes_in_motif = extract_shapes_in_motif(motif)
                        if all([sh in ALREADY_USED_SHAPES for sh in shapes_in_motif]):
                            # Then continue, since both shapes are already used.
                            pass
                        else:
                            inds_sub.append(i)
                    # print(motifgroup_ind, crit, len(list_motifs_all), inds_sub)
                    inds_idxs_all = [inds_idxs_all[i] for i in inds_sub]
                    list_motifs_all = [list_motifs_all[i] for i in inds_sub]
                    # print(len(inds_idxs_all))
                elif crit=="within_motif_diff_shapes":
                    # Prune motifs that are (shape1, shape1), eg.., circle to circle.
                    inds_sub = []
                    for i, motif in enumerate(list_motifs_all):
                        shapes_in_motif = extract_shapes_in_motif(motif)
                        if len(list(set(shapes_in_motif)))==1:
                            continue
                        else:
                            # then keep
                            inds_sub.append(i)
                    # print(motifgroup_ind, crit, len(list_motifs_all), inds_sub)

                    inds_idxs_all = [inds_idxs_all[i] for i in inds_sub]
                    list_motifs_all = [list_motifs_all[i] for i in inds_sub]
                elif crit=="mult_trials_exist_for_at_least_two_motifs":
                    # Then check that at leat 2 of the motif is associated with at least 2 trials
                    n_good = 0
                    for motif in set(list_motifs_all): # [motif1,, ..., motif2, ...]
                        n = len([m for m in list_motifs_all if m==motif]) # num instances (trials) for this motif
                        if n>1:
                            n_good+=1
                    if n_good<2:
                        FAIL = True
                else:
                    print(crit)
                    assert False
            if FAIL:
                continue
            
            # 1b) Don't take this if only one trial found. i.e., need at least two to make a set
            if len(inds_idxs_all)<2:
                continue
                
            # 1c) Dont take this if the motifs are same for all trials. ingore this if your
            # goal is _actually_ to get same across trials
            if which_group!='same':
                if len(list(set(list_motifs_all)))==1:
                    for m in list_motifs_all:
                        print(m)
                    print("Confirm that indeed these are identical motifs")
                    assert False

            ### Success! Keep this.
            print("Success! taking motifgroup with index: ", motifgroup_ind)
            if len(inds_idxs_all)>ntrials_max_per_motifgroup:
                # Take a random subset
                n = len(inds_idxs_all)
                inds_sub = random.sample(range(n), ntrials_max_per_motifgroup)
                inds_idxs_all = [inds_idxs_all[i] for i in inds_sub]
                list_motifs_all = [list_motifs_all[i] for i in inds_sub]
            
            return inds_idxs_all, list_motifs_all, motifgroup, motifgroup_ind
        
        # If got here, then failed to find anything
        return None, None, None, None


    ################### 
    # - put most important to the right.
    # list_criteria_inorder = ["mult_trials_exist_per_motif", "dont_reuse_shape", "within_motif_diff_shapes"] # order: will prune from left to right
    list_criteria_inorder = ["dont_reuse_shape_complete", "mult_trials_exist_for_at_least_two_motifs", "dont_reuse_shape", "within_motif_diff_shapes"] # order: will prune from left to right
    # list_criteria_inorder = [ "dont_reuse_shape", "within_motif_diff_shapes"] # order: will prune from left to right

    DatTrials = {}
    DatGroupsUsedSimple = {which_group:[]} # keep track of which motifgroups are used

    for i in range(len(list_criteria_inorder)+1): # +1 so runs last time with no constraints
        list_criteria_inorder_THIS = list_criteria_inorder[i:]
        print("Criteria: ", list_criteria_inorder_THIS)
        
        inds_idxs_all, list_motifs_all, motifgroup, motifgroup_ind = get_best_motifgroup(which_group, 
            list_criteria_inorder_THIS)
        while inds_idxs_all is not None:
            
            ### For each trial, save this iformation
            inds_trials_all = [x[0] for x in inds_idxs_all]
            inds_strokenums_all = [x[1] for x in inds_idxs_all]
            for trial, strokenum, motif in zip(inds_trials_all, inds_strokenums_all, list_motifs_all):
                item = (which_group, motifgroup, motif, strokenum)
                if trial in DatTrials.keys():
                    DatTrials[trial].append(item)
                else:
                    DatTrials[trial] = [item]

            # Track which groups are used
            ntrials = len(inds_idxs_all)
            DatGroupsUsedSimple[which_group].append({"motifgroup":motifgroup, "ntrials":ntrials, "inds_idxs_all":inds_idxs_all, "list_motifs_all":list_motifs_all})
            
            # Track what already used
            ALREADY_USED_MOTIFGROUPS.append(motifgroup)
            ALREADY_USED_MOTIFGROUPS_INDS.append(motifgroup_ind)

            for motif in list_motifs_all:
                shapesthis = extract_shapes_in_motif(motif)
                ALREADY_USED_SHAPES.extend(shapesthis)

            ### Try to get another motifgroup
            inds_idxs_all, list_motifs_all, motifgroup, motifgroup_ind = get_best_motifgroup(which_group, 
                list_criteria_inorder_THIS)
        print("Got this many motifgorups so far: ", len(ALREADY_USED_MOTIFGROUPS))
    
    return DatTrials, DatGroupsUsedSimple, ALREADY_USED_MOTIFGROUPS_INDS
        
            
def prune_dataset_unique_shapes_only(D):
    """ Only keep trials where no shape is repeated within the trial
    """
    assert False,'in progress. copied from ntoebook...'
    indtrial_keep = []
    print(len(D.Dat))
    for indtrial in range(len(D.Dat)):
        shapes = D.taskclass_shapes_extract(indtrial)
        nshape_uniq = len(set(shapes))
        n = len(shapes)
        if n - nshape_uniq > 1:
            print(indtrial, '-', shapes)
        else:
            indtrial_keep.append(indtrial)

    print(len(indtrial_keep))
    D.subsetDataframe(indtrial_keep)