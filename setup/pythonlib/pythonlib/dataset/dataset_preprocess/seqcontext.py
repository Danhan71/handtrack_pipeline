""" Dataset preprocessed so that 
has information about sequential context
"""


import pandas as pd
import numpy as np

def preprocess_dataset(D, n_strok_max = 8, version="tokens", plot_examples=False):
    """ Extract new columns into self.Dat, for each trial noting sequence 
    context inforamtion ,such as n strokes, shape of first stroke, etc
    PARAMS:
    - n_strok_max, int, max num strokes to etract info for, large is fine.
    - version, where to find stroke shape labels from.
    NOTE: This works for character (clust labels) too.
        shape:
        - SP, PIG --> easy, use taskstrokes.
        - CHAR --> use clust labels (behtokens)

        shape semantic:
        - SP, PIG --> easy, use taskstrokes.
        - CHAR --> use behstrokes, then find that in prim database, then label that.
    """
    
    D.behclass_preprocess_wrapper()

    datall = []
    list_dat = []
    for ind in range(len(D.Dat)):

        dat = {}
        # trial-level info
        # # trialcode = D.Dat.iloc[ind]["trialcode"]
        # tokens_beh_using_beh = None
        # if version=="tokens_using_beh":
        #     # Suimiarl to tokens, but using recomputed location info from beh (as well as older tokens, from task).
        #     tokens_beh = D.taskclass_tokens_extract_wrapper(ind, which_order="beh_using_task_data")
        COLLECT_LOCATION = True
        if version == "tokens":
            # use the tokens, e.g., for PIG
            tokens_beh_using_task = D.taskclass_tokens_extract_wrapper(ind, which_order="beh_using_task_data")
            tokens_beh_using_beh = D.taskclass_tokens_extract_wrapper(ind, which_order="beh_using_beh_data")
            tokens_task = D.taskclass_tokens_extract_wrapper(ind, which_order="task")
            # COLLECT_LOCATION = True
        # elif version == "char_clust_labels":
        #     assert False, "this is obsolete, instead, shapes are already integrated into tokens, in general.preprocess"
        #     assert False, "otherwise this runs into problem becuase not chagin shapes eraly enoguh."
        #     # use labels extracted from clkustering strokes, usualyl for chars
        #     shape_seq = D.Dat.iloc[ind]["charclust_shape_seq"] # list of str.
        #     # Fake it into list of dicts
        #     tokens_beh_using_task = [{"shape":sh} for sh in shape_seq]
        #     COLLECT_LOCATION = False
        else:
            print(version)
            assert False

        # NOTE: tokens_beh_using_task is not necesarily length of strokes_beh
        if False:
            if len(tokens_beh_using_task)!=len(D.Dat.iloc[ind]["strokes_beh"]):
                # This occurs when datsegs throws out a beh stroke that doesnt match
                # any task stroke.
                print("weird: ", ind)
                print(len(tokens_beh_using_task))
                print(len(D.Dat.iloc[ind]["strokes_beh"]))
                print(tokens_beh_using_task)
                D.plotSingleTrial(ind)
                assert False, "why?"

        nstrokes_beh = len(tokens_beh_using_task)
        dat["seqc_nstrokes_beh"] = nstrokes_beh
        dat["seqc_nstrokes_task"] = len(tokens_task)
        # dat["nstrokes_task"] = len(D.Dat.iloc[ind]["strokes_task"])

        ############## COLLECT ITEMS FROM TOKENS
        # shapes in order
        try:
            GET_CENTER = "center_binned" in tokens_beh_using_task[0].keys()
            GET_SHAPE_SEM = "shape_semantic" in tokens_beh_using_task[0].keys()

            GET_ANGLE = "angle_binned" in tokens_beh_using_beh[0].keys()
            GET_LOC = "loc_on" in tokens_beh_using_beh[0].keys()
            GET_LOC_WITHIN = False # Since this should be replaced with cluster within ...
            # GET_LOC_WITHIN = "locon_bin_in_loc" in tokens_beh_using_beh[0].keys()

            GET_LOC_CLUST = "loc_on_clust" in tokens_beh_using_beh[0].keys()

            for j in range(n_strok_max):
                if j < nstrokes_beh:
                    # Varialbes using task-stroke (aligned to beh)
                    tok = tokens_beh_using_task[j]
                    assert isinstance(tok["shape"], str)
                    dat[f"seqc_{j}_shape"] = tok["shape"]
                    dat[f"seqc_{j}_loc"] = tok["gridloc"]
                    dat[f"seqc_{j}_loc_local"] = tok["gridloc_local"]
                    if GET_CENTER:
                        dat[f"seqc_{j}_center_binned"] = tok["center_binned"]
                    if GET_SHAPE_SEM:
                        dat[f"seqc_{j}_shapesem"] = tok["shape_semantic"]
                        dat[f"seqc_{j}_shapesemcat"] = tok["shape_semantic_cat"]

                    # Variables using beh stroke (algined to beh)
                    tok = tokens_beh_using_beh[j]
                    if GET_ANGLE:
                        dat[f"seqc_{j}_angle_binned"] = tok["angle_binned"]
                        dat[f"seqc_{j}_angle"] = tok["angle"]
                    if GET_LOC:
                        dat[f"seqc_{j}_locon"] = tok["loc_on"] # stroke onset location.
                        dat[f"seqc_{j}_locx"] = tok["loc_on"][0]
                        dat[f"seqc_{j}_locy"] = tok["loc_on"][1]
                        dat[f"seqc_{j}_locon_binned"] = tok["loc_on_binned"] # sol, binned over entire page.
                    if GET_LOC_WITHIN:
                        dat[f"seqc_{j}_locon_bin_in_loc"] = tok["locon_bin_in_loc"] # sol, binned over entire page.
                    if GET_LOC_CLUST:
                        dat[f"seqc_{j}_loc_on_clust"] = tok["loc_on_clust"] # sol, binned over entire page.
                        dat[f"seqc_{j}_loc_off_clust"] = tok["loc_off_clust"] # sol, binned over entire page.

                else:
                    # Use same type as the actuals.
                    dat[f"seqc_{j}_shape"] = "IGN"
                    dat[f"seqc_{j}_loc"] = ("IGN", "IGN")
                    dat[f"seqc_{j}_loc_local"] = ("IGN", "IGN")
                    if GET_CENTER:
                        dat[f"seqc_{j}_center_binned"] = ("IGN", "IGN")
                    if GET_SHAPE_SEM:
                        dat[f"seqc_{j}_shapesem"] = "IGN"
                        dat[f"seqc_{j}_shapesemcat"] = "IGN"

                    # --------------------
                    if GET_ANGLE:
                        dat[f"seqc_{j}_angle_binned"] = "IGN"
                        dat[f"seqc_{j}_angle"] = np.nan

                    if GET_LOC:
                        dat[f"seqc_{j}_locon"] = (np.nan, np.nan)
                        dat[f"seqc_{j}_locon_binned"] = ("IGN", "IGN")
                        dat[f"seqc_{j}_locx"] = np.nan
                        dat[f"seqc_{j}_locy"] = np.nan
                    if GET_LOC_WITHIN:
                        dat[f"seqc_{j}_locon_bin_in_loc"] = ("IGN", "IGN")
                    if GET_LOC_CLUST:
                        dat[f"seqc_{j}_loc_on_clust"] = "IGN"
                        dat[f"seqc_{j}_loc_off_clust"] = "IGN"

                # Conjunctions
                dat[f"seqc_{j}_loc_shape"] = (dat[f"seqc_{j}_loc"], dat[f"seqc_{j}_shape"]) # conjunction
                if False:
                    # do this later, or else is innacurate if gridloc is modified.
                    dat[f"seqc_{j}_shapeloc"] = (dat[f"seqc_{j}_loc"], dat[f"seqc_{j}_shape"]) # conjunction

        except Exception as err:
            print(ind, j)
            print(tok.keys())
            print("didnt add this feature to tok?")
            raise err

        list_dat.append(dat)

    ###########################
    dfdat = pd.DataFrame(list_dat)
    if True:
        # 580ms
        # Otheriwse shows unecesary waribng: PerformanceWarning: DataFrame is highly fragmented.
        # This is usually the result of calling `frame.insert` many times, which has poor performance.
        # Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
        # https://stackoverflow.com/questions/68292862/performancewarning-dataframe-is-highly-fragmented-this-is-usually-the-result-o
        from warnings import simplefilter
        simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

        # Put back into D.Dat
        # pd.concat()
        # # print(pd.merge(D.Dat, dfdat))
        # # D.Dat = pd.merge(D.Dat, dfdat)
        # D.Dat = pd.concat([D.Dat, dfdat], axis=1)
        # # print(D.Dat.columns)

        for col in dfdat.columns:
            D.Dat[col] = dfdat[col]
    else:
        # 480ms
        from pythonlib.tools.pandastools import merge_subset_indices_prioritizing_second
        D.Dat = merge_subset_indices_prioritizing_second(D.Dat, dfdat, index_col=None)
    # assert False, "merge it"
    # D.Dat["seqc_nstrokes_beh"] = dfdat["nstrokes_beh"]
    # D.Dat["seqc_nstrokes_task"] = dfdat["nstrokes_task"]
    # for i in range(n_strok_max):
    #     D.Dat[f"seqc_{i}_shape"] = dfdat[f"{i}_shape"]
    #     if COLLECT_LOCATION:
    #         D.Dat[f"seqc_{i}_loc"] = dfdat[f"{i}_loc"]
    #         D.Dat[f"seqc_{i}_loc_local"] = dfdat[f"{i}_loc_local"]
    #         D.Dat[f"seqc_{i}_loc_shape"] = dfdat[f"{i}_loc_shape"]

    if plot_examples:
        # Print/plot showing taskshape config
        import random
        n = 5
        inds = random.sample(range(len(D.Dat)), n)
        fig, axes, idxs = D.plotMultTrials2(inds)
        fig, axes, idxs = D.plotMultTrials2(inds, "strokes_task")
        # D.Dat.loc[inds, ["taskconfig_loc", "taskconfig_shp", "taskconfig_shploc"]]

        for feat in ["shape", "shapesem", "loc", "center_binned", "locon", "locon_binned"]:
            for j in range(n_strok_max):
                print("------ ", feat, j)
                # D.Dat[f"seqc_{j}_{feat}"]
                print(D.Dat.loc[inds, [f"seqc_{j}_{feat}"]].values)
