"""
5/20/24 - To deal with (usually single prim) expts with single sahpes, varying in graded fashion (therefore called
"psychometric"), doing the extraction of the psycho params and plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pythonlib.tools.plottools import savefig
import os
import seaborn as sns


def params_extract_psycho_groupings_manual(animal, date):
    """
    Repo of params for each (animal ,date), for psychogood. 
    See code within for how to generate these params.
    
    """
    PARAMS = []
    date = int(date)

    if animal == "Diego" and date == 240523:
        if False:
            # 0.33 (not enought data)
            # PARAMS.append(
            # {'los_within': ('singleprims_psycho', 5, 25), 
            # 'los_base_1': ('singleprims_psycho', 6, 2), 
            # 'los_base_2': ('singleprims_psycho', 6, 16), 
            #  'psycho_params': {'psycho_ver': 'extra_tforms_each_prim', 'idx_prim': 0, 'tform_key': 'sx'}}
            # )

            0.5
            PARAMS.append(
            {'los_within': ('singleprims_psycho', 5, 61), 
            'los_base_1': ('singleprims_psycho', 6, 2), 
            'los_base_2': ('singleprims_psycho', 6, 16), 
            'psycho_params': {'psycho_ver': 'extra_tforms_each_prim', 'idx_prim': 0, 'tform_key': 'sx'}}
            )

            # 0.66
            PARAMS.append(
            {'los_within': ('singleprims_psycho', 5, 67), 
            'los_base_1': ('singleprims_psycho', 6, 2), 
            'los_base_2': ('singleprims_psycho', 6, 16), 
            'psycho_params': {'psycho_ver': 'extra_tforms_each_prim', 'idx_prim': 0, 'tform_key': 'sx'}}
            )

            # 0.83
            PARAMS.append(
            {'los_within': ('singleprims_psycho', 5, 19), 
            'los_base_1': ('singleprims_psycho', 6, 2), 
            'los_base_2': ('singleprims_psycho', 6, 16), 
            'psycho_params': {'psycho_ver': 'extra_tforms_each_prim', 'idx_prim': 0, 'tform_key': 'sx'}}
            )
        else:
            # Combine all...
            PARAMS.append(
            {'los_within': ('singleprims_psycho', 5, 19), 
            'los_base_1': ('singleprims_psycho', 6, 2), 
            'los_base_2': ('singleprims_psycho', 6, 16), 
            'psycho_params': {'psycho_ver': 'extra_tforms_each_prim', 'idx_prim': [0,2], 'tform_key': 'sx'}}
            )


        PARAMS.append(
        {'los_within': ('singleprims_psycho', 5, 16), 
        #  'los_base_1': ('singleprims_psycho', 5, 19), 
        #  'los_base_2': ('singleprims_psycho', 5, 32), 
        'los_base_1': ('singleprims_psycho', 6, 2), 
        'los_base_2': ('singleprims_psycho', 6, 12), 
        'psycho_params': {'psycho_ver': 'extra_tforms_each_prim', 'idx_prim': 2, 'tform_key': 'sx'}
        }
        )

        PARAMS.append(
        {'los_within': ('singleprims_psycho', 5, 18), 
        'los_base_1': ('singleprims_psycho', 6, 11),  
        'los_base_2': ('singleprims_psycho', 6, 5), 
        'psycho_params': {'psycho_ver': 'attachpt1_interp', 'idx_motif_char_flex': 0, 'idx_rel': 1, 'convert_extra_tforms_to_sign': True}
        }
        )

        PARAMS.append(
        {'los_within': ('singleprims_psycho', 5, 30), 
        'los_base_1': ('singleprims_psycho', 6, 6), 
        'los_base_2': ('singleprims_psycho', 6, 17),  
        'psycho_params': {'psycho_ver': 'attachpt1_interp', 'idx_motif_char_flex': 0, 'idx_rel': 2, 'idx_rel_within': 1, 'convert_extra_tforms_to_sign': True}}
        )

        PARAMS.append(
        {'los_within': ('singleprims_psycho', 5, 14), 
        'los_base_1': ('singleprims_psycho', 6, 3), 
        'los_base_2': ('singleprims_psycho', 6, 12),  
        'psycho_params': {'psycho_ver': 'attachpt1_interp', 'idx_motif_char_flex': 0, 'idx_rel': 1, 'idx_rel_within': 1, 'convert_extra_tforms_to_sign': False}}
        )

        PARAMS.append(
        {'los_within': ('singleprims_psycho', 5, 44), 
        'los_base_1': ('singleprims_psycho', 6, 12), 
        'los_base_2': ('singleprims_psycho', 6, 1),  
        'psycho_params': {'psycho_ver': 'extra_tforms_each_prim', 'idx_prim': 1, 'tform_key': 'th'}
        }
        )

        PARAMS.append(
        {'los_within': ('singleprims_psycho', 5, 33), 
        'los_base_1': ('singleprims_psycho', 6, 14), 
        'los_base_2': ('singleprims_psycho', 6, 15),  
        'psycho_params': {'psycho_ver': 'extra_tforms_each_prim', 'idx_prim': 1, 'tform_key': 'th'}
        }
        )


        los_allowed_to_miss = [('singleprims', 86, 1),
            ('singleprims', 86, 2),
            ('singleprims', 86, 3),
            ('singleprims', 86, 4),
            ('singleprims', 86, 5),
            ('singleprims', 86, 6),
            ('singleprims', 86, 7),
            ('singleprims', 86, 8),
            ('singleprims', 86, 9),
            ('singleprims', 86, 10),
            ('singleprims', 86, 11),
            ('singleprims', 86, 12),
            ('singleprims', 86, 13),
            ('singleprims', 86, 14),
            ('singleprims', 86, 15),
            ('singleprims', 86, 16),
            ('singleprims', 86, 17),
            ('singleprims', 86, 18),
            ('singleprims', 86, 19),
            ('singleprims', 86, 20),
            ('singleprims', 86, 21),
            ('singleprims', 86, 22),
            ('singleprims', 86, 23),
            ('singleprims', 86, 24),
            ('singleprims', 86, 25),
            ('singleprims', 86, 26),
            ('singleprims', 86, 27),
            ('singleprims', 86, 28),
            ('singleprims', 86, 29),
            ('singleprims', 86, 30),
            ('singleprims', 86, 31),
            ('singleprims', 86, 32),
            ('singleprims', 86, 33),
            ('singleprims', 86, 34),
            ('singleprims', 86, 35),
            ('singleprims', 86, 36),
            ('singleprims', 86, 37),
            ('singleprims', 86, 38),
            ('singleprims', 86, 39),
            ('singleprims', 86, 40),
            ('singleprims', 86, 41),
            ('singleprims', 86, 42),
            ('singleprims', 86, 43),
            ('singleprims', 86, 44),
            ('singleprims', 86, 45),
            ('singleprims', 86, 46),
            ('singleprims', 86, 47),
            ('singleprims', 86, 48),
            ('singleprims_psycho', 6, 13),
            ('singleprims_psycho', 6, 18),
        ]   
    elif animal == "Pancho" and date == 240524:
        #### GOOD
        PARAMS.append({
            "los_base_1":("singleprims_psycho", 7, 8),
            "los_within":("singleprims_psycho", 8, 11),
            "los_base_2":("singleprims_psycho", 7, 10),
            "psycho_params":{
                "psycho_ver":"extra_tforms_each_prim",
                "idx_prim": 1,
                "tform_key": "sx"
            }
        })

        PARAMS.append({
            "los_base_1":("singleprims_psycho", 7, 8),
            "los_within":("singleprims_psycho", 8, 15),
            "los_base_2":("singleprims_psycho", 7, 4),
            "psycho_params":{
                "psycho_ver":"extra_tforms_each_prim",
                "idx_prim": 0,
                "tform_key": "sx"}
        })

        PARAMS.append({
            "los_base_1":("singleprims_psycho", 7, 18),
            "los_within":("singleprims_psycho", 8, 12),
            "los_base_2":("singleprims_psycho", 7, 15),
            "psycho_params":{
                "psycho_ver":"extra_tforms_each_prim",
                "idx_prim": 0,
                "tform_key": "sx"
            }    
        })

        PARAMS.append({
            "los_base_1":("singleprims_psycho", 7, 10),
            "los_within":("singleprims_psycho", 8, 6),
            "los_base_2":("singleprims_psycho", 7, 17),
            "psycho_params":{
                "psycho_ver":"attachpt1_interp",
                "idx_motif_char_flex": 0,
                "idx_rel": 1,
            }
        })


        PARAMS.append({
            "los_base_1":("singleprims_psycho", 7, 14),
            "los_within":("singleprims_psycho", 8, 7),
            "los_base_2":("singleprims_psycho", 7, 7),
            "psycho_params":{
                "psycho_ver":"attachpt1_interp",
                "idx_motif_char_flex": 0,
                "idx_rel": 1,
            }    
        })

        PARAMS.append({
            "los_base_1":("singleprims_psycho", 7, 20),
            "los_within":("singleprims_psycho", 8, 13),
            "los_base_2":("singleprims_psycho", 7, 22),
            # Here two things are varying: interp and extra tforms. Solve this problem by convering tform to its sign.
            "psycho_params":{
                "psycho_ver":"attachpt1_interp",
                "idx_motif_char_flex": 0, # Usuall dont change (if this is one "prim"), i.e,. this is the outer index.
                "idx_rel": 1, # This is the index into the relations, within each plan
                "exclude_extra_tform_values":False,
                "exclude_interp_values":False,
                "convert_extra_tforms_to_sign":True 
            }    
        })

        PARAMS.append({
            "los_base_1":("singleprims_psycho", 7, 5),
            "los_within":("singleprims_psycho", 8, 16),
            "los_base_2":("singleprims_psycho", 7, 9),
            "psycho_params":{
                "psycho_ver":"attachpt1_interp",
                "idx_motif_char_flex": 0, # Usuall dont change (if this is one "prim"), i.e,. this is the outer index.
                "idx_rel": 1, # This is the index into the relations, within each plan
                "exclude_extra_tform_values":False,
                "exclude_interp_values":False,
                "convert_extra_tforms_to_sign":True 
            }    
        })

        PARAMS.append({
            "los_base_1":("singleprims_psycho", 7, 3),
            "los_within":("singleprims_psycho", 8, 35),
            "los_base_2":("singleprims_psycho", 7, 17),
            "psycho_params":{
                "psycho_ver":"extra_tforms_each_prim",
                "idx_prim": [0,1],
                "tform_key": "th"
            }    
        })


        # Arc --> For each one find both locations.
        PARAMS.append({
            "los_base_1":("singleprims", 114, 9), # arc
            "los_within":("singleprims_psycho", 9, 1),
            "los_base_2":("singleprims", 114, 30), # circle
            "psycho_params":{
                "psycho_ver":"morph_frac",
                "replace_morph_params_with_binary_whether_exists":True,
            }
        })

        los_allowed_to_miss = [('singleprims', 114, 1),
            ('singleprims', 114, 2),
            ('singleprims', 114, 3),
            ('singleprims', 114, 4),
            ('singleprims', 114, 5),
            ('singleprims', 114, 6),
            ('singleprims', 114, 7),
            ('singleprims', 114, 8),
            ('singleprims', 114, 10),
            ('singleprims', 114, 11),
            ('singleprims', 114, 12),
            ('singleprims', 114, 13),
            ('singleprims', 114, 14),
            ('singleprims', 114, 15),
            ('singleprims', 114, 16),
            ('singleprims', 114, 17),
            ('singleprims', 114, 19),
            ('singleprims', 114, 20),
            ('singleprims', 114, 21),
            ('singleprims', 114, 22),
            ('singleprims', 114, 23),
            ('singleprims', 114, 24),
            ('singleprims', 114, 25),
            ('singleprims', 114, 26),
            ('singleprims', 114, 28),
            ('singleprims', 114, 29),
            ('singleprims', 114, 31),
            ('singleprims', 114, 32),
            ('singleprims', 114, 33),
            ('singleprims', 114, 34),
            ('singleprims_psycho', 7, 2),
            ('singleprims_psycho', 7, 6),
            ('singleprims_psycho', 7, 12),
            ('singleprims_psycho', 7, 21)]


    else:
        assert False



    return PARAMS, los_allowed_to_miss

def preprocess_and_plot(D, var_psycho, PLOT=True):
    """
    Wrapper to do preprocess and plots, two kinds, either with strong restriction on strokes (only
    clean like singleprims) or lenient, including all even abort (useful to see vacillation).
    """

    if var_psycho == "angle":
        # OLD, should place into below, general purpose methods.
        SAVEDIR = D.make_savedir_for_analysis_figures_BETTER("psycho_singleprims")
        SAVEDIR = f"{SAVEDIR}/{var_psycho}"

        DS, DSlenient, map_shape_to_psychoparams = preprocess(D, var_psycho)

        if PLOT:
            # Plot both strict and lenient
            plot_overview(DS, D, f"{SAVEDIR}/clean_strokes_singleprim", var_psycho=var_psycho)
            plot_overview(DSlenient, D, f"{SAVEDIR}/lenient_strokes", var_psycho=var_psycho)
        
        #####################
        # Also make plot of mean_sim_score (trial by trial var)
        savedir = f"{SAVEDIR}/using_primitivenessv2"
        os.makedirs(savedir, exist_ok=True)

        from pythonlib.dataset.dataset_analy.primitivenessv2 import preprocess_plot_pipeline
        # First, extract all the derived metrics
        PLOT=True
        plot_methods = ("tls", "drw")
        DSnew, _, dfres, grouping = preprocess_plot_pipeline(D, PLOT=PLOT, plot_methods=plot_methods)
        _apply_psychoparams_to_ds(DSnew.Dat, map_shape_to_psychoparams, var_psycho)
        _apply_psychoparams_to_ds(dfres, map_shape_to_psychoparams, var_psycho)

        # Make plots
        fig = sns.catplot(data=dfres, x="angle_idx_within_shapeorig", y="mean_sim_score", col="shapeorig", 
            kind="point", sharey=True)
        savefig(fig, f"{savedir}/mean_sim_score-1.pdf")
        fig = sns.catplot(data=dfres, x="angle_idx_within_shapeorig", y="mean_sim_score", col="shapeorig", 
            alpha=0.5, sharey=True)
        savefig(fig, f"{savedir}/mean_sim_score-2.pdf")

        plt.close("all")
    
        return DS, DSlenient, map_shape_to_psychoparams
    
    elif var_psycho == "cont_morph":
        DS, DSmorphsets, map_morph_set_idx_to_shapes, _, _, SAVEDIR = preprocess_cont_morph(D)
        plot_overview_cont_morph(D, DS, DSmorphsets, map_morph_set_idx_to_shapes, SAVEDIR)

        return DS, DSmorphsets, map_morph_set_idx_to_shapes
    
    else:
        print(var_psycho)
        assert False


def _apply_psychoparams_to_ds(df, map_shape_to_psychoparams, var_psycho):
    """
    """

    var_psycho_unique = f"{var_psycho}_unique"
    var_psycho_str = f"{var_psycho}_str"
    var_psycho_idx = f"{var_psycho}_idx_within_shapeorig"

    names = ["shapeorig_psycho", "shapeorig", var_psycho_unique, var_psycho_str, var_psycho_idx]
    for i, na in enumerate(names):
        # D.Dat[f"seqc_0_{na}"] = [map_shape_to_psychoparams[sh][i] for sh in D.Dat["seqc_0_shape"]]
        df[na] = [map_shape_to_psychoparams[sh][i] for sh in df["shape"]]
        # DSlenient.Dat[na] = [map_shape_to_psychoparams[sh][i] for sh in DSlenient.Dat["shape"]]

def preprocess(D, var_psycho="angle", SANITY=True):
    """
    For psychometric variables, such as angle, determines for each stroke what its original shape is, and 
    its angle relative to that shape. 
    """
    from pythonlib.tools.pandastools import find_unique_values_with_indices
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from pythonlib.tools.pandastools import append_col_with_index_of_level, append_col_with_index_of_level_after_grouping
    from pythonlib.tools.pandastools import grouping_print_n_samples
    from pythonlib.tools.checktools import check_objects_identical
    from pythonlib.dataset.dataset_strokes import preprocess_dataset_to_datstrokes
    from math import pi

    assert var_psycho == "angle", "coded only for this currently."

    # First, for each token, extract variables that reflect the psycho variable/param.
    token_ver = "task"
    for i, row in D.Dat.iterrows():
        # for each shape get its concrete params
        Tk = D.taskclass_tokens_extract_wrapper(i, token_ver, return_as_tokensclass=True)
        Tk.features_extract_wrapper(["loc_on", "angle"], angle_twind=[0, 2])
    df = D.tokens_extract_variables_as_dataframe(["shape", "loc_on", "angle", "Prim", "gridloc"], token_ver)
    
    # Extract the original shape, which should have been overwritten in rprepovessing, but is useufl as a category
    # to anchor the variations.
    list_shape_orig = []
    for P in df["Prim"]:
        list_shape_orig.append(P.shape_oriented())
    df["shape_orig"] = list_shape_orig

    # For each shape_orig, get an ordered indices for angle. 
    if True:
        ################### FIND THE ANGLE RELATIVE TO THE BASE_PRIM'S ANGLE. THis will be used for defining the psycho index
        # Get list of all shapes which dont have extra tform
        base_shapes = []
        for ind in range(len(D.Dat)):
            tforms = D.taskclass_extract_prims_extra_params_tforms(ind)
            tform_first_stroke = tforms[0]
            if len(tform_first_stroke)==0:
                base_shapes.append(D.taskclass_shapes_extract(ind)[0])
        base_shapes = list(set(base_shapes))

        df["is_base_shape"] = df["shape"].isin(base_shapes)

        # For each shape_orig, find its one base shape
        from pythonlib.tools.pandastools import find_unique_values_with_indices

        map_shape_orig_to_angle_base = {}
        for grp in df.groupby(["shape_orig"]):
            shape_orig = grp[0][0]
            dfthis = df[(df["shape_orig"] == shape_orig) & (df["is_base_shape"]==True)]
            dfthis = grp[1][(grp[1]["is_base_shape"]==True)]

            # angle_base = dfthis["angle"].unique()
            # angle_base
            unique_values, _, _ = find_unique_values_with_indices(dfthis, "angle")
            assert len(unique_values)==1, "This means there are multiple trials without tform, which are thus called base prims, for this shape_orig"

            angle_base = unique_values[0]

            map_shape_orig_to_angle_base[shape_orig] = angle_base

        # Map the base angle to each shape_orig
        df["angle_base"] = df["shape_orig"].map(map_shape_orig_to_angle_base)

        # Calculate the relative angle
        df["angle_rel_base"] = df["angle"] - df["angle_base"]

        # Adjust angles that are negative
        df.loc[df["angle_rel_base"] < 0, "angle_rel_base"] += 2 * pi
        df.loc[df["angle_rel_base"] > 0.99*2*pi, "angle_rel_base"] = 0 # anything 6.28 should actually be 0.

        # HACK - change name to angle
        HACK = True
        if HACK:
            df["angle"] = df["angle_rel_base"]
            var_psycho = f"angle"
        else:
            var_psycho = f"{var_psycho}_rel_base"


    # col = var_psycho
    var_psycho_unique = f"{var_psycho}_unique"
    var_psycho_str = f"{var_psycho}_str"
    var_psycho_idx = f"{var_psycho}_idx_within_shapeorig"
    unique_values, indices, map_index_to_value = find_unique_values_with_indices(df, var_psycho, 
        append_column_with_unique_values_colname=var_psycho_unique)
    df[var_psycho_str] = [f"{a:.2f}" for a in df[f"{var_psycho}_unique"]]
    # df[var_psycho_str] = [f"{a:.2f}" for a in df[f"{var_psycho}"]]
    # print(var_psycho)
    # assert False
    
    ##########################

    if SANITY:
        df["shape_hash"] = [P.label_classify_prim_using_stroke(return_as_string=True, version="hash") for P in df["Prim"]]
        assert np.all(df["shape_hash"] == df["shape"]), "for psycho, you need to turn on reclassify_shape_using_stroke_version=hash in preprocess/general"

    # Redefine each token to have a unique identifier (shape-psycho)
    df = append_col_with_grp_index(df, ["shape_orig", var_psycho_str], "shape_pscho", True)

    # Convert psycho var levels in to indices that start at 0 for each shape-orig.
    df = append_col_with_index_of_level_after_grouping(df, ["shape_orig"], var_psycho_str, var_psycho_idx)

    # Finally, get map from shape to psycho stuff
    map_shape_to_psychoparams = {}
    for i, row in df.iterrows():
        shape = row["shape"]
        params = (row["shape_pscho"], row["shape_orig"], row[var_psycho_unique], row[var_psycho_str], row[var_psycho_idx])
        if shape in map_shape_to_psychoparams:
            if not check_objects_identical(map_shape_to_psychoparams[shape], params):
                print(shape)
                print(map_shape_to_psychoparams[shape])
                print(params)
                assert False
        else:
            map_shape_to_psychoparams[shape] = params

    # grouping_print_n_samples(df, ["shape", "shape_pscho"])
    # grouping_print_n_samples(df, ["shape_pscho", "shape_hash", "angle_idx_within_shapeorig", "gridloc", "shape"])
    if SANITY:
        grouping_print_n_samples(df, ["shape_pscho", "shape_hash", var_psycho_idx, "gridloc"])
        df.groupby(["shape_pscho", "shape_hash", "shape_orig", "shape", var_psycho_str]).size().reset_index()
    
    if False: # Not needed
        D.tokens_assign_dataframe_back_to_self_mult(df, tk_ver=token_ver)

    ############# EXTRACT DS
    DSlenient = preprocess_dataset_to_datstrokes(D, "singleprim_psycho")
    DS = preprocess_dataset_to_datstrokes(D, "singleprim")

    # Assign columns to DS and D
    names = ["shapeorig_psycho", "shapeorig", var_psycho_unique, var_psycho_str, var_psycho_idx]
    for i, na in enumerate(names):
        D.Dat[f"seqc_0_{na}"] = [map_shape_to_psychoparams[sh][i] for sh in D.Dat["seqc_0_shape"]]
        # DS.Dat[na] = [map_shape_to_psychoparams[sh][i] for sh in DS.Dat["shape"]]
        # DSlenient.Dat[na] = [map_shape_to_psychoparams[sh][i] for sh in DSlenient.Dat["shape"]]
    _apply_psychoparams_to_ds(DS.Dat, map_shape_to_psychoparams, var_psycho)
    _apply_psychoparams_to_ds(DSlenient.Dat, map_shape_to_psychoparams, var_psycho)

    return DS, DSlenient, map_shape_to_psychoparams


def plot_overview(DS, D, SAVEDIR, var_psycho="angle"):
    """
    """
    from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
    from pythonlib.tools.pandastools import grouping_print_n_samples

    assert var_psycho == "angle", "code it for others, need to change variable strings below"

    # Only include first stroke in these plots
    DS = DS.copy()
    DS.Dat = DS.Dat[DS.Dat["stroke_index"] == 0].reset_index(drop=True)
    DS.distgood_compute_beh_task_strok_distances()

    ### PLOT DRAWINGS
    savedir = f"{SAVEDIR}/drawings"
    os.makedirs(savedir, exist_ok=True)

    niter = 2
    for _iter in range(niter):

        print("Plotting drawings...")
        figholder = DS.plotshape_multshapes_egstrokes("shapeorig_psycho", 6, ver_behtask="beh");
        for i, (fig, axes) in enumerate(figholder):
            savefig(fig, f"{savedir}/egstrokes-{i}-iter{_iter}.pdf")

        figholder = DS.plotshape_multshapes_egstrokes("shapeorig_psycho", 6, ver_behtask="task_aligned_single_strok");
        for i, (fig, axes) in enumerate(figholder):
            savefig(fig, f"{savedir}/egstrokes-{i}-task-iter{_iter}.pdf")
        plt.close("all")

        figbeh, figtask = DS.plotshape_row_col_vs_othervar("angle_idx_within_shapeorig", "shapeorig", n_examples_per_sublot=8, plot_task=True);
        savefig(figbeh, f"{savedir}/shapeorig-vs-idx-{i}-beh-iter{_iter}.pdf")
        savefig(figtask, f"{savedir}/shapeorig-vs-idx-{i}-task-iter{_iter}.pdf")
        plt.close("all")

    ### PLOT COUNTS
    print("Plotting counts...")
    savedir = f"{SAVEDIR}/counts"
    os.makedirs(savedir, exist_ok=True)

    fig = grouping_plot_n_samples_conjunction_heatmap(DS.Dat, "angle_idx_within_shapeorig", "shapeorig", ["gridloc"])
    savefig(fig, f"{savedir}/counts-idx-vs-shapeorig.pdf")

    fig = grouping_plot_n_samples_conjunction_heatmap(DS.Dat, "angle_str", "shapeorig", ["gridloc"])
    savefig(fig, f"{savedir}/counts-str-vs-shapeorig.pdf")
    
    fig = grouping_plot_n_samples_conjunction_heatmap(DS.Dat, "angle_binned", "angle_idx_within_shapeorig", ["shapeorig", "gridloc"])
    savefig(fig, f"{savedir}/counts-angle_binned-vs-idx.pdf")

    fig = grouping_plot_n_samples_conjunction_heatmap(DS.Dat, "loc_on_clust", "angle_idx_within_shapeorig", ["shapeorig", "gridloc"])
    savefig(fig, f"{savedir}/counts-loc_on_clust-vs-idx.pdf")

    fig = grouping_plot_n_samples_conjunction_heatmap(DS.Dat, "angle_idx_within_shapeorig", "shapeorig_psycho", ["shapeorig", "gridloc"])
    savefig(fig, f"{savedir}/counts-idx-vs-shapeorig_psycho.pdf")

    fig = grouping_plot_n_samples_conjunction_heatmap(DS.Dat, "shape", "shapeorig_psycho", ["gridloc"])
    savefig(fig, f"{savedir}/counts-shape-vs-shapeorig_psycho.pdf")

    savepath = f"{savedir}/groupings.txt"
    grouping_print_n_samples(DS.Dat, 
        ["shapeorig_psycho", "shapeorig", "angle_idx_within_shapeorig", "angle_str", "shape", "character"], 
        savepath=savepath)
    plt.close("all")

    #### ANALYSES (e..,g, timing and velocity)
    savedir = f"{SAVEDIR}/analyses"
    os.makedirs(savedir, exist_ok=True)
    print("Plotting motor analyses...")

    ## First stroke
    dfthis = DS.Dat
    dfthis = dfthis[dfthis["gap_from_prev_dur"]<5].reset_index(drop=True)

    fig = sns.catplot(data=dfthis, x="angle_idx_within_shapeorig", y="angle", hue="gridloc", col="shapeorig", alpha=0.5)
    savefig(fig, f"{savedir}/angle-vs-idx-1.pdf")
    fig = sns.catplot(data=dfthis, x="angle_idx_within_shapeorig", y="angle", hue="gridloc", col="shapeorig", kind="point")
    savefig(fig, f"{savedir}/angle-vs-idx-2.pdf")

    fig = sns.catplot(data=dfthis, x="angle_idx_within_shapeorig", y="loc_on_clust", hue="gridloc", col="shapeorig", jitter=True, alpha=0.5)
    savefig(fig, f"{savedir}/loc_on_clust-vs-idx-1.pdf")
    fig = sns.catplot(data=dfthis, x="angle_idx_within_shapeorig", y="loc_on_clust", hue="gridloc", col="shapeorig", jitter=True, alpha=0.5)
    savefig(fig, f"{savedir}/loc_on_clust-vs-idx-1.pdf")
    plt.close("all")

    # Slower for psycho?
    for sharey in [True, False]:
        fig = sns.catplot(data=dfthis, x="angle_idx_within_shapeorig", y="velocity", hue="gridloc", col="shapeorig", 
            alpha=0.5, sharey=sharey)
        savefig(fig, f"{savedir}/velocity-vs-idx-sharey={sharey}-1.pdf")

        fig = sns.catplot(data=dfthis, x="angle_idx_within_shapeorig", y="velocity", hue="gridloc", col="shapeorig", 
            kind="point", sharey=sharey)
        savefig(fig, f"{savedir}/velocity-vs-idx-sharey={sharey}-2.pdf")

        fig = sns.catplot(data=dfthis, x="angle_idx_within_shapeorig", y="gap_from_prev_dur", hue="gridloc", col="shapeorig", 
            alpha=0.5, sharey=sharey)
        savefig(fig, f"{savedir}/gap_from_prev_dur-vs-idx-sharey={sharey}-1.pdf")

        fig = sns.catplot(data=dfthis, x="angle_idx_within_shapeorig", y="gap_from_prev_dur", hue="gridloc", col="shapeorig", 
            kind="point", sharey=sharey)
        savefig(fig, f"{savedir}/gap_from_prev_dur-vs-idx-sharey={sharey}-2.pdf")
        plt.close("all")

        fig = sns.catplot(data=dfthis, x="angle_idx_within_shapeorig", y="dist_beh_task_strok", hue="gridloc", col="shapeorig", 
            alpha=0.5, sharey=sharey)
        savefig(fig, f"{savedir}/dist_beh_task_strok-vs-idx-sharey={sharey}-1.pdf")

        fig = sns.catplot(data=dfthis, x="angle_idx_within_shapeorig", y="dist_beh_task_strok", hue="gridloc", col="shapeorig", 
            kind="point", sharey=sharey)
        savefig(fig, f"{savedir}/dist_beh_task_strok-vs-idx-sharey={sharey}-2.pdf")
        plt.close("all")

    ## Dataset
    savedir = f"{SAVEDIR}/analyses_dataset_lenient_strokes"
    os.makedirs(savedir, exist_ok=True)
    print("Plotting motor analyses...")

    D = D.copy()
    # D.extract_beh_features(["num_strokes_beh"])
    # frac_touched_min = 0.6
    # ft_decim_min = 0.3
    # shortness_min = 0.2
    # D.preprocessGood(params=["beh_strokes_at_least_one",
    #                             "no_supervision",
    #                             "remove_online_abort"],
    #                     frac_touched_min=frac_touched_min,
    #                     ft_decim_min=ft_decim_min,
    #                     shortness_min = shortness_min)
    D.extract_beh_features(["num_strokes_beh"])
    D.preprocessGood(params=["beh_strokes_at_least_one",
                                "no_supervision"])

    # Number of strokes used
    dfthis = D.Dat

    fig = sns.catplot(data=dfthis, x="seqc_0_angle_idx_within_shapeorig", y="FEAT_num_strokes_beh", col="seqc_0_shapeorig", 
        alpha=0.4, jitter=True)
    savefig(fig, f"{savedir}/FEAT_num_strokes_beh-vs-idx-1.pdf")

    fig = sns.catplot(data=dfthis, x="seqc_0_angle_idx_within_shapeorig", y="FEAT_num_strokes_beh", col="seqc_0_shapeorig", kind="box")
    savefig(fig, f"{savedir}/FEAT_num_strokes_beh-vs-idx-2.pdf")
    plt.close("all")

def preprocess_cont_morph(D):
    """
    ENtire preprocess pipeline for continuous morph (i.e., 2 base shapes, and a single, or multiple, shapes that interpolate bewteen them)

    """
    from pythonlib.tools.pandastools import find_unique_values_with_indices
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from pythonlib.tools.pandastools import append_col_with_index_of_level, append_col_with_index_of_level_after_grouping
    from pythonlib.tools.pandastools import grouping_print_n_samples
    from pythonlib.tools.checktools import check_objects_identical
    from pythonlib.dataset.dataset_strokes import preprocess_dataset_to_datstrokes
    from math import pi

    # Check that it is one stroke.
    assert all(D.Dat["task_kind"] == "prims_single"), "assuming this to make things eaiser..."
    
    # # First, for each token, extract variables that reflect the psycho variable/param.
    # token_ver = "task"
    # for i, row in D.Dat.iterrows():
    #     # for each shape get its concrete params
    #     Tk = D.taskclass_tokens_extract_wrapper(i, token_ver, return_as_tokensclass=True)
    #     # Tk.features_extract_wrapper(["loc_on", "angle"], angle_twind=[0, 2])
    # df = D.tokens_extract_variables_as_dataframe(["shape", "Prim", "gridloc"], token_ver)
    
    # # Extract the original shape, which should have been overwritten in rprepovessing, but is useufl as a category
    # # to anchor the variations.
    # list_shape_orig = []
    # for P in df["Prim"]:
    #     list_shape_orig.append(P.shape_oriented())
    # df["shape_orig"] = list_shape_orig

    _, map_shape_orig_to_shape = D.shapesemantic_taskclass_map_between_shape_and_shape_orig()

    ############################################
    # Map from shape to its base shapes
    map_shape_to_base_prims = {}
    # map_shape_to_base_LOSs = {}
    for ind in range(len(D.Dat)):

        # get the shape name 
        shape = D.taskclass_shapes_extract(ind)[0]
        morph_info = D.shapesemantic_taskclass_cont_morph_extract_params(ind)

        if morph_info is None:
            base_shapes = None
            # base_LOSs = None
        else:
            base_shapes = [mi["shape"] for mi in morph_info] # list of two 3-tuples
            # base_LOSs = [mi["los"] for mi in morph_info] 

        # Base shapes -- convert these to whatever they are called now,w hich might be diff, beaause of redefine
        # name by stroke hash
        if base_shapes is not None:
            base_shapes = ["-".join([str(x) for x in shtuple]) for shtuple in base_shapes]
            print(map_shape_orig_to_shape.keys())
            
            base_shapes = [map_shape_orig_to_shape[sh] for sh in base_shapes]

        if shape in map_shape_to_base_prims.keys():
            # Shape already added. check that previous item matches current.
            assert map_shape_to_base_prims[shape] == base_shapes
            # print(map_shape_to_base_LOSs[shape])
            # print(base_LOSs)
            # assert map_shape_to_base_LOSs[shape] == base_LOSs
        else:
            # Newly encountered shape. Add it
            map_shape_to_base_prims[shape] = base_shapes
            # map_shape_to_base_LOSs[shape] = base_LOSs

    ###################### sanity check, mapping other way around is also unique
    allow_incorrect_shape_names = True

    # Base shapes
    map_base_prims_to_morphed_shape = {}
    for shape, base_prims in map_shape_to_base_prims.items():
        if base_prims is not None:
            base_prims = tuple(base_prims)
            if base_prims in map_base_prims_to_morphed_shape.keys():
                if not allow_incorrect_shape_names:
                    assert map_base_prims_to_morphed_shape[base_prims] == shape
                else:
                    map_base_prims_to_morphed_shape[base_prims].append(shape)
            else:
                map_base_prims_to_morphed_shape[base_prims] = [shape]
    
    # Base LOSs
    if False:
        map_base_LOSs_to_morphed_shape = {}
        for shape, LOSs in map_shape_to_base_LOSs.items():
            if LOSs is not None:
                LOSs = tuple(LOSs)
                if LOSs in map_base_LOSs_to_morphed_shape.keys():
                    if not allow_incorrect_shape_names:
                        assert map_base_LOSs_to_morphed_shape[LOSs] == shape
                    else:
                        map_base_LOSs_to_morphed_shape[LOSs].append(shape)
                else:
                    map_base_LOSs_to_morphed_shape[LOSs] = [shape]


    ############# EXTRACT DS
    # DSlenient = preprocess_dataset_to_datstrokes(D, "singleprim_psycho")
    # DS = preprocess_dataset_to_datstrokes(D, "singleprim")
    # - Decided to just get this, since it would be too much to run through both versions below.
    DS = preprocess_dataset_to_datstrokes(D, "singleprim_psycho")

    # Append column with base prims.
    D.Dat["seqc_0_base_shapes"] = [map_shape_to_base_prims[sh] for sh in D.Dat["seqc_0_shape"]]
    # DSlenient.Dat["base_shapes"] = [map_shape_to_base_prims[sh] for sh in DSlenient.Dat["shape"]]
    DS.Dat["base_shapes"] = [map_shape_to_base_prims[sh] for sh in DS.Dat["shape"]]

    # Minor dumb things
    DS.Dat["loc_on_x"] = [loc[0] for loc in DS.Dat["loc_on"]]
    DS.Dat["loc_on_y"] = [loc[1] for loc in DS.Dat["loc_on"]]

    if False:
        D.Dat["seqc_0_base_LOSs"] = [map_base_LOSs_to_morphed_shape[sh] for sh in D.Dat["seqc_0_shape"]]
        # DSlenient.Dat["base_LOSs"] = [map_base_LOSs_to_morphed_shape[sh] for sh in DSlenient.Dat["shape"]]
        DS.Dat["base_LOSs"] = [map_base_LOSs_to_morphed_shape[sh] for sh in DS.Dat["shape"]]

    # Also save other things related to morphs and shape sets.
    # for ds in [DS, DSlenient]:
    for ds in [DS]:
        ds.Dat["morph_is_morphed"] = [bs is not None for bs in ds.Dat["base_shapes"]] # bool, whether is base (False) or morphed (True)
    D.Dat["seqc_0_morphed"] = [bs is not None for bs in D.Dat["seqc_0_base_shapes"]]

    # Also extract motor params for DS (i.e,, primitivenessv2). 
    # SHould do this here, so that the next step has all the relevant motor params already
    from pythonlib.dataset.dataset_analy.primitivenessv2 import preprocess_directly_from_DS
    DS, _ = preprocess_directly_from_DS(DS, prune_strokes=False, microstim_version=False)

    ### And then get version of DF that groups into morph sets
    # Get groups of shape including the base shapes, and all morphed shapes
    # Here defines all variables that depends on cosnidering entire set of shapes (base and morphs).
    list_df = []
    map_morph_set_idx_to_shapes = {}

    for i, (base_prims, morphed_shapes) in enumerate(map_base_prims_to_morphed_shape.items()):

        # Pull out rows 
        dfbase = DS.Dat[DS.Dat["shape"].isin(base_prims)].copy()
        dfmorph = DS.Dat[DS.Dat["shape"].isin(morphed_shapes)].copy()

        dfbase["morph_set_idx"] = i
        assert all(dfbase["morph_is_morphed"] == False)

        # For each base row, determine if it is the first or second base
        dfbase["morph_idxcode_within_set"] = [base_prims.index(sh) for sh in dfbase["shape"]]
        assert all(dfbase["morph_idxcode_within_set"].isin([0,1]))

        dfmorph["morph_set_idx"] = i
        assert all(dfmorph["morph_is_morphed"] == True)
        dfmorph["morph_idxcode_within_set"] = -1 # morphed
        
        list_df.append(dfbase)
        list_df.append(dfmorph)

        # Save mapping from morph index to prims
        # idx:[base_prim_1, [morph prims], base_prim_2]
        base_prim_1 = dfbase[dfbase["morph_idxcode_within_set"]==0]["shape"].values[0]
        base_prim_2 = dfbase[dfbase["morph_idxcode_within_set"]==1]["shape"].values[0]
        morph_prims = sorted(dfmorph["shape"].unique().tolist())
        map_morph_set_idx_to_shapes[i] = [
            base_prim_1, 
            morph_prims,
            base_prim_2
        ]

    DF = pd.concat(list_df).reset_index(drop=True)    
    DSmorphsets = DS.copy()
    DSmorphsets.Dat = DF

    # Create save dir
    SAVEDIR = D.make_savedir_for_analysis_figures_BETTER("psycho_singleprims")
    SAVEDIR = f"{SAVEDIR}/cont_morph"
    
    return DS, DSmorphsets, map_morph_set_idx_to_shapes, map_shape_to_base_prims, map_base_prims_to_morphed_shape, SAVEDIR

# def plot_motor_stats_primitivenessv2():

#     from pythonlib.dataset.dataset_analy.primitivenessv2 import preprocess_directly_from_DS, extract_grouplevel_motor_stats, plot_grouplevel_results

#     ########## First, use shape as datapt, therefore not trying to group based on a shape set (which has both novel and base)
#     grouping = ["shape", "epoch", "block", "morph_is_morphed"]
#     microstim_version = False
#     DS, SAVEDIR = preprocess_directly_from_DS(DS, prune_strokes=False, microstim_version=microstim_version)


#     dfres, grouping = extract_grouplevel_motor_stats(DS, grouping, 
#         microstim_version=microstim_version)



#     contrast = "morph_is_morphed"
#     context = "shape"

#     # Plot, comparing mean across levels of contrast variable.
#     # Each datapt is a single level of grouping.
#     savedir = f"{SAVEDIR}/grouplevel"
#     os.makedirs(savedir, exist_ok=True)
#     print(savedir)
#     plot_grouplevel_results(dfres, DS, D, grouping, contrast, savedir, context=context)

#     # Hacky, using grouped data.


#     grouping = ["shape", "epoch", "block", "morph_is_morphed", "morph_set_idx"]
#     contrast = "morph_is_morphed"
#     context = ["morph_set_idx", "shape"]

#     ds = DS.copy()
#     ds.Dat = DF
#     dfres, grouping = extract_grouplevel_motor_stats(ds, grouping, 
#         microstim_version=microstim_version)

#     from pythonlib.tools.pandastools import append_col_with_grp_index
#     dfres = append_col_with_grp_index(dfres, context, "context")
#     tmp = {
#         False:0,
#         True:1
#     }
#     dfres["morph_is_morphed_int"] = [tmp[x] for x in dfres["morph_is_morphed"]]

#     # Plot, comparing mean across levels of contrast variable.
#     # Each datapt is a single level of grouping.

#     contrast = "morph_is_morphed_int"
#     savedir = f"{SAVEDIR}/grouplevel_split_into_shape_sets"
#     os.makedirs(savedir, exist_ok=True)
#     print(savedir)
#     plot_grouplevel_results(dfres, ds, D, grouping, contrast, savedir, context="morph_set_idx")

# def _plot_structuredmorph_drawings_each_morph_set(DS, map_base_prims_to_morphed_shape, savedir):
#     """
#     For each morph set, plot the base prims and the morphed
#     """
#     from pythonlib.tools.plottools import savefig

#     for i, (base_prims, morphed_shapes) in enumerate(map_base_prims_to_morphed_shape.items()):
#         assert len(base_prims)==2
#         list_shape = [base_prims[0]] + list(morphed_shapes) + [base_prims[1]]

#         figholder = DS.plotshape_multshapes_egstrokes_grouped_in_subplots(levels_subplots=list_shape, n_examples=3)
#         for j, (fig, axes) in enumerate(figholder):
#             path = f"{savedir}/{base_prims[0]}|{base_prims[1]}--idx_{i}--sub_{j}--BEH.pdf"
#             savefig(fig, path)
            
#         figholder = DS.plotshape_multshapes_egstrokes_grouped_in_subplots(levels_subplots=list_shape, n_examples=1, ver_behtask="task")
#         for j, (fig, axes) in enumerate(figholder):
#             path = f"{savedir}/{base_prims[0]}|{base_prims[1]}--idx_{i}--sub_{j}--TASK.pdf"
#             savefig(fig, path)
        
#         plt.close("all")

def _plot_contmorph_drawings_each_morph_set(DS, map_morph_set_idx_to_shapes, savedir):
    """
    For each morph set, plot the base prims and the morphed
    """
    from pythonlib.tools.plottools import savefig

    for morph_set_idx, (base_prim_1, morphed_prims, base_prim_2) in map_morph_set_idx_to_shapes.items():

        list_shape = [base_prim_1] + morphed_prims + [base_prim_2]

        figholder = DS.plotshape_multshapes_egstrokes_grouped_in_subplots(levels_subplots=list_shape, n_examples=3)
        for j, (fig, _) in enumerate(figholder):
            path = f"{savedir}/morph_set_idx={morph_set_idx}-base_prims={base_prim_1}|{base_prim_2}-sub={j}-BEH.pdf"
            savefig(fig, path)
            
        figholder = DS.plotshape_multshapes_egstrokes_grouped_in_subplots(levels_subplots=list_shape, n_examples=1, ver_behtask="task")
        for j, (fig, _) in enumerate(figholder):
            path = f"{savedir}/morph_set_idx={morph_set_idx}-base_prims={base_prim_1}|{base_prim_2}-sub={j}-TASK.pdf"
            savefig(fig, path)
        
        plt.close("all")


def plot_overview_cont_morph(D, DS, DSmorphsets, map_morph_set_idx_to_shapes, SAVEDIR,
                             use_task_stroke_or_los="stroke"):
    """
    [GOOD] All plots for continuous morph (betwene two shapes)
    This should be folded into psychogood code. and here should subsuming the angle stuff. 
    Here is based on that but shoudl be more general.
    """

    # For every pair of base prims, plot them along with their interpolated across all locations.
    from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
    from pythonlib.tools.pandastools import grouping_print_n_samples

    # Only include first stroke in these plots
    DS = DS.copy()
    DS.Dat = DS.Dat[DS.Dat["stroke_index"] == 0].reset_index(drop=True)

    ### PLOT DRAWINGS
    savedir = f"{SAVEDIR}/drawings"
    os.makedirs(savedir, exist_ok=True)
    _plot_contmorph_drawings_each_morph_set(DS, map_morph_set_idx_to_shapes, savedir)

    ### PLOT COUNTS
    print("Plotting counts...")
    savedir = f"{SAVEDIR}/counts"
    os.makedirs(savedir, exist_ok=True)

    from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
    fig = grouping_plot_n_samples_conjunction_heatmap(DS.Dat, "shape", "morph_is_morphed", ["gridloc"])
    savefig(fig, f"{savedir}/counts-shape-vs-ismorph.pdf")

    fig = grouping_plot_n_samples_conjunction_heatmap(DSmorphsets.Dat, "shape", "morph_set_idx", None, FIGSIZE=12)
    savefig(fig, f"{savedir}/MORPHSETS-counts-shape-vs-morph_set_idx.pdf")

    fig = grouping_plot_n_samples_conjunction_heatmap(DSmorphsets.Dat, "shape", "morph_set_idx", ["morph_is_morphed"], FIGSIZE=12)
    savefig(fig, f"{savedir}/MORPHSETS-counts-shape-vs-morph_set_idx-2.pdf")

    fig = grouping_plot_n_samples_conjunction_heatmap(DSmorphsets.Dat, "morph_is_morphed", "morph_set_idx", ["gridloc"])
    savefig(fig, f"{savedir}/MORPHSETS-counts-morph_is_morphed-vs-morph_set_idx_gridloc.pdf")

    fig = grouping_plot_n_samples_conjunction_heatmap(DSmorphsets.Dat, "morph_idxcode_within_set", "morph_set_idx", ["gridloc"])
    savefig(fig, f"{savedir}/MORPHSETS-counts-morph_idxcode_within_set-vs-morph_set_idx_gridloc.pdf")

    fig = grouping_plot_n_samples_conjunction_heatmap(DSmorphsets.Dat, "morph_idxcode_within_set", "morph_set_idx", None)
    savefig(fig, f"{savedir}/MORPHSETS-counts-morph_idxcode_within_set-vs-morph_set_idx.pdf")


    from pythonlib.tools.pandastools import grouping_print_n_samples
    savepath = f"{savedir}/groupings_gridloc.txt"
    grouping_print_n_samples(DS.Dat, ["morph_is_morphed", "shape", "gridloc", "character"], savepath=savepath)

    savepath = f"{savedir}/groupings_shape_semantic.txt"
    grouping_print_n_samples(DS.Dat, ["morph_is_morphed", "shape", "shape_semantic"], savepath=savepath)

    savepath = f"{savedir}/MORPHSETS-groupings_morphsets.txt"
    grouping_print_n_samples(DSmorphsets.Dat, 
                             ["morph_set_idx", "morph_is_morphed", "morph_idxcode_within_set", "shape", "gridloc", "character"], 
                             savepath=savepath)
    plt.close("all")

    ### PLOT SCORES
    _plot_overview_scores(D, DSmorphsets, SAVEDIR, use_task_stroke_or_los, DS)



def _plot_overview_scores(D, DSmorphsets, SAVEDIR, use_task_stroke_or_los, DS=None):
    """
    """
    from pythonlib.dataset.dataset_analy.primitivenessv2 import preprocess_plot_pipeline_directly_from_DS
   

    # #### ANALYSES (e..,g, timing and velocity)
    savedir = f"{SAVEDIR}/analyses"
    os.makedirs(savedir, exist_ok=True)
    print("Plotting motor analyses...")


    if True:
        ## First stroke
        dfthis = DSmorphsets.Dat
        dfthis = dfthis[dfthis["gap_from_prev_dur"]<5].reset_index(drop=True)
        for y in ["angle", "loc_on_x", "loc_on_y", "velocity", "gap_from_prev_dur", "dist_beh_task_strok"]:
            for sharey in [True, False]:
                fig = sns.catplot(data=dfthis, x="morph_idxcode_within_set", y=y, hue="gridloc", col="morph_set_idx", col_wrap=8,
                    alpha=0.5, sharey=sharey)
                savefig(fig, f"{savedir}/{y}-vs-idx-sharey={sharey}-1.pdf")

                fig = sns.catplot(data=dfthis, x="morph_idxcode_within_set", y=y, hue="gridloc", col="morph_set_idx", col_wrap=8,
                    kind="point", sharey=sharey)
                savefig(fig, f"{savedir}/{y}-vs-idx-sharey={sharey}-2.pdf")

                plt.close("all")

        ## Dataset
        savedir = f"{SAVEDIR}/analyses_dataset_lenient_strokes"
        os.makedirs(savedir, exist_ok=True)
        print("Plotting motor analyses...")


        # Number of strokes used
        if "seqc_0_morphed" in D.Dat.columns:
            D = D.copy()
            D.extract_beh_features(["num_strokes_beh"])
            D.preprocessGood(params=["beh_strokes_at_least_one",
                                        "no_supervision"])
            
            dfthis = D.Dat
            fig = sns.catplot(data=dfthis, x="seqc_0_morphed", y="FEAT_num_strokes_beh", 
                alpha=0.4, jitter=True)
            savefig(fig, f"{savedir}/FEAT_num_strokes_beh-1.pdf")

            fig = sns.catplot(data=dfthis, x="seqc_0_morphed", y="FEAT_num_strokes_beh", 
                kind="boxen")
            savefig(fig, f"{savedir}/FEAT_num_strokes_beh-2.pdf")


        if DS is not None:
            grouping = [var, "epoch", "block", "morph_is_morphed"]
            context = var
            contrast = "morph_is_morphed"
            plot_methods = ("grp", "tls", "tl", "tc", "drw")
            preprocess_plot_pipeline_directly_from_DS(DS, grouping=grouping, 
                                                    contrast=contrast, context=context,
                                                    prune_strokes=prune_strokes, 
                                                    plot_methods=plot_methods)
    plt.close("all")
    
    #     # Same thing, but grouping trials (which might lead to repeated cases of base prims)

    #     grouping = [var, "epoch", "block", "morph_is_morphed", "morph_set_idx"]
    #     contrast = "morph_is_morphed"
    #     context = "morph_set_idx"

    #     # savedir = f"{SAVEDIR}/grouplevel_split_into_shape_sets"
    #     # os.makedirs(savedir, exist_ok=True)
    #     # print(savedir)
    #     # plot_grouplevel_results(dfres, ds, D, grouping, contrast, savedir, context="morph_set_idx")
    #     plot_methods = ("grp", "tls", "tl", "tc")
    #     preprocess_plot_pipeline_directly_from_DS(DSmorphsets, grouping=grouping, 
    #                                             contrast=contrast, context=context,
    #                                             prune_strokes=prune_strokes,
    #                                             savedir_suffix="MORPH_SETS")
    

    # # from pythonlib.tools.pandastools import append_col_with_grp_index
    # # DSmorphsets.Dat = append_col_with_grp_index(DSmorphsets.Dat, ["morph_set_idx", "morph_idxcode_within_set"], "morph_idx_idx")
    # grouping = [var, "epoch", "block", "morph_is_morphed", "morph_set_idx_global"]
    # contrast = "morph_is_morphed"
    # context = "morph_set_idx_global"
    # plot_methods = ("grp", "tls", "tl", "tc")
    # preprocess_plot_pipeline_directly_from_DS(DSmorphsets, grouping=grouping, 
    #                                         contrast=contrast, context=context,
    #                                         prune_strokes=prune_strokes,
    #                                         savedir_suffix="MORPH_SETS_EACH_IDX_GLOBAL")
    

def psychogood_preprocess_wrapper(D, PLOT_DRAWINGS = True, PLOT_EACH_TRIAL = True):
    """
    GOOD - For semi-auto detecting psycho groups, and making all plots.
    [Written for structured morph, shold subsume all, esp cont morph]

    Does entire preprocess and plot pipeline.

    """
    from pythonlib.tools.listtools import stringify_list
    from pythonlib.dataset.dataset_analy.psychometric_singleprims import psychogood_find_tasks_in_this_psycho_group_wrapper, params_extract_psycho_groupings_manual
    import pandas as pd
    from pythonlib.tools.pandastools import append_col_with_index_of_level_after_grouping, append_col_with_index_of_level
    from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
    from pythonlib.tools.plottools import savefig

    SAVEDIR = D.make_savedir_for_analysis_figures_BETTER("psycho_singleprims")
    SAVEDIR = f"{SAVEDIR}/general"

    animal = D.animals(True)[0]
    date = D.dates(True)[0]

    # Extract pre-saved params
    PARAMS, los_allowed_to_miss = params_extract_psycho_groupings_manual(animal, date)

    # Generate data for eahc psycho group
    list_dfres = []
    for psycho_group, _params in enumerate(PARAMS):
        los_base_1 = _params["los_base_1"]
        los_within = _params["los_within"]
        los_base_2 = _params["los_base_2"]
        psycho_params = _params["psycho_params"]

        # Extract trials that are in this psycho group
        dfres_within = psychogood_find_tasks_in_this_psycho_group_wrapper(D, los_within, psycho_params)
        dfres_base_1 = psychogood_find_tasks_in_this_psycho_group_wrapper(D, los_base_1, None)
        dfres_base_2 = psychogood_find_tasks_in_this_psycho_group_wrapper(D, los_base_2, None)

        # If there are any base prims that were detected by dfres_within, remove them from dfres_within
        # dfres_within = dfres_within[~dfres_within["los"].isin([los_base_1, los_base_2])].reset_index(drop=True)
        dfres_within = dfres_within[~dfres_within["los"].isin(dfres_base_1["los"].unique().tolist())].reset_index(drop=True)
        dfres_within = dfres_within[~dfres_within["los"].isin(dfres_base_2["los"].unique().tolist())].reset_index(drop=True)

        # Add features
        dfres_within = append_col_with_index_of_level_after_grouping(dfres_within, None, "psycho_value", "morph_idxcode_within_set")
        dfres_within["morph_is_morphed"] = True
        dfres_within["morph_idxcode_within_set"] += 1

        dfres_base_1["morph_is_morphed"] = False
        dfres_base_1["morph_idxcode_within_set"] = 0

        dfres_base_2["morph_is_morphed"] = False
        # dfres_base_2["morph_idxcode_within_set"] = max(dfres_within["morph_idxcode_within_set"]) + 1
        dfres_base_2["morph_idxcode_within_set"] = 99

        dfres_all = pd.concat([dfres_base_1, dfres_base_2, dfres_within]).reset_index(drop=True)
        dfres_all["morph_set_idx"] = psycho_group

        list_dfres.append(dfres_all)

        ############ MAKE PLOTS (drawings)
        if PLOT_DRAWINGS:
            savedir = f"{SAVEDIR}/psycho_group_{psycho_group}"
            import os
            os.makedirs(savedir, exist_ok=True)

            # Print results 
            fig = grouping_plot_n_samples_conjunction_heatmap(dfres_all, "los", "morph_idxcode_within_set", ["morph_is_morphed"])
            savefig(fig, f"{savedir}/counts-los-vs-morph_idxcode_within_set.pdf")

            fig = grouping_plot_n_samples_conjunction_heatmap(dfres_all, "psycho_value", "morph_idxcode_within_set", ["morph_is_morphed"])
            savefig(fig, f"{savedir}/counts-psycho_value-vs-morph_idxcode_within_set.pdf")
        
            for ver in ["beh", "task"]:

                if ver=="beh":
                    nrand = 3
                elif ver =="task":
                    # then you want to plot all, to sanity check that correctly detected
                    nrand = None
                else:
                    assert False

                SIZE = 4
                list_morph_idx = sorted(dfres_all["morph_idxcode_within_set"].unique())

                n = len(list_morph_idx) + 2
                ncols = 6
                nrows = int(np.ceil(n/ncols))

                fig_combined, axes = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE), sharex=True, sharey=True)

                # base 1
                ax = axes.flatten()[0]
                inds = dfres_base_1["idx_dat"].tolist()
                D.plot_mult_trials_overlaid_on_axis(inds, ax, ver=ver, single_color="b", nrand=nrand)
                ax.set_title(f"base_prim 1")

                # - also plot individual trials
                if False: # is included with morph index 0 and end
                    fig, _, _ = D.plotMultTrials2(inds, "strokes_beh")
                    savefig(fig, f"{savedir}/indivtrials-base_1-beh.pdf")
                    fig, _, _  = D.plotMultTrials2(inds, "strokes_task")
                    savefig(fig, f"{savedir}/indivtrials-base_1-task.pdf")

                # Within, in order
                inds_base1 = dfres_base_1["idx_dat"].tolist()
                inds_base2 = dfres_base_2["idx_dat"].tolist()

                ct = 1
                for morph_idx in list_morph_idx:
                    ax = axes.flatten()[ct]

                    if ver == "task":
                        # Ovelray base inds
                        D.plot_mult_trials_overlaid_on_axis(inds_base1, ax, ver=ver, single_color="b", alpha=0.25, nrand=10)
                        D.plot_mult_trials_overlaid_on_axis(inds_base2, ax, ver=ver, single_color="r", alpha=0.25, nrand=10)

                    inds = dfres_all[dfres_all["morph_idxcode_within_set"] == morph_idx]["idx_dat"].tolist()
                    D.plot_mult_trials_overlaid_on_axis(inds, ax, ver=ver, single_color="k")


                    ax.set_title(f"morph_idx: {morph_idx}")
                    ct += 1

                    if PLOT_EACH_TRIAL:
                        # - also plot individual trials
                        fig, _, _  = D.plotMultTrials2(inds, "strokes_beh")
                        savefig(fig, f"{savedir}/indivtrials-morph_idx_{morph_idx}-beh.pdf")
                        if False: # not needed, since tasks are already plot in the overlaid plot
                            fig, _, _  = D.plotMultTrials2(inds, "strokes_task")
                            savefig(fig, f"{savedir}/indivtrials-morph_idx_{morph_idx}-task.pdf")
            
                    plt.close("all")

                # base 2
                ax = axes.flatten()[ct]
                inds = dfres_base_2["idx_dat"].tolist()
                D.plot_mult_trials_overlaid_on_axis(inds, ax, ver=ver, single_color="r", nrand=nrand)
                ax.set_title(f"base_prim 2")

                # - also plot individual trials
                if False: # is included with morph index 0 and end
                    fig, _, _  = D.plotMultTrials2(inds, "strokes_beh")
                    savefig(fig, f"{savedir}/indivtrials-base_2-beh.pdf")
                    fig, _, _  = D.plotMultTrials2(inds, "strokes_task")
                    savefig(fig, f"{savedir}/indivtrials-base_2-task.pdf")

                savefig(fig_combined, f"{savedir}/all_overlaid-{ver}.pdf")

                plt.close("all")
    DFRES = pd.concat(list_dfres).reset_index(drop=True)

    # CHECK that all los have been assigned to something
    los_all = D.Dat["los_info"]
    tmp = DFRES["los"].unique().tolist()
    los_missing = []
    for los in los_all:
        if los not in tmp:
            los_missing.append(los)

    los_missing = sorted(set(los_missing))
    los_missing_actual = [los for los in los_missing if los not in los_allowed_to_miss]
    for los in los_missing_actual:
        print(f"{los},")
    assert len(los_missing_actual)==0, "missing these los... is that ok? Copy and paste the los that you are fine missing, and place above into los_allowed_to_miss"


    ##################################### PLOTS
    from pythonlib.tools.pandastools import find_unique_values_with_indices
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from pythonlib.tools.pandastools import append_col_with_index_of_level, append_col_with_index_of_level_after_grouping
    from pythonlib.tools.pandastools import grouping_print_n_samples
    from pythonlib.tools.checktools import check_objects_identical
    from pythonlib.dataset.dataset_strokes import preprocess_dataset_to_datstrokes
    from math import pi
    import pandas as pd
    from pythonlib.dataset.dataset_analy.primitivenessv2 import preprocess_directly_from_DS
    from pythonlib.dataset.dataset_analy.primitivenessv2 import preprocess_plot_pipeline_directly_from_DS
    from pythonlib.dataset.dataset_analy.psychometric_singleprims import _plot_overview_scores

    ############# EXTRACT DS
    # DSlenient = preprocess_dataset_to_datstrokes(D, "singleprim_psycho")
    # DS = preprocess_dataset_to_datstrokes(D, "singleprim")
    # - Decided to just get this, since it would be too much to run through both versions below.
    DS = preprocess_dataset_to_datstrokes(D, "singleprim_psycho")
    DS.dataset_append_column("los_info")

    # Also extract motor params for DS (i.e,, primitivenessv2). 
    # SHould do this here, so that the next step has all the relevant motor params already

    DS, _ = preprocess_directly_from_DS(DS, prune_strokes=False, microstim_version=False)
    # Minor dumb things
    if DS is not None:
        DS.Dat["loc_on_x"] = [loc[0] for loc in DS.Dat["loc_on"]]
        DS.Dat["loc_on_y"] = [loc[1] for loc in DS.Dat["loc_on"]]

    # Map from psycho group to trial info
    map_group_to_los = {}
    for morph_group in DFRES["morph_set_idx"].unique().tolist():
        list_idx = DFRES[DFRES["morph_set_idx"]==morph_group]["morph_idxcode_within_set"].unique().tolist()
        map_group_to_los[morph_group] = {idx:None for idx in list_idx}
        for idx_within_group in list_idx:
            los = DFRES[(DFRES["morph_set_idx"]==morph_group) & (DFRES["morph_idxcode_within_set"]==idx_within_group)]["los"].unique().tolist()
            map_group_to_los[morph_group][idx_within_group] = los

    # Params for PRIMITIVENESS plots
    use_task_stroke_or_los = "los"
    if use_task_stroke_or_los == "stroke":
        var = "stroke"
    elif use_task_stroke_or_los == "los":
        var = "los_info"
    else:
        assert False
    prune_strokes = False

    ##### Two variations on DS
    # (1) DS, concatting df where each df is a single morph set, e.g., morphj idnices [0 1 2 3... 99], where 0 and 99 are base prims
    list_df = []
    for group, idx_dict in map_group_to_los.items():
        for idx, list_los in idx_dict.items():
            df = DS.Dat[DS.Dat["los_info"].isin(list_los)].copy()
            df["morph_set_idx"] = group
            df["morph_idxcode_within_set"] = idx
            df["morph_is_morphed"] = idx not in [0, 99]
            list_df.append(df)
    DF = pd.concat(list_df).reset_index(drop=True)
    DSmorphsets = DS.copy()
    DSmorphsets.Dat = DF

    _plot_overview_scores(D, DSmorphsets, SAVEDIR, use_task_stroke_or_los="los")

    grouping = [var, "epoch", "block", "morph_is_morphed", "morph_set_idx"]
    contrast = "morph_is_morphed"
    context = "morph_set_idx"
    plot_methods = ("grp", "tls", "tl", "tc")
    preprocess_plot_pipeline_directly_from_DS(DSmorphsets, grouping=grouping, 
                                            contrast=contrast, context=context,
                                            prune_strokes=prune_strokes, plot_methods=plot_methods,
                                            savedir_suffix="MORPH_SETS")


    # (2) DS, where each df is a single morph set x morph index, e.g.., morph indices [0 1 99], [0 2 99] will be separated dfs that are concatted
    list_df = []
    morph_set_idx_global = 0
    for group, idx_dict in map_group_to_los.items():
        for idx, list_los in idx_dict.items():
            if idx not in [0, 99]:
                # Then this is a morph:

                df_morph = DS.Dat[DS.Dat["los_info"].isin(list_los)].copy()
                df_morph["morph_set_idx"] = group
                df_morph["morph_set_idx_global"] = morph_set_idx_global
                df_morph["morph_idxcode_within_set"] = idx
                df_morph["morph_is_morphed"] = True

                list_los_base_1 = idx_dict[0] # base 1
                df_base1 = DS.Dat[DS.Dat["los_info"].isin(list_los_base_1)].copy()
                df_base1["morph_set_idx"] = group
                df_base1["morph_set_idx_global"] = morph_set_idx_global
                df_base1["morph_idxcode_within_set"] = 0
                df_base1["morph_is_morphed"] = False

                list_los_base_2 = idx_dict[99] # base 2
                df_base2 = DS.Dat[DS.Dat["los_info"].isin(list_los_base_2)].copy()
                df_base2["morph_set_idx_global"] = morph_set_idx_global
                df_base2["morph_set_idx"] = group
                df_base2["morph_idxcode_within_set"] = 99
                df_base2["morph_is_morphed"] = False

                assert len(df_morph)>0
                assert len(df_base1)>0
                assert len(df_base2)>0
                
                list_df.extend([df_morph, df_base1, df_base2])
                morph_set_idx_global += 1

    DF = pd.concat(list_df).reset_index(drop=True)
    DSmorphsets = DS.copy()
    DSmorphsets.Dat = DF

    grouping = [var, "epoch", "block", "morph_is_morphed", "morph_set_idx_global"]
    contrast = "morph_is_morphed"
    context = "morph_set_idx_global"
    plot_methods = ("grp", "tls", "tl", "tc")
    preprocess_plot_pipeline_directly_from_DS(DSmorphsets, grouping=grouping, 
                                            contrast=contrast, context=context,
                                            prune_strokes=prune_strokes, plot_methods=plot_methods,
                                            savedir_suffix="MORPH_SETS_EACH_IDX_GLOBAL")

    return DFRES, PARAMS, los_allowed_to_miss
        

def simplify_morph_info_dict(morph_info):
    """
    Return simplified version of morph info, human-readable.
    RETURNS:
    - list of 2 dicts, one for each of 2 morph endpoints
    [
      {'los': ('singleprims', 114, 9), 'shape': ('arcdeep', 4, 4, 0), 'frac': 0.16666666666666666, 'flip_task1': False, 'flip_task2': False, 'frac_func': array([], shape=(0, 0), dtype=float64)}, 
      {'los': ('singleprims', 114, 30), 'shape': ('circle', 6, 1, 0), 'frac': 0.16666666666666666, 'flip_task1': False, 'flip_task2': False, 'frac_func': array([], shape=(0, 0), dtype=float64)}
    ]
    """

    if morph_info is None or len(morph_info)==0:
        return None
    else:   
        assert isinstance(morph_info, list)
        for x in morph_info:
            assert isinstance(x, dict)
            assert "frac" in x.keys()
        return morph_info

def simplify_extra_tform_dict(extra_tform_params):
    """
    Returns human readable version of tforms
    E.g., if no global tform, but have specifi tform for sencond prim:
    RETURNS:
    - methods_tforms, either None (no tform) or dict (e.g., {sx:1.2, th:0.2})
    - methods_tforms_prim, list of len strokes, where each stroke is represnted liek above, 
    either NOne, or dict

        example:
        (None, [None, {'sx': array(1.)}])
    """
    
    if extra_tform_params is None:
        return None, None
    
    if "tforms" in extra_tform_params and len(extra_tform_params["tforms"])>0:
        # methods_tforms = [_tform[0] for _tform in extra_tform_params["tforms"]]
        assert len(list(extra_tform_params.keys())) == len(set(list(extra_tform_params.keys()))), "you have tforms that repeat. in matlab this would compound. how to deal with here?"
        methods_tforms = {_tform[0]:_tform[1] for _tform in extra_tform_params["tforms"]}
        if len(methods_tforms)==0:
            methods_tforms = None
    else:
        methods_tforms = None

    if "tforms_each_prim_p" in extra_tform_params and len(extra_tform_params["tforms_each_prim_p"])>0:
        methods_tforms_prim = []
        for prim_tform in extra_tform_params["tforms_each_prim_p"]:
            if len(prim_tform)==0:
                methods_tforms_prim.append(None)
            else:
                _keys = [_tform[0] for _tform in prim_tform]
                assert len(_keys) == len(set(_keys)), "you have tforms that repeat. in matlab this would compound. how to deal with here?"
                methods_tforms_prim.append({_tform[0]:_tform[1].item() for _tform in prim_tform})
                # methods_tforms_prim = sorted(methods_tforms_prim)
    else:
        methods_tforms_prim = None
    
    return methods_tforms, methods_tforms_prim

        
def psychogood_make_task_template_to_find_psycho_variants(plan, extra_tform_params, morph_info, psycho_params):
    """ Generate a template (reprenetaiton of thet ask) which will be identical across all
    tasks in the same psycho group, by returning task after excising infomation about the 
    value of the psycho param

    out==out can be used to find tasks in the same psycho
    
    PARAMS:
    - plan
    - extra_tform_params
    - morph_info
    - psycho_params
    ---> See calling code for how get these.

    """
    from pythonlib.tools.listtools import stringify_list
    import copy

    # Simplify the inputs
    methods_tforms, methods_tforms_prim = simplify_extra_tform_dict(extra_tform_params)
    morph_info = simplify_morph_info_dict(morph_info)

    idx_prim_plan = 0

    def _remove_tform_values(methods_tforms, methods_tforms_prim, idx_prim_remove):
        """
        Remove any information aboiut the valuye of the trforms
        for tforms, keep just the keys
        PARAMS:
        EXAMPLE:
            methods_tforms_prims = [{'sx': 0.16666666666666666}, None, {'sx': 0.5}]
            -->
            (if idx_prim_remove == 0)
            methods_tforms_prim_keys = [['sx'], None, ['sx', 0.5]]

            (if idx_prim_remove == "all")
            methods_tforms_prim_keys = [['sx'], None, ['sx']]

            (if idx_prim_remove == [0, 2])
            methods_tforms_prim_keys = [['sx'], None, ['sx']]

            (if idx_prim_remove == 2)
            methods_tforms_prim_keys = [['sx', 0.16666666666666666], None, ['sx']]

            (if idx_prim_remove == None)
            methods_tforms_prim_keys = [['sx', 0.16666666666666666], None, ['sx', 0.5]]

        """

        if methods_tforms is not None:
            methods_tforms_keys = list(methods_tforms.keys())
        else:
            methods_tforms_keys = None

        methods_tforms_prim_keys = []
        if methods_tforms_prim is not None:
            for _i, tform in enumerate(methods_tforms_prim): # each prim
                if tform is not None and idx_prim_remove=="all":
                    # Then this is the idx that you want to remove values from
                    methods_tforms_prim_keys.append(list(tform.keys()))
                elif tform is not None and _i == idx_prim_remove:
                    # Then this is the idx that you want to remove values from
                    methods_tforms_prim_keys.append(list(tform.keys()))
                elif tform is not None and isinstance(idx_prim_remove, list) and _i in idx_prim_remove:
                    # Then this is the idx that you want to remove values from
                    methods_tforms_prim_keys.append(list(tform.keys()))
                elif tform is not None:
                    # Keep all values
                    methods_tforms_prim_keys.append(list(tform.keys()) + list(tform.values()))
                else:
                    methods_tforms_prim_keys.append(None)
        else:
            methods_tforms_prim_keys = None

        return methods_tforms_keys,  methods_tforms_prim_keys

    # Based on psychoparams, decide what strings to exclude from template, i.e., exclude the values for the psycho var.
    if psycho_params is None:
        # Then template will include all info, i.e., doesnt excise out the psycho value.
        # useful for finding base prims, at diff locations (which ahve same plan but diff LOS)
        plan_prim = plan[idx_prim_plan] # Ignore the global rels
        template = stringify_list(plan_prim) + stringify_list(methods_tforms) + stringify_list(methods_tforms_prim) + stringify_list(morph_info)
    else:
        psycho_ver = psycho_params["psycho_ver"]
        if "exclude_extra_tform_values" in psycho_params:
            exclude_extra_tform_values = psycho_params["exclude_extra_tform_values"]
        else:
            exclude_extra_tform_values = False

        if "exclude_interp_values" in psycho_params:
            exclude_interp_values = psycho_params["exclude_interp_values"]
        else:
            exclude_interp_values = False

        if "convert_extra_tforms_to_sign" in psycho_params:
            convert_extra_tforms_to_sign = psycho_params["convert_extra_tforms_to_sign"]
        else:
            convert_extra_tforms_to_sign = False

        if "replace_morph_params_with_binary_whether_exists" in psycho_params:
            replace_morph_params_with_binary_whether_exists = psycho_params["replace_morph_params_with_binary_whether_exists"]
        else:
            replace_morph_params_with_binary_whether_exists = False

        # Clean up morph info.
        if morph_info is None or len(morph_info)==0:
            morph_info = [] 
        else:
            if replace_morph_params_with_binary_whether_exists:
                # Still note down that params exist, just exclude the values.
                morph_info = [{"exists":"exists"}, {"exists":"exists"}]
            else:
                # Inculde all params
                tmp = [None, None]  
                tmp[0] = {k:v for k, v in morph_info[0].items() if not k=="frac"}
                tmp[1] = {k:v for k, v in morph_info[1].items() if not k=="frac"}
                morph_info = tmp
        
        ### Generate template
        if psycho_ver == "attachpt1_interp":
            idx_motif_char_flex = psycho_params["idx_motif_char_flex"]
            idx_rel = psycho_params["idx_rel"]
            if "idx_rel_within" in psycho_params:
                idx_rel_within = psycho_params["idx_rel_within"]
            else:
                idx_rel_within = 0

            # (1) Remove any informaiton about the value of the interp
            # make sure this is the correct index
            try:
                # print(plan[idx_motif_char_flex][1][1][2*idx_rel+1][2][0])
                # print(idx_motif_char_flex)
                # print(idx_rel)
                # print("PLAN:", plan[idx_motif_char_flex])
                # assert False
                if "interp_end1_end2" in plan[idx_motif_char_flex][1][1][2*idx_rel+1][2][idx_rel_within]:
                    # print(plan[idx_motif_char_flex][1][1][2*idx_rel+1][2][0])
                    plan = copy.deepcopy(plan)
                    # Then this is has the interp at the correct place. excise the value.
                    # replace it by removing the actual interp vallue
                    plan[idx_motif_char_flex][1][1][2*idx_rel+1][2][idx_rel_within] = "interp_end1_end2"
            except IndexError as err:
                # HACKY, this plan just doesnt even have this index...
                pass
            except Exception as err:
                raise err
            
            plan_prim = plan[idx_prim_plan] # Ignore the global rels

            # Optional ways to modify the template.
            if convert_extra_tforms_to_sign and methods_tforms_prim is not None:
                # useful if sign of this variable distinguishes this psycho group from another similar one.
                methods_tforms_prim = [{k:np.sign(v) for k, v in tform.items()} for tform in methods_tforms_prim if tform is not None]
            if convert_extra_tforms_to_sign and methods_tforms is not None:
                methods_tforms = {k:np.sign(v) for k, v in methods_tforms.items()}
            
            if exclude_extra_tform_values:
                methods_tforms, methods_tforms_prim = _remove_tform_values(methods_tforms, methods_tforms_prim, idx_prim_remove="all")
            
            template = stringify_list(plan_prim) + stringify_list(methods_tforms) + stringify_list(methods_tforms_prim) + stringify_list(morph_info)

        elif psycho_ver in ["extra_tforms", "extra_tforms_each_prim"]:
            # Tforms are in extra tforms.
            
            idx_prim_remove = psycho_params["idx_prim"]
            
            methods_tforms_keys, methods_tforms_prim_keys = _remove_tform_values(methods_tforms, methods_tforms_prim, idx_prim_remove)

            plan_prim = plan[idx_prim_plan] # Ignore the global rels
            template = stringify_list(plan_prim) + stringify_list(methods_tforms_keys) + stringify_list(methods_tforms_prim_keys) + stringify_list(morph_info)
        
        elif psycho_ver in ["morph_frac"]:
            # value is in morph info

            plan_prim = plan[idx_prim_plan] # Ignore the global rels

            if morph_info is None or len(morph_info)==0:
                morph_info = [] 
            else:
                # Inculde all params
                tmp = [None, None]  
                tmp[0] = {k:v for k, v in morph_info[0].items() if not k=="frac"}
                tmp[1] = {k:v for k, v in morph_info[1].items() if not k=="frac"}
                morph_info = tmp

            template = stringify_list(plan_prim) + stringify_list(methods_tforms) + stringify_list(methods_tforms_prim) + stringify_list(morph_info)
        else:
            print(psycho_ver)
            assert False
        
        if exclude_interp_values:
            template = [t for t in template if "interp_end1_end2" not in t]

    return template


def psychogood_extract_psycho_var(plan, extra_tform_params, morph_info, psycho_params):
    """ 
    Extract scalar variable that defines vairation that defines differences across
    tasks within same psychoemtric group
    PARAMS:
    - plan, 
    - extra_tform_params,
    - morph_info
    - psycho_params
    ---> See calling code for how get above.
    RETURNS:
    - psycho_var, scalar value of the psycho variable
    """

    psycho_ver = psycho_params["psycho_ver"]
    
    if psycho_ver == "attachpt1_interp":
        # Interp of attach pt.

        # plan = 
        # (['motif_char_flex',
        #   ['direct_chunk_entry',
        #    [['line',
        #      ['prot',
        #       array(18.),
        #       array(2.),
        #       array([], shape=(0, 0), dtype=float64),
        #       array(0.)]],
        #     ['translate_xy', array(0.), ['center', 'center', array([0., 0.])]],
        #     ['line',
        #      ['prot',
        #       array(18.),
        #       array(1.),
        #       array([], shape=(0, 0), dtype=float64),
        #       array(0.)]],
        #     ['translate_xy',
        #      array(-1.),
        #      ['interp_end1_end2_50', 'end1', array([0., 0.])]]]]],
        #  ['translate_xy',
        #   array(0.),
        #   ['center_prim_sketchpad', 'center_prim_sketchpad', array([ 1. , -0.2])]])
        # ---> pulls out 50 (from interp_end1_end2_50)

        idx_motif_char_flex = psycho_params["idx_motif_char_flex"]
        idx_rel = psycho_params["idx_rel"]
        if "idx_rel_within" in psycho_params:
            idx_rel_within = psycho_params["idx_rel_within"]
        else:
            idx_rel_within = 0

        assert plan[idx_motif_char_flex][1][0] == "direct_chunk_entry"
        _prims = plan[idx_motif_char_flex][1][1][0::2]
        _rels = plan[idx_motif_char_flex][1][1][1::2]

        if psycho_ver == "attachpt1_interp":
            interp_string = _rels[idx_rel][2][idx_rel_within] # e.g, interp_end1_end2_50
            i = interp_string.find("interp_end1_end2")
            interp_frac = int(interp_string[i+17:]) # .e.g, 50
        
        psycho_value = interp_frac

    elif psycho_ver == "extra_tforms_each_prim":
        # Variable in extra tforms

        idx_prim = psycho_params["idx_prim"]
        tform_key = psycho_params["tform_key"]

        methods_tforms, methods_tforms_prim = simplify_extra_tform_dict(extra_tform_params)

        if isinstance(idx_prim, list):
            # Take sum of the values... Hacky way to convert 2d psycho to 1d...
            _vals = [methods_tforms_prim[_i][tform_key] for _i in idx_prim]
            _val_diff = np.abs(max(_vals) - min(_vals)) # Take diff, as tiebreak
            _val_diff_2 = _vals[0] - _vals[1] # Take diff, as tiebreak
            
            tform_value = sum(_vals) - 0.1*_val_diff - 0.01*_val_diff_2

        else:
            tform_value = methods_tforms_prim[idx_prim][tform_key]

        psycho_value = tform_value
    elif psycho_ver == "morph_frac":
        # Morph between two base prims.

        # morph_info = D.shapesemantic_taskclass_cont_morph_extract_params(1113)
        
        # morph_info = 
        # [{'los': ('singleprims', 114, 9),
        # 'shape': ('arcdeep', 4, 4, 0),
        # 'frac': 0.6666666666666666,
        # 'flip_task1': False,
        # 'flip_task2': False,
        # 'frac_func': array([], shape=(0, 0), dtype=float64)},
        # {'los': ('singleprims', 114, 30),
        # 'shape': ('circle', 6, 1, 0),
        # 'frac': 0.6666666666666666,
        # 'flip_task1': False,
        # 'flip_task2': False,
        # 'frac_func': array([], shape=(0, 0), dtype=float64)}]

        assert np.all(np.isclose(morph_info[0]["frac"], morph_info[1]["frac"]))
        
        psycho_value = morph_info[0]["frac"].item()
    else:
        print(psycho_ver)
        assert False, "code it for this psycho_ver"

    return psycho_value

def psychogood_find_tasks_in_this_psycho_group_wrapper(D, los_within, psycho_params):
    """
    Good, Return all trials (los) that are in the psycho group that includes this los_within, and
    with variation consistent with pyscho_params.
    PARAMS:
    - los_within, (str, int, int).
    - psycho_params, dict of params defining what the psycho variable is.
    NOTE: if None, then looks for identical task plans (allowing for variation in location)
    """
    df = D.Dat[D.Dat["los_info"] == los_within]
    if len(df)==0:
        print(los_within)
        assert False, "Cant find any trials with this los... you made mistake in entering params"

    ind = D.Dat[D.Dat["los_info"] == los_within].index[0]

    # (1) Make a template for a single case within the interp
    plan = D.taskclass_extract_low_level_plan(ind)
    extra_tform_params = D.taskclass_extract_prims_extra_params_tforms(ind)
    morph_info = D.shapesemantic_taskclass_cont_morph_extract_params(ind)
    template = psychogood_make_task_template_to_find_psycho_variants(plan, extra_tform_params, morph_info, psycho_params)

    # (2) Find all trials that match this template -- i.e. tasks within same psycho group
    res = []
    for ind in range(len(D.Dat)):
        plan_this = D.taskclass_extract_low_level_plan(ind)
        extra_tform_params_this = D.taskclass_extract_prims_extra_params_tforms(ind)
        morph_info_this = D.shapesemantic_taskclass_cont_morph_extract_params(ind)
        template_this = psychogood_make_task_template_to_find_psycho_variants(plan_this, extra_tform_params_this, morph_info_this, psycho_params)

        if template == template_this:
            # Found a match to los_within.
            los_this = D.taskclass_extract_los_info(ind)

            if psycho_params is not None:
                psycho_value = psychogood_extract_psycho_var(plan_this, extra_tform_params_this, morph_info_this, psycho_params)
            else:
                psycho_value = np.nan

            res.append({
                "idx_dat":ind,
                "psycho_value":psycho_value,
                "los":los_this
            })
            print(ind, " --- los:", los_this, " -- value:", psycho_value)
    
    return pd.DataFrame(res)

def psychogood_find_tasks_in_this_psycho_group_wrapper_manual_helper(D, example_los_1, example_los_2, psycho_params_try):
    """ 
    Helper to identify what params to input to manually enter required info
    for identifying psyho groups uisng find_tasks_in_this_psycho_group_wrapper. Does this by entering two
    example los that are within group, and a guess as params. 
    Determines whether the psycho params inputed successfully calls los1 and los1 as same group. If so, then copy and paster
    the output in to PARAMS> If Not ,then read what's printed, and updated psycho_params_try accordingly...

    See 210506_.. notebook for example how to run this.

    PARAMS:
    - example_los_1, los (str, num, num), which is one of two los that are within this group for use in testing 
    these params.
    - example_los_2, los, the other that is also in the group.
    RETURNS:
    - Prints to console --> bool, whether these two los would be identified as being in the same psycho group,
    given these psycho_params. If they are then you can use these params in find_tasks_in_this_psycho_group_wrapper:
    -- los_within = <example_los_1> or <example_los_2>
    -- psycho_params = psycho_params_try
    """
    # [GOOD HELPER] To decode what psycho params to enter

    # (1) Input tasks that are in the same psycho group, such that they have one param difference, the param
    # you want to vary. Find these using your notes for this experiment, and findings tasks in the makeDrawTaskSets folder.
    # los1 = ("singleprims_psycho", 9, 1)
    # los2 = ("singleprims_psycho", 10, 1)
    los1 = example_los_1
    los2 = example_los_2

    # (2) Enter candidate psycho params. This picks out the specific one param that is expected to vary across los in this group.
    # Here, can enter and iterate, with goal that template1 == template2 below.
    psycho_params = psycho_params_try

    # Example ways to entee:
    
    # (i) Varying a prim-specific tform [structured morphs]
    # psycho_params = {
    #     "psycho_ver":"extra_tforms_each_prim",
    #     "idx_prim": 1, # the substroke index
    #     "tform_key": "sx" # scale x
    # }
    # -- Example task:
    # Extra tform params:
    # {'tforms': {}, 'tforms_each_prim_p': [{}, [['sx', array(0.16666667)]]]}
    # Extra tform params:
    # {'tforms': {}, 'tforms_each_prim_p': [{}, [['sx', array(0.5)]]]}

    # (ii) Varying attach pt [structured morphs]
    # psycho_params = {
    #     "psycho_ver":"attachpt1_interp",
    #     "idx_motif_char_flex": 0, # Usuall dont change (if this is one "prim"), i.e,. this is the outer index.
    #     "idx_rel": 1, # This is the index into the relations, within each plan
    #     "exclude_extra_tform_values":False, # Turn this True if the morphs also have diff in extra tforms (each prim), and you want to be invariant over those
    # }
    # -- Example task:
    # (['motif_char_flex',
    #   ['direct_chunk_entry',
    #    [['line',
    #      ['prot',
    #       array(18.),
    #       array(1.),
    #       array([], shape=(0, 0), dtype=float64),
    #       array(0.)]],
    #     ['translate_xy', array(0.), ['center', 'center', array([0., 0.])]],
    #     ['line',
    #      ['prot',
    #       array(18.),
    #       array(2.),
    #       array([], shape=(0, 0), dtype=float64),
    #       array(0.)]],
    #     ['translate_xy',
    #      array(-1.),
    #      ['interp_end1_end2_17', 'end2', array([0., 0.])]]]]],
    #  ['translate_xy',
    #   array(0.),
    #   ['center_prim_sketchpad', 'center_prim_sketchpad', array([-1. ,  0.8])]])
    # Plan:
    # (['motif_char_flex',
    #   ['direct_chunk_entry',
    #    [['line',
    #      ['prot',
    #       array(18.),
    #       array(1.),
    #       array([], shape=(0, 0), dtype=float64),
    #       array(0.)]],
    #     ['translate_xy', array(0.), ['center', 'center', array([0., 0.])]],
    #     ['line',
    #      ['prot',
    #       array(18.),
    #       array(2.),
    #       array([], shape=(0, 0), dtype=float64),
    #       array(0.)]],
    #     ['translate_xy',
    #      array(-1.),
    #      ['interp_end1_end2_50', 'end2', array([0., 0.])]]]]],
    #  ['translate_xy',
    #   array(0.),
    #   ['center_prim_sketchpad', 'center_prim_sketchpad', array([-1. ,  0.8])]])

    # (iii) Varying attach pt [structured morphs] and also varying extra tforms, but you want to be invariant over tforms,
    # so you convert tforms to their sign (I did this when + and - tforms indicated two different psycho groups)
    # psycho_params = {
    #     "psycho_ver":"attachpt1_interp",
    #     "idx_motif_char_flex": 0, # Usuall dont change (if this is one "prim"), i.e,. this is the outer index.
    #     "idx_rel": 1, # This is the index into the relations, within each plan
    #     "exclude_extra_tform_values":False,
    #     "exclude_interp_values":False,
    #     "convert_extra_tforms_to_sign":True 
    # }    

    # (iv) Morphing between two tasks (e.g,, arc <--> circle)
    # psycho_params = {
    #     "psycho_ver":"morph_frac",
    #     "replace_morph_params_with_binary_whether_exists":True, # Turn True so that is not affected by minute differences in params.
    # }


    # (3) Extract task infomraiton
    print("----- Printing information about each task")

    # Task 1
    ind = D.Dat[D.Dat["los_info"] == los1].index[0]
    plan1 = D.taskclass_extract_low_level_plan(ind)
    extra_tform_params1 = D.taskclass_extract_prims_extra_params_tforms(ind)
    morph_info1 = D.shapesemantic_taskclass_cont_morph_extract_params(ind)
    template1 = psychogood_make_task_template_to_find_psycho_variants(plan1, extra_tform_params1, morph_info1, psycho_params)
    psycho_var_1 = psychogood_extract_psycho_var(plan1, extra_tform_params1, morph_info1, psycho_params)

    print("= Task 1")
    print("  Plan:")
    display(plan1)
    print("  Extra tform params:")
    print(extra_tform_params1)
    print("  Morph info:")
    print(morph_info1)
    print("  Psycho var:")
    print(psycho_var_1)

    # Task 2
    ind = D.Dat[D.Dat["los_info"] == los2].index[0]
    plan2 = D.taskclass_extract_low_level_plan(ind)
    extra_tform_params2 = D.taskclass_extract_prims_extra_params_tforms(ind)
    morph_info2 = D.shapesemantic_taskclass_cont_morph_extract_params(ind)
    template2 = psychogood_make_task_template_to_find_psycho_variants(plan2, extra_tform_params2, morph_info2, psycho_params)
    psycho_var_2 = psychogood_extract_psycho_var(plan2, extra_tform_params2, morph_info2, psycho_params)

    print("= Task 2")
    print("  Plan:")
    display(plan2)
    print("  Extra tform params:")
    print(extra_tform_params2)
    print("  Morph info:")
    print(morph_info2)
    print("  Psycho var:")
    print(psycho_var_2)

    ### TEST whether these tasks are considered same psycho group
    print(" ")
    print("------ Results from comparing these two tasks:")
    same_psycho_group = template1 == template2
    print("Were tasks identified as same psycho group? ", same_psycho_group)

    if not same_psycho_group:
        print("Differences between task represnetations (Use these to determine how to change psycho params, if needed):")
        print(len(template1), len(template2))
        for i, (x, y) in enumerate(zip(template1, template2)):
            if not x==y:
                print(i, " -- ", x, y)
    
    # print(f"los_within = {example_los_1}")
    # print(f"psycho_params = {psycho_params}")
    print("If this is same group, then copy and paste the following in to PARAMS:")
    print(f"'los_within': {example_los_1}, 'psycho_params': {psycho_params}")
    