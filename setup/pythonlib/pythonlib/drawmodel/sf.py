"""
works with sf, dataframe where each column is single stroke.
    this can be extracted into Dataset class,
    using methods prefixed by "sf"
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from ..dataset.dataset_strokes import DatStrokes

# def sim_matrix_compute_wrapper(strokes_data, strokes_basis, ver):
#     """ high-level, differen versions that are known to work well
#     """
    

#     # euclidian (2x)
#     if ver in ["euclidian", "euclidian_diffs"]:
#         # align to onset
#         strokes_data = align_to_onset(strokes_data)
#         strokes_data = align_to_onset(strokes_data)
#         simmat = computeSimMatrixGivenBasis(strokes_data, strokes_basis, 
#             distancever=distancever) 


#     # hausdorff (not mean)

#     # in all cases take one-Minus (not divide by max)

# def computeSimMatrixGivenBasis(strokes_data, strokes_basis, distancever,
#         rescale_strokes_ver=None, npts_space=50, DEBUG=False):
#     """ [gOOD] Wrapper to compute sim matrix between strokes_data and strokes_bsais
#     PARAMS:
#     - strokes_data, list of np array
#     - strokes_basis, same, but the basis set
#     RETURNS:
#     - similarity_matrix, siimlarity matrix, (ndat, nbasis).
#     NOTE:
#     - range_norm were decided by inspecting histrograms of distances using DEBUG, for
#     Pancho charstrokeseqpan1b (around Jan 14th 2023). Erring on side of range_norm[1] being
#     lower, to increase the dynamic range for the strokes with low distance (high sim).
#     """
#     from pythonlib.tools.stroketools import rescaleStrokes, strokesInterpolate2, strokes_alignonset, strokes_centerize
#     from .strokedists import distStrokWrapperMult
#
#     if DEBUG:
#         print("TO debug, switch debug ona nd off by hand for each one below...")
#         assert False
#
#     assert len(strokes_data)>0
#     assert len(strokes_basis)>0
#
#     ############# Extract data
#     assert not isinstance(strokes_data, pd.core.frame.DataFrame), "deprecated. use list strokes"
#
#     ############# Preprocess
#     if rescale_strokes_ver=="stretch_to_1":
#         assert False, "confirm this doesnt mutate strokes_data, then remove this assert"
#         strokes_data = [rescaleStrokes([s])[0] for s in strokes_data]
#         strokes_basis = [rescaleStrokes([s])[0] for s in strokes_basis]
#     else:
#         assert rescale_strokes_ver is None, "which?"
#
#     # interpolate
#     if distancever in ["euclidian", "euclidian_diffs", "euclidian_both"]:
#         # then need to be same length
#         def _all_same_length(stroklist):
#             """Returns True if all strok in stroklist are same npts
#             otherwise False"""
#             list_n = [len(strok) for strok in stroklist]
#             if len(set(list_n))==1:
#                 return True
#             else:
#                 return False
#
#         if _all_same_length(strokes_data + strokes_basis):
#             pass
#         else:
#             strokes_data = strokesInterpolate2(strokes_data, N=["npts", npts_space], base="space")
#             strokes_basis = strokesInterpolate2(strokes_basis, N=["npts", npts_space], base="space")
#
#     ### Cmpute sim matrix
#     if distancever == "euclidian":
#         # Pt by pt euclidian, after aligning by onsets.
#
#         # align to onset
#         strokes_data = strokes_alignonset(strokes_data)
#         strokes_basis = strokes_alignonset(strokes_basis)
#
#         similarity_matrix = distStrokWrapperMult(strokes_data, strokes_basis, distancever=distancever,
#                                                  convert_to_similarity=True, similarity_method="squared_one_minus",
#                                                  normalize_by_range=True, range_norm=[0, 220], DEBUG=False)
#
#     elif distancever == "euclidian_diffs":
#         # PT by pt euclidian of the differences between pts at adajcent timepoints
#
#         # align to onset
#         strokes_data = strokes_alignonset(strokes_data)
#         strokes_basis = strokes_alignonset(strokes_basis)
#
#         similarity_matrix = distStrokWrapperMult(strokes_data, strokes_basis, distancever=distancever,
#                                                  convert_to_similarity=True, similarity_method="squared_one_minus",
#                                                  normalize_by_range=True, range_norm=[0, 16], DEBUG=False)
#
#     elif distancever == "hausdorff_alignedonset":
#         # Hausdorff (spatial), after aligning strokes by their onsets
#
#         # align to onset
#         strokes_data = strokes_alignonset(strokes_data)
#         strokes_basis = strokes_alignonset(strokes_basis)
#
#         similarity_matrix = distStrokWrapperMult(strokes_data, strokes_basis, distancever="hausdorff",
#                                                  convert_to_similarity=True, similarity_method="squared_one_minus",
#                                                  normalize_by_range=True, range_norm=[2, 145], DEBUG=False)
#
#     elif distancever == "hausdorff_centered":
#         # Hausdorff (spatial), after centering strokes in space
#
#         # center
#         strokes_data = strokes_centerize(strokes_data)
#         strokes_basis = strokes_centerize(strokes_basis)
#
#         similarity_matrix = distStrokWrapperMult(strokes_data, strokes_basis, distancever="hausdorff",
#                                                  convert_to_similarity=True, similarity_method="squared_one_minus",
#                                                  normalize_by_range=True, range_norm=[2, 145], DEBUG=False)
#
#     elif distancever == "hausdorff_max":
#         # Hausdorff (spatial), after centering strokes in space
#
#         # center
#         strokes_data = strokes_centerize(strokes_data)
#         strokes_basis = strokes_centerize(strokes_basis)
#
#         similarity_matrix = distStrokWrapperMult(strokes_data, strokes_basis, distancever="hausdorff_max",
#                                                  convert_to_similarity=True, similarity_method="squared_one_minus",
#                                                  normalize_by_range=True, range_norm=[2, 145], DEBUG=False)
#
#
#     else:
#         print(distancever)
#         assert False
#
#     return similarity_matrix


def computeSimMatrix(SF, rescale_strokes_ver="idxs_stroklist_dat", 
        distancever="euclidian_diffs", npts_space=70, Nbasis=300):
    """ Get distance matrix, entire dataset, with random instances chosen for basis
    PARAMS:
    """

    if Nbasis>len(SF):
        print(Nbasis)
        print(len(SF))
        assert False, "not enough data..."

    idxs_stroklist_dat = list(range(len(SF)))
    idxs_stroklist_basis = random.sample(range(len(SF)), Nbasis)

    # Get sim matrix
    similarity_matrix = computeSimMatrixGivenBasis(SF, SF.iloc[idxs_stroklist_basis], 
        rescale_strokes_ver, distancever, npts_space)

    return similarity_matrix, idxs_stroklist_basis


def preprocessStroksList(strokes, params):
    """ Preprioess strokes same as
    preprocessStroks, but takes in list of np arrays
    PARAMS:
    - strokes, list of np arrays
    - params, dict, see preprocessStroks
    RETURNS:
    - strokes, list of np arrays modified. does not modify input.
    """

    dfthis = pd.DataFrame({"strok":strokes})
    dfthis = preprocessStroks(dfthis, params)
    strokes = dfthis["strok"].tolist()
    return strokes

def preprocessStroks(df, params):
    """
    preprocessing, including normalizing, centering, etc, on a set of strok data.
    - df, dataframe with column "strok", holding a np.array for its strok
    - params, dict, telling what preprocessing to do.
    RETURNS:
    - df, shape shape as df, but with strok replaced with processed.
    (no modify in place)
    NOTE: resets index for df in return.
    NOTE:
    - Tested speed using two methods, either modify each row in df, or first convert to
    list then do list comprehension. 
    Run: %timeit preprocessStroks(SF, params)
    """
    from pythonlib.tools.pandastools import applyFunctionToAllRows
    from pythonlib.tools.stroketools import rescaleStrokes, strokesInterpolate2

    # ANOTHER METHOD, applied to df directly. is slower, see notse above, so do not use
#     if params["align_to_onset"]:
#         def F(x):
#             return x["strok"] - x["strok"][0,:]
#         df = applyFunctionToAllRows(df, F, "stroknew")
# #             df["strok"] = df["stroknew"] # replace old
# #             del df["stroknew"]
# #             print(df.iloc[0])
            
    # Fill gaps with defaults
    params_default = {
        "align_to_onset":False,
        "min_stroke_length_percentile":None,
        "min_stroke_length":None,
        "max_stroke_length_percentile":None,
        "centerize":False,
        "rescale_ver":None
    }
    for k, v in params_default.items():
        if k not in params.keys():
            params[k] = v

    # Sanity check
    assert params["align_to_onset"] != params["centerize"], "cannot do both"

    print("starting length of dataframe:")
    print(len(df))

    # pre-extract features,. if needed
    if params["max_stroke_length_percentile"] or params["min_stroke_length_percentile"] or params["min_stroke_length"]:
        # then get distance(length) for each strok
        if "distance" not in df.columns:
            from pythonlib.drawmodel.features import strokeDistances
            strok_lengths = strokeDistances(list(df["strok"].values))
            df["distance"] = strok_lengths
    
    if params["max_stroke_length_percentile"] is not None:
        thresh =  np.percentile(df["distance"], [params["max_stroke_length_percentile"]])
        # if False:
        #     plt.figure(figsize=(15,5))
        #     plt.hist(SF["distance"], 100);
        #     plt.axvline(thresh)
        #     print(thresh)
        #     assert False
            
        def F(x):
            return x["distance"]<thresh
        df = applyFunctionToAllRows(df, F, "keep")
        df = df[df["keep"]==True]
        del df["keep"]
        # df = df.reset_index()

    # if "min_stroke_length_percentile" in params.keys():
    if params["min_stroke_length_percentile"] is not None:
        thresh =  np.percentile(df["distance"], [params["min_stroke_length_percentile"]])
        # if False:
        #     plt.figure(figsize=(15,5))
        #     plt.hist(SF["distance"], 100);
        #     plt.axvline(thresh)
        #     print(thresh)
        #     assert False
        def F(x):
            return x["distance"]>thresh
        df = applyFunctionToAllRows(df, F, "keep")
        df = df[df["keep"]==True]
        del df["keep"]
        # df = df.reset_index()

    # if "min_stroke_length" in params.keys():
    if params["min_stroke_length"] is not None:
        thresh =  params["min_stroke_length"]
        def F(x):
            return x["distance"]>thresh
        df = applyFunctionToAllRows(df, F, "keep")
        df = df[df["keep"]==True]
        del df["keep"]
        # df = df.reset_index()
 
    # === pull out stroks.
    stroklist = list(df["strok"].values)
        
    assert False, "see strokes_alignonset and strokes_centerize in stroketools"
    if params["align_to_onset"]:
        def F(strok):
            return strok - strok[0,:]
        stroklist = [F(strok) for strok in stroklist]
        
    if params["centerize"]:
        def F(strok):
            c = np.mean(strok[:,:2], axis=0)
            s = strok.copy()
            s[:,:2] = s[:,:2] - c
            return s
        stroklist = [F(strok) for strok in stroklist]

    if params["rescale_ver"] is not None:
        stroklist = [rescaleStrokes([s], ver=params["rescale_ver"])[0] for s in stroklist]

    # === finally, put back into df
    df["strok"] = stroklist
    
    print("after filtering:")
    print(len(df))

    df = df.reset_index(drop=True)
    
    return df