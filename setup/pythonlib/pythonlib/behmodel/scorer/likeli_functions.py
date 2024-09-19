""" quickly coded likeli functions

"""

import numpy as np
from pythonlib.drawmodel.strokedists import distscalarStrokes



def makeLikeliFunction(ver="segments", norm_by_num_strokes=True, 
    standardize_strokes=False, asymmetric=False):
    """ returns a function that can pass into Model,
    which does the job of computing likeli (ie., model human dist) 
    CONVENTION: more positive is better match.
    NOTE: by defai;t tje split_segments version does take into acocunt directionality.
    # 
    - norm_by_num_strokes this default, shoudl be, for cases where distance is
    summed over strokes. This is true for distanceDTW.
    - standardize_strokes, then will subtract mean and divide by x liomits range
    - combine_beh_segments, then combines all segments for animal into a single stroke.
    """
    assert False, "old code, merge this with below"
    if ver in ["split_segments", "timepoints"]:
        # these differ depending on direction fo strokes
        print("NOTE: should get parses in both directions [default for getParses()], since this distance function cares about the chron order.")

    def likeliFunction(t):
        dists_all = []
        for p in t["model_parses"]:
            strokes_parse = p["strokes"]
            if ver=="modHaussdorf":
                dist = -distanceBetweenStrokes(t["behavior"]["strokes"], strokes_parse)
                if norm_by_num_strokes:
                    dist = dist/len(t["behavior"]["strokes"])
            elif ver in ["segments", "combine_segments"]:
                from pythonlib.tools.stroketools import distanceDTW
                if ver=="combine_segments":
                    S = [np.concatenate(t["behavior"]["strokes"], axis=0)]
                    v = "segments"
                else:
                    S = t["behavior"]["strokes"]
                    v = "segments"
                dist = -distanceDTW(S, strokes_parse, 
                    ver=v, norm_by_numstrokes=norm_by_num_strokes, asymmetric=asymmetric)[0]
            else:
                assert False, "not coded"

            dists_all.append(dist)
        return dists_all
    return likeliFunction



def likeli_dataset(parser_names = None):
    """ 
    Get likeli function that operates on a datset (single ind)
    - Also assumes that operates on all parses.
    OUT:
    - Scorer
    """    
    from pythonlib.behmodel.scorer.scorer import Scorer
    if parser_names is None:
        parser_names = ["parser_graphmod", "parser_nographmod"]
    def F(D, ind):
        strokes_beh = D.Dat.iloc[ind]["strokes_beh"]
        list_of_parsestrokes = D.parser_list_of_parses(ind, kind="strokes", parser_names=parser_names)
        scores = np.array([distscalarStrokes(strokes_beh, strokes_parse, "dtw_segments") for strokes_parse in list_of_parsestrokes])
        return 1/scores
    
    Li = Scorer()
    Li.input_score_function(F)
    return Li


def likeli_scorer_quick(ver):
    """ quick return of scorere, given ver
    OUT:
    - Scorer
    """
    from pythonlib.behmodel.scorer.scorer import Scorer
    from pythonlib.behmodel.scorer.utils import normscore

    if ver=="base":
        def F(D, ind):
            strokes_beh = D.Dat.iloc[ind]["strokes_beh"]
            list_of_parsestrokes = D.parserflat_extract_strokes(ind) # list of strokes
            scores = np.array([distscalarStrokes(strokes_beh, strokes_parse, "dtw_segments") for strokes_parse in list_of_parsestrokes])
            return 1/scores
    elif ver=="base_graphmodonly":
        # For gridlinecircle+, only one parser per troal. not super important, but here makes sure doesnt
        # mistake the index by flattening the parses.
        def F(D, ind):
            strokes_beh = D.Dat.iloc[ind]["strokes_beh"]
            P = D.parser_get_parser_helper(ind)
            scores = []
            # print("[likeli_functions] getting scores")
            for i in range(len(P.Parses)):
                strokes_parse = P.extract_parses_wrapper(i, "strokes")
                scores.append(distscalarStrokes(strokes_beh, strokes_parse, "dtw_segments"))
            scores = np.array(scores)

                # scores = np.array([distscalarStrokes(strokes_beh, strokes_parse, "dtw_segments") for strokes_parse in list_parses_strokes])

            # list_parses_strokes = P.extract_parses_wrapper("all", "strokes")
            # list_of_parsestrokes = D.parserflat_extract_strokes(ind) # list of strokes
            # scores = np.array([distscalarStrokes(strokes_beh, strokes_parse, "dtw_segments") for strokes_parse in list_parses_strokes])

            return 1/scores
    elif ver=="base_graphmodonly_bestperm":
        """ Only bother computing for the bestperm 
        """
        
    else:
        print(ver)
        assert False, "not coded"
    
    Li = Scorer()
    Li.input_score_function(F)

    # Normalization (to log probs)
    def norm(list_scores, params):
        beta = params[0]
        # beta = 1
        log_probs = normscore(list_scores, "log_softmax", [beta, False])
        return log_probs

    Li.input_norm_function(norm)
    Li._do_score_with_params=False
    Li._do_norm_with_params=True

    return Li


### DATABASE OF PRIOR FUNCTIONS
def likeli_function_database(ver, params=None):
    """ general purpose, holder of likeli fucntions which can pass into scorere
    """

    if ver=="monkey_vs_monkey":
        """ compare beh strokes to parses
        """
        def func(D, ind, modelname):
            """ 
            modelname, name of epoch for parses that will compare to strokes_beh
            """
            strokes_beh = D.Dat.iloc[ind]["strokes_beh"]
            parsesname = f"strokes_beh_group_{modelname}"
            list_of_parsestrokes = D.Dat.iloc[ind][parsesname]
            scores = np.array([distscalarStrokes(strokes_beh, strokes_parse, "dtw_segments") for strokes_parse in list_of_parsestrokes])
            return 1/scores
        Li = likeli_scorer_quick(ver="base")
        Li.input_score_function(func)
    elif ver in ["lines5", "default"]:
        Li = likeli_scorer_quick(ver="base")
    elif ver in ["chunks"]:
        # like base, but assuming only a single parser.
        Li = likeli_scorer_quick(ver="base_graphmodonly")
    else:
        print(ver)
        assert False
    return Li
