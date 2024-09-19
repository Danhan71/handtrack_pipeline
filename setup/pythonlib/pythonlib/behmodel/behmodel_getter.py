""" Functions to help extract scorers for specific kinds of models 
"""

from .scorer.likeli_functions import *
from .scorer.poster_functions import *
from .scorer.prior_functions import *
from pythonlib.behmodel.behmodel import BehModel
from pythonlib.behmodel.scorer.scorer import Scorer



def quick_getter(modelclass, vectorized_priors=False):
    """ For this modelclass (string) returns
    list of beh models (BMs).
    Sort of hard coded currently, just for testing)
    """
    assert False, "obsolete, see below."

    if modelclass=="lines5":

        # Generate models
        list_mod = []
        list_modnames = []
        for rule_model in ["straight", "bent"]:

            # Prior
            if vectorized_priors:
                Pr = prior_scorer_quick(ver="lines5", input_parsesflat=True)
            else:
                Pr = prior_scorer_quick(ver="lines5", input_parsesflat=False)

            # Likeli
            Li = likeli_scorer_quick(ver="base")

            # Post
            Po = poster_dataset()

            # Model
            BM = BehModel()
            BM.input_model_components(Pr, Li, Po)
            
            # BM._list_input_args_likeli = ("dat", "trial", "modelname")
            # BM._list_input_args_prior = ("parsesflat")
            BM._list_input_args_likeli = ("dat", "trial", "modelname")
            
            if vectorized_priors:
                BM._list_input_args_prior = ("parsesflat", "trialcode", "modelname")
            else:
                BM._list_input_args_prior = ("dat", "trial", "modelname")

            list_mod.append(BM)
            list_modnames.append(rule_model)
    else:
        assert False

    return list_mod, list_modnames


def get_single_model(ver, params_input=None):
    """ GOOD , get single BM.
    given ver and params, should always give same object
    """
    from pythonlib.behmodel.scorer.prior_functions import prior_function_database
    from pythonlib.behmodel.scorer.likeli_functions import likeli_function_database

    if ver=="lines5":
        Pr = prior_function_database("lines5")        
        Li = likeli_function_database("lines5")
        Po = poster_dataset(ver="logsumexp")
        BM = BehModel()
        BM.input_model_components(Pr, Li, Po,
            list_input_args_likeli = ("dat", "trial"),
            list_input_args_prior = ("parsesflat", "trialcode"),
            poster_use_log_likeli = True,
            poster_use_log_prior = True)
        BM.Prior.Params = {
            "norm":(1.,)
        }
        BM.Likeli.Params = {
            "norm":(50.,)
        }
    elif ver=="bd":
        # bendiness + direction
        params_mc = {}
        params_mc["thetas"] = {
                "circ":0.,
                "angle_travel":0.}
        params_mc["transformations"] = {
                "angle_travel":tuple([0.])}                
        params_fe = {
            "features":["circ", "angle_travel"]   
        }
        Pr = prior_function_database("default_features", 
            {"params_mc":params_mc,
            "params_fe":params_fe
            })        
        Li = likeli_function_database("default")
        Po = poster_dataset(ver="logsumexp")
        BM = BehModel()
        BM.input_model_components(Pr, Li, Po,
            list_input_args_likeli = ("dat", "trial"),
            list_input_args_prior = ("parsesflat", "trialcode"),
            poster_use_log_likeli = True,
            poster_use_log_prior = True)
        BM.Prior.Params = {
            "norm":(1.,)
        }
        BM.Likeli.Params = {
            "norm":(50.,)
        }

    elif ver=="bdn":
        # bendiness + direction
        params_mc = {}
        params_mc["thetas"] = {
                "circ":0.,
                "angle_travel":0., 
                "nstrokes":0.
                }
        params_mc["theta_limits"] = [
            (-20, 20),
            (0., 10),
            (0., 10)
        ]
        params_mc["transformations"] = {
                "angle_travel":tuple([0.]),
                "nstrokes":tuple([3.])}                
        params_fe = {
            "features":["circ", "angle_travel", "nstrokes"]   
        }
        Pr = prior_function_database("default_features", 
            {"params_mc":params_mc,
            "params_fe":params_fe
            })        
        Li = likeli_function_database("default")
        Po = poster_dataset(ver="logsumexp")
        BM = BehModel()
        BM.input_model_components(Pr, Li, Po,
            list_input_args_likeli = ("dat", "trial"),
            list_input_args_prior = ("parsesflat", "trialcode"),
            poster_use_log_likeli = True,
            poster_use_log_prior = True)
        BM.Prior.Params = {
            "norm":(1.,)
        }
        BM.Likeli.Params = {
            "norm":(50.,)
        }

    elif ver[:12]=="mix_features" and len(ver)>12:
        # e.g., "mix_features_bdn", then the codes are bdn.
        # This useful since allows this string to be a unique model class code.
        codes = ver[13:]
        # convert to list
        codes = [x for x in codes]
        return get_single_model(ver="mix_features",params_input= {"feature_codes":codes})


    elif ver=="mix_features":
        # then pass in params. e.g,:
        # params_input["features"]= {
        #     "circ": [0, None],
        #     "angle_travel": [0., tuple([0.])],
        #     "nstrokes": [0., tuple([3.])]
        # }

        if "features" in params_input.keys():
            # Then you give me the actual values
            F = params_input["features"]
            params_mc = {}
            params_mc["thetas"] = {feat: prms[0] for feat, prms in F.items()}
            params_mc["transformations"] = {feat: prms[1] for feat, prms in F.items() if prms[1] is not None}
            params_fe = {
                "features":list(F.keys())
            }
        else:
            # Then you use pre-saved values:
            def _get_param(feat_code):
                """
                """
                if feat_code=="b":
                    # bendiness
                    return "circ", [0, None]
                elif feat_code=="d":
                    # direction
                    return "angle_travel", [0., tuple([0.])]
                elif feat_code=="n":
                    # num strokes
                    return "nstrokes", [0., tuple([3.])]
                elif feat_code=="t":
                    # travel distance (center to center)
                    return "dist_travel", [0., None]
                else:
                    assert False

            feature_codes = params_input["feature_codes"]
            F = {_get_param(code)[0]:_get_param(code)[1] for code in feature_codes}
            params_mc = {}
            params_mc["thetas"] = {feat: prms[0] for feat, prms in F.items()}
            params_mc["transformations"] = {feat: prms[1] for feat, prms in F.items() if prms[1] is not None}
            params_fe = {
                "features":list(F.keys())
            }

        Pr = prior_function_database("default_features", 
            {"params_mc":params_mc,
            "params_fe":params_fe
            })        
        Li = likeli_function_database("default")
        Po = poster_dataset(ver="logsumexp")
        BM = BehModel()
        BM.input_model_components(Pr, Li, Po,
            list_input_args_likeli = ("dat", "trial"),
            list_input_args_prior = ("parsesflat", "trialcode"),
            poster_use_log_likeli = True,
            poster_use_log_prior = True)
        BM.Prior.Params = {
            "norm":(1.,)
        }
        BM.Likeli.Params = {
            "norm":(50.,)
        }

    elif ver=="random":
        # pick random parse (weigh them evenly)
        Pr = prior_function_database("random")        
        Li = likeli_function_database("default")
        Po = poster_dataset(ver="logsumexp")
        BM = BehModel()
        BM.input_model_components(Pr, Li, Po,
            list_input_args_likeli = ("dat", "trial"),
            list_input_args_prior = ("parsesflat", "trialcode"),
            poster_use_log_likeli = True,
            poster_use_log_prior = True)
        BM.Prior.Params = {
            "norm":(1.,)
        }
        BM.Likeli.Params = {
            "norm":(50.,)
        }

    elif ver=="chunks_redo_extract":
        """ Like chunks, but only chunk-specific preprocessing needed is 
        entering of the chunks into parses,
        and getting of random parses (like non-chunk stuff). the best-parse for each chunk 
        will be gotten here
        """


    elif ver=="chunks":
        # You must have (1) extracted all parses for each task amnd
        # then (2) run D._parser_extract_chunkparses for each beh trial, which saves infor
        # for wich perms are assigned to each baeparse, with each baseparse assigned to a rule for
        # chunking. And then says which perm is the best ranked, in comaprison to beh
        # #TODO: instead of precompting match between perms aned beh shoudl just extract all perms,
        # then this modeling code extract likelis. the best fit is then  likelis. prior is just based 
        # on label given to the perms (whether aligned to the model)

        # expt = params_input["expt"] # e..g, gridlinecircle
        # rule = params_input["rule"] # e..g, lolli

        # (hacky) assuming that have extracted the best-fit already, whihc is a single
        # item in P.Parses[ind], and indicated within the P.Parses[ind] dict. Not fully tested
        # as there were bugs in the extraction of best-fit.
        # that SINGLE parse is all prior peaked on
        # TODO: new method, where the best-fit parses and determined not in preceding step, but instead
        # here as part of behmodel.
        

        Pr = prior_function_database(ver)
        Li = likeli_function_database(ver)
        Po = poster_dataset(ver="logsumexp")
        # Useful for anything with multiple base parses, take the max likeli after 
        # pooling across them.
        PoTest = poster_dataset(ver="maxlikeli_for_permissible_traj") 
        BM = BehModel()
        BM.input_model_components(Pr, Li, Po,
            list_input_args_likeli = ("dat", "trial"),
            list_input_args_prior = ("dat", "trial", "modelname"),
            poster_use_log_likeli = True,
            poster_use_log_prior = True,
            poster_scorer_test = PoTest)
        # BM.Prior.Params = {
        #     "norm":(1.,)
        # }
        BM.Prior.Params = {
            "norm":(500.,)
        }
        BM.Likeli.Params = {
            "norm":(50.,)
        }


    elif ver=="mkvsmk":
        Pr = prior_function_database("monkey_vs_monkey")
        Li = likeli_function_database("monkey_vs_monkey")
        Po = poster_dataset(ver="logsumexp")
        BM = BehModel()
        BM.input_model_components(Pr, Li, Po,             
            list_input_args_likeli = ("dat", "trial", "modelname"),
            list_input_args_prior = ("dat", "trial", "modelname"),
            poster_use_log_likeli = True,
            poster_use_log_prior = True)
        BM.Prior.Params = {
            "norm":(1.,)
        }
        BM.Likeli.Params = {
            "norm":(50.,)
        }
    else:
        assert False

    return BM



def quick_getter_with_params(modelclass, list_mrules):
    """ 
    [GOOD] 
    For this modelclass (string) returns
    list of beh models (BMs).
    NOTE:
    - with_params, means params for prior are exposed, to allow fitting.
    """
    list_mod = []
    list_modnames = []

    # This not important, by default is overwritten by auto detection.
    parsers_to_flatten = ['parser_graphmod', 'parser_nographmod']

    if modelclass=="lines5":
        # list_mrules = ["straight", "bent"]
        for rule_model in list_mrules:
            BM = get_single_model("lines5")
            list_mod.append(BM)
            list_modnames.append(rule_model)
        allow_separate_likelis = False
    elif modelclass in ["mkvsmk", "bd", "bdn", "random", "chunks"]:
        for i, grp in enumerate(list_mrules):
            BM = get_single_model(modelclass)
            list_mod.append(BM)
            list_modnames.append(grp)
        allow_separate_likelis = True
    elif modelclass[:12]=="mix_features":
        for i, grp in enumerate(list_mrules):
            BM = get_single_model(modelclass)
            list_mod.append(BM)
            list_modnames.append(grp)
        allow_separate_likelis = True

    else:
        assert False

    kwargs = {
        "allow_separate_likelis":allow_separate_likelis,
        "parsers_to_flatten":parsers_to_flatten
    }
    return list_mod, list_modnames, kwargs
