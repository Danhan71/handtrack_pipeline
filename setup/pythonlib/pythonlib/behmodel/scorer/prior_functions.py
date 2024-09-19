""" holds quick coded prior functions
(modified from drawmodel.taskmodel)
"""

import numpy as np


# FROM parsing code (older)
# def score_function(parses, ver="ink", normalization = "inverse", test=False,
#                   use_torch=False, origin=None):
#     """ 
#     - ver, str, determines what score to use
#     --- "ink", then total distnace traveled on page
#     --- "travel", then total distance traveled, including
#     gaps, starting from position of first touch.
#     - normalization, how to normalize raw distnace. distance will
#     be that more positive is more worse. 
#     --- inverse take inverse, so that now less positive is worse.
#     --- negative, ...
#     """
#     from pythonlib.drawmodel.features import strokeDistances, computeDistTraveled

#     if test:
#         # then just return random number, one for each parse
#         return torch.tensor([random.random() for _ in range(len(parses))])    
    
#     if ver=="ink":
#         # === Total ink used
#         distances = [np.sum(strokeDistances(strokes)) for strokes in parses]
#     elif ver=="travel":
#         # conisder origin to be onset of first storke.
#         # Note: has issue in that a single stroke task, flipped, is idnetical cost to the same task unflipped.
#         # leads to problems later since unique(score) is used to throw out redundant parses.
#         distances_traveled = [computeDistTraveled(strokes, origin=strokes[0][0,[0,1]]) for strokes in parses]
#         distances = distances_traveled
#     elif ver=="travel_from_orig":
#         # pass in origin. 
#         assert origin is not None, " must pass in coordinate for origin"
#         distances_traveled = [computeDistTraveled(strokes, origin=origin) for strokes in parses]
#         distances = distances_traveled

#     elif ver=="nstrokes":
#         # num strokes
#         # == plit histogram of num strokes
#         nstrokes = [len(p) for p in parses]        
#     else:
#         print(ver)
#         assert False, "not codede"
        
#     if use_torch:
#         distances = torch.tensor(distances)
#     else:
#         distances = np.array(distances)
        
#     if normalization=="inverse":
#         return 1/distances
#     elif normalization=="negative":
#         return -distances
#     else:
#         print(normalization)
#         assert False, "not coded"


# def makePriorFunction(ver="uniform"):
#     """ returns a function that can pass into Model,
#     which does the job of computing prior (i.e., for each model parse 
#     give unnromalized score)
#     - convention: positive is better
#     """
#     NORM_VER = "softmax"

#     if isinstance(ver, str):
#         # pre-coded prior models
#         if ver=="uniform":
#             # just give it a constant
#             priorFunction = lambda x, trial:1
#         elif ver=="prox_to_origin":
#             # first find closest line, then touch closest poitn
#             # on that line, and so on.
#             def getDistFromOrig(point, orig):
#                 return np.linalg.norm(point-orig)

#             def priorFunction(p, trial):
#                 strokes = p["strokes"]
#                 centers = getCentersOfMass(strokes)
#                 orig = trial["task"]["fixpos"]
#                 distances = [getDistFromOrig(c, orig) for c in centers]
#                 s = np.sum(np.diff(distances)) # this is most positive when strokes are ordered from close to far
#                 return s
#         elif ver=="distance_travel":
#             from pythonlib.tools.stroketools import computeDistTraveled
#             def priorFunction(p, trial):
#                 strokes = p["strokes"]
#                 orig = trial["task"]["fixpos"]
#                 cumdist = computeDistTraveled(strokes, orig, include_lift_periods=True)
#                 s = -cumdist # better if distance traveled is shorter
#                 return s
#         elif ver=="angle_test":
#             # qwuickly putting together fake angles, e.g, empirical distrubtion
#             # based on single stroke tasks.

#             from math import pi
#             probs_empirical = {
#                 (0, pi/2):1,
#                 (pi/2, pi):0.25,
#                 (pi, 3*pi/2):0.25,
#                 (3*pi/2, 2*pi):1
#             }

#             def _getprob(dat):
#                 for angles, prob in probs_empirical.items():
#                     if dat>=angles[0] and dat<angles[1]:
#                         return prob
                    
#             def priorFunction(p, trial):
#                 from pythonlib.tools.stroketools import stroke2angle
#                 strokes = p["strokes"]
#                 angles = stroke2angle(strokes)
#                 probs = [_getprob(A) for A in angles]
#                 s = np.sum(probs) # for now take sum over all strokes
#                 return s      
#             NORM_VER = "divide" # since this is already in units of probabilti.
#         else:
#             assert False, "not coded"
#     else:
#         # Then this needs to be the function. I won't check but assume so.
#         priorFunction = ver

#     return priorFunction, NORM_VER


def makePriorFunction(ver="uniform"):
    """ 
    Simplified version, mainly for testing
    """

    # pre-coded prior models
    if ver=="distance_travel":
        from pythonlib.tools.stroketools import computeDistTraveled
        def priorFunction(p, trial):
            strokes = p["strokes"]
            orig = trial["task"]["fixpos"]
            cumdist = computeDistTraveled(strokes, orig, include_lift_periods=True)
            s = -cumdist # better if distance traveled is shorter
            return s
    else:
        print(ver)
        assert False, "not coded"

    return priorFunction



############## USING FEATURE EXTRACTOR
def prior_base():
    """ [GOOD] has norm, but no scoring fucntion
    This outputs log probs normalized priors.
    """
    
    from pythonlib.behmodel.scorer.scorer import Scorer
    from pythonlib.behmodel.scorer.utils import normscore
    from pythonlib.behmodel.feature_extractor.feature_extractor import FeatureExtractor
    from pythonlib.drawmodel.efficiencycost import Cost


    def norm(list_scores, params):
        beta = params[0]
        # beta = 1
        log_probs = normscore(list_scores, "log_softmax", [beta, False])
        return log_probs

    # Scorer
    Pr = Scorer()
    Pr.input_norm_function(norm)
    Pr.Objects = {}
    Pr._do_norm_with_params=True

    return Pr


def prior_feature_extractor(hack_lines5=True, 
    parser_names = None, rule=None):
    """ combines FeatureExtractor and MotorCost to do:
    - features, scoring, normalizing to probs
    OUT:
    - Scorer object
    """
    from pythonlib.behmodel.scorer.scorer import Scorer
    from pythonlib.behmodel.scorer.utils import normscore
    from pythonlib.behmodel.feature_extractor.feature_extractor import FeatureExtractor
    from pythonlib.drawmodel.efficiencycost import Cost

    if parser_names is None:
        parser_names = ["parser_graphmod", "parser_nographmod"]

    assert False, "use prior_function_database"
    def priorscorer(D, indtrial):
        # 1) Extract features
        F = FeatureExtractor()
        list_featurevec = F.list_featurevec_from_mult_parser(D, indtrial, 
            parser_names=parser_names, hack_lines5=hack_lines5)

        # Convert a list of featurevectors to a list of scores
        a = lambda: np.random.rand()
        params = {}
        if rule=="bent":
            params["thetas"] = {
                "circ":10.,
                "dist":0.,
                "circ_max":10.,
                "nstrokes":0.,
            }
        elif rule=="straight":
            params["thetas"] = {
                "circ":-10.,
                "dist":0.,
                "circ_max":-10.,
                "nstrokes":0.,
            }
        else:
            assert False


        # 2) Dot product of features with thetas
        MC = Cost(params)
        list_scores = np.array([MC.score_features(feature_vec) for feature_vec in list_featurevec])
        
        return list_scores

    # Normalizer
    def norm(list_scores):
        beta = 1
        params = [beta, False]
        probs = normscore(list_scores, "softmax", params)
        return probs

    # Scorer
    Pr = Scorer()
    Pr.input_score_function(priorscorer)
    Pr.input_norm_function(norm)

    return Pr


def prior_scorer_quick(ver, input_parsesflat=False):
    """ Good quick, better than above (here latest) since doesnt pass in Parsers, but 
    passes in list_of_p, so dont bneed to specify here how to parse.
    """

    from pythonlib.behmodel.scorer.scorer import Scorer
    from pythonlib.behmodel.scorer.utils import normscore
    from pythonlib.behmodel.feature_extractor.feature_extractor import FeatureExtractor
    from pythonlib.drawmodel.efficiencycost import Cost

    assert False, "use prior_function_database"

    if ver=="lines5":

        # 1) Extract features
        # One feature extract for all models, since they are working on the same parses
        F = FeatureExtractor()
        
        # 2) Scorers, these will differ based on model.
        dict_MC = {}
        params1 = {}
        params1["thetas"] = {
            "circ":10.,
            "dist":0.,
            "circ_max":10.,
            "nstrokes":0.,
        }
        dict_MC["bent"] = Cost(params1)

        params1 = {}
        params1["thetas"] = {
            "circ":-10.,
            "dist":0.,
            "circ_max":-10.,
            "nstrokes":0.,
        }
        dict_MC["straight"] = Cost(params1)


        if input_parsesflat:
            def priorscorer(list_parses, trialcode, rule):
                MC = dict_MC[rule]
                # NOTE: computing feature vectors is slow part.
                mat_features = F.list_featurevec_from_flatparses_directly(list_parses, trialcode, 
                    hack_lines5=True)
                list_scores = np.array([MC.score_features(feature_vec) for feature_vec in mat_features])
                return list_scores
        else:
            def priorscorer(D, indtrial, rule):
                MC = dict_MC[rule]
                # NOTE: computing feature vectors is slow part.
                if True:
                    # input dataset
                    mat_features = F.list_featurevec_from_flatparses(D, indtrial, 
                        hack_lines5=True)
                else:
                    mat_features = [np.zeros(4) for _ in range(len(D.Dat.iloc[indtrial]["parses_behmod"]))]
                # for feature_vec in mat_features:
                #     print(feature_vec.shape)
                #     assert False
                if True:
                    list_scores = np.array([MC.score_features(feature_vec) for feature_vec in mat_features])
                else:
                    list_scores = np.zeros(mat_features.shape[0])
                return list_scores

        # Normalizer
        def norm(list_scores):
            beta = 1
            params = [beta, False]
            probs = normscore(list_scores, "softmax", params)
            return probs

        # Scorer
        Pr = Scorer()
        Pr.input_score_function(priorscorer)
        Pr.input_norm_function(norm)
    else:
        assert False, "not coded"

    return Pr


def prior_scorer_quick_with_params(params0):
    """ Good quick, better than above (here latest) since doesnt pass in Parsers, but 
    passes in list_of_p, so dont bneed to specify here how to parse.
    """
    assert False, "use prior_function_database"
    
    # 1) Extract features
    # One feature extract for all models, since they are working on the same parses
    F = FeatureExtractor()
    MC = Cost(params0)
    Pr = prior_base()

    def priorscorer(list_parses, trialcode):
        """ params = tuple of scalars
        """
        # NOTE: computing feature vectors is slow part.
        mat_features = F.list_featurevec_from_flatparses_directly(list_parses, trialcode, 
            hack_lines5=True)
        list_scores = np.array([MC.score_features(feature_vec) for feature_vec in mat_features])
        return list_scores

    Pr.input_score_function(priorscorer)
    Pr.Objects = {
        "FeatureExtractor":F,
        "MotorCost":MC
    }
    Pr._do_score_with_params=False

    return Pr


### DATABASE OF PRIOR FUNCTIONS
def prior_function_database(ver, params=None):
    """ general purpose, holder of prior fucntions which can pass into scorere
    """

    from pythonlib.behmodel.scorer.scorer import Scorer
    from pythonlib.behmodel.scorer.utils import normscore
    from pythonlib.behmodel.feature_extractor.feature_extractor import FeatureExtractor
    from pythonlib.drawmodel.efficiencycost import Cost

    if ver=="monkey_vs_monkey":
        """ uniform distribution over all monkey (other epcoh) parses
        """
        def func(D, ind, modelname):
            """ 
            modelname, name of epoch from which parses taken
            """
            parsesname = f"strokes_beh_group_{modelname}"
            nparses = len(D.Dat.iloc[ind][parsesname])
            return np.ones(nparses)    

        Pr = prior_base()
        Pr.input_score_function(func)

    elif ver=="lines5":
        params0 = {}
        params0["thetas"] = {
                "circ":0.,
                "dist":0.,
                "circ_max":0.,
                "nstrokes":0.}
        F = FeatureExtractor()
        MC = Cost(params0)
        Pr = prior_base()

        def priorscorer(list_parses, trialcode):
            """ params = tuple of scalars
            """
            # NOTE: computing feature vectors is slow part.
            mat_features = F.list_featurevec_from_flatparses_directly(list_parses, trialcode, 
                hack_lines5=True)
            list_scores = np.array([MC.score_features(feature_vec) for feature_vec in mat_features])
            return list_scores

        Pr.input_score_function(priorscorer)
        Pr.Objects = {
            "FeatureExtractor":F,
            "MotorCost":MC
        }
        Pr._do_score_with_params=False
    elif ver=="random":
        # assign same prob to each parse

        def priorscorer(list_parses, trialcode):
            """ params = tuple of scalars
            """
            return np.ones((len(list_parses)))

        Pr = prior_base()
        Pr.input_score_function(priorscorer)
        Pr.Objects = {}
        Pr._do_score_with_params=False

    elif ver=="default_features":
        # feature extractor and motor cost
        F = FeatureExtractor(params["params_fe"])
        MC = Cost(params["params_mc"])
        Pr = prior_base()

        def priorscorer(list_parses, trialcode):
            """ params = tuple of scalars
            """
            # NOTE: computing feature vectors is slow part.
            mat_features = F.list_featurevec_from_flatparses_directly(list_parses, trialcode, 
                hack_lines5=False)
            mat_features = MC.transform_features(mat_features) 
            list_scores = MC.score_features_mat(mat_features)
            # list_scores = np.array([MC.score_features(feature_vec) for feature_vec in mat_features])
            return list_scores

        Pr.input_score_function(priorscorer)
        Pr.Objects = {
            "FeatureExtractor":F,
            "MotorCost":MC
        }
        Pr._do_score_with_params=False

    elif ver=="chunks":
        # (hacky) assuming that have extracted the best-fit already, whihc is a single
        # item in P.Parses[ind], and indicated within the P.Parses[ind] dict. Not fully tested
        # as there were bugs in the extraction of best-fit.
        # that SINGLE parse is all prior peaked on
        # TODO: new method, where the best-fit parses and determined not in preceding step, but instead
        # here as part of behmodel.

        def func(D, ind, modelname):
            """ 
            modelname, name of "rule" for which this is the best parse
            """
            # GRAPHMOD = "parser_graphmod"
            # exrtract the parser
            P = D.parser_get_parser_helper(ind) # assumes only one parser per trial.

            # find the parse that maximizes the fit between this beh trial 
            trial_tuple = D.trial_tuple(ind)

            if False:
                # This is v2 (saved in P.Parses, link to whether it is best-fit to something in P.ParseBase)
                # Don't use this, too convoluted
                inds = P.findparses_bycommand("best_fit_helper", 
                    {"rule":modelname, "trial_tuple":trial_tuple}
                    )
            else:
                # New version, where saves directly in P.ParseBase which is its best parse
                # inds1 = P.findparses_bycommand("best_fit_helper", 
                #     {"rule":modelname, "trial_tuple":trial_tuple}
                #     )
                inds = [P["best_fit_perms"][trial_tuple]["index"] for P in P.ParsesBase if P["rule"]==modelname]
            # get each parse as strokes, get their likelis
            # list_parses_as_strokes = [P.extract_parses_wrapper(i, "strokes") for i in inds]

            # strokes for this beh trial
            # strokes_beh = D.Dat.iloc[ind]["strokes_beh"]
            # print(list_parses_as_strokes[0])
            # D.plotMultStrokes([strokes_beh] + list_parses_as_strokes)
            # assert False, "check units"

            # return delta function over these parses
            nparses = len(P.Parses)
            prior = np.zeros(nparses)
            prior[inds] = 1
            return prior

        Pr = prior_base()
        Pr.input_score_function(func)

    else:
        print(ver)
        assert False
    return Pr
