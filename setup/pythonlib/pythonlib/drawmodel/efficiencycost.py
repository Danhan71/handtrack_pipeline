""" model that takes in beahviora nd outputs scalar representing efficiencyc ost.
Is related to "Planner" model in drawgood, but here making more general, e.g., can take
in strokes objects (list of np arrays) and generic "task" object as pairs, instead of
needing the complex "datseg" object I used in drawgood.
- Make this as simple as possioble. Model fitting, etc, will work using the taskmodel
object, which will have this efficiencycost as a sub object
- Note: I made this by coping over the Planner code, then pruning to based elements.
"""

# from preprocess import loadPreprocessedData, getFlatData
# import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
# from itertools import permutations
from scipy.special import logsumexp
# import copy
# from math import factorial
# from operator import itemgetter

# from pythonlib.tools.listtools import permuteRand
# from pythonlib.tools.dicttools import printOverviewKeyValues, filterSummary
# from pythonlib.tools.snstools import addLabel

DEBUG = False # setting up Jax version

class Cost:
    def __init__(self, params=None):
        """ params is dict. flexible, and whatever keys are active will dictate what
        features are considered
        """
        if params is None:
            params = self.getDefaultParams()
        
        self._initialize(params)
        self._updateThetas()

    def getDefaultParams(self):
        """
        in all cases, params are tuple, where
        params[0] is the model params, and 
        params[1,2 ,...] are flexible params useful 
        for computing that feature.
        """

        a = lambda: np.random.rand()
        params = {
            "screenHV":[1024, 768]
        }
        params["thetas"] = {
            "strokedir":(a(), 0),
            "numstrokes":a(),
            "startloc":(a(), 0., 0.),
            "endloc":(a(),  0., 0.),
            "jumpdir":(a(), 0.)
        }
        params["transformation"] = {}
        return params


    def _initialize(self, params):
        """ initialize model based on keys in params
        """

        if isinstance(params, tuple):
            # Then you passed in thetavec directly, without names
            thetavec = params
            params = self.getDefaultParams()
            assert False, "not completed need to figure out how deal with tuple"



        if "screenHV" not in params.keys():
            _params = self.getDefaultParams()
            params["screenHV"] = _params["screenHV"]
        if "thetas" not in params.keys():
            _params = self.getDefaultParams()
            params["thetas"] = _params["thetas"]
        if "transformations" not in params.keys():
            params["transformations"] = _params["transformations"]

        # 2) what is scale of screen? for normalizing units
        # (goal is diagoal is ~2)
        if "screenHV" in params:
            params["screenDiagPixHalf"] = \
            0.5*np.linalg.norm(params["screenHV"])

        self.Params = params

    def updateThetaVec(self, thetavec):
        """
        """
        self.Params["thetavec"] = np.array(thetavec)

    def params_ravel(self):
        """ return a fl;attened array of params
        use params_unravel to place back in
        """
        th = []

        # 1) thetas
        th.append(self.Params["thetavec"])
        # 2) transformations
        for k, v in self.Params["transformations"].items():
            th.append(v)
        out = np.concatenate(th)
        assert len(out.shape)==1, "tform params are mats?"
        return out


    # def params_unravel(self, th, return_leftover=False):
    #     """ place back flattened params into self.Params.
    #     th must be size (N,), and must fit perfectly inot 
    #     the params. will fail if not.
    #     """
    #     import jax.numpy as np

    #     assert len(th.shape)==1
    #     assert not isinstance(th, list) or isinstance(th, tuple), "should use np or jax"
        
    #     # 1) thetavec
    #     inds = np.arange(len(self.Params["thetavec"]))
        
    #     ct = 0
    #     print("input len:", th)
    #     print("return return_leftover", return_leftover)
    #     inds2 = np.arange(ct, ct+len(self.Params["thetavec"]))
    #     print(ct, inds, inds2)
    #     ct+=len(inds2)

    #     self.Params["thetavec"] = th[inds]

    #     th = np.delete(th, inds)

    #     # 2) Transformations
    #     for k, v in self.Params["transformations"].items():
    #         inds = np.arange(len(v))

    #         inds2 = np.arange(ct, ct+len(v))
    #         print(ct, inds, inds2)
    #         ct+=len(inds2)

    #         self.Params["transformations"][k] = th[inds]
    #         th = np.delete(th, inds)
            
    #     print("leftover", th)
    #     print("leftover inds", np.arange(ct, len(th)))
    #     assert False
    #     if return_leftover:
    #         return th
    #     else:
    #         assert len(th)==0, "inputed too many params"


    def params_unravel(self, th, return_leftover=False):
        """ place back flattened params into self.Params.
        th must be size (N,), and must fit perfectly inot 
        the params. will fail if not.
        # NOTE: new version, where doesnt use delete..
        """
        if DEBUG:
            import jax.numpy as np
        else:
            import numpy as np

        assert len(th.shape)==1
        assert not isinstance(th, list) or isinstance(th, tuple), "should use np or jax"
        
        # 1) thetavec
        ct = 0
        inds = np.arange(ct, ct+len(self.Params["thetavec"]))
        ct+=len(inds)
        try:
            self.Params["thetavec"] = th[inds]
        except Exception as err:
            print(th)
            print(inds)
            print(th.shape)
            print(inds.shape)
            raise err

        # 2) Transformations
        for k, v in self.Params["transformations"].items():

            inds = np.arange(ct, ct+len(v))
            ct+=len(inds)

            self.Params["transformations"][k] = th[inds]
            
        # print("leftover", th)
        inds = np.arange(ct, len(th))
        # print("leftover inds", np.arange(ct, len(th)))
        # assert False
        if return_leftover:
            return th[inds]
        else:
            assert len(th)==0, "inputed too many params"



    def _updateThetas(self):
        """ takes params["thetas"], which is dict, with
        paramname: (params), and updates self.Params.
        Leaves untouched things in self.Params not related to 
        thetas.
        - usually want to keep params["thetas"] keys uinchanges, but
        subsequent code should work fine even if dont.
        """

        # self.Params["thetas"] = params["thetas"]

        # 1) make sure all params are in tuple or list format.
        for k, p in self.Params["thetas"].items():
            if not isinstance(p, tuple) and not isinstance(p, list):
                # then is scalar.
                self.Params["thetas"][k] = tuple([p])

        # 3) save feature param vectors, easy to take dot products with features
        self.Params["thetavec"] = []
        self.Params["thetanames"] = []
        for k, v in self.Params["thetas"].items():
            self.Params["thetavec"].append(v[0])
            self.Params["thetanames"].append(k)

        self.Params["thetavec"] = np.array(self.Params["thetavec"])

    def score_features_mat(self, feature_mat):
        """
        - feature_mat, shape (N x k), 
        - thetavec, shape (k,)

        OUT:
        - scores, (N,)

        """
        thetavec = self.Params["thetavec"]
        assert thetavec.shape[0]==feature_mat.shape[1]
        if DEBUG:
            import jax.numpy as np
        else:
            import numpy as np
        score_vec = np.dot(feature_mat, thetavec)
        return score_vec

    def score_features(self, feature_vec):
        """ given feature vec, return score. So this doesnt work directly with
        the motor behavior. assumes your've already extracted features 
        INPUT:
        - feature_vec, type in {np.array, list, dict}
        --- if list, then must be in order of self.Params["thetanames"]
        --- if dict, then the names of features must be subsets of params. 
        format is feature_vec = {paramname: value}
        """

        if isinstance(feature_vec, list):
            assert False, "not coded"

        if isinstance(feature_vec, dict):
            thetas = self.Params["thetas"]
            # Get weighted sum of feature vec.
            score = 0.
            for k, v in feature_vec.items():
                score += thetas[k]*feature_vec[k]
        else:
            thetavec = self.Params["thetavec"]
            assert thetavec.shape==feature_vec.shape
            if DEBUG:
                import jax.numpy as np
            else:
                import numpy as np
            score = np.dot(feature_vec, thetavec)

        return score

    def transform_features(self, feature_mat):
        """ 
        applies any transfomaritons instructed in params[transformations] for the particulare
        IN:
        - feature_mat, N x nfeats
        OUT:
        - feature_mat (also modifies in place, i think.)
        """

        def _index(tform):
            return self.Params["thetanames"].index(tform)

        for tform, prms in self.Params["transformations"].items():
            if tform=="angle_travel":
                from pythonlib.tools.vectools import angle_diff_vectorized
                # find the index for this
                ind = _index(tform)
                angle0 = prms[0]
                if not DEBUG:
                    feature_mat[:, ind] = -angle_diff_vectorized(feature_mat[:,ind], angle0)
            elif tform=="nstrokes":
                # absolute value of difference from a mean num strokes
                # take negative, 
                ind = _index(tform)
                nstrokes0 = prms[0]
                if not DEBUG:
                    nstrokes0 = np.max([0, nstrokes0])
                    # feature_mat[:, ind] = -np.abs(feature_mat[:, ind] - nstrokes0)
                    # feature_mat[:, ind] = -(feature_mat[:, ind] - nstrokes0)**2
                    feature_mat[:, ind] = 1/(1+(feature_mat[:, ind] - nstrokes0)**2) # so that stays positive.
            elif tform=="dist_travel":
                # take inverse, so lower numbers are worse.
                ind = _index(tform)
                if not DEBUG:
                    feature_mat[:, ind] = 1/feature_mat[:, ind]

            else:
                assert False, "not coded"
        return feature_mat



    ########################### BATCH OPERATIONS (multiple trials)
    # --- PROBABILITIES
    def scoreSoftmax(self, value, valueall, debug=False):
        # returns softmax probability for cost
        # valueall would be the distribution of options (e.g, all permuations)
        """get log probs for this sequence (value) rleative to its permutations (valueall).
        confirmed that np.exp(valuenorm) is prob"""
        # wraps score and scoreSoftmax
        # takes unnormalized score (cost) and logsumexp of all sequences (costall), or a subset (Nperm)
        # and outputs the noramlzied score (costnorm)
        if debug:
            print("value")
            print(value)
            print("value all:")
            print(valueall[:10])

            print("these are log of softmax probabiltiies. should be identical")
            print(value - logsumexp(valueall))
            print(np.log(np.exp(value)/sum(np.exp(valueall))))
            print("this is probabilit:")
            print(np.exp(value)/sum(np.exp(valueall)))
            print(np.exp(value - logsumexp(valueall)))
        logprob = value - logsumexp(valueall)
        return value - logsumexp(valueall)




    ########################### DEALS WITH FEATURES
    def score(self, strokes, task=None, return_feature_vecs = False):
        """ given behavior (strokes) and task (task), 
        return a scalar score 
        - Note on conventions: 
        - normalization in features is goal to 
        put params in range (-1,1), or generally range 
        of 2 units (or order of that magnitude)
        - smaller values are "better" meaning less costly.
        (although this is arbitrary, since params could be neg.)
        """


        if False:
            # 1) Feature vectors for each stroke
            fstrokes = []
            for s in strokes:
                fstrokes.append(self.featuresSingleStrok(s))

            # 2) Feature vector considering entire task
            ftask = self.featuresWholeTask(strokes)

            # -- make sure no Nones, which indicates feature
            # extraction failed
            for f in fstrokes:
                assert f is not None
            for f in ftask:
                assert f is not None

            # == Combine scores into a feature vector across strokes
            # (average faet vec over strokes)
            featvec = np.r_[np.array(fstrokes).mean(0), ftask]
        else:
            featvec = self.features(strokes)
            for f in featvec:
                assert f is not None

        # score, dot using combined feat vec
        self.score_features(featvec)
        # score = np.dot(featvec, self.Params["thetavec"])

        if return_feature_vecs:
            return featvec, score
        else:
            return score

    def features(self, strokes):
        from ..tools.vectools import get_angle, angle_diff
        from math import pi
        from .features import stroke2angle

        def _features(pname):
            pval = self.Params["thetas"][pname]
            if pname == "numstrokes":
                # number of strokes (0, 1, 2, ...)
                # not normalized
                nstrokes = len(strokes)
                return nstrokes
            elif pname == "startloc":
                """ position of first touch on first
                stroke.
                pval[1,2] = (x,y) location.
                returns euclidian distance from this 
                location
                """
                loc0 = np.array((pval[1], pval[2]))
                locbeh = strokes[0][0, [0,1]]
                d = np.linalg.norm(locbeh - loc0)

                # normalize ()
                return d/self.Params["screenDiagPixHalf"]
            elif pname == "endloc":
                """ position of last touch on last
                stroke.
                pval[1,2] = (x,y) location.
                returns euclidian distance from this 
                location
                """
                loc0 = np.array((pval[1], pval[2]))
                locbeh = strokes[-1][-1, [0,1]]
                d = np.linalg.norm(locbeh - loc0)

                # normalize ()
                return d/self.Params["screenDiagPixHalf"]
            elif pname == "jumpdir":
                """ overall direction of s-s transitions
                pval[1] is "ground truth" angle.
                see featuresSingleStroke to see how angle differences
                and normalization occur.
                """

                # get angle of transition between endpoint of one stroke
                # to onset of next stroke.
                jumpangles = []
                for s1, s2 in zip(strokes[:-1], strokes[1:]):
                    jump = s2[0,[0,1]] - s1[-1, [0,1]]
                    jumpangles.append(get_angle(jump))

                # for each jump angle, get its distance fromt he desired angle
                angle0 = pval[1]
                anglediffs = [angle_diff(angle0, anglej) for anglej in jumpangles]

                # normalize each, and take avearge
                return np.mean([a/pi for a in anglediffs])

            elif pname == "totaldist":
                from pythonlib.tools.stroketools import computeDistTraveled
                assert False, "not finished coding"
                orig = trial["task"]["fixpos"]
                cumdist = computeDistTraveled(strokes, orig, include_lift_periods=True)
                s = -cumdist # better if distance traveled is shorter
                return s

            elif pname == "strokedir":
                # angle of strok, in radians, distance from
                # pval[1]. angle 0 is to right, and CCW 
                # increases.
                
                vals = []
                angle0 = pval[1]
                anglelist = stroke2angle(strokes)
                for anglebeh in anglelist:
                    a = angle_diff(angle0, anglebeh)

                    # - normalize to [0,1]
                    vals.append(a/pi)
                return np.mean(vals)
            else:
                assert False, "not coded"

        fvec = []
        for pname in self.Params["thetanames"]:
            fvec.append(_features(pname))
        return fvec


    # def featuresWholeTask(self, strokes):
    #     """ feature vector for entire task.
    #     """
    #     from ..tools.vectools import get_angle, angle_diff
    #     from math import pi


    #     def _features(pname, pval):
    #         if pname == "numstrokes":
    #             # number of strokes (0, 1, 2, ...)
    #             # not normalized
    #             nstrokes = len(strokes)
    #             return nstrokes
    #         elif pname == "startloc":
    #             """ position of first touch on first
    #             stroke.
    #             pval[1,2] = (x,y) location.
    #             returns euclidian distance from this 
    #             location
    #             """
    #             loc0 = np.array((pval[1], pval[2]))
    #             locbeh = strokes[0][0, [0,1]]
    #             d = np.linalg.norm(locbeh - loc0)

    #             # normalize ()
    #             return d/self.Params["screenDiagPixHalf"]
    #         elif pname == "endloc":
    #             """ position of last touch on last
    #             stroke.
    #             pval[1,2] = (x,y) location.
    #             returns euclidian distance from this 
    #             location
    #             """
    #             loc0 = np.array((pval[1], pval[2]))
    #             locbeh = strokes[-1][-1, [0,1]]
    #             d = np.linalg.norm(locbeh - loc0)

    #             # normalize ()
    #             return d/self.Params["screenDiagPixHalf"]
    #         elif pname == "dir":
    #             """ overall direction of s-s transitions
    #             pval[1] is "ground truth" angle.
    #             see featuresSingleStroke to see how angle differences
    #             and normalization occur.
    #             """

    #             # get angle of transition between endpoint of one stroke
    #             # to onset of next stroke.
    #             jumpangles = []
    #             for s1, s2 in zip(strokes[:-1], strokes[1:]):
    #                 jump = s2[0,[0,1]] - s1[-1, [0,1]]
    #                 jumpangles.append(get_angle(jump))

    #             # for each jump angle, get its distance fromt he desired angle
    #             angle0 = pval[1]
    #             anglediffs = [angle_diff(angle0, anglej) for anglej in jumpangles]

    #             # normalize each, and take avearge
    #             return np.mean([a/pi for a in anglediffs])

    #         elif pname == "totaldist":
    #             from pythonlib.tools.stroketools import computeDistTraveled
    #             assert False, "not finished coding"
    #             orig = trial["task"]["fixpos"]
    #             cumdist = computeDistTraveled(strokes, orig, include_lift_periods=True)
    #             s = -cumdist # better if distance traveled is shorter
    #             return s

    #         else:
    #             assert False, "not coded"

    #     fvec = []
    #     for pname, pval in self.Params["task"].items():
    #         fvec.append(_features(pname, pval))
    #     return fvec

    # def featuresSingleStrok(self, strok):
    #     """ get feature vector for a single strok
    #     """
    #     from .features import stroke2angle
    #     from ..tools.vectools import angle_diff
    #     from math import pi

    #     def _features(pname, pval):
    #         if pname == "dir":
    #             # angle of strok, in radians, distance from
    #             # pval[1]. angle 0 is to right, and CCW 
    #             # increases.
                
    #             # get distance between angles
    #             angle0 = pval[1]
    #             anglebeh = stroke2angle([strok])[0]
    #             a = angle_diff(angle0, anglebeh)

    #             # - normalize to [0,1]
    #             return a/pi
    #         else:
    #             assert False, "not coded"

    #     fvec = []
    #     for pname, pval in self.Params["strok"].items():
    #         fvec.append(_features(pname, pval))
    #     return fvec

    def oldFeatureExtractors():
        assert False, "this shoudlnt be here. figure out which should go to .vectools, and which throw out"
        def _convert_to_rel_units(x,y):
            # converts from x,y to relative units from top-left corner
            # top left = (0,0)
            # bottom right = (1,1)
#             print("---")
#             print((x,y))
#             print((xmin, xmax))
#             print((ymin, ymax))
#             print("---")
            x = (x - xmin)/(xmax-xmin)
            y = (ymax - y)/(ymax-ymin)
            
            return (x,y)

        def get_lefttop_extreme(t):
            """get x and y extreme that is most top left
            """
            x = min(t["x_extremes"])
            y = max(t["y_extremes"])
            x,y = _convert_to_rel_units(x,y)
            return x,y

        def get_center_pos(t):
            x,y = _convert_to_rel_units(t["centerpos"][0], t["centerpos"][1])
            return x,y

        
        def get_vector_between_points(A,B):
            # vector from A --> B
            x = B[0] - A[0]
            y = B[1] - A[1]
            return (x,y)
            
        def get_vector_norm(x,y):
            # norm of vector (x,y), scaled so that 1 is maximum (corner to corner)
            dist = ((x**2 + y**2)**0.5)/(2**0.5) #
            return dist

        def get_dot_between_unit_vectors(u, v):
            # ignores length, just get dot of unit vectors.
            u = np.array(u)
            v = np.array(v)
            c = np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v) 
            return c
            
        def get_projection_along_diagonal(x,y):
            # if x,y is vector with origin at top left, with positive values pointing towards bottom right, then
            # this gives projection along diagonal. 0 means is at top-left. 1 means is at bottom right.
            
            proj = (x*1+y*1)/(2**0.5) # get projection
            proj = proj/(2**0.5) # normalize
            return proj
        
        
        def score_start(t):
            """scores the starting token"""

            # get distance from top-left corner
            if True:
                x,y = get_lefttop_extreme(t)
            else:
                x,y = get_center_pos(t)
            # print("###")
            # print(x,y)
            # print(get_vector_norm(x,y))
            # print("###")

            cost=0
            if hasattr(self, "params_start"):
                cost+=3*self.params_start*get_vector_norm(x,y)

            return cost


        def score_trans(t1, t2):
            """t1-->t2. score this transition"""

            # === motor costs
            def motor_distance(t1, t2):
                # distance between centers
                c1 = get_center_pos(t1)
                c2 = get_center_pos(t2)
                x,y = get_vector_between_points(c1, c2)
                
                cost = get_vector_norm(x,y)
                return cost
            
            def motor_direction(t1,t2):
                # direction between strokes
                # weak bias to go left to right
                c1 = get_center_pos(t1)
                c2 = get_center_pos(t2)                
                x,y = get_vector_between_points(c1, c2)
                
                cost = -get_projection_along_diagonal(x,y)
                return cost
            
            def motor_inertia(t1, t2):
                # v is vector 
                global previous_vector
#                 print("previous vector {}".format(previous_vector))
                c1 = get_center_pos(t1)
                c2 = get_center_pos(t2)    
                x,y = get_vector_between_points(c1, c2)
#                 print("current vector {}".format((x,y)))

                if len(previous_vector)==0:
                    cost = 0
                else:
                    # - get angle between current vector and old vector
                    cost = -get_dot_between_unit_vectors(previous_vector, (x,y))
                
                previous_vector = (x,y)
                    
                return cost
                
             

            cost = 0
            if hasattr(self, "params_motor_dist"):
                cost+=3*self.params_motor_dist * motor_distance(t1, t2)    
            if hasattr(self, "params_motor_dir"):
                cost+=2*self.params_motor_dir * motor_direction(t1, t2)
            if hasattr(self, "params_motor_inertia"):
                cost+=self.params_motor_inertia * motor_inertia(t1, t2)
            
            return cost
    
        # cognitive costs
        def score_cognitive(t1, t2):
            
            def cog_primtype(t1,t2):
                # cost for switching primitive type
                def get_prim_type(t):
                    if "L" in t["codes_unique"]:
                        return "line"
                    elif "C" in t["codes_unique"]:
                        return "circle"
                if get_prim_type(t1) == get_prim_type(t2):
                    cost = 0
                else:
                    cost = 1
                return cost
            
            def cog_vertchunker(t1, t2):
                # cost for moving horizontally.
                c1 = get_center_pos(t1)
                c2 = get_center_pos(t2)
                x,_ = get_vector_between_points(c1, c2)
                
                cost = get_vector_norm(x,0)
                return cost
                
            def cog_horizchunker(t1, t2):
                # cost for moving vertically.
                c1 = get_center_pos(t1)
                c2 = get_center_pos(t2)
                _,y = get_vector_between_points(c1, c2)
                
                cost = get_vector_norm(0, y)
                return cost

            
            

            cost=0
            if hasattr(self, "params_cog_primtype"):
                cost+=self.params_cog_primtype * cog_primtype(t1, t2)
            if hasattr(self, "params_cog_vertchunker"):
                # print("ASDAS")
                # -- decide whether to add to param a value indicating this transition start from LL
                if hasattr(self, "params_cog_vertchunker_LL") and "LL" in t1["codes_unique"]:
                    # print("ADASd")
                    paramthis = self.params_cog_vertchunker + self.params_cog_vertchunker_LL
                else:
                    paramthis = self.params_cog_vertchunker
                cost+=2*paramthis*cog_vertchunker(t1, t2)
            if hasattr(self, "params_cog_horizchunker"):
                cost+=2*self.params_cog_horizchunker * cog_horizchunker(t1, t2)
            
            return cost
        

    ## ====== HELPER FUNCTIONS
    def printPlotSummary(self, strokes, task):
        """ overview for this data,
        - prints featurevecs
        - prints score
        - plots task
        """
        featvec, score = self.score(strokes, task,
            return_feature_vecs=True)

        print("Feature names:")
        print(self.Params["thetanames"])

        print("Feature vector: ")
        print(featvec)

        print("Feature thetas:")
        print(self.Params["thetavec"])        

        print("Score")
        print(score)



# ==== OBJECTIVE FUNCTION
# def getObjFun(datsegs, Nperm, paramslist, regularize=None):
#     # call fun to use memoized paths to generate loss
#     planner = Planner(paramslist=paramslist)
#     planner.memoizePaths(datsegs, N=Nperm)
    
#     def fun(params):
#         planner.updateParams(params)
#         score = planner.scoreMultTasks(datsegs, usememoizedPerms=True, minimalprint=True)
#         if regularize=="l2":
#             lam=(1/25) # for a sigma=5?
#             score = score-lam*np.sum(params**2)
#         elif regularize is not None:
#             print("PROBLEM - dont knwo this regularizer")
#             assert False

#         cost = -score # since want to maximize score and minimize cost.
#         return cost
    
#     return fun, planner


# from scipy.optimize import minimize as minim
# def minimize(fun, numparams):
#     params0=np.random.uniform(-8, 8, size=numparams)
#     # params = tuple([1])
#     # fun(params)
#     # res = minimize(fun, (1,), method="L-BFGS-B")
#     # res = minimize(fun, (0.5,0.5,0.5,0.5,0.5,0.5,0.5), method="L-BFGS-B")
#     # params0 = np.random.uniform(-10, 10, size=6)
#     res = minim(fun, params0, method="L-BFGS-B")
#     # res = minimize(fun, (1,), method="Nelder-Mead")
#     return res


def rank_beh_out_of_all_possible_sequences_quick(strokes_beh, strokes_task_perms, beh_task_distances, 
    task_inefficiency, plot_rank_distribution=False, efficiency_score_ver="weighted_avg", 
    confidence_ver="diff_relative_all", plot_strokes=False, plot_n_strokes=4, **kwargs):
    """ 
    QUICK VERSION: must pass in precomputed beh_task_distances and task_inefficiency
    out of all possible way sot sequence the task (orders and directions)
    how close is beh to the most efficient sequence? 
    - efficiency_score_ver, how to summarize.
    RETURNS:
    - rank, confidence, summaryscore, always
    --- rank, {0, 1, ... num sequences} where 0 means the task sequence that beh 
    is most aligned to is also the one that is most efficent
    - adds num possible seq (return_num_possible_seq)
    - adds strokeslist (return_chosen_task_strokes)
    - None, if somehow fails, e.g. if scoring function expects asme length, but it is not.
    """
    # Get all permutations (and orders) for task
    from pythonlib.tools.stroketools import getStrokePermutationsWrapper
    from pythonlib.drawmodel.features import computeDistTraveled
    from pythonlib.drawmodel.strokedists import distscalarStrokes, scoreAgainstAllPermutations
    import scipy.stats as ss

    if np.any(np.isnan(beh_task_distances)):
        return None
    if np.any(np.isnan(task_inefficiency)):
        return None


    # (4) Find the rank (based on effiicencey) of the task sequence closest aligned to behavior.
    inefficiency_dissimilarity = [[d, b, s] for d, b, s in zip(task_inefficiency, beh_task_distances, strokes_task_perms)]
    inefficiency_dissimilarity = sorted(inefficiency_dissimilarity, key=lambda x:x[0]) # sort in order of incresae inefficiency
    inefficiency_ranked = [i[0] for i in inefficiency_dissimilarity]
    beh_task_distances_ranked = [i[1] for i in inefficiency_dissimilarity]
    strokes_task_ranked = [i[2] for i in inefficiency_dissimilarity]
    
    # rank_list = ss.rankdata(beh_task_distances_ranked, method="min")
    # rank = beh_task_distances_ranked.index(min(beh_task_distances_ranked)) # find the index of the task sequence that is most aligned to beh.

    from pythonlib.tools.listtools import rankinarray1_of_minofarray2
    rank, inefficiency_picked, beh_task_distance_picked, strokes_task_picked = \
        rankinarray1_of_minofarray2(inefficiency_ranked, beh_task_distances_ranked)[:4]

    # Compute confidence
    if confidence_ver:
        if confidence_ver =="diff_first_vs_second":
            tmp = sorted(beh_task_distances_ranked)
            confidence = np.abs(tmp[1] - tmp[0])
        elif confidence_ver =="diff_relative_all":
            # like diff_first_vs_second, but normalize to the diffs for all others.
            tmp = sorted(beh_task_distances_ranked)
            numer = np.abs(tmp[1] - tmp[0])
            denom = np.mean(np.abs(np.diff(tmp[1:])))
            confidence = numer -denom
        else:
            print(confidence_ver)
            assert False, "not coded"
    else:
        confidence = np.nan


    # Compute efficiency score
    if efficiency_score_ver=="weighted_avg":
        # summary efficnecy score (weighted sum of efficiencey, weighted by 1-distance)
        # <1 good, 1 =ranodm, >1 worse than random.
        # "proibability", based on beh_task_dist
        a = 1/np.array(beh_task_distances_ranked)
        a = a/np.sum(a) # must be prob, to get weighted sum

        # weighted sum of inefficiencey
        b = np.array(inefficiency_ranked)
        summaryscore = np.dot(a, b)

        # normalize to average inefficiencey
        summaryscore = summaryscore/np.mean(inefficiency_ranked) # noramlize to the mean over all efficiencies
    else:
        print(efficiency_score_ver)
        assert False, "not coded"

    if plot_strokes:

        # First rerank based on sim to beh
        # inefficiency_dissimilarity = [[d, b, s] for d, b, s in zip(task_inefficiency, beh_task_distances, strokes_task_perms)]
        inefficiency_dissimilarity_2 = sorted(inefficiency_dissimilarity, key=lambda x:x[1]) # sort in order of incresae inefficiency
        beh_task_distances_ranked_2 = [i[1] for i in inefficiency_dissimilarity_2]
        inefficiency_ranked_2 = [i[0] for i in inefficiency_dissimilarity_2]
        strokes_task_ranked_2 = [i[2] for i in inefficiency_dissimilarity_2]

        # Plot the picked strokes
        from pythonlib.drawmodel.strokePlots import plotDatStrokes

        fig, axes = plt.subplots(1, 2, figsize=(2*2, 2))

        ax = axes[0]
        plotDatStrokes(strokes_beh, ax=ax, clean_ordered=True)
        ax.set_title("behavior")

        ax = axes[1]
        plotDatStrokes(strokes_task_picked, ax=ax, clean_ordered=True)
        ax.set_title(f"task, bestmatch (rank:{rank})")

        if isinstance(plot_n_strokes, str):
            assert plot_n_strokes=="all"
            ntaskplot = len(strokes_task_perms)
        else:
            ntaskplot = plot_n_strokes

        ncols = 6
        nrows = int(np.ceil((ntaskplot+1)/ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2))

        # Beh
        ax = axes.flatten()[0]
        plotDatStrokes(strokes_beh, ax=ax, clean_ordered=True)
        ax.set_title("behavior")

        # all tasks
        for i in range(1, ntaskplot+1):
            ax = axes.flatten()[i]
            st = strokes_task_ranked_2[i-1]
            beh_task_dist = beh_task_distances_ranked_2[i-1]
            task_score = inefficiency_ranked_2[i-1]
            plotDatStrokes(st, ax=ax, clean_ordered=True)
            ax.set_title(f"b-t dist: {beh_task_dist:.2f}")
            ax.set_xlabel(f"t score: {task_score:.2f}")
            ax.set_ylabel(f"t score: {task_score:.2f}")



    if plot_rank_distribution:
        x = range(len(inefficiency_dissimilarity))

        fig, axes = plt.subplots(1,2, figsize=(10, 3))

        ax=axes.flatten()[0]
        ax.plot(x, inefficiency_ranked, "-ok", label="ineffiiency")
        ax1=ax.twinx()
        ax1.plot(x, beh_task_distances_ranked, "-xr", label="beh_task_distance")
        ax1.axhline(0)
        # plt.ylabel("inefficiency")
        plt.xlabel("sroted by inefficnecy")
        ax.legend()
        ax1.legend()

        # OTher way around, sort by beh task distance.
        tmp = sorted(inefficiency_dissimilarity, key=lambda x:x[1]) # 
        inefficiency_this = [i[0] for i in tmp]
        beh_task_distances_sorted = [i[1] for i in tmp]

        ax=axes.flatten()[1]
        ax.plot(x, inefficiency_this, "-ok", label="ineffiiency")
        ax1=ax.twinx()
        ax1.plot(x, beh_task_distances_sorted, "-xr", label="beh_task_distance")
        ax1.axhline(0)
        # plt.ylabel("inefficiency")
        plt.xlabel("sroted by beh task dist")
        ax.legend()
        ax1.legend()

        plt.title(f"sumscore {summaryscore:.3f}")


    if False:
        # plot
        plt.figure()
        x = [i[0] for i in inefficiency_dissimilarity]
        y = [i[1] for i in inefficiency_dissimilarity]
        plt.plot(x,y, "ok")
        plt.xlabel("inefficiency")
        plt.ylabel("beh_task_dist")

    return rank, confidence, summaryscore, strokes_task_picked, inefficiency_picked, beh_task_distance_picked


def rank_beh_out_of_all_possible_sequences(strokes_beh, strokes_task, return_num_possible_seq=False,
    return_chosen_task_strokes=False, plot_rank_distribution=False, efficiency_score_ver="weighted_avg", 
    confidence_ver="diff_first_vs_second", **kwargs):
    """ out of all possible way sot sequence the task (orders and directions)
    how close is beh to the most efficient sequence? 
    - efficiency_score_ver, how to summarize.
    RETURNS:
    - rank, confidence, summaryscore, always
    --- rank, {0, 1, ... num sequences} where 0 means the task sequence that beh 
    is most aligned to is also the one that is most efficent
    - adds num possible seq (return_num_possible_seq)
    - adds strokeslist (return_chosen_task_strokes)
    - None, if somehow fails, e.g. if scoring function expects asme length, but it is not.
    """
    # Get all permutations (and orders) for task
    from pythonlib.tools.stroketools import getStrokePermutationsWrapper
    from pythonlib.drawmodel.features import computeDistTraveled
    from pythonlib.drawmodel.strokedists import distscalarStrokes, scoreAgainstAllPermutations

    # (1) get all permutations of the task strokes and distances
    out = scoreAgainstAllPermutations(strokes_beh, strokes_task, 
        confidence_ver=confidence_ver, **kwargs)
    if out is None:
        return None
    else:
        beh_task_distances, strokes_task_perms, confidence = out


    # (2) Score each one based on distance traveled
    distances = [computeDistTraveled(strokes, include_origin_to_first_stroke=False, 
                        include_transition_to_done=False) for strokes in strokes_task_perms]
    if False:
        plt.figure()
        plt.hist(distances, 20)


    # # (3) Find the one that beh is best aligned to
    # beh_task_distances = [distscalarStrokes(strokes_beh, S, ver="dtw_split_segments", splitnum1=2) 
    #                       for S in strokes_task_perms]


    # (4) Find the rank (based on effiicencey) of the task sequence closest aligned to behavior.
    inefficiency_dissimilarity = [[d, b, s] for d, b, s in zip(distances, beh_task_distances, strokes_task_perms)]
    inefficiency_dissimilarity = sorted(inefficiency_dissimilarity, key=lambda x:x[0]) # sort in order of incresae inefficiency
    beh_task_distances_ranked = [i[1] for i in inefficiency_dissimilarity]
    inefficiency_ranked = [i[0] for i in inefficiency_dissimilarity]
    strokes_task_ranked = [i[2] for i in inefficiency_dissimilarity]
    rank = beh_task_distances_ranked.index(min(beh_task_distances_ranked)) # find the index of the task sequence that is most aligned to beh.
    

    # Compute efficiency score
    if efficiency_score_ver=="weighted_avg":
        # summary efficnecy score (weighted sum of efficiencey, weighted by 1-distance)
        # <1 good, 1 =ranodm, >1 worse than random.
        # "proibability", based on beh_task_dist
        a = 1/np.array(beh_task_distances_ranked)
        a = a/np.sum(a) # must be prob, to get weighted sum

        # weighted sum of inefficiencey
        b = np.array(inefficiency_ranked)
        summaryscore = np.dot(a, b)

        # normalize to average inefficiencey
        summaryscore = summaryscore/np.mean(inefficiency_ranked) # noramlize to the mean over all efficiencies
    else:
        print(efficiency_score_ver)
        assert False, "not coded"

    if plot_rank_distribution:
        x = range(len(inefficiency_dissimilarity))

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1,2, figsize=(10, 3))

        ax=axes.flatten()[0]
        ax.plot(x, inefficiency_ranked, "-ok", label="ineffiiency")
        ax1=ax.twinx()
        ax1.plot(x, beh_task_distances_ranked, "-xr", label="beh_task_distance")
        ax1.axhline(0)
        # plt.ylabel("inefficiency")
        plt.xlabel("sroted by inefficnecy")
        ax.legend()
        ax1.legend()

        # OTher way around, sort by beh task distance.
        tmp = sorted(inefficiency_dissimilarity, key=lambda x:x[1]) # 
        inefficiency_this = [i[0] for i in tmp]
        beh_task_distances_sorted = [i[1] for i in tmp]

        ax=axes.flatten()[1]
        ax.plot(x, inefficiency_this, "-ok", label="ineffiiency")
        ax1=ax.twinx()
        ax1.plot(x, beh_task_distances_sorted, "-xr", label="beh_task_distance")
        ax1.axhline(0)
        # plt.ylabel("inefficiency")
        plt.xlabel("sroted by beh task dist")
        ax.legend()
        ax1.legend()

        plt.title(f"sumscore {summaryscore:.3f}")


    if False:
        # plot
        plt.figure()
        x = [i[0] for i in inefficiency_dissimilarity]
        y = [i[1] for i in inefficiency_dissimilarity]
        plt.plot(x,y, "ok")
        plt.xlabel("inefficiency")
        plt.ylabel("beh_task_dist")



    if return_num_possible_seq and return_chosen_task_strokes:
        return rank, confidence, summaryscore, len(strokes_task_perms), strokes_task_ranked[rank]
    elif return_num_possible_seq:
        return rank, confidence, summaryscore, len(strokes_task_perms)
    elif return_chosen_task_strokes:
        return rank, confidence, summaryscore, strokes_task_ranked[rank]        
    else:
        return rank, confidence, summaryscore