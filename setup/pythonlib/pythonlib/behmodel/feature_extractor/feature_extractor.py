""" one job - take in a single datapoint (e..g, parse) applies stored parameters to 
score this datapoint 
single parse --> single feature vector (i.e., not by timepoint or strokes)

"""
import numpy as np

from pythonlib.drawmodel.features import strokeCircularity, strokeDistances

class FeatureExtractor(object):
    def __init__(self, params, STROKE_DIR_DOESNT_MATTER=True):
        """ params is dict. flexible, and whatever keys are active will dictate what
        features are considered
        - Main focus is working with parses in Parser format.
        - Does not score or save any scoring-related params.
        NOTE:
        - STROKE_DIR_DOESNT_MATTER, mainly placeholder, in future maybe modify. this just 
        now used for hashing the edges.
        """

        self.Params = {}
        self.Params["STROKE_DIR_DOESNT_MATTER"] = STROKE_DIR_DOESNT_MATTER
        self.ListFeatures = params["features"]

        self._Tracker = {}

    # def list_featurevec_from_list_p(self, list_of_p, hack_lines5 = False):
    #     """ 
    #     IN:
    #     - list_of_p is list of parses for a single trial.
    #     """

    #     """ 
    #     Given single parser (one trial/task, multipel parses, one skeleton graph)
    #     returns a list of featuresvecs.
    #     IN:
    #     - P, parser object, holds list of parses
    #     OUT:
    #     - list of featurevecs, one for each parse in P
    #     """
    #     from pythonlib.drawmodel.features import strokeCircularity, strokeDistances

    #     tracker = {}
    #     STROKE_DIR_DOESNT_MATTER = self.Params["STROKE_DIR_DOESNT_MATTER"]

    #     def _path_to_key(walker):
    #         return walker.unique_path_id(invariant=STROKE_DIR_DOESNT_MATTER)

    #     def _path_to_feature_vec(traj):
    #         feat = (
    #             strokeCircularity([traj])[0],
    #             strokeDistances([traj])[0])
    #         return feat

    #     def feature_single_p(p):
    #         # first check if this is already saved
    #         key = _path_to_key(p["walker"])
    #         if key in tracker:
    #             return tracker[key]
    #         else:
    #             # get new
    #             feat = _path_to_feature_vec(p["traj"])
    #             tracker[key]=feat
    #             return feat

    #     def featurevec_single_trial(list_of_p):
    #         """ a single trial is a list of ps
    #             e..g, for trial ind, 
    #             list_of_p = P.extract_parses_wrapper(ind, "summary")
    #         OUT:
    #         - feature_vec, (N,) shape np array.
    #         """

    #         list_of_features = []
    #         for p in list_of_p:
    #             list_of_features.append(feature_single_p(p))

    #         # Take average over all strokes
    #         mat_of_features = np.stack(list_of_features) # N x nfeat
    #         feature_vec = np.mean(mat_of_features, axis=0)

    #         # Add other features
    #         # - max circ across trajs
    #         feature_vec = np.append(feature_vec, np.max(mat_of_features[:,0]))

    #         #- nstrokes
    #         nstrokes = len(list_of_p)
    #         feature_vec = np.append(feature_vec, nstrokes)

    #         return feature_vec
        
    #     list_featurevecs = []
    #     for ind in range(len(P.Parses)):
    #         list_of_p = P.extract_parses_wrapper(ind, "summary")
    #         feature_vec = featurevec_single_trial(list_of_p)
    #         list_featurevecs.append(feature_vec)
            
    #     return list_featurevecs


    def list_featurevec_from_mult_parser(self, D, ind, 
        parser_names = None,
        hack_lines5 = False):
        """ Get list of feature vecs, convatenated across all parses
        IN:
        - D, Dataset
        - ind, index into D
        - parser_names, columns for D
        OUT:
        - list of np arrays.
        """
        if parser_names is None:
            parser_names = ["parser_graphmod", "parser_nographmod"]
        
        list_of_parsers = D.parser_list_of_parsers(ind, parser_names=parser_names)
        list_featurevec_cat = []
        for P in list_of_parsers:
            if hack_lines5:
                list_featurevec = self.list_featurevec_from_parser_lines5(P)
            else:
                assert False, "need general purpose feature vec parasm"
            list_featurevec_cat.extend(list_featurevec)
        return list_featurevec_cat


    def list_featurevec_from_parser_lines5(self, P):
        """ 
        Given single parser (one trial/task, multipel parses, one skeleton graph)
        returns a list of featuresvecs.
        IN:
        - P, parser object, holds list of parses
        OUT:
        - list of featurevecs, one for each parse in P
        """

        tracker = {}
        STROKE_DIR_DOESNT_MATTER = self.Params["STROKE_DIR_DOESNT_MATTER"]

        def _path_to_feature_vec(traj):
            feat = (
                strokeCircularity([traj])[0],
                strokeDistances([traj])[0])
            return feat

        def feature_single_p(p):
            # first check if this is already saved
            key = self._path_to_key(p)
            if key in tracker:
                return tracker[key]
            else:
                # get new
                feat = _path_to_feature_vec(p["traj"])
                tracker[key]=feat
                return feat

        def featurevec_single_trial(list_of_p):
            """ a single trial is a list of ps
                e..g, for trial ind, 
                list_of_p = P.extract_parses_wrapper(ind, "summary")
            OUT:
            - feature_vec, (N,) shape np array.
            """

            list_of_features = []
            for p in list_of_p:
                list_of_features.append(feature_single_p(p))

            # Take average over all strokes
            mat_of_features = np.stack(list_of_features) # N x nfeat
            feature_vec = np.mean(mat_of_features, axis=0)

            # Add other features
            # - max circ across trajs
            feature_vec = np.append(feature_vec, np.max(mat_of_features[:,0]))

            #- nstrokes
            nstrokes = len(list_of_p)
            feature_vec = np.append(feature_vec, nstrokes)

            return feature_vec
        
        list_featurevecs = []
        for ind in range(len(P.Parses)):
            list_of_p = P.extract_parses_wrapper(ind, "summary")
            feature_vec = featurevec_single_trial(list_of_p)
            list_featurevecs.append(feature_vec)
            
        return list_featurevecs


    ############### USING pre-flattened parses
    def _path_to_feature_vec(self, p, list_features=None):
        if list_features is None:
            list_features = []
        feat = []
        traj = p["traj"]

        for f in list_features:
            if f =="circ":
                x = strokeCircularity([traj])[0]
            elif f=="dist":
                x = strokeDistances([traj])[0]/100 # 100
            else:
                print(f)
                assert False
            feat.append(x)
        # feat = (
        #     strokeCircularity([traj])[0],
        #     strokeDistances([traj])[0])

        return feat

    def _strokes_to_feature_vec(self, list_of_p, list_features=None):
        if list_features is None:
            list_features = []

        feat = []
        strokes = [p["traj"] for p in list_of_p]

        for f in list_features:
            if f=="angle_travel":
                # from center to center
                # NOTE: also see motor efficiency, for endpoitn to enpoitn, and norm to ground truth vector.
                from pythonlib.drawmodel.features import strokesAngleOverall
                if len(strokes)==1:
                    # then split into 2
                    from pythonlib.tools.stroketools import splitTraj
                    strokesthis = splitTraj(strokes[0])
                else:
                    strokesthis = strokes
                x = strokesAngleOverall(strokesthis)
                # print(x)
                # print(strokes)
                # print(len(strokes))
                # from pythonlib.drawmodel.strokePlots import plotDatStrokes
                # import matplotlib.pyplot as plt
                # fig, ax = plt.subplots()
                # plotDatStrokes(strokes, ax=ax)
                # assert False, "just check"
            elif f=="nstrokes":
                x = len(strokes)
            elif f=="dist_travel":
                # total distance travel, center to center, including position of fixation.
                from pythonlib.drawmodel.features import computeDistTraveled

                if len(strokes)==1:
                    # then split into 2
                    from pythonlib.tools.stroketools import splitTraj
                    strokesthis = splitTraj(strokes[0])
                else:
                    strokesthis = strokes

                orig = list_of_p[0]["origin"]
                x = computeDistTraveled(strokesthis, orig, center_to_center=True)/200

            else:
                print(f)
                assert False
            feat.append(x)
        return feat


    def _tracker_keys(self, D, indtrial, p):
        """ returns len 4 list
        D can be datsaet, or can be a string (since dataset shouldnt bvary for
        this featureextracter, normalyl.)
        """
        if isinstance(D, str):
            a = D   
        else:
            a = D.identifier_string()
        b = indtrial
        c = p["parser_name"]
        d = self._path_to_key(p)

        keys = [a,b,c,d]

        return keys

    def tracker_get(self, D, indtrial, p):
        """ checks if this combo of dataset, trial, and parse is done
        - need this conjunction in case the same feature extractor used across
        different datasets and trials
        OUT:
        - feature_vec if it exists,
        - otherwise None
        """

        # if a not in self._tracker.keys():
        #     return None
        # if b not in self._tracker[a].keys():
        #     return None
        # if c not in self._tracker[a][b]:
        #     return None
        # if d not in self._tracker[a][b][c][d]
        # return self._tracker[a][b][c][d]

        list_keys = self._tracker_keys(D, indtrial, p)
        this = self._Tracker
        for key in list_keys:
            if key not in this:
                return None 
            else:
                this = this[key]
        return this

    def tracker_add(self, D, indtrial, p, feature_vec):
        """ checks if this combo of dataset, trial, and parse is done
        - need this conjunction in case the same feature extractor used across
        different datasets and trials
        OUT:
        None
        """

        list_keys = self._tracker_keys(D, indtrial, p)
        this = self._Tracker
        for key in list_keys[:3]:
            if key not in this:
                this[key] = {}
            this = this[key]

        this[list_keys[3]] = feature_vec


    def list_featurevec_from_flatparses_directly(self, list_parses, indtrial, 
        hack_lines5=False):
        """ same as 
        list_featurevec_from_flatparses, but here pass in list of parses, 
        IN:
        - list_parses, list of list of p, i.e., this holds all parses for one
        - indtrial, is important for keeping track of edges.
        row of D.Dat, e.g,, list_parses = D.Dat.iloc[0]["parses_behmod"]
        """
        if hack_lines5:
            list_feature_names = ["circ", "dist"]
        else:
            list_feature_names = self.ListFeatures

        list_feature_names_trajlevel = [name for name in list_feature_names if name in ["circ", "dist"]]
        list_feature_names_strokeslevel = [name for name in list_feature_names if name in ["angle_travel", "nstrokes", "dist_travel"]]
        assert len(list_feature_names_trajlevel) + len(list_feature_names_strokeslevel) == len(list_feature_names)
        assert len(set(list_feature_names))==len(list_feature_names), "otherwise the putting back in place algo will not work, since uses index"
        # Inds that will use for putting things back in place
        inds_list_features = []
        for fthis in list_feature_names_trajlevel:
            inds_list_features.append(list_feature_names.index(fthis))
        for fthis in list_feature_names_strokeslevel:
            inds_list_features.append(list_feature_names.index(fthis))


        def feature_single_p(p):
            # first check if this is already saved
            # key = _path_to_key(p["walker"])

            feat = self.tracker_get("dummy", indtrial, p)
            if feat is not None:
                # print("got saved")
                return feat
            else:
                # get new
                feat = self._path_to_feature_vec(p, list_features=list_feature_names_trajlevel)
                self.tracker_add("dummy", indtrial, p, feat)
                # print("got new")
                return feat

        def feature_single_listofp(list_of_p):
            feat = self._strokes_to_feature_vec(list_of_p, list_features=list_feature_names_strokeslevel)
            return feat

        # def featurevec_single_parse(list_of_p):
        #     """ a single parse is equibalent to strokes.
        #          is a list of ps
        #         e..g, for trial ind, 
        #         list_of_p = P.extract_parses_wrapper(ind, "summary")
        #     OUT:
        #     - feature_vec, (N,) shape np array.
        #     """
            
        #     for f in list_feature_names:
        #         if f in ["circ", "dist"]:
        #             # do separateyl for each traj, then take average


        #         elif f in ["angle_travel"]:

        #         else:
        #             assert False


        #     list_of_features = []
        #     for p in list_of_p:
        #         list_of_features.append(feature_single_p(p))

        #     # Take average over all strokes
        #     mat_of_features = np.stack(list_of_features) # N x nfeat
        #     feature_vec = np.mean(mat_of_features, axis=0)

        #     # Add other features
        #     # - max circ across trajs
        #     feature_vec = np.append(feature_vec, np.max(mat_of_features[:,0]))

        #     #- nstrokes
        #     nstrokes = len(list_of_p)
        #     feature_vec = np.append(feature_vec, nstrokes)

        #     return feature_vec

        def featurevec_single_parse(list_of_p):
            """ a single parse is equibalent to strokes.
                 is a list of ps
                e..g, for trial ind, 
                list_of_p = P.extract_parses_wrapper(ind, "summary")
            OUT:
            - feature_vec, (N,) shape np array.
            """

            list_of_features = []
            for p in list_of_p:
                list_of_features.append(feature_single_p(p))

            # Take average over all strokes
            mat_of_features = np.stack(list_of_features) # N x nfeat
            feature_vec = np.mean(mat_of_features, axis=0)

            if hack_lines5==True:
                # Add other features
                # - max circ across trajs
                feature_vec = np.append(feature_vec, np.max(mat_of_features[:,0]))

                #- nstrokes
                nstrokes = len(list_of_p)
                feature_vec = np.append(feature_vec, nstrokes)

            return feature_vec
        
        nfeat = len(list_feature_names)
        mat_features = np.empty((len(list_parses), nfeat))


        for i, list_of_p in enumerate(list_parses):
            # list_of_p = P.extract_parses_wrapper(ind, "summary")
            
            feature_vec_p = featurevec_single_parse(list_of_p)
            feature_vec_listofp = feature_single_listofp(list_of_p)

            # put features back into main thing
            out = np.empty(len(list_feature_names))
            ct = 0
            for fthis, val in zip(list_feature_names_trajlevel, feature_vec_p):
                ind = inds_list_features[ct]
                out[ind] = val
                ct+=1
            for fthis, val in zip(list_feature_names_strokeslevel, feature_vec_listofp):
                ind = inds_list_features[ct]
                out[ind] = val
                ct+=1


            mat_features[i, :] = out
            # list_featurevecs.append(feature_vec)

        return mat_features


    def list_featurevec_from_flatparses(self, D, indtrial, 
        hack_lines5=False):
        """ 
        Similar to above.
        Tracker keeps track of edges for all parsers in the dict.
        OUT:
        - list of np arrays, one for each parse in this indtrial.
        """

        if hack_lines5:
            list_features = ["circ", "dist"]
        else:
            assert hack_lines5==True, "not coded"

        STROKE_DIR_DOESNT_MATTER = self.Params["STROKE_DIR_DOESNT_MATTER"]

        # initialize tracker if needed
        # parser_names = D.parser_names()
        # for name in parser_names:
        #     if name not in tracker.keys():
        #         tracker[name] = {}


        def feature_single_p(p):
            # first check if this is already saved
            # key = _path_to_key(p["walker"])

            feat = self.tracker_get(D, indtrial, p)
            if feat is not None:
                # print("got saved")
                return feat
            else:
                # get new
                feat = self._path_to_feature_vec(p, list_features=list_features)
                self.tracker_add(D, indtrial, p, feat)
                # print("got new")
                return feat

        def featurevec_single_trial(list_of_p):
            """ a single trial is a list of ps
                e..g, for trial ind, 
                list_of_p = P.extract_parses_wrapper(ind, "summary")
            OUT:
            - feature_vec, (N,) shape np array.
            """

            list_of_features = []
            for p in list_of_p:
                list_of_features.append(feature_single_p(p))

            # Take average over all strokes
            mat_of_features = np.stack(list_of_features) # N x nfeat
            feature_vec = np.mean(mat_of_features, axis=0)

            # Add other features
            # - max circ across trajs
            feature_vec = np.append(feature_vec, np.max(mat_of_features[:,0]))

            #- nstrokes
            nstrokes = len(list_of_p)
            feature_vec = np.append(feature_vec, nstrokes)

            return feature_vec
        
        if False:
            # list appending
            list_featurevecs = []
            list_list_p = D.Dat.iloc[indtrial]["parses_behmod"]
            for list_of_p in list_list_p:
                # list_of_p = P.extract_parses_wrapper(ind, "summary")
                feature_vec = featurevec_single_trial(list_of_p)
                list_featurevecs.append(feature_vec)
            mat_features = np.asarray(list_featurevecs)
        else:
            # np array
            list_list_p = D.Dat.iloc[indtrial]["parses_behmod"]
            nfeat = 4
            mat_features = np.empty((len(list_list_p), nfeat))
            for i, list_of_p in enumerate(list_list_p):
                # list_of_p = P.extract_parses_wrapper(ind, "summary")
                feature_vec = featurevec_single_trial(list_of_p)
                mat_features[i, :] = feature_vec
                # list_featurevecs.append(feature_vec)


        # if want to keep printing to see how much is cached.
        # if "Red_lines5_straight" in self._Tracker:
        #     print(len(self._Tracker["Red_lines5_straight"]))

        return mat_features


    ########### HELPERS
    def _path_to_key(self, p):
        """ tuple of ints, a unique path id
        (coincatnates list of edges)
        """
        x = p["walker"].unique_path_id(invariant=self.Params["STROKE_DIR_DOESNT_MATTER"])
        assert isinstance(x, tuple)
        return x
