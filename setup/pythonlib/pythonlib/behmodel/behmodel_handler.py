"""
See below.
WOrks with parses (graph repr of copntinuos strokes).
"""

import numpy as np
import matplotlib.pyplot as plt

DEBUG = False

class BehModelHandler(object):
    """ Combine single dataset D (beh) with multiple models (BM). Can share likeli computations
    """

    def __init__(self):
        """
        """

        # self.Likelis = None # hold likeli for each parse for each trial
        self.PriorProbs = None
        self.PriorLogProbs = None
        self.LikelisLogProbs = None
        self.Posteriors = None
        self.ParamsInputHelper = None
        self.D = None

        self.ParsesVer = "parses_behmod" # where to look to extract list_parses

    def input_data_helper(self, dataset, modelclass, list_mrules, auto_get_list_parsers=True):
        """
        Useful for reinstating a model after saving and loading
        IN:
        - auto_get_list_parsers, then uses column names in Datast to figure out list. e..g, ["parser_graphmod"]
        Otherwise uses hard coded in quick_getter_with_params
        """
        from .behmodel_getter import quick_getter_with_params
        list_mod, list_modnames, kwargs = quick_getter_with_params(modelclass, list_mrules)

        if auto_get_list_parsers:
            kwargs["parsers_to_flatten"] = [col for col in dataset.Dat.columns if "parser_" in col]
            assert len(kwargs["parsers_to_flatten"])>0

        self.input_data(dataset, list_mod, list_modnames, **kwargs) 

        # Save
        self.ParamsInputHelper = {
            "modelclass":modelclass,
            "list_mrules":list_mrules
        }

    def input_data(self, dataset, list_models, 
        list_model_ids=None, allow_separate_likelis=False,
        parsers_to_flatten = None):
        """
        IN:
        - dataset, a single D object
        - list_models, list of BehModel objects
        - list_model_ids, list of strings. if dont enter, then will generate
        auto based on timestamps.
        - allow_separate_likelis, then issue is will take longer if they dont share.
        - parsers_to_flatten, by default will flatten these parsers into list of parses, 
        where ach parse is a list of walks. This will go into a new column: "parses_behmod"
        OUT:
        - assigns to self.Dat, self.ListModels
        NOTE:
        - Requirements:
        --- Assumes all models use the same likeli scoring function (so can share likelis)
        --- Assume all models applied to the same one dataset.
        --- Otherwise models hve different priors and posetiro scorers.
        """

        if parsers_to_flatten is None:
            parsers_to_flatten = ['parser_graphmod', 'parser_nographmod']

        self._parsers_to_flatten = parsers_to_flatten
        self._allow_separate_likelis = allow_separate_likelis

        self.D = dataset
        self.ListModels = list_models

        if list_model_ids is not None:
            assert len(list_model_ids)==len(list_models)
            for M, name in zip(self.ListModels, list_model_ids):
                M.unique_id(set_id_to_this=name)
        else:
            list_model_ids = []
            for M in self.ListModels:
                list_model_ids.append(M.unique_id(ver="random_num"))

        self.ListModelsIDs = list_model_ids

        # dict from id to model
        self.DictModels = {name:model for name, model in zip(self.ListModelsIDs, self.ListModels)}

        assert len(self.DictModels)==len(self.ListModels), "probably keys are same."

        if self._allow_separate_likelis:
            self.Likelis = {name:[] for name in self.DictModels.keys()} # list of likelsi.
            self.LikelisLogProbs = {name:[] for name in self.DictModels.keys()} # list of likelsi.
        else:
            self.Likelis = []  # model --> list of likelis
            self.LikelisLogProbs  = []

        # Preprocess
        self.preprocess()

    def preprocess(self):
        """
        """
        # Flatten parsers
        self.D.parser_flatten(self._parsers_to_flatten)

        if self._allow_separate_likelis==False:
            # self.Likeli = self.ListModels[0].Likeli # asssuems all use same likeli
            if len(self.ListModels)>1:
                # Sanity check, make sure likelis do same thing
                list_likelis = [] # one for each model
                for name, Mod in self.DictModels.items():
                    list_likelis.append(self._score_helper("likeli", name, 0))
                    # list_likelis.append(Mod.Likeli.score(self.D, 0))

                nparse = len(list_likelis[0])
                for n in range(nparse):
                    a = [likelis[n] for likelis in list_likelis] # (x1,x2,x3) if 3 models, on this parse for first trial.
                    for aa in a[1:]:
                        if not np.isclose(aa,a[0]):
                            print(a)
                            assert False, "model likeli functions are different. just make separate BehModelHandlers if this is the case, or set allow_separate_likelis True"



    def _score_helper(self, scorever, modelname, i):
        """
        - General purpose for extracting list of scores (priors or likelis) for this trial(i)
        this model (modelname, string)
        NOTE:
        - this useful becuase figures out what arguments to pass in (based on params I set
        when initializing)
        """
        Mod = self.DictModels[modelname]

        # what args this needs?

        def _get_args(list_of_argstrings):
            args = None
            if list_of_argstrings==("dat","trial"):
                args = (self.D, i)
            elif list_of_argstrings == ("dat", "trial", "modelname"):
                args = (self.D, i, modelname)
            elif list_of_argstrings == ("parsesflat"):
                assert False, "if this, then shouldnt pass in args one by one... should use applyFuncToAllRows"
                args = (self.D, i, modelname)
            else:
                assert False

            if args is None:
                print(list_of_argstrings)
                assert False
            return args


        if scorever=="prior":
            args = _get_args(Mod._list_input_args_prior)
            return Mod.Prior.score_and_norm(*args)
        elif scorever=="likeli":
            args = _get_args(Mod._list_input_args_likeli)
            return Mod.Likeli.score(*args)
        else:
            print(scorever)
            assert False, "not coded"


    ##### PROCESS FOR SCORING
    def compute_store_likelis(self, force_run=False):
        """ doesnt care about model. just compute likelis and save them, for each
        line in datsaet.
        NOTE;
        - uses the likeli scorer for the first model, since assumes all models use same likeli.
        """

        def _get_list_likelis(modname):
            list_likelis = []
            for i in range(len(self.D.Dat)):
                if i%50==0:
                    print("likeli", i)
                likelis = self._score_helper("likeli", modname, i)
                # likelis = Mod.Likeli.score(self.D, i)
                list_likelis.append(likelis)
                # self.Likelis.append(likelis)
            return list_likelis

        if self._allow_separate_likelis==False:
            if len(self.Likelis)==0 or force_run==True:
                # use the first
                name = self.ListModelsIDs[0]
                self.Likelis = _get_list_likelis(name)
            else:
                print("skipping compute likelis since done")
        else:
            for name in self.DictModels.keys():
                if len(self.Likelis[name])==0 or force_run==True:
                    print("behmodel_handler - getting likelis for ", name)
                    self.Likelis[name] = _get_list_likelis(name)
                else:
                    print("skipping compute likelis since done, mod= ", name)

    def compute_store_likelis_logprobs(self, force_run=False):

        if self._allow_separate_likelis==False:
            # use the first
            if len(self.LikelisLogProbs)==0 or force_run==True:
                name = self.ListModelsIDs[0]
                # for i in range(len(self.Likelis)):
                    # scores = self.Likelis[i]
                    # scores = 100*scores
                    # logprobs = self.DictModels[name].Likeli.norm(scores)
                self.LikelisLogProbs = [self.DictModels[name].Likeli.norm(scores) for scores in self.Likelis]
            else:
                print("skipping compute log-likelis since done")

        else:
            for name in self.DictModels.keys():
                if len(self.LikelisLogProbs[name])==0 or force_run==True:
                    self.LikelisLogProbs[name] = [self.DictModels[name].Likeli.norm(scores) for scores in self.Likelis[name]]
                else:
                    print("skipping compute log-likelis since done")

    def compute_store_priorprobs(self, force_run=False): 
        """
        - force_run, then redoes, even if already donea nd saved.
        OUT:
        - svaes into : self.PriorProbs[modelid][trial]
        """
        assert "False, both vecorrized and not are in the compute_store_priorprobs_vectorized code"
        assert False, "dotn use this anymore. this takes probs, but I prior scorefun normalizes to logprobs."
        if self.PriorProbs is not None and force_run==False:
            print("skipping prior compute, since already done")
            return

        self.PriorProbs = {}
        for name, Mod in self.DictModels.items():
            self.PriorProbs[name] = []

            for indtrial in range(len(self.D.Dat)):
                if indtrial%50==0:
                    print("priors", name, indtrial)
                probs = self._score_helper("prior", name, indtrial)
                # probs = Mod.Prior.score_and_norm(self.D, indtrial)
                self.PriorProbs[name].append(probs)


    def compute_store_priorprobs_vectorized(self, force_run=False, just_sanity_check=False):
        """ 
        - Assumes that prior scorer takes in list of parses as an argument.
        this way can vectorize computation over all rows of Dat
        INPUTS:
        - just_sanity_check, then will run and return without modifying anything in self.
        NOTE:
        This seems to be about 4-5x faster than compute_store_priorprobs, on datsaet
        with about 250 rows (trials) and about 20-100 parses each.
        Tested on Red - lines5 - straight.
        See notebook: devo_taskmodel_finalized_071121
        """
        from pythonlib.tools.pandastools import applyFunctionToAllRows

        


        def _extract_probs(Mod):
            """ returns list of probs for this Mod
            """
            if Mod._list_input_args_prior[0]=="parsesflat":
                print("logprobs, getting vectorized across all trials", name)
                assert Mod._list_input_args_prior[0]=="parsesflat", "vectorized means need list parses as first argument."
                def F(x):
                    if Mod._list_input_args_prior==("parsesflat", "trialcode", "modelname"):
                        args = (x["parses_behmod"], x["trialcode"], name)
                    elif Mod._list_input_args_prior==("parsesflat", "trialcode"):
                        args = (x["parses_behmod"], x["trialcode"])
                    else:
                        print(Mod._list_input_args_prior)
                        assert False
                    logprobs = Mod.Prior.score_and_norm(*args)
                    return logprobs
                dfthis = applyFunctionToAllRows(self.D.Dat, F, "priorprobs")
                list_probs = dfthis["priorprobs"].to_list()
            else:
                # old version, not vecotorized
                print("Cant do vectorized priors - will be slower.")
                list_probs = []
                for indtrial in range(len(self.D.Dat)):
                    if indtrial%50==0:
                        print("priors", name, indtrial)
                    logprobs = self._score_helper("prior", name, indtrial)
                    # probs = Mod.Prior.score_and_norm(self.D, indtrial)
                    list_probs.append(logprobs)
            return list_probs


        if just_sanity_check:
            PriorProbsCheck = {}
            for name, Mod in self.DictModels.items():

                list_probs = _extract_probs(Mod)
                # self.PriorLogProbs[name] = []
                PriorProbsCheck[name] = list_probs
            return PriorProbsCheck


        if self.PriorLogProbs is not None and force_run==False:
            print("skipping prior compute, since already done")
            return

        self.PriorLogProbs = {}
        self.PriorProbs = {}
        
        for name, Mod in self.DictModels.items():

            # if Mod._list_input_args_prior[0]=="parsesflat":
            #     print("logprobs, getting vectorized across all trials", name)
            #     assert Mod._list_input_args_prior[0]=="parsesflat", "vectorized means need list parses as first argument."
            #     def F(x):
            #         if Mod._list_input_args_prior==("parsesflat", "trialcode", "modelname"):
            #             args = (x["parses_behmod"], x["trialcode"], name)
            #         elif Mod._list_input_args_prior==("parsesflat", "trialcode"):
            #             args = (x["parses_behmod"], x["trialcode"])
            #         else:
            #             print(Mod._list_input_args_prior)
            #             assert False
            #         logprobs = Mod.Prior.score_and_norm(*args)
            #         return logprobs
            #     dfthis = applyFunctionToAllRows(self.D.Dat, F, "priorprobs")
            #     list_probs = dfthis["priorprobs"].to_list()
            # else:
            #     # old version, not vecotorized
            #     print("Cant do vectorized priors - will be slower.")
            #     list_probs = []
            #     for indtrial in range(len(self.D.Dat)):
            #         if indtrial%50==0:
            #             print("priors", name, indtrial)
            #         logprobs = self._score_helper("prior", name, indtrial)
            #         # probs = Mod.Prior.score_and_norm(self.D, indtrial)
            #         list_probs.append(logprobs)

            list_probs = _extract_probs(Mod)
            # self.PriorLogProbs[name] = []
            self.PriorLogProbs[name] = list_probs

            # convert to probs
            if False:
                self.PriorProbs[name] = [np.exp(thisarr) for thisarr in self.PriorLogProbs[name]]

    def convert_prior_logprobs_to_probs(self):
        """ only runs if probs is not yet gotten 
        - iterates overa ll models
        """

        for name in self.ListModelsIDs:
            if name not in self.PriorProbs.keys():
                self.PriorProbs[name] = [np.exp(thisarr) for thisarr in self.PriorLogProbs[name]]


    def compute_store_posteriors(self, force_run=True, mode="train"):
        """
        OUT:
        - svaes into : self.Posteriors[modelid][trial]
        """

        assert force_run==True, "no point, since is quick."

        self.Posteriors = {}
        for name, Mod in self.DictModels.items():
            self.Posteriors[name] = []
            for indtrial in range(len(self.D.Dat)):
                if indtrial%50==0:
                    print("posterior", name, indtrial)

                if mode=="train":
                    likelis = self._get_list_likelis(name, indtrial)
                elif mode=="test":
                    # Then likelis should not be normalized, since want
                    # posterior to be in units of (spatial distance)
                    likelis = self._get_list_likelis(name, indtrial, log=False)
                else:
                    assert False

                if mode=="train":
                    probs = self._get_list_probs(name, indtrial)
                elif mode=="test":
                    probs = self._get_list_probs(name, indtrial, log=False)
                else:
                    assert False

                if mode=="train": 
                    post = Mod.Poster.score(likelis, probs)
                elif mode=="test":
                    post = None
                    if hasattr(Mod, "PosterTest"):
                        if Mod.PosterTest is not None:
                            # added for chunks, want to take max likeli over permissible parses.
                            post = Mod.PosterTest.score(likelis, probs)
                    if post is None:
                        # take weighted sum of likelis.
                        from pythonlib.behmodel.scorer.utils import posterior_score
                        post = posterior_score(likelis, probs, ver="weighted")
                else: 
                    assert False

                # print(likelis)
                # print(probs)
                # print(post)
                # print(np.argmax(likelis))
                # print(likelis[np.argmax(likelis)])
                # print(probs[np.argmax(likelis)])
                # assert False
                self.Posteriors[name].append(post)


    def compute_all(self, mode="train"):
        """ scoring data 
        INPUT:
        - mode, str, {'train', 'test'}, 
        --- if test, then also gets priors in probs (not just log),
        and reruns everything (forces)

        """
        if mode=="train":
            self.compute_store_priorprobs_vectorized(force_run=True)
            self.compute_store_likelis(force_run=False)
            self.compute_store_likelis_logprobs(force_run=False)
            self.compute_store_posteriors(force_run=True, mode=mode)
        elif mode=="test":
            self.compute_store_priorprobs_vectorized(force_run=True)
            # make sure that probs gotten
            self.convert_prior_logprobs_to_probs()
            self.compute_store_likelis(force_run=True)
            self.compute_store_likelis_logprobs(force_run=True)
            self.compute_store_posteriors(force_run=True, mode=mode)


    def final_score(self, modelname, per_trial=False, prob=False):
        """ 
        Sum of posteriors (which are log probs)
        """
        post_score = np.sum(self.Posteriors[modelname])
        if per_trial:
            post_score = post_score/len(self.Posteriors[modelname])
        if prob:
            post_score = np.exp(post_score)

        return post_score


    ##### GETTERS
    def _get_list_probs(self, modname, indtrial, log=None):
        """ get prior, in formate needed for computing posterior
        INPUT:
        - log, if None, then uses whatever prescribed by the model
        otherwise True/False forces to use that.
        """

        if log==True:
            this = self.PriorLogProbs
        elif log==False:
            this = self.PriorProbs
            if len(this)==0:
                # then extract
                self.convert_prior_logprobs_to_probs()
            this = self.PriorProbs
        elif log==None:
            if self.DictModels[modname]._poster_use_log_prior:
                this = self.PriorLogProbs
            else:
                this = self.PriorProbs
        else:
            assert False
        if len(this)==0:
            assert False, "did nto extract prior..."
        return this[modname][indtrial]

    def _get_list_likelis(self, modname, indtrial, log=None):
        """ get likeli, in formate needed for computing posterior
        log, if None, then uses whatever prescribed by the model
        otherwise True/False forces to use that.
        """

        if log==True:
            this = self.LikelisLogProbs
        elif log==False:
            this = self.Likelis
        elif log==None:
            if self.DictModels[modname]._poster_use_log_likeli:
                this = self.LikelisLogProbs
            else:
                this = self.Likelis
        else:
            assert False

        if self._allow_separate_likelis:
            return this[modname][indtrial]
        else:
            return this[indtrial]


    ##### PARAMS
    def params_prior(self, modelname):
        """
        Get the params for prior, including 
        - norm function
        - for motor cost model in the prior
        """
        out = {}
        if hasattr(self.DictModels[modelname].Prior, "Objects"):
            if "MotorCost" in self.DictModels[modelname].Prior.Objects.keys():
                out["MotorCost"] = self.DictModels[modelname].Prior.Objects["MotorCost"].Params
        out["main"] = self.DictModels[modelname].Prior.Params

        return out

    def params_prior_set(self, modelname, indict):
        """ 
        NOTE:
        indict should correspond to out in self.params_prior
        """
        if "MotorCost" in indict.keys():
            print("* Setting param, MotorCost", modelname, "to" , indict["MotorCost"])
            self.DictModels[modelname].Prior.Objects["MotorCost"].Params = indict["MotorCost"]
        self.DictModels[modelname].Prior.Params = indict["main"]
        print("* Setting param, Main", modelname, "to" , indict["main"])

    def params_dset(self, get_dataset=False):
        out = {}
        if self.D is not None:
            out["dset_id"] = self.D.identifier_string()
            out["dset_trialcodes"] = self.D.Dat["trialcode"].to_list()
            if get_dataset:
                out["dset"] = self.D
        return out

    def params_model(self):
        out = {
            "modelnames":self.ListModelsIDs,
            # "modelobjects":self.ListModels,
            "allow_separate_likelis":self._allow_separate_likelis,
            "parsers_to_flatten": self._parsers_to_flatten}
        return out



    ##### POST-PROCESS
    def results_to_dataset(self, suffix=""):
        """ assign results (posterior scores, one for each row) 
        back into the dataset.
        will make new column called:
        "modelpost_{modename}", for each model
        """

        list_col_names = []
        for name, scores in self.Posteriors.items():
            assert len(self.D.Dat)==len(scores)
            newcol = f"behmodpost_{name}_{suffix}"

            self.D.Dat[newcol] = scores
            list_col_names.append(newcol)
            d = self.D.identifier_string()
            print(f"scores into Dat: {d} , col:", newcol)

        return list_col_names


    #### EXTRACT DATA
    def extract_list_parsesstrokes(self, indtrial, modelname=None):
        if self.ParsesVer == "parses_behmod":
            return self.D.parserflat_extract_strokes(indtrial)
        elif self.ParsesVer == "mkvsmk":
            assert modelname is not None, "mk vs mk has different parses per model."
            colname = f"strokes_beh_group_{modelname}"
            return self.D.Dat.iloc[indtrial][colname]

    def extract_inds_trials_sorted(self, modelname, sort_by="posterior"):
        """ returns list of inds, sorted in increaseing order of sortby
        """
        if sort_by=="posterior":
            scores = self.Posteriors[modelname]
        else:
            print(sort_by)
            assert False, "not coded"

        inds = np.argsort(scores)
        return inds, np.array([scores[i] for i in inds])


    def extract_priors_likelis(self, modelname, indtrial, inds_parses=None, sort_by=None,
        log_probs = True, log_likelis=False):
        """ can choose to get for specific inds paress. will do this _before_ sorting.
        returns lists:
        - inds, priors, likelis
        """

        if sort_by=="poster":
            assert False, "poster is a scalar for each trial. can get per parse, have to code that..."

        # if log_probs:
        #     priors = self.PriorLogProbs[modelname][indtrial]
        # else:
        #     priors = self.PriorProbs[modelname][indtrial]
        priors = self._get_list_probs(modelname, indtrial, log=log_probs)
        likelis = self._get_list_likelis(modelname, indtrial, log=log_likelis)
        # posters = self.Posteriors[modelname][indtrial]

        if inds_parses is None:
            inds_parses = range(len(priors))
        if any(a>=len(priors) for a in inds_parses):
            print(len(priors))
            print(inds_parses)
            assert False
        if any(a>=len(likelis) for a in inds_parses):
            print(len(likelis))
            print(inds_parses)
            assert False

        priors = [priors[i] for i in inds_parses]
        likelis = [likelis[i] for i in inds_parses]
        # posters = [posters[i] for i in inds_parses]

        if sort_by is not None:
            tmp = [(p, l, i) for p,l, i in zip(priors, likelis, inds_parses)]
            if sort_by=="prior":
                tmp = sorted(tmp, key=lambda x:x[0])
            elif sort_by=="likeli":
                tmp = sorted(tmp, key=lambda x:x[1])
            elif sort_by=="poster":
                tmp = sorted(tmp, key=lambda x:x[3])
            else:
                assert False
            priors = [t[0] for t in tmp]
            likelis = [t[1] for t in tmp]
            inds_parses = [t[2] for t in tmp]

        return priors, likelis, inds_parses

    def extract_info_single_parse(self, indtrial, indparse, log_prob = False,
     increase_probs = False, increase_likeli=False, abbrev=False, modelname=None):
        """
        For ghenerating a string that can use as title for this single parse.
        combines likeli, prior (across all models).
        """

        out = []
        out.append(int(indparse))

        if log_prob:
            assert increase_probs==False

        if self.ParsesVer=="mkvsmk":
            # then parses are specifc to each model.
            assert modelname is not None
            priors, likelis, _ = self.extract_priors_likelis(
                modelname, indtrial, inds_parses=[indparse], log_probs=log_prob, sort_by=None)
            if increase_likeli is not False:
                likelis = [increase_likeli*l for l in likelis]
            if increase_probs is not False:
                priors = [increase_probs*p for p in priors]
            if abbrev:
                out.append(modelname[0])
            else:
                out.append(modelname)
            out.append(priors[0])
            if not abbrev:
                out.append("likeli")
            else:
                out.append("L")
            out.append(likelis[0])
        else:
            for modelname in self.ListModelsIDs:
                priors, likelis, _ = self.extract_priors_likelis(
                    modelname, indtrial, inds_parses=[indparse], log_probs=log_prob, sort_by=None)
                if increase_likeli is not False:
                    likelis = [increase_likeli*l for l in likelis]
                if increase_probs is not False:
                    priors = [increase_probs*p for p in priors]
                if abbrev:
                    out.append(modelname[0])
                else:
                    out.append(modelname)
                out.append(priors[0])
            if not abbrev:
                out.append("likeli")
            else:
                out.append("L")
            out.append(likelis[0])

        outstring = ""
        for i, o in enumerate(out):
            if not abbrev:
                if i>0:
                    outstring += "_"

            if isinstance(o, str):
                outstring += o[:3]
            elif isinstance(o, int):
                outstring += str(o)
            else:
                outstring += f"{o:.1f}"

        return out, outstring


    def summarize_results_trial(self, modelname, indtrial):
        """
        return dict with some useful summary scores
        """

        out = {}

        likelis = self._get_list_likelis(modelname, indtrial)
        # likelis = self.Likelis[indtrial]
         # self.PriorLogProbs[modelname][indtrial]

        logprobs = self._get_list_probs(modelname, indtrial, log=True)

        # Prob assigned to max likeli:
        from pythonlib.behmodel.scorer.utils import posterior_score
        out["logprob_of_maxlikeli"] = posterior_score(likelis, logprobs, "prob_of_max_likeli")

        # Rank of max-likeli parse, based on prior scores.
        # i.e. take the highest prior for model. where does that trial rank compared to animal?
        plist, llist, _ = self.extract_priors_likelis(modelname, indtrial, sort_by="likeli")
        from pythonlib.tools.listtools import rankinarray1_of_minofarray2
        plist = plist[::-1]
        llist = llist[::-1]

        ind = rankinarray1_of_minofarray2(llist, plist, look_for_max=True)[0]
        out["rank_of_maxprior_compared_to_monkey"] = ind        

        out["num_parses"] = len(plist)

        out["prob_if_random"] = 1/len(plist)

        return out



    ##### SAVING 
    def save_state(self, path):
        state_dict = self.extract_state()

    def extract_state(self, get_dataset=False):
        """ can use to repopulate state, save etc
        """

        state = {}

        # Prior params
        state["prior_params"] = {}
        for modname in self.ListModelsIDs:
            state["prior_params"][modname] = self.params_prior(modname)

        # Dataset information
        out = self.params_dset(get_dataset=get_dataset)
        for k, v in out.items():
            state[k] = v

        # Model info
        out = self.params_model()
        for k, v in out.items():
            state[k] = v

        # if used input helper
        if self.ParamsInputHelper is not None:
            state["input_helper_used"] = True
            state["input_helper_params"] = self.ParamsInputHelper
        else:            
            state["input_helper_used"] = False
            state["input_helper_params"] = None

        # Extracted values
        state["PriorProbs"] = self.PriorProbs
        state["PriorLogProbs"] = self.PriorLogProbs
        state["LikelisLogProbs"] = self.LikelisLogProbs
        state["Likelis"] = self.Likelis
        state["Posteriors"] = self.Posteriors

        return state

    def apply_state(self, state):
        """ load state, then repopulate.
        will overwrite when needed
        """

        # modelnames = 


        # TODO:
        # if datasets already present, check that they match
        # if not present, then add these
        # out["dset_id"] = self.D.identifier_string()
        # out["dset_trialcodes"] = self.D.Dat["trialcode"].to_list()
        # if get_dataset:
        #     out["dset"] = self.D

        self.ListModelsIDs = state["modelnames"] 
        self._allow_separate_likelis = state["allow_separate_likelis"]
        self._parsers_to_flatten = state["parsers_to_flatten"]

        for modelname in self.ListModelsIDs:
            self.params_prior_set(modelname, state["prior_params"][modelname])
        
        self.ParamsInputHelper = state["input_helper_params"]

        self.PriorProbs = state["PriorProbs"]
        self.PriorLogProbs = state["PriorLogProbs"]
        self.LikelisLogProbs = state["LikelisLogProbs"]
        self.Likelis = state["Likelis"]
        self.Posteriors = state["Posteriors"]


    ##### PLOTTING
    def plotMultStrokes(self, list_strokes, **kwargs):
        return self.D.plotMultStrokes(list_strokes, ncols=8, SIZE=3, **kwargs)


    def plot_parses_trial(self, indtrial, list_indparses=None, Nmax = 20, modelname=None, 
        plot_beh_task = True, abbrev=True):
        """ plot for this trial all parses in the order given by
        list_indparses. If dont specitfy, then plot 20 random
        - abbrev, for plotting titles (priors - liklelis)
        """
        import random
        def make_title(ind):
            pri = self._prior_probs[ind]*100
            likeli = self._likeli_scores[ind]*100

            return f"pri:{pri:.2f}, li{likeli:.2f}"
        
        list_fig = []
        if plot_beh_task:
            ## -- Plot the actual behavior on this trial
            # self.D.plotMultTrials([indtrial], which_strokes="strokes_beh")
            # self.D.plotMultTrials([indtrial], which_strokes="strokes_task")
            fig = self.D.plotSingleTrial(indtrial)
            list_fig.append(fig)
            # add posetiro scores
            tmp = ""
            for name in self.ListModelsIDs:
                p = self.Posteriors[name][indtrial]
                tmp += f"{name}_{p:.3f}"
            fig.suptitle(tmp)


        list_of_parsestrokes = self.extract_list_parsesstrokes(indtrial, modelname=modelname)
        
        if list_indparses is None:
            list_indparses = range(len(list_of_parsestrokes))
            # subsample
            if len(list_indparses)>Nmax:
                list_indparses = random.sample(list_indparses, Nmax)
                print("Got random sample of inds")

        # titles 
        titles = None
        titles = [self.extract_info_single_parse(indtrial, indparse, False, 100, 100, abbrev=abbrev, modelname=modelname)[1] for indparse in list_indparses]


        ## -- Plot trials, sorted by their scores
        liststrokes = [list_of_parsestrokes[i] for i in list_indparses]
        fig = self.plotMultStrokes(liststrokes, titles=titles, titles_on_y=not abbrev)
        list_fig.append(fig)

        return list_fig

    ####### OVERVIEW PLOTS
    def plot_prior_likeli_sorted(self, modelname, indtrial, sort_by, title=None):
        """ simple, plot a single graph of scores, soerted by either prior or likeli,
        """
        fig, ax = plt.subplots(1, 1, sharex=False, figsize=(13,3))
        # ax = axes.flatten()[0]
        priors, likelis, _ = self.extract_priors_likelis(modelname, indtrial, sort_by=sort_by, log_probs=False)
        post = self.Posteriors[modelname][indtrial]
        x = np.arange(len(priors))
        if title is None:
            ax.set_title(f"sorted by: {sort_by}; post={post:.2f}")
        else:
            title = title + f"-sorted by: {sort_by}; post={post:.2f}"
            ax.set_title(title)

        ax.plot(x, priors, "-ok", label="prior")
        ax.set_ylim(0)
        ax.set_ylabel("bk=prior")
        ax2 = ax.twinx()
        ax2.plot(x, likelis, "-or", label="likeli")
        ax2.set_ylabel("rd=likeli")

        return fig

    def plot_overview_trial(self, modelname, indtrial):
        """ 
        """

        # 1) Plot likeli and priors
        fig, axes = plt.subplots(3, 1, sharex=False, figsize=(13,10))
        for sort_by, ax in zip([None, "prior", "likeli"], axes.flatten()):
            priors, likelis, _ = self.extract_priors_likelis(modelname, indtrial, sort_by=sort_by, log_probs=False)
            x = np.arange(len(priors))

        #     ax = axes[0]
            ax.set_title(f"sorted by: {sort_by}")
            ax.plot(x, priors, "-ok", label="prior")
            ax.set_ylim(0)
            ax.set_ylabel("bk=prior")
            ax2 = ax.twinx()
            ax2.plot(x, likelis, "-or", label="likeli")
            ax2.set_ylabel("rd=likeli")

        # 1b) Plot, using log probs (both prior and likeli)
        fig, axes = plt.subplots(3, 1, sharex=False, figsize=(13,10))
        for sort_by, ax in zip([None, "prior", "likeli"], axes.flatten()):
            priors, likelis, _ = self.extract_priors_likelis(modelname, indtrial, sort_by=sort_by, log_probs=True, log_likelis=True)
            x = np.arange(len(priors))

        #     ax = axes[0]
            ax.set_title(f"sorted by: {sort_by}")
            ax.plot(x, priors, "-ok", label="prior")
            ax.set_ylabel("bk=prior")
            ax.set_ylim(top=0.)
            ax2 = ax.twinx()
            ax2.plot(x, likelis, "-or", label="likeli")
            ax2.set_ylabel("rd=likeli")
            ax2.set_ylim(top=0.)


            
        # 2) Scatter likeli and prior
        fig, axes = plt.subplots(1, 2, sharex=False, figsize=(12,6))
        priors, likelis, _ = self.extract_priors_likelis(modelname, indtrial, sort_by=None, log_probs=False)

        ax = axes.flatten()[0]
        ax.plot(priors, likelis, "ok")
        ax.set_ylabel('likeli');
        ax.set_xlabel('prior probs');
        ax = axes.flatten()[1]
        ax.plot(np.log(priors), likelis, "ok")
        ax.set_ylabel('likeli');
        ax.set_xlabel('log probs');


        # 3) Plot parses, given list of inds
        # pri
        self.plot_parses_ordered(indtrial, modelname, 8)
        # sort_by = "prior"
        # inds_parses = self.extract_priors_likelis(modelname, indtrial, sort_by=sort_by)[2]
        # self.plot_parses_trial(indtrial, inds_parses[::-1][:8])
        # sort_by = "likeli"
        # inds_parses = self.extract_priors_likelis(modelname, indtrial, sort_by=sort_by)[2]
        # self.plot_parses_trial(indtrial, inds_parses[::-1][:8])

    def plot_parses_ordered(self, indtrial, modelname=None, Nplot = 8, plot_beh_task=True, 
        plots=None, title=None):
        """ 
        plot top N ordered by prior, likeli, and post
        If modelname is None, then plots all
        """

        if plots is None:
            plots = ["prior", "likeli"]

        if modelname is None:
            for modelname in self.ListModelsIDs:
                self.plot_parses_ordered(indtrial, modelname)

        #3) Plot parses, given list of inds
        list_figs = []
        # pri
        pbh = plot_beh_task
        for this in plots:
            inds_parses = self.extract_priors_likelis(modelname, indtrial, sort_by=this)[2]
            fig = self.plot_parses_trial(indtrial, inds_parses[::-1][:Nplot], modelname=modelname, plot_beh_task=pbh)
            list_figs.append(fig)
            if title is not None:
                fig.suptitle(f"{title}-{this}")
            pbh = False
        # if "likeli" in plots:
        #     sort_by = "likeli"
        #     inds_parses = self.extract_priors_likelis(modelname, indtrial, sort_by=sort_by)[2]
        #     self.plot_parses_trial(indtrial, inds_parses[::-1][:Nplot], modelname=modelname, plot_beh_task=False)
        # sort_by = "poster"
        # inds_parses = self.extract_priors_likelis(modelname, indtrial, sort_by=sort_by)[2]
        # self.plot_parses_trial(indtrial, inds_parses[::-1][:Nplot])
        return list_figs

    def print_overview_params(self):
        """ Print params for all models
        """

        try:
            modelclass = self.ParamsInputHelper["modelclass"]
        except Exception as err:
            modelclass = ""

        # Prior
        for mod in self.ListModelsIDs:
            print(" ")
            prms = self.params_prior(mod)
            print("*", modelclass, mod)
            for k, v in prms.items():
                if isinstance(v, dict):
                    print("**", k)
                    for kk, vv in v.items():
                        print("***", kk, vv)
                else:
                    print("**", k, v)

    def plot_overview_results(self, modelname):
        """
        """
        from pythonlib.tools.plottools import hist_with_means

        list_outs = [self.summarize_results_trial(modelname, i) for i in range(len(self.D.Dat))]

        fig, axes = plt.subplots(3,2, figsize=(18, 12))

        # Rank of max prior trial (where 0 means this is also what monkey did)
        list_ranks = np.array([o["rank_of_maxprior_compared_to_monkey"] for o in list_outs])
        ax = axes.flatten()[0]
        # ax.hist(list_ranks, bins = range(max(list_ranks)))
        ax.set_title("rank_of_maxprior_compared_to_monkey")
        ax.set_xlabel("0 is best")
        bins = hist_with_means(ax, list_ranks, bins = range(max(list_ranks)))
        # overlay num parses
        list_nparses = np.array([o["num_parses"] for o in list_outs])
        ax.hist(list_nparses, bins=bins, histtype="step")

        # Rank of max prior trial (where 0 means this is also what monkey did)
        list_ranks = np.array([o["rank_of_maxprior_compared_to_monkey"] for o in list_outs])
        ax = axes.flatten()[1]
        # ax.hist(list_ranks, bins = 20)
        bins = hist_with_means(ax, list_ranks, bins = 20)
        ax.set_title("rank_of_maxprior_compared_to_monkey")
        ax.set_xlabel("0 is best")
        # overlay num parses
        list_nparses = np.array([o["num_parses"] for o in list_outs])
        ax.hist(list_nparses, bins=bins, histtype="step")

        # Model prob for max likeli
        list_lprobs = np.array([o["logprob_of_maxlikeli"] for o in list_outs])
        list_probs_if_random = np.array([o["prob_if_random"] for o in list_outs])
        ax = axes.flatten()[2]
        bins= hist_with_means(ax, list_lprobs, bins = 20)
        ax.set_title("model prob for max likeli trial")
        ax.set_xlabel("log probs")
        ax.hist(np.log(list_probs_if_random), bins=bins, histtype="step")

        # Model prob for max likeli
        ax = axes.flatten()[3]
        bins = hist_with_means(ax, np.exp(list_lprobs), bins = 20)
        ax.set_title("model prob for max likeli trial")
        ax.set_xlabel("probs")
        ax.hist(list_probs_if_random, bins=bins, histtype="step")

        ax = axes.flatten()[4]
        ax.plot(list_ranks, list_lprobs, "ok")
        ax.set_xlabel("rank_of_maxprior_compared_to_monkey")
        ax.set_ylabel("model prob for max likeli trial")

        # model params
        prms = self.params_prior(modelname)
        if "MotorCost" in prms.keys():
            thetavec = prms["MotorCost"]["thetavec"]
            ax = axes.flatten()[5]
            ax.plot(range(len(thetavec)),thetavec, "-ok")
            ax.set_title(f"softmax-beta: {prms['main']['norm'][0]:.2f}")

        return fig



def prepare_optimization_scipy(H, modelname, hack_lines5=False):
    """ prepare a function that can pass into scipy optimize.
    assumes that thetavec = (bendiness, _,_,_),. where _ is stuff that not optimize.
    """
    
    if DEBUG:
        import jax.numpy as np
    else:
        import numpy as np

    if hack_lines5:
        def func(prms, reg_coeff=np.array([0.0001, 0.001]), reg_mu=np.array([0, 1])):
        #     norm = [prms[1]]
        #     theta = [prms[:4]
            
            # do update
            H.DictModels[modelname].Prior.Params["norm"] = [prms[1]]
            H.DictModels[modelname].Prior.Objects["MotorCost"].Params["thetavec"][0] = prms[0]

            # recompute 
            H.compute_store_priorprobs_vectorized(force_run=True)
            H.compute_store_likelis()
            H.compute_store_likelis_logprobs()
            H.compute_store_posteriors()
            loss_per_task = -H.final_score(modelname, per_trial=True)

            # regularization
            loss_reg = np.sum(reg_coeff*(np.array(prms) - reg_mu)**2)
            # loss_reg = np.sum(tmp[:])
            loss = loss_per_task + loss_reg

            return loss
            
        params0 = (0., 1.)
    else:

        params0, reg_mu = H.DictModels[modelname].Prior.params_ravel(return_reg_mus=True)
        reg_coeff = 0.0005 * np.ones_like(params0)

        bounds = [(0, 10.) for _ in range(len(params0))]

        # Make sure circulaitry can be both neg and positive (straight vs. bend)
        if "MotorCost" in H.params_prior(modelname):
            if "circ" in H.params_prior(modelname)["MotorCost"]["thetanames"]:
                ind = H.params_prior(modelname)["MotorCost"]["thetanames"].index("circ")
                bounds[ind] = (-20., 20)

        def func(prms, reg_coeff=reg_coeff, reg_mu=reg_mu):
            # do update
            H.DictModels[modelname].Prior.params_unravel(prms)

            # recompute 
            H.compute_store_priorprobs_vectorized(force_run=True)
            H.compute_store_likelis()
            H.compute_store_likelis_logprobs()
            H.compute_store_posteriors()
            loss_per_task = -H.final_score(modelname, per_trial=True)

            # regularization
            loss_reg = np.sum(reg_coeff*(np.array(prms) - reg_mu)**2)
            # loss_reg = np.sum(tmp[:])
            loss = loss_per_task + loss_reg

            return loss
            

    return func, params0, bounds


def cross_dataset_model(Dlist, list_models, list_model_names, allow_separate_likelis=False, 
                        name_suffix=""):
    list_bmh = []
    for Dthis in Dlist:
        print(Dthis.identifier_string())

        # Generate models
        H = BehModelHandler()
        H.input_data(Dthis, list_models, 
                     list_model_names, allow_separate_likelis=allow_separate_likelis)

        # get likelis
        H.compute_store_likelis()
        H.compute_store_priorprobs()
        H.compute_store_posteriors()

        list_bmh.append({
            "D":Dthis,
            "H":H})

    # Assign modeling results back into Dataset
    for x in list_bmh:
        H = x["H"]
        list_col_names = H.results_to_dataset(suffix=name_suffix)

def cross_dataset_model_wrapper_params(Dlist, model_class, GROUPING_LEVELS):
    """ GOOD: higher level wrapper, where now use H itself to load all model
    classes. This useful since allows to save and reload states easily
    """
    ListBMH = []
    list_dsets = []
    ListH = []
    for Dthis in Dlist:

        # Generate models
        H = BehModelHandler()
        H.input_data_helper(Dthis, model_class, GROUPING_LEVELS)
        ListH.append(H)

        # H.input_data(Dthis, list_mod, 
        #              list_mod_names, allow_separate_likelis=allow_separate_likelis)

        # # get likelis
        # H.compute_store_likelis()
        # H.compute_store_priorprobs()
        # H.compute_store_posteriors()

        # for each combo of model and dataset, generate a function that takes in 
        # params to optimize, and outputs score.

        list_dsets.append(Dthis.identifier_string())
        list_mod_names = H.ListModelsIDs

        if model_class=="mkvsmk":
            H.ParsesVer = "mkvsmk" # important since two models have differnt paress.

        for modname in list_mod_names:
            print("** dset: ", Dthis.identifier_string(), "model: ", modname)

            if model_class=="lines5":
                func, params0 = prepare_optimization_scipy(H, modname, hack_lines5=True)
            else:
                func, params0, bounds = prepare_optimization_scipy(H, modname)

            ListBMH.append({
                "id_dset":Dthis.identifier_string(),
                "id_mod":modname,
                "func":func,
                "func_params0":params0,
                "func_bounds":bounds,
                "D":Dthis,
                "H":H})

    return ListBMH, list_dsets, ListH

def cross_dataset_model_wrapper(Dlist, list_mod, list_mod_names, 
    allow_separate_likelis=False):
    """ OK, but old version, since doesnt use H to load, but you must pass in preloaded listmod, etc.:
    Returns dict allowing you to index by dataset and model, while behind the scenes
    it takes care of the shared dependencies between them.
    OUT:
    - list_bmh, list of dicts, where each dict represents one combo of dataset and model
    NOTE:
    - assumes that all models in list_mod apply to same dataset, and can either share likelis (True)
    or not.
    """

    ListBMH = []
    list_dsets = []
    for Dthis in Dlist:

        # Generate models
        H = BehModelHandler()
        H.input_data(Dthis, list_mod, 
                     list_mod_names, allow_separate_likelis=allow_separate_likelis)

        # # get likelis
        # H.compute_store_likelis()
        # H.compute_store_priorprobs()
        # H.compute_store_posteriors()

        # for each combo of model and dataset, generate a function that takes in 
        # params to optimize, and outputs score.

        list_dsets.append(Dthis.identifier_string())
        for modname in list_mod_names:
            print("** dset: ", Dthis.identifier_string(), "model: ", modname)

            func, params0 = prepare_optimization_scipy(H, modname)

            ListBMH.append({
                "id_dset":Dthis.identifier_string(),
                "id_mod":modname,
                "func":func,
                "func_params0":params0,
                "D":Dthis,
                "H":H})

    return ListBMH, list_dsets


def bmh_prepare_optimization(ListBMH):
    for L in ListBMH:
        modname = L["id_mod"]
        H = L["H"]
        func, params0 = prepare_optimization_scipy(H, modname)

        L["func"] = func
        L["func_params0"] = params0



def bmh_score_single(ListBMH, id_dset, id_mod):
    """ dont optimize params, just score using whataver params currently
    present in model
    """

def bmh_score_grid(ListBMH, id_dset, id_mod, params_grid):
    """ 
    params_grid is list of np.arrays, list must be same length as 
    params that pass into func. 
    """
    # Grid search
    # COST LANDSCAPE
    # # array_norms = np.linspace(0.1, 2, 5)
    # #     array_norms = np.linspace(0.1, 2, 4)
    # array_norms = np.asarray([1.])
    from itertools import product

    this = [L for L in ListBMH if L["id_dset"]==id_dset and L["id_mod"]==id_mod]
    assert len(this)==1
    this = this[0]
    func = this["func"]
    H = this["H"]

    out = []
    for prms in product(*params_grid):
        prms = np.array(prms)
        print("Currently doing params: ", prms)
        cost = func(prms)
        out.append({
            "cost":cost,
            "prms":prms,
            "dset":id_dset,
            "mod":id_mod            
        })
    return out



def bmh_optimize_single(ListBMH, id_dset, id_mod):
    """ runs optimization for this single case
    """
    from pythonlib.tools.modfittools import minimize

    this = [L for L in ListBMH if L["id_dset"]==id_dset and L["id_mod"]==id_mod]
    if len(this)!=1:
        print(ListBMH)
        print(id_dset)
        print(id_mod)
        assert False
    assert len(this)==1
    this = this[0]

    res = minimize(this["func"], this["func_params0"], bounds=this["func_bounds"])

    return res

def bmh_optimize_single_jax(ListBMH, id_dset, id_mod):
    """ runs optimization for this single case
    Uses Jax
    Doesnt work with bounds.
    """
    from jax.scipy.optimize import minimize

    this = [L for L in ListBMH if L["id_dset"]==id_dset and L["id_mod"]==id_mod]
    assert len(this)==1
    this = this[0]

    res = minimize(this["func"], this["func_params0"], method="BFGS")
    return res

def bmh_results_to_dataset(ListBMH, suffix=""):
    """ helper to reassign scores back into Datasets, after you have finalized the
    params for each dset-model combo
    - suffix, for naming, usually this model class.
    """

    # Assign modeling results back into Dataset
    for L in ListBMH:
        H = L["H"]
        H.results_to_dataset(suffix=suffix)


def bmh_save(SDIR, Dlist, model_class, GROUPING_LEVELS, ListH, train_or_test):
    """ 
    NOTE:
    - Only compatible with 
    cross_dataset_model_wrapper_params
    """
    import os
    import pickle

    SAVEDICT = {}
    SAVEDICT["Dlist"] = Dlist
    SAVEDICT["model_class"] = model_class
    SAVEDICT["GROUPING_LEVELS"] = GROUPING_LEVELS
    SAVEDICT["ListH_statedicts"] = [H.extract_state() for H in ListH]


    path = f"{SDIR}/BHM_SAVEDICT-{model_class}-{'_'.join(GROUPING_LEVELS)}-{train_or_test}.pkl"    
    with open(path, "wb") as f:
        pickle.dump(SAVEDICT, f) 

def bmh_load(SDIR, model_class, GROUPING_LEVELS, train_or_test):
    """ 
    e.g, SDIR = '/data2/analyses/main/model_comp/planner/pilot-210719_210222/Red-lines5'
    """
    import os
    import pickle
    path = f"{SDIR}/BHM_SAVEDICT-{model_class}-{'_'.join(GROUPING_LEVELS)}-{train_or_test}.pkl"    

    with open(path, "rb") as f:
        SAVEDICT = pickle.load(f) 
    
    # 1) Extract skeleton
    Dlist = SAVEDICT["Dlist"]
    model_class = SAVEDICT["model_class"]
    GROUPING_LEVELS = SAVEDICT["GROUPING_LEVELS"]
    ListBMH, list_dsets, ListH= cross_dataset_model_wrapper_params(Dlist, model_class, GROUPING_LEVELS)

    # update trained params
    for H, statedict in zip(ListH, SAVEDICT["ListH_statedicts"]):
        list_modname = statedict["prior_params"].keys()
        for modname in list_modname:
            H.params_prior_set(modname, statedict["prior_params"][modname])

    return ListBMH, list_dsets, ListH