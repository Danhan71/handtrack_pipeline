from .modelexpt import *
from pythonlib.drawmodel.taskmodel import *

class ProbedatTaskmodel():
    """ integrates probedat with Taskmodel
    Taskmodel is optional. 
    """
    def __init__(self, PROBEDAT, metadat=None):
        self.Probedat = PROBEDAT
        self.Metadat = metadat
        self.ProbedatPandas = None
        self.ListFeatureNames = []
        self.preprocess() 


    ## ========== PREPROCESSING STUFF
    def preprocess(self):
        # --- 1) assign each trial a code.
        for t in self.getIndsTrials():
            p = self.t(t)
            trialcode = f"{p['date']}-{p['session']}-{p['trial']}"
            self.Probedat[t]["trialcode"] = trialcode

        col_names = {
            "ErrorCode":"getTrialsErrorCode",
            "IsAbort":"getTrialsIsAbort"
        }
        print("[preprocess] adding basic columns")
        self.pandasAddBasicColumns(col_names)
        print("[preprocess] adding computed columns")
        self._pandasAddComputedColumns()
        
    def _pandasAddComputedColumns(self):
        """ Add set of recomputed columns
        Probably shoudl have extracted when gotten probedat, but just recompute here
        """

        # 1) For each feature, also extract the "rescaled" value (between 0 and 1).
        # Previusly only extracted the raw value.
        print("Extracting feature list")
        feature_list = self.extract_feature_list(False) 
        print("Done")
        for f in feature_list:
            # define extraction function
            def F(fd, trial):
                O = getTrialsOutcomesWrapper(fd, trial)
                if f in O["beh_evaluation"]["output"].keys():
                    return O["beh_evaluation"]["output"][f]["rescale"][0][0]
                else:
                    return np.nan
            self.pandasAddBasicColumns({f"{f}_rescaled":F})
            self.ListFeatureNames.append(f)

    def clean(self):
        """ various cleaning steps
        returns a modified P object.
        """

        print("Running P.clean() ... ")
        # Remove trials with no strokes.
        # -- track n strokes.
        strokes_list = self.getTrialsHelper("getTrialsStrokesByPeanuts", "all")
        self.pandas()["nstrokes"] = [len(s) for s in strokes_list]
        # -- keep only if >0 strokes (modify self)
        nstrokes_keep = [i for i in self.pandas()["nstrokes"].unique() if i!=0]
        # nstrokes_keep = [4]
        inds = self.filterPandas({"nstrokes":nstrokes_keep}, return_indices=True)
        P = self.subsampleProbedat(inds)

        # Make names actually unique
        P.rename_unique_tasks_good()
        return P

    def rename_unique_tasks_good(self, VERSION = 2):
        """ Solving issue where tasknames are not unique in utils method.
        Hhere aken from TaskClass method. shoiuld ideally extrac tthat maehtod outside of class.
        PARAMS;
        - VERSION, 2 for new version that matches Dataset
        RETURNS:
        - modifies unique_task_name in self.pandas(), to append an additional hash number
        """
        from pythonlib.tools.pandastools import applyFunctionToAllRows
            
        self.pandas() # Genrate pandas

        if VERSION==2:
            self.pandasAddBasicColumns({"unique_task_name":getTrialsUniqueTasknameGood}, if_exists="overwrite")
        elif VERSION==1:
            # Need strokes
            # self.pandasAddBasicColumns({"strokes_beh":getTrialsStrokesByPeanuts}, if_exists="overwrite")
            self.pandasAddBasicColumns({"strokes_task":getTrialsTaskAsStrokes}, if_exists="overwrite")

            # This taken directly from TaskClass
            def rehash(strokes):
                nhash = 6
                MIN = 1000

                # Collect each x,y coordinate, and flatten it into vals.
                vals = []
                for S in strokes:
                    for SS in S:
                        vals.extend(SS)

                vals = np.asarray(vals)
                # vals = vals+MIN # so that is positive. taking abs along not good enough, since suffers if task is symmetric.

                # Take product of sum of first and second halves.
                # NOTE: checked that np.product is bad - it blows up.
                # do this splitting thing so that takes into account sequence.
                tmp1 = np.sum(vals[0::4])
                tmp2 = np.sum(vals[1::4])
                tmp3 = np.sum(vals[2::4])
                tmp4 = np.sum(vals[3::4])

                # rescale to 1
                # otherwise some really large, some small.
                # divie by 10 to make sure they are quite large.
                tmp1 = tmp1/np.max([np.floor(tmp1)/10, 1])
                tmp2 = tmp2/np.max([np.floor(tmp2)/10, 1])
                tmp3 = tmp3/np.max([np.floor(tmp3)/10, 1])
                tmp4 = tmp4/np.max([np.floor(tmp4)/10, 1])


                # Take only digits after decimal pt.
                tmp = tmp1*tmp2*tmp3*tmp4
                # print(tmp)
                tmp = tmp-np.floor(tmp)
                tmp = str(tmp)
                # print(tmp)
                # assert False
                _hash = tmp[2:nhash+2]
                return _hash

            # Append this new hash to each trials' already existing taskname
            def F(x):
                strokes = x["strokes_task"]
                code = rehash(strokes)
                return f"{x['unique_task_name']}-{code}"
            self.ProbedatPandas = applyFunctionToAllRows(self.ProbedatPandas, F, "unique_task_name")
        else:
            assert False


    def clean_remove_abort_trials(self):
        """
        Removes trials that ended on abort.
        RETURNS:
        - Either:
        --- Returns a new ProbeDat instance, without modifying this one.
        --- None, if no data.
        """
        inds = self.filterPandas({"IsAbort":[False]}, return_indices=True)
        return self.subsampleProbedat(inds)

    ## =========== HELPERS
    def t(self, ind, ignore=["filedata", "fd"]):
        """ returns dict for this trial, excluding filedata,.
        whcih is too big and makes things slow"""
        return {k:v for k,v in self.Probedat[ind].items() if k not in ignore}

    def fd(self, ind):
        """ return filedata for this index (NOT same as trial)"""
        return self.Probedat[ind]["filedata"]

    def fd_trial(self, ind):
        """ for index, get (filedata, trial)
        """
        return (self.fd(ind), self.t(ind)["trial"])

    def idxlist(self, idxs):
        """ helper, to standardize index list. Always returns list of inds.
        - idxs are indices in Probedat - NOT the asme thing as trial
        --- <int> , then returns that int in a list [int]
        --- <list> , then returns that list
        --- "all", then all inds, returned in a list (0,1, ...
        """
        if isinstance(idxs, int):
            return [idxs]
        elif isinstance(idxs, list):
            return idxs
        elif idxs=="all":
            return self.getIndsTrials() # 0, 1, 1...
        else:
            print(idxs)
            assert False


    def pandasAddThings(self, DF):
        """ extra things to add to dataframe
        (Preprocessing, mainly)."""
        from pythonlib.tools.pandastools import applyFunctionToAllRows

        assert len(DF)>0, "empty dataframe"

        # == also convert times to "faket imes, day.timewithinday"
        if "tval" in DF.columns:
            def F(x):
                return np.floor(x["tval"])+0.6
            DF = applyFunctionToAllRows(DF, F, newcolname="tvalday")

        # == also note if in summay days
        if self.Metadat is not None:
            datestokeep = self.Metadat["dates_for_summary"]
            def F(x):
                if len(datestokeep)>0:
                    return x["date"] in datestokeep
                else:
                    return True
            DF = applyFunctionToAllRows(DF, F, newcolname="keepforsummary")
        
            if self.Metadat["expt"]=="lines5":
                # == lines5 things
                # === flip score around 0 depending on epoch
                def F(x):
                    if x["epoch"]==1:
                        return x["modelcomp"]
                    elif x["epoch"]==2:
                        return -x["modelcomp"]
                    else:
                        print(x["epoch"])
                        assert False, "which eopch?"
                DF = applyFunctionToAllRows(DF, F, newcolname="modelcompflipped")
            
            # === train or test
            task_train_test = self.Metadat["task_train_test"]
            # print(self.Metadat)
            # # print(DF["kind"].value_counts())
            # print(DF)
            F = lambda x:task_train_test[x["kind"]]
            DF = applyFunctionToAllRows(DF, F, newcolname="traintest")

        # === convenient...
        if "hausdorff" in DF.columns:
            def F(x):
                return -x["hausdorff"]
            DF = applyFunctionToAllRows(DF, F, newcolname="hausdorff_positive")


        return DF


    def getTrialsHelper(self, funcname, idxs):
        """ applies funcname (e.g., getTrialsFix(fd, t)) to 
        idxs. 
        - funcname, {string, func handle}
        --- if is a string, exactly the name of thing to call
        needs to take in funcname(filedata, trial)
        --- if func handle, then must be funcname (filedata, trial)
        - idxs are indices in Probedat - NOT the asme thing as trial
        --- <int> , then that single trial
        --- <list> , then list of trials.
        --- "all", then all, returned in a list
        NOTE: output is always a list, len num trials taken
        """
        from tools import utils

        # if isinstance(idxs, int):
        #     return getattr(utils, funcname)(*self.fd_trial(idxs))
        # elif isinstance(idxs, list):
        #     return [getattr(utils, funcname)(*self.fd_trial(i)) for i in idxs]
        # elif idxs=="all":
        #     idxs = self.getIndsTrials() # 0, 1, 1...
        #     return self.getTrialsHelper(funcname, idxs)
        # else:
        #     print(idxs)
        #     assert False

        idxs = self.idxlist(idxs)
        if isinstance(funcname, str):
            return [getattr(utils, funcname)(*self.fd_trial(i)) for i in idxs]
        else:
            # [print(self.fd_trial(i)[1]) for i in idxs]
            # assert False
            return [funcname(*self.fd_trial(i)) for i in idxs]


    ## ========== DIFFERENT VIEWS OF PROBEDAT
    def pandas(self, reextract=False):
        """ uses cached version, so only runs this once.
        Can force to rerun
        """
        if self.ProbedatPandas is None:
            self.ProbedatPandas = self.asPandas()
        
        if reextract:
            self.ProbedatPandas = self.asPandas()

        return self.ProbedatPandas

    def pandasAddBasicColumns(self, col_names=None, if_exists = "error_if_different"):
        """ always run this when generating pandas.
        adds commonly used things, like block num, etc.
        - col_names, dict, <colname>:<funcname>, leave as 
        None to add the basic defaults.
        """

        # name: function
        if col_names is None:
            col_names = {
            "block":"getTrialsBlock",
            "bloque":"getTrialsBloque",
            "blokk":"getTrialsBlokk"
            }

        # 1) extract vals
        cols_to_add = {}
        for k, v in col_names.items():
            cols_to_add[k] = self.getTrialsHelper(v, "all")

        # 2) add columns
        self.pandasAddColumns(cols_to_add, if_exists=if_exists)

        print("Done, added following colunms to self.pandas():")
        print(cols_to_add.keys())

        # Add pntu size. do here since ned to stack arrays.
        pnuts = self.getTrialsHelper("getTrialsPeanutSampCollisExt", "all")
        pnuts = np.stack(pnuts)
        Pp = self.pandas()
        Pp["pnut_ext"] = pnuts



    def pandasAddColumns(self, cols_to_add, if_exists="error"):
        """ append one or more new columnt to self.ProbedatPandas.
        - cols_to_add, dict, name:vals, 
        where each item must be same length as dataframe.
        Will add this as a column
        - if_exists, string:
        --- "error", then error if column already exists. if False,
        --- "ignore", then doesnt add column, silent
        --- "overwrite", then overwrites
        --- "error_if_different", then error if different from whats arleady tehre, if
        is same, then ignores.
        - Modifies in place.
        """

        # make sure actually have self.Pandas
        for k, v in cols_to_add.items():
            assert not isinstance(v, str), "maybe you are looking for pandasAddBasicColumns?"
            if k in self.pandas().columns:
                if if_exists=="error":
                    assert False, f"youre trying to add a volumn ({k}) that already exists"
                elif if_exists=="ignore":
                    print(f"skipping volumn ({k}) that already exists")
                    continue
                elif if_exists=="overwrite":
                    print(f"ioverwriting volumn ({k}) that already exists")
                elif if_exists=="error_if_different":
                    assert all(v==self.pandas()[k].values), f"volumn ({k})  already exists, error since values differ from what you try to addd"

            assert len(v)==len(self.pandas()), "youre tyring to add a column diff length from dataframe"
            self.pandas()[k] = v


    def asPandas(self, cols_to_add = None):
        """ convert to apandas daframt. makes sure to not include 
        filedata, since too larghe
        - cols_to_add, dict, where each item must be same length as dataframe.
        Will add this as a column
        """
        import pandas as pd
        tmp = [self.t(t) for t in self.getIndsTrials()]
        DF = pd.DataFrame(tmp)

        DF = self.pandasAddThings(DF)

        if cols_to_add is not None:
            for k, v in cols_to_add.items():
                assert k not in DF.columns, "youre trying to add a volumn that already exists"
                assert len(v)==len(DF), "youre tyring to add a column diff length from dataframe"
                DF[k] = v

        return DF

    def asListOfDicts(self, exclude_filedata=True):
        """ returns in list of dicts, where list is len num trials.
        excludes fd by default.
        """
        trials = self.getIndsTrials()
        if exclude_filedata is True:
            return [self.t(n) for n in trials]
        else:
            return [self.t(n, ignore = []) for n in trials]


    ### get stuff
    def getIndsTrials(self):
        """ just idnices, 0, 1, 2, ...
        NOT original trial numbers"""
        return list(range(len(self.Probedat)))

    def getTrialNums(self):
        """ list of trial numbers.
        these are accurate and can be paired with fd
        """
        return [P["trial"] for P in self.Probedat]

    def getList(self, item):
        """ get list, length of num trials, for
        the item as key"""
        return [self.t(t)[item] for t in self.getIndsTrials()]
        

    # ==== EXTRACT THINGS
    def extractBlockParams(self):
        """ get summary of *DEFAULT* blockparams (inclyuding blockparams anbd runparams)
        across all data. 
        RETURNS:
        - BlockParamsByDateSessBlock, dict of (date,sess,block):inner_dict, where inner_dict is
        inner_dict["blockparams"] = blockparams for this block
        inner_dict["trialcodes_included"] = list of str trialcodes that use this blcokparams (useful as sanityc hceck when load this)
        NOTE: 
        - again, these are the defaut params, not hotkey updated.
        - will only inlcude blocks that actually have data in P
        """
        from pythonlib.tools.checktools import check_objects_identical

        # Collect blockparams (defualts) into a dict
        BlockParamsByDateSessBlock = {}

        for i in self.getIndsTrials():
            fd, trial = self.fd_trial(i)
            BP = getTrialsBlockParamsHotkeyUpdated(fd, trial, default_blockparams=True)
            block = getTrialsBlock(fd, trial)
            
            date = self.pandas().iloc[i]["date"]
            sess = self.pandas().iloc[i]["session"]
            trialcode = self.pandas().iloc[i]["trialcode"]

            # Sanity checks
            assert block==self.pandas().iloc[i]["block"]
            assert trial == self.pandas().iloc[i]["trial"]
            assert f"{date}-{sess}-{trial}" == trialcode
            
            # Save in output dict
            key = (date, sess, block)
            # print(trial)
            if key in BlockParamsByDateSessBlock.keys():
                # 1) add this trialcode to inlcuded
                BlockParamsByDateSessBlock[key]["trialcodes_included"].append(trialcode)
                
                # 2) confirm that BP are idenctal
                if not check_objects_identical(BP, BlockParamsByDateSessBlock[key]["blockparams"]):
                    print("----")
                    print(BP.keys())
                    print("----")
                    print(BlockParamsByDateSessBlock[key]["blockparams"].keys())
                    for k in BP.keys():
                        if not check_objects_identical(BP[k], BlockParamsByDateSessBlock[key]["blockparams"][k]):
                            for kk in BP[k].keys():
                                if not check_objects_identical(BP[k][kk], BlockParamsByDateSessBlock[key]["blockparams"][k][kk]):
                                    print(k, kk)
                                    print("----")
                                    print(BP[k][kk], type(BP[k][kk]))
                                    print("----")
                                    print(BlockParamsByDateSessBlock[key]["blockparams"][k][kk], type(BlockParamsByDateSessBlock[key]["blockparams"][k][kk]))
                                    print("ASDASD", check_objects_identical(BP[k][kk], BlockParamsByDateSessBlock[key]["blockparams"][k][kk]))
                                # check_objects_identical(BP, BlockParamsByDateSessBlock[key]["blockparams"], PRINT=True)
                                    assert False
                    assert False

                # if True:
                #     BPold = BlockParamsByDateSessBlock[key]["blockparams"]
                #     # print(1, BP.keys())
                #     # print(2, BPold.keys())
                #     # print(type(BP))
                #     # print(type(BlockParamsByDateSessBlock[key]["blockparams"]))
                #     # print(BP==BlockParamsByDateSessBlock[key]["blockparams"])
                #     # for k in BP.keys():
                #     #     # print(k)
                #     #     pass
                #     #     # assert BP[k].keys() == BPold[k].keys()
                #     #     for kk in BP[k].keys():
                #     #         pass
                #     #         # print("-", kk)
                #     #         # print(BP[k][kk])
                #     #         # print(BPold[k][kk])

                #     #         # print(BP[k][kk]==BPold[k][kk])
                #     assert BP==BlockParamsByDateSessBlock[key]["blockparams"]

            else:
                # initialize
                BlockParamsByDateSessBlock[key] = {}
                BlockParamsByDateSessBlock[key]["trialcodes_included"] = [trialcode]
                BlockParamsByDateSessBlock[key]["blockparams"] = BP
                
        return BlockParamsByDateSessBlock


    def extractStrokes(self):
        """ get strokeslist for beh and task.
        length is num trials.
        """
        strokes_beh = []
        strokes_task = []
        for p in self.Probedat:
            fd = p["filedata"]
            t = p["trial"]
            # S = getTrialsStrokesByPeanuts(fd, t)
            strokes_beh.append(getTrialsStrokesByPeanuts(fd, t))
            strokes_task.append(getTrialsTaskAsStrokes(fd, t))
        return strokes_beh, strokes_task

    def extractTasksAsClass(self, inds):
        """ returns list of Task Class objects, correspodning
        to inds.
        - inds, list of indices into Probedat. (not trials), or, whatever
        could pass into getTrialsHelper, namely:
        --- <int> , then that single trial
        --- <list> , then list of trials.
        --- "all", then all, returned in a list
        RETURN:
        TaskClassList, list where each element is TaskClass(task), 
        same length as inds.
        """
        from pythonlib.drawmodel.tasks import TaskClass

        # extract all tasks
        tasklist = self.getTrialsHelper("getTrialsTask", inds)

        # sketchpads
        spadlist = self.getTrialsHelper("getTrialsSketchpad", inds)

        for spad in spadlist:
            assert spad is not None, "likely is old version of task where didnt specify sketchpad (e.g, Luca) - see utils.getTrialsSketchpad, fix it there"

        # fixpos
        fixlist = self.getTrialsHelper("getTrialsFix", inds)
        fixlist = [fix["fixpos_pixels"] for fix in fixlist]

        # extract strokes for each task
        taskstrokeslist = self.getTrialsHelper("getTrialsTaskAsStrokes",inds)

        # Put all into datk dict
        for task, strokes, spad, fix in zip(tasklist, taskstrokeslist, spadlist, fixlist):
            task["strokes"] = strokes
            task["sketchpad"] = spad
            task["fixpos"] = fix

        # for each task, convert to TaskClass
        TaskClassList = [TaskClass(t) for t in tasklist]

        return TaskClassList




    ## === PROBEDAT STUFF
    # def addFeature(self,)

    def subsampleProbedat(self, idxs):
        """ outputs a new Probedat object, which is using only 
        idxs.
        - inds, list of indices into Probedat. (not trials), or, whatever
        could pass into getTrialsHelper, namely:
        --- <int> , then that single trial
        --- <list> , then list of trials.
        --- "all", then all, returned in a list
        """

        idxs = self.idxlist(idxs)
        if len(idxs)==0:
            return None
        PD = [self.Probedat[i] for i in idxs]
        return ProbedatTaskmodel(PD, self.Metadat)

    def filterByBehPerformance(self, filterparams):
        """ to clean up trials by performance, removing high noise.
        - filterparams, dict, with keys:
        --- hausdorff_filter, if True, then throws out tirlas lower than
        a thyreshold hd comapred to task (in neg values, so highly negative is
        bad
        --- hausdorff_filter_prctile, if passed in, then will use this percentiel
        across trials to determine the minimum hd below which trials are
        thrown out.
        RETURNS:
        - a new Probedat class instance, without modifying this one.
        NOTE:
        - by defult throws out trials without any beahvior.
        """
        
        if len(filterparams)==0:
            return self.Probedat   
        Pp = self.pandas()

        # 1) Find minimum hausdorff, all trials lower than this will be excluded.
        if filterparams["hausdorff_filter"]:
            # currently only using percential method for geting therhsod..
            thresh_hausdorff = np.percentile(Pp["hausdorff"].values, [filterparams["hausdorff_filter_prctile"]])[0]

        ProbedatFiltered = []
        norig = len(self.Probedat)
        for i in self.getIndsTrials():
        #     print(i)
            idx = self.fd_trial(i)
            
            strokes_beh = getTrialsStrokesByPeanuts(*idx)
            # strokes_task = getTrialsTaskAsStrokes(*idx)
            out = getTrialsOutcomesWrapper(*idx)
            
            # Decide if keep this datapt
            if len(strokes_beh)==0:
                print("skipping, strokes len 0")
                print(i)
                continue
            if "hausdorff_filter" in filterparams:
                if Pp["hausdorff"].values[i]<thresh_hausdorff:
                    print("skipping, didnt pass min hausdorff")
                    print(i)
                    continue
                
            ProbedatFiltered.append(self.Probedat[i])
        nnew = len(ProbedatFiltered)
        print(f"Done filtering, len, {norig} reduced to {nnew}")

        return ProbedatFiltered
    


    def filterProbedat(self, filtdict, modify_in_place=True):
        """ only keep subset of probedat, based 
        on filtdict.
        - Can choose whethe rto Modifies in place,
        - always returns self.Probedat"""
        from pythonlib.tools.dicttools import filterDict
        print(f"starting len ofprobedat: {len(self.Probedat)}")

        P = filterDict(self.Probedat, filtdict)
        print(f"after filtering: {len(P)}")
        if modify_in_place:
            print("Modified in place: self.Probedat")
            self.Probedat = P
        else:
            print("did not modify self.Probedat")

        # reextract pandas
        self.pandas(reextract=True)
        return P

    def filterPandas(self, filtdict, return_indices=False):
        """
        doesnt modify in place.
        returns filtered df, or if flagged, returns
        indices
        """
        from pythonlib.tools.pandastools import filterPandas
        # traintest = ["test"]
        # random_task = [False]
        # filtdict = {"traintest":traintest, "random_task":random_task}
        return filterPandas(self.pandas(), filtdict, return_indices=return_indices)



    ## === MODEL 

    def pd2strokes(self, probedat, return_all_trials=False):
        """ if return_all_trials, then some might be empty, if
        no behavior found.
        """

        strokes, tasks, fix, strokes_task = [[], [], [], []]
        for p in probedat:
            fd = p["filedata"]
            t = p["trial"]
            S = getTrialsStrokesByPeanuts(fd, t)
            if return_all_trials==False and len(S)==0:
                continue
            strokes.append(S)
            tasks.append(getTrialsTask(fd, t))
            fix.append(getTrialsFix(fd, t)["fixpos_pixels"])
            strokes_task.append(getTrialsTaskAsStrokes(fd, t))
        for s, t, f in zip(strokes_task, tasks, fix):
            t["strokes"] = s
            t["fixpos"] = f
        return strokes, tasks
    
    def transferModelFrom(self, datmodel):
        """ transfer model params from another model
        (datmodel."""
        assert False, "Not coded"

    def getParams(self, priorver="distance_travel", parse_ver="permutations",
                  chunkmodel = None, name="test", posterior_ver="weighted",
                  likeliver="segments"):
        """ default + modified model params
        """
        from pythonlib.drawmodel.taskmodel import getParamsWrapper

        PARAMS_DATA, PARAMS_MODEL = getParamsWrapper(priorver, parse_ver, chunkmodel,
            name, posterior_ver, likeliver)

        return PARAMS_DATA, PARAMS_MODEL


    def applyModel(self, params_data, params_model):
        from pythonlib.drawmodel.taskmodel import Model, Dataset
        
        strokes, tasks = self.pd2strokes(self.Probedat) 
        mod = Model(params_model)
        data = Dataset(strokes, tasks, PARAMS=params_data)
        data.applyModel(mod)
        
        self.Datamodel = data
        print("built and applied model (but did not optimize params)")

    def fitModel(self):
        assert False, "not coded"    

    def assignTaskmodelResults(self, ProbedatTaskmodelDicts, 
        scorefun=None):
        """ runs taskmodel on all trials, reassigns a 
        modelcomp offline score for each element
        - the model should be a Dataset object with 
        parameters already finalezed. here no fitting,
        just applying the model to score.
        - ProbedatTaskmodelDicts is dict, where each item is 
        (name of model, ProbedatTaskmodel). will assume each 
        is same Probedat, but different model score.
        - scorefun is tuple of (name, function), where function(p) 
        returns a scalar score, and p is a single modelscire item (after reassiging
        all model scores). This is useful for model comaprison, e.g,.,
            def sf(m):
                a = m["3line"]
                b = m["linePlusL"]
                return 2*((a/(a+b))-0.5)
        - Note: if give scorefun, then it will be assigned back into probedat
        (in addition to assiging to Modelscores)
        """
        PDdict = ProbedatTaskmodelDicts

        # 1) chcek that lengths match
        for D in PDdict.values():
            assert len(D.Datamodel.trials)==len(self.Probedat)
        print("good, len of dataset and probedat match")

        # 2) get modelscores, same length as probedat
        modelscores = []
        for i, p in enumerate(self.Probedat):
            modelscores.append({})
            modelscores[-1]["online_modelcomp"] = p["modelcomp"]
            modelscores[-1]["epoch"] = p["epoch"]

            for name, PD in PDdict.items():
                modelscores[-1][name] = PD.Datamodel.trials[i]["posterior"]

        # 3) assign summary score if desried
        if scorefun is not None:
            for p, m in zip(self.Probedat, modelscores):
                m[scorefun[0]] = scorefun[1](m)
                p[scorefun[0]] = m[scorefun[0]]
            print(f"added sumary score called {scorefun[0]}")
        
        self.Modelscores = modelscores

    ### ============== PLOTTING STUFF
    def extract_feature_list(self, include_extras=True):
        # Useful for plotting, gets features that exist in dataset.

        featurelist = []
        blocks_gotten = [] # only check a single trial per block..
        for i in range(len(self.Probedat)):
            print("extract_feature_list", "trial", i)
            blocknum = getTrialsBlock(*self.fd_trial(i))
            if blocknum not in blocks_gotten:
                # NOTE: This is not strictly correct, as same blocknum might be different on differnt days 
                # But is usually correct.
                featurelist.extend(getMultTrialsBehEvalFeatures(self.fd(i))) # across all trials, to extract entire set of features.
                blocks_gotten.append(blocknum)
        featurelist = list(set(featurelist))
        if include_extras:
            featurelist.extend(["score_final", "binary_evaluation", "rew_total"])
        featurelist = [f for f in featurelist if f in self.pandas().columns]
        return featurelist

    def extract_feature_list_frompandas(self, rescaled=False, include_extras=True):
        """ Extract the features quicly.
        PARAMS:
        - rescaled, bool, then gets feature columsn for rescaled between 0 and 1, .e,g, 
        "hausdorff_rescaled"
        - include_extras, bool, then includes things that arent features, but are useful to
        plot alongside them: score_final", "binary_evaluation", "rew_total
        RETURN:
        - featurelist, list of str.
        """

        featurelist = [f for f in self.ListFeatureNames]
        if rescaled:
            featurelist = [f"{f}_rescaled" for f in featurelist]
        if include_extras:
            featurelist.extend(["score_final", "binary_evaluation", "rew_total"])
        return featurelist


    def get_beheval_params_adaptive(self):
        """ Get adaptive beh eval params, such as hard limits for automatic upating
        TODO: currenyl is hacky, takes the middle trial, assuming this doesnt change. 
        Ideally shld search thru all blocks, returning indexed by block.
        But uslally this work.
        """
        n = len(self.Probedat)
        nthis = int(n/2)
        params_adaptive = self.getTrialsHelper("getTrialsBehEvalAdaptiveParams", nthis)
        return params_adaptive[0] # return just this trial



    def get_beheval_params_minmax(self, feature):
        """ 
        For this feature, get its xmin and xmax across all trials. Accounts for possible 
        adaptive change over trials.
        RETURNS:
        - np array, (ntrials, 2), where columns are (xmin, xmax). Any rows without
        this feature are replaced with nans.
        """
        
        def F(fd, t):
            x = getTrialsBehEvaluationParams(fd, t)
            if feature in x.keys():
                return np.asarray([x[feature]["xmin"], x[feature]["xmax"]])
            else:
                return np.array([[np.nan, np.nan]]).T
        return np.stack(self.getTrialsHelper(F, "all")).squeeze()

    def plotModelscores(self):
        """ distribution of model scores, including those
        extracted using assignTaskmodelResults. 
        pairplot
        """
        MS = pd.DataFrame(self.Modelscores)
        v = [k for k in self.Modelscores[0].keys() if k not in ["epoch"]]
        fig = sns.pairplot(data=MS, vars = v, hue="epoch")
        return fig


    def plotMultTrials(self, idxs):
        """ idxs is list of trials
        """
        from tools.plots import plotMultTrialsSimple2

        # 1) get matching list of filedata
        fdlist = [self.fd(i) for i in idxs]
        triallist = [self.t(i)["trial"] for i in idxs]

        fig = plotMultTrialsSimple2(fdlist, triallist)
        return fig



    def plotOverviewBlokks(self, SDIR, n_task_at_boundary=8):
        """
        - Bunch of plots, looking at block trnaistions, seaprated into distinct
        blokks. 
        - Useful for experiments where training using adaptive blocks
        """

        from tools.dayanalysis import goodfig
        import os
        from pythonlib.tools.timeseriestools import getChangePoints

        os.makedirs(SDIR, exist_ok=True)

        # == 1) extract the features you want to plot
        col_names = {
            "ErrorCode":"getTrialsErrorCode",
            "IsAbort":"getTrialsIsAbort"
        }
        self.pandasAddBasicColumns(col_names)
        self.pandasAddBasicColumns()

        ##################################
        sesslist = self.pandas()["session"].unique().tolist()
        # print(sesslist)
        for s in sesslist:

            # filter to a new probedat
            # print(s)
            # # print(len(PD))
            # print(self.pandas())
            PD = self.filterProbedat({"session":[s]}, modify_in_place=False)
            assert len(PD)>0, "overwrote PD?"
            Psub = ProbedatTaskmodel(PD)
            col_names = {
                "ErrorCode":"getTrialsErrorCode",
                "IsAbort":"getTrialsIsAbort"
            }
            Psub.pandasAddBasicColumns(col_names)
            Psub.pandasAddBasicColumns()

            Pp = Psub.pandas()

            # Pp = Ppmain[Ppmain["session"]==s].reset_index(drop=True)

            if True:

                # == Plot trial error codes, over time, and distributions
                vals = Pp["ErrorCode"].values
                fig, ax = plt.subplots()
                ax.hist(vals)
                ax.set_xlabel("error code")
                fig.savefig(f"{SDIR}/hist_trialerrors-sess{s}.pdf")
                plt.close("all")

                # == Plot overall timecourse across all trials
                # idxs = self.getIndsTrials()
                idxs = Pp.index.tolist()
                fig, ax = plt.subplots(figsize=(30*len(idxs)/150, 7))
                plotTimecourseGeneral(ax, Pp, idxs)
                goodfig(ax, Pp)
                fig.savefig(f"{SDIR}/timecourse-entire-sess{s}.pdf")
                plt.close("all")

                # == Plot, algined to blokk transitions
                idx_onsets = getChangePoints(Pp["blokk"].values)
                if len(idx_onsets)>0:
                    fig = plotAlignedToIndex(Pp, idx_onsets, N = 10, yver="behscore")
                    fig.savefig(f"{SDIR}/timecourse-blokk_transitions_all-sess{s}.pdf")
                    plt.close("all")

                    # == Plot overlaid, one plot for each kind of block transition
                    # Get dict: (bk1, bk2)--> index at start of bk2
                    def _block_transition_type(Pp, idx):
                        """ idx indexst in Pp.
                        returns [bk1, bk2], which are blocks in
                        index idx-1 and idx. i..e, assumes idx is the
                        first indiex for a new block
                        """
                        bk1 = Pp["block"].values[idx-1]
                        bk2 = Pp["block"].values[idx]
                        return bk1, bk2

                    print("TODO: make module to save block transitions")
                    block_transitions_by_type = {}
                    for i in idx_onsets:
                        bk1, bk2 = _block_transition_type(Pp, i)
                        if (bk1, bk2) not in block_transitions_by_type.keys():
                            block_transitions_by_type[(bk1, bk2)] = []
                        block_transitions_by_type[(bk1, bk2)].append(i)
                            
                    bklist = list(block_transitions_by_type.keys())
                    bklist.sort()

                    for bktype in bklist:
                        idxlist = block_transitions_by_type[bktype]
                        fig = plotAlignedToIndex(Pp, idxlist, plotorg="overlaid")
                        fig.savefig(f"{SDIR}/timecourse-blokk_transitions-{bktype[0]}_{bktype[1]}-sess{s}.pdf")
                        plt.close("all")

            # == PLOT ACTUAL BEAHVIOR, for trials aligned 
            # === PLOT BEHAVIOR TRIALS, ALIGNED TO INDEX
            # 1) Collect information about each blokk, based on blokk transitions.
            # blokk transitions.
            BlockDict = []

            idx_onsets = getChangePoints(Pp["blokk"].values)
            if len(idx_onsets)>0:
                idx_onsets = np.r_[0, idx_onsets]
                for i, ii in zip(idx_onsets[:-1], idx_onsets[1:]):
                    on = i
                    off = ii-1
                    
                    thisblock = Pp["block"].values[off]
                    thisblokk = Pp["blokk"].values[off]
                    
                    assert len(set(Pp["block"][i:ii].values))==1
                    assert len(set(Pp["blokk"][i:ii].values))==1
                    
                    BlockDict.append({
                        "blokk":thisblokk,
                        "block":thisblock,
                        "idx_on":on,
                        "idx_off":off
                    })

                
            # 2) For each blokk transition, plot N trials pre and post
            from pythonlib.tools.plottools import saveMultToPDF
            
            for i, (bkk1, bkk2) in enumerate(zip(BlockDict[:-1], BlockDict[1:])):
                
                idx_pre = list(range(bkk1["idx_on"], bkk1["idx_off"]+1))
                idx_post = list(range(bkk2["idx_on"], bkk2["idx_off"]+1))
                
                # pull out N flanking trials. 
                idx_pre = idx_pre[-n_task_at_boundary:]
                idx_post = idx_post[:n_task_at_boundary]
                
                # plot those trials
                fig1 = Psub.plotMultTrials(idx_pre)
                fig2 = Psub.plotMultTrials(idx_post)
                
                # save
                bk1 = bkk1["block"]
                bk2 = bkk2["block"]
                
                saveMultToPDF(f"{SDIR}/behavior_blokk_transitions-{i}-bk{bk1}_{bk2}-sess{s}", [fig1, fig2])
                plt.close("all")

    def plotOverviewFeatureByBlock(self, SDIR):
        """ distribution, for a feature, across blocks
        automaitlcaly gets all rfeatures.
        """
        Pp = self.pandas()

        from tools.utils import getMultTrialsBehEvalFeatures
        feature_list = getMultTrialsBehEvalFeatures(self.fd(0))
        feature_list += ["rew_total"]
        feature_list = [f for f in feature_list if f in Pp.columns]

        for f in feature_list:
        #     sns.catplot(data=Pp, x="block", y="rew_total")

            fig = sns.catplot(data=Pp, x="block", y=f, kind="boxen", aspect=1.5)
            fig.savefig(f"{SDIR}/feature_by_block-{f}.pdf")



    def plotOverviewTaskPerformance(self, SDIR):
        """ sort tasks by perforamnce, and plot summaries
        """

        os.makedirs(SDIR, exist_ok=True)

        # == 1) extract the features you want to plot
        col_names = {
            "ErrorCode":"getTrialsErrorCode",
            "IsAbort":"getTrialsIsAbort"
        }
        self.pandasAddBasicColumns(col_names)
        self.pandasAddBasicColumns()
        Pp = self.pandas()

        # === 1) Collect tasks
        tasklist = sorted(set(Pp["unique_task_name"]))
        taskdict = []
        for task in tasklist:
            df = Pp[Pp["unique_task_name"]==task]
            scores = df["beh_multiplier"].values
            
            taskdict.append({
                "task":task,
                "scores":scores
            })
        taskdict = sorted(taskdict, key=lambda x:np.mean(x["scores"]))

        # === 2) Plot each task in its own plot. sort tasks by score
        # for each task, plot
        featurelist = self.extract_feature_list_frompandas()
        for i, T in enumerate(taskdict):
            idxs = self.filterPandas({"unique_task_name":[T["task"]]}, return_indices=True)
            if len(idxs)<2:
                continue
            if len(idxs)>50:
                import random
                idxs = random.sample(idxs, 50)
                # assert False, "write to take care of large data. also, only take fixed tasks"
                
            fig = self.plotMultTrials(idxs)
            fig.savefig(f"{SDIR}/beh-task_{T['task']}-scorerank_{i}.pdf")

            # For each task (with minimu N trials), plot all timecourses across day (this hsould
            # alignt with the above plot #2)
            nfeats = len(featurelist)
            fig, axes = plt.subplots(nfeats, 1, sharex=True, figsize=(6, nfeats*2))
            for f, ax in zip(featurelist, axes.flatten()):
                vals = Pp.iloc[idxs][f]
                ax.plot(np.arange(len(vals)), vals, '-ok')
                ax.set_ylabel(f)
                if f==featurelist[-1]:
                    ax.set_xticks(np.arange(len(idxs)))
                    ax.set_xticklabels(idxs)
            fig.savefig(f"{SDIR}/beh-task_{T['task']}-scoreplot.pdf")

            plt.close("all")


        # === 3) Plot single overview, score acorss tasks.
        fig, ax = plt.subplots(figsize=(8, len(taskdict)*0.35))

        for row, T in enumerate(taskdict):
            task = T["task"]
            
            Ppthis = Pp[Pp["unique_task_name"]==task]
            
            # -- trial outcomes
            ecodes = Ppthis["ErrorCode"].values
            scores = Ppthis["beh_multiplier"].values
            aborts = Ppthis["IsAbort"].values
            
            y = row*np.ones_like(scores)
        #     plt.plot(scores, y, 'ok')
            ax.scatter(scores, y, c=ecodes, marker="o", cmap="jet", alpha=0.5)
            ax.plot(scores[aborts==True], y[aborts==True], "kx", alpha=1)
            
        ax.set_yticks(range(len(taskdict)))
        ax.set_yticklabels([T["task"] for T in taskdict])
        ax.set_xlabel("beh score")
        fig.savefig(f"{SDIR}/overview.pdf", bbox_inches='tight')


    def plotSummaryExptAllDays(self):
        from pythonlib.tools.snstools import rotateLabel

        Pp = self.pandas()

        ALPHA1 = 0.3
        ALPHA2 = 0.3

        fig1 = sns.catplot(x="date", y="task_stagecategory", hue="random_task", data=Pp, aspect=3, 
                           row_order=sorted(set(Pp["task_stagecategory"])), alpha=ALPHA1)
        fig2 = sns.catplot(x="date", y="task_stagecategory", hue="random_task", row="traintest", 
                           data=Pp, aspect=3, alpha=ALPHA1)

        # for fixed tasks only
        # sns.catplot(x="date", y="task", hue="task_category", row="traintest", data=SF[SF["random_task"]==False], 
        #             height=10, aspect=1, row_order=sorted(set(SF["task"])))
        H = len(list(set(Pp[Pp["random_task"]==False]["unique_task_name"])))*(1/5)
        fig3 = sns.catplot(x="date", y="unique_task_name", hue="task_stagecategory", row="traintest", data=Pp[Pp["random_task"]==False], 
                    height=H, aspect=1, )

        fig4 = sns.catplot(x="date", y="task_stagecategory", row="random_task", col="traintest", hue="taskgroup",
                         sharex=True, sharey=False, aspect=2, data=Pp, orient="v", alpha=ALPHA2)
        rotateLabel(fig4, 25)
        # fig1.savefig(f"{SAVEDIR}/overview-1.pdf")
        # fig2.savefig(f"{SAVEDIR}/overview-2.pdf")
        # fig3.savefig(f"{SAVEDIR}/overview-3.pdf")
        # fig4.savefig(f"{SAVEDIR}/overview-4.pdf")

        return fig1, fig2, fig3, fig4

    def plot_featuredists_byblock(self, savedir):
        """ plot single histogram for each block, arranged in a single plot.
        Useful for seeing change over blocks for each feature.
        """

        # Make save dir

        # Extract features
        self = self.clean()
        featurelist = self.extract_feature_list()
        Pp = self.pandas()

        # Separate by session
        for sess in Pp["session"].unique():
            dfthis = Pp[Pp["session"]==sess]
            if len(dfthis)==0:
                continue
            for f in featurelist:
                try:
                    fig1 = sns.catplot(data=dfthis, x="block", y=f, aspect=2)
                    fig2 = sns.catplot(data=dfthis, x="block", y=f, kind="boxen", aspect=2)

                    fig1.savefig(f"{savedir}/sess{sess}-{f}-1.pdf")
                    fig2.savefig(f"{savedir}/sess{sess}-{f}-2.pdf")
                    plt.close("all")
                except Exception as err:
                    # KeyError: 'y'
                    pass


    def plot_featuredists_overlying_params(self, savedir=None, savename=None):
        """ GOOD - plots histograms for each features, and overlays (1) tiemocurse of 
        xmin and xmax for beh eval and (2) hard limits. Useful for seeing if beh eval limits
        are appropriate wrt to behavior
        """

        # Initialize things
        feature_list = self.extract_feature_list()
        Pp = self.pandas()
        params_adaptive = self.get_beheval_params_adaptive()

        # Initialize figures
        ncols = 2
        nrows = int(np.ceil(len(feature_list)/ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize = (ncols*5, nrows*3))

        for feature, ax in zip(feature_list, axes.flatten()):
            try:
                ax.hist(Pp[feature], bins=20, alpha=0.5)
                ax.set_title(feature)
                ax.set_ylabel('counts')

                # Overlay actual limits
                ax2 = ax.twinx()
                ax2.set_ylabel("trial index")
                xminmax = self.get_beheval_params_minmax(feature)
                for col in [0, 1]:
                    # - what color to plot lines?
                    if col==0:
                        color = "r" # minimun
                    elif col==1:
                        color = "g" # max
                    else:
                        assert False
                    
                    # - get data
                    x = xminmax[:,col]
            #         indexes = np.unique(x, return_index=True)[1]
            #         x = np.array([x[index] for index in sorted(indexes)])
                    x = x[~np.isnan(x)]
                    if len(x)>0:
                        if False:
                            # v1, showing vertical lines. but dont know what is latest
                            for xx in x:
                                ax2.axvline(xx, color=color, alpha=0.3)
                        else:
                            # v2, showing timecourse
                            ax2.plot(x, range(len(x)), '-', color=color, alpha=0.5)
                            
                
                # Overlay hard limits
                if params_adaptive["on"]:
                    tmp = [x for x in params_adaptive["hardlimits"] if x["feature"]==feature]
                    if len(tmp)>1:
                        print(params_adaptive)
                        assert False, "how possible?"
                    elif len(tmp)==1:
                        # Then this does have hard limits
                        lim = tmp[0]["limits_low"]
                        ax.fill_betweenx(ax.get_ylim(), lim[0], lim[1], color="r", alpha=0.1)
                        lim = tmp[0]["limits_high"]
                        ax.fill_betweenx(ax.get_ylim(), lim[0], lim[1], color="g", alpha=0.1)
            except Exception as err:
                print(err)
                print("[error in plot_featuredists_overlying_params]; catching as skipping")

        if savedir:
            if savename is None:
                savename = "featuredists_overlying_params"
            fig.savefig(f"{savedir}/{savename}.pdf")

        return fig
                    
            
            
                
            


    ############################ SAVE PRINTED DATA
    def print_all_tasknames(self, savedir=None, pandasfilter=None):
        """ Print names of all tasks, sorted by both name and ntrials
        Then print to a text file (yaml)
        """
        from pythonlib.tools.expttools import writeStringsToFile, writeDictToYaml

        if pandasfilter is not None:
            Pp = self.filterPandas(pandasfilter)
        else:
            Pp = self.pandas()
        for INCLUDEABORT in [True, False]:
            for SORTBY in ["taskname", "ntrials"]:

                # Keep aborted trials?
                if INCLUDEABORT == False:
                    df = Pp[~Pp['IsAbort']]
                else:
                    df = Pp

                # Get tasks and counts
                x = df["unique_task_name"].value_counts()
                
                # Sort?
                if SORTBY=="taskname":
                    xsorted = x.sort_index(ascending=True)    
                elif SORTBY=="ntrials":
                    xsorted = x
                else:
                    assert False
                
                # Write
                out = []
                for taskname, n in zip(xsorted.index, xsorted.tolist()):
                    out.append({taskname:n})
                writeDictToYaml(out, f"{savedir}/all_tasks_sortedby{SORTBY}_includeabort{INCLUDEABORT}.yaml")


    def print_all_pnut_size(self, savedir=None):
        """ For each trial, prints pnut size. Useful
        for saving into a text file during preprocess
        PARAMS:
        - savedir, string directory, where will save yaml file.
        RETURNS:
        - outdict, list of dicts, each a trial with <feature>:score
        Also saves to <savedir>/all_feature_scores.yaml
        """
        
        # Extract peanuts
        pnuts = self.getTrialsHelper("getTrialsPeanutSampCollisExt", "all")
        pnuts = np.stack(pnuts)
        Pp = self.pandas()
        Pp["pnut_ext"] = pnuts

        # Write
        outdict = {}
        outlist = []
        outlist.append("session-trial-block: value")
        for i in range(len(Pp)):
            key = f"{Pp.iloc[i]['session']}-{Pp.iloc[i]['trial']}-{Pp.iloc[i]['block']}"
            val = f"{Pp.iloc[i]['pnut_ext']}"
            outdict[key] = val
            outlist.append(f"{key}:   {val}")
                           
        #     outlist = [key, val]
        if savedir is not None:
            from pythonlib.tools.expttools import writeStringsToFile, writeDictToYaml
            writeDictToYaml(outdict, f"{savedir}/all_pnut_size.yaml")
            # writeDictToYaml(outdict, f"{savedir}/all_pnut_size.yaml")
            writeStringsToFile(f"{savedir}/all_pnut_size.txt", outlist)
            # writeDictToYaml(outdict, f"{savedir}/all_feature_scores.yaml")

        return outdict


    def print_all_feature_scores(self, savedir=None):
        """ For each trial, prints out the score across all features. Useful
        for saving into a text file during preprocess
        PARAMS:
        - savedir, string directory, where will save yaml file.
        RETURNS:
        - outdict, list of dicts, each a trial with <feature>:score
        Also saves to <savedir>/all_feature_scores.yaml
        """

        # - Extracat all features
        feature_list = self.extract_feature_list()
        fl = ["session", "trial", "IsAbort"] # add session and trial info.
        fl.extend(feature_list)
        fl_actual_features = self.extract_feature_list_frompandas(False, False) # no extras

        def _get_line(i, f):
            # return a single line for this
            # trial (i) and feature
            value = Pp.iloc[i][f]
            if f in fl_actual_features:
                rescaled = Pp.iloc[i][f"{f}_rescaled"]
                return f" --- {value:.2f} [{rescaled:.2f}]"
            else:
                return f"{value:.2f}"

        # Extract scores for all features across all trials.
        Pp = self.pandas()
        outdict = []
        for i in range(len(Pp)):
            outdict.append(
                {f:_get_line(i,f) for f in fl}
            )

        if savedir is not None:
            from pythonlib.tools.expttools import writeDictToYaml
            writeDictToYaml(outdict, f"{savedir}/all_feature_scores.yaml")

        return outdict

    ## ==== PARSE FUNCTIONS
    def makeParseFunction(self, ver):
        parsefun = makeParseFunction(ver)
        return parsefun

    ## === FLATTEN TO STROKEDAT
    def flattenToStrokdat(self):
        """ flatten to pd dataframe where each row is one
        strok (np array)
        RETURNS:
        - Strokdat, pd dataframe. Includes reference to Probedat.
        NOTE: does not modify self.
        """
        strokeslist = self.getTrialsHelper("getTrialsStrokesByPeanuts", "all")
        Strokdat = []
        for i, strokes in enumerate(strokeslist):
            if i%500==0:
                print(i)
            for ii, strok in enumerate(strokes):
                Strokdat.append({
                    "stroknum":ii,
                    "strok":strok,
                    "index_probedat":i,
                    "Probedat":self})

        return pd.DataFrame(Strokdat)
            

    def animal(self):
        tmp = list(set(self.pandas()["animal"]))
        if len(tmp)>1: 
            print(tmp)
            assert False, "multiple animals, not sure which to take"
        return tmp[0]


    def generateDataset(self, savedir=None, savenote=None, extraction_params=None):
        """ generate Dataset object, which is a clearn version of
        data, filtered, and has many methods for doing other things. 
        in general Dataset can combine multiple probedats, if first save
        each probedat, then load all into one Dataset object
        INPUT:
        - savedir, full path. usually I save at "/data2/analyses/database".
        Leave None if not saving.
        - savenote, will add this to save path as a suffix.
        - extraction_params, things to do during extraction
        NOTE: dataset will not necessarilty be same size as Probedat, since can filter out
        bad trials.
        """
        from analysis.dataset import Probedat2Dat
        if extraction_params is None:
            extraction_params={
            "probedat_filter_params":{"hausdorff_filter":True,"hausdorff_filter_prctile":2.5},
            "pix_add_to_sketchpad_edges":20}
        # Add a few things to extraction params, useful for saving
        extraction_params["expt"] = self.Metadat["expt"]
        extraction_params["animal"] = self.animal()
        extraction_params["savedir"] = savedir
        extraction_params["savenote"] = savenote
        if savedir is None:
            save=False
        else:
            save=True

        # ==== filter trials based on behavioral criteria, to throw out noise.
        print("Filtering Probedat before generating dataset")
        ProbedatFiltered = self.filterByBehPerformance(extraction_params["probedat_filter_params"])

        # Reconstruct P
        P = ProbedatTaskmodel(ProbedatFiltered, self.Metadat)

        # === Convert Probedat to DAT
        DAT, METADAT = Probedat2Dat(P, extraction_params, save=save, keep_all_in_probedat=True)

        # === convert to dataset object
        from pythonlib.dataset.dataset import Dataset
        inputs = [
            (DAT, METADAT)
        ]
        self.Dataset = Dataset(inputs)


    ################### plots
    def plot_timecourse_pnut_size(self, sdir):
        """ Line plots of pnuts size over expt.
        """

        # Make save dir
        sdirthis = f"{sdir}"
        import os
        os.makedirs(sdirthis, exist_ok=True)

        Pp = self.pandas()

        # 1) timescourse
        fig = sns.relplot(data=Pp, x="trial", y="pnut_ext", hue="block", kind="line", row="session", aspect=2)
        plt.grid("on")
        fig.savefig(f"{sdirthis}/peanuts_timecourse_trials.pdf")

        fig = sns.relplot(data=Pp, x="trial", y="pnut_ext", hue="block", row="session", aspect=2)
        plt.grid("on")
        fig.savefig(f"{sdirthis}/peanuts_timecourse_trials_2.pdf")

        # 2) Blocks.
        fig = sns.catplot(data=Pp, x="block", y="pnut_ext", hue="block", col="session") 
        plt.grid("on")
        fig.savefig(f"{sdirthis}/peanuts_blocks.pdf")






#######################
def probedatOfflineScore(probedat, filtdict=None,
                        ploton=False, expt="lines5", return_all_models=False,
                        ):
    """ given probedat, get models and assign scores back
    - filtdict, default is using fixed tasks only. use None or empty dict
    to not filter
    - default models below are meant for expt lines5
    - return_all_models, then returns PDdict
    """
    if filtdict is None:
        filtdict = {"random_task":[False]}
    assert expt=="lines5", "have not coded models gfor other expeirmts"

    # -- 1) filter probedat
    probedat = ProbedatTaskmodel(probedat).filterProbedat(filtdict)

    # -- 2) Fit different models.
    PDdict = {}
    # for priorver, parse_ver, chunkmodel, posterior_ver, name in zip(
        # ["distance_travel","uniform","uniform"], 
        # ["permutations", "chunks", "chunks"],
        # [None, "3line", None],
        # ["weighted", "maxlikeli", "maxlikeli"],
        # ["distance_travel", "3line", "linePlusL"]):
        # ["uniform","uniform","uniform"], 
        # ["chunks", "chunks", "chunks"],
        # [None, "3line", None],
        # ["maxlikeli", "maxlikeli", "maxlikeli"],
        # ["onechunk", "3line", "linePlusL"]):
    for likeliver, priorver, parse_ver, chunkmodel, posterior_ver, name in zip(
        ["segments", "segments", "segments", "combine_segments", "combine_segments"],
        ["uniform","uniform","uniform", "uniform", "uniform"], 
        ["chunks", "chunks", "chunks", "chunks", "chunks"],
        [None, "3line", None, "3line", None],
        ["maxlikeli", "maxlikeli", "maxlikeli", "maxlikeli", "maxlikeli"],
        ["onechunk", "3line", "linePlusL", "3line_combine", "linePlusL_combine"]):

        # 1) new object
        PD = ProbedatTaskmodel(probedat)
        
        # 3) Buidl model
        if name in ["linePlusL", "linePlusL_combine", "onechunk"]:
            chunkmodel = PD.makeParseFunction(name)

        assert chunkmodel is not None, "need to replace this with ParseFunction.."

        PARAMS, PARAMS_MODEL = PD.getParams(priorver=priorver, parse_ver=parse_ver, 
                                            chunkmodel=chunkmodel, name=name,
                                            posterior_ver=posterior_ver, 
                                            likeliver=likeliver)
        PD.applyModel(PARAMS, PARAMS_MODEL)
        # save
        PDdict[name] = PD

    # --- 3) Score all trials
    # make new Probedat that will hodl summary scores.
    Probedat = ProbedatTaskmodel(probedat)
    def sf(m):
        """compare two models, retgurns index between -1,1
        """   
        a = m["3line"]
        b = m["linePlusL"]
        return 2*((a/(a+b))-0.5)

    scorefun = (
        "modelcomp_offline",
        sf
    )

    # NOTE: this is tailored for lines5
    Probedat.assignTaskmodelResults(PDdict, scorefun)
    Probedat.assignTaskmodelResults(PDdict, ("3line", lambda m:m["3line"]))
    Probedat.assignTaskmodelResults(PDdict, ("onechunk", lambda m:m["onechunk"]))
    Probedat.assignTaskmodelResults(PDdict, ("linePlusL", lambda m:m["linePlusL"]))
    Probedat.assignTaskmodelResults(PDdict, ("3line_combine", lambda m:m["3line_combine"]))
    Probedat.assignTaskmodelResults(PDdict, ("linePlusL_combine", lambda m:m["linePlusL_combine"]))

    if return_all_models:
        return Probedat.Probedat, PDdict
    else:
        if ploton:
            fig = Probedat.plotModelscores()
            return Probedat.Probedat, fig
        else:
            return Probedat.Probedat


############## EXTRACTIONS
# e..g, to process Probedat, get a scalar score, and append as a new column.
def assignStrokeToLabel(Probedat, ver):
    """ for each stroke get a label. e..g, could label 0, 1, 2, .. from left to
    right of screen.
    - ver, controls method for labeling
    Note: returns all trials, even if empty behavior.
    RETURNS:
    list that is same length as Probedat. Can make dataframe, appending this as column
    using Probedat.asPandas()
    """
    from pythonlib.drawmodel.tasks import TaskClass

    if ver in ["first_stroke_horiz_pos", "first_stroke_vert_pos"]:
        # First touch position; 0, 1, 2, from left.
        ########## GET LABELS FOR STROKES
        strokeslist_beh, tasks = Probedat.pd2strokes(Probedat.Probedat, return_all_trials=True)

        # ==== CONVERT TASKS TO TASK CLASS OBJECTS
        tasks = [TaskClass(t) for t in tasks]

        # === 1) Assign each stroke to its corresponding task stroke.
        stroke_assignments = []
        for strokes_beh, T in zip(strokeslist_beh, tasks):
            if len(strokes_beh)==0:
                stroke_assignments.append([])
            else:
                A = T.behAssignClosestTaskStroke(strokes_beh)
                stroke_assignments.append(A)

        # === 2) each task stroke, give it a label.
        def func(task):
            # for each stroke, get its center's x coord
            coms = getCentersOfMass(task["strokes"])
            from pythonlib.tools.nptools import rankItems
            
            if ver=="first_stroke_horiz_pos":
                # rank their x positions (0 is leftmost,,,)
                ranks = rankItems([c[0] for c in coms])    
            elif ver=="first_stroke_vert_pos":
                ranks = rankItems([c[1] for c in coms])    

            return ranks    
        [t.assignLabelToEachStroke(func) for t in tasks]
            
        # === 3) each beh stroke can now get a label
        beh_stroke_labels = [T.TaskLabels[ass] for ass, T in zip(stroke_assignments, tasks)]
        first_stroke_label = [b[0] if len(b)>0 else None for b in beh_stroke_labels]
        return first_stroke_label
    else:
        print(ver)
        assert False, "not codede"

def extractFeature(Probedat, feat, returnlist=False):
    """ extracts a list, same lengtha s Probvedat"""
    if feat=="blokk":
        """ appends both blokk within day (blokk) and blokk across days, 
        in order (blokk_across_days) """
        # 1) Separate by blokks
        blokklist = []
        for i in Probedat.getIndsTrials():
            fd, t = Probedat.fd_trial(i)
            blokk = getTrialsBlokk(fd, t)
            blokklist.append(blokk)
            
        # - convert to pandas
        DF = Probedat.asPandas(cols_to_add={"blokk":blokklist})

        # - convert to global blokk (not just within the day).
        """ adds a column called 'blokk_across_days' to dataframe,
        which is in order sorted by (day first, bkk second), 
        """
        # days = np.floor(DF["tval"].values)
        days = DF["date"].values
        blokks = DF["blokk"].values

        daylist = sorted(list(set(days)))

        blokkcount = 0
        bkcounter = {} # maps (day, bkk_within_day) --> global_bkk
        for d in daylist:
            # blokksthis = DF[np.floor(DF["tval"].values)==d]["blokk"]
            blokksthis = DF[DF["date"].values==d]["blokk"]
            blokksthislist = sorted(list(set(blokksthis)))
            
            for b in blokksthislist:
                bkcounter[(d,b)] = blokkcount
                blokkcount+=1

        from pythonlib.tools.pandastools import applyFunctionToAllRows
        def F(x):
            # return bkcounter[(np.floor(x["tval"]), x["blokk"])]
            return bkcounter[x["date"], x["blokk"]]
        DF = applyFunctionToAllRows(DF, F, "blokk_across_days")
        if returnlist:
            return DF["blokk_across_days"].values
        return DF
    else:
        print(feat)
        assert False, "not coded"


def plotTimecourseGeneral(ax, Pp, idxs, xver="trial", yver="behscore", idx_to_call_zero=None, 
                         xlabelver="intervals"):
    """ general purpose, plotting timecourse, all trials, with flexibilty for what 
    to use in x an y axis.
    e.g., 
    can align, to overlay many plots...
    or plot over day
    - Pp, dataframe, could get from P.pandas()
    - idxs, list of indices (not trials) to plot
    - xver, what units to actually use for plot
    - yver, what y to plot
    - idx_to_call_zero, index to align to, if choose xver as "rel_idx"
    - xlabelver, how to label plot.
    RETURNS:
    x, y
    - x, y, are actual plotted valeus.
    """

    df = Pp.iloc[idxs]

    # == tranform x and y into desired plot balues.
    if yver=="errorcode":
        y = df["ErrorCode"].values
        YLIM = [-0.2, 6.2]
    elif yver=="behscore":
        y = df["beh_multiplier"].values
        YLIM = [0, 1]
    else:
        assert False, "not coded"

    if xver=="trial":
        x = df["trial"].values
    elif xver=="idx":
        x = idxs
    elif xver=="rel_idx":
        assert idx_to_call_zero is not None, "need to tell mje wghere to algin. this is first tiral (0) for 2nd block"
        x = idxs-idx_to_call_zero

    # === PLOT
    ax.plot(x, y, "-k")
    # Color, based on error code
    y_errorcodes = df["ErrorCode"].values
    ax.scatter(x, y, c=y_errorcodes, cmap = "jet")

    # Mark, if this was online abort
    y_isabort = df["IsAbort"].values
    idx_isabort = np.where(y_isabort)[0]
    x_isabort = x[idx_isabort]
    ax.plot(x_isabort, 0.95*YLIM[1]*np.ones_like(x_isabort), "xr")

    # Make look pretty
    if xver=="rel_idx":
        ax.axvline(0-0.5)
    if xlabelver=="intervals":
        xforlab = x[::10]
    elif xlabelver=="endpoints":
        xforlab = x[[0, N-1, -1]]        
    ax.set_xticks(xforlab)
    ax.set_xticklabels(xforlab)
    ax.set_ylim(YLIM)
    
    return x, y


def plotAlignedToIndex(Pp, idx_onsets, N=10, yver="behscore", plotorg="separate"
                      , xlabelver="endpoints"):
    """ many subplots, each aligned to a different index, 
    index is into Pp (which is P.pandas()), so is NOT trial.
    - N is how many trials pre and post (total 2N). is in indices, not trials.
    - e.g., useful if ind_onsets are block transitions.
    - idx_onsets should be the first trial in the 2nd block.
    """
    def _block_transition_type(Pp, idx):
        """ idx indexst in Pp.
        returns [bk1, bk2], which are blocks in
        index idx-1 and idx. i..e, assumes idx is the
        first indiex for a new block
        """
        bk1 = Pp["block"].values[idx-1]
        bk2 = Pp["block"].values[idx]
        return bk1, bk2


    def _plot(ax, i,  xver = "trial", idx_to_call_zero=None, xlabelver="endpoints"):
        if i+N>len(Pp):
            Nend = len(Pp) - i
        else:
            Nend = N

        idxs = np.arange(i-N, i+Nend)
        idxs = idxs[idxs>=0]
        x, y = plotTimecourseGeneral(ax, Pp, idxs, xver = xver, yver=yver, 
            idx_to_call_zero=idx_to_call_zero)
        try:
            ax.axvline(x[N]-0.5)
        except:
            pass
        
        #  title this by its transition features
        bk1, bk2 = _block_transition_type(Pp, i)
        ax.set_ylabel(f"bk:{bk1}-{bk2}")

    if plotorg=="separate":
        ncols = 5
        nrows = int(np.ceil(len(idx_onsets)/ncols))
        fig, axes = plt.subplots(nrows, ncols, sharex=False, sharey=True, figsize=(ncols*3, nrows*2))
        for i, ax in zip(idx_onsets, axes.flat):
            _plot(ax, i, xlabelver=xlabelver)
    elif plotorg=="singleplot":
        # then single plot but not overlaid (actual trial on x axis)
        fig, ax = plt.subplots(figsize=(15,5))
        for i in idx_onsets:
            _plot(ax, i, xlabelver=xlabelver)
    elif plotorg=="overlaid":
        # then single plot aligned to save x positions
        fig, ax = plt.subplots(figsize=(15,5))
        for i in idx_onsets:
            _plot(ax, i, xver="rel_idx", idx_to_call_zero=i, xlabelver=xlabelver)
    else:
        assert False
            
    return fig

def plotAllTrialsByErrorCode(P, sdir):
    """ plots all trials, but split based on error code
    - sdir is base dir.
    e.g., will make dirs: sdir/alltrials_by_errorcode/code_[]-<figures>
    """

    # Make save dir
    sdirthis = f"{sdir}/alltrials_by_errorcode"
    import os
    os.makedirs(sdirthis, exist_ok=True)

    NplotPerFig = 40

    P.pandas()["error_code"] = P.getTrialsHelper("getTrialsErrorCode", "all")
    P.pandas().reset_index(drop=True)
    errorcode_list = sorted(P.pandas()["error_code"].unique().tolist())

    for ec in errorcode_list:
        trials = P.pandas()[P.pandas()["error_code"]==ec].index.tolist()
        nfigs = int(np.ceil(len(trials)/NplotPerFig))
        for n in range(nfigs):
            if n==nfigs-1:
                idx = range(n*NplotPerFig, len(trials))
                trialsthis = [trials[i] for i in idx]
            else:
                idx = range(n*NplotPerFig, (n+1)*NplotPerFig)
                trialsthis = [trials[i] for i in idx]

            fig = P.plotMultTrials(trialsthis)
        #     fig.savefig(f"{SAVEDIRDAY}/trialsAllChronOrder-{n}.pdf")
            print("saving ", f"{sdirthis}/code_{ec}-subset_{n}.pdf")
            fig.savefig(f"{sdirthis}/code_{ec}-subset_{n}.pdf")
            plt.close("all")

