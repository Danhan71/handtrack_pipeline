
from .behmodel_handler import *
from pythonlib.tools.pandastools import applyFunctionToAllRows
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class MultBehModelHandler(object):
    """ holds multipel H (behmodel handlers).
    - Ideally each H represents a single model class, with potentially multiple
    individual model rules (all applied to the same dataset)
    - Across H, could apply to same or different dataset. But I envision this applying to same,
    where this single dataset includes all "test" datasets concatenated. I..e, the same task 
    can be present across multipel epochs in this datsaet.
    - The result is each H is (beh epoch) x (model rule, for a single class), and 
    each H is a different model class.
    - Ideally this is more analysis and plotting, not for training models.
    """

    def __init__(self):
        
        # Combiantoins of H and D, where H is model shold be trained on D. 
        self.DictTrainedH = {}
        self.DictTrainedBMH = {}
        self.ListMrule = {}
        self.DictMrule = {}

        self.ListMclass = []

        # Loaded H and D. where now is all D that want to score using H.
        self.DictTestDH = {}
        self.DictTestD = {}

    def load_untrained_models(self, Dlist, model_class, list_rules):
        """ Load a set of models, all of same class (model_class)
        but can be multiple rules.
        - Useful if want models that dont beed to be trained - e.g., random
        or positive controls
        - Dlist can be len >1 if you would like to somehow update each model's params (i.e, fit)
        based on Dlist. But usually set it to len 1
        NOTE:
        - idea is that each item is a single (model, rule) and the training data for it. If
        you want all combos of dsets and models,  
        NOTE:
        - if you just want to test out model using untrained params, DONT use this. Instead, use
        apply_models_to_mult_new_dataset
        """

        assert len(list_rules)==1, "in doing chunks, realized best to have each key (class, rule) link to a single model. otherwise "
        # Note: otherwise each (class, rule) links to a single H that holds multiple models. This is fine, bu
        # need to ensure fitting doesnt then go thru each (class, rule) multip,e times
        
        ListBMH, list_dsets, ListH= cross_dataset_model_wrapper_params(Dlist, 
            model_class, list_rules)

        # Convert ListBMH to DictBMH
        for L in ListBMH:
            model_rule = L["id_mod"]
            key = (model_class, model_rule)
            self.DictTrainedH[key] = L["H"]
            self.DictTrainedBMH[key] = L
            print("-- GOT THIS UNTRAINED MODEL: ", model_class, model_rule)

        # save
        self.ListMclass = list(set(self.ListMclass + [model_class]))
        list_rules = [x[1] for x in self.DictTrainedH.keys() if x[0]==model_class]
        self.DictMrule[model_class] = list_rules

        print("** OUTCOME:")
        print("classes:", self.ListMclass)
        print("rules:", self.DictMrule)
        print("extracted trained models: ", self.DictTrainedH)

    def load_pretrained_models(self, SDIR, list_mclass=None, list_mrule=None, delete_saved_D=True):
        """ 
        Will append loaded models every time load. (Wil overwrite, if these models
        have already prevlsioly loaded)
        """
        from pythonlib.tools.expttools import findPath, extractStrFromFname

        list_path = findPath(SDIR, [],"BHM_SAVEDICT", return_without_fname=False)
        list_traintest = [extractStrFromFname(path, "-", 3) for path in list_path]
        list_traintest = sorted(list(set(list_traintest)))
        assert len(list_traintest)==1 and list_traintest[0] == "train", "not fit to training tasks?"
        train_or_test = list_traintest[0]

        if list_mclass is None:
            # search for path
            list_mclass = [extractStrFromFname(path, "-", 1) for path in list_path]
            list_mclass =sorted(list(set(list_mclass)))
            print("AUTO got this list_mclass: ", list_mclass)
        if list_mrule is None:
            list_mrule = [extractStrFromFname(path, "-", 2) for path in list_path]
            list_mrule =sorted(list(set(list_mrule)))
            print("AUTO got this list_mrule: ", list_mrule)

        ## First, reload pre-saved models
        for model_class in list_mclass:
            for mrule in list_mrule:
                ListBMH, _, _ = bmh_load(SDIR, model_class, [mrule], train_or_test)
                # Convert ListBMH to DictBMH
                for L in ListBMH:
                    model_rule = L["id_mod"]
                    key = (model_class, model_rule)
                    if delete_saved_D:
                        # Delete dataset
                        del L["D"]
                        L["H"].D = None
                    self.DictTrainedH[key] = L["H"]
                    self.DictTrainedBMH[key] = L

                    print("-- GOT THIS PRETRAINED MODEL: ", model_class, model_rule)

        # save
        self.ListMclass = list(set(self.ListMclass + list_mclass))
        for mclass in list_mclass:
            list_rules = [x[1] for x in self.DictTrainedH.keys() if x[0]==mclass]
            self.DictMrule[mclass] = list_rules

        print("** OUTCOME:")
        print("classes:", self.ListMclass)
        print("rules:", self.DictMrule)
        print("extracted trained models: ", self.DictTrainedH)


    def compute_scores_all_old_dataset(self, mode="test"):
        """ compute scores for all inputed models and datsets so far
        Useful if want to debug models, etc
        NOTE:
        - applies only to datasets either loaded as untrained data, or loading
        from presaved data. doesnt apply to new test dataset from 
        apply_models_to_mult_new_dataset
        - 
        """

        print("NOTE: better to use testing pipeline....")

        # Plot summary for each model (for scores)
        for H in set(self.DictTrainedH.values()): 
            # Take set, since an H instance can be indifferent loc in DictTrainedH
            H.compute_all(mode=mode)
                

    def apply_models_to_single_new_dataset(self, D):
        """ applies pretrained models to new dataset D.
        IN: 
        - D, a single dataset. holding multiple epochs for exampkle, if want to apply all models to all epochs data
        OUT:
        - Modifies self, to add self.DictTestH_SameDset
        """
        assert False, "use apply_models_to_mult_new_dataset"

        DlistThis = [D]
        self.DictTestH_SameDset = {}
        for mclass in self.ListMclass:
            list_mrule = self.DictMrule[mclass]
            ListBMH, list_dsets, ListH= cross_dataset_model_wrapper_params(DlistThis, mclass, list_mrule)
            assert len(ListH)==1
            H = ListH[0]

            # 2) For each mrule, update with trained params.
            # get statedict from trained model
            for mrule in list_mrule:
                print("*** APPLYING TRAINED PARAMS FOR: ", (mclass, mrule))
                state_trained = self.DictTrainedH[(mclass, mrule)].extract_state()
                H.params_prior_set(mrule, state_trained["prior_params"][mrule])

                # 3) Evaluate
                H.compute_all(mode="test")
                
            # 4) assign to D
            H.results_to_dataset(mclass)

            # 5) save this H
            self.DictTestH_SameDset[mclass] = H

        self.Dtest_SameDset = D


    def apply_models_to_mult_new_dataset(self, Dlist, list_mclass=None, hack_chunks=True):
        """ applies pretrained models to multiple new dataset D.
        IN: 
        - D, a single dataset. holding multiple epochs for exampkle, if want to apply all models to all epochs data
        - list_mclass, which model classes to run. i..e, this takes mclasses that are defined in Training, and 
        converst to test. If None, then runs all.
        - hack_chunks, 9/26/21, seems like best doing this going forward, where allow single H do 
        multiple models, shared Dataset.
        OUT:
        - Modifies self, to add self.DictTestH_SameDset
        """

        if list_mclass is None:
            list_mclass = self.ListMclass

        for D in Dlist:
            d_id = D.identifier_string()
            DlistThis = [D]
            for mclass in list_mclass:
                list_mrule = self.DictMrule[mclass]

                ListBMH, list_dsets, ListH= cross_dataset_model_wrapper_params(DlistThis, 
                    mclass, list_mrule)
                assert len(ListH)==1
                H = ListH[0]

                # 2) For each mrule, update with trained params.
                # get statedict from trained model
                for mrule in list_mrule:
                    print("*** APPLYING TRAINED PARAMS FOR: ", (mclass, mrule))
                    state_trained = self.DictTrainedH[(mclass, mrule)].extract_state()
                    # print(state_trained, mclass, mrule)
                    # print(H.ListModelsIDs)
                    # assert False
                    H.params_prior_set(mrule, state_trained["prior_params"][mrule])

                if hack_chunks is False:
                    for mrule in list_mrule:
                        # 3) Evaluate
                        H.compute_all(mode="test")
                else:
                
                    # this H contains all rules for this Dataset
                    print("COMPUTING FOR:" , "models:", H.ListModelsIDs, "datset: ", H.D)
                    H.compute_all(mode="test")

                    
                # 4) assign to D
                H.results_to_dataset(mclass)

                # 5) save this H
                self.DictTestDH[(d_id, mclass)] = {
                    "D":D,
                    "H":H
                }
            self.DictTestD[d_id] = D

        # update all modellists
        self._update_test_models()

        self.dataset_assign_all()

    def dataset_assign_all(self, use_training=False):
        """ Assign scores back to Dat
        """

        if all([len(list_mrule)==2 for list_mrule in self.DictMrule.values()]):
            get_diffs=True
        else:
            # cant get diffs if >2 rules
            get_diffs=False

        if use_training:
            for classmod, H in self.DictTrainedH.items():
                H.results_to_dataset(classmod)
        else:
            if len(self.DictTestDH)==0:
                assert False, "you want trainig dataset?"
            for key, DH in self.DictTestDH.items():
                mclass = key[1]
                DH["H"].results_to_dataset(mclass)

        self._initialize_colnames() 
        if get_diffs:
            print("GETTING DIFFS")
            self._dataset_get_diffs(use_training=use_training)

    def _update_test_models(self):
        # Repopulate MBH_load.ListMclass

        self.ListMclass = []
        for k, v in self.DictTestDH.items():
            self.ListMclass.append(k[1])
        self.ListMclass = sorted(list(set(self.ListMclass)))

        # Extract rules
        for mclass in self.ListMclass:
            list_mrule = sorted(set([mrule for H in self.list_H(mclass) for mrule in H.ListModelsIDs]))
            self.DictMrule[mclass] = list_mrule


    def _initialize_colnames(self):
        """ assumes that each class has same set of rules..
        """
        from pythonlib.dataset.beh_model_comparison import ColNames
        CN = ColNames(self.ListMclass, self.DictMrule[self.ListMclass[0]])
        self.ColNames = CN


    def _dataset_get_diffs(self, use_training=False):
        """ get all score diffs"
        """

        if use_training:
            list_D = [H.D for H in self.DictTrainedH.values()]
        else:
            list_D = self.list_D()

        for D in list_D:
            for mclass in self.ListMclass:
                coldiff = self.ColNames.colnames_minus_usingnames(mclass)
                col1 = self.ColNames.colnames_score(mclass)[1]
                col0 = self.ColNames.colnames_score(mclass)[0]
                D.Dat[coldiff] = D.Dat[col1] - D.Dat[col0]
                
                print(coldiff, "from ", col1, "minus", col0)



    ##### GETTERS

    def list_H(self, mclass=None):
        """ return list of H
        """
        if mclass is not None:
            return [v["H"] for k, v in self.DictTestDH.items() if k[1]==mclass]
        else:
            return [v["H"] for v in self.DictTestDH.values()]

    def list_D(self):
        """ return unique list of D
        """
        return list(set([v["D"] for v in self.DictTestDH.values()]))



    #### EXTRACT
    def extract_concatenated_dataset(self):
        """ 
        Returns single D, concatenating all Test Datasets
        """
        from pythonlib.dataset.analy_dlist import concatDatasets

        Dlist = [D for D in self.DictTestD.values()]
        return concatDatasets(Dlist)


    def extract_concatenated_aggregated_dataset(self, monkey_prior_col_name="epoch", monkey_prior_list=None,
        list_classes=None, model_score_name_list =None):
        """ 
        Returns single D, but also returns variations after different
        kinds of aggregations.
        """
        from pythonlib.tools.pandastools import applyFunctionToAllRows, filterPandas, aggregGeneral, summarize_feature

        # Get Single dataset
        D = self.extract_concatenated_dataset()
        Dat = D.Dat

        # Get list of monkye priors
        if monkey_prior_list is None:
            monkey_prior_list = sorted(Dat[monkey_prior_col_name].unique().tolist())
        # Get all scores
        if model_score_name_list is None:
            # Then get it 
            if list_classes is not None:
                model_score_name_list = []
                for c in list_classes:
                    model_score_name_list.extend(self.ColNames.colnames_score(c))
            else:
                list_classes = self.ColNames.Classes
                model_score_name_list = self.ColNames.colnames_score()

        # get list of alignments column names
        list_colnames_alignment = [col for col in D.Dat.columns if "alignment_" in col]
        print("Monkey priors (datasets): ", monkey_prior_list)
        print("Model scores (colnames): ", model_score_name_list)
        print("Model alignments (colnames): ", list_colnames_alignment)


        # 1) wide-form to long-form (a single column of "scores") (inlcude alignments)
        DatWide, _ = summarize_feature(Dat, "epoch", model_score_name_list+list_colnames_alignment, ["character", "trialcode"], 
            newcol_variable="model", newcol_value="score")

        # 1) wide-form to long-form (a single column of "scores")
        _, DatFlat = summarize_feature(Dat, "epoch", model_score_name_list, ["character", "trialcode"], 
            newcol_variable="model", newcol_value="score")

        # 1) wide-form to long-form (a single column of "scores") (inlcude alignments)
        _, DatFlatAlignment = summarize_feature(Dat, "epoch", list_colnames_alignment, ["character", "trialcode"], 
            newcol_variable="model", newcol_value="score")

        # 2) Wide, agg over tasks
        DatThisAgg = aggregGeneral(Dat, group = ["character", monkey_prior_col_name], 
                                   values=model_score_name_list)
        
        # 3) Long, agg over tasks
        DatFlatAgg = aggregGeneral(DatFlat, group = ["character", monkey_prior_col_name, "model"], 
                                   values=["score"])
        # 3) Long, agg over tasks
        DatFlatAlignmentAgg = aggregGeneral(DatFlatAlignment, group = ["character", monkey_prior_col_name, "model"], 
                                   values=["score"])

        return D, DatWide, DatFlat, DatThisAgg, DatFlatAgg, DatFlatAlignment, DatFlatAlignmentAgg




    #### PRING
    def print_pretrained_models(self):
        assert False, "not checked"
        for k, H in DictH.items():
            print("---")
            print(k)
            print(H.params_prior(k[1]))
        #     print(k, v.D.Dat["task_stagecategory"].value_counts())

    def print_summary(self):
        print("APPLIED TO TEST DATA:")
        print("self.ListMclass:", self.ListMclass)
        print("self.DictMrule:", self.DictMrule)

    def print_summary_untrained_models(self):
        """ Summarize models gotten by self.load_untrained_models
        """

        print("* NOTE: if oging to use these models on new dataset (self.apply_models_to_mult_new_dataset), then ignore what is in Dataset here")
        for i, (class_rule, DH) in enumerate(self.DictTrainedBMH.items()):
            print("-- ", i)
            
            D = DH["D"]
            H = DH["H"]
            
            print("- Dataset (two lines sould be identical):")
            print(D.identifier_string(), ' -- ', D.rules())
            print(H.D.identifier_string(), ' -- ', H.D.rules())
            print(D)
            
            print("- Model Class: ", class_rule[0])
            print("- Model Rules, used in key: ", class_rule[1])
            
            print("- Models")
            print(H.ListModelsIDs)

    def print_summary_test_data_models(self):
        """ Summarize cross of testdata x models
        NOTE:
        - accumulate things here as we go.
        """

        ### Print params (fitted) for each model
        print("Print params (fitted) for each model")
        for name, DH in self.DictTestDH.items():
            H = DH["H"]
            H.print_overview_params()
            

    #################################### ANALYSIS
    def iter_test_dat(self):
        """ Returns iterator over all test datasets/models, etc.
        RETURNS:
        - yiels dset(string), mclass(string), Dataset, H
        """
        for (dset, mclass), DH in self.DictTestDH.items():
            assert  DH["D"] is DH["H"].D, "expect 1 dataset, and these the same"
            list_rule = self.DictMrule[mclass]
            yield dset, mclass, DH["D"], DH["H"], list_rule

    def analy_compute_alignment_wrapper(self, ploton=False, plot_alignment_ver="diff"):
        """ All ways of computing alignemnt:
        In general, for each trial, how well does the model
        with same rule for that trial score beh, compared to all
        altenraitve models?
        NOTE: see dataset.analy.score_alignment. There was only 2x2
        INPUT:
        - ploton, plots alignment in rank order, and also prints out the trial inds so that 
        you can inspect each trial by hand if you want.
        - plot_alignment_ver, which alignmetn score to use for plotting
        """

        def _alignment(x, mclass, ver="rank"):
            """ Compute different versions of alignemnt, for this row
            Only gets within all rules(models) for this mclass
            """
            rthis = x["epoch"]

            colname_this = self.ColNames.colnames_score(mclass, rthis)
            colname_others = [self.ColNames.colnames_score(mclass, r) for r in self.DictMrule[mclass] if not r==rthis]

            score_this = x[colname_this]
            score_others = [x[c] for c in colname_others]

            if ver=="rank":
                # 0, 1, .., best to worst
                rank = sum(np.array(score_others)>score_this)
                return rank
            elif ver=="this_minus_meanothers":
                # more positive the better.
                return score_this - np.mean(score_others)
            elif ver=="diffindex":
                # (this - other)/(this + other)
                tmp = np.mean(score_others)
                return (score_this - tmp)/(score_this + tmp)
            else:
                assert False

        # For each row, compute alignment in various ways.
        for dset, mclass, D, H, _ in self.iter_test_dat():

            # 1) Alignment rank
            colthis = f"alignment_rank_{mclass}"
            D.Dat = applyFunctionToAllRows(D.Dat, lambda x: _alignment(x, mclass, "rank"), colthis)
            print("DONE", dset, mclass, colthis)
            
            # 2) Alignment (score diff from mean of others)
            colthis = f"alignment_diff_{mclass}"
            D.Dat = applyFunctionToAllRows(D.Dat, lambda x: _alignment(x, mclass, "this_minus_meanothers"), colthis)
            print("DONE", dset, mclass, colthis)
            
            # 3) Alignment - score index?
            colthis = f"alignment_diffindex_{mclass}"
            D.Dat = applyFunctionToAllRows(D.Dat, lambda x: _alignment(x, mclass, "diffindex"), colthis)
            print("DONE", dset, mclass, colthis)

            # For each trial, alignment, as rank out of all model scores.
            if False:
                sns.catplot(data=DatWide, x="epoch", y=colthis)
                sns.catplot(data=DatWide, x="epoch", y=colthis, kind="box")    

        if ploton:
            # Plot alignment across all trials in a sorted fasion. 

            ncols = 2
            nrows = int(np.ceil(len(self.DictTestDH)/ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*7, nrows*4))

            ct = 0
            for dset, mclass, D, H, _ in self.iter_test_dat():
                col = f"alignment_{plot_alignment_ver}_{mclass}"
                print(D.identifier_string(), ' -- ', mclass)

                # sort dataframe by a column
                dfthis = D.Dat.sort_values(col).reset_index(drop=True)
                inds_sorted_best2worst = dfthis.index.tolist()
                alignments = dfthis[col].tolist()
                
                print(f"dataset: {dset}, indstrials, low to high value, based on {col}")
                print(inds_sorted_best2worst)
            #     (alignments)
                
                ax = axes.flatten()[ct]
                ax.plot(np.arange(len(alignments)), alignments)
            #     ax.plot(inds_sorted_best2worst, alignments)
                ax.set_ylabel(col)
                ax.set_title((dset, mclass))
                ax.set_xlabel('rank order')
                ax.axhline(0)
                ct+=1        

    #################################### PLOTS
    def plot_summary_alignments(self, list_tasknames_good=None, YLIM=None):
        """ alignment vs. epoch, where each trial(or task, if agged)
        contributes one datapt, for how well it beh is aligned
        with the trained model
        INPUT:
        - list_tasknames_good, then will only plot for these tasks.
        - maxy, hack, so that malkes the 2 plots same yaxis.
        """
        
        # D, DatWide, DatFlat, DatThisAgg, DatFlatAgg, DatFlatAlignment, DatFlatAlignmentAgg 
        DatFlatAlignment, DatFlatAlignmentAgg = self.extract_concatenated_aggregated_dataset()[5:7]

        if list_tasknames_good is not None:
            DatFlatAlignment = DatFlatAlignment[DatFlatAlignment["character"].isin(list_tasknames_good)].reset_index(drop=True)
            DatFlatAlignmentAgg = DatFlatAlignmentAgg[DatFlatAlignmentAgg["character"].isin(list_tasknames_good)].reset_index(drop=True)

        dfthis = DatFlatAlignmentAgg    
        list_rule = self.ColNames.Rules

        list_figs = []
        for alignment_ver in ["rank", "diff", "diffindex"]:
            name = f"alignment_{alignment_ver}_chunks"

            dftmp = dfthis[dfthis["model"]==name]
        #     sns.catplot(data=dftmp, x="epoch", y="score", hue="model", kind="point", ci=68)
            fig = sns.catplot(data=dftmp, x="epoch", y="score", hue="model")
            if YLIM is not None:
                plt.ylim(YLIM)
            list_figs.append(fig)
            
            fig = sns.catplot(data=dftmp, x="epoch", y="score", hue="model", kind="bar", ci=68)
            if YLIM is not None:
                plt.ylim(YLIM)
            list_figs.append(fig)

            if alignment_ver == "rank":
                for i in range(len(list_rule)):
                    plt.axhline(i)
            elif alignment_ver in ["diff", "diffindex"]:
                plt.axhline(0)

        # fraction of trials that is highest ranked.
        alignment_ver = "rank"
        name = f"alignment_{alignment_ver}_chunks"

        for epoch in list_rule:
            inds = (DatFlatAlignment["model"]==name) & (DatFlatAlignment["epoch"]==epoch)
            dfthis = DatFlatAlignment[inds]

            rank_counts = [
            np.sum(dfthis["score"]==0.),
            np.sum(dfthis["score"]==1.),
            np.sum(dfthis["score"]==2.)]

            fig = plt.figure()
            plt.bar(np.arange(len(rank_counts)), rank_counts)
            plt.title(epoch)
            plt.xlabel("rank (0=best)")
            list_figs.append(fig)

        return list_figs


    # NOTE: also see "beh_model_comparisoin"
    def plot_datxmodel_overview_results(self):
        """ Plot summaries from within H, for each combiantion of 
        dataset and (mclass, mrule)
        """
        # Plot summary for each model (for scores)
        for name, DH in self.DictTestDH.items():
            H = DH["H"]
            for mname in H.ListModelsIDs:
                print("dataset:", name, ". modelrule: ", mname)
                H.plot_overview_results(mname)


    def plot_overview_trial(self, dset_rule, indtrial, PLOT_BASE_PARSES=False, alignment_col=None):
        """ [USEFUL] Inspect single trial for single dset, showing all
        the bnest parses for each model. Useful if want to see exactly 
        how score for this trial was deribed
        INPUT:
        - dset_rule, string, e..g, "lolli" which define this dset.
        - indtrial, ind within this dset.
        - PLOT_BASE_PARSES, plots all base parses (i;.e., for each chunk model).
        - alignment_col, str, e.g, "alignment_diff_chunks", to print out the laignemtn score for 
        this trial (sanity check)
        NOTE:
        - can use analy_compute_alignment_wrapper(ploton=True) to print 
        trials ordered by alginemtn, then entere those trials here to inspect
        """
        from pythonlib.dataset.dataset_analy.parses import plot_baseparses_all

        # -- Params
        plots = ["likeli", "prior"] # separte plots, sorted by these
        plot_beh_task = True
        list_figs = []
        print("(Row orders of pares plots: ", plots)
        for dset, mclass, D, H, list_rule in self.iter_test_dat():
            if not dset_rule == D.rules(force_single=True)[0]:
                print("SKIPPING dset (doesnt match input): ", dset)
                continue

            print("This trialtuple", D.trial_tuple(indtrial))

            # 2) Plot those tasks.
            if alignment_col is not None:
                print("Dataset:", D.identifier_string())
                print(alignment_col, D.Dat.iloc[indtrial][alignment_col])
                print("alignment_rank_chunks", D.Dat.iloc[indtrial]["alignment_rank_chunks"])

            for mrule in list_rule:
                print("class/rule: ", mclass, mrule)
                figs = H.plot_parses_ordered(indtrial, modelname=mrule, plots=plots, plot_beh_task=plot_beh_task);
                list_figs.extend(figs)
                for sortby in plots:
                    fig = H.plot_prior_likeli_sorted(mrule, indtrial, sortby)
                    list_figs.append(fig)
                plot_beh_task = False
                
                
            # Plot all bases parses for this task
            if PLOT_BASE_PARSES:
                print("-- BASE PARSE INFO:")
                figs = plot_baseparses_all(D, indtrial)
                list_figs.extend(figs)
        return list_figs

                            

if __name__=="__main__":
    from pythonlib.behmodel.score_dataset import score_dataset
    # from pythonlib.behmodel.scorer.prior_functions import prior_feature_extractor
    from pythonlib.behmodel.scorer.likeli_functions import likeli_dataset
    from pythonlib.behmodel.scorer.poster_functions import poster_dataset
    from pythonlib.behmodel.behmodel import BehModel
    from pythonlib.behmodel.scorer.scorer import Scorer
    from pythonlib.dataset.dataset import Dataset
    from pythonlib.dataset.dataset import concatDatasets
    from pythonlib.dataset.analy import preprocessDat
    import numpy as np
    import matplotlib.pyplot as plt
    from pythonlib.dataset.beh_model_comparison import plots_cross_prior_and_model_combined, ColNames
    from pythonlib.behmodel.behmodel_handler import cross_dataset_model_wrapper_params
    from pythonlib.behmodel.behmodel_handler import cross_dataset_model_wrapper, bmh_optimize_single, bmh_results_to_dataset, cross_dataset_model_wrapper_params, bmh_save, bmh_load
    from pythonlib.dataset.analy import extract_strokes_monkey_vs_self


    # Load datasets
    SDIR = "/data2/analyses/main/model_comp/planner"

    SUBSAMPLE = False
    N = 100
    animal = "Red"
    expt = "lines5"
    # rule_list = ["straight", "bent"]
    rule_list = ["bent"]
    LOAD_ALL_PARSES = True
    ONLY_SUMMARY_DATES = True


    # GROUPING_LEVELS = rule_list
    # model_class = "mkvsmk"
    # list_model_class = ["mix_features_bdn", "mix_features_bd", "mix_features_bn", "mix_features_dn", 
    #                     "mix_features_b", "mix_features_d", "mix_features_n", "mkvsmk", "lines5"]
    # list_model_class = ["mix_features_bd", "mix_features_dn", "mix_features_d", "mkvsmk"]
    # list_model_class = ["mix_features_bd", "mkvsmk"]
    # list_model_class = ["mix_features_tbnd", "mix_features_bnd", "mix_features_tnd", "mix_features_tbd"]
    list_model_class = ["mix_features_td", "mix_features_bd", "mix_features_nd"]

    # LOAD ALL DATASETS
    Dlist  = []
    for rule in rule_list:
        D = Dataset([])
        D.load_dataset_helper(animal, expt, rule=rule)
        
        if SUBSAMPLE:
            D.subsampleTrials(1,3)
            if len(D.Dat)>N:
                import random
                inds = sorted(random.sample(range(len(D.Dat)), N))
                D = D.subsetDataset(inds)
            
        D.load_tasks_helper()
        D.filterPandas({"random_task":[False], "insummarydates":[ONLY_SUMMARY_DATES]}, "modify")
        
        if LOAD_ALL_PARSES:
            list_parse_params = [
                {"quick":True, "ver":"graphmod", "savenote":"fixed_True"},
                {"quick":True, "ver":"nographmod", "savenote":"fixed_True"}]
            list_suffixes = ["graphmod", "nographmod"]
        else:
            list_parse_params = [
                {"quick":True, "ver":"graphmod", "savenote":"fixed_True"}]
            list_suffixes = ["graphmod"]
        D.parser_load_presaved_parses(list_parse_params, list_suffixes)
        assert len(D.Dat)>0, "why"
        Dlist.append(D)
            
            
            
    # Preprocess
    D, GROUPING, GROUPING_LEVELS, FEATURE_NAMES, SCORE_COL_NAMES = preprocessDat(Dlist[0], expt)


    ### SPLIT DATASET
    DictD = {}

    for D, rule in zip(Dlist, rule_list):

        inds_train, inds_val, inds_test = D.splitTrainTestMonkey(expt, epoch=rule)
        
        DictD[(rule, "train")] = D.subsetDataset(inds_train)
        DictD[(rule, "test")] = D.subsetDataset(inds_test)

        print("**", (rule, "train"), "this many:", len(inds_train))
        print("**", (rule, "test"), "this many:", len(inds_test))


    from pythonlib.behmodel.behmodel_handler import cross_dataset_model_wrapper, bmh_optimize_single, bmh_results_to_dataset, cross_dataset_model_wrapper_params, bmh_save, bmh_load
    import os

    from pythonlib.tools.expttools import makeTimeStamp, writeDictToYaml
    dset = f"{animal}-{expt}"
    ts = makeTimeStamp()
    SDIR = f"/data2/analyses/main/model_comp/planner/pilot-{ts}/{dset}"
    os.makedirs(SDIR, exist_ok=True)

    # Prepare strokes (other monkey) in dataset
    extract_strokes_monkey_vs_self(Dlist, GROUPING, GROUPING_LEVELS)

    for model_class in list_model_class:
        for rule in rule_list:


            # Get Dataset
            Dthis = DictD[(rule, "train")]
            Dlist_this = [Dthis]
            GROUPING_LEVELS_this = [rule]
            ListBMH, list_dsets, ListH= cross_dataset_model_wrapper_params(Dlist_this, model_class, GROUPING_LEVELS_this)

            if len(Dthis.Dat)==0:
                print(rule)
                assert False

            if len(Dthis.animals())==0 or len(Dthis.expts())==0 or len(Dthis.rules())==0:
                print(Dthis.Dat)
                assert False

            # Optimize
            if model_class=="mkvsmk":
                for H in ListH:
                    H.compute_store_priorprobs_vectorized()
                    H.compute_store_likelis()
                    H.compute_store_likelis_logprobs()
                    H.compute_store_posteriors()
    #         if model_class=="lines5":
            else:
                bmh_optimize_single(ListBMH, Dthis.identifier_string(),  rule)
        
            ### Save
            bmh_save(SDIR, Dlist_this, model_class, GROUPING_LEVELS_this, ListH, "train")

            bmh_results_to_dataset(ListBMH, suffix=model_class)