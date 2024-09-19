""" Single class for analysis of grammar/sequencing for discrete shapes, combining the 
following:
- Generating parses for given (task, rule) combinations
--- Relates to stereotyped parses from Matlab objectclass (grammar plots)
--- Relates to diagnostic model (gridlinecircle)
- Compute diagnostic scores (gridlinecircle), for (beh, model) x (task) x (score_feature)
- Compute performance (fraction correct) as used in things like neuralbiasdir
"""

class GrammarDat(object):
    """
    See docs above.
    """

    def __init__(self, input_data_dict, input_version):
        """ Input data for a single trial, icnlduing the 
        Beh, the Task
        PARAMS;
        - input_data_dict, depends on input_version, see below
        - input_version, str, what kind of data is inputed.
        """

        # Properties
        self.ChunksListClassAll = {} # dict[rulename] = CLC for that rule
        self.ParsesGenerated = {} # to store parses that have already been gnenerated.
        self.ParsesGeneratedYokedChunkObjects = {} # yoked, the oriignal chunksclass tied to each parse.
        # Input data
        if input_version=="dataset":
            """ Input the dataset and a trial index
            """
            ind = input_data_dict["ind_dataset"]
            self.Dataset = input_data_dict["dataset"].copy(just_df=True)
            # self.DatasetInd = ind
            self.DatasetTrialcode = self.Dataset.Dat.iloc[ind]["trialcode"]

            Beh = self.Dataset.Dat.iloc[ind]["BehClass"]
            Task = self.Dataset.Dat.iloc[ind]["Task"]
            expt_list = self.Dataset.expts()
            assert len(expt_list)==1, "modify code to input the expt for this specifig trial..."
            exptname = expt_list[0]
            self._input_data(Beh, Task, exptname)

        elif input_version=="direct":
            # Directly input the Beh and Task
            assert False, "code it"
        else:
            print(input_version)
            print(input_data_dict)
            assert False


    def _input_data(self, Beh, Task, exptname):
        """ Input data. 
        """
        self.Beh = Beh
        self.Task = Task
        self.Expt = exptname


    def parses_generate_batch(self, list_rules):
        """ Generate all parses for this list of rules
        PARAMS;
        - list_rules, lsit of str, each a rule, like lolli
        RETURNS:
        - adds to self.ChunksListClassAll[rule] = parses, where paress
        are list of orderings (chunks)
        OVERWRITES even if alreayd done.
        """

        assert isinstance(list_rules, list)
        for rule in list_rules:
            if rule in self.ChunksListClassAll.keys():
                # Then parses already generated. ignore
                # parses = GD.ChunksListClassAll[rule]
                pass
            else:
                # Then generate, save, and return parses
                self.parses_generate(rule)

    def parses_generate(self, rule_name, DEBUG=False):
        """
        Generate and cache parses for this rule, in form of ChunksListClass
        PARAMS:
        - rule_name, string name of rule that will be applied to generate parses.
        Must be in format of rulestring, <category>-<subcat>-<rule>
        RETURNS:
        """
        from pythonlib.chunks.chunksclass import ChunksClassList
        from pythonlib.behavior.behaviorclass import BehaviorClass

        if rule_name in self.ChunksListClassAll.keys():
            assert False, "already generated. if generate again, might be different (randomized). you want to use parses_extract_generated"

        # motifkind = motifname
        # motif_params = motifparams

        # Extract all chunks consistent with this rule for this task
        # Task = D.Dat.iloc[indtrial]["Task"]
        params = {
            "Task":self.Task,
            "expt":self.Expt, 
            "rule":rule_name
        }
        CL = ChunksClassList(method="task_entry", params=params)
        CL.remove_chunks_that_concat_strokes() # Only keep the chunks that are defined by hierarhcy, not by concatting strokes        
        
        if DEBUG:
            CL.print_summary()

        if False:
            # Actually, is fine to be empty.
            if len(CL.ListChunksClass)==0:
                # Then no chunks exist
                return None, None
        
        # Save it
        self.ChunksListClassAll[rule_name] = CL

        # Genreate all concrete parses and return them
        return self.parses_extract_generated(rule_name)

    def parses_extract_generated(self, rulestring):
        """ Extract parses that are already genreated
        Genrates them if not yet done
        - First tries getting already-saved parses...
        """

        # generate clc?
        if rulestring not in self.ChunksListClassAll.keys():
            self.parses_generate(rulestring)

        # extract parses? (and cache)
        if rulestring not in self.ParsesGenerated.keys():
            CL = self.ChunksListClassAll[rulestring]
            out, out_chunkobj = CL.search_permutations_chunks(return_ver="list_of_flattened_chunks",
                return_out_chunkobj=True)
            self.ParsesGenerated[rulestring] = out
            self.ParsesGeneratedYokedChunkObjects[rulestring] = out_chunkobj

        return self.ParsesGenerated[rulestring]


    ############### DIAGNOSTIC MODELING
    def _score_beh_in_parses(self, taskstroke_inds_beh_order, rulestring):
        """ 
        Returns true if taskstroke_inds_beh_order is identival to one of the
        parses for this rulestring
        PARAMS
        - taskstroke_inds_beh_order, list of ints, behavior strokes, in format
        of task stroke indices.
        - return_matching_parse, bool, if True, then additioanlly returns the index
        of match, if match exists. otherwise returns None.
        RETURNS:
        - bool
        """
        parses = self.parses_extract_generated(rulestring)
        if isinstance(taskstroke_inds_beh_order, list):
            taskstroke_inds_beh_order = tuple(taskstroke_inds_beh_order)
        return taskstroke_inds_beh_order in parses

    def _score_beh_in_parses_find_index_match(self, taskstroke_inds_beh_order, rulestring,
                                              print_parses_if_fail=True):
        """ 
        returns the index of the parse for this rulestring that matches this beh,
        if it exists. otherwise returns None
        """
        parses = self.parses_extract_generated(rulestring)
        if isinstance(taskstroke_inds_beh_order, list):
            taskstroke_inds_beh_order = tuple(taskstroke_inds_beh_order)
        if taskstroke_inds_beh_order in parses:
            return parses.index(taskstroke_inds_beh_order)
        else:
            if print_parses_if_fail:
                print("=== parses:")
                for p in parses:
                    print(p)
                print("=== taskstroke_inds_beh_order:")
                print(taskstroke_inds_beh_order)
            return None

    ################ utils
    def dataset_trialcode_to_ind(self, trialcode):
        """ Return the index (int) into self.Dataset. 
        asserts there exactly one match
        """
        return self.Dataset.index_by_trialcode(trialcode)

    def strokes_extract(self, ind_strokes=None):
        """ Returns strokes, optioanlly ordered as in ind_strokes
        """
        strokes = self.Task.Strokes
        if ind_strokes is None:
            ind_strokes = range(len(strokes))        
        return [strokes[i] for i in ind_strokes]

    def rules_extract(self):
        """ REturn list of rules (strings), 3-part strings)
        """
        return list(self.ChunksListClassAll.keys())

    ############### PLOT
    # Plot all parses for each rule, and compare to the beh drawing.
    def plot_beh_and_parses(self, rulestr, nrand = 20):
        """ Plot all (or subset) of parses for this rulestring
        PARAMS:
        - rulestr, 3-part string,e g. "ch-dir2-(AB)n"
        RETURNS:
        - fig1, fig2, for plots of beh and parses
        """

        # 1) Plot behavior
        ind = self.dataset_trialcode_to_ind(self.DatasetTrialcode)
        fig1 = self.Dataset.plotSingleTrial(ind, task_add_num=True)
        # fig1, ax = self.Beh.plotStrokes()
        # ax.set_title("Behavior")
        # self.Beh.plotTaskStrokes()
        # # ax.set_title("Behavior")

        # 2) Plot parses
        parses = self.parses_extract_generated(rulestr)
        inds = list(range(len(parses)))
        if len(parses)>nrand:
            import random
            indsthis = random.sample(inds, nrand)
            # indsthis = list(range(nrand))
        else:
            indsthis = inds
        parses = [parses[i] for i in indsthis]
        list_strokes = [self.strokes_extract(par) for par in parses]
        # print(parses)
        fig2, axes2 = self.Dataset.plotMultStrokes(list_strokes)
        return fig1, fig2, axes2, indsthis, parses

    #################
    def print_plot_summary(self, doplot=True, only_this_rulestring=None):
        """ Print and plot things summarizing this grammardat, including parses"""


        for rule, CLC in self.ChunksListClassAll.items():
            if only_this_rulestring is not None and not rule == only_this_rulestring:
                continue
            print("----- CLC for This rule: ", rule)
            print(CLC.print_summary())
            print("... with these parases:")
            if len(self.ParsesGenerated[rule])<20:
                print(self.ParsesGenerated[rule])
            else:
                print(f"(just first N, since too many {len(self.ParsesGenerated[rule])}):")
                print(self.ParsesGenerated[rule][:20])

            if doplot:
                self.plot_beh_and_parses(rule)

        if doplot:
            self.Beh.alignsim_plot_summary()
