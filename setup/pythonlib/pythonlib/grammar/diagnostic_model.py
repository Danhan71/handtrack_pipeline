from pythonlib.dataset.dataset_analy.motifs_search_ordered import find_motif_in_beh_wildcard
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class DiagnosticModel(object):
    """"""
    def __init__(self):
        self.Dat = None
        self.DatAgg = None

    def preprocess_dataset_extract_scores(self, D, LIST_MODELS = None,
        LIST_MOTIFS= None, GET_MODELS=True, 
        COLS_TO_KEEP = ("taskcat_by_rule")):
        """ Full pipeline to extract diagnostic features/scores, given a dataset and 
        scorers (models)
        """

        ## PARAMS
        animal = "|".join(D.animals())

        if LIST_MODELS is None:
            LIST_MODELS = ["rand-null-uni"] # just random

        if LIST_MOTIFS is None:
            # This determines what kinds of scores are collected
            expt = D.expts()[0]
            if expt=="gridlinecircle":
                LIST_MOTIFS = [
                    ("repeat", {"shapekey":"shape",
                                 "shape":"circle",
                                 "nmin":2,
                                 "nmax":4}),
                    ("repeat", {"shapekey":"shape",
                                 "shape":"line",
                                 "nmin":2,
                                 "nmax":4}),
                    ("lolli", {})
                ]
                # This determines which models (controls) to get
                LIST_MODELS = ["alternate", "baseline", "linetocircle", "circletoline", "lolli"]
                
            else:
                LIST_MOTIFS = [
                    ("repeat", {"shapekey":"shape",
                                 "shape":"line-8-4-0",
                                 "nmin":2,
                                 "nmax":6}),
                    ("repeat", {"shapekey":"shape",
                                 "shape":"line-8-3-0",
                                 "nmin":2,
                                 "nmax":6}),
                ]

        self.Dataset = D
        self.DatasetCopy = D.copy()
        self.Params = {
            "LIST_MODELS":LIST_MODELS,
            "LIST_MOTIFS":LIST_MOTIFS
        }

        ##### EXTRACT SCORES FOR BEH
        DAT = []

        # For behvior, collect across all trials
        for indtrial in range(len(D.Dat)):
            if indtrial%50==0:
                print(indtrial)
                
            character = D.Dat.iloc[indtrial]["character"]
            epoch = D.Dat.iloc[indtrial]["epoch"]
            monkey_train_or_test = D.Dat.iloc[indtrial]["monkey_train_or_test"]

            # print("KISTI MORIFS")
            # for x in LIST_MOTIFS:
            #     print(x)
            for motifkind, motif_params in LIST_MOTIFS:
                
                # print("HERE")
                # print("kind", motifkind)
                # print("paramd", motif_params)
                # 2) Collect behavior
                tokens = D.taskclass_tokens_extract_wrapper(indtrial, "beh_firsttouch")
                matches_summary_frac_beh = find_motif_in_beh_wildcard(tokens, motifkind, motif_params, return_as_number_instances=False)
                ntok_expected_pertrial = _count_num_tokens_used(matches_summary_frac_beh)
        #         ntok_expected_pertrial = sum([_ntok_per_group(motifkind, k)*v for k, v in matches_summary_frac_beh.items()])
                DAT.append({
                    "indtrial":indtrial,
                    "trialcode":D.Dat.iloc[indtrial]["trialcode"],
                    "taskgroup":D.Dat.iloc[indtrial]["taskgroup"],
                    "probe":D.Dat.iloc[indtrial]["probe"],
                    "agent":f"{animal}-{epoch}",
                    "score_name": _score_name(motifkind, motif_params),
                    "score": ntok_expected_pertrial,
                    "score_list_matching_parses":"IGNORE",
                    "parses":"IGNORE",
                    "character": character,
                    "epoch": epoch,
                    "monkey_train_or_test": monkey_train_or_test})

                for col in COLS_TO_KEEP:
                    if col in D.Dat.columns:
                        DAT[-1][col] = D.Dat.iloc[indtrial][col]

        ##### EXTRACT FOR MODELS
        # For model, collect across all unique tasks
        # For behvior, collect across all trials

        list_characters = D.Dat["character"].unique().tolist()
        def _get_single_trial(char):
            """ Get the first trial that uses this char
            """
            return D.Dat[D.Dat["character"]==char].index.tolist()[0]

        if GET_MODELS:
            for i, character in enumerate(list_characters):
                if i%10==0:
                    print("character", i, "/", len(list_characters), character)

                indtrial = _get_single_trial(character)
                # epoch = D.Dat.iloc[indtrial]["epoch"]
                # monkey_train_or_test = D.Dat.iloc[indtrial]["monkey_train_or_test"]
                
                for motifkind, motif_params in LIST_MOTIFS:

                    # 1) Collect all models    
                    for rule_control_model in LIST_MODELS:
                        # collect expected number of tokens in each motif
                        list_matches, parses = find_motif_in_beh_wildcard_control(D,
                            motifkind, motif_params, rule_control_model, indtrial,
                            return_as_number_instances=False)

                        # For each match, compute the num tokens used
                        if list_matches is None:
                            ntok_expected_pertrial = 0
                        else:
                            list_ntok_used = [_count_num_tokens_used(dict_match) for dict_match in list_matches]
                            ntok_expected_pertrial = np.mean(list_ntok_used)

                        DAT.append({
                            "indtrial":indtrial,
                            "agent":f"model-{rule_control_model}",
                            "score_name": _score_name(motifkind, motif_params),
                            "score": ntok_expected_pertrial,
                            "character": character,
                            "probe":"IGNORE",
                            "trialcode":"IGNORE",
                            "taskgroup":"IGNORE", 
                            "epoch": "IGNORE",
                            "score_list_matching_parses":list_ntok_used,
                            "parses":parses,
                            "monkey_train_or_test": "IGNORE"})    

                        for col in COLS_TO_KEEP:
                            if col in D.Dat.columns:
                                DAT[-1][col] = D.Dat.iloc[indtrial][col]
        from pythonlib.tools.pandastools import aggregGeneral, applyFunctionToAllRows

        dfall = pd.DataFrame(DAT)

        # add a column indicating if is monkey or model
        def F(x):
            if animal in x["agent"]:
                return "monkey"
            elif "model" in x["agent"]:
                return "model"
            else:
                print(x)
                assert False
        dfall = applyFunctionToAllRows(dfall, F, "agent_kind")


        # add a column indicating the rule (either for monkey or model)
        def F(x):
            ind = x["agent"].find("-")
            return x["agent"][ind+1:]

        dfall = applyFunctionToAllRows(dfall, F, "agent_rule")

        # Merge linetocircle and circletoline? If only care about repeats then do this
        # - add a new column
        if False:
            def F(x):
                if x["agent_rule"] in ["linetocircle", "circletoline"]:
                    return "repeats"
                else:
                    return x["agent_rule"]
            dfall = applyFunctionToAllRows(dfall, F, "agent_rule")

        # Aggregate, taking mean over trials
        dfall_agg = aggregGeneral(dfall, group=["score_name", "character", "agent_kind", "agent_rule"], 
                      values=["score"], nonnumercols=[])

        # Conventions
        if "score_value" in dfall.columns:
            map_old_to_new = {
                "score_value":"score",
                # "score_name":"diagfeat",
            }
            dfall = dfall.rename(map_old_to_new, axis=1)

        ### SAVE
        self.Dat = dfall
        self.DatAgg = dfall_agg


    def grammardat_extract(self, character):
        """ Return the GD for this char, picking one example trial
        """
        dfthis = self.Dat[(self.Dat["character"]==character)]
        this = dfthis["indtrial"].unique().tolist()
        # assert len(this)==1
        indtrial = this[0]
        # Plot each parse, overlay the scores
        GD = self.Dataset.GrammarDict[indtrial]
        return GD

    def rulestring_extract_model_rules(self):
        """ Return list of rulestrings for models
        applies tot his character
        """
        # return self.grammardat_extract(character).rules_extract()
        return self.Params["LIST_MODELS"]

    def plotwrapper_example_allcharacter(self, SDIR):
        """ Wrapper to make all useful plots across all rulestrings and characters,
        such as beh, image, parases, and scores. Makes ~6 plots for each character.
        """

        import os
        sdir = f"{SDIR}/example_char_beh_parses_drawings"
        os.makedirs(sdir, exist_ok=True)

        list_char = sorted(self.Dat["character"].unique().tolist())
        list_rule = self.rulestring_extract_model_rules()

        for char in list_char:
            fig1, fig2, fig3, fig4, fig5 = self.plot_example_character_beh_scores(char)
            fig1.savefig(f"{sdir}/{char}|taskimage.pdf")
            fig2.savefig(f"{sdir}/{char}|beh_strokes.pdf")
            fig3.savefig(f"{sdir}/{char}|beh_discrete.pdf")
            fig4.savefig(f"{sdir}/{char}|scores_raw.pdf")
            fig5.savefig(f"{sdir}/{char}|scores_mean.pdf")

            for rulestring in list_rule:
                figbehtask, figparses = self.plot_example_character_model_parses(rulestring, char)
                figparses.savefig(f"{sdir}/{char}|parses|{rulestring}.pdf")
            plt.close("all")

    def plot_example_character_model_parses(self, rulestring, character):
        """ Plot model parses, along with scores (for each score name) for
        each pasrses
        PARAMS:
        - rulestring, e.g, "chmult-dirdir-LolDR"
        """

        # 1) extract rows for this char and model
        # char = list_char[0]
        agent_kind = "model"

        # Get parses and scores
        parsesdict = self.parses_scores_extract(rulestring, character)

        # Plot 
        dfthis = self.Dat[(self.Dat["character"]==character) & (self.Dat["agent"]==f"{agent_kind}-{rulestring}")]
        this = dfthis["indtrial"].unique().tolist()
        assert len(this)==1
        indtrial = this[0]
        # Plot each parse, overlay the scores
        GD = self.Dataset.GrammarDict[indtrial]
        figbehtask, figparses, axes2, list_ind_parses, parses_plotted = GD.plot_beh_and_parses(rulestring, nrand=10)

        # titles the plots with thier score
        # def _get_score_this_parse(parse):
        #     scoresthis = []
        #     list_scorenames = []
        #     for sn, scores in parsesdict.items():
        #         sc = scores[indparse]["score"]
        #         assert scores[indparse]["parse"] == parses_plotted[indparse] # ensure they are matched (plotted and scores)
        #         scoresthis.append(sc)
        #         list_scorenames.append(sn)
        #     return scoresthis, list_scorenames

        list_titles =[]
        # print("HERE")
        # print(list_ind_parses)
        # print(parsesdict)
        # print(parsesdict.keys())
        list_scorenames = self.scorenames_extract()
        for par in parses_plotted:
            list_sc = []
            for scorename in list_scorenames:
                score = parsesdict[(scorename, tuple(par))]
                list_sc.append(score)
            # sthis, list_scorenames = _get_score_this_parse(indparse)
            list_titles.append("|".join([str(s) for s in list_sc]))

        for i, (ax, title) in enumerate(zip(axes2.flatten(), list_titles)):
            ax.set_title(f"score:{title}")      
            if i==0:
                ax.set_ylabel(list_scorenames)

        return figbehtask, figparses


    def plot_example_character_beh_scores(self, character=None, indtrial=None):
        """ Plot each beh trial for this char, along with 
        scores for each model and monkey.
        """

        from pythonlib.tools.pandastools import filterPandas
        from pythonlib.tools.snstools import rotateLabel
        
        D = self.Dataset
        dfall = self.Dat
        SIZE=3# for drawings

        if indtrial is None and character is None:
            # Pick a random trial, to use its character    
            indtrial = random.choice(dfall["indtrial"].tolist())
            character = D.Dat.iloc[indtrial]["character"]    
        elif character is not None:
            assert indtrial is None
            assert isinstance(character, str)
        elif indtrial is not None:
            assert isinstance(indtrial, int)
            character = D.Dat.iloc[indtrial]["character"]    
        else:
            assert False
            
        list_trials = D.Dat[D.Dat["character"]==character].index.tolist()
        
            
        ## Extract information
        list_epochs = D.Dat[D.Dat["character"]==character]["epoch"].tolist()
        list_strokes_beh = D.Dat[D.Dat["character"]==character]["strokes_beh"].tolist()
        list_tc = D.Dat[D.Dat["character"]==character]["trialcode"].tolist()

        list_title_beh = [f"{ep}|{tc}" for ep, tc in zip(list_epochs, list_tc)]

        list_Beh = D.behclass_extract(list_trials)
        list_strokes_task = [Beh.extract_strokes("task_after_alignsim") for Beh in list_Beh]

        # Collect all scores as titles
        list_titles = []
        for i, indthis in enumerate(list_trials):
            list_score_names, list_score_values = self._get_list_scores(indthis)
            if i==0:
                # title = "|".join([f"{self._shorthand(n)}={v[0]:.1f}" for n,v in zip(list_score_names, list_score_values)])
                title = "|".join([f"{n[-5:]}={v[0]:.1f}" for n,v in zip(list_score_names, list_score_values)])
            else:
                title = "|".join([str(v) for v in list_score_values])
            list_titles.append(title)

        # 1) Plot the task image
        if indtrial is None:
            # pick a random case of the task
            indtrial = list_trials[0]
        fig1 = D.plotSingleTrial(indtrial, ["task"], task_add_num=True, number_from_zero=True)

        # - plot all trials to confirm
        fig2, _ = D.plotMultStrokesByOrder(list_strokes_beh, titles=list_title_beh, plotkwargs={"number_from_zero":True}, SIZE=SIZE);
        fig3, _ = D.plotMultStrokesByOrder(list_strokes_task, titles=list_titles, titles_on_y=False, 
            plotkwargs={"number_from_zero":True}, SIZE=SIZE);


        # TODO: plot the aligned parse (two rows, beh vs. parse)
        df_matches = filterPandas(dfall, {"character":[character]})
        fig4 = sns.catplot(data=df_matches, hue="agent", y="score", x="score_name", aspect=1.5, height=3, jitter=True, alpha=0.7)
        rotateLabel(fig4)
        fig5 = sns.catplot(data=df_matches, hue="agent_rule", y="score", x="score_name", row="agent_kind",
                    kind="point", aspect=1.5, height=3)
        rotateLabel(fig5)

        if False:
            df_matches = filterPandas(dfall_agg, {"character":[character]})
            sns.catplot(data=df_matches, hue="agent_rule", y="score", x="score_name", row="agent_kind", aspect=1.5)
            sns.catplot(data=df_matches, hue="agent_rule", y="score", x="score_name", row="agent_kind",
                        kind="point", aspect=1.5, height=2)
        
        return fig1, fig2, fig3, fig4, fig5
            
    #################### HELPERS.
    # def slice_:
    #     # Extract all parses for a given rule and model and their scores.
    def parses_scores_extract(self, model_rule_string, character, agent_kind = "model"):
        """
        get dict, holding score for each parse. Uses parses that are already genreated.
        PARAMS:
        - model_rule_string, e.g.,, "ss-rankdir-CLr2"
        - character, string
        RETURNS:
        - dict, keys are tuple of (scorenmaes, parse), and values are scores
        e.g,:
            {('ntokens-repeat-line', (3, 1, 2, 0, 4)): 2,
             ('ntokens-repeat-line', (1, 2, 3, 4, 0)): 2,
             ('ntokens-repeat-line', (1, 0, 3, 2, 4)): 0,
             ('ntokens-repeat-line', (0, 4, 3, 2, 1)): 1,
         """

        D = self.Dataset

        # Extract parses
        dfthis = self.Dat[(self.Dat["character"]==character) & (self.Dat["agent"]==f"{agent_kind}-{model_rule_string}")]
        # this = dfthis["indtrial"].unique().tolist()
        # assert len(this)==1
        # indtrial = this[0]
        # parses = D.grammar_parses_extract(indtrial, [model_rule_string])[model_rule_string]

        # Get scores for each parse
        list_scorename = dfthis["score_name"].tolist()
        list_scores_matching_parses = dfthis["score_list_matching_parses"].tolist()
        list_parses = dfthis["parses"].tolist()
        # # sanity check
        # for list_sc in list_scores_matching_parses:
        #     assert len(list_sc)==len(parses), "one score for each parse"

        # Collect all into dict, and do sanity check
        score_parse_dict = {}
        for sn, scores, parses in zip(list_scorename, list_scores_matching_parses, list_parses):
            assert len(scores)==len(parses)
            # score_parse_dict[sn] = []
            # for sc, pa in zip(scores, parses):
            #     score_parse_dict[sn].append({
            #         "score":sc,
            #         "parse":pa
            #     })
            # score_parse_dict[sn] = []
            for sc, pa in zip(scores, parses):
                score_parse_dict[(sn, tuple(pa))] = sc
        
        return score_parse_dict           


    def _get_list_scores(self, indtrial, agent_kind = "monkey"):
        """
        """
            
        dfall = self.Dat

        list_score_values =[]
        list_score_names = sorted(dfall["score_name"].unique().tolist())
        for score_name in list_score_names:
            dfthis = dfall[(dfall["indtrial"]==indtrial) & (dfall["agent_kind"]==agent_kind) & (dfall["score_name"]==score_name)]
            if len(dfthis)!=1:
                print(dfthis)
                assert False

            list_score_values.append(dfthis["score"].values)
        return list_score_names, list_score_values

    def print_parses_summary_this(self, rulestr, char):
        """ Print summary of parses fro this rule and char
        """
        # CLC
        GD = self.grammardat_extract(char)
        GD.parses_extract_generated(rulestr)
        CLC = GD.ChunksListClassAll[rulestr]
        CLC.print_summary()        

        # Score for across each parse.
        parses_dict = self.parses_scores_extract(rulestr, char)
        print(parses_dict)


    def scorenames_extract(self):
        return sorted(self.Dat["score_name"].unique().tolist())

    # shorten them to add to title
    def _shorthand(self, scorename):
        if scorename=="ntokens-lolli":
            return "L"
        elif scorename=="ntokens-repeat_circle":
            return "Rc"
        elif scorename=="ntokens-repeat_line":
            return "Rl"
        else:
            return scorename
        


def find_motif_in_beh_wildcard_control(D, motifname, motifparams, 
                                                rulestring, indtrial,
                                                list_beh_mask = None, 
                                                return_as_number_instances=False, 
                                                DEBUG=False):
    """
    RETURNS:
    - list_matches, len of parses.
    (None, None) if this rule_control_model doesnt have any parses for this trial's task
    """
    # 1) get parses
    parses = D._grammarparses_parses_extract(indtrial, [rulestring])[rulestring]

    # 2) Convert each parse to a tokens
    T = D.Dat.iloc[indtrial]["Task"]
    list_matches = []
    for par in parses:
        tokens_this = T.tokens_reorder(par)
        matches = find_motif_in_beh_wildcard(tokens_this, motifname, motifparams, list_beh_mask=list_beh_mask,
                                                              return_as_number_instances=return_as_number_instances)
        list_matches.append(matches)
        if DEBUG:
            print(chunk_flat)
            print(matches)
            display(matches)
            BehFake.alignsim_plot_summary()

    return list_matches, parses

# Sample  a single ordering of tokens that does alternation
# 
import random

def _eligible_tokens(tokens_remain, tokens_taken):
    """ 
    all are lists of indices
    """
    # only those not taken and not identical shape to prev taken
    if len(tokens_taken)==0:
        tokens_elegible = tokens_remain
    else:
        tokens_elegible = [t for t in tokens_remain if list_tokens[t] != list_tokens[tokens_taken[-1]]]
        if len(tokens_elegible)==0 and len(tokens_remain)>0:
            # then open up eligibility to all tokens
            tokens_elegible = tokens_remain
    return tokens_elegible

def _sample_token(tokens_remain, tokens_taken):
    tokens_elig = _eligible_tokens(tokens_remain, tokens_taken)
    ind_tok = random.choice(tokens_elig)
    
    tokens_taken.append(ind_tok)
    tokens_remain = [t for t in tokens_remain if t!=ind_tok]
    return tokens_remain, tokens_taken
    
def _sample_single_chunk(list_tokens):
    tokens_remain = range(len(list_tokens))
    tokens_taken = []
    while len(tokens_remain)>0:
        tokens_remain, tokens_taken = _sample_token(tokens_remain, tokens_taken)
    return tokens_taken


def _count_num_tokens_used(dict_matches, unique_tokens=True):
    """ Counts number of tokens used across motifs within
    dict_matches. 
    PARAMS:
    - dict_matches, dict where keys are motifs and values 
    are lists of lists of ints (found storke groupings_)
    - unique_tokens, bool, if True, then only counts unique
    tokens used
    RETURNS:
    - scalar, num tokens
    """
    
    for k, v in dict_matches.items():
        for vv in v:
            assert isinstance(vv, list), "you probably gave me already counted data"
    
    list_tok_all = []
    for v in dict_matches.values(): # get each list of list
        for vv in v: # list of int
            list_tok_all.extend(vv)
    if unique_tokens:
        list_tok_all = list(set(list_tok_all))
    return len(list_tok_all)
    
    
def _ntok_per_group(motifkind, key):
    if motifkind=="repeat":
        return key
    elif motifkind=="lolli":
        return 2
    else:
        assert False

        
def _score_name(motifkind, motif_params):
    if motifkind=="lolli":
        return f"ntokens-lolli"
    elif motifkind=="repeat":
        return f"ntokens-repeat-{motif_params['shape']}"
    else:
        assert False

        