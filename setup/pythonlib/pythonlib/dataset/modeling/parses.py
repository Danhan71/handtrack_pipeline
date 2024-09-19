""" Helpers to visualize parsers, including chunk- and model-based parses,
and best-fit perms of those parses.
After in Dataset have run various parse extractions.

WOrks with continuous parses in D.Dat (bpl)
"""


def plot_and_print_perms_for_each_baseparse(D, indtrial):
    """ plots best-fit, and prints all perms
    """
    
    # Plot beahvior
    # 1) Plot behavior
    D.plotSingleTrial(indtrial)
    
    # Find the "chunk"
    trial_tuple = D.trial_tuple(indtrial)
    P = D.parser_get_parser_helper(indtrial)

    list_rules = []
    list_bestfitperm =[]
    list_allperms = [] # list of list of inds
    for i, p in enumerate(P.ParsesBase):

        # 1) Collect the best-fit perms
        list_rules.append(p["rule"])
        list_bestfitperm.append(p["best_fit_perms"][trial_tuple])
        list_allperms.append(P.findparses_bycommand("permutation_of_v2", {"parsekind": "base", "ind":i}))


        print("all perms in self.Parses for this BaseParse:", i, p["rule"], list_allperms[-1])



    #     list_perms =[]
    #     for indparse in list_indparse:
    #         list_perms.append(P.ParsesBase[indparse]["best_fit_perms"][trial_tuple])
    #         print(rule, indparse, P.ParsesBase[indparse]["hier"], "--", P.ParsesBase[indparse]["fixed_order"], "--", P.ParsesBase[indparse]["chunks"])
    # #         print(rule, indparse, 
    #     list_perm_indices = [p["index"] for p in list_perms]

    # Plot each best perm
    list_perms_asstrokes = [p["strokes"] for p in list_bestfitperm]
    titles = [f"{i}_{t}" for i, t in zip([p["index"] for p in list_bestfitperm], list_rules)]
    D.plotMultStrokes(list_perms_asstrokes, titles=titles)
    


def plot_baseparses_all(D, indtrial):
    """ Plot base parses (all) for this trial
    """
    
    # 1) Collect infor for all base parases
    P = D.parser_get_parser_helper(indtrial)
    P.parses_fill_in_all_strokes()

    list_rule = []
    list_strokes = []
    list_paths = []
    list_hier = []
    list_fixed = []
    list_chunks =[]
    for p in P.ParsesBase:
        list_rule.append(p["rule"])
        list_strokes.append(p["strokes"])
        list_hier.append(p["hier"])
        list_fixed.append(p["fixed_order"])
        list_chunks.append(p["chunks"])
                
    list_figs = []
    # 2) Plot parses for each base parse
#     P.extract_parses_wrapper(list(range(len(P.ParsesBase))), "strokes", is_base_parse=True)
    fig = D.plotMultStrokes(list_strokes, titles=[f"{i}_{r}" for i, r in enumerate(list_rule)]);
    list_figs.append(fig)
#     P.ParsesBase[0]["list_ps"]
#     print(P.extract_parses_wrapper("all", "list_of_paths", True))
    
    # 3) Print metadat (e.g, hier) for each base parse
    for i, (rule, hier, fixed, chunks) in enumerate(zip(list_rule, list_hier, list_fixed, list_chunks)):
        print(i, '--', rule, '--hier:', hier, '--', fixed, "--chunks:", chunks)
    
    # 4) plot graph
    fig = P.plot_graph()
    list_fig.append(fig)

    return figs


def print_summary_bestparses_alltrials(D, list_rules = ("baseline", "linetocircle", "circletoline", "lolli")):
    # same as below, differnt printing order
    # Print the best-fit for each combo of rule and trial
    for indtrial in range(len(D.Dat)):

    #     print("== DATASET beh:", D.identifier_string())
        trial_tuple = D.trial_tuple(indtrial)
        print("---", indtrial, trial_tuple)
        for rule in list_rules:

            P = D.parser_get_parser_helper(indtrial)

            list_indparse = P.findparses_bycommand("rule", {"list_rule":[rule]}, is_base_parse=True)

            # Collect the perms
            list_perms =[]
            for indparse in list_indparse:
                list_perms.append(P.ParsesBase[indparse]["best_fit_perms"][trial_tuple])
            list_perm_indices = [p["index"] for p in list_perms]

            print(f"parses:", list_indparse, f"num bestperms {len(list_perms)}", list_perm_indices, f"rule_{rule}", "--", )

        #         # Plot to compare
        #         if PLOT:
        #             D.plotSingleTrial(indtrial)
        #             list_parses_strokes = P.extract_parses_wrapper(inds, "strokes")
        #             D.plotMultStrokes(list_parses_strokes)


def print_summary_all_bestfits_each_baseparse(D, indtrial):
    
    P = D.parser_get_parser_helper(indtrial)

    for ind_base_par, p in enumerate(P.ParsesBase):
        print("--- baseparse ", ind_base_par, p["rule"])
        print("done_permutations:", P.ParsesBase[ind_base_par]["done_permutations"])
        for k, p in p["best_fit_perms"].items():
            print(k, "permnum: ", p["index"])
