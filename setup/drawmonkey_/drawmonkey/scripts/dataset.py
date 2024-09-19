""" Scripts to preprocess and analyze datasets, general-purpose code
Mainyl for "main" (i..e, across day) analysis, but also applies to daily (note that
daily is uisually ran in tools.preprocess as part of preprocessing).
"""

# Example calls:


# -- Generate simple summary plots, for main analy, if already generated metadat (possibly across multiple rules)
# python -m scripts.dataset plot_simplesummary charstrokeseqpan None None Diego main null True



if __name__=="__main__":
    from pythonlib.dataset.dataset import load_dataset, load_dataset_daily_helper

    import sys
    method = sys.argv[1]
    exptname = sys.argv[2] # <str>
    date_first = sys.argv[3] # YYMMDD
    date_last = sys.argv[4]
    animal = sys.argv[5] 

    if len(sys.argv)>6:
        dataset_dir_kind = sys.argv[6] # daily, main
    else:
        dataset_dir_kind = "main"
    
    if len(sys.argv)>7:
        rule_to_generate_metadat = sys.argv[7] # null, <str>
    else:
        rule_to_generate_metadat = "null"

    if len(sys.argv)>8:
        overwrite = sys.argv[8] # bool
    else:
        overwrite = False

    # For extraction and plots
    if dataset_dir_kind=="daily":
        rulelist = [str(date_first)]
    elif dataset_dir_kind=="main":
        rulelist = None # gets all rules
    else:
        print(dataset_dir_kind)
        assert False

    def extract():
        """ Extract metadat and dataset"""
        from analysis.dataset import generate_metadat, generate_dataset_file_from_raw

        # 1) Make metadat
        generate_metadat(exptname, date_first, date_last, animal, rule=rule_to_generate_metadat,
            dataset_dir_kind=dataset_dir_kind, overwrite=overwrite)    

        # 2) Run dataset extraction
        generate_dataset_file_from_raw(animal, exptname, dataset_dir_kind=dataset_dir_kind, 
            rulelist=rulelist, SKIP_IF_EXISTS=not overwrite)
    
    def extract_dataset():
        """ Get dataset from already-created metadata files
        If there are multiple files (each diff rule) will get all of them.
        """
        from analysis.dataset import generate_dataset_file_from_raw

        # 2) Run dataset extraction
        generate_dataset_file_from_raw(animal, exptname, dataset_dir_kind=dataset_dir_kind, 
            rulelist=rulelist, SKIP_IF_EXISTS=not overwrite)

    def load():
        if dataset_dir_kind=="daily":
            assert date_first==date_last, "this currently only loads single date.. to get mult and concat, use load_dataset, input list of dates. not ready, since need to also autoamticlaly extract expt name."
            D = load_dataset_daily_helper(animal, date_first)
        elif dataset_dir_kind=="main":
            D = load_dataset(animal, exptname, rulelist)
        else:
            print(dataset_dir_kind)
            assert False
        return D

    def plot_simplesummary():
        # Summary of drawings and overview of expt.

        from pythonlib.dataset.dataset_analy.summary import plotall_summary      

        # 3) Simple summary plots.
        plotall_summary(animal, exptname, rulelist, dataset_dir_kind)

    def plot_charstrok():
        from pythonlib.dataset.dataset_analy.characters import pipeline_generate_and_plot_all
        # D = load_dataset(animal, exptname, rulelist)
        D = load()
        pipeline_generate_and_plot_all(D)            

    def plot_grammar():
        from pythonlib.dataset.dataset_analy.grammar import pipeline_generate_and_plot_all

        # 3) Simple summary plots.
        D = load()
        pipeline_generate_and_plot_all(D, which_rules="recompute_parses")

    def plot_primsingrid():
        from pythonlib.dataset.dataset_analy.prims_in_grid import preprocess_dataset

        # 3) Simple summary plots.
        D = load()
        preprocess_dataset(D, doplots=True)

    def plot_singleprim():
        from pythonlib.dataset.dataset_analy.singleprims import preprocess_dataset

        # 3) Simple summary plots.
        D = load()
        preprocess_dataset(D, PLOT=True)

    def plot_primitivenessv2():
        from pythonlib.dataset.dataset_analy.primitivenessv2 import preprocess_plot_pipeline
        D = load()
        preprocess_plot_pipeline(D)

    def plot_microstim():
        from pythonlib.dataset.dataset_analy.microstim import plot_all_wrapper
        D = load()
        plot_all_wrapper(D)

    if method=="generate_metadat":
        # just generate the metadat. dont extract.
        # python -m scripts.dataset generate_metadat charstrokeseqpan1 230112 230114 Pancho main rulename True

        from analysis.dataset import generate_metadat

        # 1) Make metadat
        generate_metadat(exptname, date_first, date_last, animal, rule=rule_to_generate_metadat,
            dataset_dir_kind=dataset_dir_kind, overwrite=overwrite)    

    elif method=="extract_dataset":
        # python -m scripts.dataset extract_dataset charstrokeseqpan1 null null Pancho main null True
        extract_dataset()

    elif method=="extract_dataset_simplesummary":
        # Extract dataset and plot simple summary
        # python -m scripts.dataset extract_dataset_simplesummary charstrokeseqpan1 null null Pancho main null True
        extract_dataset()
        plot_simplesummary()


    elif method=="extract":
        # Generates metadat and extracts dataset
        # Usually use this for "main" (across days)
        # python -m scripts.dataset extract charstrokeseqpan1 230112 230114 Pancho main null True
        extract()

    elif method=="plot_simplesummary":
        # Summary of drawings and overview of expt.
        # python -m scripts.dataset plot_simplesummary charstrokeseqpan1 null null Pancho main null True

        plot_simplesummary()

    elif method=="plot_charstrokiness":
        # MAIN: python -m scripts.dataset plot_charstrokiness charstrokeseqpan1 null null Pancho main null True
        # DAILY: python -m scripts.dataset plot_charstrokiness null 230421 230421 Diego daily null True
        plot_charstrok()

    elif method=="plot_primsingrid":
        # python -m scripts.dataset plot_primsingrid charstrokeseqpan1 null null Pancho main null True
        # python -m scripts.dataset plot_primsingrid null 220709 220709 Pancho daily null True
        plot_primsingrid()

    elif method=="plot_singleprim":
        # python -m scripts.dataset plot_singleprim charstrokeseqpan1 null null Pancho main null True
        # python -m scripts.dataset plot_singleprim null 220709 220709 Pancho daily null True
        plot_singleprim()

    elif method=="plot_grammar":
        # python -m scripts.dataset plot_grammar charstrokeseqpan1 null null Pancho main null True
        # python -m scripts.dataset plot_grammar null 220709 220709 Pancho daily null True
        plot_grammar()

    elif method=="plot_primitiveness":
        plot_primitivenessv2()

    elif method=="plot_microstim":
        # All the plots for microstim
        plot_microstim()

    elif method=="extract-summary":
        # Then do all three steps
        extract()
        plot_simplesummary()

    elif method=="extract-grammar":
        # Then do all three steps
        extract()
        plot_grammar()

    elif method=="extract-summary-grammar":
        # Then do all three steps
        extract()
        plot_simplesummary()
        plot_grammar()

    elif method=="extract-summary-charstrok":
        # Then do all three steps
        extract()
        plot_simplesummary()
        plot_charstrok()

    else:
        print(method)
        assert False




