
def prepare_trial(D, ind):
    """ outputs this trial beh and task in format that
    can pass directly to any behmodel.
    OUT:
    Beh, Task, class instances
    NOTE:
    will modify Parser and Task otherwise will modify D.
    """
    from pythonlib.behmodel.behavior import Behavior

    # D = D.copy() 

    ######## TASK (task and parses)
    # parses
    Parser = D.Dat["parser"][ind]
    Parser.finalize()
    Task = D.Dat["Task"][ind]
    # strokes_task = D.Dat["strokes_task"][ind]

    # from pythonlib.tools.stroketools import check_strokes_identical
    # assert check_strokes_identical(Task.Strokes, strokes_task)==True
    # make sure parses are in original coordinate system
    
    if False:
        # Old version, thought of extracrting parses as strokes, but better to keep
        # as Parser format.
        strokes_task_out, parses_list = Parser.strokes_translate_back_batch()

        # Processed correctly.
        Task.input_parses_strokes(parses_list)
        Task.Strokes = strokes_task_out
    else:
        Task.input_parser(Parser)
        # Task.Strokes = None # dont make mistake of using this. only use parses.

    if False:
        # just to confirm that the parses and strokes_task look correct, coords, etc.
        D.plotMultStrokes([strokes_task, strokes_task_out, parses_list[0]]);

    ####### Behavior
    strokes_beh = D.Dat["strokes_beh"][ind]
    Beh = Behavior()
    Beh.input_data(strokes_beh)

    return Beh, Task


def score_dataset(D, BehModel, saveon=True, sdir=""):
    """ scores each trial using this BehModel
    Returns list of posterior scores, same length as D.Dat
    """

    assert False, "also save trialcodes"
    assert False, "also save preprocessing params for this dataset"
    
    if saveon:
        assert len(sdir)>0

    list_postscores = []
    for indtrial in range(len(D.Dat)):
        if indtrial%50==0:
            print(indtrial)
        post = BehModel.score_single_trial_dataset(D, indtrial)
        list_postscores.append(post)

    if saveon:
        import pickle
        path = f"{sdir}/posterior_scores.pkl"
        with open(path, "wb") as f:
            pickle.dump(list_postscores, f)

