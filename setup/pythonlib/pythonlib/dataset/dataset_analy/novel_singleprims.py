"""
Novel prims in general, and usually are novel prims made by concatenating subprims.
"""

def preprocess_and_plot(D, PLOT=True, plot_methods = ("grp", "tls", "drw", "tc")):
    """ 
    Entire pipeline for extraction and plots
    """
    from pythonlib.dataset.dataset_analy.primitivenessv2 import preprocess_plot_pipeline
    
    # Combine across locations, so that this really emphasizes how novel prims are not invariant to location
    grouping = ["shape", "epoch", "block", "shape_is_novel"]
    # grouping = ["locshape_pre_this", "epoch", "block", "shape_is_novel"]
    
    DS, SAVEDIR, dfres, grouping = preprocess_plot_pipeline(D, PLOT=PLOT, grouping=grouping,
        contrast="shape_is_novel", plot_methods=plot_methods, lenient_preprocess=True)

    return DS, SAVEDIR, dfres, grouping