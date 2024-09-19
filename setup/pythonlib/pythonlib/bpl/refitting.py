""" for refitting BPL model to new data
Currently implemented where first convert beh to MPs (given default, human model), then update
params based on those MPs
"""
import numpy as np
import torch
import matplotlib.pyplot as plt





def update_param_1D(MPlist_flat, lib, lib_update, param, prior_counts_percentile = 8., ploton=True):
    """ 
    - lib, original library, will not modify
    - lib_update, new library, 
    RETURNS:
    - lib_update, updated.
    """
    from pythonlib.tools.listtools import get_counts, counts_to_pdist
    
    # -- prelims
    if param=="kappa":
        dtype = lib.pkappa.dtype
        categories = list(range(1, len(lib.pkappa)+1))
        vals = np.array([MP.k.numpy() for MP in MPlist_flat])
    elif param=="rel_type_mixture":
        dtype = lib.rel["mixprob"].dtype
        categories = ['unihist', 'start', 'end', 'mid']
        vals = []
        for mp in MPlist_flat:
            for rel in mp.relation_types:
                vals.append(rel.category)
    elif param=="prim_type_mixture":
        dtype = lib.logStart.dtype
        categories = range(len(lib.logStart)) # 0-indexed.
        vals = []
        for mp in MPlist_flat:
            for parts in mp.part_types:
                assert len(parts.ids)==1
                vals.append(parts.ids)
        vals = torch.tensor(vals)

        # Old version - mistake.
        # dtype = lib.shape["mixprob"].dtype
        # categories = range(len(lib.shape["mixprob"])) # 0-indexed.
        # vals = []
        # for mp in MPlist_flat:
        #     for parts in mp.part_types:
        #         assert len(parts.ids)==1
        #         vals.append(parts.ids)
        # vals = torch.tensor(vals)
    else:
        print(param)
        assert False
    
    
    # extract new distribution
    print("-- Extracted these counts:")
    counts_dict = get_counts(vals)
    print([k for k in counts_dict.keys()])
    print([v for v in counts_dict.values()])

    # Get prior counts based on percentile of distribution
    if True:
        # based on percentiles.
        prior_counts = np.percentile([v for v in counts_dict.values()], [prior_counts_percentile])
    else:
        # based on range
        prior_counts = (1/100)*prior_counts_percentile* max([v for v in counts_dict.values()])
    print("-- Using this for prior counts:")
    print(prior_counts)
    pdist = counts_to_pdist(counts_dict, categories, dtype, prior_counts=prior_counts)

    # update
    if param=="kappa":
        print("-- Updated lib_update.pkappa")
        lib_update.pkappa = pdist
        pold = lib.pkappa
        pnew = lib_update.pkappa
    elif param=="rel_type_mixture":
        print("-- Updated lib_update.rel['mixprob']")
        lib_update.rel["mixprob"] = pdist
        pold = lib.rel["mixprob"]
        pnew = lib_update.rel["mixprob"]
    elif param=="prim_type_mixture":
        print("-- Updated lib_update.shape['mixprob']")
        # lib_update.shape["mixprob"] = pdist
        # pold = lib.shape["mixprob"]
        # pnew = lib_update.shape["mixprob"]
        lib_update.logStart = torch.log(pdist)
        pold = lib.logStart
        pnew = lib_update.logStart
    else:
        assert False

    # Plot
    if ploton:
        fig, ax = plt.subplots()
        ax.plot(pold, label="model (original lib)")
        ax.plot(pnew, label="empirical (i.e, lib_update)")
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories)
        ax.legend()
    
    return lib_update


def update_param_2D(MPlist_flat, lib, lib_update, param, ploton=True):
    """ 2d distribitons"""
    from pybpl.library.spatial_OLD.spatial_model import SpatialModel as SpatialModelOriginal

    def _extract_spatial_hist_params(libthis):
        # extract old params that will reuse
        clump_id = libthis.Spatial.clump_id # 0, 1, 2, ...
        xlim = libthis.Spatial.xlim
        ylim = libthis.Spatial.ylim
        nbin_per_side = len(libthis.Spatial.list_SH[0].xlab)-1 # shoud be 25
        assert nbin_per_side == libthis.Spatial.list_SH[0].logpYX.shape[0]
        prior_count = libthis.Spatial.list_SH[0].prior_count # 

        print("clump_id, xlim, ylim, nbin_per_side, prior_count")
        print(clump_id, xlim, ylim, nbin_per_side, prior_count)

        return clump_id, xlim, ylim, nbin_per_side, prior_count

    clump_id, xlim, ylim, nbin_per_side, prior_count = _extract_spatial_hist_params(lib)

    # === to print new params
    # libthis = lib_update
    # _extract_spatial_hist_params(libthis)

    # find all cases of indep relations, and get their start positions and stroke nums
    sid_list = []
    gpos_list = []
    for mp in MPlist_flat:
        for sid, rel in enumerate(mp.relation_types):
            if rel.category=="unihist":
                gpos_list.append(rel.gpos)
                sid_list.append(sid)
    gpos_list = torch.stack(gpos_list) 
    sid_list = torch.tensor(sid_list) # should be same format as clump_id, 0, 1,2 ....

    # # plot all touch postioins.
    # plt.figure()
    # plt.plot(gpos_list[:,0], gpos_list[:,1], "ok", alpha=0.002)

    # New model
    SM = SpatialModelOriginal(data_start=gpos_list, data_id=sid_list, clump_id=clump_id, xlim=xlim, ylim=ylim, 
                         nbin_per_side=nbin_per_side, prior_count=prior_count)
    lib_update.Spatial = SM

    # plot extract params
    # libthis = lib_update
    # _extract_spatial_hist_params(libthis)

    # plot result 
    if ploton:
        for ind in range(len(lib.Spatial.list_SH)):
            plt.figure()
            plt.imshow(lib.Spatial.list_SH[ind].logpYX, origin="lower")
            plt.title(f"old - {ind}")
            plt.figure()
            plt.imshow(lib_update.Spatial.list_SH[ind].logpYX, origin="lower")        
            plt.title(f"new - {ind}")

    return lib_update





def libRefitted(MPlist_flat, 
    params_to_update = None,
    lib_update=None, ploton=False):
    """ Get updated library from monkeyt data , motor programs previously fit to monkey.
    Will use empriical distributions from MPlist_flat.
    TODO, Things used in stroke type scoring. All updated except the following:
    self.shapes_mu[subid], self.shapes_Cov
    self.scales_con[subid], self.scales_rate
    """ 

    if params_to_update is None:
        params_to_update = ["kappa", "rel_type_mixture", "prim_type_mixture", "spatial_hist"]

    # from pythonlib.tools.listtools import get_counts

    # Load a default model first
    import copy
    from pythonlib.bpl.strokesToProgram import lib as lib_orig
    if lib_update is None:
        # from pythonlib.bpl.strokesToProgram import lib, model 
        lib_update = copy.deepcopy(lib_orig)
    else:
        print("using passed in lib_update")

    # Apply each update to lib_udate
    for paramthis in params_to_update:
        print(f"====== DOING {paramthis}")
        if paramthis in ["kappa", "rel_type_mixture", "prim_type_mixture"]:
            lib_update = update_param_1D(MPlist_flat, lib_orig, lib_update, param=paramthis, ploton=ploton)
        elif paramthis in ["spatial_hist"]:
            lib_update = update_param_2D(MPlist_flat, lib_orig, lib_update, param=paramthis, ploton=ploton)
        else:
            print(paramthis)
            assert False

    # Some things that should always update by default, although should't differ across any monkey dataasets.
    # 1) transition probaiblties between substrokes. this should be just repeat of marginal (logStart), since
    # currently only using single substrokes
    print("Updating lib.logT (based simply on repeating lib.logStart)")
    lib_update.logT = lib_update.logStart.reshape(1,-1).repeat(len(lib_update.logStart), 1)

    # 2) pmat (prob num substrokes, given current stroke) should not depend on num strokes.
    # just take first and repeat
    print("Updating lib.pmat_nsub (based simply on repeating lib.logStart)")
    lib_update.pmat_nsub = lib_update.pmat_nsub[0].repeat((10,1))


    return lib_update
        

def plotLibDists1D(libraries_list, indlist = None, paramlist = None):

    # == PLOT DIFFERENCES IN LIB DISTRIBUTIONS
    # paramlist = ["kappa", "rel_type_mixture", "prim_type_mixture", "spatial_hist"]
    # paramlist = ["kappa", "rel_type_mixture", "prim_type_mixture"]
    # indlist = [5, 7] # indices into libraryes
    # libraries_to_plot = [libraries_list[i]["lib"] for i in range(2)]

    if paramslist is None:
        paramslist = ["kappa", "rel_type_mixture", "prim_type_mixture"]
        
    if indlist is None:
        indlist = range(len(libraries_list))

    nplots = len(paramlist)
    ncols = 1
    nrows = int(np.ceil(nplots/ncols))
    fig1, axes = plt.subplots(nrows, ncols, figsize=(ncols*10, nrows*3))

    def _getpdist(i, param):
        """ 
        - i, index into libraries_list
        - param, str
        """
        libthis = libraries_list[i]["lib"]
        index_grp = libraries_list[i]["index_grp"]
        if param=="kappa":
            pdist = libthis.pkappa
        elif param=="rel_type_mixture":
            pdist = libthis.rel["mixprob"]
        elif param=="prim_type_mixture":
            # pdist = libthis.shape["mixprob"]
            pdist = libthis.logStart
        else:
            assert False
        return pdist, index_grp

    for param, ax in zip(paramlist, axes.flatten()):
    #     ax = axes.flatten()[0]
        for i in indlist:
            pdist, index_grp = _getpdist(i, param)
            ax.plot(pdist, label=index_grp)
        ax.set_title(param)
        ax.legend()
        
    # also plot an xy plot to compare two models
    from pythonlib.tools.plottools import plotScatter45
    if len(indlist)==2:
        nplots = len(paramlist)
        ncols = 2
        nrows = int(np.ceil(nplots/ncols))
        fig2, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*5))
        for param, ax in zip(paramlist, axes.flatten()):
            vals =[]
            labels=[]
            for i in indlist:
                pdist, index_grp = _getpdist(i, param)
                vals.append(pdist)
                labels.append(index_grp)
            plotScatter45(vals[0], vals[1], ax, True)
    #         ax.plot(vals[0], vals[1], "o")
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_title(param)
    else:
        fig2=None

    return fig1, fig2


def plotLibDists2D(libraries_list, indlist=None):
    # == 2d plots

    if indlist is None:
        indlist = range(len(libraries_list))

    nplots = len(libraries_list[0]["lib"].Spatial.list_SH) # num historgrams

    figs = []
    for indhist in range(nplots):
        nsubs = len(indlist)
        ncols = 2
        nrows = int(np.ceil(nplots/ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3))
        
        for i, ax in zip(indlist, axes.flatten()):
            libthis = libraries_list[i]["lib"]
            index_grp = libraries_list[i]["index_grp"]
            ax.imshow(libthis.Spatial.list_SH[indhist].logpYX, origin="lower", cmap="plasma")
            ax.set_title(f"{indhist} - {index_grp}")
    #     fig.colorbar()
        figs.append(fig)
    return figs