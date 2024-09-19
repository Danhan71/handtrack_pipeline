""" 
stuff with stroks, e.g., preprocessing for clustering, etc.
NOTE: also strok stuff exists in line2, but that is clunky, not working
with probedat new code. used it for strok feature models.

Generally works with output from Probedat.flattenToStrok
"""
import numpy as np
import matplotlib.pyplot as plt

from analysis.modelexpt import *
# from tools.utils import * 
# from tools.plots import *
# from tools.analy import *
# from tools.calc import *
# from tools.analyplot import *
# from tools.preprocess import *
# from tools.dayanalysis import *
# from analysis.strok import *
from analysis.line2 import *
from analysis.probedatTaskmodel import *
# from pythonlib.drawmodel.analysis import *
from pythonlib.tools.stroketools import *

from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture as GMM

from pythonlib.tools.plottools import plotScatterOverlay
from pythonlib.tools.pandastools import applyFunctionToAllRows
from pythonlib.tools.listtools import tabulate_list
from pythonlib.tools.expttools import findPath
import pickle5 as pickle

from pythonlib.drawmodel.sf import preprocessStroks, computeSimMatrix
from pythonlib.drawmodel.strokedists import distMatrixStrok

from pythonlib.drawmodel.strokePlots import plotStroksInGrid

def _probedat2stroks(P):
    """ to flatten P"""


def probedat2stroks(P):
    """ 
    Convert data where each element is a single trial 
    to data where each element is single stroke.
    INPUT:
    - P, is probedat class.
    RETURNS:
    - Stroks, a dataframe, each row a single stroke.
    """
    from analysis.line2 import probedat2strokefeats, strokfeats2Dataframe
    # Pp = P.pandas()

    # === 2) Flatten into strokes
    print("Flattening Probedat into strokes")
    datelist = sorted(set(P.pandas()["date"].values))
    strokfeats, TASKLIST = probedat2strokefeats(P.Probedat, datelist, only_shared_tasks=False, keep_strok=True)

    print("Converting to dataframe")
    matchedstrokes=None
    SF = strokfeats2Dataframe(strokfeats, P.Metadat, only_first_last_trials=False,
                 matchedstrokes=matchedstrokes, 
                 traintest="bothtraintest", only_if_has_model_res=False,
                only_if_task_in_all_epochs=False)

    return SF








def loadStrokeDat(params):
    """ general purpose mechanism to load a single strokedat dataframe, either
    (1) probedat --> dataset --> strokedat [less flexible, dont use]
    (2) presaved datasets --> strokedat (more flexible, since can do things
    like extract parses as strokes, etc)
    INPUT:
    - params, dict, with:
    --- ver, either "probedat" or "dataset"
    --- inputs, list of strings, either exptlist (if using probedat) or 
    pathlist (if uisng dataset)
    --- animal, string, required if uysing probedat, not if uisng dataset.
    --- strokes_ver, string, either "strokes_beh", "strokes_task", or "strokes_parse".
    default is strokes_beh. this control which data to extract. strokes_parse 
    only works for ver=dataset, and have to have already saved those parses.
    RETURN:
    - a single dataframe, where each row is a stroke.
    """

    if params["ver"]=="probedat":
        USE_DATASET = True # genearlly go thru dataset converstion.

        # === 1) load stroks across multiple experiments
        exptlist = params["inputs"]
        animal = params["animal"]

        SFall = []
        for expt in exptlist:
        #     expt = "arc2"

            FD, exptMetaDat = loadMultDataForExpt(expt,animal, whichdates="all", metadatonly=False)
            PD = loadProbeDatWrapper(FD, exptMetaDat)
            P = ProbedatTaskmodel(PD, exptMetaDat)

            if USE_DATASET:
                P.generateDataset()
                P.Dataset.subsampleTrials()
                Strokedat = P.Dataset.flattenToStrokdat()
            else:
                Strokedat = P.flattenToStrokdat()
            SFall.append(Strokedat)
        
        # === combine across multiple probedat/SF datasets
        SF = SFall[0]
        for i in range(1, len(SFall)):
            SF = SF.append(SFall[i])
        SF = SF.reset_index(drop=True)

    elif params["ver"]=="dataset":
        from pythonlib.dataset.dataset import Dataset
        assert params["strokes_ver"] in ["strokes_beh", "strokes_task", "strokes_parse", "strokes_beh_splines"]
        
        D = Dataset(params["inputs"], None)
        D.subsampleTrials()
        if params["strokes_ver"]=="strokes_parse":
            D.parsesLoadAndExtract()
        if params["strokes_ver"]=="strokes_beh_splines":
            # convert to splines (beh)
            D.strokesToSplines(strokes_ver='strokes_beh', make_new_col=True)
        SF = D.flattenToStrokdat(strokes_ver=params["strokes_ver"])

    return SF


def similarityClustering(load_params, 
        Nbasis = 200, PCAdim = 5, gmm_n_mixtures = tuple(range(5, 27)), 
        perplist = (15, 25, 45, 55, 65), rescale_strokes_ver = "stretch_to_1", 
        distancever = "euclidian_diffs", npts_space = 50, USE_DATASET=True):
    """ 
    - USE_DATASET, then uses new method where first converst probedat to datsaet, then
    extracts strokes. This allows using all methods in datset for filtering data, if needed.
    """


    # # === 1) load stroks across multiple experiments
    # # exptlist = ["shapes2", "arc1"]
    # SFall = []
    # for expt in exptlist:
    # #     expt = "arc2"


    #     FD, exptMetaDat = loadMultDataForExpt(expt,animal, whichdates="all", metadatonly=False)
    #     PD = loadProbeDatWrapper(FD, exptMetaDat)
    #     P = ProbedatTaskmodel(PD, exptMetaDat)

    #     if USE_DATASET:
    #         P.generateDataset()
    #         Strokedat = P.Dataset.flattenToStrokdat()
    #     else:
    #         Strokedat = P.flattenToStrokdat()
    #     SFall.append(Strokedat)

    
    # # === combine across multiple probedat/SF datasets
    # SF = SFall[0]
    # for i in range(1, len(SFall)):
    #     SF = SF.append(SFall[i])
    # SF = SF.reset_index(drop=True)

    assert False, "use fromSF instead"

    SF = loadStrokeDat(load_params)
    animallist = list(set(SF["animal"]))
    exptlist = list(set(SF["expt"]))

    # == Filter
    params = {
        "align_to_onset":True,
        "min_stroke_length_percentile":2,
        "min_stroke_length":50,
        "max_stroke_length_percentile":99.5,
    }
    SF = preprocessStroks(SF, params)

    # 1) Get distance matrix, entire dataset, with random instances chosen for basis
    # preprocess stroklist
    stroklist = list(SF["strok"].values)

    # rescale
    if rescale_strokes_ver=="stretch_to_1":
        stroklist = [rescaleStrokes([s])[0] for s in stroklist]
    else:
        print("keeping strokes scale unchaged")

    # interpolate
    if distancever in ["euclidian", "euclidian_diffs"]:
        # then need to be same length
        stroklist = strokesInterpolate2(stroklist, N=["npts", npts_space], base="space")

    idxs_stroklist_dat = list(range(len(stroklist)))
    idxs_stroklist_basis = random.sample(range(len(stroklist)), Nbasis)
    similarity_matrix = distMatrixStrok(idxs_stroklist_dat, idxs_stroklist_basis, stroklist=stroklist,
                       normalize_rows=False, normalize_cols_range01=True, distancever=distancever)


    # 1) Use output of PCA for below modeling
    pca_model = PCA(n_components=PCAdim)
    Xpca = pca_model.fit_transform(similarity_matrix)


    # -- TSNE
    # perplist = np.linspace(5, 50, 2)

    out = []
    for perp in perplist:
        print(perp)
        D_tsne = TSNE(n_components=2, perplexity=perp).fit_transform(Xpca)
        out.append({
            "perp":perp,
            "D_fit":D_tsne
        })
    models_tsne = out


    # === PRELIM, fitting GMM to PCA-transformed data

    covariance_type="full"
    nsplits = 1
    n_init = 1
    out = []
    for isplit in range(nsplits):
        Xtrain, Xtest = train_test_split(Xpca, test_size=0.1)
        for n in gmm_n_mixtures:
            gmm = GMM(n_components=n, n_init=1, covariance_type=covariance_type)
            gmm.fit(Xtrain)
            # bic.append(gmm.bic(np.array(s)))
        #     gmm.bic(Xin)

            out.append({
                "mod":gmm,
                "n":n, 
                "isplit":isplit,
                "bic":gmm.bic(Xtest),
                "cross_val_score":gmm.score(Xtest)
            })
    models_gmm = out

    # ============= SAVE

    if not USE_DATASET:
        # = before save, extract useful information and then discard probedat.
        def F(x):
            indprobe = x["index_probedat"]
            return x["Probedat"].pandas().iloc[indprobe].to_dict()

        from pythonlib.tools.pandastools import applyFunctionToAllRows

        SF = applyFunctionToAllRows(SF, F)

        #  then delete probedat
        del SF["Probedat"]
    else:
        # dont do anything, since should have already inehrited all columns
        del SF["Dataset"]


    # ============= SAVE
    SAVEDAT = {}
    params = {
        "rescale_strokes_ver":rescale_strokes_ver,
        "distancever":distancever
    }

    # strok data
    animal = animallist[0]
    SAVEDAT["animallist"] = animallist
    SAVEDAT["load_params"] = load_params
    SAVEDAT["exptlist"] = exptlist
    SAVEDAT["params"] = params
#     SAVEDAT["SF"] = SF
#     SAVEDAT["SF"] = TEST
    SAVEDAT["similarity_matrix"] = similarity_matrix
    SAVEDAT["Xpca"] = Xpca
#     SAVEDAT["Xtsne"] = Xtsne
    SAVEDAT["tsne_models"] = models_tsne
    SAVEDAT["gmm_models"] = models_gmm
    SAVEDAT["pca_model"] = pca_model

    if "strokes_ver" in load_params.keys():
        sver = load_params["strokes_ver"]
    else:
        sver = "strokes_beh"

    
    SDIR = f"/data2/analyses/database/clustering/bysimilarity/{'_'.join(exptlist)}-rescale_{rescale_strokes_ver}-dist_{distancever}-{sver}"
    from pythonlib.tools.expttools import makeTimeStamp

    ts = makeTimeStamp(f"{animal}", False)

    SDIRTHIS = f"{SDIR}/{ts}"
    os.makedirs(SDIRTHIS, exist_ok=True)

    fname = f"{SDIRTHIS}/SAVEDAT.pkl"

    with open(fname, "wb") as f:
        pickle.dump(SAVEDAT, f)

    SF.to_pickle(f"{SDIRTHIS}/SF.pkl")




def similarityClusteringFromSF(SF, Nbasis = 200, PCAdim = 5, gmm_n_mixtures = tuple(range(8, 30)), 
        perplist = (15, 25, 35, 45, 55, 65), rescale_strokes_ver = "stretch_to_1", 
        distancever = "euclidian_diffs", npts_space = 50, savename=None, just_get_SF=False):
    """ Pass in SF, otherwise the same"""

    # == Filter
    params = {
        "align_to_onset":True,
        "min_stroke_length_percentile":2,
        "min_stroke_length":50,
        "max_stroke_length_percentile":99.5,
    }
    SF = preprocessStroks(SF, params)

    if just_get_SF:
        # then skip all distance matrix, clustering, modeling, etc. 
        # just get SF, save.
        from pythonlib.tools.expttools import makeTimeStamp
        savenamethis = f"rescale_{rescale_strokes_ver}-dist_{distancever}-{savename}"
        SDIR = f"/data2/analyses/database/clustering/bysimilarity/indiv/{savenamethis}"
        animallist = sorted(list(set(SF["animal"])))
        # exptlist = sorted(list(set(SF["expt"])))
        animals = "_".join(animallist)

        ts = makeTimeStamp(f"{animals}", False)
        SDIRTHIS = f"{SDIR}/{ts}"
        os.makedirs(SDIRTHIS, exist_ok=True)

        fname = f"{SDIRTHIS}/SAVEDAT.pkl"

        if "Dataset" in SF.columns:
            del SF["Dataset"]
        SF.to_pickle(f"{SDIRTHIS}/SF.pkl")

        SAVEDAT = {}
        params = {
            "rescale_strokes_ver":rescale_strokes_ver,
            "distancever":distancever
        }

        # strok data
        animallist = sorted(list(set(SF["animal"])))
        exptlist = sorted(list(set(SF["expt"])))
        animals = "_".join(animallist)
        SAVEDAT["animallist"] = animallist
        SAVEDAT["exptlist"] = exptlist
        SAVEDAT["params"] = params
        fname = f"{SDIRTHIS}/SAVEDAT.pkl"
        with open(fname, "wb") as f:
            pickle.dump(SAVEDAT, f)

        return


    similarity_matrix, idxs_stroklist_basis = computeSimMatrix(SF, rescale_strokes_ver, distancever, npts_space, Nbasis)

    # # 1) Get distance matrix, entire dataset, with random instances chosen for basis
    # # preprocess stroklist
    # stroklist = list(SF["strok"].values)

    # # rescale
    # if rescale_strokes_ver=="stretch_to_1":
    #     stroklist = [rescaleStrokes([s])[0] for s in stroklist]
    # else:
    #     print("keeping strokes scale unchaged")

    # # interpolate
    # if distancever in ["euclidian", "euclidian_diffs"]:
    #     # then need to be same length
    #     stroklist = strokesInterpolate2(stroklist, N=["npts", npts_space], base="space")

    # idxs_stroklist_dat = list(range(len(stroklist)))
    # idxs_stroklist_basis = random.sample(range(len(stroklist)), Nbasis)
    # similarity_matrix = distMatrixStrok(idxs_stroklist_dat, idxs_stroklist_basis, stroklist=stroklist,
    #                    normalize_rows=False, normalize_cols_range01=True, distancever=distancever)


    # 1) Use output of PCA for below modeling
    pca_model = PCA(n_components=PCAdim)
    Xpca = pca_model.fit_transform(similarity_matrix)


    # -- TSNE
    from sklearn.manifold import TSNE
    # perplist = np.linspace(5, 50, 2)

    out = []
    for perp in perplist:
        print(perp)
        D_tsne = TSNE(n_components=2, perplexity=perp).fit_transform(Xpca)
        out.append({
            "perp":perp,
            "D_fit":D_tsne
        })
    models_tsne = out


    # === PRELIM, fitting GMM to PCA-transformed data
    from sklearn.model_selection import train_test_split
    from sklearn.mixture import GaussianMixture as GMM

    covariance_type="full"
    nsplits = 1
    n_init = 1
    out = []
    for isplit in range(nsplits):
        Xtrain, Xtest = train_test_split(Xpca, test_size=0.1)
        for n in gmm_n_mixtures:
            gmm = GMM(n_components=n, n_init=1, covariance_type=covariance_type)
            gmm.fit(Xtrain)
            # bic.append(gmm.bic(np.array(s)))
        #     gmm.bic(Xin)

            out.append({
                "mod":gmm,
                "n":n, 
                "isplit":isplit,
                "bic":gmm.bic(Xtest),
                "cross_val_score":gmm.score(Xtest)
            })
    models_gmm = out

    # ============= SAVE
    # if not USE_DATASET:
    #     # = before save, extract useful information and then discard probedat.
    #     def F(x):
    #         indprobe = x["index_probedat"]
    #         return x["Probedat"].pandas().iloc[indprobe].to_dict()

    #     from pythonlib.tools.pandastools import applyFunctionToAllRows

    #     SF = applyFunctionToAllRows(SF, F)

    #     #  then delete probedat
    #     del SF["Probedat"]
    # else:
    # dont do anything, since should have already inehrited all columns


    # ============= SAVE

    SAVEDAT = {}
    params = {
        "rescale_strokes_ver":rescale_strokes_ver,
        "distancever":distancever
    }

    # strok data
    animallist = sorted(list(set(SF["animal"])))
    exptlist = sorted(list(set(SF["expt"])))
    animals = "_".join(animallist)
    SAVEDAT["animallist"] = animallist
    SAVEDAT["exptlist"] = exptlist
    SAVEDAT["params"] = params
#     SAVEDAT["SF"] = SF
#     SAVEDAT["SF"] = TEST
    SAVEDAT["similarity_matrix"] = similarity_matrix
    SAVEDAT["Xpca"] = Xpca
#     SAVEDAT["Xtsne"] = Xtsne
    SAVEDAT["tsne_models"] = models_tsne
    SAVEDAT["gmm_models"] = models_gmm
    SAVEDAT["pca_model"] = pca_model

    savenamethis = f"{'_'.join(exptlist)}-rescale_{rescale_strokes_ver}-dist_{distancever}-{savename}"

    SDIR = f"/data2/analyses/database/clustering/bysimilarity/{savenamethis}"
    
    from pythonlib.tools.expttools import makeTimeStamp
    ts = makeTimeStamp(f"{animals}", False)
    SDIRTHIS = f"{SDIR}/{ts}"
    os.makedirs(SDIRTHIS, exist_ok=True)


    fname = f"{SDIRTHIS}/SAVEDAT.pkl"
    with open(fname, "wb") as f:
        pickle.dump(SAVEDAT, f)

    if "Dataset" in SF.columns:
        del SF["Dataset"]
    SF.to_pickle(f"{SDIRTHIS}/SF.pkl")


############################################## PLOTS

def gmm_extract_model(SAVEDAT, gmm_n):
    mod = None
    out = SAVEDAT["gmm_models"]

    for o in out:
        if o["n"]==gmm_n:
            mod = o["mod"]
            break
    if not mod:
        print(out)
        assert False, "did not extract a gmm model"
    return mod

def gmm_labels(SAVEDAT, gmm_n, SF, assign_as_column_in_SF=True):
    """
    - assign_as_column_in_SF, then replaces column called "label"
    """
    mod = gmm_extract_model(SAVEDAT, gmm_n)

    # === remap labels, sorted by order of curvature
    Xpca = SAVEDAT["Xpca"]
    SF["label"] = mod.predict(Xpca)

    distances =[]
    labelsthis = []
    for group in SF.groupby(["label"]):

        distances.append(np.mean(group[1]["distance"]))
        labelsthis.append(np.mean(group[1]["label"].values[0]))

    tmp = [[l, d] for l, d in zip(labelsthis, distances)]

    tmp_sorted = sorted(tmp, key=lambda x: x[1])

    label_map = {int(lab[0]):i for i, lab in enumerate(tmp_sorted)}
    label_map # maps gmm output to new label name

    # === use new labels
    labels = mod.predict(Xpca)
    print(labels)
    labels = np.array([label_map[l] for l in labels])
    print(labels)
    labellist = list(set(labels))

    if assign_as_column_in_SF:
        SF["label"] = labels
    else:
        if "label" in SF:
            del SF["label"]
    
    return labels, SF

def gmm_labels_resort(SF):
    """  arbitrarily resort gmm labels
    e.g., if want to renumber so sorted from most different
    across to leats.
    RETURNS:
    SF, in columns "label_resorted" (doenst modify in place)
    """
    # will sort by difference in frequencies between these datsaets
    dset1 = "Pancho_beh"
    dset2 = "Red_beh"

    # =====================
    dat = {}
    for g in SF.groupby("animal_dset"):
        tmp = tabulate_list(g[1]["label"])
        
        total = np.sum([v for v in tmp.values()])
        tmp = {k:v/total for k, v in tmp.items()}
        dat[g[0]] = tmp
        

    keys = sorted(set(SF["label"]))

    def _getProb(dset, key):
        if key not in dat[dset].keys():
            # then this doesnt even eixst. it is 0
            return 0
        return dat[dset][k]
        
    diffs = [] # list of tuples
    for k in keys:
        p2 = _getProb(dset2, k)
        p1 = _getProb(dset1, k)
    #     pdiff.append(p2 - p1)
        diffs.append((k, p2-p1))

    # sort by increasing
    diffs = sorted(diffs, key=lambda x:x[1])

    # assign new column to SF, using resorted labels
    map_labels_old_to_new = {}
    for i in range(len(diffs)):
        map_labels_old_to_new[diffs[i][0]] = i

    def F(x):
        return map_labels_old_to_new[x["label"]]
    if "label_resorted" in SF.columns:
        del SF["label_resorted"]
    SF = applyFunctionToAllRows(SF, F, "label_resorted")

    print("DONE - resorted labels are in SF.label_resorted")
    return SF


# == PLOT
def plotStrokOrderedByLabel(labels, SF, labels_in_order=None):
    """ plot example (rows) of each label(cols), ordred as in 
    labels_in_order.
    INPUTS:
    - labels_in_order, if None, then will use sorted(set(labels))
    - labels, list, same len as SF
    """

    # === for each cluster, plot examples
    if labels_in_order is None:
        labels_in_order = sorted(list(set(labels)))

    indsplot =[]
    titles=[]
    for ii in range(3):
        # collect inds
        for lab in labels_in_order:
            inds = [i for i, l in enumerate(labels) if l==lab]
            indsplot.append(random.sample(inds, 1)[0])
            if ii==0:
                titles.append(lab)
            else:
                titles.append('')

    # plot    
    stroklist = [SF["strok"].values[i] for i in indsplot]
    fig = plotStroksInGrid(stroklist, ncols=len(labels_in_order), titlelist=titles);

def extractX(SF, ver="tsne"):
    """ from SF, pull out array size N x d
    INPUT:
    ver, {"tsne", "pca", "sim"}
    RETURN:
    array
    """

    if ver=="tsne":
        key = "Xtsne"
    elif ver=="pca":
        key="Xpca"
    elif ver=="sim":
        key="Xsim"

    return np.stack(SF[key].values)


def plotTsneSeparateLabels(SF, SAVEDAT=None, labels=None, perp = 45):
    # == PREP
    
    # pull out saved tsne models
    # Xtsne = extractTsne(SAVEDAT, perp)
    # assert Xtsne.shape[0]==len(SF)
    Xtsne = extractX(SF, "tsne")

    # aasign each row a label, based on animal_dataset
    if labels is None:
        labels = SF["animal_dset"].values

    # overlay, separate
    fig, ax = plotScatterOverlay(Xtsne, labels, 
        ver="separate_no_background", alpha=0.08, downsample_auto=True)

def plotTsneSeparateLabelsHeatmap(SF, nbins = 40):
    """ heatmaps in tsne space, for all datasets split, and
    also comparing datsets
    """
    from pythonlib.tools.plottools import getHistBinEdges

    Xtsne = extractX(SF, "tsne")
    bins = [getHistBinEdges(Xtsne[:,0], nbins), getHistBinEdges(Xtsne[:,1], nbins)]

    nrows = 2
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(15,15))

    # == get vmin and vmax across figures
    h_all =[]
    for dset in set(SF["animal_dset"]):
        SFthis = SF[SF["animal_dset"]==dset]
        X = extractX(SFthis, "tsne")
        h = np.histogram2d(X[:,0], X[:,1], bins=bins, density=True)[0]
        h_all.append(h)
    vmin = np.min(np.concatenate([h[:] for h in h_all]))
    vmax = np.max(np.concatenate([h[:] for h in h_all]))

    # == plot, using common vmin and vmax
    for dset, ax in zip(set(SF["animal_dset"]), axes.flatten()):
        SFthis = SF[SF["animal_dset"]==dset]
        X = extractX(SFthis, "tsne")
        h = ax.hist2d(X[:,0], X[:,1], bins=bins, density=True, vmin=vmin, vmax=vmax, cmap="plasma");
        h_all.append(h[0][:])
        ax.set_title(dset)
        
    # == plot differences
    diffs_to_plot = [
        [0,1],
        [2,3],
        [0,2],
        [1,3],
    ]
    for d in diffs_to_plot:
        hthis = h_all[d[1]] - h_all[d[0]]
        
        v = np.max(np.abs(hthis[:]))
        
        plt.figure()
        plt.imshow(np.flipud(hthis.T), vmin=-v, vmax=v,  cmap="PuOr")
        plt.colorbar()
        plt.title(f"{d[1]} minus {d[0]}")



# 1) which slice (2 dimensions) to trake?
def plotScatter(X, SF, dims_to_take, nplot = 20, ax=None, 
                       color="k", textcolor="r", alpha=0.05):
    """ 
    Scatter plot of X, picking out 2 dimensions. useful is X is output after
    dim reduction. 
    INPUT:
    - X, array, shape N x D, where N is n samp, D is dim.
    - dims_to_take, list, which 2 dims to take, e.g., [0, 1]
    """
    from pythonlib.tools.linalgtools import plotScatterXreduced
    
    fig, ax, indsrand = plotScatterXreduced(X, dims_to_take, nplot, ax, color, textcolor, alpha, return_inds_text=True)
    
    # === plot each of these cases
    if SF is not None:
        strokstoplot = [SF["strok"][i] for i in indsrand]
        titlelist = [f"{i}" for i in indsrand]
        plotStroksInGrid(strokstoplot, titlelist=titlelist);

    return fig, ax


def prepSF(SF, SAVEDAT, perp):
    """ also assgins all things in SAVEDAT to SF, so if subsample SF later, is fine
    """

    # - expt, epoch
#     def F(x):
#         return f"{x['newcol']['expt']}-{x['newcol']['epoch']}"
    def F(x):
        return f"{x['expt']}-{x['epoch']}"
    SF = applyFunctionToAllRows(SF, F, "expt-epoch")

    # - Train, test
    if False:
        def F(x):
            return x['newcol']['traintest']
        SF = applyFunctionToAllRows(SF, F, "traintest")
        

    # - task 
        def F(x):
            return x['newcol']['task_stagecategory']
        SF = applyFunctionToAllRows(SF, F, "task_stagecategory")

    def F(x):
        return x['monkey_train_or_test']
    SF = applyFunctionToAllRows(SF, F, "traintest")

    print(SAVEDAT)
    print(SAVEDAT.keys())
    Xtsne = [x["D_fit"] for x in SAVEDAT["tsne_models"] if x["perp"]==perp][0]
    tmp = [x for x in Xtsne]
    SF["Xtsne"] = tmp

    Xsim = SAVEDAT["similarity_matrix"]
    tmp = [x for x in Xsim]
    SF["Xsim"] = tmp

    X = SAVEDAT["Xpca"]
    tmp = [xx for xx in X]
    SF["Xpca"] = tmp

    SF.reset_index(drop=True)
    return SF

def get_shared_tasks(SF, col_to_equalize_over="animal_dset"):
    """ finds all tasks that are done by all datasets.
    - col_to_equalize_over, will get min sample size across 
    all levels for this col.
    RETURNS:
    trials_per_task, dict, where keys are tasks (str) and
    values are num trials (minimum, across datasets)
    """
    # == get list of tasks that are common across all
    tasklist = SF["unique_task_name"].unique()
    _all = SF[col_to_equalize_over].unique()

    tasks_good = []
    for i, t in enumerate(tasklist):
        if i%500==0:
            print(i, t)
        dsets = SF[SF["unique_task_name"]==t][col_to_equalize_over].unique()
        good = all([d in dsets for d in _all])
        if good:
            # then keep this task
            tasks_good.append(t)
    print("This many shared tasks (out of total...):")
    print(len(tasks_good))
    print(len(tasklist))

    # for each good task, find minimum num trials. get that num trials from each dataset
    trials_per_task = {}
    for t in tasks_good:
        values = SF[SF["unique_task_name"]==t][col_to_equalize_over].value_counts()
        trials_per_task[t] = min(values.values)

    return trials_per_task

# for each shared task, pull out random N tasks across datasets
def get_SF_shared_tasks(SF, col_to_equalize_over="animal_dset"):
    """ 
    only uses tasks present for all dasets, and subsamples so
    exact same num rows for each dataset (for a task).
    RETURNS:
    """
    
    print("TODO, NOTE: shouuld instead match num trials, but havent coded")
    
    # 1) Get dict of shared tasks, and num trials
    trials_per_task = get_shared_tasks(SF, col_to_equalize_over=col_to_equalize_over)
    
    # 2) Subsample SF to get matched datseets.
    ndatasets = len(SF[col_to_equalize_over].unique())
    df_all = []
    for task, n in trials_per_task.items():
        def F(x):
            inds = random.sample(range(len(x)), n)
            return x.iloc[inds]

        # 1) get just for this task
        SFthis = SF[SF["unique_task_name"]==task]

        # extract n rows for each daset
        dfthis = SFthis.groupby(col_to_equalize_over).apply(F).reset_index(drop=True)
        assert len(dfthis)==n*ndatasets, "not sure why.."
        df_all.append(dfthis)

    SF_sharedtasks = pd.concat(df_all).reset_index(drop=True)

    print("New len of SF after matching tasks and num trials exactly:")
    print(len(SF_sharedtasks))
    
    return SF_sharedtasks

def plotHistDatasets(SF, label="label", sort_by="animal_dset", 
    shrink=1.):
    """ histogram over gmm labels
    INPUT:
    - label, {"label", "label_resorted"}
    """
    import seaborn as sns
    sns.displot(data=SF, x=label,hue=sort_by, stat="probability", common_norm=False, multiple="dodge", element="bars", shrink=shrink, aspect=2, height=5)
    sns.displot(data=SF, x=label, row=sort_by, stat="probability", common_norm=False, multiple="dodge", element="bars", shrink=shrink, aspect=2, height=3)


def plots(SDIR, overwrite, gmm_n = 20, perp=35, plotGrid=True):
    if overwrite==False:
        if os.path.exists(f"{SDIR}/figures"):
            return
    
    
    print(f"** Plotting this dir: {SDIR}")

    fname = f"{SDIR}/SAVEDAT.pkl"
    with open(fname, "rb") as f:
        SAVEDAT = pickle.load(f)

    fname = f"{SDIR}/SF.pkl"
    try:
        SF = pd.read_pickle(fname)
    except Exception:
        with open(fname, "rb") as f:
            SF = pickle.load(f)


    SDIRFIGS = f"{SDIR}/figures"
    os.makedirs(SDIRFIGS, exist_ok=True)

    # === add columns to SF
    SF = prepSF(SF, SAVEDAT, perp)

    # === pca plot
    pcamod = SAVEDAT["pca_model"]
    fig = plt.figure(figsize=(20,5))
    plt.plot(pcamod.explained_variance_ratio_, "ok-")
    plt.plot(np.cumsum(pcamod.explained_variance_ratio_), "or-")

    fig.savefig(f"{SDIRFIGS}/pca_variance.pdf")

    if False:
        dims_to_take = [0,1]
        fig, ax = plotScatter(Xpca, SF, dims_to_take)



    # === 4) Compute distance matrix between all stroks

    # # -- just some examples
    # idxs = random.sample(range(len(SF)), Nplot)
    # idxs_stroklist_dat = idxs
    # idxs_stroklist_basis = idxs


    # D = distMatrixStrok(idxs_stroklist_dat, idxs_stroklist_basis, stroklist=SF["strok"].values,
    #                    normalize_cols_range01=False)
    # plt.figure()
    # plt.imshow(D, cmap="plasma")
    # plt.colorbar()


    # D = distMatrixStrok(idxs_stroklist_dat, idxs_stroklist_basis, stroklist=SF["strok"].values,
    #                    normalize_cols_range01=True)
    # plt.figure()
    # plt.imshow(D, cmap="plasma")
    # plt.colorbar()


    # # === Visualize - condition on one item, take slices
    # plt.figure(figsize=(10, 10))
    # plt.imshow(D)
    # plt.colorbar()

    # N = 8
    # fig, axes = plt.subplots(ncols=5, nrows=2, sharex=True, sharey=True)
    # for i, ax in zip(range(N), axes.flatten()):
    #     ax.hist(D[:,i], 100);


    nrows = int(np.ceil(len(SAVEDAT["tsne_models"])/2))
    fig, axes = plt.subplots(nrows, 2, figsize=(15, 20))

    for mod, ax in zip(SAVEDAT["tsne_models"], axes.flatten()):
        Xtsne = mod["D_fit"]
        perp = mod["perp"]
        # print(SAVEDAT["tsne_models"])
        # Xtsne = out[0]["D_fit"]
        # Plot TSNE RESULTS
    #     labels = [0 for _ in range(len(Xtsne))]
        plotScatter(Xtsne, None, [0,1], ax=ax)
        ax.set_title(perp)
    #     plotScatterOverlay(Xtsne, labels)
        # plt.title(

    fig.savefig(f"{SDIRFIGS}/tsne-scatter-diffperplexities.pdf")

    # == select a model (TSNE)
    Xtsne = SAVEDAT["tsne_models"][3]["D_fit"]

    # TSNE PLOT - separate by epoch

    # 1) overlay
    labels = list(SF["expt-epoch"].values)
    # labels = [0 for _ in range(len(Xtsne))]
    fig, ax = plotScatterOverlay(Xtsne, labels, ver="separate", alpha=0.04)
    fig.savefig(f"{SDIRFIGS}/tsne-scatter-epochs.pdf")

    # 2) same, but only test tasks
    indsplot = SF["traintest"].values=="test"
    labels = list(SF["expt-epoch"].values)
    labels = np.array(labels)[indsplot]
    Xthis = Xtsne[indsplot, :]
    fig, ax = plotScatterOverlay(Xthis, labels, ver="separate", alpha=0.1)
    fig.savefig(f"{SDIRFIGS}/tsne-scatter-epochs-testonly.pdf")

    # == select a model (GMM)
    out = SAVEDAT["gmm_models"]
    mod = gmm_extract_model(SAVEDAT, gmm_n)

    # Plotting for GMM results
    dfthis = pd.DataFrame(out)

    plt.figure()
    fig = sns.lineplot(x="n", y="bic", data=dfthis).get_figure()
    fig.savefig(f"{SDIRFIGS}/gmm-bic.pdf")

    plt.figure()
    fig = sns.lineplot(x="n", y="cross_val_score", data=dfthis).get_figure()
    fig.savefig(f"{SDIRFIGS}/gmm-ll.pdf")

    # select GMM model
    Xpca = SAVEDAT["Xpca"]
    labels, SF = gmm_labels(SAVEDAT, gmm_n, SF)
    labellist = sorted(list(set(labels)))


    # == plot scatter of tsne, collored by label
    # labels = 
    fig, ax = plotScatterOverlay(Xtsne, labels = labels)
    fig.savefig(f"{SDIRFIGS}/tsne-scatter-gmmlabels.pdf")

    # fig, ax = plotScatter(Xtsne, SF, [0,1], color="k", textcolor="k", alpha=0.02)

    # ax.scatter(Xtsne[:,0], Xtsne[:,1], c=mod.predict(Xin), alpha=0.15, cmap="rainbow")


    # === plot a grid, based on percentiles along 2 dimensions
    # Xfit = out[0]["D_fit"]
    # 1) assign all indices to a position in grid, based on percentiles
    if plotGrid:
        Xfit = Xtsne
        values1 = Xfit[:,0]
        values2 = Xfit[:,1]
        idxslist = range(Xfit.shape[0])

        nbins = 20
        p = np.linspace(0, 100, nbins)

        binedges = np.percentile(values1, p)
        inds1 = np.digitize(values1, binedges)

        binedges = np.percentile(values2, p)
        inds2 = np.digitize(values2, binedges)
        # for each combo of inds, plot an example trial
        indslist = set(np.r_[inds1, inds2])
        fig, axes = plt.subplots(len(indslist), len(indslist), sharex=True, sharey=True, figsize=(len(indslist)*2, len(indslist)*2))
        for i1 in indslist:
            for ii2, i2 in enumerate(np.array([i for i in indslist])[::-1]): # so can go backwards.. with bottom left as 1,1
                print(i1, i2)
                ax = axes[ii2-1][i1-1]
                indsthis = list(np.where((inds1==i1) & (inds2==i2))[0])
                if len(indsthis)==0:
                    continue

                ind = random.sample(indsthis,1)[0]

                strokthis = SF["strok"][ind]
                plotDatStrokes([strokthis], ax, pcol="r")
                ax.axhline(0, color='k', alpha=0.3)
                ax.axvline(0, color='k', alpha=0.3)
                ax.set_title(f"{i1}-{i2}")
                M = 300
                ax.set_xlim([-M, M])
                ax.set_ylim([-M, M])    

        fig.savefig(f"{SDIRFIGS}/tsne-behgrid-ll.pdf")


    # === VISUALIZE CLUSTERS
    # for each cluster, plot examples at different extremes (and also centroid).
    fig = plt.figure(figsize=(5, 10))
    labelsprob = mod.predict_proba(Xpca)
    plt.imshow(labelsprob[2::500], vmin=0., vmax=1.)
    plt.colorbar()
    plt.xlabel('gmm label')
    plt.ylabel('trial')
    fig.savefig(f"{SDIRFIGS}/gmmlabels-heat-examplescores.pdf")



    # === for each cluster, plot examples
    nplot = 20
    for lab in labellist:
        inds = [i for i, l in enumerate(labels) if l==lab]

        indsplot = random.sample(inds, nplot)

        stroklist = [SF["strok"].values[i] for i in indsplot]
        fig = plotStroksInGrid(stroklist, titlelist=indsplot)
        fig.savefig(f"{SDIRFIGS}/gmmlabels-examplebeh-{lab}.pdf")

    # == reorganize heat mpa based on labels
    X = SAVEDAT["Xpca"]

    # --- before sorting
    fig, ax = plt.subplots(figsize=(5, 50))
    Xplot = X[::100]
    labelsplot = labels[::100]
    ax.imshow(Xplot)
    ax.set_yticks(range(len(labelsplot)));
    ax.set_yticklabels(labelsplot);

    fig.savefig(f"{SDIRFIGS}/pcafeatures-heat-beforesortbygmm.pdf")


    # -- sort
    A = [[XX, ll] for  XX, ll in zip(Xplot, labelsplot)]
    A = sorted(A, key=lambda x:x[1])
    Xsorted = np.stack([AA[0] for AA in A])
    labelssorted = [AA[1] for AA in A]

    fig, ax = plt.subplots(figsize=(5, 50))
    ax.imshow(Xsorted)
    ax.set_yticks(range(len(labelssorted)));
    ax.set_yticklabels(labelssorted);
    fig.savefig(f"{SDIRFIGS}/pcafeatures-heat-aftersortbygmm.pdf")


    # === plot frac strokes within each category
    if True:
        # y = []
        # for l in labellist:
        #     y.append(len([lab for lab in labels if lab == l]))


        epochs = set(list(SF["expt-epoch"].values))
        fig = plt.figure(figsize=(10,5))
        for e in epochs:
            inds = np.where(SF["expt-epoch"].values==e)[0]
            labelthis = [labels[i] for i in inds]

            binedges = np.r_[list(labellist), list(labellist)[-1]+1]-0.5
            xy = plt.hist(labelthis, bins=binedges, density=True, histtype="step")
        #     plt.plot(binedges[:-1]+0.5, xy[0])
        plt.legend(epochs)
        fig.savefig(f"{SDIRFIGS}/gmmlab-hist-bars.pdf")

        # Tighter restriction - only test
        epochs = set(list(SF["expt-epoch"].values))
        fig = plt.figure(figsize=(10,5))
        for e in epochs:

            inds = np.where(
                (SF["expt-epoch"].values==e) & (SF["traintest"].values=="test") & np.isin(SF["task_stagecategory"], ["2linePlusL", "3linePlusL", "LplusL", "linePlusLv2"])
            )[0]

            labelthis = [labels[i] for i in inds]

            binedges = np.r_[list(labellist), list(labellist)[-1]+1]-0.5
        #     xy = plt.hist(labelthis, bins=binedges, density=True, histtype="step")
    #         xy = np.histogram(labelthis, bins=binedges, density=True)
            xy = plt.hist(labelthis, bins=binedges, density=True, histtype="step")
        plt.legend(epochs)
        fig.savefig(f"{SDIRFIGS}/gmmlab-hist-bars-onlytest-onlylines5tasks.pdf")

    # === bootstrap to get category membership
    binedges = np.r_[list(labellist), list(labellist)[-1]+1]-0.5
    Nboot = 100
    only_lines5_tests = False # then restricts to common tasks
    only_tests = True # overwrite above
    result_boot =[]
    for e in epochs:

        if only_tests:
            inds = np.where(
                (SF["expt-epoch"].values==e) & (SF["traintest"].values=="test"))[0]
        elif only_lines5_tests:
            inds = np.where(
                (SF["expt-epoch"].values==e) & (SF["traintest"].values=="test") & np.isin(SF["task_stagecategory"], ["2linePlusL", "3linePlusL", "LplusL", "linePlusLv2"])
            )[0]
        else:
            inds = np.where(SF["expt-epoch"].values==e)[0]

        labelthis = [labels[i] for i in inds]

        valsboot = []
        K = len(labelthis)
        for n in range(Nboot):
            labelrand = random.choices(labelthis, k=K)

            #  get bin occupancies
            xy = np.histogram(labelrand, bins=binedges, density=True)
            valsboot.append(xy[0])

            for val, b in zip(xy[0], xy[1]):
                result_boot.append({
                    "epoch":e,
                    "nboot":n,
                    "bin":int(b+0.5),
                    "val":val})
    #     result_boot.append({
    #             "epoch":e,
    #             "valsboot":np.stack(valsboot)})




    df_boot = pd.DataFrame(result_boot)

    # sns.catplot(data=df_boot, x="bin", y="val", hue="epoch", kind="point", aspect = 2, linestyles="")
    fig = sns.catplot(data=df_boot, x="bin", y="val", hue="epoch", aspect = 2, alpha=0.4)
    fig.savefig(f"{SDIRFIGS}/gmmlab-hist-bootstrapped-onlytest-overlay.pdf")


    sns.catplot(data=df_boot, x="bin", y="val", hue="epoch", aspect = 2, row="epoch", alpha=0.4)
    fig.savefig(f"{SDIRFIGS}/gmmlab-hist-bootstrapped-onlytest-separate.pdf")


    # -- plot single example of each bin, ordred
    # === for each cluster, plot examples
    indsplot =[]
    titles=[]
    for ii in range(3):
        # collect inds
        for lab in labellist:
            inds = [i for i, l in enumerate(labels) if l==lab]
            indsplot.append(random.sample(inds, 1)[0])
            if ii==0:
                titles.append(lab)
            else:
                titles.append('')

    # plot    
    stroklist = [SF["strok"].values[i] for i in indsplot]
    fig = plotStroksInGrid(stroklist, ncols=len(labellist), titlelist=titles);
    fig.savefig(f"{SDIRFIGS}/gmmlab-hist-examplebeh.pdf")

    # === justification for using gmm
    # Xsim = SAVEDAT["similarity_matrix"]
    X = SAVEDAT["Xpca"]

    # df = pd.DataFrame(Xpca)
    df = pd.DataFrame(X)
    dims=range(5)
    fig, axes = plt.subplots(5, 2, figsize=(12, 30))
    c=0
    for d1 in range(5):
        for d2 in range(d1+1, 5):
            ax = axes.flatten()[c]
            c+=1
            x = df[d1][::2]
            y = df[d2][::2]
            lab = np.array(labels)[::2]
            ax.scatter(x, y, c=lab, alpha=0.2, marker="x", cmap = "rainbow")
            ax.set_xlabel(d1)
            ax.set_ylabel(d2)
    #         ax.legend(set(labels))


    fig.savefig(f"{SDIRFIGS}/pairwisescatters-pca-colorbygmmlabel.pdf")



    # === justification for using gmm (SAME AS ABOVE, USING SIM MATRIX)
    X = SAVEDAT["similarity_matrix"]

    # df = pd.DataFrame(Xpca)
    df = pd.DataFrame(X)

    dims=range(5)
    fig, axes = plt.subplots(5, 2, figsize=(12, 30))
    c=0
    for d1 in range(5):
        for d2 in range(d1+1, 5):
            ax = axes.flatten()[c]
            c+=1
            x = df[d1][::2]
            y = df[d2][::2]
            lab = np.array(labels)[::2]
            ax.scatter(x, y, c=lab, alpha=0.2, marker="x", cmap = "rainbow")
            ax.set_xlabel(d1)
            ax.set_ylabel(d2)
    #         ax.legend(set(labels))


    fig.savefig(f"{SDIRFIGS}/pairwisescatters-simmatrix-colorbygmmlabel.pdf")

    # == plot heatmap of distances
    Xsim = SAVEDAT["similarity_matrix"]
    fig = plt.figure(figsize=(15, 15))
    plt.imshow(Xsim[::100], cmap="plasma")
    plt.colorbar()
    plt.xlabel("basis_strokes")
    plt.ylabel("data_strokes")
    fig.savefig(f"{SDIRFIGS}/simmatrix-heatmap-colorbygmmlabel.pdf")    


def loadSF(SDIR):
    """ load saved cluistering analyses
    """
    fname = f"{SDIR}/SAVEDAT.pkl"
    with open(fname, "rb") as f:
        SAVEDAT = pickle.load(f)

    fname = f"{SDIR}/SF.pkl"
    SF = pd.read_pickle(fname)
    
    return SF, SAVEDAT


############################################### MULTIPROCESSING


def main_cluster(animal, distancever):
    exptlist = ["arc2", "lines5", "figures89"]
    # Nbasis = 5
    # gmm_n_mixtures = list(range(1, 3)) 
    # perplist = [25]
    Nbasis = 200
    gmm_n_mixtures = list(range(5, 27))
    perplist = [15, 25, 45, 55, 65]

    similarityClustering(animal, exptlist, 
        Nbasis = Nbasis, PCAdim = 5, gmm_n_mixtures = gmm_n_mixtures,
        perplist = perplist, rescale_strokes_ver = "stretch_to_1", 
        distancever = distancever, npts_space = 50)


def main_cluster_2(animal, strokes_ver, distancever="euclidian_diffs", 
    just_get_SF=False):
    # Combine multiple (pre extracted) datasets, convert to a single SF, and analyse.
    # just_get_SF, Can choose to just save the SF and metadat, without running entire pipeline.

    assert False, "obsolete I think"

    if animal=="Red":
        path_list = [
            "/data2/analyses/database/Red-lines5-formodeling-210329_005719",
            "/data2/analyses/database/Red-arc2-formodeling-210329_005550",
            "/data2/analyses/database/Red-shapes3-formodeling-210329_005200",
            "/data2/analyses/database/Red-figures89-formodeling-210329_005443"
        ]
    elif animal=="Pancho":
        path_list = [
            "/data2/analyses/database/Pancho-lines5-formodeling-210329_014835",
            "/data2/analyses/database/Pancho-arc2-formodeling-210329_014648",
            "/data2/analyses/database/Pancho-shapes3-formodeling-210329_002448",
            "/data2/analyses/database/Pancho-figures89-formodeling-210329_000418"
        ]
    load_params = {
        "ver":"dataset",
        "inputs":path_list,
        "strokes_ver":strokes_ver
    }

    Nbasis = 200
    gmm_n_mixtures = list(range(5, 27))
    perplist = [15, 25, 35, 45, 55, 65]
    
    SF = loadStrokeDat(load_params)
    
    savename = strokes_ver

    similarityClusteringFromSF(SF, 
        Nbasis = Nbasis, PCAdim = 5, gmm_n_mixtures = gmm_n_mixtures,
        perplist = perplist, rescale_strokes_ver = "stretch_to_1", 
        distancever = distancever, npts_space = 50, savename=savename,
        just_get_SF=just_get_SF)


def main_cluster_4(animallist, strokes_ver_list, DEBUG = False, skip_preprocess=False):
    """ Compute similarity matrix and do cllustering.
    """
    ### now load multiple datasets, and concat SF, preprocess and save
    if not skip_preprocess:
        D = Dataset([])
        D.load_dataset_helper(animallist, expt, "mult")
        D.sf_load_preextracted(strokes_ver_list=strokes_ver_list)

        if DEBUG:
            D.SF = D.SF[:100]
            
        # preprocess SF
        D.sf_preprocess_stroks()
        D.sf_save_combined_sf()

    ### FINAL LOAD BEFORE GET EMBEDDINGS
    D = Dataset([])
    D.load_dataset_helper(animallist, expt, "mult")
    D.sf_load_combined_sf(animallist, [expt], strokes_ver_list)

    # Load SFs, and perform embeddings and clustering
    if DEBUG:
        similarity_matrix, idxs_stroklist_basis, params = D.sf_embedding_bysimilarity(Nbasis=5)
    else:
        similarity_matrix, idxs_stroklist_basis, params = D.sf_embedding_bysimilarity()

    ### MODEL the sim matrix
    from pythonlib.tools.clustertools import clusterSimMatrix

    if DEBUG:
        DAT = clusterSimMatrix(similarity_matrix, gmm_n_mixtures=[4], perplist = [35])
    else:
        DAT = clusterSimMatrix(similarity_matrix)

    # Save clustering results

    s = params["path_embeddings_similarity"]
    sdir = f"{s}/clustering"
    os.makedirs(sdir, exist_ok=True)

    path = f"{sdir}/DAT.pkl"
    with open(path, "wb") as f:
        pickle.dump(DAT, f)
    print(path)        


def main_cluster_5(animallist, strokes_ver_list, list_rule, DEBUG = False):
    """ 
    Takes precomputed sim matrix and clusters, and saves
    - Does not plot anything
    NOTE: similar to 4, but here using pre-saved sim matrix.
    """

    ### FINAL LOAD BEFORE GET EMBEDDINGS
    D = Dataset([])
    D.load_dataset_helper(animallist, expt, "mult", rule=list_rule)
    D.sf_load_combined_sf(animallist, [expt], strokes_ver_list)

    # Load SFs, and perform embeddings and clustering
    pathlist = findPath(D.SFparams["path_sf_combined"], 
             [["embeddings"], ["similarity", "stretch_to_1", "euclidian_diffs"]], "", "", True)

    if len(pathlist)>1 or len(pathlist)==0:
        assert False
        
    pathdat = f"{pathlist[0]}/dat.pkl"
    with open(pathdat, "rb") as f:
        dat = pickle.load(f)
        
    pathdat = f"{pathlist[0]}/params.pkl"
    with open(pathdat, "rb") as f:
        params = pickle.load(f)
    similarity_matrix = dat["similarity_matrix"]

    ### MODEL the sim matrix
    from pythonlib.tools.clustertools import clusterSimMatrix

    if DEBUG:
        DAT = clusterSimMatrix(similarity_matrix, gmm_n_mixtures=[4], perplist = [35])
    else:
        DAT = clusterSimMatrix(similarity_matrix)

    # Save clustering results

    s = params["path_embeddings_similarity"]
    sdir = f"{s}/clustering"
    os.makedirs(sdir, exist_ok=True)

    path = f"{sdir}/DAT.pkl"
    with open(path, "wb") as f:
        pickle.dump(DAT, f)
    print(path)        


# def main(arg1, arg2):


# def main(arg1, arg2):
#     print(arg1, arg2)

if __name__=="__main__":

    VER = 5

    if VER==1:
        # === OLD VERSION, BEFORE INCORPORATING PARSES
        # exptlist = ["arc2"]
        # exptlist = ["arc2", "lines5", "figures89"]
        animallist = ["Red", "Pancho"]
        distancelist = ["euclidian_diffs", "euclidian", "hausdorff_means"]

        args1 = []
        args2 = []
        for animal in animallist:
            for dist in distancelist:
                args1.append(animal)
                args2.append(dist)

        from multiprocessing import Pool
        with Pool(4) as pool:
            pool.starmap(main_cluster, zip(args1, args2))

    elif VER==2:
        # === NEW VERSION (can do parsing control)
        # Can choose to ether save SF indiv, or to process.
        animallist = ["Red", "Pancho"]
        # distancelist = ["euclidian_diffs", "euclidian", "hausdorff_means"]
        # strokes_ver_list = ["strokes_beh_splines", "strokes_beh", "strokes_parse"]
        strokes_ver_list = ["strokes_beh_splines"]
        args1 = []
        args2 = []
        for animal in animallist:
            for s in strokes_ver_list:
                args1.append(animal)
                args2.append(s)

        from multiprocessing import Pool
        with Pool(4) as pool:
            pool.starmap(main_cluster_2, zip(args1, args2))

    elif VER==3:
        # Combining multiple previuos cases of saved SF, to do new SF.

        Nbase_list = [300, 400]
            # [200, 300, 400, 500, 800]:

        if False:
            # Old way of collecting indiv SFs
            # exptlist = ["arc2", "lines5", "figures89"]
            exptlist = ["lines5", "figures", "shapes3"]
            animallist = ["Red", "Pancho"]
            distancelist = ["euclidian_diffs", "euclidian", "hausdorff_means"]
            SDIRMAIN = "/data2/analyses/database/clustering/bysimilarity"
            strokesver_list = ["beh", "parse"]
            rescale = "stretch_to_1"
            overwrite = False
            dry_run=True

            import glob
            SDIR_list = []
            for anim in animallist:
                for distver in distancelist:
                    for sver in strokesver_list:
                        dirlist = glob.glob(f"{SDIRMAIN}/*{'*'.join(exptlist)}*-rescale_{rescale}-dist_{distver}-strokes_{sver}/{anim}*")
                        for d in dirlist:

                            SDIR = d
                            print(SDIR)
                            if not dry_run:
                                plots(SDIR, overwrite, gmm_n=14, plotGrid=True)
                                
                            SDIR_list.append(SDIR)
        else:
            from pythonlib.tools.expttools import findPath
            # Load paths for indiv data
            path_base = "/data2/analyses/database/clustering/bysimilarity/indiv"
            path_fname = "SAVEDAT"
            ext = ".pkl"

            path_hierarchy = [
                ["stretch_to_1", "dist_euclidian_diffs", "strokes_beh_splines"],
                [""]]
            SDIRlist1 = findPath(path_base, path_hierarchy, path_fname, ext, return_without_fname=True)

            path_hierarchy = [
                ["stretch_to_1", "dist_euclidian_diffs", "strokes_parse"],
                [""]]
            SDIRlist2 = findPath(path_base, path_hierarchy, path_fname, ext, return_without_fname=True)

            SDIR_list = SDIRlist1
            SDIR_list.extend(SDIRlist2)
            print("=== GETTING THESE SF indivs")
            print(len(SDIR_list))
            [print(s) for s in SDIR_list]
                                    
        # Load multiple presaved SF, and concatnate.
        SFall =[]
        print("LOADING SF:")
        for SDIR in SDIR_list:
            path = f"{SDIR}/SF.pkl"
            with open(path, "rb") as f:
                SF = pickle.load(f)

            # add column to map back to original dataset
            SF["path_to_sf"] = path

            if "Dataset" in SF.columns:
                del SF["Dataset"]
            print("---")
            print(path)
            print("extracted SF, length")
            print(len(SF))
            SFall.append(SF)
                
        # concatnate
        import pandas as pd
        # SF = SFall[0]
        SF = pd.concat(SFall)
        SF = SF.reset_index(drop=True)

        # == DRY RUN
        # similarityClusteringFromSF(SF, Nbasis = 3, PCAdim = 2, gmm_n_mixtures = list(range(5, 6)), 
        #         perplist = [5], rescale_strokes_ver = "stretch_to_1", 
        #         distancever = "euclidian_diffs", npts_space = 20,
        #                            savename="combined_parse_beh")
        for N in Nbase_list:
            similarityClusteringFromSF(SF, Nbasis=N, savename="combined_parse_beh_savingpath")



    elif VER==4:

        # This can run from scratch (would turn on section saying "DO_INITIAL_EXTRACTION"). if already
        # extracted SF for each dataset, then just turn that False.
        # compared to previous VEr, this ver uses Dataset and is good.

        DO_INITIAL_EXTRACTION = True

        from pythonlib.dataset.dataset import Dataset
        # expt = "lines5"
        # strokes_ver_list = ["strokes_beh_splines", "strokes_parse"]
        # animallist = ["Pancho", "Red"]


        # RUN THIS FIRST TO EXTRACT ALL STROKES
        expt = "gridlinecircle"
        # strokes_ver_list = ["strokes_beh_splines", "strokes_parse"]
        strokes_ver_list = ["strokes_beh_splines"]
        animallist = ["Pancho"]
        list_rule = ["baseline", "lolli"]

        if DO_INITIAL_EXTRACTION:
            for a in animallist:
                for rule in list_rule:
                    for sver in strokes_ver_list:
                        D = Dataset([])
                        D.load_dataset_helper(a, expt, ver="single", rule=rule)

                        # Extract single strokes feat 
                        D.sf_extract_and_save(strokes_ver = sver)


        # THEN GET ALL SUBSETS OF ANIMALLIST AND STROKESVERLIST TO DO ANALYSIS'
        # first get all combinations
        from itertools import chain, combinations

        def powerset(iterable):
            "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
            s = list(iterable)
            return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

        animallist_list = list(powerset(animallist))[1:]
        strokesverlist_list = list(powerset(strokes_ver_list))[1:]

        # === NEW VERSION (can do parsing control)
        args1 = []
        args2 = []
        # Run analysis for all single and combined datasets
        for animallist in animallist_list:
            for strokes_ver_list in strokesverlist_list:
                
                if "Pancho" in animallist and "Red" in animallist:
                    # then skip some things.
        #             if len(strokes_ver_list)==1 and "strokes_parse" in strokes_ver_list:
        #                 continue
                    if len(strokes_ver_list)==1:
                        continue

                args1.append(animallist)
                args2.append(strokes_ver_list)

        from multiprocessing import Pool
        with Pool(4) as pool:
            pool.starmap(main_cluster_4, zip(args1, args2))

    elif VER==5:
        # If have already extracted and saved similiarty matrix (i.e., ran cluster 4, but 
        # bug at line 1391, then instead of running 4, run 5.)
        from pythonlib.dataset.dataset import Dataset
        # expt = "lines5"
        # strokes_ver_list = ["strokes_beh_splines", "strokes_parse"]
        # animallist = ["Pancho", "Red"]

        expt = "gridlinecircle"
        strokes_ver_list = ["strokes_beh_splines"]
        animallist = ["Pancho"]
        list_rule = ["baseline", "lolli"]
        MULTI = False

        # RUN THIS FIRST TO EXTRACT ALL STROKES
        # THEN GET ALL SUBSETS OF ANIMALLIST AND STROKESVERLIST TO DO ANALYSIS'
        # first get all combinations
        from itertools import chain, combinations

        def powerset(iterable):
            "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
            s = list(iterable)
            return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

        animallist_list = list(powerset(animallist))[1:]
        strokesverlist_list = list(powerset(strokes_ver_list))[1:]
        rulelist_list = list(powerset(list_rule))[1:]

        # === NEW VERSION (can do parsing control)
        args1 = []
        args2 = []
        args3 = []
        # Run analysis for all single and combined datasets
        for animallist in animallist_list:
            for strokes_ver_list in strokesverlist_list:
                for list_rule in rulelist_list:
                    if "Pancho" in animallist and "Red" in animallist:
                        # then skip some things.
            #             if len(strokes_ver_list)==1 and "strokes_parse" in strokes_ver_list:
            #                 continue
                        if len(strokes_ver_list)==1:
                            continue

                    args1.append(animallist)
                    args2.append(strokes_ver_list)
                    args3.append(list_rule)


        if MULTI:
            from multiprocessing import Pool
            with Pool(4) as pool:
                pool.starmap(main_cluster_5, zip(args1, args2, args3))
        else:
            for arg1, arg2, arg3 in zip(args1, args2, args3):
                main_cluster_5(arg1, arg2, arg3)
