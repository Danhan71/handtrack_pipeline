import numpy as np

def clusterSimMatrix(similarity_matrix, PCAdim = None, PCAdim_touse = 5,
        gmm_n_mixtures = tuple(range(8, 30)), 
        perplist = (8, 15, 25, 35, 45, 55, 65),
        gmm_tsne_perp_to_use = None,
        things_to_do = ("tsne", "gmm", "gmm_using_tsne")):
    """ [GOOD] Wrapper to do:
    Given precomputed sim matrix, do various kinds of clustering, and 
    return dict with results
    - Not plotting...
    PARAMS:
    - similarity_matrix, (ndat, nfeat)
    - PCAdim, max num dims to keep, both for output
    - PCAdim_touse, int, numdims to use for tsne and gmm
    """
    from sklearn.decomposition import PCA 
    from sklearn.manifold import TSNE
    from sklearn.model_selection import train_test_split
    from sklearn.mixture import GaussianMixture as GMM

    # 1) Use output of PCA for below modeling
    if PCAdim is not None:
        assert PCAdim==similarity_matrix.shape[1]
    pca_model = PCA(n_components=PCAdim)
    Xpca = pca_model.fit_transform(similarity_matrix) # (nsamps, nPCs)
    Xpca_input_models = Xpca[:, :PCAdim_touse]


    # -- TSNE
    if "tsne" in things_to_do:
        out = []
        for perp in perplist:
            D_tsne = TSNE(n_components=2, perplexity=perp).fit_transform(Xpca_input_models)
            out.append({
                "perp":perp,
                "D_fit":D_tsne
            })
        models_tsne = out
    else:
        models_tsne = None


    # === PRELIM, fitting GMM to PCA-transformed data
    if "gmm" in things_to_do:
        covariance_type="full"
        nsplits = 1
        n_init = 1
        out = []
        for isplit in range(nsplits):
            Xtrain, Xtest = train_test_split(Xpca_input_models, test_size=0.1)
            for n in gmm_n_mixtures:
                gmm = GMM(n_components=n, n_init=1, covariance_type=covariance_type)
                gmm.fit(Xtrain)
                out.append({
                    "mod":gmm,
                    "n":n, 
                    "isplit":isplit,
                    "bic":gmm.bic(Xtest),
                    "cross_val_score":gmm.score(Xtest)
                })
        models_gmm = out
    else:
        covariance_type=None
        models_gmm = None

    if "gmm_using_tsne" in things_to_do:
        assert "tsne" in things_to_do
        assert gmm_tsne_perp_to_use in perplist, "didnt get this perp...?"

        this = [mod for mod in models_tsne if mod["perp"]==gmm_tsne_perp_to_use][0]
        X = this["D_fit"]
        covariance_type="full"
        nsplits = 1
        n_init = 1
        out = []
        for isplit in range(nsplits):
            Xtrain, Xtest = train_test_split(X, test_size=0.1)
            for n in gmm_n_mixtures:
                gmm = GMM(n_components=n, n_init=1, covariance_type=covariance_type)
                gmm.fit(Xtrain)
                out.append({
                    "mod":gmm,
                    "n":n, 
                    "isplit":isplit,
                    "bic":gmm.bic(Xtest),
                    "cross_val_score":gmm.score(Xtest)
                })
        models_gmm_using_tsne = out
        Xtsne_input_gmm = X
    else:
        covariance_type=None
        models_gmm_using_tsne = None
        Xtsne_input_gmm= None

    # OUTPUT
    DAT = {
        "pca_model":pca_model,
        "Xpca":Xpca,
        "Xpca_input_models":Xpca_input_models,
        "Xtsne_input_gmm":Xtsne_input_gmm,
        "models_tsne":models_tsne,
        "models_gmm":models_gmm,
        "models_gmm_using_tsne":models_gmm_using_tsne,
        "similarity_matrix":similarity_matrix,
        "gmm_n_mixtures":gmm_n_mixtures,
        "perplist":perplist,
        "gmm_covariance_type":covariance_type
    }

    return DAT


################### SORTING
# def sort_by_labels(X, labels, axis=0):
#     """ Sort X by labels, in incresaing order.
#     PARAMS:
#     - X, ndat x ndim np array, rows will be osrted.
#     - labels, list of ints, length ndat
#     - axis, dimeision to srot byt. if 0, then sorts rows...
#     RETURNS:
#     - X, labels, but sorted (copies)
#     """
    
#     inds = np.argsort(labels)

#     if axis==0:
#         X = X[inds,:]
#     elif axis==1:
#         X = X[:, inds]
#     else:
#         assert False

#     labels = [labels[i] for i in inds]
    
#     return X, labels

################### PLOTS
# OBSOLETE
def plot_examples_grouped_by_label_columns(SF, labels, labellist_toplot,
        n_examples = 3):
    """ columns are labels, rows are n examples. Useful to align to bottom
    of a histogram plot of frequencies for each label in dataset
    PARAMS;
    - SF, dataframe with "strok"
    - labels, list of label names associated with SF
    - labellist_toplot, list of names in order to plot
    - n_examples, num rows (unique eg)
    """
    # -- plot single example of each bin, ordred
    assert False, "obsolete..."
    
    from pythonlib.drawmodel.strokePlots import plotStroksInGrid
    import random

    assert len(SF) == len(labels)

    # === for each cluster, plot examples
    indsplot =[]
    titles=[]
    for ii in range(n_examples):
        # collect inds
        for lab in labellist_toplot:
            inds = [i for i, l in enumerate(labels) if l==lab]
            indsplot.append(random.sample(inds, 1)[0])
            if ii==0:
                titles.append(lab)
            else:
                titles.append('')

    # plot    
    stroklist = [SF["strok"].values[i] for i in indsplot]
    fig = plotStroksInGrid(stroklist, ncols=len(labellist_toplot), titlelist=titles);
#     fig.savefig(f"{SDIRFIGS}/gmmlab-hist-examplebeh.pdf")

