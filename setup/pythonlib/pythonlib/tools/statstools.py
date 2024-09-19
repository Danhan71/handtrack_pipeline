## bunch of stuff for doings stats, focussing on using pandas

from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd

def signrank_wilcoxon(x1, x2=None):
    return stats.wilcoxon(x1, y=x2) 

def ttest_paired(x1, x2=None, ignore_nan=False):
    """ x1, x2 arrays, with same lenght
    if x2 is None, then assumes is comparing to 0s.
    Automatically removes any rows with nans
    RETURNS:
    - ttest result object.
    """
    if x2 is None:
        x2 = np.zeros_like(x1)

    if ignore_nan:
        nan_policy='omit'
    else:
        nan_policy = 'propagate'
    return stats.ttest_rel(x1, x2, nan_policy=nan_policy)

def zscore(val, vals_dist):
    """ zscore val relative to the vals_dist distribtion"""
    return (val - np.mean(vals_dist)) / np.std(vals_dist)
    
# def statsTtestPaired(df1, df2, varname, pairedon="human"):
#     """ttest between two variables, assume it is paired at level of human (i.e, eachhuman 1pt each var pts)"""
#     df12 = pd.merge(df1, df2, on=pairedon)
# #     print(df12)
#     return stats.ttest_rel(df12[f"{varname}_x"], df12[f"{varname}_y"])

# def statsTtestUnpaired(df1, df2, varname):
#     return stats.ttest_ind(df1[f"{varname}"], df2[f"{varname}"])

# def statsRanksum(df1, df2, varname):
#     return stats.ranksums(df1[f"{varname}"], df2[f"{varname}"])

# NOT YET DONE!!!
if False:
    def getStatsOLS(dfthis_teststimonly, colname):
        from statsmodels.regression.linear_model import OLS
        from statsmodels.tools import add_constant

        res_dict = {}
        
        X = dfthis_teststimonly[colname].values.reshape(-1,1)
        y = dfthis_teststimonly["x4"].values.reshape(-1,1)
        X = add_constant(X, prepend=False)

        res = OLS(y, X).fit()
        res_dict["DC_vs_planner"]=res

        X = dfthis_teststimonly[colname].values.reshape(-1,1)
        y = dfthis_teststimonly["human_cond"].values.reshape(-1,1)
        X = add_constant(X, prepend=False)

        res = OLS(y, X).fit()
        res_dict["DC_vs_humancond"]=res
        
        return res_dict

def empiricalPval(stat_actual, stats_shuff, side="two"):
    """ p value, useful for permutation tests.
    - side, if "two" then two sided (uses absolute value), 
    if "left" then hypothesize that actual is small, if 'right' then large.
    """    
    if side=="two":
        n = sum(np.abs(stats_shuff)<=np.abs(stat_actual)) + 1
    elif side=="left":
        n = sum(stats_shuff<=stat_actual) + 1
    elif side=="right":
        n = sum(stats_shuff>=stat_actual) + 1
    else:
        assert False, "not coded"
    nn = len(stats_shuff) + 1
    p = n/nn

    return p



def permutationTest(data, funstat, funshuff, N, plot=True, side="two"):
    """ generic permutation test function
    - funstat(data) returns a scalar. if funstat returns a tuple (a,b), then 
    treats a as the scalar for statistics, and treats b as values that will collect
    and output after doing shuffle, as [a1, a2, ...]
    - funshuff(data) return a copy of data, shuffled
    - N is how many permutations
    - side, if "two" then two sided (uses absolute value), 
    if "left" then hypothesize that actual is small, if 'right' then large.
    """

    X = funstat(data)
    if isinstance(X, tuple) or isinstance(X, list):
        assert len(X)==2
        # then outputing list/tuple
        collect_shuffstats=True
    else:
        collect_shuffstats=False

    if collect_shuffstats:
        stat_actual, stat_actual_collected = funstat(data)
    else:
        stat_actual = funstat(data)

    stats_shuff = []
    stats_shuff_collected = []
    for i in range(N):
        if i%50==0:
            print(f"shuffle # {i}")
        data_shuff = funshuff(data)
        if collect_shuffstats:
            X = funstat(data_shuff)
            stats_shuff.append(X[0])
            stats_shuff_collected.append(X[1])
        else:
            stats_shuff.append(funstat(data_shuff))
    
    # p value
    p = empiricalPval(stat_actual, stats_shuff, side=side)

    # plot 
    if plot:
        fig = plt.figure()
        plt.hist(stats_shuff)
        plt.axvline(x=stat_actual, color="r")
        plt.title(f"p={p:.3f} | {1-p:.3f}")
    else: 
        fig = None
    
    if collect_shuffstats:
        return p, stat_actual_collected, stats_shuff_collected, fig
    else:
        return p, fig

    if False:
        # fake dataset/experiement to sanity check permutation test and see how it works
        def funstat(data):
            return np.corrcoef(data[:,0], data[:,1])[0,1]
        #     return np.mean(data[:,1]) - np.mean(data[:,0])

        def funshuff(data):
            import copy
            data_shuff = copy.deepcopy(data)
            np.random.shuffle(data_shuff[:,0])
            return data_shuff

        pvals = []
        for _ in range(500):
            data = np.random.rand(50,2)
            pvals.append(permutationTest(data, funstat, funshuff, 100, plot=False))
            
        plt.figure()
        plt.hist(pvals)
        plt.title("this should be uniformly distributed")
        print("frac cases of p<0.05 should be around 0.05")
        print(sum(np.array(pvals)<0.05)/500)


def plotmod_pvalues(ax, xvals, pvals, pthresh=0.05):
    """ Plot values on top of plots where each x locaiton is a 
    distribution with its own p value
    """

    assert len(xvals)==len(pvals)

    YLIM = ax.get_ylim()
    y = YLIM[0] + 0.75*YLIM[1]-YLIM[0]
    # xrang = XLIM[1] - XLIM[0]

    for x, p in zip(xvals, pvals):
        if p<pthresh:
            col = "r"   
        else:
            col = "k"
        ax.text(x, y, f"{p:.2E}", color=col, rotation=45, alpha=0.5)

def crossval_folds_indices(n1, n2, nfold=None):
    """ Return the indices for partitioning
    two datasets with size n1 and n2, to obtain
    crossvalkidated data of nfolds.
    Written for unbiased euclidian distnace.
    PARAMS:
    - n1, n2, int, size of two dataset
    - nfold, int, how many folds. None to use min of n1 n2
    RETURNS;
    - list_inds_1,  list of np ararys of ints, each indices into dataset 1
    - list_inds_2, same thing, for dataset 2.
    (Both lists will be len nfold).
    """
    inds_on_1, inds_on_2 = _crossval_folds_indices_onsets(n1, n2, nfold)

    def _convert_indson_to_inds(inds_on, n):
        list_inds = []
        for i, on in enumerate(inds_on):
            if i+1==len(inds_on):
                # last onset, it ends at the end of the data
                list_inds.append(np.arange(on, n))
            else:
                list_inds.append(np.arange(on, inds_on[i+1]))
        return list_inds

    list_inds_1 = _convert_indson_to_inds(inds_on_1, n1)
    list_inds_2 = _convert_indson_to_inds(inds_on_2, n2)

    assert len(list_inds_1)==len(list_inds_2)

    return list_inds_1, list_inds_2



def _crossval_folds_indices_onsets(n1, n2, nfold=None):
    """ Return the indices for subsampling
    two datasets with size n1 and n2, to obtain
    crossvalkidated data of nfolds.
    Written for unbiased euclidian distnace.
    PARAMS:
    - n1, n2, int, size of two dataset
    - nfold, int, how many folds. None to use min of n1 n2
    RETURNS;
    - inds_on_1, np arary ints, indices into dataset
    1 which are the onset boundaries of partitions.
    - inds_on_2, same thing, for dataset 2.
    """

    if nfold is None:
        nfold = min([n1, n2])

    if True:
        assert (nfold<=n1) and (nfold<=n2)
    else:
        if nfold>n1:
            nfold = n1
        if nfold>n2:
            nfold = n2

    if n1==n2==nfold:
        # sample each pt one by one.
        return np.arange(n1), np.arange(n2)
    elif n1==n2:
        # partition both datasets the same way
        inds_on = np.floor(np.linspace(0, n1+1, nfold+1)).astype(int)[:-1]
        return inds_on, inds_on
    else:
        # split into subsets
        n_larger = max([n1, n2])
        n_smaller = min([n1, n2])

        inds_on_larger = np.floor(np.linspace(0, n_larger+1, nfold+1)).astype(int)[:-1]
        inds_on_smaller = np.floor(np.linspace(0, n_smaller+1, nfold+1)).astype(int)[:-1]

        if n1>n2:
            return inds_on_larger, inds_on_smaller
        elif n1<n2:
            return inds_on_smaller, inds_on_larger
        else:
            assert False


def cluster_kmeans_with_silhouette_score(X, n_clusters=None, n_clusters_min_max=None, PLOT_SIL=False,
                                         PLOT_FINAL=False, return_figs=False):
    """
    K-means clustering, with optinaly finding n clusters by maximizing silhoutte score
    over range of possible n_clsuters.

    Code taken from: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
    :param X: (nsamp, nfeats)
    :param n_clusters:
    :param n_clusters_min_max: only needed if n_clusters is None, in which case shoudl be 2-list of int,
    will try all n clusters in range of [min, max], and use the one that maximizes the silhoutte score.
    :param PLOT:
    :return: cluster_labels, list of strings represntations of ints
    """
    from sklearn.cluster import KMeans

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)

        if n_clusters is None:
            assert isinstance(n_clusters_min_max, (list, tuple)) and len(n_clusters_min_max)==2
            range_n_clusters = list(range(n_clusters_min_max[0], n_clusters_min_max[1]+1))

            import matplotlib.cm as cm
            from sklearn.metrics import silhouette_samples, silhouette_score

            list_silhouette_avg = []
            list_n_clusters = []
            for n_clusters in range_n_clusters:

                # Initialize the clusterer with n_clusters value and a random generator
                # seed of 10 for reproducibility.
                clusterer = KMeans(n_clusters=n_clusters, random_state=10, n_init="auto")
                cluster_labels = clusterer.fit_predict(X)

                # The silhouette_score gives the average value for all the samples.
                # This gives a perspective into the density and separation of the formed
                # clusters
                silhouette_avg = silhouette_score(X, cluster_labels)
                print(
                    "For n_clusters =",
                    n_clusters,
                    "The average silhouette_score is :",
                    silhouette_avg,
                )
                list_silhouette_avg.append(silhouette_avg)
                list_n_clusters.append(n_clusters)

                if PLOT_SIL:
                    # Compute the silhouette scores for each sample
                    sample_silhouette_values = silhouette_samples(X, cluster_labels)

                    # Create a subplot with 1 row and 2 columns
                    fig, (ax1, ax2) = plt.subplots(1, 2)
                    fig.set_size_inches(18, 7)

                    # The 1st subplot is the silhouette plot
                    # The silhouette coefficient can range from -1, 1 but in this example all
                    # lie within [-0.1, 1]
                    ax1.set_xlim([-0.1, 1])
                    # The (n_clusters+1)*10 is for inserting blank space between silhouette
                    # plots of individual clusters, to demarcate them clearly.
                    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

                    y_lower = 10
                    for i in range(n_clusters):
                        # Aggregate the silhouette scores for samples belonging to
                        # cluster i, and sort them
                        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                        ith_cluster_silhouette_values.sort()

                        size_cluster_i = ith_cluster_silhouette_values.shape[0]
                        y_upper = y_lower + size_cluster_i

                        color = cm.nipy_spectral(float(i) / n_clusters)
                        ax1.fill_betweenx(
                            np.arange(y_lower, y_upper),
                            0,
                            ith_cluster_silhouette_values,
                            facecolor=color,
                            edgecolor=color,
                            alpha=0.7,
                        )

                        # Label the silhouette plots with their cluster numbers at the middle
                        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                        # Compute the new y_lower for next plot
                        y_lower = y_upper + 10  # 10 for the 0 samples

                    ax1.set_title("The silhouette plot for the various clusters.")
                    ax1.set_xlabel("The silhouette coefficient values")
                    ax1.set_ylabel("Cluster label")

                    # The vertical line for average silhouette score of all the values
                    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

                    ax1.set_yticks([])  # Clear the yaxis labels / ticks
                    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

                    # 2nd Plot showing the actual clusters formed
                    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
                    ax2.scatter(
                        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
                    )

                    # Labeling the clusters
                    centers = clusterer.cluster_centers_
                    # Draw white circles at cluster centers
                    ax2.scatter(
                        centers[:, 0],
                        centers[:, 1],
                        marker="o",
                        c="white",
                        alpha=1,
                        s=200,
                        edgecolor="k",
                    )

                    for i, c in enumerate(centers):
                        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

                    ax2.set_title("The visualization of the clustered data.")
                    ax2.set_xlabel("Feature space for the 1st feature")
                    ax2.set_ylabel("Feature space for the 2nd feature")

                    plt.suptitle(
                        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
                        % n_clusters,
                        fontsize=14,
                        fontweight="bold",
                    )
                else:
                    fig = None

            n_clusters = list_n_clusters[np.argmax(list_silhouette_avg)]
            print("Using n_clusters =", n_clusters)

            clusterer = KMeans(n_clusters=n_clusters, n_init="auto")
            cluster_labels = clusterer.fit_predict(X)
            if PLOT_FINAL:
                fig_final, ax = plt.subplots()
                for i in range(n_clusters):
                    ax.plot(X[cluster_labels==i, 0], X[cluster_labels==i, 1], "x", alpha=0.5, label=i)
                # ax.scatter(X[:,0], X[:,1], c=cluster_labels, marker="x", alpha=0.5)
                ax.set_title(f"FINAL CLUSTERS, n={n_clusters}")
            else:
                fig_final = None

            # List of strings, so that these are categorical
            # cluster_labels = [int(c) for c in cluster_labels]
            cluster_labels = [f"{c}" for c in cluster_labels]

        if return_figs:
            return cluster_labels, fig, fig_final
        else:
            return cluster_labels
