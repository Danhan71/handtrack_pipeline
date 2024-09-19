""" functions for computing distances between objects, generally time-series
represetning strokes, behavior, etc. """

import numpy as np
import matplotlib.pyplot as plt

def distmat_construct_wrapper(vals1, vals2, dist_func, cap_dist=None, normalize_rows=False,
                              normalize_cols_range01=False, normalize_by_range=False, range_norm=None,
                              convert_to_similarity=False, similarity_method=None, DEBUG=False,
                              accurately_estimate_diagonal=False,
                              inds_skip_rows_or_cols=None, PLOT=False):
    """ Wrapper to generate distance matrix.
    Assumes this is symetric. Only computes upper triangle copies to lower.
    PARAMS:
    - dist_func, function mapping from (vals1[i], vals2[j]) --> scalar
    - accurately_estimate_diagonal, bool, if True, then deals with prpoblem where
    diagonal values are all comopare to self  here solves by
    resamplig method. This only makes sense if (if val1==vals2)
    RETURNS:
        - D, np array shape (len(vals1), len(vals2))
    """

    from pythonlib.tools.checktools import check_objects_identical
    symmetric = check_objects_identical(vals1, vals2)

    # if isinstance(vals1, np.ndarray):
    #     symmetric = np.all(vals1==vals2) # This speeds things up
    # else:
    #     try:
    #         symmetric = vals1==vals2 # This speeds things up
    #     except ValueError as err:
    #         try:
    #             symmetric = np.all(vals1==vals2) # This speeds things up
    #         except Exception as err:
    #             print(vals1)
    #             print(vals2)
    #             print(type(vals1), type(vals2))
    #             raise err

    if inds_skip_rows_or_cols is None:
        inds_skip_rows_or_cols = []

    # certain params are incompatible
    if normalize_cols_range01:
        # then each column convert to range 0,1 (min, max distance)
        assert normalize_by_range==False
        assert similarity_method not in ["divide_by_maxcap", "divide_by_inputed_range"], "tehse fail becuase they change units beofre norm."
    if normalize_by_range:
        # clip into range 0,1.
        assert range_norm is not None
        assert normalize_cols_range01 ==False

    n1 = len(vals1)
    n2 = len(vals2)
    D = np.empty((n1, n2))

    if accurately_estimate_diagonal:
        assert symmetric==True
        def dist_func_i_equals_j(v1, dummy):
            return dist_vs_self_split_compute_agg(v1, dist_func, nfold=10)
    else:
        dist_func_i_equals_j = None

    if symmetric:
        # Do only one half
        for i, v1 in enumerate(vals1):
            for j, v2 in enumerate(vals2):
                if j>=i:
                    if (i in inds_skip_rows_or_cols) or (j in inds_skip_rows_or_cols):
                        # assert False
                        d = np.nan
                    else:
                        if accurately_estimate_diagonal and i==j:
                            d = dist_func_i_equals_j(v1, v2)
                        else:
                            d = dist_func(v1, v2)
                        # print(i,j)
                        # print(v1, v2)
                        # print(v1.shape)
                        # print(v2.shape)
                        # print(np.mean(v1, 0))
                        # print(np.mean(v2, 0))
                        # print(d)
                        # print(dist_func(v1, v2))
                        # print(np.isnan(d))
                        # print(np.any(np.isnan(v1)))
                        # print(np.any(np.isnan(v2)))
                        # print(DID)
                        # print(v1.shape, v2.shape)
                        # print(dist_func(v1, v2))
                        assert not np.isnan(d)
                    D[i, j] = d

        # populate the other half of matrix.
        for i, v1 in enumerate(vals1):
            for j, v2 in enumerate(vals2):
                if j<i:
                    D[i, j] = D[j, i]
    else:
        # Not symmetric..
        for i, v1 in enumerate(vals1):
            for j, v2 in enumerate(vals2):
                if (i in inds_skip_rows_or_cols) or (j in inds_skip_rows_or_cols):
                    d = np.nan
                else:
                    d = dist_func(v1, v2)
                    assert not np.isnan(d)
                    D[i, j] = d

    # Cap the distance?
    if cap_dist is not None:
        # plt.figure()
        # plt.hist(D)
        # print(cap_dist)
        # assert False
        D[D>cap_dist] = cap_dist
    # print(D)
    # print(n1, n2)
    # assert False

    if normalize_rows:
        dnorm = np.sum(D, axis=1, keepdims=True)
        D = D/dnorm

    if normalize_cols_range01:
        # then each column convert to range 0,1 (min, max distance)
        dmin = np.min(D, axis=0, keepdims=True)
        D = D-dmin
        dmax = np.max(D, axis=0, keepdims=True)
        D = D/dmax

    if normalize_by_range:
        # clip into range 0,1 (all values)
        if DEBUG:
            plt.figure()
            plt.hist(D)
            plt.title("Before norm by range")
            print(range_norm)
        assert range_norm[1]>range_norm[0]
        D[D>range_norm[1]] = range_norm[1] # first clip
        D = (D-range_norm[0])/(range_norm[1] - range_norm[0])
        if DEBUG:
            plt.figure()
            plt.hist(D)
            plt.title("After norm by range")

    # Convert from distance to similarity
    if convert_to_similarity:
        EPS = 0.00001
        # plt.figure()
        # plt.hist(D[:])
        # assert False
        if similarity_method=="squared_one_minus":
            # take differenc,e then take square
            # emprically: makes it more normal, becuase of skew.
            D = (1-D)**2
        elif similarity_method=="one_minus":
            D = (1-D)
        elif similarity_method=="divide_by_max":
            D = 1-D/(np.max(D) + EPS)
        elif similarity_method=="divide_by_median":
            tmp = D/(np.median(D) + EPS)
            D = 1-tmp/(np.max(tmp) + EPS)
        elif similarity_method=="divide_by_maxcap":
            assert cap_dist is not None
            D = 1-D/(cap_dist + 1)
        elif similarity_method=="inverse":
            D = 1./(D + 1)
        else:
            assert False

    # assert ~np.any(np.isnan(D))
    if PLOT:
        from pythonlib.tools.snstools import heatmap_mat
        fig, ax, rgba_values = heatmap_mat(D, annotate_heatmap=False)
    return D

def closest_pt_twotrajs(traj1, traj2):
    """ returns closest pt between these
    INPUT:
    - traj1, traj2, N/M x 2
    OUTPUT:
    - dist, scalar
    - ind1,
    - ind2, indexing into traj1 and 2 for those pts
    """
    from scipy.spatial.distance import cdist
    try:
        D = cdist(traj1, traj2)
    except Exception as err:
        print(traj1.shape, traj2.shape)
        raise err

    ind1, ind2 = np.unravel_index(np.argmin(D), D.shape)
    dist = D[ind1, ind2]

    return dist, ind1, ind2


def furthest_pt_twotrajs(traj1, traj2, assymetry=None):
    """
    see
    closest_pt_twotrajs
    INPUT:
    - assymetry,
    --- None, considers entire trajs
    --- 1. <int>, traj1 is contained in 2
    --- 2, vice versa
    """
    from scipy.spatial.distance import cdist
    D = cdist(traj1, traj2)

    if assymetry is None:
        # consider all pts
        ind1, ind2 = np.unravel_index(np.argmax(D), D.shape)
        dist = D[ind1, ind2]
    elif assymetry==1:
        # is OK if traj2 has many points out
        x = np.min(D, axis=1)
        dist = np.max(x)
        ind1, ind2 = None, None
    elif assymetry==2:
        x = np.min(D, axis=2)
        dist = np.max(x)
        ind1, ind2 = None, None
    else:
        assert False

    return dist, ind1, ind2


def _distPtsTimePtsMatched(pts1, pts2):
    """
    Get cumulative pt by pt distance between pts1 and pts2.
    Gets euclidian distance.
    PARAMS;
    - pts1, pts2, each np array, (N,2+) where N is same for them
    If >2 columns, will autoatmicalyl take just first 2.
    RETURNS:
    - dist, sum euclidian dist
    """
    assert pts1.shape[0]==pts2.shape[0]
    dist = 0
    for p1, p2 in zip(pts1[:,:2], pts2[:,:2]):
        dist+=np.linalg.norm(p1-p2)
    return dist


def _distStrokTimeptsMatched(strok_beh, strok_mod, fs=None, ploton=False,
                           min_strok_dur=0.175, return_separate_scores=False,
                           vec_over_spatial_ratio=1,
                             # lowpass_freq=5,
                             lowpass_freq=None, # changed 1/3/24, so now uses "clean" vesrion
                             ):
    """ if strok1 and 2 have matched timepoitns (at least
    same num timepoint) then can compare pt by pt to
    compute distance.
    - distance function in both spatial (xy pos) and velocity
    (x and y) domains. final score is the sum of these. use
    vec_over_spatial_ratio to match theri scales (since spatial is
    usually lower magnitudes)
    - min_strok_dur, if strok shorter than this, then throws error
    - return_separate_scores, then returns tuple separating spatial and
    vel scores, (spatial, vel). make it None to ignore.
    - vec_over_spatial_ratio, multiples sptial score by this, useful if want
    them match, or to more strongly weigh one over other.
    - NOTE: vec_over_spatial_ratio, if scalar, then multiplies spatial dist.
    if tuple of scalars, then should be length 2 and will multiply (spatial, vec)
    - returns cost normalzied by num timesteps. """
    from pythonlib.tools.stroketools import strokesVelocity

    if min_strok_dur is not None:
        if strok_beh[-1, 2]-strok_beh[0,2]<min_strok_dur:
            assert False, "stroke duration too short"

    # skip certain distances if instructed
    skip_spatial=False
    skip_vel=False
    if isinstance(vec_over_spatial_ratio, (list, tuple)):
        if vec_over_spatial_ratio[0]==0:
            skip_spatial=True
        if vec_over_spatial_ratio[1]==0:
            skip_vel=True

    # 1) spatial distance - use mean squared error - can do this since points are matched
    if skip_spatial:
        dist_spatial = 0.
    else:
        dist_spatial = _distPtsTimePtsMatched(strok_beh, strok_mod)
        # Old version, incorrect
        # dist_spatial2 = np.linalg.norm(strok_beh[:,[0,1]] - strok_mod[:,[0,1]])

    # 2) velocity distance - also use mean squared error of velocity timecourses
    if skip_vel:
        dist_vel = 0.
    else:
        if fs is None:
            assert False, "to get vel distance, you must pass in fs"
        strok_beh_vel = strokesVelocity([strok_beh], fs=fs, lowpass_freq=lowpass_freq)[0][0]
        strok_mod_vel = strokesVelocity([strok_mod], fs=fs, lowpass_freq=lowpass_freq)[0][0]
        dist_vel = _distPtsTimePtsMatched(strok_beh_vel, strok_mod_vel)

    # normalize by timesteps
    dist_spatial/=strok_beh.shape[0]
    dist_vel/=strok_beh.shape[0]

    # reweight the scores if desired
    if isinstance(vec_over_spatial_ratio, (list, tuple)):
        assert len(vec_over_spatial_ratio)==2
        dist_spatial*=vec_over_spatial_ratio[0]
        dist_vel*=vec_over_spatial_ratio[1]
    else:
        dist_spatial*=vec_over_spatial_ratio

    if ploton:
        import matplotlib.pyplot as plt
        from ..drawmodel.strokePlots import plotDatStrokes, plotDatStrokesTimecourse
        fig, axes = plt.subplots(3,1, figsize=(10,15))
        plotDatStrokes([strok_beh], axes[0])
        plotDatStrokes([strok_mod], axes[0])
        plotDatStrokesTimecourse([strok_beh_vel], axes[1])
        plt.ylabel("beh vel")
        plotDatStrokesTimecourse([strok_mod_vel], axes[2])
        plt.ylabel("mod vel")

        print(dist_spatial, dist_vel)

    if return_separate_scores:
        return (dist_spatial, dist_vel)
    else:
        cost = dist_spatial + dist_vel
        return cost




def modHausdorffDistance(itemA, itemB, dims=(0,1), ver1="mean", ver2="max", D=None,
    return_marginals=False, asymmetric_ver=None):
    """
    Modified Hausdorff Distance.

    M.-P. Dubuisson, A. K. Jain (1994). A modified hausdorff distance for object matching.
     International Conference on Pattern Recognition, pp. 566-568.

    :param itemA: [(n,2) array] coordinates of "inked" pixels
    :param itemB: [(m,2) array] coordinates of "inked" pixels
    :return dist: [float] distance

    dims = [0,1] means will use itemA[:,[0,1]] and so on.
    From Reuben Feynman and Brenden Lake
    ----------
    - D, distance matrix, optional. If pass this, then itemA, itemB, dims will not be used.
    - return_marginals, then returns marginals of distance matrix after taking min.

    """

    if D is None:
        from scipy.spatial.distance import cdist
        if dims:
            D = cdist(itemA[:,dims], itemB[:,dims])
        else:
            D = cdist(itemA, itemB)


    mindist_A = D.min(axis=1)
    mindist_B = D.min(axis=0)

    if ver1=="mean":
        mean_A = np.mean(mindist_A)
        mean_B = np.mean(mindist_B)
    elif ver1=="median":
        mean_A = np.median(mindist_A)
        mean_B = np.median(mindist_B)
    elif ver1=="max":
        mean_A = np.max(mindist_A)
        mean_B = np.max(mindist_B)
    elif ver1=="min":
        mean_A = np.min(mindist_A)
        mean_B = np.min(mindist_B)
        assert mean_A==mean_B, "min should be like this..."
    else:
        assert False

    if asymmetric_ver==None:
        # Then combine both scores, from perspective of pts1 and pts2
        if ver2=="mean":
            dist = np.mean((mean_A,mean_B))
        elif ver2=="max":
            dist = np.max((mean_A,mean_B))
        elif ver2=="min":
            dist = np.min((mean_A,mean_B))
        else:
            assert False
    elif asymmetric_ver=="A":
        # Then only care about "perspective" of ptsA. So as long as ptsB has pts close to A,
        # doesn't matter what other ptsB pts are doing.
        dist = mean_A
    elif asymmetric_ver=="B":
        dist = mean_B
    else:
        assert False

    if return_marginals:
        return dist, mindist_A, mindist_B
    else:
        return dist

def euclidian(dat1, dat2):
    """

    :param dat1: (n,d)
    :param dat2: (m,d)
    :return:
    """

    return np.linalg.norm(np.mean(dat1, axis=0) - np.mean(dat2, axis=0))


def euclidian_unbiased_debug(n1=10, n2=10, dim=10):
    """
    Quick test and plots of euclidian, comparing unbioased vs, biased.

    :return:
    """

    # Test unbiased estimator.
    fig, ax = plt.subplots()

    # ADDS = list(range(10))
    ADDS = np.linspace(0, 0.4, 100)
    ds = []
    ds_unb = []
    for ADD in ADDS:
        dat1 = np.random.rand(n1, dim) - 0.5
        dat2 = np.random.rand(n2, dim) - 0.5 + ADD
        # dat2[:,0] = dat2[:,0] + ADD

        d_unbias = euclidian_unbiased(dat1, dat2)
        d = np.linalg.norm(np.mean(dat1,0)-np.mean(dat2, 0))

        ds.append(d)
        ds_unb.append(d_unbias)

    ax.plot(ADDS, ds, "ok", label="biased")
    ax.plot(ADDS, ds_unb, "xr", label="unbiased")
    # ax.axis("square")
    ax.axhline(0)
    ax.legend()

def euclidian_unbiased(dat1, dat2):
    """ Get euclidian distance betweeen centroides of
    two multivar datasets. Unbiased, port of:
    https://github.com/fwillett/cvVectorStats/blob/master/cvDistance.m
    Here, does a few times, shuffling in between, to get better estimate.
    Checked that this helsp a bit, but not much.
    PARAMS:
    - dat1, dat2, (N,d) where N is num datapts and d is dim, they can
    have different N.
    NOTE: ioncreasing n_iter doesnt help much (even 20)
    """

    if dat1.shape[0]<2 or dat2.shape[0]<2:
        print(dat1.shape)
        print(dat2.shape)
        assert False

    d = _euclidian_unbiased(dat1, dat2)
    list_d = [d]
    dat1_copy = dat1.copy()
    dat2_copy = dat2.copy()
    n_iter = min([dat1.shape[0], dat2.shape[0], 3])
    # n_iter = 40
    for _ in range(n_iter):
        # Shuffle
        np.random.shuffle(dat1_copy)
        np.random.shuffle(dat2_copy)
        # dat2_rolled = np.roll(dat2, 5, axis=0)

        d = _euclidian_unbiased(dat1_copy, dat2_copy)
        list_d.append(d)

    return np.mean(list_d)

def _euclidian_unbiased(dat1, dat2):
    """ Get euclidian distance betweeen centroides of
    two multivar datasets. Unbiased, port of:
    https://github.com/fwillett/cvVectorStats/blob/master/cvDistance.m
    PARAMS:
    - dat1, dat2, (N,d) where N is num datapts and d is dim, they can
    have different N
    """
    from pythonlib.tools.statstools import crossval_folds_indices

    # Get cross-validation splits.
    n1 = dat1.shape[0]
    n2 = dat2.shape[0]
    nfold = min([n1, n2, 5])
    # nfold = None
    list_inds_1, list_inds_2 = crossval_folds_indices(n1, n2, nfold=nfold)

    # Compute squared dist (norm of vector difference) for each split.
    inds1_all = np.arange(n1, dtype=int)
    inds2_all = np.arange(n2, dtype=int)
    squared_dist_estimates = []
    for inds1, inds2 in zip(list_inds_1, list_inds_2):

        # Vector between centroids of "small set"
        d_small = np.mean(dat1[inds1, :], axis=0) - np.mean(dat2[inds2, :], axis=0)

        # Vector between centroids of remaining data (big set)
        inds1_big = np.setdiff1d(inds1_all, inds1)
        inds2_big = np.setdiff1d(inds2_all, inds2)
        d_big = np.mean(dat1[inds1_big, :], axis=0) - np.mean(dat2[inds2_big, :], axis=0)

        # get squared eucl dist.
        squared_dist_estimates.append(np.dot(d_small, d_big))

        # print(inds1, inds2, inds1_big, inds2_big, np.dot(d_small, d_big))
        # print(d_small.shape)

    # Convert to eucl distance
    squared_dist = np.mean(squared_dist_estimates)
    euclidian_dist = np.sign(squared_dist) * (np.abs(squared_dist)**0.5)
    # print(squared_dist, euclidian_dist)
    return euclidian_dist

def dist_vs_self_split_compute_agg(X, dist_func, nfold=10):
    """ Given a data dsitribiton X (ndat, ndim), compute
    its distance to itself after splitting in half, doing multiple
    times and averaging. As a "positive control" for the noisiness
    of the data, when used in distnace matrix calucaitons.
    PARAMS:
    - nfold, int, n times to rnadomly split data and average over resultgs.
    - dist_func, func(X,X), ie takes in X-like data and returns scalar.
    RETURNS:
        - d, scalar distance between splits of X. Expectation
        0 given large dataset.
    """
    from sklearn.model_selection import train_test_split, ShuffleSplit

    # Iterate. split compute, avg (5 fold?)
    rs = ShuffleSplit(n_splits=nfold, train_size=0.5)
    ds = []
    for i, (train_index, test_index) in enumerate(rs.split(X)):
        X1 = X[train_index, :]
        X2 = X[test_index, :]
        d = dist_func(X1, X2)
        ds.append(d)
    return np.mean(ds)
