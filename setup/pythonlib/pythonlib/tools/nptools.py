import numpy as np
import matplotlib.pyplot as plt

### 
# np.r_ is for quick contaneation.

def bin_values_categorical_factorize(vals):
    """ like pd.factorize.
    convert categorical value (strings) to 0,1,2,...
    """
    return np.unique(vals, return_inverse=True)[1]


def bin_values_by_rank(values, nbins=8, assert_all_vals_within_bins=True):
    """ Convert values to vranks, then bin those, thus ensuring equal n
    items in each class
    """
    from pythonlib.tools.listtools import rank_items
    ranks = rank_items(values)
    ranks_binned = bin_values(ranks, nbins, assert_all_vals_within_bins=assert_all_vals_within_bins)
    return ranks_binned

def bin_values(vals_input, nbins=8, valmin = None, valmax=None, epsilon = 0.0001,
    assert_all_vals_within_bins=True):
    """ REturn vals, but binned, unofrmly from min to max
    Values binned to new categories called 1...nbins.
    Any nan inputs will be nan outoput (ignored).
    PARAMS;
    - nbins, num bbins.
    - epsilon, scalar, small number to pad, to make sure get all values.
    RETIURNS:
    - list of len(n), with string ints, and IGN where input is nan.
    e.g. [np.nan, 1, 2, 3] --> ['IGN', '1', '5', '8']

    # DBUG CODE:
    #     x = np.random.rand(100)
        x[10:20] = np.nan
        xbinned = bin_values(x, 4)
        print(xbinned)
        plt.figure()
        # plt.plot(x, xbinned, "ok")
        plt.plot(xbinned, x, "ok")
    """

    if np.all(np.isnan(vals_input)):
        return vals_input

    if isinstance(vals_input, list):
        vals_input = np.asarray(vals_input)

    def _bin_values(vals, valmin, valmax):
        """ Operates on de-naned data
        """
        assert ~np.any(np.isnan(vals))
        if valmin is None:
            valmin = np.min(vals)
        if valmax is None:
            valmax = np.max(vals)

        bins = np.linspace(valmin-epsilon, valmax+epsilon, nbins+1)
        vals_binned = np.array([np.digitize(v, bins) for v in vals], dtype=np.int_).astype(int)

        # print(vals_binned)
        if assert_all_vals_within_bins:
            assert np.all((vals_binned>=1) & (vals_binned<=nbins))

        # make it list of ints
        # vals_out = [int(x) for x in vals_out]
        vals_binned = vals_binned.astype(int).tolist()

        # for v in vals_binned:
        #     assert i
        return vals_binned

    # bin only the non-nan values.
    inds = ~np.isnan(vals_input)
    # print(inds)
    # print(vals_input)
    vals_to_bin = vals_input[inds]
    try:
        vals_to_bin_binned = _bin_values(vals_to_bin, valmin, valmax)
    except Exception as err:
        print(vals_to_bin)
        print(vals_input)
        print(inds)
        raise err

    # Initialize list
    vals_out = vals_input.copy()
    # vals_out = np.empty(len(vals_out))
    # vals_out[~inds] = "IGN"
    vals_out[inds] = vals_to_bin_binned

    # make it list of ints
    vals_out = [str(int(v)) if ~np.isnan(v) else "IGN" for v in vals_out]
    # vals_out = vals_out.tolist()

    return vals_out

def rankItems(arr):
    """ for each item, returns its
    rank out of all items (0 = smallest);
    if arr = [3,1,5,0], then
    returns [2, 1, 3, 0]. 
    - deals with ties by assigning lower label
    to the earlyier bvalue
    """
    array = np.array(arr)
    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    return ranks


def sortPop(X, dim, filt):
    """ sort population, general purpose. 
    - X is array
    - dim, dimension to sort.
    - filt, function to apply to subarray, e..g, 
    vals_to_sort = [filt(X[:,0, ...]), filt(X[:,1, ...]), ...] 
    if dim is 1. will then sort in ascending order.
    """
    # 1) copy array, and move dim to be the new dim 0.
    Xcopy = np.copy(X)
    Xcopy = np.moveaxis(Xcopy, dim, 0)
    
    # 2) for each element in new dim 0, get scalar by applying filt
#     print(Xcopy[0].shape)
#     print([x.shape[0] for x in Xcopy])
    vals = [filt(x) for x in Xcopy]
    list_to_sort = [[v, x] for v, x in zip(vals, Xcopy)] # combine, so can sort
#     print([l[0] for l in list_to_sort])
    list_to_sort = sorted(list_to_sort, key=lambda x:x[0])
#     print([l[0] for l in list_to_sort])
    Xcopy = [l[1] for l in list_to_sort] # recreate
    Xcopy = np.array(Xcopy)
    
    # 3) return axes to original position
    Xcopy = np.moveaxis(Xcopy, 0, dim)
    return Xcopy   

def isin_close(a, v, atol=1e-06):
    """ returns True if a (scalar) is in v (vector) allowing
    for tolerace. essentially a for loop using np.isclose
    RETURNS:
    - bool, whether a is in v
    - idx, location of a in v.
    """
    x = [np.isclose(a, b, atol=atol) for b in v]
    return np.any(x), np.where(x)[0]

def isnear(arr1, arr2):
    """
    Return True if arr1==arr2 elementwise, allowing for tolerance.
    :param arr1:
    :param arr2:
    :return:
    """
    return isin_array(arr1, [arr2])

def isin_array(arr, list_arr, atol=1e-06):
    """ Returns True if arr (array) is equal to at least one of arrays in list_arr
    PARAMS;
    - arr, np array
    - list_arr, list of arrays
    """
    tmp = [(arr == x).all() for x in list_arr] # array of bools.
    tmp = [np.allclose(arr, x, atol) for x in list_arr] # array of bools.
    return any(tmp)

def stringify(arr, remove_decimels=True):
    """ Convert array to list of strings, 
    - each row in arr is a single item int he otuptu string.
    - cols are concatenated like col1|col2|, ...
    PARAMS:
    - remove_decimels, bool, if True, then uses only singles digits and up.
    RETURNS:
    - out, list of strings.
    """

    assert len(arr.shape)==2

    out = []
    for val in arr:
        s = ""
        for i, item in enumerate(val):
            if i>0:
                sep = "|"
            else:
                sep = ""
            if remove_decimels:
                s += f"{sep}{item:.0f}"
            else:
                s += f"{sep}{item}"
        out.append(s)
        # out.append(f"{val[0]:.0f}|{val[1]:.0f}")
    return out

def unique_tol(arr, decimels=4):
    """ get unique, alowoing for tolerance, based on roudning decimels
    """
    return np.unique(arr.round(decimels=decimels))


def sort_by_labels(X, labels, axis=0):
    """ Sort X by first soring labels, in incresaing order,.
    then sorting X yoked to labels. i.e, assumes you want to yoke
    each item in X to its label.
    PARAMS:
    - X, ndat x ndim np array, rows will be osrted.
    - labels, list of ints, length ndat, to suport sorting in incresaing order
    - axis, dimeision to srot byt. if 0, then sorts rows...
    RETURNS:
    - X, labels, but sorted (copies)
    """
    
    inds = np.argsort(labels)

    if axis==0:
        X = X[inds,:].copy()
    elif axis==1:
        X = X[:, inds].copy()
    else:
        print(X.shape)
        print(axis)
        assert False

    labels = [labels[i] for i in inds]
    
    return X, labels

def find_peaks_troughs(vals, PLOT=False, refrac_npts=None):
    """ Find indices of peaks and troughs in vals
    Does so by looking at diffs
    PARAMS:
    - vals, (1,N) or (N,) array of values.
    - refrac_npts, no maxes will be close to each other npts less
    than refrac_npts. same for mins
    RETURNS:
    - inds_peaks, inds_troughs, list of ints, indices into vals.
    """
    
    # Peaks are where the diffs change sign from pos to neg. and similarly
    # for troughs
    
    if len(vals.shape)>1:
        assert vals.shape[1]==1
        vals = vals.squeeze()

    valsdiff = np.diff(vals, axis=0)
    assert len(valsdiff)==len(vals)-1, "prob shape issue fior vals"

    inds_troughs = list(np.where(np.diff(np.sign(np.diff(vals)))>0)[0]+1)
    inds_peaks = list(np.where(np.diff(np.sign(np.diff(vals)))<0)[0]+1)

    def get_windowed_value(ind, flank):
        """ Get ind-flank:ind+flank, aking care to
        take subset if ind is too lclsoe tot edges"""
        if (ind-flank-1)<0:
            i1 = 0
        else:
            i1 = ind-flank-1
        assert i1>=0

        if ind+flank+1>len(vals)-1:
            i2 = -1
        else:
            i2 = ind+flank+1
        vals_windowed = vals[i1:i2]
        return vals_windowed

    # Check refractr
    if refrac_npts is not None:
        from pythonlib.tools.listtools import remove_values_refrac_period
        inds_peaks_good = []
        for ind_peak in inds_peaks:
            vals_all = get_windowed_value(ind_peak, refrac_npts)
            vals_peak = vals[ind_peak]
            # print("----", ind_peak)
            # print(vals_all)
            # print(vals_peak)
            if np.any(vals_all>vals_peak):
                # Then remove this peak. there must be another that is higher close by
                # print("removing")
                pass
            else:
                # print("keeping")
                inds_peaks_good.append(ind_peak)
        # print(inds_peaks_good)
        # print(inds_peaks)
        inds_peaks = inds_peaks_good

        inds_troughs_good = []
        for ind_trough in inds_troughs:
            vals_all = get_windowed_value(ind_trough, refrac_npts)
            vals_trough = vals[ind_trough]
            if np.any(vals_all<vals_trough):
                # Then remove this peak. there must be another that is higher close by
                pass
            else:
                inds_troughs_good.append(ind_trough)
        inds_troughs = inds_troughs_good

    if PLOT:
        fig, ax = plt.subplots()
        ax.plot(range(len(vals)), vals, "ok")
        for idx in inds_troughs:
            ax.plot(idx, 0, "xr")
        for idx in inds_peaks:
            ax.plot(idx, 0, "xb")
    else:
        fig = None

    return inds_peaks, inds_troughs, fig


def optimize_offset_to_align_tdt_ml2(vals_tdt, vals_ml2):
    """
    Find additive offset that minimaize diff (euclidian) between
    these two vectors.
    :param vals_tdt:
    :param vals_ml2:
    :return:
    - offset, scalar, to add to vals_tdt, to minimize distance from
    vals_ml2.
    """
    from scipy.optimize import minimize_scalar

    assert vals_tdt.shape==vals_ml2.shape, "must be aligned in time"

    def F(x):
        # x = -1
        v2_offset = (vals_tdt +x)
        return np.mean((vals_ml2 - v2_offset)**2)

    res = minimize_scalar(F)
    return res.x



################ PLAYING AROUND WITH INDEXING
# # pull out square slice
# rows = [0,2]
# cols = [0,1]
# x[rows,:][:, cols]
# # x[[0,1] [0]]
#
#
# # pull out speciifc indices
#
# rows = [0, 2]
# cols = [0, 1]
#
# print(x[rows, cols])
# print(x[(rows, cols)])
#
# print("Expected:")
# for r, c in zip(rows, cols):
#     print(f"coord {r,c}:", x[r,c])
#
#
# rows = [0, 0, 0, 0]
# cols = [0, 1, 0, 1]
#
# inds = tuple([(r,c) for r, c in zip(rows, cols)])
# print("Indices: ", inds)
# x[inds]
#
