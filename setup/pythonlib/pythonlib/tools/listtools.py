"""stuff to dow with lists"""
from operator import itemgetter
import numpy as np
# import torch

def check_if_list1_is_ordered_subset_of_list2(list1, list2):
    """
    Return True if every item in list1 is in list2, and the order of the first occurance of each
    of those items in list2 is monoticanlyl incresaing.
    NOTE:
        if item occurs > 1 time in lsit2, only conisders the first occurance in list2
    EXAMPLES:
        check_if_list1_is_ordered_subset_of_list2([1, "test", 2,3, 10], [1, "test", 2, 3, 4, 2, 10]) --> True
    """
    # Check that items in list1 exist in list2, and appear in order
    inds = [list2.index(x) if x in list2 else -1 for x in list1]

    if not all([i>=0 for i in inds]):
        # Then an item is missing from list2
        return False

    # Check that it is increasing
    return np.all(np.diff(inds)>0)


def stringify_list(li, return_as_str=False, separator="--"):
    """ 
    list --> list of str
    """

    if isinstance(li, (tuple, list)):
        out = []
        for x in li:
            if isinstance(x, (tuple, list)):
                out.extend(stringify_list(x))
    #         elif isinstance(x, np.ndarray):
    #             from pythonlib.tools.nptools import stringify
    #             out.extend(stringify(x))
            else:
                out.append(str(x))
    else:
        # is not list...
        out = [str(li)]
    if return_as_str:
        assert isinstance(out, (list, tuple))
        return f"{separator}".join(out)
    else:
        return out


# def sort_mixed_type(mylist):
#     """ Sort, works even if elements in mylist are mixed type.
#     Uses convention for how to rank things of different type:
#     str > num > list > other things > dict > cant hash
#     """
#     from numpy import ndarray

#     def _convert_to_sortable(val):
#         """ convert any item to number, so can compare them by sorting"""
#         if isinstance(val, (list, tuple)):
#             return sum([_convert_to_num(valval) for valval in val])
#         elif isinstance(val, str):
#             return hash(val)
#         elif isinstance(val, (float, int, ndarray)):
#             return val
#         else:
#             # NOTE: hash(None) is valid
#             # if x is int, returns that
#             return hash(val)

#     def key(x):
#         # x, item in list that want to sort. any type.
#         try:
#             if isinstance(x, (list, tuple)):
#                 return (2, sum([_convert_to_sortable(val) for val in x]))
#                 # x = sort_mixed_type(x)
#                 # return (0, hash(tuple(x)))
#             elif isinstance(x, str):
#                 # return (0, _convert_to_num(x))
#                 return (0, _convert_to_sortable(x))
#             elif isinstance(x, (float, int, ndarray)):
#                 return (1, _convert_to_sortable(x))
#             elif isinstance(x, dict):
#                 a = sum([_convert_to_sortable(val) for val in x.keys()])          
#                 b = sum([_convert_to_sortable(val) for val in x.values()])          
#                 return (4, a+b)
#             else:
#                 # eeverything else, hash
#                 return (3, _convert_to_sortable(x))

#         # elif isinstance(x, str):
#         #     return x
#         # elif isinstance(x, (float, int)):
#         #     return x
#         # else:
#         #     # NOTE: hash(None) is valid
#         #     # if x is int, returns that
#         #     return (0, hash(x))
#         except TypeError as err:
#             # Bad, just put at end
#             return (5, '')

#     return sorted(mylist, key=lambda x: key(x))

def argsort_list_of_tuples(list_of_tuples, key):
    """ Given list of tuples, sort hierarchically (e.g, first item in tuple, then second..)
    and also return the inds for sorting. The latter cannot use np.argsort; it doesnt work for
    list of tuples
    PARAMS:
    - key, e.g, lambda x:(x[1], x[2], x[3], x[4]) to sort by 2nd item, then 3rd, etc
    RETURNS:
        -
    e.g.,
    list_of_tuples =
        [('line-8-4-0', (-1, 1), 1, 0),
         ('line-8-4-0', (1, 1), 1, 0),
         ('Lcentered-4-3-0', (1, 0), 4, 0),
         ('line-8-4-0', (1, 1), 4, 1),
         ('arcdeep-4-1-0', (-1, 0), 4, 2),
         ('V-2-2-0', (-1, 1), 4, 3),
         ('Lcentered-4-3-0', (1, 1), 1, 0),
         ('Lcentered-4-3-0', (1, 0), 1, 0),
        )
    """

    assert isinstance(list_of_tuples[0], tuple)

    # find len of tuples
    tmp = list(set([len(x) for x in list_of_tuples]))
    if not len(tmp)==1:
        print(tmp)
        assert False, "otherwise doesnt work"
    n = tmp[0]

    # append index to beginign of each tuple
    labels = [list(x) + [i] for i, x in enumerate(list_of_tuples)]

    try:
        # labels_sorted = sorted(labels, key=key)
        labels_sorted = sort_mixed_type(labels, key_user=key)
    except Exception as err:
        print(labels)
        for i in range(len(labels[0])-1): # -1 since last index is just ints
            print(f"Unique items in column {i} of labels: ", set([l[i] for l in labels]))
            # print(sorted(set([l[i] for l in labels])))
        raise Exception

    # extract indices
    inds_sort = [x[n] for x in labels_sorted]

    # sanity check
    assert len(inds_sort)==len(labels)
    assert len(set(inds_sort))==len(inds_sort)

    return inds_sort

def sort_mixed_type(mylist, DEBUG=False, key_user=None):
    """ Sort, works even if elements in mylist are mixed type.
    Uses convention for how to rank things of different type:
    str > num > list > other things > dict > cant hash
    IMPROVED over the above commented out version. here works
    to try to maintain rank (within type) as much as possible, wheras
    there would convert to hash and lose it (e.g., strings wouldnt properly compare).
    PARAMS;
    - key_user, callable: <item in mylist> --> <item>
    """
    from numpy import ndarray

    if False:
        def _convert_to_num(val):
            """ convert any item to number, so can compare them by sorting"""
            if isinstance(val, (list, tuple)):
                return sum([_convert_to_num(valval) for valval in val])
            elif isinstance(val, str):
                return hash(val)
            elif isinstance(val, (float, int, ndarray, np.generic)):
                return val
            else:
                # NOTE: hash(None) is valid
                # if x is int, returns that
                return hash(val)

    def _is_list_of_comparable_types(mylist):
        # Reutnr True if all itesm in _x are type in <comparable_types>.
        # i./.e, is a one-level
        comparable_types = (str, float, int, ndarray, np.generic)
        if DEBUG:
            print([type(this) for this in mylist])
        return all([isinstance(this, comparable_types) for this in mylist])

    def _convert_to_sortable(val):
        """ convert any item to sortable object - int he end it is string or list of strings."""
        if isinstance(val, (list, tuple)):
            if _is_list_of_comparable_types(val):
                # concatenate into a long string
                tmp = "" 
                for x in val:
                    tmp+=f"{x}"
                return tmp
            else:
                return _convert_to_sortable([_convert_to_sortable(valval) for valval in val])
        elif isinstance(val, str):
            return val
        elif isinstance(val, (float, int, ndarray, np.generic)):
            # VERY HACKY.

            if isinstance(val, (ndarray, np.generic)):
                val = float(val)

            # if val<0:
            #     # so that large negative numbers are first.
            #     # print(val, f"x{-1/val}")
            #     return f"X{-1/val}"
            # else:
            #     return f"x{val}"
            # return val
            # e.g., li = [(-2, -1), -100.123, -100.122, -1, 0, 2, 100, -20, 0.11, 0.111, 20.1, 100000, np.nan, ('test', np.array(0.232))]
            # --> [-100.123,
                 # -100.122,
                 # -20,
                 # -1,
                 # 0,
                 # 20.1,
                 # 2,
                 # 0.111,
                 # 0.11,
                 # 100,
                 # 100000,
                 # nan,
                 # (-2, -1),
                 # ('test', array(0.232))]

            # Squash large positive numbers down.
            if val<0:
                tmp = -1/val
            elif val==0:
                tmp = val
            else:
                # This squashes large numbers down to <10. this is important since "10" is before "9" when sorting as string,
                # which is not what we want.
                tmp = np.log(10 + val)

            # Then convert to 9.999[]
            if tmp>10:
                tmp = 9.99 + np.log(10)/1000

            # Append strings so that negative numbners come before pos.
            # Make the string long so that numbers will cluster togethre (as opopsed to being seprated by actual trings).
            if val==0:
                # in between XXXXXXX and XXXXXXZ
                strval = "XXXXXXXY"
            if val<0:
                # so that large negative numbers are first.
                strval = f"XXXXXXX{tmp:.10f}"
            else:
                strval = f"XXXXXXZ{tmp:.10f}"
            # print(val, " -- ", tmp, " -- ", strval)
            return strval
            #
            # if val<0:
            #     # so that large negative numbers are first.
            #     # print(val, f"x{-1/val}")
            #     return f"X{-1/val}"
            # else:
            #     return f"x{val}"
        else:
            # NOTE: hash(None) is valid
            # if x is int, returns that
            
            # Note: hash is not consistent across runs, but is ok here, since just sorting objects within same run..
            return f"{hash(val)}"

    def key(x):
        # is like _convert_to_sortable, but appends at onset an index that 
        # ensures corect sorting across types.
        # x, item in list that want to sort. any type.

        # First, transform x using user input
        if key_user is not None:
            # print("input:", x)
            x = key_user(x)
            # print("Output:", x)
            # assert False

        # try:
        if DEBUG:
            print("----", x)
        # if _is_list_of_comparable_types(x):
        #     return (2, [_convert_to_sortable(val) for val in x])
        if isinstance(x, (list, tuple)):
            # print([_convert_to_sortable(val) for val in x])
            # adsad
            if False:
                # Need to do this. nested lists...
                # print("ERE")
                # print(x, sum([_convert_to_sortable(val) for val in x]))
                # assert False
                out = (2, sum([_convert_to_sortable(val) for val in x]))
            else:
                out = (2, [_convert_to_sortable(val) for val in x])
        elif isinstance(x, str):
            out = (0, _convert_to_sortable(x))
        elif isinstance(x, (float, int, ndarray, np.generic)):
            out = (1, _convert_to_sortable(x))
        elif isinstance(x, dict):
            if False:
                a = sum([_convert_to_sortable(val) for val in x.keys()])
                b = sum([_convert_to_sortable(val) for val in x.values()])
                out = (4, a+b)
            else:
                a = [_convert_to_sortable(val) for val in x.keys()]
                b = [_convert_to_sortable(val) for val in x.values()]
                out = (4, a+b)
        else:
            # eeverything else, hash
            out = (5, _convert_to_sortable(x))
        # print(out)
        return out

        # except TypeError as err:
        #     # Bad, just put at end
        #     return (5, '')

    return sorted(mylist, key=lambda x: key(x))


def permuteRand(mylist, N, includeOrig=True, not_enough_ok=False):
    """gets N random permutations from muylist, no
    replacement. works by shuffling, taking first one, and
    checking if already have, and so on. can choose to enforce that 
    the entered mylist is included. can allow for N to be larger than
    maximum num permutations. will then just ouput p[ermutations."""

    from random import shuffle
    from math import factorial
   
    if not_enough_ok:
        if N>factorial(len(mylist)):
            from itertools import permutations
            print("not enough permutations, so getting {}".format(factorial(len(mylist))))
            return list(permutations(mylist))
    else:
        assert N <= factorial(len(mylist)), "not enough unique permutations..."
       
    # convert list to list of indices, so that can put in sets.
    mylist_inds = list(range(len(mylist)))
    myset = set()
    if includeOrig:
        myset.add(tuple(mylist_inds))
        
    while len(myset) < N:
        shuffle(mylist_inds)
        myset.add(tuple(mylist_inds))
   
#     print([list(x) for x in myset])
   
    # get back permuted lists
    permutedlists = []
    from operator import itemgetter
    for inds in myset:
        permutedlists.append(itemgetter(*inds)(mylist))
#     print(permutedlists)
   
    return permutedlists
if False:
    # mylist = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T']
    mylist = ["0","1","2","3","4","5", "6", "7", "8"]
    permuteRand(mylist, 7*720)



def concatStringsInList(mylist):
    """given list """
    pass


def tabulate_list(l, return_as_list=False):
    """outputs dict with unique entries as keys, and num
    occurances as entries
    like tabulate() in matlab
    - return_as_list, then returns as list of tuples, not as dict.
    """
    l_unique = set(l);
    outdict = {}
    for key in l_unique:
        outdict[key] = len([ll for ll in l if ll==key])
    if return_as_list:
        return [(k, v) for k, v in outdict.items()]
    return outdict


def partition(collection):
    """ from https://stackoverflow.com/questions/19368375/set-partitions-in-python
    gets all partitions of the list collection.
    e..g, one partition is [[1], [2,3,4]]
    """
    if len(collection) == 1:
        yield [ collection ]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset 
        yield [ [ first ] ] + smaller

    
    # something = list(range(1,5))

    # for n, p in enumerate(partition(something), 1):
    #     print(n, sorted(p))


def get_counts(vals):
    """ returns dict, where keys are unique values in vals,
    and values are counts (frequenceis). vals can be anything
    hashable (?) by np.unique, e..g, strings, nums, etc.
    INPUTS:
    - vals, array-like,
    """
    x, y = np.unique(np.array(vals), return_counts=True)
    counts_dict = {xx:yy for xx, yy in zip(x, y)}
    return counts_dict


def counts_to_pdist(counts_dict, cats_in_order, dtype=None, 
    prior_counts=0., print_stuff=True):
    """
    Get an array of probabilities, where the
    indices are in order defined by cats_in_order.
    INPUT:
    - counts_dict, from get_counts.
    - cats_in_order, list of elements matching (superset) of unique
    values in vals. will use this to determine ordering in pdist output. 
    can also be subset, in which will ignore any keys not in it. will always
    normalize output to sum to 1.
    - dtype, dtype for torch tensor
    - prior_counts, adds this to counts before getting probs
    RETURNS:
    - pdist, torch tensor, len of cats_in_order.
    """
    import torch
    if dtype is None:
        dtype = torch.float32
    probs = []
    for c in cats_in_order:
        if c not in counts_dict:
            nelem = 0. + prior_counts
        else:
            nelem = counts_dict[c] + prior_counts
        probs.append(nelem)

    pdist = torch.tensor(probs, dtype=dtype)
    pdist = pdist/torch.sum(pdist)
    pdist = pdist.squeeze()
    
    if print_stuff:
        print("cats, this order:")
        print(cats_in_order)
        print("these probs:")
        print(pdist)
    
    return pdist

def rank_items(li):
    """ Retrurn rank of each item, in their order
    e..g, [12 13 2 2] --> [3 4 1 1]
    Note: ranks start at 1 (not 0).
    """
    import scipy.stats as ss
    ranks = ss.rankdata(li, method="min") # e..g, [12 13 2 2] --> [3 4 1 1]
    return ranks
def rankinarray1_of_minofarray2(array1, array2, look_for_max=False):
    """ for the index of min in array2, what is the rank of the
    correspodning value in array1? e.g., if
    array1 = [1,40,8,32,32] and array2 = [3,2,4,5,1], then first say
    index of min array 2 is 4, then looks in array1[4]. 32 is the 3rd 
    rank (this accounts for ties by taking the best rank), so the output 
    is 3.
    - look_for_max, then same, but looks for max in both arrays.
    RETURNS:
    - rank, item1, item2, idx, where items are the original values in arrays 1 and 2.
    and idx is the index.
    NOTES:
    - e.g., useful is array2 is beh-task distance, and array1 is some measure of
    the task efficeincey(score), for scoring how efficient behvior was.
    - 0-indexed, so best rank is 0.

    """    
    import scipy.stats as ss
        
    if look_for_max:
        array1 = [-a for a in array1]
        array2 = [-a for a in array2]
        
    # convert array1 to ranks
    array1_ranks = ss.rankdata(array1, method="min") # e..g, [12 13 2 2] --> [3 4 1 1]
    array1_ranks = array1_ranks-1 # since is 1-indexed

    # find the rank of min array2
    idx = array2.index(min(array2)) # find the index of the task sequence that is most aligned to beh.

    return array1_ranks[idx], array1[idx], array2[idx], idx

def random_inds_uniformly_distributed(vals, ntoget, return_original_values=False,
    dosort=True):
    """ sorts vals incresaing, then returns unofmrly sampled values.
    PARAMS:
    - vals, array of numbers
    - return_original_values, then instead of indices into vals returns the
    corresponding vals
    RETURNS:
    - list of indices into original vals [NOT THE VALUES THEMSELVES]
    """
        
    if ntoget>len(vals):
        ntoget = len(vals)
        
    # assert ntoget<=len(vals)

    if dosort:
        inds = np.argsort(vals)
    else:
        inds = np.arange(len(vals))

    idxs = np.linspace(0, len(inds)-1, ntoget)
    idxs = np.floor(idxs)
    idxs = np.array(idxs, dtype=int)
    indskeep = inds[idxs]

    if return_original_values:
        return [vals[i] for i in indskeep]
    else:
        return indskeep


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    from itertools import chain, combinations
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def extract_novel_unique_items_in_order(listin):
    """ 
    Get change points, when the items in listin change.
    PARAMS:
    - listin, list of items with equals method
    RETURNS:
    - list_new_items, list of items at change points
    - list_positions, list of indices in listin wherre those change pts are fond (right side)
    e.g, listin=[0,2,2,1,3, 3, 2], returns:
    [0, 2, 1, 3, 2]
    [0, 1, 3, 4, 6]
    """
    list_new_items =[]
    list_positions = []
    curr_item = None
    for i, x in enumerate(listin):
        if x==curr_item:
            continue
        else:
            list_new_items.append(x)
            list_positions.append(i)
            curr_item = x

    return list_new_items, list_positions

def remove_values_refrac_period(times_in_order, refrac_period):
    """ Given list of times, return indices to slice, to remove times that
    occur too close in time to preceding (i.e. interval less than refrac_period).
    i.e., takes the first one that encounters...
    PARAMS:
    - times_in_order, list-like of times, assumed to be in order. will not check
    - refrac_period, duration.
    RETURNS:
    - inds_keep, list of ints indices to keep.
    - inds_remove, complement of inds_keep
    EXAMPLE:
        values_in_order = [0.1, 0.11, 0.2]
        refrac_period = 0.1
        --> ([0], [1, 2])
    """
    inds_remove = np.argwhere(np.diff(times_in_order)<refrac_period)+1
    inds_remove = [x[0].astype(int) for x in inds_remove]
    inds_keep = [i for i in range(len(times_in_order)) if i not in inds_remove]
    return inds_keep, inds_remove

def indices_into_original_list_after_concat(list_of_lists):
    """
    returns a list, of sum of lents of all innter lisrts, with index back
    into orignal list

    e.g., x = [[0,1], [3,4,5], [10], [], range(5), range(2)]
    RETURNS:
    0
    0
    1
    1
    1
    2
    4
    4
    4
    4
    4
    5
    5
    """

    ct = 0
    idxs = []
    for i, xx in enumerate(list_of_lists):
        for j in range(len(xx)):
            # print(i)
            idxs.append(i)
        ct = ct+len(xx)

    return idxs

def unique_input_order(mylist):
    """ list set(mylist) but returns
    in the order inputed.
    e.g., 
    mylist = [0,4,3,4, 2,4,0, 'cow'] -->
        [0, 4, 3, 2, 'cow']
    """
    
#     mylist_uniq = list(set(mylist))
    mylist_uniq = []
    for item in mylist:
        if item not in mylist_uniq:
            mylist_uniq.append(item)
    
    return mylist_uniq

def list_roll(mylist, shift):
    """ Like np.roll but for lists. does not spruiously
    convert list of tuples into list of lists.
    PARAMS:
    - mylist, list
    - shift, int, how much to shift to right. 0 means no shift. 
    e.g.,,:
        x = [1,2,3,4,5]
        list_roll(x, 2) --> [4, 5, 1, 2, 3]
        list_roll(x, 6) --> [5, 1, 2, 3, 4] (becuase 6%5=1)
        list_roll(x, -2) --> [3, 4, 5, 1, 2] 
        list_roll(x, -7) --> [3, 4, 5, 1, 2] 
    """
    assert isinstance(mylist, list)
    shift = shift%len(mylist)
    return mylist[-shift:] + mylist[:-shift]


def slice_list_relative_to_index_out_of_bounds(mylist, ind, npre=1, npost=1):
    """
    Pull out tokens precesding and following this index.
    Any tokens that dont exist: replace with None.
    RETURNS:
    - tokens_prev, list of prev tokens,
    - tok_this, a single token, for ind, 
    - tokens_next, list of next tokens
    (for pre and next, if doesnt exist, reuturns None for that item in the list)
    EG:
    - mylist = [0, 1, 2, 3, 4]
    - ind = 2
    - npre, npost = (4,4)
    --> ([None, None, 0, 1], 2, [3, 4, None, None])
    """

    assert npre>=0
    assert npost>=0
    
    if ind-npre<0:
        on = 0
    else:
        on = ind-npre
    n_nones = npre - ind
    tokens_prev = [None for _ in range(n_nones)] + mylist[on:ind]

    tok_this = mylist[ind]

    if ind+npost+1>len(mylist):
        off = len(mylist)
    else:
        off = ind+npost+1

    n_nones = ind+npost+1 - len(mylist)
    tokens_next = mylist[ind+1:ind+npost+1] + [None for _ in range(n_nones)]

    assert len(tokens_prev)==npre, f"{len(tokens_prev)}, {npre}"
    assert len(tokens_next)==npost, f"{len(tokens_next)}, {npost}"     

    return tokens_prev, tok_this, tokens_next


def sequence_first_error(beh, task):
    """
    Compare two list of tokens (ints), usually represnting beh and task 
    Finds the first item that differs. If they don't; differ (even if they are
    diff length) returns None.
    PARAMS:
    - beh, task, list of objects than can be compared with ==. 
    Usually ints.
    RETURNS;
    - index of first difference.
    - taskind_chosen, taskind_correct, the objects in beh and task at 
    that index.
    - OR-
    None, None, None (this true even if beh and task are different length)
    EG:
    - beh = [2, 5]
    - task = [2, 3, 5, 1, 4, 0]
    returns 1, 5, 3
    """
    for i, (taskind_chosen, taskind_correct) in enumerate(zip(beh, task)):
        if not taskind_chosen==taskind_correct:
            return i, taskind_chosen, taskind_correct
    return None, None, None
