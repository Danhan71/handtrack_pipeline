""" tools for use with pandas dataframes. also some stuff using python dicts and translating between that and dataframs

3/20/21 - confirmed that no mixing of values due to index error:
- i.e., in general, if function keeps df size unchanged, then will not modify the indices, and 
will check that the output df is same as input (for matching columns). 
- if does change size, then will reset indices.
"""

import pandas as pd
import numpy as np
from pythonlib.tools.listtools import sort_mixed_type
import matplotlib.pyplot as plt
from pythonlib.tools.plottools import savefig
import seaborn as sns

def _check_index_reseted(df):
    """
    Assert that indices are 0, 1, 2....
    This is assumed for lots of my code...
    :param df:
    :return:
    """
    if len(df)>0:
        inds = df.index.tolist()
        assert (list(sorted(set(inds)))==inds) and (inds[-1] == len(inds)-1), "this dataframe.index is not 0, 1, 2.... [should do df.reset_index(drop=True)]"

def _mergeKeepLeftIndex(df1, df2, how='left',on=None):
    """ merge df1 and 2, with output being same length
    as df1, and same indices. 
    - on, what columnt to align by, need this if they are differnet
    sizes.
    RETURN:
    - dfout, without modifying in place.
    """

    df1 = df1.copy() # becuase dont want to set value on a slice, need to copy.
    df1["index_copy"] = df1.index # save old indices.
    dfout = df1.merge(df2, how=how, on=on) # 
    # dfout = df1.merge(df2, how=how, on=on, validate="one_to_one") # 
    assert len(dfout)==len(df1)
    dfout = dfout.set_index("index_copy", drop=True)
    dfout.index.names = ["index"]
    return dfout

def _checkDataframesMatch(df1, df2, check_index_match=True):
    """ Checks that:
    - for any index that is shared (between df1 and df2), 
    the values for all columns are shared. 
    - check_index_match, then also checks that df1 and df2
    are identical length, with identical indices
    NOTE: if have same values (relative to index) but index is
    shuffled rleative to row locaiton, then will fail check - i.e.
    df1[col].values muist equal df2[col].values for all col.
    NOTE: Not checking whether columns have same values, since this
    led to segmentation fault issues (if there was nan)
    """
    
    columns_shared = [c for c in df1.columns if c in df2.columns]
    
    if check_index_match:
        if len(df1)!=len(df2):
            print(df1)
            print(df2)
            assert False, "lengths must be the same to do this check"
        assert np.all(df1.index == df2.index), "indices are not the same!! maybe you applied reset_index() inadvertantely?"
        # dfthis1 = df1[df1.notna()]
        # dfthis2 = df2[df2.notna()]
        # assert dfthis1[columns_shared].equals(dfthis2[columns_shared])
        # for col in columns_shared:
        #     assert df1[col].fillna(0).equals(df2[col].fillna(0)), "values dont match"
        
        # to show that index must be in the same order to pass check.
#         assert df1[columns_shared].sort_values("hausdorff").sort_index().equals(df2[columns_shared])
                
    else:
        assert False, "not coded, since if frames are different size, not sure what youd want to check"


def mergeOnIndex(df1, df2):
    """ merge, keeping indexes unchanged, 
    df1 and df2 must come in with idnetical indices.
    will use union of columns, but making sure to not have
    duplicate columns. if duplicate, then assumes that they are
    identical in df1 and df2 (will check) and will
    take the column values from df1.
    """
    _checkDataframesMatch(df1, df2)

    cols_to_use = df2.columns.difference(df1.columns)
    dfout = pd.merge(df1, df2[cols_to_use], left_index=True, 
        right_index=True, how='outer', validate='one_to_one')
    return dfout

def join_dataframes_appending_columns(df1, df2, columns_of_df2_to_add_to_df1):
    """ join dataframes that have identical indices, such that anyuc olumns of
    df2 that dont exist in df1 will be added to those of df1.
    MUCH faster than looping thru each col and adding.
    PARAMS:
    - columns_of_df2_to_add_to_df1, list of str, columns of df2. Note: it is OK if
    columns exist in df1 (ignores them) or there are repeats in columns_of_df2_to_add_to_df1.

    Guaranteed that will work, or else fails. (e.g, checks indices match)
    RETURNS:
    - new df, without modding inputs.
    """

    if not np.all(df1.index==df2.index):
        print(len(df1.index))
        print(len(df2.index))
        print(df1.index[:10])
        print(df2.index[:10])
        assert False

    # assert len(df1)==len(df2)

    # These are required or else gets weird
    columns_of_df2_to_add_to_df1 = list(set(columns_of_df2_to_add_to_df1))
    columns_of_df2_to_add_to_df1 = [col for col in columns_of_df2_to_add_to_df1 if col not in df1.columns]

    if len(columns_of_df2_to_add_to_df1)>0:
        df12 = df1.join(df2.loc[:, columns_of_df2_to_add_to_df1])
    else:
        df12 = df1.copy()

    return df12


def merge_subset_indices_prioritizing_second(df_old, df_new, index_col=None):
    """
    Merge two dataframes, prioritizing df_new.
    Takes dataframes that have overlapping indices (in column, index_col),
    such that, for indices that that both have, take the value from df_new,
    otherwise keep the value from df_old.
    If df_new has index that doesn't exist in df_old, it will be added as a new
    row, with null values for columns that exist in df_old but not df_new.

    :param df_old: The older DataFrame to merge
    :param df_new: The newer DataFrame whose values are prioritized
    :param index_col: The column name to use as the index for merging. None means use df.index
    :return: Merged DataFrame with prioritization for df_new
    """

    if index_col is not None:
        # Then reset the index to this clumns values
        # Ensure the index_col is the index for both DataFrames
        df_old.set_index(index_col, inplace=True, drop=False)
        df_new.set_index(index_col, inplace=True, drop=False)

    # Combine DataFrames, prioritizing df_new
    merged_df = df_new.combine_first(df_old)

    # Optionally, reset the index if you want index_col to be a column again
    merged_df.reset_index(drop=True, inplace=True)

    return merged_df


#############################vvvv OBSOLETE - USE aggregGeneral
# def aggreg(df, group, values, aggmethod=["mean","std"]):
#     """
#     get group means, for certain dimensions(values). 
#     e.g., group = ["worker_model", "worker_dat"]
#     e.g. values = ["score", "cond_is_same"]
#     NOTE: will change name of balues filed, e.g. to score_mean.
#     OBSOLETE - USE aggregGeneral]
#     """
#     assert False, "[OBSOLETE - use aggregGeneral]"
#     # this version did not deal with non-numeric stuff that would liek to preset
#     # but was useful in taking both mean and std.
#     df = df.groupby(group)[values].agg(aggmethod).reset_index()
#     # df.columns = df.columns.to_flat_index()
#     df.columns = ['_'.join(tup).rstrip('_') for tup in df.columns.values]
#     return df

# def aggregMean(df, group, values, nonnumercols=[]):
#     """
#     get group means, for certain dimensions(values). 
#     e.g., group = ["worker_model", "worker_dat"]
#     e.g. values = ["score", "cond_is_same"]
#     e.g. nonnumercols=["sequence", "name"] i.e., will take the first item it encounters.
#     [OBSOLETE - USE aggregGeneral]
#     """
#     assert False, "[OBSOLETE - use aggregGeneral]"
#     agg = {c:"mean" for c in df.columns if c in values }
#     agg.update({c:"first" for c in df.columns if c in nonnumercols})
#     print(agg)
#     df = df.groupby(group).agg(agg).reset_index()
#     # df.columns = df.columns.to_flat_index()
#     # df.columns = ['_'.join(tup).rstrip('_') for tup in df.columns.values]
#     return df
########################### ^^^^ OBSOLETE - USE aggregGeneral


def aggregGeneral(df, group, values=None, nonnumercols=None, aggmethod=None):
    """
    Aggregate by first grouping (across multiple dimensions) and then applyiong
    arbnitrary method.
    PARAMS;
    - group, list of str, each a column. groups using combos fo these --> each is a new
    row.
    - values, list of str, which values to apply aggregation. the returned df will
    have these as new columns. if None, then will not add any new columsn other
    than whats in group
    - nonnumercols, list of str. these columsn will be retained, keeping only the 
    first encountered value.
    - aggmethod, list of str, applies each of these agg methods.
    """

    if values is None:
        # use dummy
        df = df.copy()
        df["_dummy"] = 1
        values = ["_dummy"]
        DELETE_DUMMY = True
    else:
        DELETE_DUMMY = False

    if aggmethod is not None:
        assert isinstance(aggmethod, list)
    assert isinstance(values, list)
    assert isinstance(group, list)

    if nonnumercols is None:
        nonnumercols = []
    else:
        # Check they all exist and has a single unique item per level of group
        for col in nonnumercols: 
            groupdict = grouping_append_and_return_inner_items(df, group, 
                groupinner=col, new_col_name="dummy")
            for k, v in groupdict.items():
                if len(v)!=1:
                    print(col)
                    print(group)
                    print(k)
                    print(v)
                    assert False, "nonnumercols takes the first. this is wrong if there are multipel levels for any nonnumercol..."

    if aggmethod is None:
        aggmethod = ["mean"]

    agg = {c:aggmethod for c in df.columns if c in values}
    agg.update({c:"first" for c in df.columns if c in nonnumercols})

    dfagg = df.groupby(group).agg(agg).reset_index()
    # df.columns = df.columns.to_flat_index()

    if len(aggmethod)==1:
        # then reanme columns so same as how they came in:
        # e.g., dist instead of dist_mean. can't do if 
        # multiple aggmethods, since will then be ambiguos.
    # df.columns = ['_'.join(tup).rstrip('_') for tup in df.columns.values]
        dfagg.columns = [tup[0] for tup in dfagg.columns.values]
    else:
        dfagg.columns = ['_'.join(tup).rstrip('_') for tup in dfagg.columns.values]

    if DELETE_DUMMY:
        del df["_dummy"]
        del dfagg["_dummy"]

    return dfagg

def df2dict(df):
    return df.to_dict("records")


def applyFunctionToAllRows(df, F, newcolname="newcol", replace=True, just_return_newcol=False):
    """F is applied to each row. is appended to original dataframe. F(x) must take in x, a row object
    - validates that the output will be identically indexed rel input.
    INPUT:
    - just_return_newcol, then returns single column without merging with df.
    """
    # To debug:
    # def F(x):
    #     return x["trial"]
    # # applyFunctionToAllRows(Pp, F, newcolname="test")
    # Ppsorted = Pp.sort_values("hausdorff").reset_index()
    # applyFunctionToAllRows(Ppsorted, F, newcolname="test")


    # NOTE: should actually jsut do this:
    # def F(x):
    #     sh = x["seqc_0_shape"]
    #     if sh in map_shape_to_shapelump.keys():
    #         return map_shape_to_shapelump[sh]
    #     else:
    #         return sh
    # SP.DfScalar["test"] = SP.DfScalar.apply(F, axis=1)

    # # OLD VERSION FAILS IF INDICES PASSED IN ARE NOT IN ORDER, 
    # # becuase of the reset_index()
    # dfnewcol = df.apply(lambda x: F(x), axis=1).reset_index()
    # dfout = df.merge(dfnewcol, left_index=True, right_index=True).rename(columns={0:newcolname})

    # GET NEW SERIES
    assert len(df)>0, "empty dataframe"
    dfnewcol = df.apply(F, axis=1).rename(newcolname) # rename, since series must be named.

    # DO MERGE
    # Make copy, since must delete if want to replace col
    dfthis = df.copy()
    if replace:
        if newcolname in dfthis.columns:
            del dfthis[newcolname]
    else:
        assert newcolname not in dfthis.columns, f"{newcolname} already exists as a col name"

    # print(dfnewcol)
    # print(dfnewcol.columns)
    # assert False
    # print(dfnewcol.index)
    if len(dfnewcol)!=len(dfthis):
        print(dfnewcol)
        print(dfthis)
        assert False

    # dfout = df.merge(dfnewcol, how="left", left_index=True, right_index=True)
    # _checkDataframesMatch(df, dfnewcol)

    if just_return_newcol:
        return dfnewcol

    dfout = dfthis.merge(dfnewcol, how="left", left_index=True, right_index=True, validate="one_to_one")

    if len(dfnewcol)!=len(dfout):
        print(dfnewcol)
        print(dfthis)
        print(dfout)
        assert False

    _checkDataframesMatch(df, dfout)

    return dfout

############3333 SCRATCH NOTES

# Apoply function to group-wise:
# Note; I was unsure whether the index pulled out is the global index or within the group.
# I think it is glocal.
# F = lambda x: x.iloc[x["distance"].idxmin()]
# F = lambda x: SF.iloc[x["distance"].idxmin()]
# # F = lambda x: SF.iloc[1]
# SF.groupby(["task", "epoch"]).apply(F)

######### take rows based on max/min of for values of some column:
# SF.loc[SF.groupby(["task", "epoch"])["distance"].idxmax()]


# pivoting (to take multiple rows and make them colums)
# Y = SFagg.pivot(index="task", columns="epoch", values="distance_median")

def filter_remove_rows_using_function_on_values(df, column, F, PRINT=False, reset_index=True):
    """ for each row, get the value and pass that in to F(value) --> bool
    Keep rows that return True
    RETURNS:
    - df, a copy of df, pruned. to remove rows that return F() = False
    """

    keeps = [F(val) for val in df[column].values.tolist()]

    if PRINT:
        print("nkeep / ntotal rows")
        print(sum(keeps), "/", len(keeps))

    df = df[keeps]

    if reset_index:
        df = df.reset_index(drop=True)

    return df


def filter_prune_min_n_rows(df, column, min_n):
    """ Returns copy of df pruning so that each  level of column has at leasnt
    min_n rows. Resets index of output
    """
    assert False, 'code it'


def prune_min_ntrials_across_higher_levels(df, col_high, col_low, n_min):
    """ Wrapper for filterGroupsSoNoGapsInData
        e.g, only keep characters that have at least 5 trials in each epoch:
    self.prune_min_ntrials_across_higher_levels("epoch", "character", 5)
    PARAMS:
    - col_high, the higher-level column, string, e.g., "epoch"
    - col_low, the lower-level column, e.g., "character".
    RETURNS:
    - df, pruned. (Does not modify self.Dat)
    """
    return filterGroupsSoNoGapsInData(df, col_low, col_high, min_n_trials=n_min)

def filterGroupsSoNoGapsInData(df, group, colname, colname_levels_to_check=None,
        min_n_trials = 1):
    """ filter df so that each group has at
    least one item for each of the desired values
    for the column of interest. useful for removing
    data which don't have data for all days, for eg...
    -- e.g., this only keeps tasks that have data for
    both epochs:
        colname_levels_to_check = [1,2]
        colname = "epoch"
        group = "unique_task_name"
    # NOTE - index of output will be reset.
    """

    if colname_levels_to_check is None:
        colname_levels_to_check = df[colname].unique().tolist()

    def F(x):
        """ True if has data for all values"""
        checks  = []

        # count n trials for each level of colname
        list_n =[]
        for v in colname_levels_to_check:
            n = sum(x[colname]==v)
            list_n.append(n)

        # check if each n is >= than min
        list_good = [n>=min_n_trials for n in list_n]
        return all(list_good)
        # for v in colname_levels_to_check:
        #     print(x[colname].values)
        #     checks.append(v in x[colname].values)
        # return all(checks)
    return df.groupby(group).filter(F).reset_index(drop=True)


def getCount(df, group, colname):
    """ return df grouped by group, and with one count value
    for each level in group, the name of that will be colname
    - colname must be a valid column from df [not sure why..]
    """

    return df.groupby(group)[colname].count().reset_index()


def bin_values_conditioned_on_class(df, var_bin, vars_condition, nbins,
                                    var_bin_ndim=1, new_col_name=None,
                                    bin_by_rank=False, value_if_grp_not_enough_data=-1):
    """
    Bin values in <var_bin> running separtely for each class of the careogtical
    varaible <vars_condition>, and appending to a new column.
    e..g, vcc is shape, and vb is location..., then bin indepently for each shape.

    NOTE: keeps throwing error in lemur: ValueError: Names should be list-like for a MultiIndex

    :param df:
    :param var_bin: str, name of variable to bin
    :param vars_condition: list of str.
    :param nbins: int, how mnay bins
    :param var_bin_ndim: 1 or 2, n dimensions. if 2, then bins each (x,y) dimenisions indepdenly. (--> tuple of ints).
    :param new_col_name: str to name new column, or None, in which case uses <var_bin>_binned
    :param value_if_grp_not_enough_data: if any grp only 1 datapt, then assigns this to it.
    :return: df copied, and with new column of binned values.
    """
    from pythonlib.tools.nptools import bin_values, bin_values_by_rank

    if new_col_name is None:
        new_col_name = f"{var_bin}_binned"
    if new_col_name in df.columns:
        df = df.drop(new_col_name, axis=1)

    # Define a function to apply binning to each group
    if var_bin_ndim==1:
        def bin_values_helper(group):
            if len(group)==1:
                # Then not enough data. call it -1
                group[new_col_name] = value_if_grp_not_enough_data
            else:
                # group['A_binned'] = pd.cut(group['A'], bins=bin_edges, labels=bin_labels, include_lowest=True)
                if bin_by_rank:
                    group[new_col_name] = bin_values_by_rank(group[var_bin], nbins=nbins)
                else:
                    group[new_col_name] = bin_values(group[var_bin], nbins=nbins)
            return group
    elif var_bin_ndim==2:
        def bin_values_helper(group):
            if len(group)==1:
                group[new_col_name] = value_if_grp_not_enough_data
            else:
                # group['A_binned'] = pd.cut(group['A'], bins=bin_edges, labels=bin_labels, include_lowest=True)

                values = np.stack(group[var_bin].tolist(), axis=0)
                assert values.shape[0]==len(group)
                assert values.shape[1]==2

                if bin_by_rank:
                    xs_binned = bin_values_by_rank(values[:,0], nbins=nbins)
                    ys_binned = bin_values_by_rank(values[:,1], nbins=nbins)
                else:
                    xs_binned = bin_values(values[:,0], nbins=nbins)
                    ys_binned = bin_values(values[:,1], nbins=nbins)
                # Convert to list of 2-tuples
                values_binned = [(x, y) for x, y in zip(xs_binned, ys_binned)]

                group[new_col_name] = values_binned
            return group
    else:
        print(var_bin_ndim)
        assert False

    # Apply binning separately for each class of B
    binned_df = df.groupby(vars_condition).apply(bin_values_helper)

    return binned_df

def binColumn(df, col_to_bin, nbins, bin_ver = "percentile"):
    """ bin values from a column, assign to a new column.
    - col_to_bin, string, name of col
    - nbins, 
    - bin_ver, how to spread out bin edges.
    NOTE:
    - bins will range from 0, 1, .., up to num gaps between edges.
    RETURN:
    - modified df in place, new col with name <col_to_bin>_binned
    - new_col_name, string
    """
    vals = df[col_to_bin].values
    df_orig = df.copy()
    # 2) Get bin edges
    if bin_ver=="uniform":
        # even bin edges
        binedges = np.linspace(min(vals)-0.1, max(vals)+0.1, nbins+1) 
    elif bin_ver=="percentile":
        # percentile bin edges
        binedges = np.percentile(vals, np.linspace(0, 100, nbins+1))
        binedges[0]-=1
        binedges[-1]+=1
    else:
        print(bin_ver)
        assert False, "not coded"
        
    # 3) assign bins
    vals_binned = np.digitize(vals, binedges, False)-1

    # 4) Put back into dataframe
    new_col_name = f"{col_to_bin}_binned"
    df[new_col_name] = vals_binned
    print(f"added column: {new_col_name}")

    _checkDataframesMatch(df, df_orig)
    return new_col_name

def aggregThenReassignToNewColumn(df, F, groupby, new_col_name, 
    return_grouped_df=False, overwrite_col=True):
    """ groups, then applies aggreg function, then reassigns that output
    back to df, placing into each row for each item int he group.
    e.g., if all rows split into two groups (0, 1), then apply function, then 
    each 0 row will get the same new val in new_col_name, and so on.
    - F, function to apply to each group.
    - groupby, list of str hwo to group
    - new_col_name, what to call new col.
    output will be same size as df, but with extra column.
    - If groupby is [], then will apply to all rows
    """

    assert isinstance(groupby, (list, tuple))

    if new_col_name in df.columns:
        del df[new_col_name]

    if len(groupby)==0:
        # then add dummy column
        dummyname="dummy"
        while dummyname in df.columns:
            dummyname+="1"
        df[dummyname] = 0
        groupby = [dummyname]
        remove_dummy=True
    else:
        remove_dummy=False

    assert 0 not in df.columns, "the new coliumn is expected to be called 0"
    dfthis = df.groupby(groupby).apply(F).reset_index()
    assert 0 in dfthis.columns
    dfthis = dfthis.rename(columns={0:new_col_name})
    # dfthis will be smaller than df. but merge will expand dfthis.

    # NOTE: sanity check, mking sure that merge does correctly repopulate:
    # from pythonlib.tools.pandastools import aggregThenReassignToNewColumn
    # def F(x):
    #     return [list(set(x[kbin])), list(set(x["stroknum"]))]
    # tmp = aggregThenReassignToNewColumn(P.Xdataframe, F, [kbin, "stroknum"], "test")
    # # LOOK THRU THIS AND MAKE SURE EACH PAIR IS IDENTICAL.
    # for _, row in tmp.iterrows():
    #     if row["test"] != [row[kbin], row["stroknum"]]:
    #         print(row["test"])
    #         print([[row[kbin]], [row["stroknum"]]])
    #         print("--")

    
    if False:
        # old version, didnt work for cases where input index were not in order.

        # df_new = pd.merge(df, dfthis, on=groupby)
        # df_new = pd.merge(df, dfthis, how="left", on=groupby)

        # df_new = df.merge(dfthis, how="left", on=groupby)
        df_new = df.reset_index().merge(dfthis, how="left", on=groupby).set_index("index") # maintains indices.
    else:
        df_new = _mergeKeepLeftIndex(df, dfthis, how='left',on=groupby)

    # remove dummy
    if remove_dummy:
        del df[dummyname]
        del dfthis[dummyname]
        del df_new[dummyname]

    _checkDataframesMatch(df_new, df)

    if return_grouped_df:
        return df_new, dfthis
    else:
        return df_new

def filterPandas(df, filtdict, return_indices=False, auto_convert_tolist=True, 
        reset_index=True):
    """ 
    filtdict is dict, where each value is a list of
    allowable values.
    PARAMS:
    - auto_convert_tolist, then any values of filtdict that arent lsits are converted
    to lists. assumes you tried to enter a single value to filter.
    - See filtdict for format
    NOTE - doesnt modify in place. just returns.
    NOTE - return_indices, returns the original row indices
    (as a list of ints) instead of the modified df
    NOTE - if return dataframe, automaticlaly resets indices.
    """
    # assert False, "dont use this, just use pandas .isin()"

    for k, v in filtdict.items():
        if not isinstance(v, list):
            if auto_convert_tolist:
                    v = [v]
            else:
                assert isinstance(v, list), "must make into lists the values of filtdict"
        if len(df)>0:
            if isinstance(v[0], np.ndarray) or isinstance(df[k].values[0], np.ndarray):
                from .nptools import isin_close
                def _F(x):
                    return isin_close(x[k], v)
                trues = applyFunctionToAllRows(df, _F, just_return_newcol=True)
            else:
                trues = df[k].isin(v)
            df = df[trues]
            # print(len(df))
    if return_indices:
        return list(df.index)
    else:
        if reset_index:
            return df.reset_index(drop=True)
        else:
            return df


def filterPandasMultOrs(df, list_varnames, list_filts, return_as = "inds",
    verbose=False):
    """ Apply multiple filters in a row, and get the union of all
    the resulting indices into df. 
    Flexible way to define filters.
    PARAMS;
    - df, dataframe
    - list_varnames, list of str, the column names to consider
    - list_filts, list of lists, where each inner list has items
    which define what levels to keep fore ach var. the final output
    is the conjunction of each of these inner items. A few ways to define.
    e..g, list_filts = [list1, list2, ... (arbitrary number of lists)], where
    list1 = [list_of_levels_for_var1, list_of_levels_for_var2], where var1 and var2 are in list_varnames
    list1 = [level_for_var1,...] so like above, but a single level
    list1 = [None, ..] where None means keep all levels (i.e., ignore this variable).
    RETURNS:
    - Depends on return_as:
    --- if "inds", then indices (corresponding to df.index), 
    --- if "dataframe", then df
    --- if "dict", then list of dicts
    NOTE:
    - useful, for eg. if taking subset of data as heldout data for model testing.
    - is "or" for outer list, "and" for inner list.
    EG:
    list_varnames = DATAPLOT_GROUPING_VARS
    list_filts = [
        ["Lcentered-4-0", (1,1)],
        ["V-2-0", None],
        [["squiggle3-1-0", "circle-1-0"], None],
        [["squiggle3-1-0", "circle-1-0"], [(-1,1), (-1,-1)]]
    ]

    """
    
    list_inds = []
    for innerlist in list_filts:
        # get trials that are conjucntion of these features
        F = {}
        for varname, levels in zip(list_varnames, innerlist):
            if levels is None:
                # skip, i.e., get all levels
                continue
            if isinstance(levels, list):
                # then keep if it is any item in the list
                F[varname] = levels
    #         elif isinstance(levels, (str, tuple)):
            else:
                # this is assumed to be a single item. put in a list
                F[varname] = [levels]

        if verbose:
            print("For this filt: ", innerlist, ' -- Using this filt: ' , F)

        # Apply filter 
        inds = filterPandas(df, F, return_indices=True)
        list_inds.extend(inds)

    list_inds = sorted(set(list_inds))
    if verbose:
        print("Got this many indices: ", len(list_inds))

    if return_as=="dataframe":
        return df.iloc[list_inds]
    elif return_as=="inds":
        return list_inds
    elif return_as=="dict":
        return df.iloc[list_inds].to_dict("records")
    else:
        assert False

def filter_by_min_n(df, colname, n_min_per_level, types_to_consider=(str, int)):
    """ for each level in df[colname] prune all of its instances if
    the n rows is less than n_min_per_level
    PARAMS;
    - colname, str
    - n_min_per_level, int
    - types_to_consider, list of types, only prune levels that are type in this list.
    This is important so that doesnt throw out all numerical data.
    RETURNS:
    - df, a copy of the input, pruned
    """

    # n_min_per_level = 40 # min total n trials for level
    from pythonlib.tools.checktools import check_is_categorical

    df["_index"] = df.index
    levels = df[colname].unique().tolist()

    indstoremove = []

    print("--- [filter_by_min_n], checking var=", colname)
    for lev in levels:
        # Skip if this level is numerical
        if check_is_categorical(lev):
        # if isinstance(lev, types_to_consider):
            n = sum(df[colname]==lev)
            if n<n_min_per_level:
                # then remove it
                inds = df[df[colname]==lev].index.tolist()
                indstoremove.extend(inds)
                print(f"level {lev}, REMOVING, n={n}")
            else:
                print(f"level {lev}, keeping, n={n}")
    print('(removing this many indices): ', len(set(indstoremove)))
    df = df.copy()
    df = df.drop(list(set(indstoremove))).reset_index(drop=True)
    return df

def findPandas(df, colname, list_of_vals, reset_index=True):
    """ returns df with only rows matchibng list_of_vals. 
    output will be same length as list_of_vals, with order matching.
    INPUT:
    - df
    - colname, the column to check
    - list_of_vals, the values to pick out the rows
    RETURNS:
    - dfout
    NOTE
    - is doing it in a slow way, iterating over items.
    - will fail if list_of_vals are not unique, or any val is not found
    - will reset index
    """
    assert len(set(list_of_vals))==len(list_of_vals), "values not all unique"

    dfout = pd.concat([df[df[colname]==v] for v in list_of_vals])

    assert len(dfout)==len(list_of_vals), "at least one val was not found in df"
    tmp = dfout[colname].tolist()
    for a,b in zip(tmp, list_of_vals):
        assert a==b, "mistake somewher"

    if reset_index:
        dfout = dfout.reset_index(drop=True)
    return dfout



def pivotAndMerge(d1, df2):
    """
    """
    assert False, "in progress"

    # from line2strokmodel.py
    DF2 = pd.pivot_table(DF, index=["trial", "strok_num_0"], values="finalcost", columns="nsubstrokes").reset_index()
    DFstrokdur = pd.pivot_table(DF, index=["trial", "strok_num_0"], values="strok_dur").reset_index()
    DF2 = pd.merge(DF2, DFstrokdur)

def printOverview(df, MAX=50):
    """
    print columns and values counts)
    - MAX, if num values greater than this, then skip printing.
    """
    if False:
        # old version, doesnt plto frequency of values
        from .dicttools import printOverviewKeyValues
        printOverviewKeyValues(df2dict(df)) 
    else:
        for col in df.columns:
            print(' ')
            print(f"-- {col}")
            if isinstance(df[col].values[0], list):
                print(f"*Skipping print, since type of values are lists")
                continue
            if isinstance(df[col].values[0], np.ndarray):
                print(f"*Skipping print, since type of values are np.ndarray")
                continue
            if isinstance(df[col].values[0], dict):
                print(f"*Skipping print, since type of values are dict")
                continue
            nvals = len(set(df[col].values))
            if nvals>MAX:
                print(f"*Skipping print, since {nvals} vals > MAX")
            else:
                print(df[col].value_counts())



# @param grp: [col1 col2 col3]
# @return: new column with index for category (based on col1/2/3 perm)
def append_col_with_grp_index(df, grp, new_col_name, use_strings=True, 
        strings_compact=True, return_col_name=False):
    """ for each col, gets its grp index (based on grp list),
    and appends as new column. first converts to string by str(list)
    INPUTS:
    - grp, list of strings, each a column. order matters!
    - use_strings, bool (True), then items are strings, otherwise are tuples.
    RETURNS:
    - df, but with new col. either values as string or tuples
    """
    from pythonlib.tools.pandastools import applyFunctionToAllRows
    
    for x in grp:
        assert not isinstance(x, (list, tuple)), "leads to errors. you prob want to flatten grp"

    # if isinstance(grp, tuple):
    #     grp = list(grp)

    # add a column, which is grp index
    def F(x):
        tmp = [x[g] for g in grp]
        if use_strings:
            if strings_compact:
                tmp_compact = []
                for t in tmp:
                    if isinstance(t, bool):
                        if t:
                            tmp_compact.append("1")
                        else:
                            tmp_compact.append("0")
                    else:
                        tmp_compact.append(str(t))
                return "|".join(tmp_compact)
            else:
                return str(tmp) 
        else:
            return tuple(tmp)
    df = applyFunctionToAllRows(df, F, new_col_name)  

    if return_col_name:
        return df, new_col_name 
    else:
        return df


def append_col_after_applying_to_group(df, groupby, cols_to_use, F, newcol):
    """ F takes in a group (dframe) and outputs a series of same length.
    appends the combined output to a new col with name newcol.
    Confirmed that the locations (rows) will be correctly aligned with the original
    dataframe.
    - cols_to_use, will pull out datafram df.groupby(groupby)[cols_to_use]. is list. THis
    goes into F.
    - F, dataframe --> series (or list)
    - newcol, name of new column
    NOTE: is like applyFunctionToAllRows but here applies function taking in a group, not a row.
    """

    # groupby = "character"
    # newcol = "test"

    assert isinstance(cols_to_use, list), "so get dataframe out"

    # 1) get the series.
    df[newcol] = df.groupby(groupby)[cols_to_use].transform(F)
    # dfnew[newcol] = dfnew[groupby]
    # del dfnew[groupby]
    # pd.concat([dfthis, dfnew], axis=1)

    return df

def append_col_with_index_of_level_after_grouping(df, grpvars, column, newcolumn):
    """ 
    Number each class of <column> after first grouping by levels of <grpvars>, and in increasing
    order after sorting. E..g, if you want to give each unqiue character an interger label, but first
    grouping characters by epoch...
    PARAMS:
    - grpvars: list of str --> first goruping step.
    - column: str, the column whose labels will be sorted and convberted to integer classes.
    - newcolumn: str, stores results here.
    """

    df = df.copy()

    _check_index_reseted(df)

    if grpvars is None:
        df = df.copy()
        assert "_dummy" not in df.columns
        df["_dummy"] = "dummy"
        grpvars = ["_dummy"]

    n_in = len(df)
    groupdict = grouping_append_and_return_inner_items_good(df, grpvars)
    list_df = []
    for _, indices in groupdict.items():
        dfthis = df.iloc[indices].reset_index(drop=True)
        append_col_with_index_of_level(dfthis, column, newcolumn)
        list_df.append(dfthis)
    dfout = pd.concat(list_df).reset_index(drop=True)   

    assert len(df)==n_in

    return dfout

def append_col_with_index_of_level(df, column, newcolumn):
    """ 
    APpends a new column which converts levels of <column> into indices, 0, 1,..
    assuming colum is categorical.
    PARAMS:
    - column, string col in df
    - newcolumn, str, name of new column 
    RETURNS:
    (mutates df to have this new column)
    - map_idx_to_level, dict
    - map_level_to_idx, dict
    """

    levels = sort_mixed_type(df[column].unique().tolist())
    
    # map_idx_to_level, map_level_to_idx = map
    map_level_to_idx = {lev:i for i, lev in enumerate(levels)}
    map_idx_to_level = {}
    for lev, idx in map_level_to_idx.items():
        assert idx not in map_idx_to_level.keys()
        map_idx_to_level[idx] = lev


    # get list, len(dtapts)
    idxs = [map_level_to_idx[lev] for lev in df[column].tolist()]
    # Append
    df[newcolumn] = idxs

    return map_idx_to_level, map_level_to_idx

def append_col_with_index_number_in_group(df, groupby, colname="trialnum_chron", randomize=False):
    """ appends a col, which holds index (0, 1, 2.) in order within its level within groupby.
    ie each row is a different index!
    e.g, if groupby has 2 levels (A and B), then this gives all rows with level A unqiue indices 0, 1,2 ...
    e.g.. like trial numbers for a given condition/task.
    - randomize, then will randomize the indices (only within trails with same level of groupby)
    """

    # OLD method - avoid since it mutates
    # dfthis = D.Dat
    # def F(x):
    #     # assign in chron order
    #     x["test"] = range(len(x))
    # #     x["test"] = 1
    #     return x
        
    # dfthis = dfthis.groupby("character").apply(F)

    df = df.copy()

    def F(x):
        # assign in chron order
        out = list(range(len(x)))
        if randomize:
            from random import shuffle
            shuffle(out)
        return out

    return append_col_after_applying_to_group(df, groupby, [groupby], F, colname)    

def convert_to_1d_dataframe_hist(df, col1, plot_hist=True, ax=None):
    """ Aggregate (usually counts for each level of col1).
    Plots a histogram, ordered by the n counts
    RETURNS:
    - labels, ncounts, list of str and ncounts, matching, sorted from high to low.
    - fig, ax
    """
    import matplotlib.pyplot as plt
    x = df[col1].value_counts()
    labels = x.index
    ncounts = x.values
    if plot_hist:
        if ax is None:
            fig, ax = plt.subplots(1,1)
        else:
            fig = None
        ax.bar(labels, ncounts)
        # ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45)
        plt.xticks(rotation=90)

        # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        # df[col1].hist(xrot=45)
    else:
        fig, ax = None, None
    return labels, ncounts, fig, ax

def convert_to_2d_dataframe(df, col1, col2, plot_heatmap=False, 
        agg_method = "counts", val_name = "val", ax=None, 
        norm_method=None,
        annotate_heatmap=True, zlims=(None, None),
        diverge=False, dosort_colnames=False,
        list_cat_1 = None, list_cat_2 = None):
    """ Reshape dataframe (and prune) to construct a 2d dataframe useful for 
    plotting heatmap. Eech element is unique combo of item for col1 and col2, 
    with a particular aggregation function (by default is counts). 
    PARAMS:
    - col1, string name of col whose values will become row indices
    - col2, string name of col whose values wil become column indices.
    - plot_heatmap, bool
    - agg_method, str, what to put into each cell (by agging across datapts)
    - val_name, str name of column, optional if the agg function requires this (e,g. mean)
    - annotate_heatmap, bool, to put numerical values in each cell as text.
    - diverge, bool, if True, then centers the heat values.
    RETURNS:
    - 2d dataframne,
    - fig, 
    - ax,
    - rgba_values, (nrows, ncols, 4), where rgba_values[0,1], means rgba value for row 0 col 1.
    """
    from pythonlib.tools.snstools import heatmap

    assert isinstance(val_name, str)

    # If col2 is None, then give a dummy varaible, so that this code runs
    if col2 is None:
        # then is really a 1d plot
        df = df.copy()
        df["dummy"] = 0
        col2="dummy"

    if list_cat_1 is None:
        list_cat_1 = df[col1].unique()

    if list_cat_2 is None:
        list_cat_2 = df[col2].unique()

    if dosort_colnames:
        list_cat_1 = sort_mixed_type(list_cat_1)
        list_cat_2 = sort_mixed_type(list_cat_2)

    arr = np.zeros((len(list_cat_1), len(list_cat_2)))
    for i, val1 in enumerate(list_cat_1):
        for j, val2 in enumerate(list_cat_2):
            dfsub = df[(df[col1] == val1) & (df[col2] == val2)]
            if agg_method=="counts":
                # get counts
                n = len(dfsub)
                valthis = n
            elif agg_method=="mean":
                valthis = dfsub[val_name].mean()
            else:
                print(agg_method)
                assert False
            arr[i, j] = valthis

    dfthis = pd.DataFrame(arr, index=list_cat_1, columns=list_cat_2)

    # if norm_method=="all_sub":
    #     # minus mean over all cells
    #     dfthis = dfthis - dfthis.mean(axis=None)
    #     diverge = True
    # elif norm_method=="col_div":
    #     # normalize so that for each col, the sum across rows is 1
    #     assert np.all(dfthis>=0), "cant norm by dividing unless all vallues are >0"
    #     dfthis = dfthis.div(dfthis.sum(axis=0), axis=1)
    # elif norm_method=="row_div":
    #     # same, but for rows
    #     assert np.all(dfthis>=0), "cant norm by dividing unless all vallues are >0"
    #     dfthis = dfthis.div(dfthis.sum(axis=1), axis=0)
    # elif norm_method=="all_div":
    #     # divide by sum of all counts
    #     assert np.all(dfthis>=0), "cant norm by dividing unless all vallues are >0"
    #     dfthis = dfthis/dfthis.sum().sum()
    # elif norm_method=="col_sub":
    #     # normalize so by subtracting from each column its mean across rows
    #     dfthis = dfthis.subtract(dfthis.mean(axis=0), axis=1)
    #     diverge = True
    # elif norm_method=="col_sub_notdiverge":
    #     # normalize so by subtracting from each column its mean across rows
    #     dfthis = dfthis.subtract(dfthis.mean(axis=0), axis=1)
    #     diverge = False
    # elif norm_method=="row_sub":
    #     # normalize so by subtracting from each column its mean across rows
    #     dfthis = dfthis.subtract(dfthis.mean(axis=1), axis=0)
    #     diverge = True
    # elif norm_method=="row_sub_firstcol":
    #     # for each item in a given row, subtract the value of the first colum in that row.
    #     dfthis = dfthis.subtract(dfthis.iloc[:,0], axis=0)
    # elif norm_method is None:
    #     pass
    # else:
    #     print(dfthis)
    #     print(norm_method)
    #     assert False

    if plot_heatmap:
        fig, ax, rgba_values = heatmap(dfthis, ax, annotate_heatmap, zlims,
                                       diverge=diverge, norm_method=norm_method)
        ax.set_xlabel(col2)
        ax.set_ylabel(col1)
        ax.set_title(agg_method)
    else:
        fig, ax, rgba_values = None, None, None

    if "dummy" in df.columns:
        del df["dummy"]
    return dfthis, fig, ax, rgba_values
            

def expand_categorical_variable_to_binary_variables(df, var_categorical,
                                                    index_vars=None):
    """
    Given a categortical variable, expand it so that each level that
    exists becomes its own binary variable. (so increases n rows by n_old_rows *
    n_levels_of_var)
    E.g., useful if you assign each row a string label (i.e., "multinomial), but you want to
    represent, for each row, True/False over all the levels (i.e., "contrasts").
    PARAMS:
    - var_categorical, column in df. each level that exists will become a new binary
    variable in output.
    - index_vars, list of columns in df. will extract these variables for each row
    of df.
    RETURNS:
    - dfout, num rows will be (num levels of var_categorical) x (num rows in df), with
    columns, <vars in index_vars>, <var_categorical> (the name of the level) and,
    "value" (int).
    EXAMPLE:
    dflabels = expand_categorical_variable_to_binary_variables(D.Dat, SEQUENCE_VAR, ["trialcode", "epoch"])
    """

    if False:
        # Old version, using for-loop so is very slow.
        list_levels = sort_mixed_type(df[var_categorical].unique().tolist())

        if index_vars is None:
            # Then get all columns that exist
            index_vars = [c for c in df.columns if not c==var_categorical]

        # Collet across each row * level
        dats = []
        for ind in range(len(df)):
            for lev in list_levels:

                # Binary variable, get its value
                match = lev == df.iloc[ind][var_categorical]

                # append a new item
                d = {
                    var_categorical:lev,
                    "value":match,
                    # "trialcode":df.iloc[ind]["trialcode"],
                    # "epoch":df.iloc[ind]["epoch"]
                }

                # collect index vars for this line in df
                for v in index_vars:
                    d[v] = df.iloc[ind][v]

                dats.append(d)
        df_binary = pd.DataFrame(dats)
    else:
        dftmp = pd.get_dummies(df, prefix="", prefix_sep="", columns=[var_categorical])
        id_vars = [c for c in df.columns if not c==var_categorical]
        list_levels = sort_mixed_type(df[var_categorical].unique().tolist())
        df_binary = pd.melt(dftmp, id_vars = id_vars, value_vars=list_levels, var_name=var_categorical)

    # NOTES: couple other approeaches I tried which difnt work well.
    # var = "choice_code_str"
    # dftmp = pd.get_dummies(df_actions, prefix_sep="-", columns=[var])
    # # - convert to long
    # pd.wide_to_long(dftmp, [var], ["trialcode", "idx_beh"], var)
    #
    # # Doesnt work... gived weird values.
    # list_levels = sorted(df_actions[var].unique().tolist())
    # groups = {
    #     f"{var}_new":list_levels
    # }
    # dftmp = pd.get_dummies(df_actions, prefix="", prefix_sep="", columns=[var])
    # pd.lreshape(dftmp, groups)

    return df_binary


def unpivot(df, id_vars, value_vars, var_name, value_name):
    """ Convert from wide-form to long-form.
    PARAMS:
    - id_vars, list of str, grouping variable defining new rows.
    - value_vars, list of str, values for these columns will be stacked vertically.
    Thus nrows = len(value_vars)*len(df)
    - var_name, str, what to call the new column for which the values are value_vars.
    Cannot use "variable"
    - value_name, str, what to call the new column holding the values. Cannot use "value"
        chan    var     var_others  event   val_kind    val_method  bregion     val     val_interaction     val_others
        0   2   epoch   (taskgroup,)    00_fixcue_50_to_600     modulation_subgroups    r2smfr_running_maxtime_twoway   M1_m    0.000854    0.014409    0.010222
        1   2   epoch   (taskgroup,)    02_samp_50_to_600   modulation_subgroups    r2smfr_running_maxtime_twoway   M1_m    0.002288    0.005289    0.004021
        2   277     epoch   (taskgroup,)    00_fixcue_50_to_600     modulation_subgroups    r2smfr_running_maxtime_twoway   vlPFC_p     0.001375    0.013763    0.007092
        3   277     epoch   (taskgroup,)    02_samp_50_to_600   modulation_subgroups    r2smfr_running_maxtime_twoway   vlPFC_p     0.160906    0.018665    0.020233  

            chan    var     var_others  event   val_kind    val_method  bregion     anova_source    peta2
        0   2   epoch   (taskgroup,)    00_fixcue_50_to_600     modulation_subgroups    r2smfr_running_maxtime_twoway   M1_m    val     0.0008541354921786013
        1   277     epoch   (taskgroup,)    00_fixcue_50_to_600     modulation_subgroups    r2smfr_running_maxtime_twoway   vlPFC_p     val     0.0013754887892285677
        2   2   epoch   (taskgroup,)    02_samp_50_to_600   modulation_subgroups    r2smfr_running_maxtime_twoway   M1_m    val     0.0022879608882661736
        3   277     epoch   (taskgroup,)    02_samp_50_to_600   modulation_subgroups    r2smfr_running_maxtime_twoway   vlPFC_p     val     0.160906210839649
        4   2   epoch   (taskgroup,)    00_fixcue_50_to_600     modulation_subgroups    r2smfr_running_maxtime_twoway   M1_m    val_others  0.010222271105982818
        5   277     epoch   (taskgroup,)    00_fixcue_50_to_600     modulation_subgroups    r2smfr_running_maxtime_twoway   vlPFC_p     val_others  0.007092412522261568
        6   2   epoch   (taskgroup,)    02_samp_50_to_600   modulation_subgroups    r2smfr_running_maxtime_twoway   M1_m    val_others  0.004020582538782944
        7   277     epoch   (taskgroup,)    02_samp_50_to_600   modulation_subgroups    r2smfr_running_maxtime_twoway   vlPFC_p     val_others  0.020232822456417488
        8   2   epoch   (taskgroup,)    00_fixcue_50_to_600     modulation_subgroups    r2smfr_running_maxtime_twoway   M1_m    val_interaction     0.014408870995814671
        9   277     epoch   (taskgroup,)    00_fixcue_50_to_600     modulation_subgroups    r2smfr_running_maxtime_twoway   vlPFC_p     val_interaction     0.013762569757076468
        10  2   epoch   (taskgroup,)    02_samp_50_to_600   modulation_subgroups    r2smfr_running_maxtime_twoway   M1_m    val_interaction     0.005289383533117516
        11  277     epoch   (taskgroup,)    02_samp_50_to_600   modulation_subgroups    r2smfr_running_maxtime_twoway   vlPFC_p     val_interaction     0.018665206122281126
    """

    assert not var_name == "variable"
    assert not value_name == "value"
    assert not "variable" in list(df.columns)
    assert not "value" in list(df.columns)

    _value_name = "value"
    tmp = [value_name] + list(df.columns)
    while _value_name in tmp:
        _value_name+="X"
    df_melt = pd.melt(df, id_vars=id_vars,
            value_vars=value_vars, value_name=_value_name)
    
    # NOTE: the reason do this insead of passing var_name into pd.melt is if do that then sometimes end with bug if
    # var_name already exists as a column name.
    df_melt[var_name] = df_melt["variable"]
    df_melt = df_melt.drop("variable", axis=1)

    df_melt[value_name] = df_melt[_value_name]
    df_melt = df_melt.drop(_value_name, axis=1)

    return df_melt

def pivot_table(df, index, columns, values, aggfunc = "mean", flatten_col_names=False,
    flatten_separator="-", col_strings_ignore=None):
    """
    Take a long-form datagrame, and convert into a wide form. 
    PARAMS:
    - index, list of str, defining a grouping variable. Each level of this variable will be a
    new row in the output dataframe.
    - columns, list of str or str, a grouping variable. each level of this variable will define a
    new column in output dataframe. (generally keep this length 1, easy to understand. if len >1, then will be hierarchical 
    columns)
    - values, list of str, the response variable, whose values will populate the cells in the output dataframe,
    will be aggregated (mean) across instance with same col and index.
    - flatten_col_names, if output is hierarchical, will flatten to <val>-<col1>-<col2>.., if 
    where col1, 2, ... are the items in columsn (if it is a list)
    RETURNS:
    - new dataframe, where can access e.g., by df["value"]["col_level"]
    NOTES:
    - Also aggregates, by taking mean (for a given value) over all cases with a given combo of (index, columns)
    - essentially groups by the uniion of index, columns, then aggregates, then reshapes so that index
    is over rows, and volumns/values are over columns (hierarhcical)
    - resets index, so that index levels will make up a column
    e.g.:
    - index=["unique_task_name"], columns=["block"], values=["time_touchdone", "time_raise2firsttouch"]
    - Look into pd.melt(dftmp, id_vars = ["block", "unique_task_name"]), if want to undo this. would need to 
    not rename columns.
    - can eitehr do:
    (1) agg, then do this or
    (2) do this directly (since it aggs by taking mean.)
    """

    dftmp = pd.pivot_table(data=df, index=index, columns=columns, values=values, aggfunc=aggfunc).copy()

    if col_strings_ignore is None:
        col_strings_ignore = []
    if flatten_col_names:
        dftmp.columns = [flatten_separator.join([str(c) for c in col if c not in col_strings_ignore]).strip() for col in dftmp.columns.values]

    # to reindex and retain indices as new columns
    dftmp = dftmp.reset_index()

    # OLD VERSION WHERE DID THE SAME THING BUT BY HAND
    # column_levels = [17, 18]
    # column_dsets = "block"
    # value = "time_touchdone"
    # def F(x):
    # #     cols=[str(column_levels[0]), str(column_levels[1])]
    #     cols=column_levels
    #     def _getval(x, level):
    #         val = x[x[column_dsets]==level][value].values
    #         if len(val)==0:
    #             return np.nan
    #         elif len(val)>1:
    #             assert False
    #         else:
    #             return val[0]
    #     val1 = _getval(x, column_levels[0])
    #     val2 = _getval(x, column_levels[1])

    #     tmp = [val1, val2]
    #     tmp = pd.Series(tmp, index=cols)
    #     return tmp

    # DF.groupby(["unique_task_name"]).apply(F).reset_index()


    return dftmp



def summarize_feature(df, GROUPING, FEATURE_NAMES,
                          INDEX= ("character", "animal", "expt"), 
                          func = None, newcol_variable=None, newcol_value=None):
    """ [USEFUL] wide-form --> long form
    aggregating and summarizing features
    See summarize_featurediff for variables.
    NOTE:
    - if have N rows, and 8 columsn with scores under 8 different models, and want to flatten to long-form
    so that there is a single "score" column with Nx8 rows (useful for plotting). Can do following:
    summarize_feature(D.Dat, "epoch", model_score_name_list, ["character", "trialcode"]), where
    model_score_name_list is list of 8 column names. The outcome is, where "value" is the "score" column.

    dfthisflat =
         epoch   character   trialcode   variable    value
    0   baseline    mixture2-ss-2_1-111763  210821-1-205    behmodpost_baseline_chunks  0.074286
    1   baseline    mixture2-ss-2_1-111763  210821-1-273    behmodpost_baseline_chunks  0.020258
    2   baseline    mixture2-ss-2_1-111763  210821-1-364    behmodpost_baseline_chunks  0.053116
    3   baseline    mixture2-ss-2_1-111763  210821-1-438    behmodpost_baseline_chunks  0.020556
    4   baseline    mixture2-ss-2_1-111763  210821-1-478    behmodpost_baseline_chunks  0.063520
    ...     ...     ...     ...     ...     ...
    307     lolli   mixture2-ss-2_1-111763  210902-1-116    behmodpost_lolli_mkvsmk     0.017931
    308     lolli   mixture2-ss-2_1-111763  210902-1-213    behmodpost_lolli_mkvsmk     0.008365
    309     lolli   mixture2-ss-6_1-854929  210901-1-399    behmodpost_lolli_mkvsmk     0.013021
    310     lolli   mixture2-ss-6_1-854929  210901-1-598    behmodpost_lolli_mkvsmk     0.007421
    311     lolli   mixture2-ss-6_1-854929  210902-1-176    behmodpost_lolli_mkvsmk     0.010960

    (If want to aggregate over all trials, then use ["character"] instead. )

    dfthis is basically looks like the input shape, but pruned to the relevant columns.
    """
    if func is None:
        func = lambda x: np.nanmean(x)
    if not isinstance(GROUPING, list):
        GROUPING = [GROUPING]
    dfagg = aggregGeneral(df, GROUPING + INDEX, FEATURE_NAMES, aggmethod=[func])
    dfaggflat = pd.melt(dfagg, id_vars = GROUPING + INDEX)

    # change name
    if newcol_variable is not None:
        dfaggflat = dfaggflat.rename(columns ={"variable":newcol_variable})
    if newcol_value is not None:
        dfaggflat = dfaggflat.rename(columns ={"value":newcol_value})

    return dfagg, dfaggflat




def summarize_featurediff(df, GROUPING, GROUPING_LEVELS, FEATURE_NAMES,
                          INDEX= ("character", "animal", "expt"), 
                          func = None, return_dfpivot=False, 
                          do_normalize=False, normalize_grouping = ("animal", "expt"),
                          get_absolute_val = False
                         ):
    """ High level summary, for each task (or grouping), get its difference 
    across two levels for grouping (e..g, epoch 1 epoch2), with indices seaprated
    by INDEX (usually, animal/expt/character).
    INPUTS:
    - GROUPING, str, will take the difference of each item in FEATURE_NAMES between the two 
    levels in GROUPING. e.g., GROUPING = "epoch", then takes differfence between two epochs.
    - GROUPING_LEVELS, list of two str, the two levels to take difference of
    - FEATURE_NAMES, list of str, column names for numerical values that you wish to take
    difference of. e..g, FEATURE_NAMES = ["score"]. if len >1 then rtake difference separtely for each item.
    - INDEX, list of str, each a column, the conjunction defines the new row levels
    - func, how to aggregate across multipel rows.
    - do_normalize, if True, then dfsummaryflat will have normalized values.
    i.e., for each (animal, expt, variable), take frac change relative to GROUPING_LEVEL[0]
    i.e., (a-b)/abs(b), where b is after averaging over all tasks. Will put this in a new column
    caleed "value_norm"
    OUTPUT:
    - dfsummary, new dataframe, with rows = unique combos of index, and columns line:
    ("total_time") [see eg below]
    - dfsummaryflat, similar but flattened, so that only columns are to identify index
    NOTES:
    - e.g, starting from D, 
    --- INDEX = ["character", "animal", "expt"], (must be rows in input dataframe)
    --- GROUPING = ["plan_time_cat"] --> levels {"short, "long"}  (must be 
    rows in input dataframe, e.g., short is a value that plantimecat can take
    --- FEATURE_NAMES = ["total_time", "distance", ...] (must be columns in 
    input datafrane)
    # EXAMPLE:
    (levels for align_to are "stroke_onset" and "go_cue")
    dfsummary, dfsummaryflat, COLNAMES_NOABS, COLNAMES_ABS, COLNAMES_DIFF = 
                    summarize_featurediff(dfmodels, GROUPING="align_to", GROUPING_LEVELS=["go_cue", "stroke_onset"], 
                      FEATURE_NAMES=["score_test_mean"], INDEX=["regions_str", "yfeat"])
        
score_test_mean-stroke_onsetmingo_cue   score_test_mean-stroke_onsetmingo_cue-ABS   ntrain-stroke_onsetmingo_cue    ntrain-stroke_onsetmingo_cue-ABS    regions_str     yfeat
0   -0.001340   0.001340    0   0   ALL     gridloc
1   0.160566    0.160566    0   0   ALL     shape_oriented
2   -0.107563   0.107563    0   0   FP_p-FP_a   gridloc
3   0.017268    0.017268    0   0   FP_p-FP_a   shape_oriented
4   0.175209    0.175209    0   0   M1_m-M1_l   gridloc
5   0.284695    0.284695    0   0   M1_m-M1_l   shape_oriented
6   -0.094072   0.094072    0   0   PMd_a-PMd_p     gridloc
7   0.287808    0.287808    0   0   PMd_a-PMd_p     shape_oriented
8   0.069778    0.069778    0   0   PMv_m-PMv_l     gridloc
9   0.004547    0.004547    0   0   PMv_m-PMv_l     shape_oriented
10  -0.034416   0.034416    0   0   SMA_p-SMA_a     gridloc
11  0.244854    0.244854    0   0   SMA_p-SMA_a     shape_oriented
12  -0.073975   0.073975    0   0   dlPFC_p-dlPFC_a     gridloc
13  0.234116    0.234116    0   0   dlPFC_p-dlPFC_a     shape_oriented
14  -0.178012   0.178012    0   0   preSMA_p-preSMA_a   gridloc
15  0.158417    0.158417    0   0   preSMA_p-preSMA_a   shape_oriented
16  -0.094409   0.094409    0   0   vlPFC_p-vlPFC_a     gridloc
17  0.107170    0.107170    0   0   vlPFC_p-vlPFC_a     shape_oriented    
    """

    if func is None: 
        func = lambda x: np.nanmean(x)
    assert len(GROUPING_LEVELS)==2
    assert isinstance(GROUPING, str), "see docs for pivot_table"
    
    # 1) Aggregate and split by grouping
    dfpivot = pivot_table(df, index=INDEX, columns=[GROUPING], values=FEATURE_NAMES, 
                          aggfunc=func)

    # ===== Compute summary statistic (e.g., difference across groupings)
    out = {}
    COLNAMES_DICT = []
    COLNAMES_NOABS = []
    COLNAMES_ABS = []
    COLNAMES_DIFF = []

    # 2) all other features, take difference
    for val2 in FEATURE_NAMES:
        if val2=="alignment":
            # 1) alignemnt, take mean
            colname = f"{val2}-MEAN"
            colvals = np.nanmean(np.c_[dfpivot[val2][GROUPING_LEVELS[0]].values, 
                                                dfpivot[val2][GROUPING_LEVELS[1]].values], axis=1)
        else:
            colname = f"{val2}-{GROUPING_LEVELS[1]}min{GROUPING_LEVELS[0]}"
            colvals = dfpivot[val2][GROUPING_LEVELS[1]] - dfpivot[val2][GROUPING_LEVELS[0]]
            COLNAMES_DIFF.append(colname)

        out[colname] = colvals

        # get the abs balue
        if get_absolute_val:
            out[f"{colname}-ABS"] = np.abs(colvals)

        COLNAMES_NOABS.append(colname)
        COLNAMES_ABS.append(f"{colname}-ABS")

    # 3) retain character name
    for colthis in INDEX:
        out[colthis] = dfpivot[colthis]

    dfsummary = pd.DataFrame(out)
    dfsummary=dfsummary.dropna().reset_index(drop=True)

    # ==== Melt into long-form dataset
    dfsummaryflat = pd.melt(dfsummary, id_vars = INDEX)
    predicted_len = (len(dfsummary.columns)-len(INDEX)) * len(dfsummary)
    assert len(dfsummaryflat)==predicted_len

    # === Noramlize?
    if do_normalize:
        # Normalization, collect denominators (for percent change, signed)
        denom_level = GROUPING_LEVELS[0]
        normalization_denoms = {}
        for g in dfpivot.groupby(normalize_grouping):
            for var in FEATURE_NAMES:
                x = np.nanmean(g[1][var][denom_level])
                normalization_denoms[g[0] + tuple([var])] = x

        # extract "original" variabl ename
        def F(x):
            return x["variable"][:x["variable"].find("-")]
        dfsummaryflat = applyFunctionToAllRows(dfsummaryflat, F, "variable_origname")

        # compute normalization
        def F(x):
            idx = (x["animal"], x["expt"], x["variable_origname"])
            denom = normalization_denoms[idx]
            xnorm = x["value"]/np.abs(denom) # take abs to retain sign
            return xnorm
        dfsummaryflat = applyFunctionToAllRows(dfsummaryflat, F, "value_norm")

    
    if return_dfpivot:
        return dfsummary, dfsummaryflat, COLNAMES_NOABS, COLNAMES_ABS, COLNAMES_DIFF, dfpivot
    else:
        return dfsummary, dfsummaryflat, COLNAMES_NOABS, COLNAMES_ABS, COLNAMES_DIFF


def datamod_normalize_row_after_grouping_within(df, class_column, value_column, new_col_name=None):
    """
    The function calculates the mean of the values within each class and then subtracts these 
    means from the corresponding values, returning the modified dataframe. You can call this function 
    with your specific dataframe and column names.    
    
    In this function:

        df is the input dataframe.
        class_column is the name of the column containing the classes.
        value_column is the name of the column containing the values to be normalized.

    RETURNS:
    - modifies df, adding a new column  f"{value_column}_norm"

    # # Sample usage
    # data = {
    #     'classes': ['A', 'A', 'B', 'B', 'B', 'C', 'C'],
    #     'values': [10, 12, 20, 25, 30, 35, 40]
    # }
    # df = pd.DataFrame(data)

    # # Normalize values by class
    # normalized_df = normalize_values_by_class(df, 'classes', 'values')

    # # Display the resulting dataframe
    # print(normalized_df)
    """
        
    # Calculate mean values for each class
    class_means = df.groupby(class_column)[value_column].transform('mean')

    if new_col_name is None:
        new_col_name = f"{value_column}_norm"

    # Subtract mean values from the original values
    df[new_col_name] = df[value_column] - class_means


import pandas as pd

def datamod_normalize_row_after_grouping_return_same_len_df(df, var_contrast, grplist_index, y_var, 
        lev_default_contrast, PLOT=False, do_normalization=True, do_pvals=False):
    """
    Normalize data within groupings by the mean value across trials of a specific level of a categorical (contrast) variable.
    
    E.g., consider all trials for a character x block, with measurement "score". 
    You want to normalize the score for each level of "epoch" to the score across trials for a specific level of epoch (e.g., baseline). 
        var_contrast = "epoch"
        grplist_index = ["character", "block"]
        y_var = "score"
        lev_default_contrast = "baseline"
    
    NOTE: if a lev of grplist_index doesnt have var_contrast level lev_default_contrast, then all rows for it get Na
    PARAMS:
    - df, long-form dataframe.
    - grplist_index, list of str. Each mean will be taken within each conjunctive level.
    - PLOT, bool, summaries. Scatterplot of distribution vs. each level of var_contrast, along with sign-rank p values.
    - do_normalization, bool, if True, perform normalization.
    - do_pvals, bool, if True, perform p-value calculations.
    
    RETURNS:
    - A copied dataframe with an additional normalized column.

    # Sample usage
    data = {
        'character': ['A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B'],
        'block': [1, 1, 1, 2, 2, 2, 2, 2, 2],
        'epoch': ['baseline', 'trial1', 'baseline', 'trial1', 'trial2', 'baseline', 'trial1', 'trial1', 'trial1'],
        'score': [10, 12, 20, 25, 30, 35, 40, 50, 10]
    }
    df = pd.DataFrame(data)

    # Normalize values by specific class level
    normalized_df, stats, fig = datamod_normalize_row_after_grouping_return_same_len_df(df, 'epoch', ['character', 'block'], 'score', 'baseline', PLOT=True)

    # Display the resulting dataframe and plot
    print(normalized_df)
    if fig:
        plt.show()

    """

    df = df.copy()
        
    if lev_default_contrast is None:
        lev_default_contrast = df[var_contrast].unique().tolist()[0]

    # Calculate the mean value for the specific level within each group
    print()
    group_means = df[df[var_contrast] == lev_default_contrast].groupby(grplist_index)[y_var].mean().reset_index()
    group_means = group_means.rename(columns={y_var: f'{y_var}_mean'})

    # Merge the mean values back into the original dataframe
    df = pd.merge(df, group_means, on=grplist_index, how='left')

    if do_normalization:
        new_col_name = f"{y_var}_norm"
        # Subtract the specific class mean from the original values
        df[new_col_name] = df[y_var] - df[f'{y_var}_mean']

    if do_pvals or PLOT:
        from scipy.stats import wilcoxon

        list_lev = [x for x in df[var_contrast].unique().tolist() if not x == lev_default_contrast]
        pvals = []
        means = []
        for lev in list_lev:
            vals = df[df[var_contrast] == lev][new_col_name].dropna().values
            res = wilcoxon(vals)
            pvals.append(res.pvalue)
            means.append(vals.mean())
        stats = {
            "levels": list_lev,
            "pvals": pvals,
            "means": means
        }
    else:
        stats = {}

    if PLOT:
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        list_lev_plot = [lev_default_contrast] + list_lev
        pvals_plot = [1.] + pvals

        x = df[var_contrast]
        y = df[new_col_name]
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        ax.axhline(0, color="k", alpha=0.5)
        sns.stripplot(ax=ax, x=x, y=y, jitter=True, alpha=0.7, order=list_lev_plot)
        sns.pointplot(ax=ax, x=x, y=y, order=list_lev_plot, linestyles="", color="k")
        ax.set_ylabel(f"{y_var} minus {lev_default_contrast}")
        ax.set_title(f"each data = {grplist_index}")
    else:
        fig = None

    return df, stats, fig

def datamod_normalize_row_after_grouping(df, var_contrast, grplist_index, y_var,
    lev_default_contrast=None, PLOT=False, do_normalization=True,
    do_pvals=False):
    """ [GOOD] Normalize data within groupings by a the mean value across trials
    of specific level of a categorical (contrast) variable
    
    E.g., consider all trials for a character x block, with measurement "score". 
    You want to normalize the sccore each level of "epoch"
    to the score across trials for a specific level of epoch (e.g, baseline). 
        var_contrast = "epoch"
        grplist_index = ["character", "block"]
        y_var = "score"
        lev_default_contrast = "baseline"
    PARAMS:
    - df, long-form dataframe.
    - grplist_index, lsit of str. each mean will be taken within each conjunctive
    level
    - PLOT, bool, summaries. Scatterplot of distribution vs. each level of var_contrast, along wiht
    sign-rank p values.
    - datamod_normalize_row_after_grouping, bool, 
    RETURNS:
    - dfpivot, dfpivot_norm, dflong_norm, stats
    NOTE:
    - this is more general versin of summarize_featurediff, which only compares a pair of levels
    (e.g., treatement vs. baseline).
    NOTE: Thios returns n rows as same as n levels (i.e. not each original row, but after averaging)
    """
    from pythonlib.tools.statstools import ttest_paired, signrank_wilcoxon, plotmod_pvalues

    assert grplist_index is not None
    
    if lev_default_contrast is None:
        lev_default_contrast = df[var_contrast].unique().tolist()[0]

    ## Pivot 
    # - each row is a single level of conjucntion of vars in grplist_index
    # - separate columns for each level of var_contrast, holding its value (mean
    # across trials if needed) of y_var.
    dfpivot = pivot_table(df, index=grplist_index, columns=[var_contrast], values=[y_var])

    ## For each row, normalize by subtracting the rows value for the default level of var_contrast
    dfpivot_norm = dfpivot.copy()
    if do_normalization:
        dfpivot_norm[y_var] = dfpivot_norm[y_var] - dfpivot_norm[y_var][lev_default_contrast].values[:,None]
    
    ## Convert from wide- to long-form
    dflong_norm = pd.melt(dfpivot_norm, id_vars=grplist_index)

    ## Sign-rank: do stats for each level
    if do_pvals or PLOT:
        list_lev = [x for x in df[var_contrast].unique().tolist() if not x==lev_default_contrast]
        pvals = []
        means = []
        for lev in list_lev:
            vals = dfpivot_norm[y_var][lev].values
        #     res = ttest_paired(vals)
            # Remove nans
            vals = vals[~np.isnan(vals)]
            res = signrank_wilcoxon(vals)
            if np.isnan(res.pvalue):
                print(vals)
                print(np.isnan(vals))
                assert False
            pvals.append(res.pvalue)
            means.append(np.mean(vals))
        stats = {
            "levels":list_lev,
            "pvals":pvals,
            "means":means
        }
    else:
        stats = {}

    if PLOT:
        # fig = sns.catplot(data=dflong_norm, x=var_contrast, y="value", jitter=True, alpha=0.7)
        # plt.axhline(0, color="k")
        
        # fig, ax = plt.subplots()
        # x = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], len(pvals))
        # plotmod_pvalues(ax, x, pvals)
        # ax.set_xticks(x, labels=list_lev);    

        list_lev_plot = [lev_default_contrast] + list_lev
        pvals_plot = [1.] + pvals

        x = dflong_norm[var_contrast]
        y = dflong_norm["value"]
        fig, ax = plt.subplots(1,1,figsize=(8,6))

        # sns.scatterplot(ax=ax, x=x, y=y, x_jitter=True)
        # sns.swarmplot(ax=ax, x=x, y=y)
        ax.axhline(0, color="k", alpha=0.5)
        sns.stripplot(ax=ax, x=x, y=y, jitter=True, alpha=0.7, order=list_lev_plot)
        sns.pointplot(ax=ax, x=x, y=y, order=list_lev_plot, linestyles="", color="k")
        # sns.histplot(ax=ax, x=x, y=y, alpha=0.7, order=list_lev_plot)
        plotmod_pvalues(ax, range(len(list_lev_plot)), pvals_plot)
        ax.set_ylabel(f"{y_var} minus {lev_default_contrast}")
        ax.set_title(f"each dat = {grplist_index}")
    else:
        fig = None

    return dfpivot, dfpivot_norm, dflong_norm, stats, fig


def extract_trials_spanning_variable(df, varname, varlevels=None, n_examples=1,
                                    F = None, return_as_dict=False, 
                                    method_if_not_enough_examples="prune_subset"):
    """ To uniformly sample rows so that spans levels of a given variable (column)
    e..g, if a col is "shape" and you want to get one example random trial of each shape,
    then varname="shape" and varlevels = list of shape names, or None to get all.
    PARAMS:
    - df, dataframe
    - varname, column name
    - varlevels, list of string, levels. or None to get all.
    - n_examples, how many to get of each (random)
    - F, dict for filtering. will use this and append the varname:varlevel.
    - return_as_dict, then returns as out[level] = <list of indices>
    RETURNS:
    - list_inds, list of ints or None (for cases where can't get this many examples, 
    will fail all examples, not just the excess... (should fix this)). if n_examples >1, then
    adjacnet inds will be for the same example.
    - varlevels
    """
    import random
    if F is None:
        F = {}

    if False: # This didnt make sense to me...
        if method_if_not_enough_examples=="prune_subset":
            assert return_as_dict==True, "otherwise will lose traick of inidces."

    # Get the levels of this vars
    if varlevels is None:
        varlevels = df[varname].unique().tolist()
    
    # For each level, find n examples
    list_inds = []
    outdict = {}
    for val in varlevels:
        F[varname] = val
        list_idx = filterPandas(df, F, True)
        if len(list_idx)>=n_examples:
            inds = random.sample(list_idx, n_examples)[:n_examples]
        else:
            if method_if_not_enough_examples == "all_none":
                # option 1< return all as None
                inds = [None for _ in range(n_examples)]
            elif method_if_not_enough_examples=="prune_subset":
                # sample size changes... keep how many you have
                # n_examples = len(list_idx)
                # inds = random.sample(list_idx, n_examples)[:n_examples]
                inds = list_idx
            elif method_if_not_enough_examples== "fail":
                assert False, "not enough trials "
            else:
                assert False
        list_inds.extend(inds)
        outdict[val] = inds
    if return_as_dict:
        return outdict, varlevels
    else:
        return list_inds, varlevels


def grouping_get_inner_items(df, groupouter="task_stagecategory", 
    groupinner="index", groupouter_levels=None, nrand_each=None, 
    sort_keys=False,
    n_min_each_conj_outer_inner=1,
    take_top_n_inner=None,
    DEBUG=False):
    """ Return dict of unique items (levels of groupinner), grouped
    by groupouter levels. 
    PARAMS:
    - groupouter, string, the first grouping.
    - groupinner, string, the second grouping. either a column or "index"
    - groupouter_levels, list of values to use for groupouter. if None, then 
    finds and uses all.
    - nrand_each, int,then gets max this much per levle of groupouter. gets 
    random, sorted. if None then gets all.
    - n_min_each_conj_outer_inner, int, if any conj of outer and inner has fewre
    than this many cases, then excludes. e.g. if is (task, epoch), then ignores
    epochs where < this many cases of this task. using 1 means ignore.
    - take_top_n_inner, if not None, then is int, caps n inner for each outer by this, 
    by taking the n with max n trials
    RETURNS:
    - groupdict, where each key is a level of groupouter, and
    items are the unique values of groupinner that exist for that
    groupouter level.
    EXAMPLE:
    - if groupouter = date and groupinner = character, then returns
    {date1:<list of strings of unique characters for date1>, 
    date2:<list of strings ....}
    """

    _check_index_reseted(df)

    if groupouter_levels is None:
        groupouter_levels = df[groupouter].unique()

    if sort_keys:
        groupouter_levels = sort_mixed_type(groupouter_levels)
        
    groupdict = {}
    for lev in groupouter_levels:
        try:
            dfthisgroup = df[df[groupouter]==lev]
        except Exception as err:
            # probably lev is np array
            print(type(lev))
            print(lev)
            print(groupouter_levels)
            print(groupouter)
            print(df[groupouter].unique())
            raise err
        if groupinner=="index":
            itemsinner = dfthisgroup.index.tolist()
            assert n_min_each_conj_outer_inner==1, "doesnt work for index, each index is already uinque.."
        else:
            itemsinner = dfthisgroup[groupinner].unique().tolist()
            if n_min_each_conj_outer_inner>1:
                if DEBUG:
                    print("-----")
                    print("N, each inner, for outer:", lev, " ... ")
                    for it in itemsinner:
                        print(it, "-->", sum(dfthisgroup[groupinner]==it))
                itemsinner = [it for it in itemsinner if sum(dfthisgroup[groupinner]==it)>=n_min_each_conj_outer_inner]
                if DEBUG:
                    print("Keeping these items: ", itemsinner, "(ie those with n >= ", n_min_each_conj_outer_inner, ")")
            if take_top_n_inner is not None and len(itemsinner)>take_top_n_inner:
                # get n for each item
                if DEBUG:
                    print("------")
                    print("Starting items.. too many: ", itemsinner)
                itemsinner_n = [sum(dfthisgroup[groupinner]==it) for it in itemsinner]
                _inds = np.argsort([-n for n in itemsinner_n])[:take_top_n_inner]
                itemsinner = [itemsinner[i] for i in _inds]
                if DEBUG:
                    print("sample size: ", itemsinner_n)
                    print("idnices in order of incresaing n:", _inds)
                    print("new items:", itemsinner)

        if nrand_each is not None:
            if len(itemsinner)>nrand_each:
                import random
                itemsinner = random.sample(itemsinner, nrand_each)
        groupdict[lev] = itemsinner
    return groupdict

def grouping_append_and_return_inner_items_good(df, list_groupouter_grouping_vars,
                                                groupouter_levels=None, sort_keys=True):
    """ Quicker version of grouping_append_and_return_inner_items
    RETURNS:
    - groupdict, grptuple:list_indices_into_df
    """
    groupdict = {}
    assert groupouter_levels is None, "implementation fails below"
    try:
        for grp in df.groupby(list_groupouter_grouping_vars):
            if (groupouter_levels is not None) and (grp not in groupouter_levels):
                continue
            groupdict[grp[0]] = grp[1].index.tolist()
    except Exception as err:
        print(list_groupouter_grouping_vars)
        print(len(df))
        # print(grp)
        for g in list_groupouter_grouping_vars:
            print(df[g].unique())
        raise err

    if sort_keys:
        keys = list(groupdict.keys())
        keys = sort_mixed_type(keys)
        groupdict = {k:groupdict[k] for k in keys}

    return groupdict

def stringify_values(df):
    """
    Convert any values that are tuples or lists into strings, with
    separator | nbetween items.
    Useful for grouping, seaborn, and other plotting stuff, which can
    often throw error in these cases.
    :param df:
    :return: copy of df, with modifications descrtibed above,.
    """
    from pythonlib.tools.listtools import stringify_list
    df_str = df.copy()
    for k in df_str.columns:
        df_str[k] = [stringify_list(v, return_as_str=True, separator="|") if isinstance(v, (list, tuple)) else v for v in df_str[k].values.tolist()]
    return df_str

def grouping_append_and_return_inner_items(df, list_groupouter_grouping_vars, 
    groupinner="index", groupouter_levels=None, new_col_name="grp",
    return_df=False, sort_keys=True):
    """ Does in sequence (i) append_col_with_grp_index (ii) grouping_get_inner_items.
    Useful if e./g., you want to get all indices for each of the levels in a combo group,
    where the group is defined by conjunction of two columns.
    PARAMS:
    - list_groupouter_grouping_vars, list of strings, to define a new grouping variabe,
    will append this to df. this acts as the groupouter.
    - groupinner, see grouping_get_inner_items
    - groupouter_levels, see grouping_get_inner_items
    - sort_keys, sorts keys of groupdict before returning.
    RETURNS:
    - groupdict, see grouping_get_inner_items
    NOTE: does NOT modify df.
    """

    # If empty...
    if len(df)==0:
        if return_df:
            return {}, df
        else:
            return {}

    _check_index_reseted(df)
    assert not isinstance(groupinner, list)

    if groupinner=="index" and return_df==False:
        groupdict = grouping_append_and_return_inner_items_good(df, list_groupouter_grouping_vars,
                                                                groupouter_levels, sort_keys)
        return groupdict

    # 1) Append new grouping variable to each row
    # while new_col_name in df.columns:
    #     new_col_name+="_"
    df = append_col_with_grp_index(df, list_groupouter_grouping_vars, new_col_name, use_strings=False)

    # 2) Get dict of eaceh group
    groupdict = grouping_get_inner_items(df, new_col_name, groupinner, groupouter_levels=groupouter_levels)

    if sort_keys:
        keys = list(groupdict.keys())
        keys = sort_mixed_type(keys)
        groupdict = {k:groupdict[k] for k in keys}

    if return_df:
        return groupdict, df
    else:
        return groupdict

def grouping_count_n_samples(df, groupvars):
    """ REturn list of ints, the n across all conjucntiosn of groupvars"""
    return df.groupby(groupvars).size().tolist()

def grouping_count_n_samples_quick(df, list_groupouter_grouping_vars):
    """ Returns the min and max n across conjjunctiosn of list_groupouter_grouping_vars
    """
    dftmp = df.groupby(list_groupouter_grouping_vars).count()
    nmax = np.max(np.max(dftmp, axis=0))
    nmin = np.min(np.min(dftmp, axis=0))
    return nmin, nmax

def grouping_plot_n_samples_conjunction_heatmap(df, var1, var2, vars_others=None, FIGSIZE=7):
    """ Plot heatmap of num cases of 2 variables (conjucntions), each subplot conditioned
    on a third variable (value of conjcjtions of vars_others).
    NOTE: this is better than extract_with_levels_of_conjunction_vars because here
    can make separate subplots conditioned on a third variable. there is only one supblot
    of 2 vars.
    PARAMS:
    - var1, var2, string, columns in df, categorical varlibels, will be axes of heatmsp
    - vars_others, list of str, columns in df, each conj is a sbuplot
    rRETURNS:s
    - fig
    """

    df = df.copy()

    if isinstance(var1, (tuple, list)):
        df = append_col_with_grp_index(df, var1, "dummy_var", use_strings=False)
        var1 = "dummy_var"

    if vars_others is not None:
        assert isinstance(vars_others, (list, tuple))
        df = append_col_with_grp_index(df, vars_others, "dummy", use_strings=False)
    else:
        df = df.copy()
        df["dummy"] = 0
    list_dummy = sort_mixed_type(df["dummy"].unique().tolist())

    list_var1 = sort_mixed_type(df[var1].unique().tolist())
    list_var2 = sort_mixed_type(df[var2].unique().tolist())

    if len(list_dummy)<3:
        ncols = len(list_dummy)
    else:
        ncols = 3
    nrows = int(np.ceil(len(list_dummy)/ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*FIGSIZE, nrows*FIGSIZE), squeeze=False)
    for ax, dum in zip(axes.flatten(), list_dummy):
        dfthis = df[df["dummy"]==dum]

        # plot 2d
        convert_to_2d_dataframe(dfthis, var1, var2, plot_heatmap=True, ax=ax,
            list_cat_1 = list_var1, list_cat_2 = list_var2)

        ax.set_title(dum)
    return fig

# def plot_

def grouping_print_n_samples(df, list_groupouter_grouping_vars, Nmin=0, savepath=None,
        save_convert_keys_to_str = False, save_as="txt", sorted_by_keys=True):
    """ print the sample size for each conjunctive level (defined by grouping list: list_groupouter_grouping_vars)
    e.g., if goruping is [shape, location, size]: prints:
    ('Lcentered-3-0', (-1, -1), 'rig3_3x3_small') 58
    ('Lcentered-3-0', (-1, 1), 'rig3_3x3_small') 42
    ('Lcentered-3-0', (1, -1), 'rig3_3x3_small') 58
    ('Lcentered-3-0', (1, 1), 'rig3_3x3_small') 66
    ('Lcentered-4-0', (-1, -1), 'rig3_3x3_small') 52
    ('Lcentered-4-0', (-1, 1), 'rig3_3x3_small') 53
    ('Lcentered-4-0', (1, -1), 'rig3_3x3_small') 59
    ('Lcentered-4-0', (1, 1), 'rig3_3x3_small') 44
    ('V-2-0', (-1, -1), 'rig3_3x3_small') 73, 
    ... and so on.
    PARAMS;
    - list_groupouter_grouping_vars, list of str
    - savepath, include extension, saves as yaml
    - Nmin, int, skips any cases that have fewer than Nmin samples.
    - print_value_not_n, then etracts not n, but the actual values (list)
    RETURNS:
    - outdict, dict[level]=n
    """

    if len(df)==0:
        return {}

    outdict = {}
    # out = grouping_append_and_return_inner_items(df, list_groupouter_grouping_vars)
    out = grouping_append_and_return_inner_items_good(df, list_groupouter_grouping_vars)

    # out[grp] = list_indices.
    for k in sort_mixed_type(list(out.keys())):
        n = len(out[k])
        if n>Nmin:
            # if print_value_not_n:
            #     print(k)
            #     print(out)
            #     assert False
            #     outdict[k] = out[k]
            # else:
            outdict[k] = n

    if savepath is not None:
        if save_as=="dict":
            from .expttools import writeDictToYaml
            if save_convert_keys_to_str:
                outdict = {str(k):v for k, v in outdict.items()}
            writeDictToYaml(outdict, savepath)
        elif save_as in ["txt", "text"]:
            # Then each is a string line, write to text
            from .expttools import writeStringsToFile, writeDictToTxtFlattened
            header = "|".join(list_groupouter_grouping_vars)
            lines = writeDictToTxtFlattened(outdict, savepath, header=header, sorted_by_keys=sorted_by_keys)
            print(lines)
            # lines = [f"{str(k)} : {v}" for k, v in outdict.items()]
            # if sorted_by_keys:
            #     # for l in lines[:10]:
            #     #     print(l)
            #     lines = sort_mixed_type(lines)
            #     # print("----")
            #     # for l in lines[:10]:
            #     #     print(l)
            #     # assert False
            # lines = [header] + lines
            # print(lines)
            # writeStringsToFile(savepath, lines)
        else:
            print(save_as)
            assert False
        print("Saved to: ", savepath)

    if sorted_by_keys:
        list_keys = list(outdict.keys())
        # print(list_keys)
        list_keys = sort_mixed_type(list_keys)
        # print(list_keys)
        # adsfsf
        outdict = {k:outdict[k] for k in list_keys}
        # print(k)
        # asdsad

    for k, v in outdict.items():
        print(k, ':    ', v)
        
    return outdict



# def replace_values_with_this(df, column, value_template, replace_with):
#     """ MOdify df, replacing all values in <column> that match
#     <value_template> with <replace_with>
#     :param value_template, eitehr None (replace values that are None) or
#     item to match
#     """
#     if value_template is None:
#         inds = df[column].isna()
#         if sum(inds)>0:
#             if isinstance(replace_with, tuple):
#                 replace_with = [replace_with for i in range(sum(inds))]
#             df.loc[inds, column] = replace_with
#     else:
#         inds = df[column]==value_template
#         if sum(inds)>0:
#             if isinstance(replace_with, tuple):
#                 replace_with = [replace_with for i in range(sum(inds))]
#             print(inds)
#             print(replace_with)
#             df.loc[inds, column] = replace_with

def replace_values_with_this(df, column, value_template, replace_with):
    """ MOdify df, replacing all values in <column> that match
    <value_template> with <replace_with>
    :param value_template, eitehr None (replace values that are None) or
    flexible, even iterables
    :param replace_with: value to replace with, works even if this is tuple/list.
    item to match
    """

    if isinstance(replace_with, (list, tuple)) and value_template is None:
        # Slow, but needed.
        df[column] = df[column].apply(lambda x: replace_with if pd.isna(x) else x)
    elif isinstance(replace_with, (list, tuple)):
        df[column] = df[column].apply(lambda x: replace_with if x == value_template else x)
    elif value_template is None:
        # Faster
        df.loc[df[column].isna(), column] = replace_with
    else:
        df.loc[df[column] == value_template, column] = replace_with

def replaceNone(dfthis, column, replace_with):
    """ replace Nones in this column with... 
    modifies in place.
    Retains the original called <column>_orig
    """

    # 1) save orig
    dfthis[f"{column}_orig"] = dfthis[column]
    # 2) replace
    tmp = dfthis[column].tolist()
    for i, x in enumerate(tmp):
        if x is None:
            tmp[i] = replace_with
    dfthis[column] = tmp
    return dfthis


def slice_by_row_label(df, colname, rowvalues, reset_index=True,
                       assert_exactly_one_each=False, prune_to_values_that_exist_in_df=True):
    """ Return a sliced dataframe, where rows are sliced
    to match the list of labels (rowvalues) for a column
    PARAMS:
    - colname, str, the column in which to compare lables
    - rowvalues, list of items (labels) to comapre. the
    returned df will have rows exactly matching these values.
    - prune_to_value_exist_in_df, if true, then keeps only rowvalues that exist in df.
    otherwise: error if rowvalues contains value not in df
    NOTE: if a value occurs in multipel rows, it extracts all rows.
    - assert_exactly_one_each, if True, then each val in rowvalues
    matches exactly one and only one. Can be confident that the OUTPUT
    matches input rowvalues exactly.
    NOTES:
        - enforces that even item in rowvalues must exist in df[col] (if prune_to_value_exist_in_df==False)
        - returns in order of rowvalues.
        - assert_exactly_one_each==False, then gets all rows in
        df which match an item in rowbales.
    EG:
    df = pd.DataFrame({'A': [5,6,3,4, 5], 'B': [1,2,3,5, 6]})
    list_of_values = [3, 6, 5]
    df.set_index('A').loc[list_of_values].reset_index()
    NOTE: Tested that this is optimal speed... ie time is taken almost all by the line
        dfout = df.set_index(colname).loc[rowvalues] # sets index to values of colname,
    """

    # print(rowvalues)
    # print(len(rowvalues))
    # for x in rowvalues:
    #     print(x, type(x))
    # assert False
    from pythonlib.tools.exceptions import NotEnoughDataException
    assert isinstance(colname, str)
    assert isinstance(rowvalues, list)

    if assert_exactly_one_each:
        prune_to_values_that_exist_in_df = False

    if prune_to_values_that_exist_in_df:
        assert assert_exactly_one_each==False, "incompatible"
        print(" prune_to_values_that_exist_in_df")
        rowvalues = [v for v in rowvalues if v in df[colname].tolist()]

    # the dataframe must have each value in rowvalues.
    try:
        dfout = df.set_index(colname).loc[rowvalues] # sets index to values of colname,
        # and then pulls out rows that have indices in rowvalues, in order they appear in
        # rowvalues (first sort) and then by order they are in df (second sort).
    except KeyError as err:
        # Then a row value doesnt exist in df
        tmp = df.index.tolist()
        for v in rowvalues:
            if v not in tmp:
                print(v)
                print("this item in rowvalues does not exist in df[colname]. Could be (i) passed in too many values, and prune_to_value_exist_in_df==False")
        print("ERROR:", err)
        raise NotEnoughDataException

    if reset_index:
        dfout = dfout.reset_index()

    if assert_exactly_one_each:
        if not dfout[colname].tolist() == rowvalues:
            print("---")
            print(len(dfout))
            print(len(rowvalues))
            print("---")

            print("values in dfout that are not in rowvalues:")
            print(dfout[~dfout[colname].isin(rowvalues)][colname])

            print("values in rowvalues that are not in dfout:")
            print([v for v in rowvalues if v not in dfout[colname].tolist()])

            # figure out whihc one is incorrect
            print("Values that are misalined.")
            for i, val in enumerate(rowvalues):
                if dfout.iloc[i][colname] != val:
                    print(i, val)
                    print(dfout.iloc[i][colname])
            assert False
        # # Check that each row of the outcome has rowval mathing the input.
        # for i, val in enumerate(rowvalues):
        #     if not dfout.iloc[i][colname] == val:
        #         print(dfout.iloc[i][colname], type(dfout.iloc[i][colname]))
        #         print(val, type(val))
        #         assert False

    return dfout


def concat(list_df):
    """ Return concatenated df
    Adds a column ("idx_df_orig") which indicates for 
    each row which item in list_df it came from
    PARAMS:
    - list_df, list of df, with matching cols
    RETURNS:
    - df_all, concated (n rows increases), with index resetted
    """

    df_all = pd.concat(list_df).reset_index(drop=True)
    from pythonlib.tools.listtools import indices_into_original_list_after_concat

    indices = indices_into_original_list_after_concat(list_df)
    df_all["idx_df_orig"] = indices

    return df_all


def conjunction_vars_prune_to_balance_stupid():
    """ See conjunction_vars_prune_to_balance(), which is the good code,
    here is bad version, which just find all levels of one var
    that starts out having all levls of other var.
    """ 
    assert False, "clean up the code...."
    PRUNE_TO_BALANCE_VARIABLES = True
    PLOT = True
    if PRUNE_TO_BALANCE_VARIABLES:

        if False:
            # print all conjunctions
            from pythonlib.tools.pandastools import grouping_print_n_samples
            for lev, dftmp in dict_levels.items():
                print("-- LEVEL OF OTHERVAR: ", lev)
                grpdict = grouping_print_n_samples(dftmp, ["vars_others", var])
            #     print(lev, len(dftmp))

        from pythonlib.tools.pandastools import convert_to_2d_dataframe

        dfcounts, _, _, _ = convert_to_2d_dataframe(dfthis, "vars_others", var, PLOT);

        import numpy as np
        x = np.all(dfcounts>0, axis=1) # the othervars that have data across all levels
        list_othervars_good = x[x==True].index.tolist()
        print(list_othervars_good)

        # prune df to only keep these othervars
        dfthis = dfthis[dfthis["vars_others"].isin(list_othervars_good)]    
        
        if PLOT:
            _, _, _, _ = convert_to_2d_dataframe(dfthis, "vars_others", var, PLOT);

def drop_using_position_index(dfthis, index, axis):
    """ Returns a copy of dfthis, removing a single position index.
    This is useful when index is not numerical., SOmetimes runs into 
    bugs when the index values are tuples
    PARAMS:
    - dfthis
    - index, int position on either axis
    - axis, 0 or 1, row or col
    RETURNS:
    - dfthis copy
    NOTE: does NOT reset index, since this would destroy index type.
    """


    if axis==0:
        n = len(dfthis.index)
    elif axis==1:
        n = len(dfthis.columns)
    else:
        assert False
    inds = np.ones(n, dtype=bool)
    inds[index] = False

    if axis==0:
        return dfthis.iloc[inds, :]
    elif axis==1:
        return dfthis.iloc[:, inds]
    else:
        assert False

def conjunction_vars_prune_to_balance(dfthis, var1, var2, PLOT=False,
    prefer_to_drop_which=None, force_to_drop_which=None):
    """
    Given two variables, prune df that that each level of var1 has 
    the same levels of var2 (not sample size, but simpl,y whether has it).
    First converts df to 2d (var1 vs var2) then does this in a greedy way,
    by considering all row and cols, giving each a score (low means low N and
    low num levels of the other varibale), removing the row or col with lowests core,
    and iterating until good.
    NOTE: is possible that df is returned empty...
    PARAMS
    - var1, var2, strings, variables to consider
    - prefer_to_drop_which, int {1, 2, None}, if 1, then prefereably drops levels from
    var1. Does this by having stronger penalty on scores for var1. the value of the
    penalty is emporical
    RETURNS:
    - copy of dfthis, balanced.
    - dfcounts, the final 2d dataframe
    """
    import numpy as np
    from pythonlib.tools.pandastools import convert_to_2d_dataframe
    DIVISOR = 4
    nstart = len(dfthis)

    if force_to_drop_which is not None:
        # "force" takes precedence.
        prefer_to_drop_which = None
        assert force_to_drop_which in (1,2)

    def _evaluate_data_size(dfcounts):
        """ Helper for summarizing size of this 2d dataframe"""
        n_othervar = len(dfcounts)
        n_var = len(dfcounts.columns)
        n_dat = int(dfcounts.sum().sum())

        return n_dat, n_var, n_othervar

    def _score_each_level(which_axis, dfcounts, DIVISOR):
        """ returns pandas series for this axis, with score....
        (see docs above). Low is bad. means low N and low n levels of other var,
        and should preferably prune"""
        if which_axis==var1:
            AXIS = 0
        elif which_axis==var2:
            AXIS = 1

        # score for n data
        tmp1 = np.sum(dfcounts, axis=AXIS)
        tmp1 = tmp1/np.sum(tmp1)

        # score for n levels of other var.
        tmp2 = np.sum(dfcounts>0, axis=AXIS)
        tmp2 = tmp2/np.sum(tmp2)

        # return sum of scores.
        score = tmp1 + tmp2

        if force_to_drop_which==1:
            # Then reduce score by n totla counts, since this guarantees it iwll
            # all values in score wil be lower than the lowst value in other score
            #( will be negative).
            if which_axis==var1:
                score = score - np.sum(np.sum(dfcounts))
        elif force_to_drop_which==2:
            if which_axis==var2:
                score = score - np.sum(np.sum(dfcounts))
        else:
            if prefer_to_drop_which==1:
                if which_axis==var1:
                    score = score/DIVISOR
            elif prefer_to_drop_which==2:
                if which_axis==var2:
                    score = score/DIVISOR
            else:
                assert prefer_to_drop_which is None

        return score

    # Convert to 2d frame
    dfcounts, _, _, _ = convert_to_2d_dataframe(dfthis, var2, var1, PLOT);
    nlev_var1 = len(dfcounts.columns)
    nlev_var2 = len(dfcounts)

    if isinstance(prefer_to_drop_which, str):
        if prefer_to_drop_which==var1:
            prefer_to_drop_which = 1
        elif prefer_to_drop_which==var2:
            prefer_to_drop_which = 2
        else:
            print("prefer_to_drop_which", prefer_to_drop_which)
            assert False

    if prefer_to_drop_which==1:
        # (nlev_var2/nlev_var1) is to make the scale same across the two vars.
        # the multiplier is to further penalize this var. if this is 3, then means that
        # you only choose to drop the "var you dont prefer to drop" if it is 
        # at lesat 3 times worse than the var you prefer to drop.
        DIVISOR = 3*(nlev_var2/nlev_var1) # 
    elif prefer_to_drop_which==2:
        DIVISOR = 3*(nlev_var1/nlev_var2) # 
    else:
        assert prefer_to_drop_which is None
        DIVISOR = 1

    while np.all(dfcounts>0)==False: # Then there is still imbalance.
        scores_others = _score_each_level(var2, dfcounts, DIVISOR)
        scores_var =  _score_each_level(var1, dfcounts, DIVISOR)

        if min(scores_others)<=min(scores_var):
            # remove the level of var2 that has the lowest score
            indexnum_to_remove = scores_others.argmin()
            lev_to_remove = scores_others.index[indexnum_to_remove]
            axis_to_remove = var2
            AXIS = 0
        else:
            # remove the level of var1 that has the lowest score
            indexnum_to_remove = scores_var.argmin()
            lev_to_remove = scores_var.index[indexnum_to_remove]
            axis_to_remove = var1
            AXIS = 1

        if PLOT:
            print("---")    
            print(AXIS, lev_to_remove)
            display(dfcounts)
            if False:
                print("scores_var1:", scores_var.tolist())
                print("scores_var2:", scores_others.tolist())
            print(f"Dropping from {axis_to_remove}, level {lev_to_remove}, which is index {indexnum_to_remove}")

        # prune one col/row, and then repeat until good.
        # print("DUPLICATEED:", dfcounts.columns.duplicated())
        # print("DUPLICATEED:", dfcounts.index.duplicated())
        # DFTHIS_EVENT.index.duplicated()
        # print(dfcounts.columns)
        if False:
            # fails if index values are tuples
            dfcounts = dfcounts.drop(lev_to_remove, axis = AXIS)    
        else:
            dfcounts = drop_using_position_index(dfcounts, indexnum_to_remove, axis = AXIS)


    # plot heatmap
    if PLOT:
        from pythonlib.tools.snstools import heatmap
        heatmap(dfcounts)


    # Prune dfthis
    vars_keep = dfcounts.columns.tolist()
    othervars_keep = dfcounts.index.tolist()

    dfthisout = dfthis[
        (dfthis[var1].isin(vars_keep)) & 
        (dfthis[var2].isin(othervars_keep))
    ].copy().reset_index(drop=True)

    # evaluate
    if PLOT:
        print("evaluation (ndat, nvar, nothervar):", _evaluate_data_size(dfcounts))
        assert len(dfthisout)==_evaluate_data_size(dfcounts)[0]
        print("Starting len: ", nstart)
        print("Ending len: ", len(dfthisout))

    return dfthisout, dfcounts


def extract_with_levels_of_var(df, var, levels=None):
    """
    Split df into multiple smaller mutually exclusive dataframes,
    each with a particular level fo var
    PARAMS:
    - var, column in df
    - levels, list of values of var, or None to use all.
    - min_n_keep_lev, int, throws out levels that dont have at least this much data
    RETURNS:
    - dict_levs, dict[lev] = df_sub
    - levels, list of levels in dict.keys()
    """

    assert len(df)>0
    if levels is None:
        levels = sort_mixed_type(df[var].unique().tolist())

    assert len(levels)>0

    dict_levs = {}
    for lev in levels:
        dict_levs[lev] = df[df[var]==lev].copy().reset_index(drop=True)

    return dict_levs, levels

def assign_epochsets_group_by_matching_levels_of_var(df, var_outer_trials, var_inner,
        epochset_col_name, PRINT=False, n_min_each_conj_outer_inner = 1, n_max_epochs = 10):
    """
    For each row, assign it a value that reflects the folliwing:
    Find all trials that have the same level of <var_outer_trials>. 
    Get the list of unique levels of <var_inner> across these trials.
    All trials that have the same list are given the same id assigned to
    a new column called <epochset_col_name>.
    For example, see Dataset.epochset_extract_common_epoch_sets()
    PARAMS:
    - n_min_each_conj_outer_inner, int, only considers levels of <var_inner> that
    have at least this many cases for a given level of <var_outer_trials>
    NOTE: if n_min_each_conj_outer_inner>1, then you can get cass where epochsets are not accurate, since
    will not include trials when defining epochset, but those trials will be given the epochset id... soluton 
    would be to remove those trials...
    n_max_epochs, int, if > this many levels of var_inner for a given level of var_outer_trials,
    then doesnt include all >10 (how does this?)
    RETURNS:
    - appends new column to df called <epochset_col_name>.
    """

    # - For each char_seq, get its list of epochs that it is present in
    groupdict = grouping_get_inner_items(df, var_outer_trials, var_inner,  
        n_min_each_conj_outer_inner=n_min_each_conj_outer_inner, take_top_n_inner=n_max_epochs)
    
    # - make the epoch set hashable, and sorted
    groupdict = {charseq:tuple(sorted(epoch_set)) for charseq, epoch_set in groupdict.items()}

    if PRINT:
        list_epochsets_unique = sorted(set([x for x in list(groupdict.values())]))
        print("Unique classes of epochs spanned by individual tasks:")
        for e in list_epochsets_unique:
            print(e)

    # - For each trial, map to one of these sets )
    def F(x):
        epoch_set = groupdict[x[var_outer_trials]]
        return epoch_set
    from pythonlib.tools.pandastools import applyFunctionToAllRows
    df = applyFunctionToAllRows(df, F, epochset_col_name)
    print(f"Defined new column: {epochset_col_name}")

    if PRINT:
        print("... value_counts:")
        print(df[epochset_col_name].value_counts())

    # sanity checl
    tmp = grouping_append_and_return_inner_items(df, [var_outer_trials], epochset_col_name)
    for k, v in tmp.items():
        assert len(v)==1

    return df, list_epochsets_unique


def extract_with_levels_of_var_good(df, grp_vars, n_min_per_var):
    """
    Return df that keeps only levels of var which have at least n_min_per_var trials.
    :param df:
    :param var:
    :param levels_var:
    :param n_min_per_var:
    :return:
    """

    _check_index_reseted(df)
    assert isinstance(grp_vars, list)

    groupdict = grouping_append_and_return_inner_items(df, grp_vars)
    inds_keep =[]
    for lev, inds in groupdict.items():
        if len(inds)>=n_min_per_var:
            # keep
            inds_keep.extend(inds)

    inds_keep = sorted(inds_keep)
    assert len(inds_keep)==len(set(inds_keep))

    dfthis = df.iloc[inds_keep].reset_index(drop=True)

    return dfthis, inds_keep

def extract_with_levels_of_conjunction_vars_helper(df, var, vars_others, n_min_per_lev=1,
                                                   plot_counts_heatmap_savepath=None,
                                                    lenient_allow_data_if_has_n_levels=2):
    """
    Heloper, setting params I usualyl use for simple task of pruning df to get just tjhose levels
    opf vars_others which have at least 2 levels of var, with <n_min_per_level> trials at least, per
    level.
    :param df:
    :param var:
    :param vars_others:
    :return:
    """
    # lenient_allow_data_if_has_n_levels = 2
    dfout, dict_dfthis = extract_with_levels_of_conjunction_vars(df, var, vars_others, n_min_across_all_levs_var=n_min_per_lev,
                    PRINT=False, lenient_allow_data_if_has_n_levels=lenient_allow_data_if_has_n_levels, DEBUG=False,
                    prune_levels_with_low_n=True, balance_no_missed_conjunctions=False,
                    balance_prefer_to_drop_which=None, PRINT_AND_SAVE_TO=None,
                    ignore_values_called_ignore=False, plot_counts_heatmap_savepath=plot_counts_heatmap_savepath,
                                            balance_force_to_drop_which=None)
    return dfout, dict_dfthis


def extract_with_levels_of_conjunction_vars(df, var, vars_others, levels_var=None, n_min_across_all_levs_var=8,
                    PRINT=False, lenient_allow_data_if_has_n_levels=None, DEBUG=False,
                    prune_levels_with_low_n=True, balance_no_missed_conjunctions=False,
                    balance_prefer_to_drop_which=None, PRINT_AND_SAVE_TO=None,
                    ignore_values_called_ignore=False, plot_counts_heatmap_savepath=None,
                                            balance_force_to_drop_which=None):
    """ Helper to extract dataframe (i) appending a new column
    with ocnjucntions of desired vars, and (ii) keeping only 
    levels of this vars (vars_others) that has at least n trials for 
    each level of a var of interest (var).
    PARAMS:
    - var, str, the variabiel you wisht ot use this dataset to compute 
    moudlation for.
    - vars_others, list of str, variables that conjucntion will define a 
    new goruping var.
    - levels_var, either list of items, for each must have at least n_min.
    if None, then takes the unique levels across entire df
    - n_min_across_all_levs_var, min n trials desired for each level of var. will only keep
    conucntions of (vars_others) which have at least this many for each evel of
    var.
    - lenient_allow_data_if_has_n_levels, either None (ignore) or int, in which case
    is more leneint. keeps a given level of vars_others if it has >= this many
    levels of var which has >=n_min datapts. usually requires _all_ levels of vars_others
    to have >=n_min datapts.
    - prune_levels_with_low_n, bool, then removes levles with n data < n_min_across_all_levs_var.
    - balance_no_missed_conjunctions, bool, if True, then makes sure the resulting
    dfthis is "square" in that each level of var has at least some data for
    each level of vars_others. Does this in an interative fashion, removing levels
    from var or vars_others that have the least impact on data.
    - ignore_values_called_ignore, bool, if True, then any rows with values (for any
    var) with "ignore" string, will throw out. could be any caps, and coiuld be within
    tuple/list. e.g,, ("ignore", "good") would throw out becuaes at l;esat one item is
    ignore. unloimited depth
    - plot_counts_heatmap_savepath, str of None. if str, then plots heatmap of num counts
    of each conj of var vs. vars_others. and saved to this path.
    EG:
    - you wish to ask about shape moudlation for each combaiton of location and 
    size. then var = shape and vars_others = [location, size]
    RETURNS:
    - dataframe, with new column "vars_others", concated across all levels of varothers
    - dict, level_of_varothers:df_sub
    NOTE: will have column "_index" guarateed to match indices of input df.
    EXAMPLE:
        from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars

        dftest = {
            'a':[1,2,3,1,2,3,1,2,3,1 ,2, 3],
            'b':[1,1,1,2,2,2,3,3,3, 3, 3, 3],
            'c':['a','a','a','b','b','b','c','c','c', 'c', 'c', 'c']
        }
        import pandas as pd
        dfthis = pd.DataFrame(dftest)

        extract_with_levels_of_conjunction_vars(dfthis, var="a", vars_others=['b', 'c'], n_min=2)
    """

    if isinstance(var, (tuple, list)):
        df = append_col_with_grp_index(df, var, "_tmp")
        var = "_tmp"

    _check_index_reseted(df)

    if len(df)==0:
        return pd.DataFrame([]), {}

    if n_min_across_all_levs_var is None:
        # assume you mean to be fully leneint
        n_min_across_all_levs_var = 1
        # assert n_min_across_all_levs_var is not None, "not coded. just make this one"

    if DEBUG:
        PRINT = True

    # make a copy, becuase will have to append to this dataframe
    df = df.copy().reset_index(drop=True)
    # Store index
    df["_index"] = df.index

    if lenient_allow_data_if_has_n_levels is None:
        # levels_var must be extracted here from entire dataset 
        assert levels_var is None, "you cant both rerquest to get all levels and also tell me which levels to get."
        levels_var = df[var].unique().tolist()

    # throw out rows with "ignore?"
    if ignore_values_called_ignore:
        from pythonlib.tools.pandastools import filter_remove_rows_using_function_on_values

        def _check_if_ignore_value(value):
            """ Returns True if this value has "IGNORE" string,in any inner list-lkike. recursive any depths
            e.g all these fail:
                # # value = "IGNORE"
                # # value = ("IGNORE", )
                # value = (("IGNORE", ), ("IGNORE", ))
                # _check_if_ignore_value(value)
            """
            if isinstance(value, (list, tuple)):
                ignores = [_check_if_ignore_value(v) for v in value]
                return any(ignores)
            elif isinstance(value, str):
                return value.lower()=="ignore"
            else:
                # Is not ignore.
                return False            

        def _keep(value):
            # To invert ignore --> True if keep.
            return not _check_if_ignore_value(value)      

        # remove rows that are IGNORE
        df = filter_remove_rows_using_function_on_values(df, var, _keep, reset_index=True)
        # print(df)
        # assert False
        if vars_others is not None:
            for v in vars_others:
                if len(df)>0:
                    df = filter_remove_rows_using_function_on_values(df, v, _keep, reset_index=True)

        # if len(df)==0:
        #     dfout = pd.DataFrame([])
        #     dict_dfthis = {}
        #     return dfout, dict_dfthis

    if len(df)>0:
        # Want to use entier data for this site? or do separately for each level of a given
        # conjunction variable.
        if vars_others is None:
            # then place a dummy variable so that entire thing is one level
            vars_others = ["dummy_var"]
            # if "dummy_var" in df.columns:
            #     df = df.drop("dummy_var", axis=1)
            # assert "dummy_var" not in df.columns
            df.loc[:, "dummy_var"] = "dummy"
            REMOVE_DUMMY = True
        else:
            assert isinstance(vars_others, (list, tuple))
            REMOVE_DUMMY = False

        # 1) Append conjucntions
        var_conj_of_others = "vars_others"
        df = append_col_with_grp_index(df, vars_others, new_col_name=var_conj_of_others, use_strings=False)
        # 2_ get for each conjunction of other variables
        levels_others = df[var_conj_of_others].unique().tolist()
        levels_others = sort_mixed_type(levels_others)
        # except TypeError as err:
        #     pass
        #     # print("not sorting (TypeError): ", levels_others)

        # 3) check each sub datfarme
        list_dfthis = []
        dict_dfthis = {} # level:df
        
        for lev in levels_others:

            if PRINT:
                print("=========== getting this level (othervar):", lev)
            # get data
            dfthis = df[df[var_conj_of_others] == lev]

            good, levels_passing = check_data_has_all_levels(dfthis, var, levels_var, n_min_across_all_levs_var,
                                                             lenient_allow_data_if_has_n_levels=lenient_allow_data_if_has_n_levels,
                                                             PRINT=PRINT)
            if DEBUG:
                print(lev, ' -- ',  var, ' -- ',  levels_var, ' -- ', good)
            if good:
                # keep, first pruning to just the levels that passed
                # print(len(dfthis))
                # print(dfthis[var].value_counts())
                # print(levels_passing)
                if prune_levels_with_low_n:
                    dfthis = dfthis[dfthis[var].isin(levels_passing)]
                    if PRINT:
                        print(f"-- n for each level of {var}, for the othervar level {lev}")
                        for _lev in levels_passing:
                            print(_lev, "...", sum(dfthis[var]==_lev))
                list_dfthis.append(dfthis)
                dict_dfthis[lev] = dfthis.copy()

        if REMOVE_DUMMY:
            del df["dummy_var"]
    else:
        list_dfthis = []
        dict_dfthis = {}

    # merge
    if len(list_dfthis)==0:
        dfout = pd.DataFrame([])
    else:
        dfout = pd.concat(list_dfthis).reset_index(drop=True)

    if balance_no_missed_conjunctions:
        # 2) balance (var, othervars)
        # print("Starting len:", len(dfthis))
        if len(dfout)>0:
            dfout, _ = conjunction_vars_prune_to_balance(dfout, var, "vars_others",
                prefer_to_drop_which=balance_prefer_to_drop_which,
                force_to_drop_which=balance_force_to_drop_which,
                PLOT=DEBUG)
            # print("Ending len:", len(dfthis))

    if PRINT_AND_SAVE_TO is not None:
        # print and save text of sampel size fo alll concuutison
        from pythonlib.tools.expttools import writeStringsToFile
        if len(dfout)>0:
            levels_var_this = sort_mixed_type(dfout[var].unique().tolist())
            list_s = []
            for lev_other, dftmp in dict_dfthis.items():
                list_s.append(f"LEVEL OTHER: {lev_other}")
                for lev in levels_var_this:
                    n = sum(dftmp[var]==lev)
                    if n>0:
                        list_s.append(f"    {lev} : [{n}]")
                    else:
                        list_s.append(f"    {lev} : {n}")
        else:
            list_s = []
        writeStringsToFile(PRINT_AND_SAVE_TO, list_s)

    if len(dfout)>0 and plot_counts_heatmap_savepath:
        if False:
            fig = convert_to_2d_dataframe(dfout, var, "vars_others", plot_heatmap=True)[1]
        else:
            fig = grouping_plot_n_samples_conjunction_heatmap(dfout, var, "vars_others", 
                vars_others=None)
        savefig(fig, plot_counts_heatmap_savepath)

    # Finalyl, extract each sub df
    dict_dfthis = {} # level:df
    if len(dfout)>0:
        for lev in levels_others:
            dfthis = dfout[dfout["vars_others"] == lev]
            if len(dfthis)>0:
                dict_dfthis[lev] = dfthis.copy()

    return dfout, dict_dfthis

# only keep if this has all the levels
def check_data_has_all_levels(dfthis, var, levels_to_check=None, 
    n_trials_min=5, lenient_allow_data_if_has_n_levels=None,
    PRINT=False):
    """ Retuirns True if this dataframe as at least n_trials_min for each
    level of var
    PARAMS:
    - var, str, column in dfthis
    - levels_to_check, either list of categorival levles of var, or None (to get all of them)
    - n_trials_min, int
    - lenient_allow_data_if_has_n_levels, None or int. if None, then must have at least n_trials_min for 
    each level in levels_to_check. if int, then only need to have at least n_trials_min for 
    lenient_allow_data_if_has_n_levels many levels.
    RETURNS:
    - passing, bool
    - levels_passing, list of levels that pass, only applies if lenient_allow_data_if_has_n_levels is not None,
    otherwise returns None.
    """
        
    assert len(dfthis)>0
    # print(levels_to_check)

    if lenient_allow_data_if_has_n_levels is None:
        assert levels_to_check is not None, "this code doesnst know what all the levels are... so you must pass it in"
    
    if levels_to_check is None:
        levels_to_check = dfthis[var].unique().tolist()
    # else:
    #     # check that each level exists in dfthis
    #     for lev in levels_to_check:
    #         assert lev in dfthis[var].tolist(), "you passed in levels_to_check that doesnt exist"
    assert isinstance(levels_to_check, list)

    if lenient_allow_data_if_has_n_levels is None:
        STRICT = True
    else:
        STRICT = False

    list_n = []
    for lev in levels_to_check:
        dfthisthis = dfthis[dfthis[var]==lev]
        n = len(dfthisthis)
        list_n.append(n)
        # assert n>0
        if PRINT:
            print(lev, ' -- ', n)

    if STRICT:
        if any([n<n_trials_min for n in list_n]):
            if PRINT:
                print(f"Skipping {var} becuase using strict mode, and at least one level is note nough data...")
                print("n across levels: ", list_n)
                print("min allowed n:", n_trials_min)
        # if n<n_trials_min:
            return False, None
        else:
            # Then you have suceeded iof havent failed aboev.
            return True, levels_to_check
    else:
        # Not strict, make sure at least some levels pass n_min.
        passes = [nthis >= n_trials_min for nthis in list_n] #  whether enogh trials for each level of var: [True, False, ...]
        levels_passing = [lev for n, lev in zip(list_n, levels_to_check) if n>=n_trials_min]
        npass = sum(passes)
        good = npass >= lenient_allow_data_if_has_n_levels

        if PRINT:
            print("----")
            print(f"{good}, because {npass}/{lenient_allow_data_if_has_n_levels} passed... ({list_n}) >= {n_trials_min}")

        if npass >= lenient_allow_data_if_has_n_levels:
            return True, levels_passing
        else:
            return False, levels_passing

def sort_by_two_columns_separate_keys(df, col1, col2, key2=None):
    """
    Sort hierarhccyalyl by two colummn, and allow passing in key for the second column.
    REturns copy
    for each
    :param df:
    :param col1:
    :param col2:
    :param key2:
    :return:
    """

    # NOT using this, since it fails the groupby step (as there it then groups again by the variable before keying)
    # if key1 is not None:
    #     def _key1(series1):
    #         # print(series1)
    #         # print([key1(x) for x in series1])
    #         # assert False
    #         return [key1(x) for x in series1]
    # else:
    #     _key1 = None

    if key2 is not None:
        def _key2(series2):
            return [key2(x) for x in series2]
    else:
        _key2 = None

    def custom_score_sort(x):
        return x.sort_values(by=col2, key=_key2)

    # Sort the DataFrame by 'Age' column and apply custom sorting on 'Score' column'
    df = df.sort_values(col1, key=None).groupby(col1, group_keys=False).apply(custom_score_sort).reset_index(drop=True)

    return df


def sort_rows_by_trialcode(dfthis):
    """ return copy of dfthis, rows sorted so that trialcode is increaseing downwards.
    This helps by converting trialcode string to tuple.
    Appends column "trialcode_tuple" as intermediate step.
    """
    from pythonlib.tools.stringtools import decompose_string

    def tc_to_tupleints(tc):
        """ 
        tc "2022-1-10" --> tupleints (2022, 1, 10)
        """
        this = decompose_string(tc, "-")
        return [int(x) for x in this]


    # 1) apopoend column 
    def F(x):
        return tc_to_tupleints(x["trialcode"])
    dfthis = applyFunctionToAllRows(dfthis, F, "trialcode_tuple")
    
    # 2) sort
    return dfthis.sort_values(by="trialcode_tuple").reset_index(drop=True)

# shuffle_dataset_singlevar_hierarchical
def shuffle_dataset_hierarchical(df, list_var_shuff, list_var_noshuff,
                          maintain_block_temporal_structure=False, shift_level="datapt",
                          return_in_input_order=True):
    """ Hierarhical, this means shuffle only the variables in list_var_shuff, within
    each level of list_var_noshuff. Moreover, all var in lisT_var_shuff will
    remain correlated. e.g., if want to shuffle all cases of (char, seq) within
    each epoch, have list_var_noshuff = ["epoch"], and list_var_shuff = ["char", "seq"],
    which does dfs for each level of "epoch" one by one, and within each one, shuffle
    the values of ["char", "seq"] (without changing values of other variables), and then
    concatenate.
    PARAMS:
    - return_in_input_order, bool, if True, then the rows match the inpout rows. in this case,
    will not have "index" column
    RETURNS:
        - df, with new column "index" holding the original indices.
    NOTE: confirmed doesnt mod input. and works correctly, including "index" correctly mapping
    to original data.
    """

    # Stringify, or else will fail groupby step
    df = stringify_values(df)

    # For each group of var, shuffle the othervar
    list_grp = []
    for i, grp in df.groupby(list_var_noshuff):
        if len(list_var_shuff)==1:
            grp = shuffle_dataset_singlevar(grp, list_var_shuff[0], maintain_block_temporal_structure=maintain_block_temporal_structure,
                                   shift_level=shift_level)
        else:
            grp = shuffle_dataset_varconj(grp, list_var_shuff, maintain_block_temporal_structure=maintain_block_temporal_structure,
                                           shift_level=shift_level)
        list_grp.append(grp)
    if return_in_input_order:
        df_shuff = pd.concat(list_grp).sort_index()
    else:
        df_shuff = pd.concat(list_grp).reset_index()

    # from pythonlib.tools.checktools import check_objects_identical
    # assert check_objects_identical(df_shuff.loc[:, list_grp].tolist(),
    #                     df.loc[:, list_grp].tolist())
    if True:
        # Sanityh check that no change in the noshuff variablves
        assert np.all(df_shuff.loc[:, list_var_noshuff] == df.loc[:, list_var_noshuff])

    return df_shuff

    # """
    # HIerarhical shuffle, which means shuffle within each level of the categorical
    # variable <var_grp_hier>. E.g., var_grp_hier is "character", then shuffle
    # onle within each char
    # PARAMS:
    # - var_shuff, str, the variable to shuffle.
    # - list_grp_hier, list of str, grouping variable conjunction.
    # """
    #
    # # Only shuffle within groups
    # list_df = []
    # for dfthis in df.groupby(list_grp_hier):
    #     # get shuffled version (copy)
    #     if DEBUG:
    #         print("-------------")
    #         print(dfthis[1].loc[:, list_grp_hier+[var_shuff]])
    #     _dfshuff = shuffle_dataset_singlevar(dfthis[1].copy(), var_shuff,
    #         maintain_block_temporal_structure, shift_level, DEBUG)
    #     if DEBUG:
    #         print(_dfshuff.loc[:, list_grp_hier+[var_shuff]])
    #     # assert False
    #     list_df.append(_dfshuff)
    #
    # # Concat
    # return pd.concat(list_df).reset_index(drop=True)

def shuffle_dataset_varconj(df, list_var, maintain_block_temporal_structure=True,
        shift_level="datapt", DEBUG=False, PRINT=False):
    """ Like _shuffle_dataset, but allowing you maintain the correlation between
    multiple varialbes. e..g, if you are doing two-way anova, want to make sure the sample
    size of conjunction var1xvar2 does not change.
    Here, does it by making dummy variable that
    is conjunciton, shuffle using that varaibale, then pull out new var1 and var2 from
    dummy
    PARAMS:
    - list_var, list of str,
    RETURNS:
    - copy of df, with labels for each var in list_var shuffed
    """
    # make a new temp var
    from pythonlib.tools.pandastools import append_col_with_grp_index, applyFunctionToAllRows, grouping_print_n_samples

    # assert len(list_var)==2, "not yet coded for >2"

    # 1) Make a dummy conjunction variable.
    dfthis = append_col_with_grp_index(df, list_var, "dummy", use_strings=False)

    # 2) Shuffle
    dfthis = shuffle_dataset_singlevar(dfthis, "dummy", maintain_block_temporal_structure, shift_level, DEBUG)

    # 3) Pull out the var1 and var2
    # resassign vars
    for i, varname in enumerate(list_var):
        def F(x):
            return x["dummy"][i]
        dfthis = applyFunctionToAllRows(dfthis, F, varname)

    if PRINT:
        print("=== Original, first 5 inds. SHOULD NOT CHANGE")
        print(df[list_var[0]][:5])
        print(df[list_var[1]][:5])

        print("=== Shuffled, first 5 inds. SHOULD CHANGE")
        print(dfthis[list_var[0]][:5])
        print(dfthis[list_var[1]][:5])

        print("=== Orig/Shuffled, n for each conj. SHOULD NOT CHANGE")
        print("-orig")
        grouping_print_n_samples(df, list_var)
        print("-shuffled")
        grouping_print_n_samples(dfthis, list_var)

    if "dummy" in dfthis.columns:
        del dfthis["dummy"]

    return dfthis

def shuffle_dataset_singlevar(df, var, maintain_block_temporal_structure=True,
        shift_level="datapt", DEBUG=False):
    """ returns a copy of df, with var labels shuffled
    NOTE: confirmed that does not affect df
    PARAMS:
    - maintain_block_temporal_structure, bool, if True, then shuffles by
    circular shifting of trials. THis is better if you didn't randopmly interleave 
    trials. It is a better estimate of variance of shuffles. 
    """
    import random
    from pythonlib.tools.stringtools import decompose_string
    from pythonlib.tools.listtools import extract_novel_unique_items_in_order
    from pythonlib.tools.listtools import list_roll

    levels_orig = df[var].tolist()
    # shuffle a copy
    levels_orig_shuff = [lev for lev in levels_orig]

    # maintain_block_temporal_structure=False
    if maintain_block_temporal_structure:
        # make sure dataframe is in order of trials.

        def tc_to_tupleints(tc):
            """ 
            tc "2022-1-10" --> tupleints (2022, 1, 10)
            """
            this = decompose_string(tc, "-")
            return [int(x) for x in this]

        trialcodes = df["trialcode"].tolist()
        trialcodes_sorted = sorted(trialcodes, key=lambda x: tc_to_tupleints(x))
        if not trialcodes==trialcodes_sorted:
            print("Trialcodes, -- , trialcodes_sorted")
            for t1, t2 in zip(trialcodes, trialcodes_sorted):
                print(t1, t2)
            assert False, "ok, you need to code this. sort dataframe"
        # return the levels in this order
        # print(trialcodes)
        # print(sorted(trialcodes))
        # this = [(x, y) for x, y in zip(levels_orig, trialcodes)]

        if shift_level=="trial":
            # then shift to not break within-trial correlations
            possible_shifts = extract_novel_unique_items_in_order(trialcodes)[1]
            shift = random.sample(possible_shifts, 1)[0]
        elif shift_level=="datapt":
            # shift at any datpt.
            shift = random.randint(0, len(levels_orig)-1) # 0, 1, 2, ... n-1, possible shifts
        else:
            print(shift_level)
            assert False

        # Do shuffle
        # Dont use this. it converts list of tuples to list of list.
        # levels_orig_shuff = np.roll(levels_orig_shuff, -shift).tolist() # negative, so that works for trial.
        levels_orig_shuff = list_roll(levels_orig_shuff, -shift)

        if DEBUG:
            print("trialcodes, levels(orig):")
            for t1, t2 in zip(trialcodes, levels_orig):
                print(t1, t2)   
            # print(possible_shifts)
            # print(levels_orig)
            print(levels_orig_shuff)
            print(shift)
            assert False
    else:
        # independently shuffle eachj location
        random.shuffle(levels_orig_shuff)

    dfthis_shuff = df.copy(deep=False)
    dfthis_shuff[var] = levels_orig_shuff

    if False:
        # dont need this sanity check
        if type(levels_orig_shuff[0])!=type(levels_orig[0]):
            print("orig: ", levels_orig)
            print("shuffed:", levels_orig_shuff)
            print(type(levels_orig_shuff[0]))
            print(type(levels_orig[0]))
            assert False
    
    return dfthis_shuff

def plot_subplots_heatmap(df, varrow, varcol, val_name, var_subplot,
                          diverge=False, share_zlim=False, norm_method=None,
                          annotate_heatmap=False, return_dfs=False,
                          ZLIMS=None, title_size=6, ncols=3,
                        row_values= None, col_values=None, W=5):
    """
    Plot heatmaps, one for each level of var_subplot, with each having columsn and rows
    given by those vars. Does aggregation to generate one scalar perc
    cell, if needed.
    :param df:
    :param varcol:
    :param varrow:
    :param var_subplot:
    :param ZLIMS: overwrite the zmin and max independently, 2-tuple, an item use None to 
    use the auto value. e.g,., (0., None) means fiz min to 0, but use auto for max.
    :return:
    """

    assert isinstance(var_subplot, str)

    # list_row = df[varrow].unique().tolist()
    list_subplot = df[var_subplot].unique().tolist()
    if ncols is None:
        # Then ncols is the num suibplots
        ncols = max([len(list_subplot), 2]) # for axes sake

    # ncols = 3
    # W = 5
    H = (4/5) * W
    nrows = int(np.ceil(len(list_subplot) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*W, nrows*H))

    if share_zlim:
        # Compute the min and max
        grpdict = grouping_append_and_return_inner_items(df, [varrow, varcol, var_subplot])
        mins = []
        maxs = []
        for grp, inds in grpdict.items():
            mins.append(np.min(df.iloc[inds][val_name]))
            maxs.append(np.max(df.iloc[inds][val_name]))
        zmin = min(mins)
        zmax = max(maxs)
        # print([varrow, varcol, var_subplot])
        # df_agg = aggregGeneral(df.loc[:, [varrow, varcol, var_subplot, val_name]], [varrow, varcol, var_subplot], [val_name])
        # zmin = np.min(df_agg[val_name])
        # zmax = np.max(df_agg[val_name])
        zlims = [zmin, zmax]
    else:
        zlims = [None, None]

    if ZLIMS is not None:
        # each item in ZLIM replaces independently
        if ZLIMS[0] is not None:
            zlims[0] = ZLIMS[0]
        if ZLIMS[1] is not None:
            zlims[1] = ZLIMS[1]
        # zlims = ZLIMS
    zlims = tuple(zlims)
    
    DictSubplotsDf ={}
    for lev_subplot, ax in zip(list_subplot, axes.flatten()):
        a = df[var_subplot]==lev_subplot
        dfthis = df[(a)]
        df2d, _, _, rgba_values = convert_to_2d_dataframe(dfthis, varrow, varcol,
                                True,
                                "mean",
                                val_name,
                                ax=ax, annotate_heatmap=annotate_heatmap,
                                diverge=diverge, zlims=zlims, norm_method=norm_method,
            list_cat_1 = row_values, list_cat_2=col_values
        )
        ax.set_title(lev_subplot, color="r", fontsize=title_size)
        DictSubplotsDf[lev_subplot] = df2d

    if return_dfs:
        return fig, axes, DictSubplotsDf
    else:
        return fig, axes


def plot_pointplot_errorbars(df, xvar, yvar, ax, hue=None, yvar_err=None):
    """ Given precomputed errorbars, plot them on a line or bar plot.
    This beucase seaborn doesnt support pre-computed errbars.
    """

    if hue is None:
        df = df.copy()
        df["_dummy"] = 0
        hue = "_dummy"

    list_hue = sort_mixed_type(df[hue].unique().tolist())
    for hue_lev in list_hue:
        dfthis = df[df[hue]==hue_lev]
        x=dfthis[xvar]
        y=dfthis[yvar]
        if yvar_err is not None:
            yerr = dfthis[yvar_err]
        else:
            yerr = None
        lab = f"{hue_lev}"

        ax.errorbar(x=x, y=y, yerr=yerr, label=lab)
        # ax.bar(x, y, yerr=yerr, label=lab, alpha=0.4)
        # sns.barplot(data=dfthisthis, x="bregion", y="same_mean", yerr=dfthisthis["same_sem"])
    return list_hue

def plot_45scatter_means_flexible_grouping(dfthis, var_manip, x_lev_manip, y_lev_manip,
                                           var_subplot, var_value, var_datapt,
                                           plot_text=True,
                                           alpha=0.8, SIZE=3, shareaxes=False,
                                           plot_error_bars=True,
                                           map_subplot_var_to_new_subplot_var=None,
                                           ):
    """ Multiple supblots, each plotting
    45 deg scatter, comparing means for one lev
    vs. other lev
    PARAMS:
    - var_manip, str, name of col whose values will define x and y values.
    - x_lev_manip, str, categorical level of var_manip, each value defines the x-values
    that will be averaged for each datapt.
    - var_subplot, str, categorical col, levels dedefine subplots. if List, then groups
    first so that each subploit is conjucntion of these vars
    - var_value, str, column, whose values/sdcore to plot.
    - var_datapt, str, categorical col, each lev is separate datapt. (note: will assert that max 1 datapt per).
    - ignore_if_sum_values, this useful to exclude any
    - map_subplot_var_to_new_subplot_var, dict that helps you remap the subplot variable to a new set of variables.
    e.g,, keys that dont exist will be grouped in "LEFTOVER". e.g., if var_subplot is "bregion" and you want to have
    three subplots, one preSMA_a, one preSMA_p, and one with all the others, then do this (along with var_subplot="bregion")
        map_subplot_var_to_new_subplot_var = {
        "preSMA_a":"preSMA_a",
        "preSMA_p":"preSMA_p",
            }

    EXAMPLE:
    - To compare score during stim 9stim_epoch) vs. during off, one datapt per character...
    separate subplot for each epoch_orig...
    dfres, fig = plot_45scatter_means_flexible_grouping(dfthis, "microstim_epoch_code",
                                                "off", y_lev_manip=stim_epoch,
                                                var_value="strokes_clust_score", var_subplot="epoch_orig,
                                                var_datapt="character",
                                   plot_text=plot_label, alpha=0.2, SIZE=4)

    """
    from scipy.stats import sem
    from pythonlib.tools.plottools import plotScatter45

    assert not x_lev_manip==y_lev_manip

    list_manip = [x_lev_manip, y_lev_manip]
    if (x_lev_manip not in dfthis[var_manip].tolist()) or (y_lev_manip not in dfthis[var_manip].tolist()):
        return None, None
    # assert x_lev_manip in dfthis[var_manip].tolist()
    # assert y_lev_manip in dfthis[var_manip].tolist()

    nmin = 1
    if isinstance(var_subplot, (tuple, list)):
        assert map_subplot_var_to_new_subplot_var is None, "the subplot var name will change and thus fail."
        dfthis = append_col_with_grp_index(dfthis, var_subplot, "_var_subplot", strings_compact=True)
        var_subplot = "_var_subplot"
    elif var_subplot is None:
        # dummy
        dfthis = dfthis.copy()
        dfthis["dummy45"] = "dummy"
        var_subplot = "dummy45"

    if map_subplot_var_to_new_subplot_var is not None:
        def F(x):
            if x[var_subplot] in map_subplot_var_to_new_subplot_var:
                return map_subplot_var_to_new_subplot_var[x[var_subplot]]
            else:
                return "LEFTOVER"
        dfthis = applyFunctionToAllRows(dfthis, F, "_var_subplot")
        var_subplot = "_var_subplot"

    list_lev = dfthis[var_subplot].unique().tolist()
    list_datapt = dfthis[var_datapt].unique().tolist()

    # dict_res = {}
    dict_res_2 = []
    for date in list_datapt:
        for lev in list_lev:
            # collect all data for each value of stim
            for manip in list_manip:
                dfthisthis = dfthis[(dfthis[var_datapt]==date) & (dfthis[var_subplot]==lev) & (dfthis[var_manip]==manip)]

                if len(dfthisthis)>=nmin:

                    # get stats
                    stats_mean = np.mean(dfthisthis[var_value])
                    stats_sem = sem(dfthisthis[var_value])

                    # dict_res[(date, lev, manip)] = (stats_mean, stats_sem)
                    dict_res_2.append({
                        var_value:stats_mean,
                        f"{var_value}_sem":stats_sem,
                        var_datapt:date,
                        var_subplot:lev,
                        var_manip:manip,
                    })
    dfres = pd.DataFrame(dict_res_2)

    # Make plots
    ncols = 3
    nrows = int(np.ceil(len(list_lev)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE))
 # sharex=sharex, sharey=sharey)

    list_xs = []
    list_ys = []
    for ax, lev in zip(axes.flatten(), list_lev):

        # Collect data
        dftmp_x = dfres[(dfres[var_subplot]==lev) & (dfres[var_manip]==x_lev_manip)]
        dftmp_y = dfres[(dfres[var_subplot]==lev) & (dfres[var_manip]==y_lev_manip)]

        # get intersection of datapts.
        list_date = sorted([d for d in dftmp_x[var_datapt].tolist() if d in dftmp_y[var_datapt].tolist()])
        dftmp_x = slice_by_row_label(dftmp_x, var_datapt, list_date, assert_exactly_one_each=True)
        dftmp_y = slice_by_row_label(dftmp_y, var_datapt, list_date, assert_exactly_one_each=True)

        # Plot
        xs = dftmp_x[var_value]
        ys = dftmp_y[var_value]
        if plot_error_bars:
            x_errors = dftmp_x[f"{var_value}_sem"]
            y_errors = dftmp_y[f"{var_value}_sem"]
        else:
            x_errors, y_errors = None, None
        if plot_text:
            labels = list_date
        else:
            labels = None
        try:
            plotScatter45(xs, ys, ax, labels=labels, marker="o",
                          x_errors=x_errors, y_errors=y_errors, alpha=alpha)
        except Exception as err:
            raise err
        ax.set_title(lev, fontsize=8)
        ax.set_xlabel(x_lev_manip)
        ax.set_ylabel(y_lev_manip)

        list_xs.extend(xs)
        list_ys.extend(ys)

        ax.axhline(0, color="k", alpha=0.25)
        ax.axvline(0, color="k", alpha=0.25)

    if shareaxes and len(list_xs)>0:
        MIN = min(list_xs + list_ys)
        MAX = max(list_xs + list_ys)
        # print(MIN, MAX)
        # assert False
        delt = 0.1*(MAX-MIN)

        # if False:
        #     MIN = 0
        # else:
        MIN = MIN - delt
        MAX = MAX + delt

        if MIN>0:
            # force min to be 0
            MIN = 0

        for ax in axes.flatten():
            ax.set_xlim(MIN, MAX)
            ax.set_ylim(MIN, MAX)

    return dfres, fig


def find_unique_values_with_indices(df, col, tolerance = 1e-3,
        append_column_with_unique_values_colname=None):
    """Return each unique value of col (numerical) as key to dict,
    with values being the indicies into df whose values of col 
    are np.isclose() to this value. 

    NOTE: is similar to append_col_with_index_of_level_after_grouping(), except
    here works with continuous values.

    :param df: _description_
    :param col: _description_
    :param append_column_with_unique_values_colname: either None (ingnore)
    or str, adds this column using the unique values.
    """

    # series = df[col]
    unique_values = []
    indices = []

    # # first, sort df by col
    # df = df.copy()
    # df = df.sort_values(col).reset_index(drop=True)

    map_index_to_value = {}
    for i, value in enumerate(df[col].values):
        found = False
        for j, unique_value in enumerate(unique_values):
            if np.isclose(value, unique_value, atol=tolerance):
                indices[j].append(i)
                found = True
                map_index_to_value[i] = unique_value
                break
        if not found:
            unique_values.append(value)
            indices.append([i])
            map_index_to_value[i] = value

    if append_column_with_unique_values_colname is not None:
        values = [map_index_to_value[i] for i in range(len(df))]
        df[append_column_with_unique_values_colname] = values
    
    return unique_values, indices, map_index_to_value

    # # Find unique values and their indices for each column
    # unique_values, indices = find_unique_with_indices(df[col], tolerance)
    # unique_indices[col] = list(zip(unique_values, indices))

    # print("Unique values and their indices:")
    # print(unique_indices)
