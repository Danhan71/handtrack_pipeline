import numpy as np

def preprocess_dates(D, groupby, animal, expt, nmin = 5, return_filtered_dataset=False,
    SAVEDIR=None):
    """ Figure out good dates for summary analyses based on flexible criteria
    - groupby, variable for which you desire there to exist trials for all levels in all date
    epochs. e..g, if groupby is "character" then extracts dates that have at least one trial for each char.
    NOTE: throws out levels that are not present across all datse.
    INPUT:
    - groupby, col name
    - nmin = 5, at lesat this many trials to use a day.
    - return_filtered_dataset, then returns dataset pruned to only have the levels of groupby that are kept.
    - SAVEDIR, if provided, then will save date information as yaml
    NOTE:
    - default criteria is to find dates that have at lesat nmin trials for all levels of groupby. finds the earlierst
    and latest within the window hand coded inside. throws out any levels which fail.
    """

    # Split all into early and late
    # for each prim, determine its early and late
    day_window = None
    if expt=="primitivesall":
        if animal=="Pancho":
            day_window = list(range(210605, 210613+1)) # the days in which to look.
        elif animal=="Diego":
            day_window = [210610, 210611, 210612, 210613] # the days in which to look.
        else:
            assert False
    elif expt=="primcat12":
        day_window = list(range(210731, 210804+1)) # the days in which to look.
    assert day_window is not None


    list_levels = D.Dat[groupby].unique()
    list_days = D.Dat["date"].unique()

    # For each primtuple, check if has enough trials early and late
    #### Output is, for each primtuple, all the inds present on each date.
    DictDateInds = {}
    for level in list_levels:

        DictDateInds[level] = {}
        for day in list_days:
            # save the indices and dates for this primtuple
            inds = D.filterPandas({"date":[day], groupby:[level]})
            DictDateInds[level][str(day)] = inds

    #### For each day, find the levels which have at least a min num trials.
    DictDay = {}
    for day in list_days:
        list_levels_this = []
        for level, val in DictDateInds.items():
            if len(val[day])>nmin:
                list_levels_this.append(level)

        DictDay[day] = list_levels_this

    for day, prims in DictDay.items():
        print(day, len(prims))

    # given two days, find set of overlapping prims
    def _olap(day1, day2):
        return [this for this in DictDay[str(day1)] if this in DictDay[str(day2)]]

    def _intersect(list_of_lists):
        result = set(list_of_lists[0])
        for s in list_of_lists[1:]:
            result.intersection_update(s)
        return result

    def intersect_days(list_days):
        list_of_l = [DictDay[str(d)] for d in list_days]
        return _intersect(list_of_l)


    # will take the earliers and latest which exist for each prim.
    DictDateEarlyLate = {}
    for prim, days in DictDateInds.items():
        # ealiest day in this window
        day_candidates = [d for d, inds in days.items() if len(inds)>nmin and int(d) in day_window] # days with at least nmin trials
        if len(day_candidates)<2:
            continue
    #     print(sorted(day_candidates))
        # take the first day 
        firstday = day_candidates[0]
        lastday =  day_candidates[-1]

        DictDateEarlyLate[prim] = [firstday, lastday]

    if SAVEDIR is not None:
        from pythonlib.tools.expttools import writeDictToYaml, makeTimeStamp
        writeDictToYaml(DictDateEarlyLate, f"{SAVEDIR}/DictDateEarlyLate.yaml")
        writeDictToYaml(DictDay, f"{SAVEDIR}/DictDay.yaml")
        writeDictToYaml(DictDateInds, f"{SAVEDIR}/DictDateInds.yaml")


    if return_filtered_dataset:
        Dfilt = D.filterPandas({groupby:list(DictDateEarlyLate.keys())}, "dataset")
        return DictDateEarlyLate, DictDay, DictDateInds, Dfilt
    else:
        return DictDateEarlyLate, DictDay, DictDateInds
