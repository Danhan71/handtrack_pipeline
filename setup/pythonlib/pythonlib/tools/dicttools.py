"""generally takes in list of dicts, where all dicts have same keys"""


def printOverviewKeyValues(summarydict_all, keystoget=None):
    """Prints all keys and their unique values. ignoers some keys that have numerical vbalues."""
    if keystoget is None:
        keystoget = set([kk for k in summarydict_all for kk in k.keys()])

    out = {}
    for key in keystoget:
        if key not in ["planner", "res", "cost", "x0", "x1", "x2", "x3", "x4", "x5", "x6", "filedata", "fd"]:
            if not isinstance(summarydict_all[0][key], dict):
                try:
                    values = set([s[key] for s in summarydict_all])
                except:
                    try:
                        set([s[key] for s in summarydict_all])
                        # for s in summarydict_all:
                        #     print(s[key])
                        values = set([", ".join(s[key]) for s in summarydict_all])
                    except Exception:
                        print(f"=== skipping {key}, because of Exception: {Exception}")
                        continue
            N = len(values)
            print("-----")
            print("- {} (N = {})".format(key, N))
            if N>100:
                print("SKIPPING PRINTING OF VALUES FOR {}, SINCE LEN {}".format(key, len(values)))
                continue
            [print(v) for v in values]
            out[key]=values
    return out


def filterSummary(summarydict, filterdict):
    """first converts to pandas dataframe, then filters
    e.g, filterdict = {"colname":[things,I,want], ...}    
    """
#     df = filterSummary(summaryflat, {"xnum":[1,2,3,4]})
    print("[OBSOLETE] - see pandastools for filterPandas")
    import pandas as pd
    df = pd.DataFrame(summarydict)
    for key in filterdict.keys():
        df = df[df[key].isin(filterdict[key])]
    return df

def filterDict(dat, filt):
    """ filter is dict, which says which items in 
    dat to keep. dat is list of dicts. e..g, if
    filter = {"key1":[list of vals], key2:val, ...}
    then will keep only items in dat which are consisteth
    with filter.
    """
    if filt is None or len(filt)==0:
        return dat
    else:
        datout = []
        for d in dat:
            keep=True
            for k, v in filt.items():
                if isinstance(v, list):
                    if d[k] not in v:
                        keep=False
                else:
                    if d[k]!=v:
                        keep=False
            if keep:
                datout.append(d)
        return datout


