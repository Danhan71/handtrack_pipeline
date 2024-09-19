""" verious functions useful for running experiemnts/analysis"""

import os

def makeTimeStamp(exptID="", datefirst=True):
    """useful for saving things, 
    gives yymmdd_hhmmss_{exptID}"""
    import time
    ts = time.localtime()
    tstamp = time.strftime("%y%m%d_%H%M%S", ts)

    if len(exptID)>0:
        if not datefirst:
            exptID+="-"
        else:
            exptID = "-" + exptID

    if datefirst==False:
        return f"{exptID}{tstamp}"
    else:
        return f"{tstamp}{exptID}"


def getDateList(sdate=None, edate=None):
    """    sdate = 200226, for Feb 26 2020
    RETURNS:
    - dates in str
    """
    
    from datetime import date, timedelta, datetime
    
    if sdate is None:
        sdate=date(2020, 1, 1)
    else:
        sdate = datetime.strptime(f"{sdate}", '%y%m%d')
    
    if edate is None:
        edate =  date.today()
    else:
        edate = datetime.strptime(f"{edate}", '%y%m%d')

    delta = edate - sdate       # as timedelta
    date_list = [sdate + timedelta(days=i) for i in range(delta.days+1)]
    # for i in range(delta.days + 1):
    #     day = sdate + timedelta(days=i)
    #     print(day)
    date_list = [d.strftime("%y%m%d") for d in date_list]
    return date_list

def writeStringsToFile(fname, stringlist, silent=True, append=False):
    """ atuoamtically overwrite
    fname, add .txt if you want.
    PARAMS:
    - append, bool, if True, then doesnt overwrite.
    """
    if append:
        mode = "a"
    else:
        mode = "w"
    with open(fname, mode) as f:
        for s in stringlist:
            if not silent:
                print(s)
            f.write(f"{s}\n")

def writeDictToYaml(dictdat, path):
    """ path shold include extension (.yaml)"""
    import yaml

    with open(path, 'w') as f:
        # yaml.dump(dictdat, f, default_flow_style=False)
        yaml.dump(dictdat, f, sort_keys=False)

def writeDictToTxt(dictdat, path):
    """ Write each key:val of dict as as new line in text file
    f"{key} : {val}"
    PARAMS:
    - path, imcludes extension.
    """
    list_s = []
    for key, val in dictdat.items():
        list_s.append(f"{key} : {val}")
    writeStringsToFile(path, list_s)

def writeDictToTxtFlattened(dictdat, path, header=None,
                            sorted_by_keys=False):
    """
    Each k:v pair is a row...
    :param dictdat:
    :param path:
    :return: lines: list of str
    """
    from pythonlib.tools.listtools import sort_mixed_type
    lines = [f"{str(k)} : {v}" for k, v in dictdat.items()]
    if sorted_by_keys:
        lines = sort_mixed_type(lines)
    if header is not None:
        lines = [header] + lines
    writeStringsToFile(path, lines)
    return lines

def extractStrFromFname(fname, sep, pos, return_entire_filename=False):
    """ given fname like '/data2/animals/Pancho/201030/201030_164324_arc2_Pancho_3.h5'
    pull out arc2 if give me sep="_" and pos = "2"
    - pos is 0,1, ..., or -1 to get end
    --- or, pos="all", to return list of strings
    - if return_entire_filename, then overwrites sep and pos, and instead returns 
    201030_164324_arc2_Pancho_3
    RETURNS:
    - either:
    --- single string (if pos is numer)
    --- list of strings (if pos=="all")
    """
    import os
    import re
    
    # 1) get the filename without leading dir, or training extension
    # e.g., gets 201030_164324_arc2_Pancho_3
    path = os.path.split(fname)[1]
    path = os.path.splitext(path)[0]

    if return_entire_filename:
        return path

    # 2) get posoptions of the separator
    idxs = [m.start() for m in re.finditer(sep, path)]
    idxs.append(len(path)) # to allow getting of last substr
    idxs.insert(0, -1) # to allow getting of first
    if isinstance(pos, str):
        if pos=="all":
            # return list of strings
            return [path[idxs[i]+1:idxs[i+1]] for i in range(len(idxs)-1)]
        else:
            assert False
    else:
        if pos==-1:
            # shift it back one, isnce appended -1 to end
            pos = -2
        # 2) get substring
    #     print(idxs)
        if pos==-2 or len(idxs)>=pos+2:
            # print(path)
            # print(idxs)
            # print(pos)
            # assert False
            return path[idxs[pos]+1:idxs[pos+1]]

        else:
            return None

def count_n_files_in_dir(dirpath, ext=None):
    """ 
    Count number of files that exist in dirpath, with extension
    ext (or ignore extgnesion if None)
    PARAMS:
    - ext, str, e.g., txt (dont give the period)
    RETURNS:
    - n, int
    - list_paths, list of str, full paths of all files found
    """
    import glob
    if ext:
        list_paths = glob.glob(f"{dirpath}/*.{ext}")
    else:
        list_paths = glob.glob(f"{dirpath}/*")

    return len(list_paths), list_paths

def check_if_dir_not_empty(dirpath):
    """ Reutrns True if any file exists in dirpath"""
    with os.scandir(dirpath) as it:
        if any(it):
            return True
    return False

def checkIfDirExistsAndHasFiles(dirname):
    """ returns (exists?, hasfiles?), 
    - hasfiles is obviously always False if exists
    is False
    """
    if os.path.isdir(dirname):
        exists = True
            
        if True:
            # Should be faster when many files in dir
            # https://stackoverflow.com/questions/49284015/how-to-check-if-folder-is-empty-with-python
            hasfiles = check_if_dir_not_empty(dirname)
        else:
            if len(os.listdir(dirname))>0:
                hasfiles = True
            else:
                hasfiles = False
    else:
        exists = False
        hasfiles = False

    return exists, hasfiles

def fileparts(path, return_up=False, append_slash=True):
    """ breaks down path into dir, path, ext. always works regaridless of whether give 
    relative or fullpath.
    PARAMS:
    - return_up, then also return one dir up.
    NOTES:
    - if path is full, then pathdir will be string.
    - if path is rel, then pathdir is empyt string.
    - if path is full, but only one level (e/g. path = /tmp.ext), pathdir is "/"
    - Will alkways be able to reconstruct path as pathdir + pathname + ext
    """
    # f = "/210517_152802_plan3_Red_1.pkl"
    import os
    pathdir, pathend = os.path.split(path)
    pathname, ext = os.path.splitext(pathend)
    if return_up:
        pathdir_up, pathdir_end = os.path.split(pathdir)

    if append_slash:
        if len(pathdir)>0 and pathdir!="/":
            pathdir+="/" # to allow reconstruction as pathdir + pathname + ext
    
    # also return one path up.
    if return_up:
        return pathdir, pathname, ext, pathdir_up, pathdir_end, pathend
    else:
        return pathdir, pathname, ext

def deconstruct_filename(filename):
    """ wrapper to do all procedures to get filename parts.
    Assumes that you use "-" as separator for lowest part of ilename
    PARAMS:
    - filename, str, absolute path
    RETURNS:
    - dict, various deconstructions. check it to see.
    """

    # Extract all parts of entire string of filename (i.e., absolute path components)
    list_entire_path_parts = fileparts(filename, True, append_slash=False)
    list_entire_path_dirs = [list_entire_path_parts[i] for i in [3, 0]]
    list_entire_path_subdirnames = [list_entire_path_parts[i] for i in [4, 5]]

    path_final_string = list_entire_path_parts[5]

    # Extract all parts of lowest string in filename 
    list_string_components = []
    ct = 0
    x = extractStrFromFname(filename,"-", ct, False)
    while x is not None:
        list_string_components.append(x)
        ct+=1
        x = extractStrFromFname(filename,"-", ct, False)


    path_parts = {}
    path_parts["filename_components_hyphened"] = list_string_components
    path_parts["basedirs"] = list_entire_path_dirs
    path_parts["basedirs_filenames"] = list_entire_path_subdirnames
    path_parts["filename_final_ext"] = path_final_string
    path_parts["filename_final_noext"] = list_entire_path_parts[1]

    return path_parts

def get_filename_most_recent_by_date(pathlist):
    """ Given paths with datestrings, extract the most recent.
    PARAMS:
    - pathlist, list of str, paths with format <stuff>-YYMMDD_HHMMSS.<ext>, (Note the hyphen!!)
    RETURNS:
    - path, single path, the most recent.
    """
    import numpy as np
    
    # 1) get list of dates machine the filenames
    dates = []
    for path in pathlist:
        fileparts = deconstruct_filename(path)["filename_components_hyphened"]
        datethis = fileparts[-1]
        if len(datethis)!=13:
            print(datethis)
            print(path)
            assert False, "should be YYMMDD_HHMMSS"
        dates.append(datethis)
    # 2) sort
    indices = np.argsort(dates) # string sorting sorts the digits (starting from leftmost)
    
    return pathlist[indices[-1]]


    
def modPathFname(path, prefix=None, suffix=None, do_move=True):
    """ moves file in path, to nbew filename either
    {prefix}-{path} or {path, without ext}-{suffix}-{ext}
    Only moves if source exists and target doesnt. eitehr way wioll return the target path.
    NOTES:
    - ok to entire full path, this code will pick out the lowest level filename.
    - only moves if both do_move==True and the above
    RETURNS:
    - new pathname
    """
    from pathlib import Path
    pathdir,pathname, ext = fileparts(path)

    if prefix:
        pathname = prefix + "-" + pathname
    if suffix:
        pathname = pathname + "-" + suffix
    pathout = pathdir + pathname + ext

    # Only do this if the source exists and the target doesnt exist.
    if do_move:
        if os.path.exists(path) and not os.path.exists(pathout):
            Path(path).rename(pathout)
            print("Renamed:")
            print("from: ", path)
            print("to: ", pathout)
    return pathout


def findPath(path_base, path_hierarchy, path_fname="", ext="",
    return_without_fname=False, sort_by="name", path_hierarchy_wildcard_on_ends=True,
    strings_to_exclude_in_path=None):
    """ get list of data, searches thru paths.
    INPUT:
    - path_base, str, static, common across all possible
    paths
    - path_hierarchy, list, where each element is another list,
    where each element in that list is a string, each of which
    will be separated by wildcards to find paths. 
    - path_fname, {str, None} make this None if want to have the last level of path_hierarchy
    be the filename.
    - ext, extensions for files. leave empty to not care.
    - return_without_fname, then returns path name but splt off from the
    final file name. (i..e, jsut gets the dir)s
    - sort_by, if "name", then alphabetically. otherwise {"size", "date"}. always incresaing order.
    - path_hierarchy_wildcard_on_ends, if true, allows /*[]*[]*/... otherwise does /[]*[]/
    - strings_to_exclude_in_path, list of strings, will exclude  apath if any of these strings occurs anywhere in the
    path.

    NOTES:
    The order
    matter. e..g,: 
    path_hierarchy = [['pancho', 'beh'], ['figures']) 
    ext = ".pkl"
    means look for 
    {path_base}/*pancho*beh*/*figures*/*.pkl
    NOTE:
    - length of path_hierarchy must match the hierarchy of the paths.
    EXAMPLE USAGE:
        path_base = "/data2/analyses/database/clustering/bysimilarity"
        path_hierarchy = [
            ["combined"],
            ["Red", "Pancho"],
            ["Pancho", "Red"],
        ]
        # path_hierarchy = [
        # ]
        path_fname = "SAVEDAT"
        ext = ".pkl"
        findPath(path_base, path_hierarchy, path_fname, ext)

    """
    import glob
    
    if strings_to_exclude_in_path is None:
        strings_to_exclude_in_path = []

    def _summarize(pathlist):
        print("Found this many paths:")
        print(len(pathlist))
        for p in pathlist:
            print("---")
            print(p)
    
    # Construct path
    path = path_base
    
    for p in path_hierarchy:
        if path_hierarchy_wildcard_on_ends:
            path += "/*"
        else:
            path += "/"
        for pp in p:
            path += f"{pp}*"
        if not path_hierarchy_wildcard_on_ends:
            path = path[:-1] # remove the last wildcard
    
    if path_fname is None:
        path += "*" + ext
    else:
        path += "/*" + path_fname + "*" + ext
    
    print("Searching using this string:")
    print(path)
    
    # Search
    pathlist = glob.glob(path)

    # Process
    if return_without_fname:
        print("-- Splitting off dir from fname")
        # pull out just the path before the file name
        import os
        pathlist = [os.path.split(p)[0] for p in pathlist]
        pathlist = list(set(pathlist))

    if sort_by == "name":
        pathlist = sorted(pathlist)
    elif sort_by == "size":
        import os
        fsizes = [os.path.getsize(f) for f in pathlist]

        tmp = [(s, p) for s, p in zip(fsizes, pathlist)]
        tmp = sorted(tmp, key=lambda x:x[0])
        pathlist = [t[1] for t in tmp]
    elif sort_by == "date":
        import os
        fsizes = [os.path.getmtime(f) for f in pathlist]
        tmp = [(s, p) for s, p in zip(fsizes, pathlist)]
        tmp = sorted(tmp, key=lambda x:x[0])
        pathlist = [t[1] for t in tmp]
    else:
        print(sort_by)
        assert False, "not coded"

    def string_check(path):
        """ returns True if none of the strings to check are in the path
        """
        out = all([s not in path for s in strings_to_exclude_in_path])
        return out

    pathlist = [path for path in pathlist if string_check(path)]

    _summarize(pathlist)

    return pathlist
    
def get_common_path(pathlist):
    """ if pathlist len>1, then gets the most specific path 
    that is common across all. if len==1, then gets the final directory.
    """
    import os

    assert isinstance(pathlist, list)

    if len(pathlist)>1:
        path_shared = os.path.commonpath(pathlist)
    elif len(pathlist)==1:
        path_shared = os.path.split(pathlist[0])[0]
    else:
        assert False
    return path_shared


def load_yaml_config(path, make_if_no_exist=False):
    """ 
    Load a file.yaml into an output dict
    RETURNS:
    - a dict, if the file empty, will be emty dict.
    NOTE:
    - path should have .yaml extension.
    """
    import yaml

    if make_if_no_exist:
        if not os.path.isfile(path):
            # 1) start this file
            print("initialized file: ", path)
            writeDictToYaml({}, path)

    with open(path) as file:
        outdict = yaml.load(file, Loader=yaml.FullLoader)

    if outdict is None:
        outdict = {}
    return outdict

def update_yaml_dict(path, key, val, allow_duplicates=True):
    """ loads and updates this file and appends val to this key,
    if key doesnt existt, then starts it, etc.
    never delets anything.
    INPUT:
    - allow_duplicates, if False, then doesnt add this val to key if it alreayd tehre
    """
    
    if not os.path.isfile(path):
        # 1) start this file
        print("initialized file: ", path)
        writeDictToYaml({key:val}, path)
    else:
        x = load_yaml_config(path)
        if key in x.keys():
            if allow_duplicates is False and val in x[key]:
                print("Skipping, since already there.", path, " with: ", key, val)
                # Already there - skip
                pass
            else:
                print("updating ", path, " with: ", key, val)
                x[key].append(val)
        else:
            print("updating ", path, " with new key:val: ", key, val)
            x[key] = [val]
        writeDictToYaml(x, path)

def read_all_lines_from_textfile(filepath):
    """ REturns list of strings
    """
    list_lines = []
    with open(filepath, 'r') as f:
        line = f.readline()
        while line:
            list_lines.append(line)
            line = f.readline()
    return list_lines


