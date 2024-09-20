
from pyvm.utils.experiments import get_params
from pyvm.globals import BASEDIR


def find_expt_config_paths(exptname, condition, animal):
    """
    Return dict, holding path of config file for all cameras under this expt and condition. 
    e..g,:
    - expt = "camtest5"
    - condition = "beahvior", or "wand"
    RETUNRS:
    - dict_path[camname] = path
    - base_paths[camname] = "/data2/camera/211106_cagetest2/behavior/DLC/combined-bfs_flea_ffly-Lucas-2021-11-09" (i.e., wituout config.yaml)
    NOTE:
    - fails if any camera doesnt find one and only one DLC config
    """
    from pythonlib.tools.expttools import findPath
    # from pyvm.dlc.initialize import get_params

    params = get_params(exptname, animal)

    # Find the ind for the desired condition
    list_conditions = params["list_conditions"]
    # ind_condition = [i for i, cond in enumerate(list_conditions) if cond==condition]
    # assert len(ind_condition)==1
    ind_condition = 0
    list_combinecams = params["list_combinecams"]
    combine_cameras = list_combinecams[ind_condition]

    # Find the path for this expt/cond, for all cams
    dirname = params["dirname"]
    list_camnames = params["list_camnames"]
    animal = params["animal"]

    dict_path = {}
    if combine_cameras:
        list_paths = findPath(f"{BASEDIR}/{animal}/{dirname}/{condition}/DLC", [["combined"]+list_camnames], "config")
        assert len(list_paths)==1
        dict_path["combined"] = list_paths[0]
    else:
        for camname in list_camnames:
            basepath = f"{BASEDIR}/{animal}/{dirname}/{condition}/DLC"
            list_paths = findPath(f"{BASEDIR}/{animal}/{dirname}/{condition}/DLC", [[camname]], "config")
            assert len(list_paths)==1
            dict_path[camname] = list_paths[0]

    base_paths = {}
    from pythonlib.tools.expttools import fileparts
    for k, v in dict_path.items():
        base_paths[k] = fileparts(v)[0][:-1] # remove / at end

    return dict_path, base_paths


def find_analysis_path(base_path, analysis_suffix, do_iterate_if_exists=False):
    """ 
    Find the latest analysis path using this analysis_suffix. if it exists, then can choose whether
    to return it, or to iterate +1. if noit yet created, then
    returns with iteration = 0
    PARAMS:
    - base_path, like "/data2/camera/211106_cagetest2/behavior/DLC/combined-bfs_flea_ffly-Lucas-2021-11-09"
    - analysis_suffix, str
    - return_latest_without_iterating, then returns the latest.
    RETURNS:
    - analysis_path, str
    - index, iteration number, 0, 1,2 ..
    """

    from pythonlib.tools.expttools import findPath, extractStrFromFname, writeDictToYaml, load_yaml_config

    list_path = findPath(base_path, [[analysis_suffix]], None)

    if len(list_path)==0:
        # then never done analyses before:
        if do_iterate_if_exists:
            # then create new with iteration 0
            index = 0
            # analysis_path = f"{base_path}/analyze_videos-{analysis_suffix}-0"
        else:
            print(f"{base_path}/analyze_videos-{analysis_suffix}-0")
            assert False, "did not find"
    else:
        list_indices = [extractStrFromFname(path, "-", -1) for path in list_path] 
        if do_iterate_if_exists:
            # start a new analysis.
            index = int(list_indices[-1])+1
        else:
            index = int(list_indices[-1])
    
    analysis_path = f"{base_path}/analyze_videos-{analysis_suffix}-{index}"

    return analysis_path, index



