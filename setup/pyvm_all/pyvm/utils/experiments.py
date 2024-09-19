""" helpers to extract infomration about experiments, etc.
"""
from pyvm.utils.directories import get_metadata
from pyvm.globals import PYVM_DIR
from pythonlib.tools.expttools import load_yaml_config


def get_params(x):
    return get_params_yaml(x)

def get_params_yaml(DATE):
    exptname_or_date = DATE
    """ Loads params from yaml file
    PARAMS;
    - exptname_or_date, str, either name or date, the file is <exptname_or_date>.yaml
    RETURNS:
    - dict
    """
    

    from pythonlib.tools.expttools import load_yaml_config

    path = f"{PYVM_DIR}/metadata/{exptname_or_date}.yaml"
    params = load_yaml_config(path)
    animal = params["animal"]

    # Convert the input format (start, stop) into range(start, stop+1)
    for i, this in enumerate(params["list_vidnums"]):
        print(this)
        print(type(this))
        if this is not None:
            assert len(this)==2
            params["list_vidnums"][i] = range(this[0], this[1]+1)

    # Newer verions, some attributes detected automaticlaly. include these for legacy.
    if "list_camnames" not in params.keys():
        # make allow_gen ... false, so avoid recusurive call of get_params_yaml
        metadat = get_metadata(DATE=exptname_or_date, animal=animal,allow_generate_from_scratch=False) # get auto params
        if metadat is not None:
            params["list_camnames"] = metadat["list_camnames"]
        else:
            params['list_camnames'] = None

    if "dirname" not in params.keys():
        params["dirname"] = exptname_or_date

    return params


    # # ####### DONT DEPEND ON CONDITION
    # if exptname == "camtest5":
    #     # Things that dont depend on condition
    #     list_camnames = ["cam1bfu", "cam2bfs", "cam3flea"]
    #     dirname = "210826_camtest5"

    # elif exptname == "cagetest1":
    #     list_camnames = ["bfu", "bfs", "flea"]
    #     dirname = "210903_cagetest1"
    # else:
    #     assert False

    # ######## DOES DEPEND ON CONDITION
    # list_conditions = ["behavior", "wand"]
    # list_bodyparts = [
    #     ["fingertip"],
    #     ["blue", "red"]
    # ]
    # list_skeletons = [[], []]
    # list_start = [0.15, 0.] # fraction of video, for extraction frmaes.
    # list_stop = [0.85, 1.]
    # list_combinecams = [True, True]

    # if exptname == "camtest5":
    #     # One per condition
    #     list_vidnums = [
    #         range(3, 17+1), # trials where there is behavior
    #         None
    #     ]
    #     list_numframes2pick = [5, 75] # frames per video, so make low if many videos.
    # elif exptname == "cagetest1":
    #     list_vidnums = [
    #             range(22, 102+1), # trials where there is behavior
    #             None
    #         ]
    #     list_numframes2pick = [2, 75] # frames per video, so make low if many videos.
    # else:
    #     assert False

    # params["list_conditions"] = list_conditions
    # params["list_bodyparts"] = list_bodyparts
    # params["list_skeletons"] = list_skeletons
    # params["list_start"] = list_start
    # params["list_stop"] = list_stop
    # params["list_combinecams"] = list_combinecams

    # params["list_camnames"] = list_camnames
    # params["dirname"] = dirname
    # params["list_vidnums"] = list_vidnums
    # params["list_numframes2pick"] = list_numframes2pick

    # return params

