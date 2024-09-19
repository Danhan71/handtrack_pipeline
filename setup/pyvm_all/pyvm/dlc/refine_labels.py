import deeplabcut
from pyvm.utils.experiments import get_params
from pyvm.dlc.utils import find_expt_config_paths
from pyvm.globals import BASEDIR, WINDOWS
from initialize import find_expt_config_paths
from pythonlib.tools.expttools import load_yaml_config


dict_paths, base_paths= find_expt_config_paths(name, condition)
pcf = list(dict_paths.values())
pcf = pcf[0]

%gui wx
deeplabcut.refine_labels(pcf)