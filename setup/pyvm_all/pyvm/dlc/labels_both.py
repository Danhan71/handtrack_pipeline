import deeplabcut
import argparse
from pyvm.utils.experiments import get_params
from pyvm.dlc.utils import find_expt_config_paths
from pyvm.globals import BASEDIR, WINDOWS
from initialize import find_expt_config_paths
from pythonlib.tools.expttools import load_yaml_config


dict_paths_wand, base_paths_wand= find_expt_config_paths(name, 'wand')
pcf_wand = list(dict_paths_wand.values())
pcf_wand = pcf_wand[0]

%gui wx
deeplabcut.label_frames(pcf_wand)

dict_paths_b, base_paths_b= find_expt_config_paths(name, 'behavior')
pcf_b = list(dict_paths_b.values())
pcf_b = pcf_b[0]

%gui wx
deeplabcut.label_frames(pcf_b)