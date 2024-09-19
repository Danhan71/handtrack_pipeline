from tools.utils import * 
from tools.plots import *
from tools.analy import *
from tools.calc import *
from tools.analyplot import *
from tools.preprocess import *
from tools.dayanalysis import *
from analysis.line2 import *
from analysis.probedatTaskmodel import *
from pythonlib.drawmodel.analysis import *
from pythonlib.tools.stroketools import *

# Load filedata, quick




expt_info_list = name.split('_')
date = str(expt_info_list[0])
expt = expt_info_list[1]
if len(expt_info_list) == 3:
    sess = expt_info_list[2]
else:
    sess = 1

print (animal, name, expt, sess)

fd = loadSingleDataQuick(animal, date, expt, sess)

from tools.handtrack import HandTrack, getTrialsCameraFrametimes

# if expt=="camtest5":
#     ind1_vid = 3
#     ind1_ml2 = 1
# elif expt=="chunkbyshape4":
#     ind1_vid = 4
#     ind1_ml2 = 1
# elif expt=="primpancho1d":
#     ind1_vid=1
#     ind1_ml2=1
# else:
#     assert False, "Please enter a new statement for the current expt"

# I think it doesn't matter now, since I decided all starting at 1

ind1_vid=1
ind1_ml2=1

HT = HandTrack(ind1_vid, ind1_ml2, fd, date=name, expt=expt)