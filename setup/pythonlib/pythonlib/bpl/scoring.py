""" for scoring parses"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pybpl.model import CharacterModel
from pybpl.library import Library

# Load Library
USE_HIST = True
LIB_PATH = '/data1/code/python/pyBPL/lib_data/'
lib = Library(LIB_PATH, use_hist=USE_HIST)
model = CharacterModel(lib)




def scoreMPs_factorized(MPlist_flat, use_hist=True, lib=None, return_as_tensor=True):
    """ outputs score type only, factorized into different compoentns of the prir.
    INPUT
    - MPlist, list of ctype (motor programs)
    - lib, a Library object.
    - use_hist, only applies if dont pass in lib, in which case will
    have to load library, using use_hist
    RETURNS:
    - scorelist, list of dicts, each dict has k:v where k is prior component and
    v is ll.
    
    """
    from pybpl.model import CharacterModel

    # Load model
    if lib is None:
        lib = Library(LIB_PATH, use_hist=use_hist)
    model = CharacterModel(lib)

    def _score(ctype):
        score_dict = model.score_type_monkey(ctype)
        return score_dict

    scorelist = [_score(ctype) for ctype in MPlist_flat]
    
    if not return_as_tensor:
        # Then convert to numpy
        scorelist2 = []
        for score in scorelist:
            scorelist2.append({k:v.numpy() for k, v in score.items()})
        scorelist = scorelist2
    return scorelist


def scoreMPs(MPlist_flat, use_hist=True, lib=None, scores_to_use = None):
    """ outputs score (tokenscore + typescore, ignoring image score)
    as list of scores
    INPUT
    - MPlist, list of ctype (motor programs)
    - lib, a Library object.
    - use_hist, only applies if dont pass in lib, in which case will
    have to load library, using use_hist
    - scores_to_use, list, elements from {"type", "token", "image"}
    NOTE: image is not implmeneted yet.
    """

    if scores_to_use is None:
        scores_to_use = ["type", "token"]
        
    from pybpl.model import CharacterModel

    # Load model
    if lib is None:
        lib = Library(LIB_PATH, use_hist=use_hist)
    model = CharacterModel(lib)

    def _score(ctype):
        s = torch.tensor([0.])
        for stype in scores_to_use:
            if stype=="type":
                s += model.score_type(ctype)
            elif stype=="token":
                ctoken = model.sample_token(ctype)
                s += model.score_token(ctype, ctoken)                
            elif stype=="image":
                assert "image" not in scores_to_use, "not implmemneted, need to save image."
            else:
                print(stype)
                assert False, "dont know this"
        return s.float().reshape(1)

    scorelist = [_score(ctype) for ctype in MPlist_flat]
    return scorelist
