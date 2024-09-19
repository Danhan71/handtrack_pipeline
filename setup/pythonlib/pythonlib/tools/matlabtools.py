""" For working with matlab
"""


import numpy as np

def convert_to_npobject(listthis):
    """ Converts nested list into nested numpy object, which
    can be saved and opened in  matlab as a nested cell array/
    PARAMS;
    - listthis, list to convert. if this not a list, just returns
    the same. (can also be tuple)
    RETURNS:
    - listthis_obj, same, but nested np object.
    NOTE: listthis_obj can be saved using scipy:
        from scipy.io import savemat
        fname = "/tmp/test.mat"
        # x = np.asarray(T.PlanDat["Plan"], dtype='object')
        savemat(fname, savedict)
    """
    if not isinstance(listthis, (list, tuple)):
        return listthis
    
    listthis_obj = np.zeros((len(listthis),), dtype='object')
    for i, item in enumerate(listthis):
        if isinstance(item, (list, tuple)):
            listthis_obj[i] = convert_to_npobject(item)
        else:
            listthis_obj[i] = item
    return listthis_obj

