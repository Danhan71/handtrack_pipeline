""" tools for working with hdf5 groups"""

## method to walk throough the file.
def find_foo(name):
    print(name)

	# TrialRecord.visit(find_foo)

def print_level(group):
    print('--- printing each level and first entry')
    for key in group.keys():
        print(group[key])
        try:
            print(group[key][0])
        except:
            pass


def group2dict(group):
    if False:
        # this is old version. will stop going down tree once it
        # stops seeing an hdf5 group.
        """recursive, so goes through all
        levels and stops when encounter not hdf5. 
        this means that if group is dict, then does not
        search lower levels for hdf5, just
        returns the input"""
        import h5py
        outdict = {}
        if isinstance(group, dict):
            # then is not h5, so just output the input

            outdict = group    
        else:
            for key in group.keys():
                # print(key)
                # print(type(group[key]))
                # print(group[key])
                if isinstance(group[key], h5py._hl.group.Group):
                    outdict[key] = group2dict(group[key])
                    # print('skipping, since this is a group')
                elif isinstance(group[key], dict):
                    outdict[key] = group[key]
                elif isinstance(group[key][()], bytes):
                    outdict[key] = group[key][()].decode()
                else:
                    outdict[key] = group[key][()]
        return outdict
    else:
        """recursive, so goes through all
        levels and does not stop even if not an hdf5. stops at explicit type matching"""
        import h5py
        import numpy as np
        outdict = {}
        # print(group.keys())
        for key in group.keys():
            # print(f"group2dict - at key: {key}")
            if isinstance(group[key], (h5py._hl.group.Group, dict)):
                # keep going, for dict could still be hhdf45 below
                outdict[key] = group2dict(group[key])
            elif isinstance(group[key], (tuple, np.ndarray, str, int)):
                # then this is data, stop here
                outdict[key] = group[key]
            elif isinstance(group[key], h5py._hl.dataset.Dataset):
                if isinstance(group[key][()], bytes):
                    # then this is data and is text. decode it and stop here
                    outdict[key] = group[key][()].decode()
                else:
                    # otherwise this should be h5 dataset that is not text.
                    outdict[key] = group[key][()]
            elif group[key] is None:
                outdict[key] = group[key]
            else:
                print("group")
                print(group)
                print("key")
                print(key)
                print(group[key])
                print(type(group[key]))
                print(type(group[key][()]))
                assert False, "what is it?"
        return outdict
