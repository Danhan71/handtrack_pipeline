import numpy as np

def dict2list(line):
    """ given dict, where each item is indexed by number (in string format),
    converts to list, where order is maintained, using values from dict,
    and taking care of convering numbers to squeezed arrays, 
    NOTE: Is recursive.
    PARAMS:
    - line, dict, formatted like this:
    {'1': 'transform', '2': array([[0.]]), '3': array([[0.]]), '4': array([[0.]]), '5': array([[1.]]), '6': array([[1.]]), '7': 'trs'}
    where each value can itself be a line.
    OUTPUT:
    - line_list = ['transform', array(0.), array(0.), array(0.), array(1.), array(1.), 'trs']"""

    def _keys_are_all_ints(x):
        """ Returns True if the keys for x (a dict) are all
        integers"""
        assert False, "not working, since keys are strings encoding ints"
        for k in x.keys():
            print(k, type(k))
            if not isinstance(k, int):
                return False
        return True

    def _f(x):
        if isinstance(x, str):
            return x
        elif isinstance(x, np.ndarray):
            return x.squeeze()
        elif isinstance(x, dict) and list(x.keys())[0]=="1":
            # Then this is nested dict...
            # print(x.keys())
            return dict2list(x)
        elif isinstance(x, dict):
            print(x.keys())
            return x
        else:
            print(type(x))
            assert False, "if this is dict, then you are looking at nested dict."
    line_list = []
    for i in range(len(line)):
        idx = f"{i+1}"
        val = _f(line[idx])
        line_list.append(val)
    return line_list


def dict2list2(line, fix_problem_with_single_items=False):
    """ 
    BETTER - this deals with hierarchical cases better, including cases where skips a level of hierachy 
    and then suddenly does the 1,2,3 keys thing again.
    Otherwise should deal with all cases dict2list can deal with.
    given dict, where each item is indexed by number (in string format),
    converts to list, where order is maintained, using values from dict,
    and taking care of convering numbers to squeezed arrays, 
    NOTE: Is recursive.
    PARAMS:
    - line, dict, formatted like this:
    {'1': 'transform', '2': array([[0.]]), '3': array([[0.]]), '4': array([[0.]]), '5': array([[1.]]), '6': array([[1.]]), '7': 'trs'}
    where each value can itself be a line.
    - fix_problem_with_single_items, bool, see within code.
    OUTPUT:
    - line_list = ['transform', array(0.), array(0.), array(0.), array(1.), array(1.), 'trs']"""

    def _keys_are_all_ints(x):
        """ Returns True if the keys for x (a dict) are all
        integers"""
        # assert False, "not working, since keys are strings encoding ints"
        for k in x.keys():
            if not k.isnumeric():
                return False
            # if not isinstance(k, int):
            #     return False
        return True

    def _f(x):
        if isinstance(x, str):
            return x
        elif isinstance(x, np.ndarray):
            return x.squeeze()
        elif isinstance(x, dict) and len(x.keys())==0:
            # Then just return, its empty dict
            return x
        # elif isinstance(x, dict) and list(x.keys())[0]=="1":
        elif isinstance(x, dict) and _keys_are_all_ints(x):
            # Then this is nested dict like:
            # {'1': 'transform', '2': array([[0.]]), '3': array([[0.]]), '4': array([[0.]]), '5': array([[1.]]), '6': array([[1.]]), '7': 'trs'}
            line_list = []
            for i in range(len(x)):
                idx = f"{i+1}"
                val = _f(x[idx])
                line_list.append(val)
            return line_list
        elif isinstance(x, dict):
            return {k:_f(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [_f(v) for v in x]
        else:
            print(type(x))
            assert False, "if this is dict, then you are looking at nested dict."

    if fix_problem_with_single_items:
        if isinstance(line, dict) and len(line.keys())>0 and not _keys_are_all_ints(line):
            # becuase if it ismatlab cell array length 1, then returns not {"1":dict} [which is desired] but instead just the inner item, dict..
            line = {"1":line}

    return _f(line)
