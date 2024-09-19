import numpy as np

def check_objects_identical(obj1, obj2, PRINT=False):
    """ Works with commonly used objects, checks if identical,
    Returns False if any difference found
    otherwise True.
    """

    # Go thru all cases. if any are false, returns False
    if not type(obj1)==type(obj2):
        if PRINT:
            print('obj1',obj1)
            print('obj2',obj2)
        return False
    elif isinstance(obj1, (tuple, list)):
        # if not len(obj1)==len(obj2):
        #     if PRINT:
        #         print('obj1',obj1)
        #         print('obj2',obj2)
        #     return False
        for x, y in zip(obj1, obj2):
            good = check_objects_identical(x,y)
            if not good:
                if PRINT:
                    print('obj1',obj1)
                    print('obj2',obj2)
                return False
    elif isinstance(obj1, dict):
        if not sorted(obj1.keys()) == sorted(obj2.keys()):
            if PRINT:
                print('obj1',obj1, obj1.keys())
                print('obj2',obj2, obj2.keys())
            return False
        for k in obj1.keys():
            good = check_objects_identical(obj1[k], obj2[k])
            if not good:
                if PRINT:
                    print(k)
                    print('obj1',obj1, obj1[k])
                    print('obj2',obj2, obj2[k])
                return False
    elif isinstance(obj1, (np.ndarray, np.generic)):
        # if np.isempty(obj1) and np.isempty(obj2):
        #     pass

        if np.all(np.isnan(obj1)) and np.all(np.isnan(obj2)):
            pass
        elif not obj1.shape == obj2.shape:
            return False
        elif not np.all(np.isclose(obj1, obj2)):
            if PRINT:
                print('obj1',obj1)
                print('obj2',obj2)
            return False
        else:
            pass
    elif isinstance(obj1, float):
        if not np.all(np.isclose(obj1, obj2)):
            return False
    elif isinstance(obj1, (int, str, float)):
        if not obj1==obj2:
            if PRINT:
                print('obj1',obj1)
                print('obj2',obj2)
            return False
    elif obj1 is None:
        pass
        # if not obj1==obj2:
        #     if PRINT:
        #         print('obj1',obj1)
        #         print('obj2',obj2)
        #     return False
    else:
        # Use built in method
        try:
            if not obj1==obj2:
                if PRINT:
                    print('obj1',obj1)
                    print('obj2',obj2)
                return False
        except Exception as err:
            print('obj1',obj1)
            print('obj2',obj2)
            print(type(obj1))
            print(type(obj2))
            print("Might ahve to add this type... See the error message")
            raise err
        # # use standard equality
        # try:
        #     if not obj1==obj2:
        #         if PRINT:
        #             print('obj1',obj1)
        #             print('obj2',obj2)
        #         return False
        # except Exception as err:
        #     print('obj1',obj1)
        #     print('obj2',obj2)
        #     print(type(obj1))
        #     print(type(obj2))
        #     print("Might ahve to add this type...")
        #     raise err
    if PRINT:
        print("tests passed")
    # got here, passed all tests.
    return True
        

def check_is_categorical(item, types_categorical=(str, int)):
    """ Check the type, returen whether is categorical.
    PARAMS;
    - types_categorical, tuple of types. if item is in this, or is 
    a tuple with inner items all in this, then is categorical.
    RETURNS:
    - bool, if is categoritgcal.
    """

    if isinstance(item, tuple):
        # check taht each inner item is in good types
        return all([isinstance(x, types_categorical) for x in item])
    else:
        return isinstance(item, types_categorical)
