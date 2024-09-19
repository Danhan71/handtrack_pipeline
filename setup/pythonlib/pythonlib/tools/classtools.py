""" To work with python class objects
Made this as was setting up for copying Dataset.
"""

def attributes_get_capitalized_or_underscore_capitalized(obj):
    """ Get list of attributes (strings) that either start with
    capital, or with _<capital>.
    """
    attributes_with_capital = [attr for attr in dir(obj) if (attr[0].isupper()) or (attr[0]=="_" and attr[1].isupper())]
    return attributes_with_capital

def concat_objects_attributes_flexible(obj_new, list_obj, concat_dicts_keyed_by_trialcode=True):
    """ Concatenate attributes across obj in list_obj, and assign the result as
    attributes of obj_new
    PARAMS:
    - obj_new, the obj to add attributres to, usually is already cocnatetted of list_obj.
    - list_obj, the objs for which attributes are combined.
    - concat_dicts_keyed_by_trialcode, concat atributes which are dicts, keyed by trialcode.
    Must have obj.Dat be a dataframe with "trialcode" as column. Will check that
    keys present across obj has identical values.
    RETURNS:
    - list_attr_identical_and_concat, list of str that are attributes that were concatted across obj in list_obj.
    (NOTE: (modifies obj_new))
    """
    from pythonlib.tools.checktools import check_objects_identical
    import copy
    ######### DATA THAT YOU NEED TO MERGE ACROSS DATASETS

    # (1) Attributes that start with captial, or _<capital>, and are string type --> they must be
    if len(list_obj)<2:
        # for code to work. This should still work.
        list_obj = [list_obj[0], list_obj[0]]

    # identical across input objects. Will inherit that value.
    list_attr_identical = attributes_get_capitalized_or_underscore_capitalized(list_obj[0])
    # Only those that are strings
    list_attr_identical = [attr for attr in list_attr_identical if isinstance(getattr(list_obj[0], attr), str)]
    # print(list_attr_identical)
    # list_attr_identical = ["TokensVersion"]
    for attr in list_attr_identical:
        items = [getattr(d, attr) for d in list_obj]
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                if check_objects_identical(items[i], items[j], PRINT=True)==False:
                    print("These items are different across object you are trying to concat:")
                    print(items)
                    assert False
        item_take = items[0]
        print(f"- Assigning to D.{attr} this value:", item_take)
        setattr(obj_new, attr, item_take)

    if concat_dicts_keyed_by_trialcode:
        # (2) Attributes which are dicts, and which has keys that are trialcodes, they will be concated.

        def _is_trialcode(x, Dthis):
            """ return True if is string and is like yyyyyy-{}-{}"""
            from pythonlib.tools.stringtools import trialcode_to_tuple
            return trialcode_to_tuple(x) is not None
        
        list_attr_identical = attributes_get_capitalized_or_underscore_capitalized(list_obj[0])
        list_attr_identical_and_concat = []
        for attr in list_attr_identical:
            obj = getattr(list_obj[0], attr)
            if isinstance(obj, dict) and len(obj)>0 and _is_trialcode(list(obj.keys())[0], list_obj[0]):
                list_attr_identical_and_concat.append(attr)
        print("These attributes are dicts with keys that are trialcodes, which will concat:", list_attr_identical_and_concat)

        for attr in list_attr_identical_and_concat:
            dict_this = copy.copy(getattr(list_obj[0], attr))
            for d in list_obj[1:]:
                for tc, val in getattr(d, attr).items():
                    if tc in dict_this:
                        if not check_objects_identical(dict_this[tc], val):
                            check_objects_identical(dict_this[tc], val, PRINT=True)
                            print("----------------")
                            print(tc)
                            print(dict_this[tc])
                            print(val)
                            # for k in dict_this[tc].keys():
                            #     print(dict_this[tc][k] == val.)
                            # print(len(dict_this[tc].Tokens))
                            # print(len(val.Tokens))
                            for tk1, tk2 in zip(dict_this[tc].Tokens, val.Tokens):
                                print(1, tk1)
                                print(2, tk2)
                                for k, v in tk1.items():
                                    print(k, "-", tk1[k], ' .. ', tk2[k])
                            assert False, "[Probably class object needs an __eq__() method!. how can diff datasets have overlapping trialcodes with diff values?"
                    else:
                        # append it
                        dict_this[tc] = val
            setattr(obj_new, attr, dict_this)
            print(f"- Assigning to D.{attr} this dict (concatted, with trialcode keys, showing first 3):")
            print(list(dict_this.items())[:3])
            # for k, v in dict_this.items():

    # TODO (3) Attributes which are lists, in which case take either union or intersection

    return list_attr_identical_and_concat

