""" string edit distances
"""

if False:
    def stringDist(a, b):
        import pyxdameraulevenshtein as dl
        # rerutns value between 0 and 1 (1 is max difference)
        # first map all the items to idx identifiers (e.g. from  ["C1", "C2", "L1", "L2"] to [1,2,3,4]
    #     alphabet = ["C1", "C2", "L1", "L2"]
    #     A = [alphabet.index(aa) for aa in a]
    #     B = [alphabet.index(bb) for bb in b]
        # e.g,, dl.damerau_levenshtein_distance(["C2", "C0", "L1", "L2"], ["C9", "LLL", "C", "L"])

        return dl.normalized_damerau_levenshtein_distance(a,b)


def nmatch_until_first_diff(A, B):
    """ score how many tokens are identical starting from onset of a and b.
    Is symmetric
    RETURNS:
    - score from 0 to 1, where 1 means all tokesn are diff, and 0 is idnteical.
    EG:
    - A = [1,2,3]
    - B = [1,2,3,4]
    then it counts 2, which is noirmalize to 1-2/4 = 0.5    
    """

    nmatch = 0
    for i, (b, t) in enumerate(zip(A, B)):
        if b==t:
            nmatch+=1
        else:
            break

    d_nmatch = 1- nmatch/max([len(A), len(B)])

    return d_nmatch
