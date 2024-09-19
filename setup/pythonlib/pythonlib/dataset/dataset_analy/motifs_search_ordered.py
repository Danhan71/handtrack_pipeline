""" All methods to take in tokens (lsit of dicts) and search for desired motifs 
(lits of dicts), WITH caring about temporal order. 
Is low-level, so it doenst try to ask what a given rule/model would do. It just takes 
search keys (at various levels of abstraction) and returns findings.

Can operate without considering the specific order used by monkey (just the abstract tokens).

Combining what used to be in:
- behaviorclass.py
- chunks.py

Related to:
- chunks, interprets rules/models 
- motifs.py, searches/extract all motifs and organized them, not based on a sesarch key
"""


#################################### MOTIFS 

# def alignsim_find_motif_in_beh_specific_byindices(self, taskstroke_inds, list_beh_mask=None):
#     """ Find this motif, where motif is defined as a specific sequence of taskstroke indices.
#     PARAMS:
#     - taskstroke_inds, list of list of ints, where each inner list is a specific sequence 
#     of taskstroek inds to look for. E..g, [1,2] means look for beh that got 1 --> 2. beh, if 
#     found, will be same length as taskstroke_inds.
#     NOTE: will use beh representation that is based on only useing each taskstroke once.
#     RETURNS:
#     - list_behstroke_inds, list of inds into datsegs.
#     """
    
#     tokens = self.alignsim_extract_datsegs()
#     motifthis = [tokens[i] for i in taskstroke_inds]
#     return self.alignsim_find_motif_in_beh_specific(motifthis, list_beh_mask)


def find_motif_in_beh_bykind(tokens, kind, params=None, list_beh_mask=None,
    DEBUG = False):
    """ Helper to search for this kind of motif. More abstract, since kind and params
    are used to construct the specific motif.
    PARAMS:
    - kind, string code, like "repeat"
    - params, dict of params, depends on kind.
    """

    def motifkind_generate_searchstring(kind, params=None, expt=None):
        """
        Generate a motif "search string" that can be used for filtering or 
        searching within datsegs.
        PARAMS:
        - kind, str, category of motif
        - params, dict, params which depend on kind
        - expt, str, name of expt, which sometimes needed.
        NOTE: this returns a specific string, without wildcards or regular expressions.
        """
        if kind=="repeat":
            # Repeat a token n times
            n = params["n"] 
            token = params["token"] 
            assert isinstance(token, dict)
            motif = [token for _ in range(n)] 
        elif kind=="chunk":
            # Simplest, a concrete chunk. is just looking for a specific sequence
            motif = params["tokens"]
        elif kind=="lolli":
            # circle and adjacent line, orinetation can be one of 4. 
            # order can be one of two
            orientation = params["orientation"] # token1-->token2, str {up, down, left, right}
            first_shape = params["first_shape"] # {circle, line}
            motif = [{}, {}]

            motif[1]["rel_from_prev"] = orientation

            if False:
                # old version, which used human strings for shapes. 
                if first_shape=="circle":
                    motif[0]["shape_oriented"] = "circle"
                    if orientation in ["up", "down"]:
                        motif[1]["shape_oriented"] = "vline"
                    elif orientation in ["left", "right"]:
                        motif[1]["shape_oriented"] = "hline"
                    else:
                        print(orientation)
                        assert False
                elif first_shape=="line":
                    motif[1]["shape_oriented"] = "circle"
                    if orientation in ["up", "down"]:
                        motif[0]["shape_oriented"] = "vline"
                    elif orientation in ["left", "right"]:
                        motif[0]["shape_oriented"] = "hline"
                    else:
                        print(orientation)
                        assert False
                else:
                    assert False
            else:
                # New version.
                if first_shape=="circle":
                    motif[0]["shapeabstract"] = "circle"
                    motif[1]["shapeabstract"] = "line"
                    if orientation in ["up", "down"]:
                        motif[1]["orient_string"] = "vert"
                    elif orientation in ["left", "right"]:
                        motif[1]["orient_string"] = "horiz"
                    else:
                        print(orientation)
                        assert False
                elif first_shape=="line":
                    motif[0]["shapeabstract"] = "line"
                    motif[1]["shapeabstract"] = "circle"
                    if orientation in ["up", "down"]:
                        motif[0]["orient_string"] = "vert"
                    elif orientation in ["left", "right"]:
                        motif[0]["orient_string"] = "horiz"
                    else:
                        print(orientation)
                        assert False
                else:
                    assert False

        else:
            print(kind)
            assert False

        return motif

    motif = motifkind_generate_searchstring(kind, params)
    x = find_motif_in_beh_specific(tokens, motif, list_beh_mask)
    if DEBUG:
        print("Motif:", motif)
        print("Tokens:")
        for t in tokens:
            print(t["shape_oriented"], t["rel_from_prev"])
        print(x)
    return x


def find_motif_in_beh_wildcard(list_beh, motifname, motifparams_abstract=None, 
    list_beh_mask=None, return_as_number_instances=False, 
    force_no_shared_tokens_across_motifs=True, 
    force_no_shared_tokens_across_motifs_usealltokens=False):
    """ [GOOD] 
    Helper - basically a "for loop" over different parametriazations of
    a motif kind (e.g., repeat is the kind, here get repeat 2,3, 4)...
    to search for a kind of motif (flexibly) within datsegs. 
    The most abstract, since will automatically generate many specific motifs,
    as many as needed. THink of this almost as running 
    alignsim_find_motif_in_beh_bykind, but in a loop, returning dict where 
    each key is one run with different params.
    """
    """ More flexible way to find motifs, such as "a circle repeated N times" where 
    N is wildcard so can be anything (or with constraint). Is the most abstract method.
    PARAMS:
    - list_beh, list of tokens. Usually datseg, which is list of dicts
    - motifname, str, a keyword for the kind of motif, e.g, {'repeats', 'lollis'}
    - motifparams_abstract, dict of params, flexible and depends on motifname
    - return_as_number_instances, bool (False), return in format of number of occurances found, 
    instaed of the list of instanc indices.
    - force_no_shared_tokens_across_motifs, (False), maeks sure motifs don't share toekns. PRoblem:
    if True, then can lead to undercounting, e.g, if there really are two lollis (l c l c) might only 
    keep the midlde one (cl) thereby undercounting (1 instead of 2). This is turned on for cases where
    it must be (e;..g repeats)
    - force_no_shared_tokens_across_motifs_usealltokens, see inside code
    RETURNS:
    - dict, where keys will depend on motifname, but generally reflect different subkinds of motifname.
    e.g., for repeats will be different length repeats. values are always lists of lists of ints, where
    ints are the indices.
    NOTE:
    - by default ensures that no tokens are reused across different keys in dict. e.g., 
    a two different lollis must use different tokens. does this by keeping matches that are
    found later.
    """

    assert force_no_shared_tokens_across_motifs==True, "to help me diagnose -- i expect this true, where did I switch off?"

    # assert force_no_shared_tokens_across_motifs==False, 'problem if [0,1], [1,2], [2,3], it keeps only [0,1]. should also keep [2,3]'
    if motifparams_abstract is None:
        motifparams_abstract = {}
    dict_matches = {}
    tokens = list_beh

    def _is_in(match, key, use_all_tokens_in_match=force_no_shared_tokens_across_motifs_usealltokens):
        """
        Remove previous found matches (stored in dict_matches) 
        if they use the same tokens as in match
        Returns True if match is a subset of one of the dict_matches in the list 
        of dict_matches for key key
        - match, list of ints
        - key, key into dict_matches. e.g., could be int, num repeats
        - use_all_tokens_in_match, bool, if True, the says match has been used
        (True) by searching that all tokens in match are subset of a previously found
        token. If False, then only needs this to apply for one token in match
        NOTE:
        e.g., 
        - match = [1,2]
        - dict_matches[n] = [[1,2,3], [7,8,9]]
        Then returns True
        """
        for match_check in dict_matches[key]:
            if use_all_tokens_in_match:
                if all([m in match_check for m in match]):
                    return True
            else:
                # print("HERE")
                # print(key)
                # print(match_check)
                # print(match)
                if any([m in match_check for m in match]):
                    return True                    
        return False

    def _remove_motifs_share_tokens(dict_matches, key_this):
        """
        Use this if want to make sure you throw out older chunks if find a newer chunk
        """
        for key_prev in dict_matches.keys():
            if key_prev != key_this:
                dict_matches[key_prev] = [match for match in dict_matches[key_prev] if not _is_in(match, key_this)]
        return dict_matches

    def _remove_motifs_share_tokens_v2(dict_matches):
        """ Better, in that it maximizes then num of chunks, doesnt
        care about order you foudn them.
        """
        from pythonlib.chunks.chunks import sample_all_possible_chunks

        # Collect all chunks (groups)
        list_groups = []
        for k, v in dict_matches.items():
            list_groups.extend(v)

        # get the single chunk with no redundant prims which maximizes n chunks.
        if len(list_groups)>0:
            list_hier, list_is_grp = sample_all_possible_chunks(list_groups, list(range(max([xx for x in list_groups for xx in x])+1)), 
                                       append_unused_strokes_as_single_group=False, return_single_grouping_max_n_grps=True)
            hier = list_hier[0]

            # map this hier back, only keeping if it is in hier
            dict_matches_new = {}
            for k, v in dict_matches.items():
                dict_matches_new[k] = [grp for grp in v if grp in hier]
        else:
            dict_matches_new = dict_matches

        return dict_matches_new


    if motifname=="repeat":
        # find all cases of repeats of the same shape. will make sure doesnt take the same 
        # token more than once - e..g, for a 3-repeat, will not also return the 2-repeat. will
        # take the max length rep.
        shapekey = motifparams_abstract["shapekey"] # "shape" or "shape_oriented"
        shape = motifparams_abstract["shape"] # e..g, circle
        nmin = motifparams_abstract["nmin"] # min num repeats (inclusive)
        nmax = motifparams_abstract["nmax"] # max num repeats (inclusive)
        if "allowed_rank_first_token" in motifparams_abstract.keys():
            allowed_rank_first_token = motifparams_abstract["allowed_rank_first_token"] # list of ints, only keeps motifs whose first item are in these ranks
        else:
            allowed_rank_first_token = None
        if "allowed_rank_last_token" in motifparams_abstract.keys():
            allowed_rank_last_token = motifparams_abstract["allowed_rank_last_token"] # same as above.
        else:
            allowed_rank_last_token = None
        # None to ignore
        # -1, -2, etc. interpreted as last, second-to-last..


        # construct single token
        token = {shapekey:shape}

        # search for repeats of incresaing n in the range of nmin to nmax, until fail.
        # each time add a repeat, remove the repeat of the preceding length.
        for n in range(nmin, nmax+1):

            motif = [token for _ in range(n)]
            list_matches = find_motif_in_beh_specific(list_beh, motif, list_beh_mask)

            # check if matches pass criteria
            def _convert_rank_to_positive(list_rank):
                """ Convert all the neg ranks in list_rank to positive, based
                on legnth of motif. returns a copy of list_rank
                """
                nmax = len(list_beh)
                list_rank_pos = []
                for rank in list_rank:
                    if rank<0:
                        list_rank_pos.append(nmax+rank)
                    else:
                        list_rank_pos.append(rank)
                return list_rank_pos

            if allowed_rank_first_token is not None:
                # convert negative ranks to positive
                allowed_rank_first_token_positives = _convert_rank_to_positive(allowed_rank_first_token)
                list_matches = [m for m in list_matches if m[0] in allowed_rank_first_token_positives]
                # assert False
            if allowed_rank_last_token is not None:
                # check their first toekns
                allowed_rank_last_token_positives = _convert_rank_to_positive(allowed_rank_last_token)
                # print("HER")
                # print(allowed_rank_last_token)
                # print(len(list_beh))
                # print(allowed_rank_last_token_positives)
                # print(list_matches)
                # A = len(list_matches)
                list_matches = [m for m in list_matches if m[-1] in allowed_rank_last_token_positives]
                # print(list_matches)
                # B = len(list_matches)
                # if A>0 and B==0:                    
                #     assert False
                # print("---")

            # store list matches
            key_this = n
            dict_matches[key_this] = list_matches

            # remove previous found matches if they use the same tokens.
            if force_no_shared_tokens_across_motifs:
                dict_matches = _remove_motifs_share_tokens(dict_matches, key_this)

    elif motifname=="lolli":
        # Find all the lollis, which means all cases of circle to line or line to circle, in any
        # direction (u d l r).
        # If two lollis share anything (e..g, circle) will keep only one. does this by findings
        # best combo across lollis to maximize n lollis.
        
        # list_orientation = ["up", "down", "left", "right"]
        # list_first_shape = ["circle", "line"]
        list_orientation = motifparams_abstract["list_orientation"]
        list_first_shape = motifparams_abstract["list_first_shape"]

        key_prev = None
        for o in list_orientation:
            for s in list_first_shape:
                par = {"orientation":o, "first_shape":s}
                m = find_motif_in_beh_bykind(tokens, "lolli", par)

                key_this = (o,s)
                dict_matches[key_this] = m

                # Remove previous lollis that have any overlap with the current
                # - go thru all previous keys
                # if force_no_shared_tokens_across_motifs:
                #     dict_matches = _remove_motifs_share_tokens(dict_matches, key_this)
        
        if force_no_shared_tokens_across_motifs:
            dict_matches = _remove_motifs_share_tokens_v2(dict_matches)
    else:
        print(motifname)
        assert False, "??"

    if return_as_number_instances:
        x = list(dict_matches.items())
        for k, v in x:
            dict_matches[k] = len(v)

    return dict_matches


def find_motif_in_beh_specific(list_beh, motif, list_beh_mask=None):
    """ Helper to search for this motif in datsegs, extracted from aligned beh-task
    using sim matrix alignement. Must enter specific motif
    PARAMS:
    - motif, list of tokens. Will search for this specific list. This can be like wildcard,
    if each dict in motif (lsit of dicts) only uses a subset of the keys in datsegs.
    i.e., Generic - given list of beh tokens, and a motif, find if/where this
    motif occurs.
    PARAMS:
    - list_beh, list of tokens, either int or str. e..g, [line, circle, ...], or list of
    objects that can be checked for equality using =. So datsegs (list of dicts) are also
    doable.
    - motif, list of tokens, same type as in list_beh. this is the filter.
    If motif is list of dicts, then will only check the dict keys here. So if list_beh has more
    keys, they will be ignored. Moreover, each element in motif can use different keys if
    desired. e.g., 
        motif = [{'shape': 'line'},
            {'shape_oriented': 'circle'}]
    - list_beh_mask, np array of bool int, same len as list_beh. baseically says that 
    only conisder substrings in list_beh for which all tokens are True in this mask.
    RETURNS:
    - list_matches, list of list, where each inner list are the indices into list_beh which 
    would match motif. e.g., [[0,1], [4,5]]
    """

    # for t in list_beh:
    #     print(t)
    # print(motif)
    # assert False
    assert isinstance(motif, list) and isinstance(list_beh, (list, tuple))
    def _motifs_are_same(behstring, motif):
        assert len(behstring)==len(motif)
        for a, b in zip(behstring, motif):
            if isinstance(a, dict) and isinstance(b, dict):
                # Then only check the keys in motif
                keys_to_check = b.keys()
                for k in keys_to_check:
                    if not a[k] == b[k]:
                        return False
            else:
                # Then check for complete ewqulaiyt
                if not a==b:
                    return False
        return True

    nend = len(list_beh) - len(motif)
    nmotif = len(motif)
    if list_beh_mask is not None:
        assert len(list_beh)==len(list_beh_mask)
    
    if len(list_beh)<nmotif:
        return []

    list_matches = []
    for i in range(nend+1):
        
        behstring = list_beh[i:i+nmotif]

        if list_beh_mask is not None:
            behstring_mask = list_beh_mask[i:i+nmotif]
            if ~np.all(behstring_mask):
                # skip this, since not all beh strokes are unmasked
                continue

        if _motifs_are_same(behstring, motif):
            list_matches.append(list(range(i, i+nmotif)))

    return list_matches


def motif_shorthand_name(motif_kind, motif_params):
    """ Return string, useful for dtaframe columns,.
    Works for motif_kind and motif_params that would pass into any of the two
    methods for abstractly defining motifs:
    alignsim_find_motif_in_beh_wildcard and alignsim_find_motif_in_beh_bykind
    """
    s = motif_kind

    def _append_param(s, param_key):
        if param_key in motif_params.keys():
            s += f"-{motif_params[param_key]}"
        elif "token" in motif_params.keys():
            if param_key in motif_params["token"]:
                s += f"-{motif_params['token'][param_key]}"
        return s
        
    if motif_kind=="repeat":
        for param_key in ["shape", "n"]:
            s=_append_param(s, param_key)
    elif motif_kind =="lolli":
        for param_key in ["orientation", "first_shape"]:
            s=_append_param(s, param_key)
    else:
        assert False
    return s


