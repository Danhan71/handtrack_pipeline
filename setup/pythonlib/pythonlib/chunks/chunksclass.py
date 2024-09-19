""" for working with chunks.
Each task can be chunked in different ways. Each instance here is a single way of chunking. 
Can then input beh and so on to compute things wrt this chunking. 

Also can keep track of the labels of task stroke, et.c, Line circle, etc., and can then do things
like search for beh that matches a given motif - e.g, line-->circle. 
Can do the same for chunks, e.g., chunk1 --> chunk2.
"""

import numpy as np

class ChunksClassList(object):
    """ A single ChunksClass instance represents all ways of chunking this Task given an
    expt and rule.
    Two methods exist for inputting chunks:
    1. "task_entry", meaning input a Task and rule, and will automatically find chunks
    2. "chunkslist_entry", input chunks directly. Useful if input chunks saved in monkeylogic code
    """

    def __init__(self, method, params):
        """ Different ways of initializing this list of chunks
        """

        self.Task = None
        self.Expt = None
        self.Rule = None

        if method=="task_entry":
            # enter the TaskClass object directly. Here will search for all chunks
            # based on set of rules (expt, rule)
            self._init_task_entry(params["Task"], params["expt"], params["rule"])
        elif method=="chunkslist_entry":
            # Enter a list of lists, each outer list correspoding to a single way of chunking
            # assumes no "chunking" but instead is using hierarchy.
            # This is the way you would input PlanClass information saved into ml2 tasks
            # i.e., see drawmodel.tasks
            if "shapes" in params.keys():
                shapes = params["shapes"]
            else:
                shapes = None
            self._init_chunkslist_entry(params["chunkslist"], params["nstrokes"], shapes)
        else:
            print(method)
            print(params)
            assert False, "code it"


    def _init_chunkslist_entry(self, chunkslist, nstrokes, list_shapes=None, 
            check_matches_nstrokes=True):
        """ Enter list of chunks, inputed in chunkslist format.
        PARAMS:
        - chunkslist, chunkslist[0] = [name, hier, flips, index, color],
        --- name, string
        --- hier, list of list of ints (or like)
         e.g.,, [['default',
              [array([1., 2.]), array([3., 4.])],
              [array([0., 0.]), array([0., 0.])],
              0, 
              colorlist_hier
              ]]
            - colorlist_hier,optional, currently not functioning, but list of array:
            --- e,.g., [array([0.44048823, 0.47708101, 0.44188513]),
                    array([0.47163037, 0.65416502, 0.40344111])], len of hier

        - nstrokes, int, need this since needs to make chunks all single strokes.
        - shapes, list of string or ints, categories for these shapes, same len as nstrokes
        - check_matches_nstrokes, bool, then asserts that got all and only these storkes. helpful for 
        catching 1-indexing from matlab.
        NOTE:  does Confirm that each model/index is unique
        """

        if list_shapes is not None:
            assert len(list_shapes)==nstrokes

        already_entered = {}

        self.ListChunksClass = []
        chunks = [i for i in range(nstrokes)] # assume that no stroke chunking
        for chunkthis in chunkslist:
            name = chunkthis[0]
            hier = chunkthis[1]
            flips = chunkthis[2]
            index = chunkthis[3]
            if len(chunkthis)>4:
                color = chunkthis[4]
            else:
                color = None
            fixed_order = None # uses default.
            C = ChunksClass(chunks, hier, fixed_order=fixed_order, 
                task_stroke_labels=list_shapes, name=name, 
                flips=flips, index=index, colorlist_hier=color)
            if check_matches_nstrokes:
                from .chunks import check_all_strokes_used
                all_used = check_all_strokes_used(C.Hier, nstrokes)
                if not all_used: 
                    print(nstrokes)
                    C.print_summary()
                    assert False, "maybe iyou entered using 1-indexing"

            # Confirm that each model/index is unique
            if name in already_entered.keys():
                assert index not in already_entered[name]
                already_entered[name].append(index)
            else:
                already_entered[name] = [index]

            # Add it
            self.ListChunksClass.append(C)



    def _init_task_entry(self, Task, expt, rule, DEBUG=False):
        """
        Input a single Task.
        If pass in expt and rule, then will generate chunks autoatmically, 
        given a Task and rule (i.e., model).
        Procedures for genreating usually should be hand-coded for each rule.
        PARAMS:
        - Task, TaskClass instnace
        - expt, str, the experiment name.
        - rule, str, the rule name
        NOTE:
        - expt and rule are used for deciding what constitites a "chunk". Leave them as None if
        want to enter other method for deciding chunk
        """
        from pythonlib.dataset.modeling.discrete import find_chunks_hier

        self.Task = Task
        self.Expt = expt
        self.Rule = rule
        self.ListChunksClass = []

        # Extract all chunks, hierarhchies, etc
        use_baseline_if_dont_find = True # e..g, for lolli.

        # list_chunks, list_hier, list_fixed_order = find_chunks_wrapper(Task, 
        #     expt, rule, use_baseline_if_dont_find=use_baseline_if_dont_find)

        list_chunks, list_hier, list_fixed_order = find_chunks_hier(self.Task,
            self.Rule, use_baseline_if_dont_find=use_baseline_if_dont_find)
        if DEBUG:
            print("list_chunks", list_chunks)
            print("list_hier", list_hier)
            print("list_fixed_order", list_fixed_order)


        # - Make one ChunksClass instance for each chunk:
        list_shapes = [s for s in Task.Shapes]
        # e.g., [['line-3-0', {'x': -1.146, 'y': 0.05, 'sx': None, 'sy': None, 'theta': None, 'order': None}], ['V-4-0', {'x': 0.05, 'y': -1.146, 'sx': None, 'sy': None, 'theta': None, 'order': None}], ['Lcentered-3-0', {'x': 1.246, 'y': -2.342, 'sx': None, 'sy': None, 'theta': None, 'order': None}]]
        for chunks, hier, fixedorder in zip(list_chunks, list_hier, list_fixed_order):
            C = ChunksClass(chunks, hier, fixedorder, task_stroke_labels=list_shapes)
            self.ListChunksClass.append(C)

    def remove_chunks_that_concat_strokes(self):
        """ Remove chunks from ListChunksClass that concatenates strokes.
        i.e, wherever C.Chunks has a chunk with multiple strokes
        e.g., [[1,2], [0]].
        RETURNS:
        - modifies self.ListChunksClass
        """

        def _does_concat(C):
            chunks = C.Chunks
            for c in chunks:
                if len(c)>1:
                    return True
            else:
                return False

        self.ListChunksClass = [C for C in self.ListChunksClass if not _does_concat(C)]

    def find_chunk(self, chunkname, chunkindex=None):
        """ Retunr the single "chunks" out of the list of chunks, based on matching the name 
        and (optionally) the index
        PARAMS;
        - chunkname, indexs Chunk.Name
        - chunkindex, indexes Chunk.Index. by defualt is None, but this allows for multiple indixes
        under a given name/model.
        RETURNS:
        - Chunk
        """
        for C in self.ListChunksClass:
            if C.Name==chunkname and C.Index==chunkindex:
                return C
        print("HERE")
        print(chunkname, chunkindex)
        self.print_summary()
        assert False, "diud not find this chunk"


    def print_summary(self):
        """
        """
        for i, C in enumerate(self.ListChunksClass):
            print("--- Chunk num:", i)
            C.print_summary()

    ########## PERMUTATIONS
    def search_permutations_chunks(self, return_ver="list_of_flattened_chunks", 
        max_perms=10000, return_out_chunkobj=False):
        """ Return unique permutations of each Chunk. If num possible
        perms is less than max_perms, then returns all.
        PARAMS:
        - return_ver, str, how to structure output:
        --- list_of_list_of_chunks, each inner list corresponds to one ChunksClass
        --- list_of_chunks, each inner list is a chunk, i..e, combines chunks acorss
        ChunksClass
        --- list_of_flattened_chunks, each inner list is a flattened chunk.
        - return_out_chunkobj, if True, also returns a list holding the Chunks() object
        aligned with each parse.
        RETURNS:
        - out (structure depends on return_ver). Ensures no reduntant chunks.
        If no parses, returns []
        """
        out = []
        # out_heir_of_chunk = []
        out_chunkobj = []
        for C in self.ListChunksClass:
            # Get all permutations of grounded concrete sequences, given a chunks
            list_chunk_permutations = C.search_permutations_chunks(max_perms=max_perms)

            if return_ver=="list_of_list_of_chunks":
                out.append(list_chunk_permutations)
                # out_heir_of_chunk.append(C.Hier)
                out_chunkobj.append(C)
            elif return_ver=="list_of_chunks":
                for chunk in list_chunk_permutations:
                    out.append(chunk)
                    # out_heir_of_chunk.append(C.Hier)
                    out_chunkobj.append(C)
            elif return_ver=="list_of_flattened_chunks":
                for chunk in list_chunk_permutations:
                    out.append(C._flatten_chunk(chunk))
                    # out_heir_of_chunk.append(C.Hier)
                    out_chunkobj.append(C)
            else:
                print(return_ver)
                assert False

        # print(out)
        # # print(out_heir_of_chunk)
        # print(out_chunkobj)

        # test = [(tuple(out[0]), out_chunkobj[0]) for _ in range(2)]
        # print(test)
        # print(list(set(test)))
        # assert False

        # 1) make sure no redundant ones.
        if return_ver in ["list_of_chunks", "list_of_flattened_chunks"]:
            out = [tuple(o) for o in out]
            if False:
                out = list(set(out))
            else:
                # make sure out and out_chunkobj are yoked.
                tmp = [(a, b) for a, b in zip(out, out_chunkobj)]
                tmp = list(set(tmp))
                out = [x[0] for x in tmp]
                out_chunkobj = [x[1] for x in tmp]
        if out_chunkobj:
            assert len(out)==len(out_chunkobj)
            return out, out_chunkobj
        else:
            return out



class ChunksClass(object):
    """ Represents a single way of chunking a specific task"""
    def __init__(self, chunks, hier, fixed_order=None, task_stroke_labels = None,
        assume_all_strokes_used=True, name=None, flips=None, index=None, colorlist_hier=None):
        """
        PARAMS:
        - chunks, list of list, like [[1 2], [0 3], [5]], meaing chunk 0 uses strokes 1 and 2...
        Can also be [[1 2] 3].
        Can only be 2 levels deep max (i.e., [[1 2] 3])
        Meanning of chunks: use it to concatenate strokes.
        - hier, same type as chunks, but meaning is not to concat strokes, but to structure the ordering
        - fixed_order, dict, which indicate whether teh elements at a given level are aloowed to 
        be reordered, or remain fixed in order. e.g. {0: False, 1: [False, False, False]}, if 
        hier is [[1, 0], [3, 2], [5, 4]],
        means level0 is not fixed, and the items within the 3 chunks are also not fixed.
        If None, then assumes both levels are allows to be reordered.
        - task_stroke_labels, list of ints or str, which are names to give to each task stroke. e.g..
        ["L", "C", ..] for line and circle.
        - assume_all_strokes_used, asserts that each task stroke id is used only once. 
        assumes that all ids are used, but cannot know for sure since not passing in how many
        strokes there were in the task.
        - name, string name (useful if correspnd to a model, for eg)
        - flips, same type as hier, says for each chunkstroke in hier, whether to flip (might not be used).
        - colorlist, list of (3,) asrrays, each a rgb color for a hier chunk, mathces len hier
        """

        self.Chunks = chunks
        self.Hier = hier
        if fixed_order is None:
            # Enter default. assume both levels can be reordered:
            from .chunks import fixed_order_for_this_hier
            self.FixedOrder = fixed_order_for_this_hier(hier, True, True)
        else:
            self.FixedOrder = fixed_order
        self.Labels = task_stroke_labels
        self.Name = name
        self.Index = index # canhave multiple indices within name
        self.Flips = flips

        if colorlist_hier is not None:
            assert len(colorlist_hier)==len(hier)
        self.ColorlistHier = colorlist_hier

        self._preprocess()


    def _preprocess(self, DEBUG=False):
        """ run once
        """
        import numpy as np

        def _clean(input_list):
            assert len(input_list)>0
            output_list = []

            if DEBUG:
                print("---")
                print(input_list)

            for i, c in enumerate(input_list):
                if DEBUG:
                    print("CHUNKSCLASS:", i, c)

                if isinstance(c, int):
                    output_list.append([c])
                elif isinstance(c, np.ndarray) and len(c.shape)==0:
                    output_list.append([int(c)])
                elif isinstance(c, np.ndarray) and len(c)>0:
                    # array([1])
                    # array([1, 2])
                    output_list.append([int(cc) for cc in c])
                # elif isinstance(c, np.ndarray) and len(c)==1:
                #     print([int(cc) for cc in c])
                #     # print(c.shape)
                #     # print(len(c))
                #     assert False    
                elif isinstance(c, np.float64):
                    # THis is like 3.0. check it is int, then convert to list.
                    assert np.floor(c) == c
                    output_list.append([int(c)])
                elif isinstance(c, list):
                    # e..g, [0]
                    assert isinstance(c[0], int)
                    output_list.append([int(cc) for cc in c])
                else:
                    print(c)
                    print(type(c))
                    print(len(c))
                    assert False
            return output_list

        # 1) no ints, make sure all items are lists.
        self.Chunks = _clean(self.Chunks)
        # print("HERE1", self.Hier, len(self.Hier))
        self.Hier = _clean(self.Hier)
        # print("HERE2", self.Hier, len(self.Hier))
        if self.Flips is not None:
            self.Flips = _clean(self.Flips)
            # I used to assert that h and f were same len. but this 
            # doesnt have to be the case, if h is mult strokes...
            # for h,f in zip(self.Hier, self.Flips):
            #     if len(h)!=len(f):
            #         print("hier", h)
            #         print("flips", f)
            #         assert False

        # N task strokes
        chunks_flat = [c for C in self.Chunks for c in C]
        n_taskstrokes = len(chunks_flat)
        assert len(chunks_flat)==len(set(chunks_flat)), "you netered same stroke multipel times across chunks"

        # 2) Check that shapes are correct
        assert len(self.FixedOrder[1]) == len(self.Hier)
        if self.Labels is not None:
            assert len(self.Labels)==n_taskstrokes

        # # Check that each stroke has been assigned, and no assigned strokes outside of range
        # print(self.Hier)
        # print(self.Chunks)



    def _find_group_for_this_ind(self, ind, ver="chunks"):
        """ 
        Generic: Find either the chunk or hier for this ind. see params.
        PARAMS:
        - ind, int, 
        --- if ver=="chunks", then ind is the original stroke id before chunking.
        --- if ver=="hier", then ind is the chunk id after chunking.
        """
        
        if ver=="chunks":
            groups = self.Chunks
        elif ver=="hier":
            groups = self.Hier
        else:
            print(ver)
            assert False

        for i, c in enumerate(groups):
            if ind in c:
                return i
        assert False, f"did not find this ind in this chunks, {ind}, {chunks}"

    def find_hier_for_this_taskstroke(self, indstroke):
        """ Given this original task stroke ind, return its hier chunk. takes into 
        account chunking, then hierarchy.
        RETURNS:
        - index into self.Hier
        e./g., if self.HIer = [[0, 1], [2, 3], [4, 5]], and indstroke = 5, returns 2
        """
        indchunk = self._find_group_for_this_ind(indstroke)
        hierthis = self._find_group_for_this_ind(indchunk, "hier")
        assert indstroke in self.Hier[hierthis]
        return hierthis

    def find_hier_for_list_taskstroke(self, list_indstrokes):
        """ Given list of taskstroke inds, return the sequence of hieraarchical chunks.
        Useful e.g., if list_indstrokes is seuqnece of task inds drawn by subject
        PARAMS:
        - list_indstrokes, list of ints
        RETURNS:
        - list_hier, list of ints.
        """

        return [self.find_hier_for_this_taskstroke(ind) for ind in list_indstrokes]


    def find_hier_for_list_list_taskstroke(self, list_list_indstrokes, only_one_hier_allowed=False):
        """ Similar to find_hier_for_list_taskstroke, but assumes that each beh stroke
        can potentialyl get multiple task strokes. For each beh stroke, returns the list of 
        all hier that this beh stroke got. So this works on on entire trial.
        PARAMS:
        - list_list_indstrokes, list of list of ints. e..g, [[1 2], [0], [0, 1]] means first 
        beh stroke got task strokes 1 and 2, etc.
        - only_one_hier_allowed, then for cases where a single beh stroke matches two task hier chunks,
        then replaces that with np.nan. 
        RETURNS:
        if only_one_hier_allowed==False"
        - list_list_indhier, list of list of ints, e.g., [[1], [0], [0, 1]] mean sfirst beh stroke
        got hierchunk 1, etc. Note: the items within each inner list (e..g, [0,1]) will not be
        in order given by input strokes. e..g, [[1], [0, 2], [2], [0], [1]]
        if only_one_hier_allowed==True:
        - array of ints, where np.nan means this beh stroke did not get just a single hier.
        e.g., array([ 1., nan,  2.,  0.,  1.])
        """

        list_list_indhier =[]
        for list_indstrokes in list_list_indstrokes:
            
            this = self.find_hier_for_list_taskstroke(list_indstrokes)

            # clean up, if any ind hier repeated for the same beh stroke, then combine them into one int
            this = list(set(this))

            list_list_indhier.append(this)


        if only_one_hier_allowed:
            out = np.ndarray(len(list_list_indhier))
            for i, val in enumerate(list_list_indhier):
                if len(val)>1:
                    out[i] = np.nan
                else:
                    out[i] = val[0]
            return out
        else:
            return list_list_indhier

    ########################### GROUDNING/CONVERTING TO CONCRETE SEQUENCES
    def search_permutations_chunks(self, ver="hier", max_perms=1000 ):
        """
        Returns all permutatoins of chunks or hierarchy
        PARAMS:
        - ver, str, either "chunks", or "hier", which one to return permutations of.
        RETURNS:
        - list of chunks-like, each a permutation of the original chunks.
        """
        from .chunks import search_permutations_chunks

        if ver=="chunks":
            assert False, "currently only works for Hier, since Fixed Order is only defined for that. can easlity make one for CHunks"
        elif ver=="hier":
            return search_permutations_chunks(self.Hier, self.FixedOrder, max_perms=max_perms)
        else:
            assert False


    ########################### WORKING WITH LABELS

    ######################## EXTRACTION
    def extract_strokeinds_as(self, how, which="hier"):
        """ WRapper to extract strokes indices for this chunks, in vairous formats
        PARAMS:
        - how, str, in {'chunks', 'flat'}, either as chunks list of list of ints, or flat (list of ints)
        - which, str, in {'hier', 'chunks'}, which one to extract
        RETURNS:
        - list, depends on "how"
        """

        if which=="hier":
            chunks = self.Hier
        elif which=="chunks":
            chunks = self.Chunks
        else:
            print(which)
            assert False

        if how=="chunks":
            return chunks    
        elif how=="flat":
            return [c for ch in chunks for c in ch]
        else:
            assert False

    ############################
    def print_summary(self):
        """ quick printing of chunks, etc
        """
        print("Name:", self.Name)
        print("Index:", self.Index)
        print("Chunks:", self.Chunks)
        print("Hier:" , self.Hier)
        print("Fixed Order:", self.FixedOrder)
        print("list labels/shapes:", self.Labels)
        print("colorlist_hier:", self.ColorlistHier)
        if self.Flips is not None:
            print("Flips:", self.Flips)
        
    def _flatten_chunk(self, chunk):
        """ COnvert chunk to list of ints
        PARAMS;
        - chunk, list of list of ints, e.g, [[1,2], [4], [3]]
        RETUNRS:
        - chunk_flat, list of ints, e.g, [1,2,4,3]
        """
        return [xx for x in chunk for xx in x]


##### FAKE CHUNKS
def generate_dummy_chunksclass():
    """ Useful for debugging
    """

    # Different kinds of chunks

    # hier = [1,2,3]
    # fixed_order = {0:False, 1:[False, False, False]}

    # hier = [[1,2,3]]
    # fixed_order = {0:False, 1:[False]}

    # hier = [[1,2], [3]]
    # fixed_order = {0:False, 1:[False, False]}

    C = ChunksClass(chunks, hier, fixed_order)
    C.print_summary()
    display(C.search_permutations_chunks())

    return C

    