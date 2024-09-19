

class ParserStroke(object):
    """
    Strokes, for use with Parser, here saves as sequence of edges in Parser skeleton graph.
    Similar to StrokeWalker in bpl, but make new version here so can flexibly assign list_ei and 
    list_ni, and use other methods.
    """

    def __init__(self):
        """
        """

        self.Params = {}
        # self.Flipped = False # keep track
        self.list_ei = None
        self.list_ni = None
        self.EdgesDirected = None

    def input_data(self, list_ni, list_ei):
        """ 
    
        - list_ni, list of nodes (integers)
        - list_ei, list of tuples (edges) each tuple (n1, n2, i) where n1 and n2 are the
        nodes (undirected so order doesnt matter) and i is indicator in case there are multiple
        edges for this pari of nodes {0, 1, ...}
        """

        self.list_ei = list_ei
        self.list_ni = list_ni
        self._check_data_consistency()
        self.convert_to_directed_edges()

    def input_data_directed(self, list_eidir):
        """ input directed edges.
        - list_eidir is list of tuples, each a directed edge, e.g.
        [(0, 1, 0), (1,4,0)].
        Will check consistency of the ordering (so that edges chain up)
        Will convert to list of nodes as well.
        """

        self.EdgesDirected = list_eidir
        self.list_ni, self.list_ei = self.convert_from_directed_edges(self.EdgesDirected)
        try:
            self._check_data_consistency()
            self._check_lists_match()
        except Exception as err:
            print("**FAILING consistent check. Printing elemtsn:")
            self.print_elements()
            raise

    def convert_from_directed_edges(self, list_eidir):
        """ convert from list of directed edges to list_ni and list_ei
        """
        list_ni = []
        for i, e in enumerate(self.EdgesDirected):
            if i==0:
                list_ni.append(e[0])
            list_ni.append(e[1])

        list_ei = self.EdgesDirected
        return list_ni, list_ei

    def extract_list_of_directed_edges(self):
        """ returns as list of directed edges (i.e., a "path")
        """
        return [ed for ed in self.EdgesDirected]


    def _check_data_consistency(self):
        """ make sure edges and nodes are consistent in  input data
        """
        list_ni = self.list_ni
        list_ei = self.list_ei

        if len(list_ei)==0 or len(list_ni)==0:
            print("No data,,,")
            self.print_elements()
            assert False

        assert isinstance(list_ni, list) and isinstance(list_ni[0], int)
        assert isinstance(list_ei, list) and isinstance(list_ei[0], tuple) and isinstance(list_ei[0][0], int) and all([len(x)==3 for x in list_ei])
        assert len(list_ni)==len(list_ei)+1

        for i in range(len(list_ni)-1):
            n1 = list_ni[i]
            n2 = list_ni[i+1]
            assert set((n1, n2)) == set(list_ei[i][:2]), "these nodes and edges dont match..."

        if self.EdgesDirected is not None:
            for i in range(len(self.EdgesDirected)-1):
                e1 = self.EdgesDirected[i]
                e2 = self.EdgesDirected[i+1]
                assert e1[1]==e2[0], "not cahined up"

            

    def _check_lists_match(self):
        """
        Check that self.EdgesDirected is consistent with input list_ni and list_ei
        """
        list_ni = []
        for i, e in enumerate(self.EdgesDirected):
            if i==0:
                list_ni.append(e[0])
            list_ni.append(e[1])
        assert list_ni==self.list_ni

        for e1, e2 in zip(self.EdgesDirected, self.list_ei):
            assert self._edges_are_same(e1, e2)

    def _edges_are_same(self, e1, e2):
        """ returns True if edges are the same (can be in oppopsite dir)"""
        if set(e1[:2])==set(e2[:2]) and e1[2]==e2[2]:
            return True
        else:
            return False

    def do_flip(self):
        """ switch state, whether stroke is flipped.
        """
        # self.Flipped = not self.Flipped
        self.list_ei = self.list_ei[::-1]
        self.list_ni = self.list_ni[::-1]
        self.convert_to_directed_edges()


    def copy(self):
        """ return a copy of self
        NOTE: probably want to use copy_object, unless you plan to modify the lists
        themselves.
        """
        # from copy import copy

        # assert self.list_ei is not None, "this is empty..."

        PS = ParserStroke()
        if self.list_ei is not None:
            # PS.input_data(self.list_ni.copy(), self.list_ei.copy())
            PS.input_data(self.list_ni.copy(), self.list_ei.copy())
        # PS.Flipped = self.Flipped
        return PS


    def copy_object(self):
        """ return a copy of self, but keeps references to list elemtns.
        This useful if many many permutations of lists of ParserStrokes, and
        each one want to have different flipping of the stroke for exmaple. This
        makes sure when flip one ParserStroke, doesnt flip another. 

        """
        # from copy import copy

        PS = ParserStroke()
        if self.list_ei is not None:
            PS.input_data(self.list_ni, self.list_ei)
        # PS.Flipped = self.Flipped
        return PS


    def unique_path_id(self, invariant=False):
        """ wrapper, will return unique id, first decide what kind of invariance automaticlaly, given
        the current path kind
        """

        if invariant:
            kind = self.get_stroke_kind()
            if kind=="notloop":
                invariance_kind="dir" # direction
            elif kind=="loop":
                invariance_kind="loop" # direction
            else:
                print(kind)
                assert False
            return self._unique_path_id(invariance_kind)
        else:
            return self._unique_path_id()


    def _unique_path_id(self, invariance_kind=None):
        """ list, which is unique id for this stroke, taking into account for
        sequenc eof nodes and the edge between them (i.e., since is muligraph, the 
        nodes alone dont uniquely define edge). ie directed edge
        - invariant_to_dir, then will be same id whether flipped or not.
        OUT:
        tuple of ints, a unique "hash" defining a path. see below.
        """

        def _hash(li):
            """ unique code for path given by directed edges. if flipped,
            is equal to the code for the path, flipped.
            e..g, if dir edges: [(7, 4, 1), (4, 2, 0), (2, 7, 0), (7, 3, 0)]
            out =  (7, 1, 4, 4, 0, 2, 2, 0, 7, 7, 0, 3)
            """
            out = []
            for e in li:
                out.append(e[0])
                out.append(e[2])
                out.append(e[1])
            return tuple(out)

        kind = self.get_stroke_kind()
        
        if invariance_kind=="dir":
            # reutrn either edges or it flipped, depending on which is increasing values
            # all that matters is that returns the same thing for both directions.

            assert kind =="notloop", "if loop, then deal with this using circle invariance"
            id_ = _hash(self.EdgesDirected)

            # Return its unique, direction invariance.
            if id_[-1]>id_[0]:
                # then dont flip
                return id_
            elif id_[-1]<id_[0]:
                return id_[::-1]
            else:
                assert False

        elif invariance_kind=="loop":
            # allows any direction, and starting from any node along this loop.

            assert kind =="loop"

            # 1) get both directions
            # copy and flip
            PS = self.copy_object()
            PS.do_flip()

            list_edir1 = self.EdgesDirected
            list_edir2 = PS.EdgesDirected

            # 2) get all circular permutations
            from more_itertools import circular_shifts
            tmp = []
            for lthis in [list_edir1, list_edir2]:
                tmp.extend([list(x) for x in circular_shifts(lthis)])

            # 3) convert each one to hash
            tmp = [_hash(li) for li in tmp]

            return sorted(tmp)[0]

        else:
            # then keep this direction exactly
            return _hash(self.EdgesDirected)


    def unique_path_id_old(self, invariant_to_dir=False):
        """ list, which is unique id for this stroke, taking into account for
        sequenc eof nodes and the edge between them (i.e., since is muligraph, the 
        nodes alone dont uniquely define edge)
        - invariant_to_dir, then will be same id whether flipped or not.
        OUT:
        - tuple, including both int and str
        e.g.,:
        nodes: [3, 4, 7, 3]
        edges: [(3, 4, 0), (4, 7, 0), (7, 3, 0)]
        out = [3, '0', 4, '0', 7, '0', 3]
        # NOTE Old version, used undirect edges. ignore.
        """

        assert False, "use new version"

        list_ni = self.list_ni
        list_ei = self.list_ei

        # if invariant_to_dir:
        #     # decide whether to flip, always incresing order.


        out = []
        for i in range(len(list_ni)-1):
            
            n1 = list_ni[i]
            n2 = list_ni[i+1]
            
            if i==0:
                out.append(n1)
            out.append(str(list_ei[i][2])) # indicator for edge
            out.append(n2)
        return tuple(out)

    def convert_to_directed_edges(self):
        """ consider directed edges as best way to summarize the path.
        combines both the list of nodes, and the edge id.
        NOTE:
        - if already converted, then this actualyl just checks that it is expected
        based on current list_ni and list_ei.
        """

        list_eidir = []

        for i, ei in enumerate(self.list_ei):

            list_eidir.append((self.list_ni[i], self.list_ni[i+1], ei[2]))

        self.EdgesDirected = list_eidir

    ############################
    def get_stroke_kind(self):
        if self.list_ni[-1] == self.list_ni[0]:
            kind = "loop"
        else:
            kind = "notloop"
        return kind


    ############################


    def print_elements(self, verbose=True):
        if verbose:
            print(f"nodes: {self.list_ni}")
            print(f"edges: {self.list_ei}")

        print(f"dir edges: {self.EdgesDirected}")






