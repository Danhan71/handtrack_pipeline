"""
7/9/21 - Good for parsing directly from strokes (task image)
NOTES:
- each parse will be in different memory loation, even if it looks the same.
"""

import numpy as np
import matplotlib.pyplot as plt

class Parser(object):
    """
    """

    def __init__(self):
        """
        """
        self.Params = {}
        self.Finalized=False # to ensure that after process strokes, dont try to do more stuff.
        self.Parses = []
        self.ParsesBase = []
        self._GraphLocked = False # will not allow graphmods after locked (e.g., already done parsing)

    def input_data(self, data, kind="strokes"):
        """ 
        General-purpose for inputing a single datapoint. 
        INPUT:
        - data, flexible type, depending on kind.
        "strokes", then is list of np arrays.
        """

        self.Params["data_kind"] = kind

        if kind=="strokes":
            self.Strokes = data
            self.Strokes = [s[:,:2] for s in self.Strokes]

        self.Translation = {}
        self.Skeleton = None

    # def graph_and_parse_pipeline(self, N=1000, nwalk_det = 10, max_nstroke=400, max_nwalk=100, 
    #         plot=False, quick=False, graph_mods = None, graph_mods_params = None,
    #         direction_invariant=False, strokes_invariant=False):

    #     self.make_graph_pipeline(plot=plot)
    #     self.parse_pipeline(N=1000, nwalk_det = 10, max_nstroke=400, max_nwalk=100, 
    #         plot=False, quick=False, graph_mods = None, graph_mods_params = None,
    #         direction_invariant=False, strokes_invariant=False)

    def make_graph_pipeline(self, plot=False, graph_mods = None, graph_mods_params = None):
        """
        didmod_all, True if actually modified graph
        """

        # 1) Recenter strokes to positive coordinates
        self.strokes_translate_positive()
        self.strokes_interpolate_fine(plot=plot)
        self.skeletonize_strokes()
        self.graph_construct()

        print(graph_mods)
        # print(graph_mods_params)
        # self.plot_graph()

        # Graph mods
        didmod_all = False
        if graph_mods is not None:
            for mod, params in zip(graph_mods, graph_mods_params):
                print("** APPLYING THIS GRAPHMOD: ", mod)
                didmod = None
                if mod=="merge":
                    didmod = self.graphmod_merge_nodes_auto(**params) 
                elif mod=="splitclose":
                    didmod = self.graphmod_split_edges_auto(**params)
                elif mod=="strokes_ends":
                    # endpoints for each stroke should match a node.
                    # params["strokes"] can be strokes, or string attribute (e.g., "StrokesInterp")
                    if isinstance(params["strokes"], str):
                        params["strokes"] = getattr(self, params["strokes"])
                    self.graphmod_add_nodes_strokes_endpoints(**params)
                    didmod = False # This never modifies the pts.
                elif mod == "merge_close_edges":
                    didmod = self.graphmod_merge_close_edges_auto(**params)
                elif mod=="loops_floating":
                    didmod = self.graphmod_auto_loops_floating_only_one_node()
                elif mod=="loops_cleanup":
                    didmod = self.graphmod_auto_loops_cleanup_excess_nodes()
                elif mod=="loops_add":
                    didmod = self.graphmod_auto_loops_add_nodes(**params)
                else:
                    print(mod)
                    assert False
                print("** DONE: ")
                assert didmod is not None
                didmod_all = didmod_all or didmod

        #     self.plot_graph()
        # print(self.Graph.edges)
        # assert False

        if plot:
            self.plot_skeleton()
            self.plot_graph()

        return didmod_all

        # print(mod, params)
        # self.plot_graph()
        # assert False

    def parse_pipeline(self, N=1000, nwalk_det = 10, max_nstroke=400, max_nwalk=100, 
            plot=False, quick=False, stroke_order_doesnt_matter=False, direction_within_stroke_doesnt_matter=False):
        """
        Full pipeline to get unique parses
        - N, how many. might not be exact since will remove reduntant parses
        - graph_mods, list of mods (strings) in order of application.
        - graph_mod_params, list of dicts, which are kwargs for each mod opertaion./
        """

        # self.make_graph_pipeline(plot=plot)

        # 1) Recenter strokes to positive coordinates
        # self.strokes_translate_positive()
        # self.strokes_interpolate_fine(plot=plot)
        # self.skeletonize_strokes()
        # self.graph_construct()
        
        if quick:
            N=500 # this is usually quick, but still reasonbale
            nwalk_det = 5
            max_nstroke=80
            max_nwalk=20

        # Walk on graph
        # self.walk_extract_parses(10,50,50)
        self.walk_extract_parses(nwalk_det=nwalk_det, 
            max_nstroke=max_nstroke, max_nwalk=max_nwalk)
        self.ParsesParams["N"] = N

        # Remove redundant
        self.parses_remove_redundant()

        # How many permutations to get?
        nnow = len(self.Parses)
        neach = int(np.ceil(N/nnow))
        print("Got this many base parses:", nnow)
        print("Getting this many permutations of each base parses:", neach)

        ### get all permutations
        # self.parses_get_all_permutations(n_each = 2)
        self.parses_get_all_permutations(n_each = neach, direction_within_stroke_doesnt_matter=direction_within_stroke_doesnt_matter)

        ### Remove redundant permutations
        self.parses_remove_redundant(stroke_order_doesnt_matter=stroke_order_doesnt_matter, 
            direction_within_stroke_doesnt_matter=direction_within_stroke_doesnt_matter)

        # Cleanup parses

        # Keep top K parses
        # TODO

        if plot:
            self.summarize_parses()

        # Lock the graph
        self._GraphLocked = True

    def _translate(self, pts):
        """ apply saved transfomration to any pts,
        where pts is Nx2 np array, or (2,)
        """
        if not pts.shape==(2,):
            assert pts.shape[1]==2
            assert len(pts.shape)==2

        return pts + self.Translation["xy"]

    def strokes_translate_forward(self, strokes):
        """ applies original transform to new strokes.
        doesnt modify self, just returns copy of strokes, mod
        """
        from pythonlib.tools.stroketools import translateStrokes
        return translateStrokes(strokes, self.Translation["xy"])

    def strokes_translate_back(self, strokes):
        """ applies transformation back to original coordiantes,
        for tihs strokes
        """
        from pythonlib.tools.stroketools import translateStrokes
        return translateStrokes(strokes, -self.Translation["xy"])

    def strokes_translate_positive(self):
        """ translate strokes so that all value are ositive. they can then act as 
        indices for constructing skeleton image
        RETURNS:
        - modifies self.Strokes
        - saves current state in self.Translation
        """
        from pythonlib.drawmodel.image import convStrokecoordToImagecoord, get_sketchpad_edges_from_strokes, getSketchpadEdges
        from pythonlib.tools.stroketools import translateStrokes

        if len(self.Translation)>0:
            if self.Translation["is_translated"]:
                assert False, "it is currently translated"

        # Recenter, so that all indices are positive
        edges = get_sketchpad_edges_from_strokes([self.Strokes])

        # translate so that x and y start at index k
        k = 4
        xy = [-edges[0,0] + k, -edges[0,1] + k]
        self.Strokes = translateStrokes(self.Strokes, xy)

        # sanity check
        edges_new = get_sketchpad_edges_from_strokes([self.Strokes])
        assert (edges_new>0).all(), "did not translate enough?"

        # save current state
        self.Translation["xy"] = np.array(xy)
        self.Translation["is_translated"] = True


    def strokes_interpolate_fine(self, plot=False):
        """ interpolate strokes so that does not skip any pixel - spatial interpolation
        RETURNS:
        - new field: self.StrokesInterp. does not modify self.Strokes
        """
        from pythonlib.tools.stroketools import strokesInterpolate2

        interval = 0.8
        self.StrokesInterp = strokesInterpolate2(self.Strokes, 
            N = ["interval", interval], base="space", plot_outcome=plot)


    def skeletonize_strokes(self):
        """ convert strokes to image, and skeletonize
        RETURNS:
        - puts ske into self.Skeleton
        """
        from skimage import morphology
        strokes_interp = self.StrokesInterp

        # Find good size of image that captures entire stroke.
        xlim = int(np.max(np.concatenate(strokes_interp)[:,0]))
        ylim = int(np.max(np.concatenate(strokes_interp)[:,1]))

        # Initiate image
        image = np.zeros((xlim+5, ylim+5), dtype=np.int16)
        tmp = np.concatenate(strokes_interp)
        for pt in tmp:
            xind = int(np.round(pt[0]))
            yind = int(np.round(pt[1]))
            image[xind, yind] = 1

        # Skeletonize the image
        ske = image
        # ske = morphology.remove_small_objects(ske)
        # ske = morphology.remove_small_holes(ske)
        ske = morphology.skeletonize(ske).astype(np.uint16)

        self.Skeleton = ske

    def graph_construct(self):
        """ make skeleton graph using self.Skeleton
        RETURNS:
        - self.Graph hold sthe networkx graph.
        """
        from sknw import build_sknw
        self.Graph = build_sknw(self.Skeleton, multi=True)

        self._graph_remove_zerolength_edge()


    ################# TOOLS, WROKING WOTH PARSERSTROKE
    def _walkerstroke_to_parserstroke(self, ws):
        """ converts a single WalkerStroke instance to a 
        single ParserStroke instance"""
        from .parser_stroke import ParserStroke
        PS = ParserStroke()
        if len(ws.list_ni)<2 or len(ws.list_ei)<1:
            self.plot_graph()
            print(ws.list_ni)
            print(ws.list_ei)
            assert False, "why?"
        PS.input_data(ws.list_ni, ws.list_ei)
        return PS

    def _listofpaths_to_parserstroke(self, path):
        """
        help convert list of paths to a single stroke
        - path, list of edges (tuples)
        """
        from .parser_stroke import ParserStroke
        PS = ParserStroke()
        PS.input_data_directed(path)
        return PS


    def _strokes_convert_to_parsestrokes(self, is_base_parse):
        """ for each parse, represent strokes each as a ParserStroke object,
        instead of WalkerStroke (bpl) which is default.
        - Usually run this right after extract parses)
        """

        ParsesList = self.ParsesBase if is_base_parse else self.Parses

        for i in range(len(ParsesList)):
            if "list_walkers" in ParsesList[i].keys():
                list_ws = ParsesList[i]["list_walkers"]
                list_ps = [self._walkerstroke_to_parserstroke(w) for w in list_ws]
                ParsesList[i]["list_ps"] = list_ps
                del ParsesList[i]["list_walkers"]


    ########## WALKING ON GRAPH (GET PARSES)
    def walk_extract_parses(self, nwalk_det = 5, max_nstroke=250, max_nwalk=100):
        """ Extract parses from skeleton graph
        NOTE:
        - uses code by Reuben Feinmam for walking on graph.
        """
        from pybpl.bottomup.initialize import RandomWalker

        # initialize random walker and empty parse list
        walker = RandomWalker(self.Graph, self.Skeleton)
        
        # Keep track of all parses
        walker_list = []
        parses_strokes = []
        parses_nodes = []

        # add deterministic minimum-angle walks
        for i in range(nwalk_det):
            parses_strokes.append(walker.det_walk())
            # walker_list.append(walker)
            parses_nodes.append(walker.list_ws)

        # sample random walks until we reach capacity
        nwalk = len(parses_strokes)
        nstroke = sum([len(parse) for parse in parses_strokes])
        ct = 0
        while nstroke < max_nstroke and nwalk < max_nwalk:
            if ct%10==0:
                print(ct, nstroke, nwalk, "target:", max_nstroke, max_nwalk)
            walk = walker.sample()
            parses_strokes.append(walk)
            parses_nodes.append(walker.list_ws)

            nwalk += 1
            nstroke += len(walk)
            ct+=1

        # save
        for s, n in zip(parses_strokes, parses_nodes):
            self.Parses.append(
                {
                "strokes":s,
                "list_walkers": n,
                "permutation_of":None
                })
        self.ParsesParams = {
            "nwalk_det":nwalk_det,
            "max_nstroke":max_nstroke,
            "max_nwalk":max_nwalk
        }

        # Convert strokes to represnetation using ParserStrokes
        self._strokes_convert_to_parsestrokes()
            
    def _check_edges_identical(self, edge1, edge2):
        if edge1[2]==edge2[2] and set(edge1[:2])==set(edge2[:2]):
            return True
        else:
            return False

    ############# MANUALLY ENTERING PARSES
    def check_if_parse_exists(self, parse, stroke_order_doesnt_matter=True,
        direction_within_stroke_doesnt_matter=True, is_base_parse=False):
        """ Check if this parse already exists in self.Parses,
        IN:
        - parse, a dict holding all parse info. like self.Parses[0]
        OUT:
        exists, bool.
        ind, int if exists. otherwise None 
        NOTE:
        - will stop at first encounter of a same. assumes that you dont have
        redundant parses. 
        """
        
        # get input parse in correct format
        list_p = parse["list_ps"]
        assert len(list_p)>0

        if is_base_parse:
            ParsesList = self.ParsesBase
        else:
            ParsesList = self.Parses

        for i in range(len(ParsesList)):
            list_p_this = self.extract_parses_wrapper(i, is_base_parse=is_base_parse)
            if self._check_parses_is_identical(list_p, list_p_this, 
                stroke_order_doesnt_matter, 
                direction_within_stroke_doesnt_matter):
                # Exists.
                # print(list_p)
                # print(list_p_this)
                # print(is_base_parse, stroke_order_doesnt_matter, direction_within_stroke_doesnt_matter)
                return True, i  
        # Doesnt exist
        return False, None

    def update_parse_with_input_parse(self, parse, keyvals_update=None,
            stroke_order_doesnt_matter=True,
            direction_within_stroke_doesnt_matter=True,
            append_keyvals=False, is_base_parse=False
            ):
        """ Looks whether there exists parse already, identical to input. if so,
        then updates the existing parse with key:val in keyvals_update
        IN:
        - parse, dict
        - keyvals_update, dict, all will overwrite existing parse, if find.
        - append_keyvals, then does not overwrite existing key, but appends to value.
        the current value MUST Be a list (input value should nto be list). if key doesnt exist, then starts a new list.
        --- False, then overwrites it with whatever is input val (doesnt hve to be ;list)
        OUT:
        - success, bool, indiicating if found
        - ind, 
        --- int, location of parse, if found
        --- None, if did not find.
        """

        # check if parse exists 

        if keyvals_update is None:
            keyvals_update = {}
        exists, ind = self.check_if_parse_exists(parse, stroke_order_doesnt_matter,
            direction_within_stroke_doesnt_matter, is_base_parse=is_base_parse)
        

        if exists: 
            self.update_existing_parse(ind, keyvals_update, append_keyvals, is_base_parse)
            # # Then update
            # parsedict = self.extract_parses_wrapper(ind, "dict")
            # for k, v in keyvals_update.items():
            #     if append_keyvals:
            #         if k in parsedict.keys():
            #             parsedict[k].append(v)
            #         else:
            #             parsedict[k] = [v]
            #     else:
            #         # Replace
            #         parsedict[k] = v
            return True, ind
        else:
            return False, None

    def update_existing_parse(self, ind, keyvals_update=None, append_keyvals=False,
            is_base_parse=False):
        """
        Helper to update an existing parse in smart way
        IN:
        - keyvals_update, dict which will enter into parsedict. will overwrite if exists.
        unless use append_keyvals
        --- {k:v} 
        - append_keyvals, then will assume values are lists, and iwll append the input item if exists
        or start a list if doesnts.
        --- enter v the item, not [v]
        OUT:
        - parsedict, updated
        """
        # Then update
        # if len(self.Parses[64]["perm_of_list"])>0:
        #     print(ind, parsedict)
        #     print(self.Parses[64])
        #     assert False
        if keyvals_update is None:
            keyvals_update = {}
        parsedict = self.extract_parses_wrapper(ind, "dict", is_base_parse=is_base_parse)
        # if len(self.Parses[64]["perm_of_list"])>0:
        #     print(ind, parsedict)
        #     print(self.Parses[64])
        #     assert False
        # for i,p in enumerate(self.Parses):
        #     print(i, len(p["perm_of_list"]))
        for k, v in keyvals_update.items():
            if append_keyvals:
                if k in parsedict.keys():
                    parsedict[k].append(v)
                else:
                    parsedict[k] = [v]
            else:
                # Replace
                parsedict[k] = v
        # for i,p in enumerate(self.Parses):
        #     print(i, len(p["perm_of_list"]))
        # if len(self.Parses[64]["perm_of_list"])>0:
        #     print(ind, parsedict)
        #     print(self.Parses[64])
        #     assert False
        return parsedict



    def wrapper_input_parse(self, parse, ver, note="", apply_transform=True,
        require_stroke_ends_on_nodes=True, params=None, is_base_parse=False):
        """
        [GOOD USE THIS] Manually enter a new parse in where input format/type is flexible.
        INPUT:
        - list_of_paths,
        --- list of list of edges (each path is a list of directed edges.)
        e.g., 
        - list_of_paths = [
        [(0,1,0), (1,2,0)],
        [(2,3,0)]
        ]
        gives two paths (strokes).
        --- dict, holding list_ps.
        --- strokes, list of Nx2 arrays
        """
        if params is None:
            params = {}


        if ver=="list_of_paths":
            self._manually_input_parse(parse, use_all_edges=True, note=note,
                is_base_parse=is_base_parse)
        elif ver=="dict":
            if is_base_parse:
                self.ParsesBase.append(parse)
                print("Added new BASE parse, ind:", len(self.ParsesBase)-1)
            else:
                self.Parses.append(parse)
                print("Added new parse, ind:", len(self.Parses)-1)
        elif ver=="strokes":
            # assume these are untransformed affine.
            self._manually_input_parse_from_strokes(parse, apply_transform=apply_transform,
                require_stroke_ends_on_nodes=require_stroke_ends_on_nodes, note=note,
                is_base_parse=is_base_parse)
        elif ver=="strokes_plus_chunkinfo":
            # enter each segment in strokes, but also enter how to chunk.
            # will automaitlcaly chunk into a parse.
            from pythonlib.drawmodel.tasks import chunks2parses, chunk_strokes, flatten_hier

            strokes = parse
            chunks = params["chunks"] # refers to strokes input. will chunk strokes before inseting
            hier = params["hier"] # refers to strokes AFTER chunkign
            fixed_order = params["fixed_order"] # dict[0] = [True]; dict[1] = [True, False], means whether allowed to 
            # shuffle on first level (0) and second (1)

            # if not self._hier_is_flat(chunks):
            #     assert False, "not sure if always succeeds in finding paths for each new chunk"

            assert sorted(flatten_hier(hier))==list(range(len(chunks))), "hierarchy must account for all strokes AFTER chunking"
            assert len(fixed_order[1])==len(hier), "need to do this"

            # chunk strokes if needed
            strokes_chunked = chunk_strokes(strokes, chunks, reorder=False)

            # Input parse
            parse_dict = self._manually_input_parse_from_strokes(strokes_chunked, 
                apply_transform=apply_transform, require_stroke_ends_on_nodes=False, note=note,
                return_parse_dict=True, is_base_parse=is_base_parse)


            # add metadat
            list_keys = ["chunks", "hier", "fixed_order", "objects_before_chunking", "rule"]
            for k in list_keys:
                assert k not in parse_dict
                parse_dict[k] = params[k]

            # print(parse)
            # print(strokes_chunked)
            # for k, v in parse_dict.items():
            #     print(k, v)

            # assert False, "chcek"

        else:
            assert False

        # if already finalized, then need to finalize this last parse
        if is_base_parse is False:
            if self.Finalized:
                indthis = len(self.Parses)-1
                self.finalize(force_just_this_ind=indthis)


    def _manually_input_parse(self, list_of_paths, use_all_edges=True, note="",
        is_base_parse=False):
        """
        INPUT:
        - list_of_paths,
        --- list of list of edges (each path is a list of directed edges.)
        e.g., 
        - list_of_paths = [
        [(0,1,0), (1,2,0)],
        [(2,3,0)]
        ]
        gives two paths (strokes).
        NOTE:
        - will check for correctness of each path
        - will check that all edges are used up, unless override.
        """ 

        # All paths need to be directed
        for i, path in enumerate(list_of_paths):
            list_of_paths[i] = self.convert_list_edges_to_directed(path)

        if use_all_edges:
            # get set of all edges
            set_of_edges = set([tuple(edge) for path in list_of_paths for edge in path])
            for ed in self.Graph.edges:
                if not any([self._check_edges_identical(ed, ed1) for ed1 in set_of_edges]):
                    print("edges inputed", set_of_edges)
                    print("edges in graph", self.Graph.edges)
                    assert False, "did not use all edges"

        list_ps = [self._listofpaths_to_parserstroke(path) for path in list_of_paths]


        newparse = {
            "strokes":None,
            "list_ps":list_ps,
            "permutation_of":None,
            "manual":True,
            "note":note
        }

        if is_base_parse:
            self.ParsesBase.append(newparse)
            print("Added new BASE parse, ind:", len(self.ParsesBase)-1)
        else:
            self.Parses.append(newparse)
            print("Added new parse, ind:", len(self.Parses)-1)

        return newparse


    def _manually_input_parse_from_strokes(self, strokes, apply_transform=True,
            require_stroke_ends_on_nodes=True, note="",
            is_base_parse=False, return_parse_dict=False):
        """ wrapper, to auto find list of nodes, and input, for this storkes.
        strokes must match the strokes coordinates that make up the graph.
        - will fail if doesnt find a complete set of edges.
        INPUT:
        - apply_transform, then first transfomrs strokes (spatial) using same params
        as did for the initial strokes entry.
        """
        from pythonlib.tools.graphtools import path_through_list_nodes, path_between_nodepair

        if apply_transform:
            strokes = self.strokes_translate_forward(strokes)

        if False:
            # Old version, problem is if there are >1 path between endpoints, then this fails.
            # Find endpoint nodes for each stroke
            list_of_nodes = self.map_strokes_to_nodelists(strokes)
            # Conver these lenght 2 nodes to lists of directed edges
            list_of_paths = []
            for nodepair in list_of_nodes:
                path = path_between_nodepair(self.Graph, nodepair)
                list_of_paths.append(path)

        # else:
        #     list_of_nodes_traj = self.map_strokes_to_nodelists_entiretraj(strokes)
        #     list_of_paths = []
        #     for nodes in list_of_nodes_traj:
        #         path = path_through_list_nodes(self.Graph, nodes)
        #         list_of_paths.append(path)

        #     # Conver these lenght 2 nodes to lists of directed edges
        #     print(list_of_paths)
        #     assert FAlse
        else:
            if require_stroke_ends_on_nodes:
                # Origianl, strict version. basucally rquires the graph to be matched up
                # identicalyl to the strokes
                list_of_paths = self.map_strokes_to_edgelist(strokes, 
                    must_use_all_edges=True, no_multiple_uses_of_edge=True)
            else:
                # Looser version, e.g., if made graph mods, but still wabnt to add strokes parses.
                list_of_paths = self.map_strokes_to_edgelist(strokes, thresh=20, 
                    assert_endpoints_on_nodes=False, only_take_edges_for_active_node=False,
                    must_use_all_edges=True, no_multiple_uses_of_edge=True)

        # Inset this as a new parse
        #self.plot_graph()
        newparse = self._manually_input_parse(list_of_paths, note=note, is_base_parse=is_base_parse)

        if return_parse_dict:
            return newparse


    ################### DO THINGS WITH PARSES [INDIV]
    def parses_to_strokes(self, ind, is_base_parse=False):
        """ extract strokes for this parse
        NOTE: automaticlaly check consistency, since this uses
        list_ni and list_ei, but other things here usual;yl use edges dir
        """
        return self._parses_to_walker(ind, is_base_parse=is_base_parse).S


    def _parses_to_walker(self, ind, is_base_parse=False):
        """ returns RandomWalker object for this parse (ind)
        """
        from pybpl.bottomup.initialize.random_walker import RandomWalker
        R = RandomWalker(self.Graph, self.Skeleton)

        if is_base_parse:
            ParsesList = self.ParsesBase
        else:
            ParsesList = self.Parses

        if False:
            R.list_ws = self.Parses[ind]["list_walkers"]
        else:
            # check consistnetcy
            [w._check_lists_match() for w in ParsesList[ind]["list_ps"]]
            # good.
            R.list_ws = [w for w in ParsesList[ind]["list_ps"]]
        return R


    ################### DO THINGS WITH PARSES [BATCH]
    def score_parses(self):
        """ quick and dirty to score parses, then can extract the top K parses.
        """
        assert False, "not done, probably just ignore this."
        from pythonlib.drawmodel.parsing import score_function
        origin = D.Dat["origin"][0]
        origin = P._translate(origin)
        score_ver = "travel_from_orig" # better, since differentiates 2 tasks thjat are just flipped (and so will not throw one of them out)
        score_norm = "negative"
        # score_fn = lambda parses: score_function(parses, ver=score_ver, 
        #                                          normalization=score_norm, use_torch=True, origin=origin)

        parses = self.extract_all_parses_as_list()
        scores = score_function(parses, ver=score_ver, normalization=score_norm, use_torch=False, origin=origin)


    def parses_fill_in_all_strokes(self):
        """ if a parse is missing strokes, then converts to strokes and saves
        in self.Parses[ind]["strokes"]
        - is smart about not runing if strokes already exists.
        """
        for i in range(len(self.Parses)):
            if self.Parses[i]["strokes"] is None:
                self.Parses[i]["strokes"] = self.parses_to_strokes(i)
        if hasattr(self, "ParsesBase"):
            for i in range(len(self.ParsesBase)):
                if self.ParsesBase[i]["strokes"] is None:
                    self.ParsesBase[i]["strokes"] = self.parses_to_strokes(i, is_base_parse=True)


    def parses_remove_redundant(self, stroke_order_doesnt_matter=True,
        direction_within_stroke_doesnt_matter=True):
        """ Finds parses that are the same and removes.
        by default, Same is defined as invariant to (1) stroke order,
        (2) within stroke direction and (3) for circle, circle permutation
        OUTPUT:
        - modifies self.Parses
        NOTE:
        - removes the parse with the later ind, always.
        """

        def _ind_already_will_remove(ind, matches):
            """ returns True if this ind is already s part of a value for
            any key in matches. this means it will be removed.
            """
            for k, v in matches.items():
                if ind in v:
                    return True
            return False



        # Find matches, return as dict[ind1] = [list of inds idential]
        # ind1 is not present unless list is len >0
        matches = {}
        nparses = len(self.Parses)
        for i in range(nparses):
            # if i is already scheduled to be removed, then just skip

            if _ind_already_will_remove(i, matches):
                print("SKIPPING, since will already remove", i)
                continue

            for ii in range(i+1, nparses):

                # compare parses
                # p1 = [p.list_ni for p in parses_tracker[i]]
                # p2 = [p.list_ni for p in parses_tracker[ii]]
                
                if self.check_parses_is_identical(i, ii, stroke_order_doesnt_matter=stroke_order_doesnt_matter,
                    direction_within_stroke_doesnt_matter=direction_within_stroke_doesnt_matter):
                    if i in matches:
                        matches[i].append(ii)
                    else:
                        matches[i] = [ii]
        #             matches.append((i,ii))
                    if False:
                        print("identical:")
                        print(p1)
                        print(p2)
        #         else:
        #             print("diff")
        #             print(p1)
        #             print(p2)
                    
        # throw out all parses identical
        # i.e., only keep the keys, which are the first instance found (unique).
        inds_to_remove = set([mm for m in matches.values() for mm in m])

        print("removing these inds", inds_to_remove)
        print("number", len(inds_to_remove))

        print("startring len of parses", len(self.Parses))
        self.Parses = [p for i, p in enumerate(self.Parses) if i not in inds_to_remove]
        print("ending len of parses", len(self.Parses))


    def parses_remove_all_except_bestfit_perms(self, rule_list, reset_index=True):
        """ Only keeps parses (self.Parse) which are best-fit perm
        for at least one of self.ParsesBase. 
        Useful for pruning dataset before chunks task model analysis,
        IN:
        - rule_list, list of str, only keeps perms that are perms of baseparse for one 
        of these rules.
        OUT:
        - self.Parses will be pruned. NOTE: will update the "index" value, so is in order 0, 1, ..
        """

        # Just get the best-fit perms 
        list_perm_indexes = []
        for p in self.ParsesBase:
            if p["rule"] in rule_list:
                for perm in p["best_fit_perms"].values():
                    list_perm_indexes.append(perm["index"])
        list_perm_indexes = sorted(set(list_perm_indexes))
        # print(list_perm_indexes)
        # print(len(list_perm_indexes))

        # Keep only the best perms
        self.Parses = [p for p in self.Parses if p["index"] in list_perm_indexes]

        if reset_index:
            self._parses_give_index()


    def parses_get_all_permutations(self, n_each = 5, direction_within_stroke_doesnt_matter=True):
        """ gets all permutations for each parse. 
        Then appends them to the parses list.
        OUTPUT:
        """

        for i in range(len(self.Parses)):
            if i%5==0:
                print(i)
            parses_list = self.get_all_permutations(i, n=n_each, direction_within_stroke_doesnt_matter=direction_within_stroke_doesnt_matter)

            # add these to ends of self.Parses
            for p in parses_list:
                self.Parses.append(
                    {
                    "strokes":None,
                    "list_ps": p,
                    "permutation_of": i
                    })

        
    ########## GETTING PERMUTATIONS
    def _hier_is_flat(self, hier):
        """ hier is 2-level
        e..g, [[1,2], 3]
        RETURNS:
        - True if hierarchy is flat. evem this [[1], 2, [0]] woudl be True
        """
        from pythonlib.chunks.chunks import hier_is_flat
        return hier_is_flat(hier)


    def get_hier_permutations(self, indparse, is_base_parse=True, 
        nconfigs=750, update_not_append=True, direction_within_stroke_doesnt_matter=True):
        """ 
        uses self.ParsesBase[indparse]["hier"] to determing all the allowable ways
        of permuting edges
        IN:
        - is_base_parse, controls which input parse to use. DOESNT affect output
        - update_not_append, then searched if parse is already exist, if so updates, instaed of appending.
        OUT:
        - output parses to self.Parses (not self.ParsesBase), regardless of whether input parse is base parse.
        TODO:
        - fixed_order controls both ordering and flipping. Instead, should separately
        control them (add another input, fixed_flipping)
        NOTE: 
        This should eventually be subsumed by ChunksClass.search_permutations_chunks()
        """
        # from .search import search_parse
        from pythonlib.drawmodel.tasks import chunk_strokes
        from pythonlib.tools.stroketools import getStrokePermutationsWrapper
            
        assert direction_within_stroke_doesnt_matter==True, "havent tested that this works if allow diff directions."

        self._parses_reset_perm_of_list()

        list_p = self.extract_parses_wrapper(indparse, is_base_parse=is_base_parse)
        parse_dict = self.extract_parses_wrapper(indparse, "dict", is_base_parse=is_base_parse)
        hier = parse_dict["hier"]
        fixed_order = parse_dict["fixed_order"]

        # combine hier and fixed order, so not broken apart after permutation
        hier_fo = [(h,f) for h, f in zip(hier, fixed_order[1])]

        # is hierarchy flat?
        if self._hier_is_flat(hier):
            # Then no hierarchy exists:
            list_parses_all_flat = self.search_parse(list_p, nconfigs,  
                direction_within_stroke_doesnt_matter=direction_within_stroke_doesnt_matter)

            # if direction_within_stroke_doesnt_matter:
            #     # Use this method, doesnt even try to flip
            #     list_parses_all_flat = getStrokePermutationsWrapper(list_p, "all_orders", num_max=nconfigs)
            # else:
            #     # flips and reorders
            #     list_parses_all_flat = self.get_all_permutations(indparse, n=nconfigs, is_base_parse=is_base_parse)
        else:
            # First, collect all the ways of chunking (without concating) list_p
            list_waystochunk_p = []
            if fixed_order[0] == True:
                # Then fixed. dont touch.
                list_waystochunk_p = [hier_fo]
            else:
                # get all permutations of the top level
                list_waystochunk_p = self.search_parse(hier_fo, 1000, 
                    direction_within_stroke_doesnt_matter=direction_within_stroke_doesnt_matter)

                # list_waystochunk_p = getStrokePermutationsWrapper(hier_fo, "all_orders", num_max=1000)

            # for x in list_waystochunk_p:
            #     print(x)
            # assert False
            # Second, for each way of hcunking...
            # -- each high-level permutation
            list_parses_all = []
            for hierthis in list_waystochunk_p:

                # run search within each chunk
                list_perms_each_chunk = [] # each item is list of all perms for that chunk
                for chunk, fixed_order in hierthis:
                    if len(chunk)==0:
                        # This can happen. e.g. if line-->circle rule, but no circles in this task...
                        print("SKIPPING CHUNK since empty")
                        continue

                    # convert chunk to list of p
                    list_p_thischunk = chunk_strokes(list_p, [chunk], False, generic_list=True)[0]
                    if len(list_p_thischunk)==0:
                        print(parse_dict)
                        print(hier_fo)
                        print(list_p)
                        print(chunk)
                        print(hier)
                        print(fixed_order)
                        assert False
                    if fixed_order==True:
                        # then dont get any perms
                        list_perms_each_chunk.append([list_p_thischunk]) # a list of list_p
                    else:

                        perms = self.search_parse(list_p_thischunk, 1000, 
                            direction_within_stroke_doesnt_matter=direction_within_stroke_doesnt_matter)
                        # if direction_within_stroke_doesnt_matter:
                        #     # quicker
                        #     perms = getStrokePermutationsWrapper(list_p_thischunk, "all_orders", num_max=1000)
                        # else:
                        #     perms, _ = search_parse(list_p_thischunk, 
                        #         configs_per=nconfigs, trials_per=800, max_configs=1e6)
                        list_perms_each_chunk.append(perms)

                # Take cross product of lists, each for one chunk in the hierarchy
                from itertools import product
                # for x in list_perms_each_chunk:
                #     print(x)
                #     print(len(x))
                #     print(x[0])
                #     print("-")
                # assert False
                # print(list_perms_each_chunk[0])

                list_parses_all.extend(product(*list_perms_each_chunk))

            # Flatten all
            list_parses_all_flat = []
            for x in list_parses_all:
                list_p_this = []
                for xx in x:
                    if isinstance(xx, list):
                        list_p_this.extend(xx)
                    else:
                        list_p_this.append(xx)
                list_parses_all_flat.append(list_p_this)

        # Add these to self.Parses
        for list_p in list_parses_all_flat:

            # First, try to update a parse if it exists
            parsenew = {
                "strokes":None,
                "list_ps": list_p,
                "perm_of_list":[("base", indparse)] if is_base_parse else ("notbase", indparse)
            }

            if update_not_append:
                # First, see if already exists
                keyvals_update = {
                    "perm_of_list":("base", indparse) if is_base_parse else ("notbase", indparse)
                    }

                #TODO: a self.Parses item can end up twith mutlipel identical entries if perm_of_list, because stroke order issues. is OK"
                #TODO: update_not_append=True should be identical to False, then running remove redundant. Somehow it is slightly different...
                # I thjink I know why - permutations is not exhaustive..
                success, ind = self.update_parse_with_input_parse(parsenew, keyvals_update, 
                    stroke_order_doesnt_matter=False,
                    direction_within_stroke_doesnt_matter=direction_within_stroke_doesnt_matter,
                    append_keyvals=True
                    )


                if success:
                    print('Updated self.Parses at ind', ind, ' with new perm of : ', ("base", indparse) if is_base_parse else ("notbase", indparse))
                if not success:
                    # Then add it to parses
                    self.wrapper_input_parse(parsenew, ver="dict")
                    print('Added self.Parses, perm of : ', ("base", indparse) if is_base_parse else ("notbase", indparse))
            else:
                # always append
                self.wrapper_input_parse(parsenew, ver="dict")
                print('Added self.Parses, perm of : ', ("base", indparse) if is_base_parse else ("notbase", indparse))

            # parse_dict = {
            #     "strokes":None,
            #     "list_ps": list_p,
            #     "permutation_of": ("base", indparse)
            #     }
            # self.wrapper_input_parse(parse_dict, "dict", is_base_parse=False)

        return list_parses_all_flat


    def search_parse(self, parse, configs_per, trials_per=800, max_configs=1e6,
            direction_within_stroke_doesnt_matter=True):
        """ wrapper, so that can decide whether to flip (takes much longer)
        INPOUT:
        - parse, list of p
        """
        from pythonlib.tools.stroketools import getStrokePermutationsWrapper
        if direction_within_stroke_doesnt_matter:
            # then just reorder. much quicker.
            list_parse = getStrokePermutationsWrapper(parse, 
                "all_orders", num_max=configs_per)
        else:
            from .search import search_parse as sp
            list_parse, _ = sp(parse, 
                configs_per=configs_per, trials_per=trials_per, 
                max_configs=max_configs)
        return list_parse


    def get_all_permutations(self, ind, n=500, ver="parser", is_base_parse=False,
        direction_within_stroke_doesnt_matter=True):
        """ get all permutations for this parse (ind)
        By default, get reordering, flipping.
        TODO: circular permtuation for any loops.
        NOTE: does on list of nodes (instead of strokes) but is identical, 
        since can convert to strokes after.
        INPUT:
        - ind, 
        - n, how many parses to keep. note that will search thru many (trials_per), picking
        out random (or select, if use scorefun). 
        OUT:
        - list of parses, each parses is a list of list.
        """
        # import sys
        # sys.path.append("/data1/code/python/GNS-Modeling/")
        # from gns.inference.parsing.top_k import search_parse
        # from .search import search_parse


        if ver=="walker":
            assert False, "old - now converting WalkerStrokes to ParserStrokes"
            parse = self.Parses[ind]["list_walkers"]
            parse = [{"nodes":p.list_ni, "edges":p.list_ei, "flipped":False} for p in parse]
            assert False, "doesnt seem to be flipping..."
            assert False, "make a new class, just like walker class."
        elif ver=="parser":
            # each stroke is a ParserStroke
            parse = self.extract_parses_wrapper(ind, is_base_parse=is_base_parse)
            # parse = self.Parses[ind]["list_ps"]
        else:
            assert False, "need to fix below, cannot use nodelists, since doesnt keep track of the edge"
            # Instead, use edgelists. then convert from edge to nodes.
            # even better, just use codes for each stroke, e.g, [[]]
            # even better, just use the walk object. then give it a method to track whether flipped or not.

            parse = self.parses_extract_nodelists(ind)
            parse = self.parses_extract_edgelists(ind)

            # Uinsg strokes.
            # parses_out, _ = search_parse(parse, lambda parses: torch.rand(len(parses)))
            # parses_out = [[p.numpy() for p in parse] for parse in parses_out]

        # parses_list, _ = search_parse(parse, configs_per=n, trials_per=800, max_configs=1e6)
        parses_list = self.search_parse(parse, configs_per=n, trials_per=800, max_configs=1e6,
            direction_within_stroke_doesnt_matter=direction_within_stroke_doesnt_matter)

        return parses_list


    ########## GRAPH UTIls
    def extract_nodelists(self, ind):
        """ for this ind, extract list of nodes, 
        which is list of list, analogous to strokes, but using sequence of 
        nodes.
        INPUT:
        - ind, integer
        OUT:
        - list of list
        """
        return [p.list_ni for p in self.Parses[ind]["list_walkers"]]


    def extract_edgelists(self, ind):
        """ see above
        NOTE:
        - these do not influence the order of the final stroke, since final stroke will always
        make it so the edges emanate from the nodes.
        i.e., (1,2,0) does not necessarily mean stroke from node 1 to 2. 
        """
        return [p.list_ei for p in self.Parses[ind]["list_walkers"]]
        

    def extract_parses_wrapper(self, ind, kind="parser_stroke_class", is_base_parse=False):
        """ helper to extract list of paths for this parse, represented in flexible waus
        IN:
        - ind, either
        --- int, get a single parse
        --- list of int, get multipel parse
        --- "all", get all parses
        OUT:
        - see within code.
        """

        if isinstance(ind, list):
            return [self.extract_parses_wrapper(i, kind, is_base_parse) for i in ind]
        elif isinstance(ind, str):
            if ind=="all":
                n = len(self.Parses) if is_base_parse==False else len(self.ParsesBase)
                return [self.extract_parses_wrapper(i, kind, is_base_parse) for i in range(n)]
            else:
                assert False
        else:
            assert isinstance(ind, int)

            if is_base_parse:
                ParsesThis = self.ParsesBase
            else:
                ParsesThis = self.Parses

            if kind=="list_of_paths":
                # path is a list of directed edges.
                parses = self.extract_parses_wrapper(ind, "parser_stroke_class", 
                    is_base_parse=is_base_parse)
                return [p.extract_list_of_directed_edges() for p in parses]
            if kind == "parser_stroke_class":
                key = "list_ps"
            elif kind=="strokes_orig_coords":
                # translates back into behavior coord system.
                assert False, "nto coded. see finalize"
            elif kind=="strokes":
                key = "strokes"
            elif kind=="summary":
                # returns list of dicts, where each item has keys: "edges", "walker", "traj"
                # i.e. combines all of the others.
                list_of_edges = self.extract_parses_wrapper(ind, "list_of_paths", 
                    is_base_parse=is_base_parse)
                list_of_walkers = self.extract_parses_wrapper(ind, 
                    is_base_parse=is_base_parse)
                list_of_trajs = self.extract_parses_wrapper(ind, "strokes", 
                    is_base_parse=is_base_parse)
                out = []
                for e,w,t in zip(list_of_edges, list_of_walkers, list_of_trajs):
                    out.append({"edgesdir":e, "walker":w, "traj":t})
                return out
            elif kind=="dict":
                return ParsesThis[ind]
            else:
                assert False, "not coded"

            return ParsesThis[ind][key]


    def extract_all_parses_as_list(self, kind="strokes"):
        """ wrapper to get list of parses
        kind, 
        "strokes"
        {"parser_stroke_class"}
        """
        # assert False, "use extract_parses_wrapper"

        return self.extract_parses_wrapper("all", 
            kind=kind, is_base_parse=False)

        # if kind=="strokes":
        #     self.parses_fill_in_all_strokes()
        #     return [p["strokes"] for p in self.Parses]
        # else:
        #     return [self.extract_parses_wrapper(i, kind) for i in range(len(self.Parses))]

    def _check_parses_is_identical(self, list_p_1, list_p_2, stroke_order_doesnt_matter=True,
        direction_within_stroke_doesnt_matter=True):
        """ Return True if parses identical, False otherwise.
        See check_parses_is_identical...
        INPUT:
        - list_p_1, list_p_2, each is outcome of self.extract_parses_wrapper(ind)
        """
        # for each stroke, find its hash
        list1 = [p.unique_path_id(invariant=direction_within_stroke_doesnt_matter) for p in list_p_1]
        list2 = [p.unique_path_id(invariant=direction_within_stroke_doesnt_matter) for p in list_p_2]

        # since dont care about direction, sort each list (of tuples of ints)
        if stroke_order_doesnt_matter:
            list1 = sorted(list1)
            list2 = sorted(list2)
        
        # check if they are identical
        return list1==list2

    def check_parses_is_identical(self, ind1, ind2, stroke_order_doesnt_matter=True,
        direction_within_stroke_doesnt_matter=True):
        """
        returns true if parses are identical (with certain invariances, see below)
        INPUT:
        - ind1 and ind2 index tinto self.Parses (e.g., self.Parses[1])
        - stroke_order_doesnt_matter, then order of storkes doenst matter
        - direction_within_stroke_doesnt_matter, then the direction within each stroke doesnt matter.
        OUT: 
        - bool

        -----------------------
        list1 = [p.list_ni for p in self.Parses[i]["list_walkers"]]

        list1 and list2 are lists of lists, where the inner lists of integers.,
        returns true if lists 1 and 2 are eequal, allowing for the items in each inner lists to
        flipped either way, and for the inner lists to be in any order.
        - if think of these as parses, indicating sequence of nodes, then this detects if have
        identical parese (ignoring direction of each stroke, and order 
        e..g,:
        if a = [[3,2,1,0], [2,1,2]], then if 
        b = [[3,2,1,0], [2,1,2]] --> True
        b = [[2,1,2], [3,2,1,0]] --> True
        b = [[2,1,2], [0,1,2,3]] --> True
        b = [[2,1,2], [0,1,3,2]] --> False
        
        NOTE: also is invariant ot circular permutation. so [4,5,6,4] is smae as [5,6,4,5]. 
        e.g,:
        if a is [[3, 4, 7, 3], [7, 2, 4, 7]] and
        b = [[3, 7, 4, 3], [4, 2, 7, 4]], then these are equal.    
        """

        list1 = self.extract_parses_wrapper(ind1)
        list2 = self.extract_parses_wrapper(ind2)
        return self._check_parses_is_identical(list1, list2, 
            stroke_order_doesnt_matter=stroke_order_doesnt_matter,
            direction_within_stroke_doesnt_matter=direction_within_stroke_doesnt_matter)


    def _OLD_parses_is_identical_ignoredir_OLD(self, ind1, ind2):
        """
        returns true if parses are identical (with certain invariances, see below)
        INPUT:
        - ind1 and ind2 index tinto self.Parses (e.g., self.Parses[1])
        OUT: 
        - bool
        NOTE:
        - OLD, beucase doesnt take into accound the edge id (fails for multigraphs).

        -----------------------
        list1 = [p.list_ni for p in self.Parses[i]["list_walkers"]]

        list1 and list2 are lists of lists, where the inner lists of integers.,
        returns true if lists 1 and 2 are eequal, allowing for the items in each inner lists to
        flipped either way, and for the inner lists to be in any order.
        - if think of these as parses, indicating sequence of nodes, then this detects if have
        identical parese (ignoring direction of each stroke, and order 
        e..g,:
        if a = [[3,2,1,0], [2,1,2]], then if 
        b = [[3,2,1,0], [2,1,2]] --> True
        b = [[2,1,2], [3,2,1,0]] --> True
        b = [[2,1,2], [0,1,2,3]] --> True
        b = [[2,1,2], [0,1,3,2]] --> False
        
        NOTE: also is invariant ot circular permutation. so [4,5,6,4] is smae as [5,6,4,5]. 
        e.g,:
        if a is [[3, 4, 7, 3], [7, 2, 4, 7]] and
        b = [[3, 7, 4, 3], [4, 2, 7, 4]], then these are equal.    
        """
        assert False, 'use new ver'
        def _hash_linear(li):
            """ make the inner list in increasing order
            this ensures that sorting the outer list always leads to same
            result, assuming the inner lists are the same (modulo flipping)
            - hash, since returns list that ids a set of equivalent lists.
            """
            if li[-1]>li[0]:
                # then dont flip
                return li
            elif li[-1]<li[0]:
                return li[::-1]
            else:
                # they are equal
                return li
        
        def _hash_circular(l):
            """ given list of nodes, where first and list are identical, so is
            a loop, returns a unique list of nodes that identifies this loop.
            invariant to order and direction.
            e.g.,: 
            [4, 2, 7, 4] --> (2, 4, 7)
            [7, 2, 4, 7] --> (2, 4, 7)
            """

            # remove endpoint
            l = l[:-1]

            # for each direction, get all circular permutations
            l_rev = l[::-1]

            tmp = []
            for lthis in [l, l_rev]:
                from more_itertools import circular_shifts
                tmp.extend([list(x) for x in circular_shifts(lthis)])
            return sorted(tmp)[0]
        
        def _hash(l):
            if l[0]==l[-1]:
                return _hash_circular(l)
            else:
                return _hash_linear(l)

        assert False, "problem: doesnt take into account multiple paths between nodes"
        # e.g., sequence of nodes [2, 7, 4, 2] is paired with edges sequence: [(2, 7, 0), (7, 4, 1), (4, 2, 0)]
        # solution: represent sequence of nodes as something like [2, a, 7, b, 4, a, 2].
        # altearnive solution: any operation on stroke involves both nodelist and edgelist.

        
        list1 = self.parses_extract_nodelists(ind1)
        list2 = self.parses_extract_nodelists(ind2)

        # first, for each inner list, flip so that is incresaing.
        list1 = [_hash(x) for x in list1]
        list2 = [_hash(x) for x in list2]
        
        # second, sort each list
        list1 = sorted(list1)
        list2 = sorted(list2)
        
        # check if they are identical
        return list1==list2




    ################ AUTOMATICALLY MODIFYING PARSE GRAPH
    def _graph_remove_zerolength_edge(self, dist_thresh=20):
        """
        Remove any edges from node to itself
        - dist_thresh, only removes if the cum dist along the edge is less
        than this.
        """
        edges_to_remove = []
        for ed in self.Graph.edges:
            if ed[0]==ed[1]:
                # print(self.Graph.edges[ed])
                # only remove if this edge has short distance for pts
                from pythonlib.drawmodel.features import strokeDistances
                d = strokeDistances([self.Graph.edges[ed]["pts"]])[0]
                # make sure each node has other edges that would still remain
                nodes = ed[:2]
                nedges_per_node = [len(list(self.Graph.edges(n))) for n in nodes]

                # only remoive if fails distance treshold, and ther are other exdges remining
                if d<dist_thresh and all([n>1 for n in nedges_per_node]):
                    edges_to_remove.append(ed)

        if len(edges_to_remove)>0:
            print("Removing edges that go from a node to istself:")
            self.modify_graph(edges_to_remove=edges_to_remove)


    def find_closest_nodes_under_thresh(self, thresh, return_sorted_list=False): 
        """ Finds nodes which are close to each other
        OUT:
        - set of 2 nodes
        --- unless return_sorted_list, in which case return list of pairs, sorted from
        closest to fuerthers. i.e., list_pairs, dists
        """
        def _find_closest_nodes_under_thresh(thresh, G):
            """ find pairs of nodes that are closer than thresh
            OUT:
            - None if dosent find.
            - or the closest pair as a set, e..g, {2,3}
            """
            pairs = []
            dists = []
            for n in G.nodes:
                for nn in G.nodes:
                    if nn>n:
                        try:
                            p1 = G.nodes[n]["o"]
                            p2 = G.nodes[nn]["o"]
                        except Exception as err:
                            print(n, nn)
                            print(G.nodes[n])
                            print(G.nodes[nn])
                            print(G.nodes)
                            print(G.edges)
                            self.plot_graph()
                            raise err
                        d = np.linalg.norm(p1-p2)
                        if np.linalg.norm(p1-p2)<thresh:
                            pairs.append({n, nn})
                            dists.append(d)
            if len(pairs)==0:
                if return_sorted_list:
                    return [], []   
                else:
                    return None
            else:
                # take the closest pair
                li = [(a, b) for a,b in zip(pairs, dists)]
                li = sorted(li, key = lambda x: x[1])
                if return_sorted_list:
                    return [x[0] for x in li], [x[1] for x in li]
                pairthis = li[0][0] # take the closest pair
                return pairthis
        
        return _find_closest_nodes_under_thresh(thresh, self.Graph)


    def graphmod_merge_nodes_auto(self, thresh=50):
        """ automaticalyl merge airs of nodes that are closert than thershold until no
        such pairs remain
        """

        did_mod = False

        def _get_node_pair():
            if False:
                # take the closest pair
                pairthis = self.find_closest_nodes_under_thresh(thresh=thresh)
            else:
                # If any exist, first take pairs that dont make loop with themselves (ie.., endpoints of an unclosed loop)
                # The reason: closing loops by merge only smooths one of the two ends. Better to leave these to end, so that
                # all smoothing is done as much as possible.
                list_pairthis, lsit_dists = self.find_closest_nodes_under_thresh(thresh=thresh, return_sorted_list=True)
                if len(list_pairthis)==0:
                    pairthis = None
                else:
                    list_pairthis_not_unclosed_loop = [pair for pair in list_pairthis if self.check_nodes_make_unclosed_loop(list(pair)[0], list(pair)[1])==False]
                    if len(list_pairthis_not_unclosed_loop)>0:
                        pairthis = list_pairthis_not_unclosed_loop[0]
                    else:
                        # take the closest
                        pairthis = list_pairthis[0]
            return pairthis

        # run
        pairthis = _get_node_pair()
        while pairthis is not None:
            print("Merging this pair of nodes: ", pairthis)         
            # Merge each pair
            # Make sure that if a node is part of multiple pairs, should do them separately, unless
            # they are all close together.
            self.merge_nodes(pairthis)

            # pairthis = self.find_closest_nodes_under_thresh(thresh=thresh)
            pairthis = _get_node_pair()

            if pairthis is None:
                print("No more pairs to merge - done!")
            else:
                print("Merging this pair of nodes: ", pairthis)         
            did_mod = True
        return did_mod


    def find_closest_edge_to_split(self, thresh):
        """ find any pts that are close to an edge (that the pt is not on)
        return that edge, and the location on that edge that is closest to the pt
        """

        G = self.Graph
        edges = []
        dists = []
        inds = []
        nodes = []
        for n in G.nodes:
            for e in G.edges:
                if n not in e[:2]:
                    d, ind = self.min_dist_node_to_edge(n, e)
                    if d<thresh:
                        edges.append(e)
                        dists.append(d)
                        inds.append(ind)
                        nodes.append(n)
        if len(edges)==0:
            return None, None, None, None
        else:
            # get the edge that has the case of closest node.
            tmp = [(e, d, i, n) for e, d, i, n in  zip(edges, dists, inds, nodes)]
            tmp = sorted(tmp, key=lambda x:x[1])
            return tmp[0]


    def graphmod_merge_close_edges_auto(self, thresh=25):
        """ auto merge edges that are close, even if they don't have any nodes in that vicinity.
        similar to graphmod_split_edges_auto, but there was edge close to node. here is edge
        close to edge. Merges into single node.
        """
        from itertools import product
        # Track state
        found_an_edge_pair = True # only turns true if can get thru all edge pairs without any triggering closeness.
        niter = 0
        maxiter = 30
        didmod = False
        while found_an_edge_pair == True:
            # Find edges that are close to each other
            
            for (ed1, ed2) in product(self.Graph.edges, self.Graph.edges):
                if ed1==ed2:
                    # print("HERE")
                    continue
                if self.check_edges_share_node(ed1, ed2):
                    # Or else all of these edges would be considered close.
                    # print("HERE2")
                    continue

                dist, ind1, ind2 = self.find_closest_pt_twoedges(ed1, ed2)

                if dist<thresh:
                    didmod = True
                    print("CONNECTING THESE EDGES, since are close:", ed1, ed2, dist, ind1, ind2)
                    # then create new nodes here and merge them
                    list_nodes = []
                    for ed, ind in zip([ed1, ed2], [ind1, ind2]):
                        pts = self.get_edge_pts(ed)
                        node_new = self.add_node([pts[ind,:], ed], "pt_edge")
                        list_nodes.append(node_new)
                    # then merge
                    if False:
                        self.graphmod_merge_nodes_auto(thresh = thresh+1)
                    else:
                        self.merge_nodes(list_nodes)
                    
                    # break and restart (since edges are now changed)
                    found_an_edge_pair=True
                    # print("HERE3")
                    break

            found_an_edge_pair = False
            # # If can get to here, that means no edge pairs are usable
            # if found_an_edge_pair:
            #     no_more_edges = False
            # else:
            #     no_more_edges = True
                
            niter+=1
            if niter>maxiter:
                self.plot_graph()
                print(self.Graph.edges)
                print(self.Graph.nodes)
                assert False, "what loop?"
        return didmod

            
    def graphmod_split_edges_auto(self, thresh=25):
        """ automatilcaly split edgfes that re close to a pt, at
        the location on edge closest to the pt.
        """

        ct = 0
        e, d, i, n = self.find_closest_edge_to_split(thresh)
        didmod = False
        while e is not None:
            
            if ct>20:
                print(G.nodes)
                print(G.edges)
                assert False, "recursive problem."
            print("**Splitting edge", e, "Merging with node:", n)
            nnew = self.split_edge(e, i)
            
            # self.plot_graph() 
            # Must then merge these two pts.
            print("Merging: ", {n, nnew})
            self.merge_nodes({n, nnew})

            # self.plot_graph() 
            # Merge any other nodes that are now too close.
            self.graphmod_merge_nodes_auto(thresh=thresh)

            e, d, i, n = self.find_closest_edge_to_split(thresh)

            # self.plot_graph()
            # assert False

            didmod=True
        return didmod


    ################ AUTOMATICALLY MODIFYING PARSE GRAPH [CIRCLES, LOOPS]
    def graphmod_auto_loops_floating_only_one_node(self, ploton=False):
        """ Floating loops will only have one node. Arbitrarily keeds the first
        node in the list of nodes.
        """

        list_cycles = self.find_cycles()
        
        # 1) first, floating loops should only have one node
        nodes_to_remove = []
        didmod=False
        for cy in list_cycles:
            if self.is_isolated(cy):
                if len(cy)>1:
                    # Then multiple edges. remove one node (arbitratiyl just keep the first)
                    list_nodes = self.get_nodes_from_list_edges(cy)
                    nodes_to_remove.extend(list_nodes[1:])
                    didmod = True
                    print("[Floating cycle has >1 node. removing excess nodes:]", list_nodes[1:])

        for node in nodes_to_remove:
            self.remove_node(node, ploton=ploton)

        return didmod


    def graphmod_auto_loops_cleanup_excess_nodes(self, ploton=False):
        """ If not isoalted cycle, then removes all nodes that are degree 2 or less
        i.e., if this circle has a branch, then no need to have other nodes on it.
        """
        list_cycles = self.find_cycles()
        nodes_to_remove = []
        didmod = False
        for cy in list_cycles:
            if not self.is_isolated(cy):
                # go thru all nodes. 
                list_nodes = self.get_nodes_from_list_edges(cy)
                for node in list_nodes:
                    degree = self.Graph.degree[node]
                    if degree<3:
                        # remove this node
                        print("[Extra node on acycle that arleady has degree>3 nodes. Removing:]", node)
                        nodes_to_remove.append(node)
                        didmod=True
        nodes_to_remove = list(set(nodes_to_remove))
        for node in nodes_to_remove:
            self.remove_node(node, ploton=ploton)
        return didmod

    def graphmod_auto_loops_add_nodes(self, list_angles_opposite = None, THRESH = 40, 
            ploton=False):
        """ Add nodes onto a loop in smart way. E.g., place a node opposite current node (that branches).
        Finds current nodes on loop, and gets candidate new nodes that are a given angle away (e.g., opposite).
        If every one of those candidates is NOT near a current node, then will pick the middle candidate (e.g.
        directly opposite) and place a single new node there. 
        - list_angles_opposite, list of angles, in radians, to check candidate pts, relative to each existing node.
        See description. New node will be the middle one of these, if they all are not close to a current node.
        - THRESH = 40, if any existing node is this close to oen of the candidates, then will not add any of them. 
        """
        from math import pi
        from pythonlib.tools.stroketools import travel_on_loop
        if list_angles_opposite is None:
            list_angles_opposite = np.linspace(pi/2, 3*pi/2, 5)

        # 3) Add new node across from currnet nodes in a cycle, if there are no nodes close by
        list_cycles = self.find_cycles()
        list_angles_opposite =  np.linspace(pi/2, 3*pi/2, 5)
        
        nodes_to_add = []
        list_edges_split = []
        list_inds_split = []
        didmod = False
        for cy in list_cycles:
            # print("")
            # print(cy)
            # self.plot_graph()

            list_nodes = self.get_nodes_from_list_edges(cy, in_order=True)

            for node in set(list_nodes):

                # Only consider nodes that are branching off of this loop.
                if self.Graph.degree[node]>2:
                    # check that opposite this node there is node close by

                    # 1) Generate pts
                    pts = self.find_pts_this_path_edges(cy)
                    # print(pts)
                    # assert False
                    # pts = self.find_pts_between_these_nodes(list_nodes, concat=True)
                    o = self.get_node_dict(node)["o"]

                    # Given loop, and pt, find opposite pt
                    list_pts = [travel_on_loop(pts, o, a) for a in list_angles_opposite]
                    
                    # for each pt, check if it is close to a node. if not, then add it.
                    pts_to_add = []
                    for pt in list_pts:
                        closenodes = self.find_close_nodes_to_pt(pt, THRESH)
                        if len(closenodes)==0:
                            # add this pt
                            pts_to_add.append(pt)

                    if len(pts_to_add)<len(list_pts):
                        # then at least one pt was close to a node. abort entirely
                        continue

                    else:
                        # only keep one of the pts (the middle one)
                        if len(pts_to_add)>1:
                            pt = pts_to_add[int(np.round(len(pts_to_add)/2))]
                        elif len(pts_to_add)==1:
                            pt = pts_to_add[0]
                        else:
                            assert False

                        # snap pt to nearest edge
                        edges, dists, inds = self.find_close_edge_to_pt(pt, dosort=True, thresh=30)
                        
                        # find the edge that is on the path and is the closest
                        eligible_edges = cy
                        # eligible_edges = self.find_paths_between_these_nodes(list_nodes)
                        found=False
                        for ed, d, i in zip(edges, dists, inds):
                            if self.edge_is_in(eligible_edges, ed):
                                if not self.edge_is_in(list_edges_split, ed):
                                    print("[Adding a new node oppposite an old node, on (edge, ind):]", ed, i)
                                    list_edges_split.append(ed)
                                    list_inds_split.append(i)
                                found=True
                                didmod = True
                                break
                        if found==False:
                            print(edges, dists, inds)
                            print(eligible_edges)
                            # print(list_nodes_tmp)
                            assert False


                    if ploton:
                        plt.figure()
                        plt.plot(pts[:,0], pts[:,1])
                        plt.plot(o[0], o[1], "ro", label="node")
                        for pt_new in list_pts:
                            plt.plot(pt_new[0], pt_new[1], "ok", label="opposite")

        if ploton:
            self.plot_graph()

        for ed, i in zip(list_edges_split, list_inds_split):
            if ed not in self.Graph.edges:
                print(ed)
                print(self.Graph.edges)
                assert False
            self.split_edge(ed, i)
        if ploton:
            self.plot_graph()

        return didmod
                            

    ############## TOOLS FOR MODIFYING NODES MANUALLY
    def remove_node(self, node, ploton=False):
        """ Carefully removes a node. Connects up other nodes that were previously
        connected to this node.
        NOTE:
        - only works if there are 2 and only 2 edges for this node. Otherwise it is not clear
        how to reconnect the new nodes.
        """

        edges_to_remove = []

        # Check that there are only 2 edges
        list_edges = self.find_edges_connected_to_this_node(node)
        if len(list_edges)<2:
            assert False, "this node is a loop with no other nodes.."
        if len(list_edges)>2:
            assert False, "multiple adjacent nodes, not sure how to reconnect them"

        # Delete these two edges
        edges_to_remove.extend(list_edges)

        # Make a new edge between the two adjacent edges
        nodes_adjacent = self.find_adjacent_nodes(node)
        if len(nodes_adjacent)==1:
            # then two edges, to same node...
            listn = [nodes_adjacent[0], node, nodes_adjacent[0]]
        else:
            listn = [nodes_adjacent[0], node, nodes_adjacent[1]]
        pts = self.find_pts_between_these_nodes(listn, concat=True)
        new_edge = (listn[0], listn[2], {"pts":pts})
        # if False:
        #     # Old, where I made mistake, assumed needed new ind...
        #     list_inds = self.find_edgeindices_for_edges_between_these_nodes(nodes_adjacent[0],
        #         nodes_adjacent[1])
        #     new_ind = max(list_inds)+1
        #     new_edge = (nodes_adjacent[0], nodes_adjacent[1], new_ind)

        # DO MOD
        self.modify_graph(nodes_to_remove=[node], edges_to_add=[new_edge], 
            edges_to_remove=edges_to_remove, plot_pre_and_post=ploton)

    def add_node(self, node_info, ver):
        """ [GOOD] Flexible way to manually adda single node.
        Takes care to (i) find where to add the node and (ii) reconnect edges
        properly.
        INPUT:
        - ver, str, method to use
        - node_info, list, depends on ver
        """

        if ver=="pt_edge":
            """ give a pt (xy) and snaps it onto the nearest location in edge
            - node_info, [pt, edge], where pt is [x,y] and edge is (n1, n2, idx)
            """
            # edge = (2,6,1)
            # pt = [75, 5]
            pt = node_info[0]
            edge = node_info[1]

            dist, ind_along_edge = self.min_dist_pt_to_edge(pt, edge)
            node_new = self.split_edge(edge, ind_along_edge)
            
        elif ver=="pt_anyedge":
            """ Given pt (xy) snaps it onto nearest location on nearest edge
            """
            assert False, "test this. and return nodenew"
            strokes = [np.array([pt, pt, pt])]
            P.graphmod_add_nodes_strokes_endpoints(strokes)

        return node_new


    def graphmod_add_nodes_strokes_endpoints(self, strokes, thresh=10):
        """ add nodes for each endpoint for each stroke (or do nothing
        if it already exists)
        """
        for strok in strokes:
            for ind in [0, -1]:
                pt = strok[ind, :2]
                # check if this strokes endpoints both have nodes assigned
                list_ni = self.find_close_nodes_to_pt(pt, thresh=thresh)
                
                if len(list_ni)==0:
                    # then add a node
                    edges, dists, inds= self.find_close_edge_to_pt(pt, thresh=thresh)
                elif len(list_ni)>1:
                    # Then skip. This is usally when edge of a line is close to another
                    # node, for natural reasons.
                    continue
                    # self.plot_graph()
                    # print(pt)
                    # print(list_ni)
                    # assert False, "first do merge"
                else:
                    # fine, already has a single node.
                    continue

                # Split graph at each of the extracted edges
                for ed, ind in zip(edges, inds):
                    print("**Splitting, adding strokes endpoints", ed, ind)
                    self.split_edge(ed, ind)

    def merge_nodes(self, nodes, THRESHMULT=2.):
        """ Merges a pair of nodes intoa  new node, with position in between.
        Takes care to update edges correctly
        IMPUT: 
        - THRESHMULT, any edges shorter than THRESHMULT*(distance between pts)
        will be removed. others will be kept.
        NOTE:
        - procedure: every edge for these nodes will be removed. new edges will be 
        added to replace them, except if they are shorted than THRESHMULT*distance between noides
        In addition, the endpoints will be shifted; first pt always to the new node, 
        last pt usually stable, unless the other node will be deleted (eitehr if it is aloop,
        or if it is the one being merged). Fiunally, will smooth the path towards the new
        node location.
        """

        def _merge_nodes(nodes, G):
            """ 
            - nodes, set of nodes, e..g, {3,4}
            - G
            OUT:
            - modifies G to have new node (ind is max + 1), and new edges replacing old edges.
            """
            assert isinstance(nodes, set)
            list_nodes = list(nodes)

            o1 = self.get_node_dict(list_nodes[0])["o"]
            o2 = self.get_node_dict(list_nodes[1])["o"]
            radius = np.linalg.norm(o1-o2)
            radius *= 1.75
            # print("--- RADIUS", radius)

            # new center
            onew = np.round(np.mean(np.stack([G.nodes[n]["o"] for n in nodes]), axis=0)).astype(np.int16)
            nnew = max(list(G.nodes))+1

            # add new edges to replace any old edges
            edges_to_add = []
            edges_to_remove = []
            if False:
                # stoped this, since below is decindeing to only remove if it is a short edge
                elist = [e for e in G.edges if set(e[:2])==nodes]
                edges_to_remove.extend(elist)

            for nthis in list_nodes:
                othis = G.nodes[nthis]["o"]

                list_edges = self.find_edges_connected_to_this_node(nthis)
                for e in list_edges:
                # for e in G.edges:
                    # edg = G.edges[e]

                    if self.edge_is_in(edges_to_remove, e):
                        # you've already considered whether to keep it
                        continue

                    ###### ALWAYS REMOVE
                    edges_to_remove.append(e)

                    ##### ADD A NEW EDGE TO REPLACE THIS?
                    if True:
                        nother = e[1]
                    else:
                        # conver this edge to a new edge, using same pts
                        nother = [n for n in e[:2] if n!=nthis]
                        if len(nother)==0:
                            print("ytou might have repeat node? a-->a")
                            print(e)
                            print(nthis)
                            self.plot_graph()
                            assert False
                        elif len(nother)>1:
                            assert False, "huh"
                        nother = nother[0]

                    if set([nthis, nother])==nodes:
                        # If this edge is shorter than the distance threshold between nodes
                        # then remove it
                        thresh = THRESHMULT*np.linalg.norm(o2-o1)
                        if self.get_edge_length(e)<thresh:
                            continue
                        else:
                            pass
                            # Keep this edge

                    ################# construct a new edge
    #                 weight = edg["weight"]
    #                 visited = edg["visited"]
                    pts = self.get_edge_pts(e, anchor_node=nthis)

                    ################ REPLACE ENDPOINTS First pt always modifed to onew
                    pts[0, :] = onew

                    # Last pt usually modified to nother, unless nother is gong to be deleted.
                    if nthis==nother:
                        # Then is a loop, modify both start and end to the same (onew)
                        pts[-1,:] = onew
                    elif set([nthis, nother])==nodes:
                        # then these nodes will be merged. the new endpoints should be the lcation
                        # of merge pt.
                        pts[-1, :] = onew
                    else:
                        # the nother will not be deleted. keep its position
                        pts[-1,:] = self.get_node_dict(nother)["o"]

                    if False:
                        pts = edg["pts"]

                        # add the center to the end that is currently closest to the center
                        d1 = np.linalg.norm(pts[0,:]-othis)
                        d2 = np.linalg.norm(pts[-1,:]-othis)
                        if d1<d2:
                            pts = np.insert(pts, 0, onew.reshape(1,-1), axis=0)
                        else:
                            pts = np.append(pts, onew.reshape(1,-1), axis=0)

                    ################# smoothening out the new edge
                    # only the starting end for now...
                    # - radius, based on distance between two nodes that you are merging
                    from pythonlib.tools.stroketools import smooth_pts_within_circle_radius
                    pts = smooth_pts_within_circle_radius(pts, radius, must_work=True)

                    #################### Add this new edge
            #         G.add_edge(nnew, nother, pts=pts, weight=weight)
                    if nthis==nother:
                        # Then this is a loop. nnew must be connected to iself, since
                        # it mus tbe coonnected to the node that replaces nother.
                        edges_to_add.append((nnew, nnew, {"pts":pts}))
                    elif set([nthis, nother])==nodes:
                        # Then these nodes will be merged
                        edges_to_add.append((nnew, nnew, {"pts":pts}))
                    else:
                        # Then nother still exists, nthis is deleted and replaced with nnew
                        edges_to_add.append((nnew, nother, {"pts":pts}))
    #                 edges_to_add.append((nnew, nother, {"pts":pts, "weight":weight}))
                    # edges_to_remove.append((nthis, nother, e[2]))

            if len(edges_to_add)==0:
                # check that this is not an "unclosed" loop (i.e., 2 endpoints being merged)
                if self.check_nodes_make_unclosed_loop(list_nodes[0], list_nodes[1]):
                # list_neigh1 = list(G.neighbors(list_nodes[0]))
                # list_neigh2 = list(G.neighbors(list_nodes[1]))
                # if set(list_neigh1 + list_neigh2)==set(list_nodes):
                # then is an unclosed loop, they are their only neighborfs, so cant delete this edge.
                    pts = self.find_pts_between_these_nodes(list_nodes, concat=True)
                    pts[0, :] = onew
                    pts[-1, :] = onew
                    edgenew = (nnew, nnew, {"pts":pts})
                    edges_to_add.append(edgenew)
                else:
                    print(nodes)
                    print(G.nodes)
                    print(G.edges)
                    print(edges_to_remove)
                    assert False, "removing edge and not replacing with anything.." 

            ################# NODES
            nodes_to_add = [(nnew, {"o":onew, "pts":onew.reshape(1,-1)})]
            nodes_to_remove = list(nodes)

            self.modify_graph(nodes_to_add=nodes_to_add, nodes_to_remove=nodes_to_remove, 
                        edges_to_add=edges_to_add, edges_to_remove=edges_to_remove)
            
            return G

        if isinstance(nodes, list):
            nodes = set(nodes)
        return _merge_nodes(nodes, self.Graph)


    def split_edge(self, edge, ind):
        """
        Split this edge into two new edges at location within edge defined by ind.
        (i.e., add a node, and connect that node to the endpoints)
        Takes care to update all edges etc.
        NOTE:
        - ind is using canonical order of pts, hard ocded for this edge.
        """

        def _split_edge(edge, ind, G):
            """ split this edge at this index along its pts
            """

            # First, make the new node
            # pt for the new node
            # print("split edge start")
            print("split this edge", edge, "at this ind", ind)
            # print(self.Graph.edges)
            print("len of edge", self.Graph.edges[edge]["pts"].shape)
            # print("all edges", G.edges)
            onew = G.edges[edge]["pts"][ind]   
            
            # figure out orientation of pts
            pts = G.edges[edge]["pts"]

            # This old method can fail.
            if False:
                n1 = edge[0]
                n2 = edge[1]
                o1 = G.nodes[n1]["o"]
                o2 = G.nodes[n2]["o"]

                try:
                    if self.check_pts_orientation(pts, o1, return_none_on_tie=True) is None:
                        # then either way is fine
                        pass
                    else:
                        if self.check_pts_orientation(pts, o1):
                            # then pts goes from n1 to n2.
                            assert self.check_pts_orientation(pts, o2)==False
                        else:
                            # n1 is always the one closest to pts[0]
                            assert self.check_pts_orientation(pts, o2)==True
                            n1 = edge[1]
                            n2 = edge[0]
                except Exception as err:
                    print("____")
                    print(edge)
                    print(n1, n2)
                    print(self.check_pts_orientation(pts,o1))
                    print(self.check_pts_orientation(pts,o2))
                    self.plot_graph()
                    raise err
            else:
                n1, n2 = self.find_nodes_this_edge_canonical_pts_order(edge)

            # New pts
            nnew = max(G.nodes)+1
            onew = pts[ind,:]
            # print("Added node: ", nnew)
            # G.add_node(nnew, o=onew, pts=onew.reshape(1,-1))
            # self.modify_graph(nodes_to_add = [(nnew, {"o":onew, "pts":onew.reshape(1,-1)})])
            self.modify_graph(nodes_to_add = [(nnew, {"o":onew})])
            
            # New edges
            edges_to_add = [
                (n1, nnew, {"pts":pts[:ind+1,:]}),
                (nnew, n2, {"pts":pts[ind:,:]})]
            
            # Remove edge
            edges_to_remove = [edge]
            
            self.modify_graph(edges_to_add=edges_to_add, edges_to_remove=edges_to_remove)
            
            return nnew

        return _split_edge(edge, ind, self.Graph)
        

    ################### LOW-LEVEL CODE FOR MANIPULATING GRAPH
    def modify_graph(self, nodes_to_add=None, nodes_to_remove=None, edges_to_add=None, 
        edges_to_remove=None, plot_pre_and_post=False):
        """ Low-level to add/remove nodes/edges. DO NOT use this unless you know what you doing,
        becuase this doesnt' autoatmically prune and add edges so that graph is appropriately
        connected.
        INPUTS:
        - nodes_to_add, list of nodes, where each is (nnew, {"o":onew, "pts":onew.reshape(1,-1)})]
        -- NODE: nnew can be None, in which case increments node for you
        - nodes_to_remove, list of ints
        - edges_to_add, list of edges, where edges[0] = (node0, node1, {pts:[...]})
        - edges_to_remove, list of edges, where each is (node0, node1, index)
        """ 


        assert self._GraphLocked==False, "already locked (probablyt because gotten parses)"
        
        # Easier, since cant iterate over None, below.
        if nodes_to_add is None: 
            nodes_to_add = []
        if nodes_to_remove is None:
            nodes_to_remove = []
        if edges_to_add is None:
            edges_to_add = []
        if edges_to_remove is None:
            edges_to_remove = []

        if plot_pre_and_post:
            self.plot_graph()

        G = self.Graph
         
        print("Graph before modifiyong:")
        print(G.nodes)
        print(G.edges)
        outdict = {}

        # Sanity checks
        # 1) not adding edges for nodes that dont exist
        for ed in edges_to_add:
            for node in ed[:2]:
                if node not in self.Graph.nodes and node not in [n[0] for n in nodes_to_add]:
                    print(edges_to_add)
                    print(self.Graph.nodes)
                    print(nodes_to_add)
                    assert False, "cant add edge for node that doesnt exist"
            # 1b) Check that gave me pts
            assert isinstance(ed[2], dict)
            assert "pts" in ed[2].keys()

    # 2) Adding, removing nodes
        for n in nodes_to_add:
            node = n[0]
            if node is not None:
                assert not isinstance(node, list), "hacky check, I might have entered node=[] at some pt."
                # 2a) Not adding and removing node at same time
                if node in nodes_to_remove:
                    print(nodes_to_add)
                    print(nodes_to_remove)
                    assert False, "trying to add and remove a node at the saemt ime"
                # 2b) adding a node that already exists:
                if node in self.Graph.nodes:
                    print(node)
                    print(self.Graph.nodes)
                    assert False, "node already excists"
            else:
                assert False, "not yet coded, get auto node"
            # 2c) must give me pts
            assert isinstance(n[1], dict)
            assert "o" in n[1].keys()



        # Removing nodes
        if len(nodes_to_add)>0:
            print("Adding new nodes:", [[e for e in ed if not isinstance(e, dict)] for ed in nodes_to_add])
        #     G.add_node(nnew, o=onew, pts=onew.reshape(1,-1))
            G.add_nodes_from(nodes_to_add)
            outdict["nodes_added"] = [n[0] for n in nodes_to_add]
            
        if len(nodes_to_remove)>0:
            print("Removing old nodes:", nodes_to_remove)
            G.remove_nodes_from(nodes_to_remove)
            outdict["nodes_removed"] = nodes_to_remove

        if len(edges_to_remove)>0:
            print("Removing edges:", edges_to_remove)
            G.remove_edges_from(edges_to_remove)
            outdict["edges_removed"] = edges_to_remove

        if len(edges_to_add)>0:
            # Remove edges
            print("Adding edges", [[e for e in ed if not isinstance(e, dict)] for ed in edges_to_add])
        #     print(", which replace these removed edges", edges_to_remove)
            keys = G.add_edges_from(edges_to_add)
            outdict["edges_added"] = [(ed[:2], key) for ed, key in zip(edges_to_add, keys)]

        print("- Done modifying, new nodes and edges")
        print(G.nodes)
        print(G.edges)

        if plot_pre_and_post:
            self.plot_graph()

        return outdict


    ######################## GRAPHMOD - tools
    def check_pts_orientation(self, pts, o, return_none_on_tie=False, return_true_on_tie=False):
        """ figures out how pts is oriented relative to 
        pt1. returns True is pts[0] is closer, or False otherwise
        - pt1, Nx2 array
        - o, (2,) array
        - return_none_on_tie, otherwise fails.
        - return_true_on_tie, overwrite return_none_on_tie, is useful if have a loop.
        """
        
        # add the center to the end that is currently closest to the center
        d1 = np.linalg.norm(pts[0,:]-o)
        d2 = np.linalg.norm(pts[-1,:]-o)
        if d1==d2:
            if return_none_on_tie and return_true_on_tie==False:
                return None
            elif return_true_on_tie==True:
                return True
            else:
                print(d1, d2)
                print(o)
                print(pts[0], pts[-1])
                assert False
        if d1<d2:
            return True
        else:
            return False



    ########## GRAPHMOD - helpers to find nodes and edges
    def find_closest_pt_twoedges(self, ed1, ed2):
        """ 
        """
        from pythonlib.tools.distfunctools import closest_pt_twotrajs
        pts1 = self.get_edge_pts(ed1)
        pts2 = self.get_edge_pts(ed2)
        dist, ind1, ind2 = closest_pt_twotrajs(pts1, pts2)
        return dist, ind1, ind2



    def find_edges_connected_to_this_node(self, node):
        """ return list of edges given a node (int)
        - input node will always be the first node in each edge.
        OUT:
        - list_edges, e.g, [(0,3,1), (..)...]
        NOTE:
        - if this node doesnt exist, returns empty list.
        """
        if node not in self.Graph.nodes:
            print(node)
            print(self.Graph.edges)
            print(self.Graph.nodes)
            assert False
        return list(self.Graph.edges(node, keys=True))
        # list_edges = [ed for ed in self.Graph.edges if node in ed[:2]]
        # return list_edges

    def find_adjacent_nodes(self, node):
        """ Return a list of nodes connected to this node.
        """
        return list(self.Graph.neighbors(node))

    def find_edges_between_these_nodes(self, node1, node2):
        """ Returns list of all edges between thse nodes
        INPUT:
        - node1, node2, ints. order doesnt matter.
        OUT:
        - list of edges
        """
        return [ed for ed in self.Graph.edges if set(ed[:2])==set([node1, node2])]

    def find_nodes_this_edge_canonical_pts_order(self, edge):
        """ return (node1, node2) such that pts goes from node1 to node2.
        NOTE: if this is loop, then not well defined. But in this case doesnt usually
        matter. will return in the order that they are present in edges.
        """

        pts = self.get_edge_pts(edge)
        node1 = edge[0]
        node2 = edge[1]
        o1 = self.get_node_dict(node1)["o"]
        o2 = self.get_node_dict(node2)["o"]

        x1 = self.check_pts_orientation(pts, o1, return_true_on_tie=True)
        x2 = self.check_pts_orientation(pts, o2, return_true_on_tie=True)
        if x1 and not x2:
            return node1, node2
        elif x2 and not x1:
            return node2, node1
        elif x1 and x2:
            # then they are both equally close to pts[0]
            return node1, node2
        else:
            assert False, "unclear"



    def find_edgeindices_for_edges_between_these_nodes(self, node1, node2):
        """ Returns list of indices for all edges that exist between thse nodes
        i.e,, if edge = (1,2,0) then 0 is the index.
        """
        list_edges = self.find_edges_between_these_nodes(node1, node2)
        return [ed[2] for ed in list_edges]

    def find_paths_between_these_nodes(self, list_nodes):
        """ 
        Find entire path (lsit of edges) in order between nodes in list_nodes.
        Must have unambiguous edge between each adjacent pari of noides in 
        list_nodes.
        e..g, list_ni = [0,5,2]
        --> [(0, 5, 0), (5, 2, 0)]
        NOTE:
        - if want a cycle e..g, 1-->2-->1, with the two edges being different, then
        will resolve ambiguity by taking the edge with lower key first.
        """
        if len(list_nodes)==3 and list_nodes[0]==list_nodes[2]:
            # then you have a cycle.
            from pythonlib.tools.graphtools import path_between_nodepair
            return path_between_nodepair(self.Graph, list_nodes[:2], False)
        else:
            from pythonlib.tools.graphtools import path_through_list_nodes
            # print("PARSER - find_paths_)betwee..")
            # print(list_nodes)
            # self.plot_graph()
            return path_through_list_nodes(self.Graph, list_nodes)

    def find_pts_between_these_nodes(self, list_nodes, concat=False):
        """ Returns the unambiguous (fails otherwise) path between 
        list of nodes in list_nodes. Must give enough info in list_nodes
        so that there is one and only one path.
        IN:
        - list_nodes, e.g, [0,7,2]
        OUT:
        - pts, 
        --- list of pts for each edge (if concat=False)
        --- Nx2 array (if concat==True)
        NOTE:
        - will ensure that the order of pts is correct
        """

        # get list of edges between these nodes
        list_edges = self.find_paths_between_these_nodes(list_nodes)

        # Get list of edges
        list_edges_pts = []
        for ed, anchor_node in zip(list_edges, list_nodes[:-1]):
            pts = self.get_edge_pts(ed, anchor_node=anchor_node)
            list_edges_pts.append(pts)
            
        if concat==True:
            return np.concatenate(list_edges_pts, axis=0)
        else:
            return list_edges_pts

    def find_pts_this_path_edges(self, list_edges):
        """ 
        Unambiguous path
        """

        path = self.convert_list_edges_to_directed(list_edges)
        list_pts = [self.get_edge_pts(ed, anchor_node =ed[0]) for ed in path]
        pts = np.concatenate(list_pts, axis=0)
        return pts


    def convert_list_edges_to_directed(self, list_edges):
        """ outputs something like (1, 2, 0), (2, 5, 0), ...
        where node1 for ed[0] is same as node0 for ed[1], ..
        and so on.
        NOTE:
        - checks that is already directed. if so, then doestnt do anything
        """

        if self.check_list_edges_directed(list_edges):
            return list_edges
        else:
            def _run(list_edges, anchor_node):
                list_edges_directed = []
                failed=False
                for ed in list_edges:
                    if ed[0]==anchor_node:
                        list_edges_directed.append(ed)
                        anchor_node = ed[1]
                    elif ed[1]==anchor_node:
                        list_edges_directed.append((ed[1], ed[0], ed[2]))
                        anchor_node = ed[0]
                    else:
                        failed = True
                        break
                return list_edges_directed, failed

            # Try combo of (1) permutations (ordering)
            # and two possible anchor nodes to initialize
            for anchor_node_ind in [0, 1]:
                anchor_node = list_edges[0][anchor_node_ind]
                list_edges_directed, failed = _run(list_edges, anchor_node)
                if not failed:
                    return list_edges_directed

            # If got to here, then try permutations
            from pythonlib.tools.listtools import permuteRand
            perm_list_edges = permuteRand(list_edges, N=1000, not_enough_ok=True)
            for this_list_edges in perm_list_edges:
                for anchor_node_ind in [0, 1]:
                    anchor_node = this_list_edges[0][anchor_node_ind]
                    list_edges_directed, failed = _run(this_list_edges, anchor_node)
                    if not failed:
                        return list_edges_directed

            # got to here, must have failed
            print(list_edges)
            self.plot_graph()
            assert False

            # anchor_node = list_edges[0][0]
            # list_edges_directed, failed = _run(list_edges, anchor_node)
            # if failed:
            #     # try seeding in other direction
            #     anchor_node = list_edges[0][1]
            #     list_edges_directed, failed = _run(list_edges, anchor_node)
            #     if failed:
            #         print(list_edges)
            #         self.plot_graph()
            #         assert False
            # return list_edges_directed

    def check_list_edges_directed(self, list_edges):
        """ Return True if list edges directed, i.e.,, end n odes are 
        same as start node for next edge.
        """
        for ed1, ed2 in zip(list_edges[:-1], list_edges[1:]):
            if ed1[1]!=ed2[0]:
                return False
        return True

    def check_edges_share_node(self, edge1, edge2):
        """ Returns True iff edges share at least one node
        """
        for node in edge1[:2]:
            if node in edge2[:2]:
                return True
        return False

    def check_nodes_make_unclosed_loop(self, node1, node2):
        """ Unclosed loop is like a "C". i.e, each nodes neighbords
        are only the othe rnode. also, nodes are not the same. Also, there must
        only be one edge between
        OUT:
        - bool
        """
        list_edges = self.find_edges_between_these_nodes(node1, node2)
        list_neigh1 = list(self.Graph.neighbors(node1))
        list_neigh2 = list(self.Graph.neighbors(node2))

        if len(list_edges)==1 and list_neigh1 == [node2] and list_neigh2==[node1]:
            return True
        return False

    # for each node, check how close it is to every other path
    def min_dist_node_to_edge(self, node, edge):
        """ returns min distance from this node to any pt along
        this path
        - node, int
        - edge, (n1, n2, key)
        OUT:
        - d, ind,
        where ind is index in pts (for edge) of min
        """
        assert node not in edge[:2], "must compare node to other edges"
        o = self.Graph.nodes[node]["o"]
        return self.min_dist_pt_to_edge(o, edge)

    def min_dist_pt_to_edge(self, pt, edge):
        """
        INPUT:
        - pt, xy coord.
        - edge, edge index.
        Retunrs
        - the distance
        - the ind along edge that has this distance.
        NOTE:
        - ind is using the "canonical" orietntaion, whichi s the 
        hard coded order of pts in edges.
        """
        G = self.Graph
        pts = G.edges[edge]["pts"]
        
        dists = np.linalg.norm(pts-pt, axis=1)
        ind = np.argmin(dists)
        return dists[ind], ind


    def find_close_edge_to_pt(self, pt, thresh=10, dosort=False):
        """ Find list of edges which are close to pt.
        Also return the ind along the edge.
        RETURNS:
        - edges, list of edges
        - dists, 
        - inds, 
        (All lists)
        NOTE:
        - sorted, then returns sorted from closest to furthers
        """
        G = self.Graph
        edges = []
        dists = []
        inds = []

        for e in G.edges: 
            d, ind = self.min_dist_pt_to_edge(pt, e)
            if d<thresh:
                edges.append(e)
                dists.append(d)
                inds.append(ind)

        if dosort:
            x = [(e,d,i) for e,d,i in zip(edges, dists, inds)]
            x = sorted(x, key=lambda x: x[1])
            edges = [xx[0] for xx in x]
            dists = [xx[1] for xx in x]
            inds = [xx[2] for xx in x]

        return edges, dists, inds

    def find_close_nodes_to_pt(self, pt, thresh=10, only_one=False, take_closest=False):
        """
        only_one, then fails if >1. ok if 0.
        take_closest, then takes closest if there are multiple.
        OUT:
        - list_nodes, list of ints. empty if none.
        """

        list_nodes = []
        distances = []
        for n in self.Graph.nodes:
            o = self.Graph.nodes[n]["o"]
            d = np.linalg.norm(pt-o)
            if d<thresh:
                list_nodes.append(n)
                distances.append(d)
        if only_one:
            if len(list_nodes)>1:
                assert False
        if take_closest:
            if len(list_nodes)>0:
                tmp = [(n,d) for n,d in zip(list_nodes, distances)]
                tmp = sorted(tmp, key=lambda x:x[1])
                list_nodes = [tmp[0][0]]
        return list_nodes


    def find_cycles(self):
        """ Find all cycles, including self-loops, 2-nde cycles, etc.
        OUT:
        - list of cycles, each cycle as list of edges, each edge a 3-tuple
        """
        from networkx.algorithms import cycles
        from networkx.classes.function import selfloop_edges
        # from pythonlib.tools.graphtools import find_all_cycles
        from pythonlib.tools.graphtools import find_all_cycles_edges
        from pythonlib.tools.graphtools import path_between_nodepair

        list_cycles_edges = []

        # Find all cycles (1 node)
        edges = selfloop_edges(self.Graph, keys=True)
        for ed in edges:
            list_cycles_edges.append([ed])
        # print("1", list_cycles_edges)

        # Find all cycles (2+ nodes)
        # self.plot_graph()
        # list_cycles = find_all_cycles(self.Graph)
        list_cycles = find_all_cycles_edges(self.Graph)

        # # - convert to list of eddges
        # for cy in list_cycles:
        #     if len(cy)==1:
        #         # skip, this already gotten as self-loops
        #         continue
        #     if len(cy)==2:
        #         list_edge = path_between_nodepair(self.Graph, cy, False)
        #     else:
        #         list_edge = self.find_paths_between_these_nodes(cy + [cy[0]])

        #     list_cycles_edges.append(list_edge)
        #     print("2", list_cycles_edges)            

        # - convert to list of eddges
        for cy in list_cycles:
            if len(cy)==1:
                # skip, this already gotten as self-loops
                continue
            list_cycles_edges.append(cy)
            # print("2", list_cycles_edges)            
        return list_cycles_edges
                

    ############# USE STROKES TO FIND/MANIPULATE NODES AND EDGES
    def map_strokes_to_nodelists(self, strokes, thresh=5):
        """ finds nodes aligned to strokes
        OUT:
        - list of paths, each path a list of nodes of len =2
        NOTE:
        - will fail iuf cannot find for some reason.
        """

        list_of_paths = []
        for strok in strokes:
            path = []
            for ind in [0, -1]:
                pt = strok[ind,:2]
                list_nodes = self.find_close_nodes_to_pt(pt, thresh)

                if len(list_nodes)==0:
                    print(pt)
                    print([self.Graph.nodes[n]["o"] for n in self.Graph.nodes])
                    assert False, "this pt not close to a node, need to add fifrst?"
                elif len(list_nodes)>1:
                    print(pt)
                    print([self.Graph.nodes[n]["o"] for n in self.Graph.nodes])
                    assert False, "found > 1 node, must merge close nodes first"
                else:
                    node = list_nodes[0]

                path.append(node)
            list_of_paths.append(path)
        return list_of_paths



    def map_strokes_to_edgelist(self, strokes, thresh=5, 
        assert_endpoints_on_nodes=True, only_take_edges_for_active_node=True, 
        must_use_all_edges=False, no_multiple_uses_of_edge=False):
        """ 
        [GOOD]
        returns list of edges [(1,2,0), ...] that the input stroke is along a trajectory of.
        makes sure that if a pair of nodes
        have multiple possible edgse, takes teh edge that occurs more often (as you travel)
        along the pts
        INPUT:
        - assert_endpoints_on_nodes, then fails if the endpoints of strokes are not close to a node.
        - only_take_edges_for_active_node, then state maintains active node, and only keep edges that
        contain this node. Generally have assert_endpoints_on_nodes True, since then makes sure that at all times
        an active node will exist.
        - must_use_all_edges, then fails if each edge is not used.
        - no_multiple_uses_of_edge, then fails if an edge is used twice, over all strokes.
        """


        def _traj_to_edgelist(traj):
            """ traj is Nx2
            """

            last_visited_node = None
            tracker = {}
            list_of_edges = []
            for i, pt in enumerate(traj[:,:2]):

                nodes = self.find_close_nodes_to_pt(pt, only_one=True, thresh=thresh, take_closest=True)
                edges, dists, inds = self.find_close_edge_to_pt(pt, thresh=thresh)

                if only_take_edges_for_active_node:
                    if len(nodes)==1:
                        if last_visited_node is None:
                            last_visited_node = nodes[0]

                        elif last_visited_node is not None and nodes[0]!=last_visited_node:
                            # then you have visisted a new node. 
                            # if change last_visited_node, then pick the most visited edge, then reset.
                            new_node = nodes[0]
                            # print(i)
                            # print(last_visited_node)
                            # print(nodes)
                            # print("HERE", tracker)
                            # # pick out candidate edges that explain the just finished traj
                            # edges_candidate = [ed for ed in tracker.keys() if set(ed[:2])==set([last_visited_node, new_node])]

                            # sort all candidate edges by how oten they visited
                            edges_candidate = [(ed, nvisit) for ed, nvisit in 
                                tracker.items() if set(ed[:2])==set([last_visited_node, new_node])]

                            # sort and pick out most highly visited edge.
                            edges_candidate = sorted(edges_candidate, key=lambda x: x[1]) # sort in ascending order.
                            edge_just_done = edges_candidate[-1][0]

                            if edge_just_done[0]==new_node:
                                edge_just_done = (edge_just_done[1], edge_just_done[0], edge_just_done[2]) # mnake sure is in order.
                            list_of_edges.append(edge_just_done)

                            # reset everything
                            last_visited_node = nodes[0]
                            tracker = {}
                        else:
                            # do nothing, since nthing changed
                            pass

                    # only keep edges that involve the current node
                    edges = [ed for ed in edges if last_visited_node in ed[:2]]
    
                    # update tracker
                    for ed in edges:
                        if ed in tracker.keys():
                            tracker[ed] +=1
                        else:
                            tracker[ed] = 1
                else:
                    for ed in edges:
                        if ed not in list_of_edges:
                            list_of_edges.append(ed)

                if assert_endpoints_on_nodes:
                    if i==0 or i==traj.shape[0]:
                        # print(i,pt)
                        # print([self.Graph.nodes[n]["o"] for n in self.Graph.nodes])
                        assert len(nodes)==1, "on and off should match a node..."


            # Only keep edges that are "fully" within the traj
            # i.e., traj can be multiple edge, but cant be vice versa.
            from pythonlib.tools.distfunctools import furthest_pt_twotrajs
            tmp = []
            for ed in list_of_edges:
                pts = self.get_edge_pts(ed)
                dist = furthest_pt_twotrajs(pts, traj[:, :2], assymetry=1)[0]
                if dist<thresh:
                    tmp.append(ed)
            list_of_edges = tmp

            return list_of_edges

        list_paths = [_traj_to_edgelist(traj) for traj in strokes]

        # check that all edges are used up
        if must_use_all_edges:
            list_edges_all = [edge for list_edge in list_paths for edge in list_edge]
            for ed in self.Graph.edges:
                assert self.edge_is_in(list_edges_all, ed)

        if no_multiple_uses_of_edge:
            # check that all edges are unique
            list_edges_all = [edge for list_edge in list_paths for edge in list_edge]
            assert len(list(set(list_edges_all)))==len(list_edges_all)

        return list_paths


    def map_strokes_to_nodelists_entiretraj(self, strokes, thresh=5, assert_endpoints_on_nodes=True):
        """ finds nodes aligned to strokes. includes all nodes, that are found along the 
        trajectory. 
        OUT:
        - list of paths, each path a list of nodes of at lest len=2, but more if more are encountered.
        - e.g,,: [[4, 2, 1], [3, 4, 5, 6], [8, 2, 0], [8, 5, 7]], means there are 4 strokes, each going
        thru those nodes in those order.

        NOTE:
        - can guarantee that these nodes should be in order (modulo reversal)
        - asserts that will find something for the first and last pts.
        """

        list_of_paths = []
        for strok in strokes:
            path = []
            # start from first pt, and continue
            for i in range(strok.shape[0]):
                pt = strok[i,:2]
                list_nodes = self.find_close_nodes_to_pt(pt, thresh)

                if assert_endpoints_on_nodes:
                    if i==0 or i==strok.shape[0]:
                        assert len(list_nodes)==1, "must find a node at start and end."

                if len(list_nodes)==1:
                    node = list_nodes[0]
                    if node not in path:
                        path.append(node)
                
                if len(list_nodes)>1:
                    print(pt)
                    print([self.Graph.nodes[n]["o"] for n in self.Graph.nodes])
                    assert False, "found > 1 node, must merge close nodes first"

            list_of_paths.append(path)
        return list_of_paths


    ########## LOW-LEVEL HELPERS WITH EDGES AND NODES
    def get_edge_helper(self, edge):
        """ Help return correct index for edge, given vareity of 
        ways of inputing edge here
        INPUT:
        - edge, e.g.,:
        --- edge = (1,2), then looks for one edge that is (1,2,x) where
        x is idx. 
        --- edge = (2,1,0), which returns the correct ordering of 1 and 2.
        OUT:
        - (n1, n2, idx)
        NOTE:
        - in all cases, fails if doiesnt find one and only one edge.
        """

        list_edges = []
        for ed in self.Graph.edges:
            if set(edge[:2]) != set(ed[:2]):
                continue

            if len(edge)>2:
                # also check the index
                if edge[2]!=ed[2]:
                    continue

            # passes all tests. keep
            list_edges.append(ed)

        if len(list_edges)==0 or len(list_edges)>1:
            print(list_edges)
            print(edge)
            print(self.Graph.edges)
            assert False, "didnt find exactly one edge"

        return list_edges[0]

    def edge_is_in(self, list_edges, edge):
        """ smart checking of ehtehr edge is in list_edges
        Solves problem of first 2 nodes can be in any order
        OUT:
        - True if edge is in list_edges, False otherwise
        """
        for ed in list_edges:
            if self.get_edge_helper(ed)==self.get_edge_helper(edge):
                return True
        return False


    def get_edge_dict(self, edge):
        """ Return dict fo this edge
        e..g, dict[pts] = [...
        """
        assert isinstance(edge, tuple)
        edge = self.get_edge_helper(edge)
        return self.Graph.edges[edge]


    def get_edge_pts(self, edge, anchor_node=None):
        """ return pts, array (N,2)
        INPUT:
        - anchor_node, pts will be oriented (flipped) so that
        pts[0,:] will be closer to anchor_node, than will be pts[-1]
        --- if None, then returns in the saved order.
        """

        pts = self.get_edge_dict(edge)["pts"]
        if len(pts)==0:
            print(edge, anchor_node)
            print(self.G.edges)
            assert False, "this edge no pts.."

        if anchor_node is None:
            return pts
        else:
            anchor_pts = self.get_node_dict(anchor_node)["o"]
            doflip = not self.check_pts_orientation(pts, anchor_pts, return_true_on_tie=True)
            if doflip:
                pts = np.flipud(pts)
            return pts

    def get_edge_length(self, edge):
        """ get lenght of edge
        """
        from pythonlib.drawmodel.features import strokeDistances
        pts = self.get_edge_pts(edge)
        return strokeDistances([pts])

    def get_node_dict(self, node):
        """ Returns dict for this node.
        fails if doesnt exist
        """
        return self.Graph.nodes[node]

    def get_nodes_from_list_edges(self, list_edges, in_order=False):
        """ return list of unique nodes, in sorted order,
        INPUT:
        - in_order, then keeps the exact order. requires entering edges that are 
        "directed", in that end nodes are identical to onset nodes. if is not liek this,
        tries to correct, if can't then fails. NOTE: will not all be unique. e.g, if pass in
        (ed1, ed2), then will output (ed1[0], ed1[1], ed2[1])
        """
        if len(list_edges)==0:
            return []
        if in_order:
            list_edges = self.convert_list_edges_to_directed(list_edges)
            list_nodes = [list_edges[0][0]]
            for ed in list_edges:
                list_nodes.append(ed[1])
            return list_nodes
        else:
            list_nodes = []
            for ed in list_edges:
                list_nodes.extend(ed[:2])
            return sorted(set(list_nodes))

    ######### CYCLES
    def is_isolated(self, list_edges):
        """ returns True if list_edges forms an isolated
        cycle. False if there are branches coming off.
        Tests by asking if every edge coming out
        of every node is in list_edges. if so, then this must be isoalted
        """
        all_edges = []
        for ed in list_edges:
            for node in ed[:2]:
                edges_connected = self.find_edges_connected_to_this_node(node)
                for ed in edges_connected:
                    if self.edge_is_in(list_edges, ed) is False:
                        # then this edge is not in the original lsit of edges.
                        return False
        # Then passed all tests. 
        return True

        
    ######### FIND PARSES
    def findparses_filter(self, filtdict, return_inds=False, is_base_parse=False):
        """ filter parses. each parse is  adict. filters by keywords.
        see code within
        RETURNS:
        - list of parses that pass filters
        """

        from pythonlib.tools.dicttools import filterDict
        if is_base_parse:
            ParsesList = self.ParsesBase
        else:
            ParsesList = self.Parses
        list_parses = filterDict(ParsesList, filtdict)
        if return_inds:
            return [P["index"] for P in list_parses]
        else:
            return list_parses

    ########## FILTER/CLEAN UP PARSES
    def filter_single_parse(self, ver, ind):
        """ returns True/False, where True means
        bad (should remove)
        """

        if ver=="any_path_has_redundant_edge":
            """ fails whether same or diff dir
            only checks within each path (turn on yourself).
            # 1) Find parses where there are paths where repeat an edge (in the same path)
            # could repeat in same or diff direction
            """
            list_of_paths = self.extract_parses_wrapper(ind, "list_of_paths")
            PS = self.extract_parses_wrapper(ind)[0]

            def _path_does_repeat_edge(path):
                """ returns True if this path has a repeated edge (can be either dir)
                """
                for i in range(len(path)):
                    for ii in range(i+1, len(path)):
                        if PS._edges_are_same(path[i], path[ii]):
                            return True
                return False

            # check each path
            if any([_path_does_repeat_edge(path) for path in list_of_paths]):
                return True
        if ver=="any_two_paths_have_common_edge":
            """ this common edge can be in either direction
            """
            list_of_paths = self.extract_parses_wrapper(ind, "list_of_paths")
            PS = self.extract_parses_wrapper(ind)[0]

            # conpare all pairs of paths
            for i in range(len(list_of_paths)):
                for ii in range(i+1, len(list_of_paths)):
                    path1 = list_of_paths[i]
                    path2 = list_of_paths[ii]
                    for e1 in path1:
                        for e2 in path2:
                            if PS._edges_are_same(e1, e2):
                                return True
        return False

    def filter_all_parses(self, ver, do_remove=True):
        """ 
        Rturns list of True False same len as parses, where True means is bad and should
        remove. 
        if do_remove, then does remove from self.Parses
        """

        badlist = [self.filter_single_parse(ver, i) for i in range(len(self.Parses))]
        self.Parses = [self.Parses[i] for i, x in enumerate(badlist) if not x]
        print(f"Removed {[i for i, x in enumerate(badlist) if x]}")
        print(f"Removed {sum(badlist)} parses.")
        return badlist
            


    ########## POST PROCESSING
    def finalize(self, convert_to_splines_and_deinterp=True,
        dist_int=10., fail_if_finalized=False, force_just_this_ind=None):
        """ converts all parses into format for analysis, coordiantes, etc.
        INPUT:
        - fail_if_finalized, True, then fails otherwise, skips all steps. assumes your
        previous finalizing is what you wanted to do.
        - force_just_this_ind, then forces to reprocess this ind. doesnt matter whether
        other stuff already finalized, since redoces this ind from stracth
        NOTES:
        - Puts all finalized things into:
        - only modifies self.Parses[ind]["strokes"]
        - 
        """
        from pythonlib.drawmodel.splines import strokes2splines

        def _process(strokes):
            # Translate coords back
            strokes = self.strokes_translate_back(strokes)
            # Spline and deinterp
            if convert_to_splines_and_deinterp:
                strokes = strokes2splines(strokes, dist_int=dist_int)
            return strokes

        if force_just_this_ind is not None:
            ind = force_just_this_ind
            
            self.Parses[ind]["strokes"] = self.parses_to_strokes(ind)

            strokes = _process(self.Parses[ind]["strokes"])
            assert strokes is not None
            self.Parses[ind]["strokes"] = strokes 

            self._parses_fill_in_missing_keys()
            self._parses_give_index()

        else:
            if hasattr(self, "Finalized"):
                if self.Finalized:
                    if fail_if_finalized:
                        assert False, "already finaled.."
                    else:
                        # skip
                        return

            self.parses_fill_in_all_strokes()


            # parses
            for i in range(len(self.Parses)):
                strokes = _process(self.Parses[i]["strokes"])
                assert strokes is not None
                self.Parses[i]["strokes"] = strokes 

            self._parses_fill_in_missing_keys()
            self._parses_give_index()

            self.Finalized=True

    def _parses_reset_perm_of_list(self):
        """ Hack, since some cases saved so that same list across all parses,
        so when apend to list, affects all parses. fixed on 9/25/21, but some
        still saved on disk, so run this.
        """

        for ParsesList in [self.Parses, self.ParsesBase]:
            for P in self.Parses:
                for k in  ["perm_of_list", "bestperm_beh_list", "bestperm_of_list"]:
                    P[k] = list(P[k]) # make copy

    def _parses_fill_in_missing_keys(self, force=False):
        """ 
        """

        default_keys = {
            "manual":False,
            "note": "",
            "rule": None,
            "perm_of_list":[],
            "bestperm_beh_list": [],
            "bestperm_of_list":[]
        } 

        def _default(key):
            if key in ["manual", "note", "rule"]:
                return default_keys[key]
            elif key in ["perm_of_list", "bestperm_beh_list", "bestperm_of_list"]:
                return []

        for P in self.Parses:
            for k, v in default_keys.items():
                if force:
                    P[k] = _default(k)
                else:
                    if k not in P.keys():
                        P[k] = _default(k)

        if hasattr(self, "ParsesBase"):
            for P in self.ParsesBase:
                for k, v in default_keys.items():
                    if force:
                        P[k] = _default(k)
                    else:
                        if k not in P.keys():
                            P[k] = _default(k)


    def _parses_give_index(self):
        for i, P in enumerate(self.Parses):
            P["index"] = i


    def strokes_translate_back_batch(self, convert_to_splines_and_deinterp=True,
        dist_int=10.):
        """ Return list of all parses, in orriginal coords
        IN:
        - convert_to_splines_and_deinterp, uses 10 pix in between pts. uses
        codes from pyBPL. Note does both processes in same code currently.
        OUT:
        - strokes_task, original strokes_task, but made sure to undergo same processing
        as the parses.
        - parses, list of parse
        """
        from pythonlib.drawmodel.splines import strokes2splines
        
        self.parses_fill_in_all_strokes()

        def _process(strokes):
            # Translate coords back
            strokes = self.strokes_translate_back(strokes)
            # Spline and deinterp
            if convert_to_splines_and_deinterp:
                strokes = strokes2splines(strokes, dist_int=dist_int)
            return strokes

        # Stroke
        strokes_task = _process(self.StrokesInterp)

        # parses
        parses_list = []
        for i in range(len(self.Parses)):
            strokes = _process(self.Parses[i]["strokes"])
            parses_list.append(strokes)
        
        return strokes_task, parses_list


    ########## PRINTING
    def print_parse_info(self, ind):
        for p in self.Parses[ind]["list_ps"]:
            p.print_elements(verbose=False)

    def print_parse_concise(self, ind):
        out = f"{ind} = "
        for i, p in enumerate(self.Parses[ind]["list_ps"]):
            out+=f"{i}:{p.EdgesDirected} - "
            # p.print_elements(verbose=False)        
        print(out)

    def print_parse_concise_all(self):
        """ print all parses, each a single line"""
        for i in range(len(self.Parses)):
            self.print_parse_concise(i)

    def print_graph(self):
        nodes = self.Graph.nodes
        edges = self.Graph.edges
        print(f"NODES: {nodes} -- EDGES: {edges}")


    ########## PLOTTING
    def plot_skeleton(self):
        """ plots the bw image
        Note: the orientation might differ from the strokes. This is ok, since 
        once convert back to strokes, will fix the orientation
        """
        if self.Skeleton is not None:
            # plt.figure()
            # plt.imshow(image, cmap="gray")
            # plt.title("before skel")
            plt.figure()
            plt.imshow(self.Skeleton, cmap='gray')
            plt.title("after skel")
        else:
            print("need to first make skeleton")

    def plot_graph(self, graph=None):
        """ plots skeleton graph overlaid on skeleton
        """

        def _plot_graph(ax, nodecol="w"):
            # draw edges by pts
            for (s,e, i) in graph.edges:
                ps = graph.edges[(s, e, i)]['pts']
                ax.plot(ps[:,1], ps[:,0], "--")
                
            # draw node by o
            nodes = graph.nodes()
            ps = np.array([nodes[i]['o'] for i in nodes])
            ax.plot(ps[:,1], ps[:,0], 'o', color=nodecol)
            for i in nodes:
                pt = nodes[i]["o"]
                ax.text(pt[1], pt[0], i, color="r", size=15)
            ax.set_xlabel("y")
            ax.set_ylabel("x")

        if graph is None:
            graph = self.Graph
        
        # draw image
        fig, axes = plt.subplots(1,2, figsize=(25,8))

        # 1) Overlay on image
        ax = axes[0]
        ax.imshow(self.Skeleton, cmap='gray')
        _plot_graph(ax)

        # 2) graph alone
        ax = axes[1]
        # ax.imshow(self.Skeleton, cmap='gray')
        _plot_graph(ax, "k")

        return fig

    def findparses_bycommand(self, inds, params=None, is_base_parse=False):
        """ inds is gneral purpose command
        INPUT:
        - inds,
        --- list of ints
        --- int, then pltos this many random ones
        --- None, then all
        --- str, other commands, see below.
        - params, dict, flexiblty hold params
        - is_base_parse, which set of parses to search
        """
        import random
        
        if params is None:
            params = {}
        if is_base_parse==True:
            ParsesList = self.ParsesBase
        else:
            ParsesList = self.Parses

        if inds==None:
            # then plot all
            inds = range(len(ParsesList))
        elif isinstance(inds, int):
            inds = min([inds, len(ParsesList)])
            inds = sorted(random.sample(range(len(ParsesList)), inds))
        elif isinstance(inds, str):
            if inds=="manual":
                # then only manually entered parses
                inds = self.findparses_filter({"manual":[True]}, True, is_base_parse=is_base_parse)
            elif inds=="manual_and_perms":
                # Then all manual and their permutations. in overall sorted order.
                inds = self.findparses_bycommand("manual", is_base_parse=is_base_parse)
                if len(inds)>0:
                    inds_perm = self.findparses_filter({"permutation_of":inds}, True, is_base_parse=is_base_parse)
                    inds = sorted(inds + inds_perm)
            elif inds=="base_parses":
                # NOTE: old version, looks in self.Parses for anything that is not a permutation., will have None for permutation_of
                inds = [i for i, p in enumerate(ParsesList) if p["permutation_of"] is None]
                # saniyt check, confirme that all referneces to bases have been accounted for
                for i, p in enumerate(ParsesList):
                    if p["permutation_of"] is None:
                        assert i in inds, "this parse refers to something that I did not extract as a base parse/.//"
            elif inds=="permutation_of":
                # permutation of params["of"]
                # perms are ints.
                assert isinstance(params["of"], int)
                inds = [i for i, p in enumerate(ParsesList) if p["permutation_of"]==params["of"]]
            elif inds=="is_best_perm":
                # all that are called "best_perm"
                assert False, "this ver failed if tried to enter new parse that already existsed... use best_perm_of"
                inds = [i for i, p in enumerate(ParsesList) if "bestperm" in p["note"]]
            
            elif inds=="best_perm_of":
                # GOOD - returns the single (or multiple) parses that is the best perm for the input:
                # if you tell me the beh trial you want, then will be single.
                # otherwise could be multiple.
                # always returns a list
                ind_base_parse = params[0] # ind of the base parse. ("base", 0) [new] or 0 [old]
                trial_tuple = params[1] # ('Pancho', 'gridlinecircle', 'linetocircle', '210823-1-587')
                # meaning (animal, expt, rule, trialcode)
                # Note: can leave any item in there as None if want to ignore,e .g, (Pancho, None, None, None)
                # will get all trials by Pancho
                assert len(trial_tuple)==4

                list_par = []
                list_par_beh = []

                def _matches_trialtuple(tp, trial_tuple):
                    # filter by trialtuple (fail if any items dont match)
                    # ignore trialtuple item if it is None
                    for x, y in zip(tp, trial_tuple):
                        if y is None:
                            continue
                        if not x == y:
                            return False
                    return True

                inds = []
                for i, p in enumerate(ParsesList):
                    if "bestperm_of_list" in p.keys():
                        assert len(p["bestperm_of_list"])==len(p["bestperm_beh_list"])
                        # bestperm_of_list = [("base, 0"), ...]
                        # bestperm_beh_list = [trialtuple1, tuple2, ..]
                        
                        # see if any tuple match as a combo
                        for _of, _beh in zip(p["bestperm_of_list"], p["bestperm_beh_list"]):
                            if _of==ind_base_parse and _matches_trialtuple(_beh, trial_tuple):
                                # take this parse
                                # print(_of, ind_base_parse)
                                # print(_beh, trial_tuple)
                                inds.append(i)
                                break

                #             indtmp = p["bestperm_of_list"].index(ind_base_parse)
                #             tp = p["bestperm_beh_list"][indtmp]

                #             if i==100:
                #                 print("here")
                #                 print(indtmp)
                #                 print(tp)
                #                 print(ind_base_parse)
                #                 print(p["bestperm_of_list"])
                #                 print(p["bestperm_beh_list"])
                #                 asdsa
                #             # filter by trialtuple (fail if any items dont match)
                #             skip=False
                #             for x, y in zip(tp, trial_tuple):
                #                 if y is None:
                #                     continue
                #                 if not x == y:
                #                     skip=True
                #             if not skip:
                #                 list_par.append(i)
                #                 list_par_beh.append(tp) # matching trialtuples
                #             if skip:
                #                 print(tp, " -- ", trial_tuple)
                # inds = list_par
                inds = sorted(set(inds))

                if False:
                    # print the tuples that were found
                    print("this parse is the best-perm for baseparse: ", ind_base_parse)
                    print(list_par)
                    for x in list_par_beh:
                        print(x)


            elif inds=="permutation_of_v2":
                # Better than permutation_of
                # perms are tuples, e.g, ("base", 1) or ("notbase", 2)
                baseind = (params["parsekind"], params["ind"])
                inds = [i for i, p in enumerate(ParsesList) if baseind in p["perm_of_list"]]
            elif inds=="rule":
                """ get inds that follow this rule"""
                list_rule = params["list_rule"]
                inds = []
                for i, p in enumerate(ParsesList):
                    if "rule" not in p.keys():
                        continue
                    if p["rule"] in list_rule:
                        inds.append(i)

            elif inds=="find_perms_for_a_rule":
                """ rule parses should be in self.ParsesBase
                finds perms that are stored in self.Parse
                """
                rule = params["rule"]

                # find inds for the base parses for this rule
                list_indparse = self.findparses_bycommand("rule", {"list_rule":[rule]}, is_base_parse=True)
                if len(list_indparse)==0:
                    print("didnt find any base parses for this rule", rule)
                    return []

                # print("base parses for this rule ", rule, "are: ", list_indparse)

                # Find all perms across these 
                inds = []
                for indparse in list_indparse:
                    # find all perms that are of this 
                    x = self.findparses_bycommand("permutation_of_v2", {"parsekind": "base", "ind":indparse}, 
                        is_base_parse=False)

                    # print("perms of ", indparse, ": ", x)
                    
                    # print("names of best fit trials for these inds:")
                    # print([p["bestperm_beh_list"] for i, p in enumerate(self.Parses) if i in x])
                    inds.extend(x)
                inds = sorted(set(inds))


            elif inds=="best_fit_helper":
                # helper to find the best-fit parse for this combination of beh (trial_tuple) and rule (base parse)
                # trial_tuple
                # rule = "lolli"
                trial_tuple = params["trial_tuple"]
                rule = params["rule"] # 

                # 1) Find all perms for this rule
                inds_perms = self.findparses_bycommand("find_perms_for_a_rule", {"rule":rule})

                # 2) of those perms, find those that have best fit trial be this trial tupel
                list_indparse = self.findparses_bycommand("rule", {"list_rule":[rule]}, is_base_parse=True)
                if len(list_indparse)==0:
                    print("didnt find any base parses for this rule", rule)
                    assert False
                    # return []

                # print("base parses for this rule ", rule, "are: ", list_indparse)

                inds = []
                for indparse in list_indparse:
                    # find_best_perm_this_beh_and_baseparse
                    x = self.findparses_bycommand("best_perm_of", [("base", indparse), trial_tuple])
                    inds.extend(x)
                # # find inds for the base parses for this rule
                # list_indparse = self.findparses_bycommand("rule", {"list_rule":[rule]}, is_base_parse=True)
                # if len(list_indparse)==0:
                #     print("didnt find any base parses for this rule", rule)
                #     return []

                # print("base parses for this rule ", rule, "are: ", list_indparse)

                # inds = []
                # for indparse in list_indparse:
                #     # find all perms which are best of this indparse
                #     ind_best = self.findparses_bycommand("best_perm_of", [("base", indparse), trial_tuple])
                #     if len(ind_best) !=1:
                #         print(ind_best)
                #         assert False, "there should only be one base fitting parse"
                #     inds.append(ind_best[0])
                # print(inds)
                # assert False

    
            else:
                print(inds)
                assert False
        else:
            assert isinstance(inds, list)
            # you already gave me inds, just s[pit back out]
        
        return inds

    def plot_parses(self, inds=None, Nmax=50):
        """ new, quick, to just plot parses, not in cluding summary states, etc
        INPUT:
        - inds,
        --- list of ints
        --- int, then pltos this many random ones
        --- None, then all
        --- str, other commands, see below.
        NOTE:
        - returns None if no inds.
        """
        from pythonlib.dataset.dataset import Dataset
        import random
        D = Dataset([])
        inds = self.findparses_bycommand(inds)

        if len(inds)==0:
            return None

        if len(inds)>Nmax:
            inds = random.sample(inds, Nmax)
        parses_list = self.extract_all_parses_as_list()
        parses_list = [parses_list[i] for i in inds]
        titles = inds
        titles = [str(i) for i in inds]
        # print(len(parses_list))
        fig = D.plotMultStrokes(parses_list,titles=titles, jitter_each_stroke=True)
        return fig

    def plot_single_parse(self, ind):
        from pythonlib.drawmodel.parsing import plotParses
        parse = self.Parses[ind]["strokes"]
        plotParses([parse], ignore_set_axlim=True)


    def summarize_parses(self, nmax = 50, inds=None):
        from pythonlib.drawmodel.parsing import summarizeParses
        from pythonlib.tools.dicttools import printOverviewKeyValues
        import random
        if len(self.Parses)==0:
            return

        printOverviewKeyValues(self.Parses)

        parses = self.extract_all_parses_as_list()
        self.plot_skeleton()
        self.plot_graph()
        if inds is not None:
            parses = [parses[i] for i in inds]
        print("FOUND THIS MANY PARSES: ", len(parses))
        if len(parses)>nmax:
            parses = random.sample(parses, nmax)
        summarizeParses(parses, plot_timecourse=True, ignore_set_axlim=True)


# from pythonlib.drawmodel.parsing import plotParses, summarizeParses

# # == PLOT PARSES (my plotting fucntion, which shows tiecourse as well
# # === 2) Given Task, general purpose parsing function
# from pythonlib.tools.stroketools import fakeTimesteps
# from pythonlib.drawmodel.strokePlots import plotDatStrokes, plotDatStrokesTimecourse

# plotParses(parses)





################ SCRATCH

# Testing the code for comparing two parses (see if identical)
# See by eye whether they are correct.
# dinv = True
# sinv = True
# for i in range(len(P.Parses)):
#     for ii in range(i+1, len(P.Parses)):
#         if P.parses_is_identical(i, ii, direction_invariant=dinv, strokes_invariant=sinv):
#             print("FOUND:")
#             print(i)
#             P.print_parse_info(i)
#             print(ii)
#             P.print_parse_info(ii)
