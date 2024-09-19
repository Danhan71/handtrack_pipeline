""" 
Can be used for either beh or task ordered tokens
"""

import numpy as np
import matplotlib.pyplot as plt
# (1) Get database for each aniaml
# from pythonlib.dataset.dataset_strokes import DatStrokes
# DS = DatStrokes()
# # For each, convert to semantic
# map_shape_to_shapesemantic_Diego = DS.shapesemantic_stroke_shape_cluster_database(which_basis_set="Diego")
# map_shape_to_shapesemantic_Pancho = DS.shapesemantic_stroke_shape_cluster_database(which_basis_set="Pancho")

# map_shape_to_shapesemantic_Diego = {
#     'Lcentered-4-2-0': 'Lcentered-DL-DL',
#      'Lcentered-4-3-0': 'Lcentered-DR-DR',
#      'Lcentered-4-4-0': 'Lcentered-UR-UR',
#      'V-2-1-0': 'V-LL-LL',
#      'V-2-2-0': 'V-DD-DD',
#      'V-2-4-0': 'V-UU-UU',
#      'arcdeep-4-1-0': 'arcdeep-LL-LL',
#      'arcdeep-4-2-0': 'arcdeep-DD-DD',
#      'arcdeep-4-4-0': 'arcdeep-UU-UU',
#      'circle-6-1-0': 'circle-XX-XX',
#      'line-8-1-0': 'line-LL-LL',
#      'line-8-2-0': 'line-UU-UU',
#      'line-8-3-0': 'line-UR-UR',
#      'line-8-4-0': 'line-UL-UL',
#      'squiggle3-3-1-0': 'squiggle3-LL-0.0',
#      'squiggle3-3-2-0': 'squiggle3-LL-1.0',
#      'squiggle3-3-2-1': 'squiggle3-LL-0.0'}

# map_shape_to_shapesemantic_Pancho = {
#     'Lcentered-4-1-0': 'Lcentered-UL-UL',
#      'Lcentered-4-2-0': 'Lcentered-DL-DL',
#      'Lcentered-4-3-0': 'Lcentered-DR-DR',
#      'Lcentered-4-4-0': 'Lcentered-UR-UR',
#      'V-2-2-0': 'V-DD-DD',
#      'V-2-3-0': 'V-RR-RR',
#      'V-2-4-0': 'V-UU-UU',
#      'arcdeep-4-2-0': 'arcdeep-DD-DD',
#      'arcdeep-4-3-0': 'arcdeep-RR-RR',
#      'arcdeep-4-4-0': 'arcdeep-UU-UU',
#      'circle-6-1-0': 'circle-XX-XX',
#      'line-8-1-0': 'line-LL-LL',
#      'line-8-2-0': 'line-UU-UU',
#      'line-8-3-0': 'line-UR-UR',
#      'line-8-4-0': 'line-UL-UL',
#      'squiggle3-3-1-0': 'squiggle3-LL-0.0',
#      'squiggle3-3-1-1': 'squiggle3-UU-0.0',
#      'squiggle3-3-2-0': 'squiggle3-LL-1.0',
#      'squiggle3-3-2-1': 'squiggle3-LL-0.0',
#      'usquare-1-2-0': 'usquare-DD-DD',
#      'usquare-1-3-0': 'usquare-RR-RR',
#      'usquare-1-4-0': 'usquare-UU-UU',
#      'zigzagSq-1-1-0': 'zigzagSq-LL-1.0',
#      'zigzagSq-1-1-1': 'zigzagSq-LL-0.0',
#      'zigzagSq-1-2-0': 'zigzagSq-UU-1.0',
#      'zigzagSq-1-2-1': 'zigzagSq-UU-0.0'}

# for k, v in map_shape_to_shapesemantic_Diego.items():
#     if k in map_shape_to_shapesemantic_Pancho:
#         assert v==map_shape_to_shapesemantic_Pancho[k]

# MAP_SHAPE_TO_SHAPESEMANTIC = {}
# for map_each_animal in [map_shape_to_shapesemantic_Diego, map_shape_to_shapesemantic_Pancho]:
#     for k, v in map_each_animal.items():
#         if k not in MAP_SHAPE_TO_SHAPESEMANTIC:
#             MAP_SHAPE_TO_SHAPESEMANTIC[k] = v
#         else:
#             assert v==MAP_SHAPE_TO_SHAPESEMANTIC[k]

# HACKY -- use the above code to generate this...
MAP_SHAPE_TO_SHAPESEMANTIC = {
    'Lcentered-4-2-0': 'Lcentered-DL-DL',
    'Lcentered-4-3-0': 'Lcentered-DR-DR',
    'Lcentered-4-4-0': 'Lcentered-UR-UR',
    'V-2-1-0': 'V-LL-LL',
    'V-2-2-0': 'V-DD-DD',
    'V-2-4-0': 'V-UU-UU',
    'arcdeep-4-1-0': 'arcdeep-LL-LL',
    'arcdeep-4-2-0': 'arcdeep-DD-DD',
    'arcdeep-4-4-0': 'arcdeep-UU-UU',
    'circle-6-1-0': 'circle-XX-XX',
    'line-8-1-0': 'line-LL-LL',
    'line-8-2-0': 'line-UU-UU',
    'line-8-3-0': 'line-UR-UR',
    'line-8-4-0': 'line-UL-UL',
    'squiggle3-3-1-0': 'squiggle3-LL-0.0',
    'squiggle3-3-2-0': 'squiggle3-LL-1.0',
    'squiggle3-3-2-1': 'squiggle3-LL-0.0',
    'Lcentered-4-1-0': 'Lcentered-UL-UL',
    'V-2-3-0': 'V-RR-RR',
    'arcdeep-4-3-0': 'arcdeep-RR-RR',
    'squiggle3-3-1-1': 'squiggle3-UU-0.0',
    'usquare-1-2-0': 'usquare-DD-DD',
    'usquare-1-3-0': 'usquare-RR-RR',
    'usquare-1-4-0': 'usquare-UU-UU',
    'zigzagSq-1-1-0': 'zigzagSq-LL-1.0',
    'zigzagSq-1-1-1': 'zigzagSq-LL-0.0',
    'zigzagSq-1-2-0': 'zigzagSq-UU-1.0',
    'zigzagSq-1-2-1': 'zigzagSq-UU-0.0'}


def generate_tokens_from_raw(strokes, shapes, gridlocs=None, gridlocs_local=None,
                             reclassify_shape_using_stroke=False):
    """
    [NOTE: ONLY use this for genreated tokens in default order. this important becuase
    generates and caches. To reorder, see tokens_reorder]
    Designed for grid tasks, where each prim is a distinct location on grid,
    so "relations" are well-defined based on adjacency and direction
    PARAMS:
    - params, dict, like things defining the grid params for this expt
    - inds_taskstrokes, list of ints, order for the taskstrokes. e..g,, [0,4,2]
    if None, then uses the default order.
    - track_order, bool, whether order is relevant. if True (e.g, for behavior)
    then tracks things related to roder (e.g., sequential relations). If False
    (e.g. for chunks where care only about grouping not order), then ignore those
    features.
    - hack_is_gridlinecircle, for gridlinecirlce epxeirments, hacked the grid...
    for both "gridlinecircle", "chunkbyshape2"
    - input_grid_xy, either None (extracts grid params for this task auto), or a
    list of two arrays [gridx, gridy] where each array is sorted (incresaing) scalar coordinates
    for each grid location.
    RETURNS:
    - datsegs, list of dicts, each a token.
    """
    from pythonlib.primitives.primitiveclass import PrimitiveClass, generate_primitiveclass_from_raw

    ############ PREPARE DATA
    # Convert to Primitives, since the rest of code requires that.
    assert len(strokes)==len(shapes)
    Prims = []
    for traj, sh in zip(strokes, shapes):
        P = generate_primitiveclass_from_raw(traj, sh)
        # shape_abstract, scale, rotation, reflect = shape_string_convert_to_components(sh)
        # P = PrimitiveClass()
        # try:
        #     P.input_prim("prototype_prim_abstract", {
        #         "shape":shape_abstract,
        #         "scale":scale,
        #         "rotation":rotation,
        #         "reflect":reflect},
        #         traj = traj)
        # except Exception as err:
        #     print(sh, shape_abstract, scale, rotation, reflect)
        #     assert False
        Prims.append(P)

    inds_taskstrokes = list(range(len(Prims)))

    ################ METHODS (doesnt matter if on grid)
    # def _orient(i):
    #     if _shape(i)=="line-8-1-0":
    #         return "horiz"
    #     elif _shape(i)=="line-8-2-0":
    #         return "vert"
    #     else:
    #         return "undef"
    #     # elif _shape(i) in ["circle-6-1-0", "arcdeep-4-4-0"]:
    #     #     return "undef"
    #     # else:
    #     #     print(_shape(i))
    #     #     assert False, "code it"

    # Spatial scales.
    def _width(i):
        return Prims[i].Stroke.extract_spatial_dimensions(scale_convert_to_int=True)["width"]
    def _height(i):
        return Prims[i].Stroke.extract_spatial_dimensions(scale_convert_to_int=True)["height"]
    def _diag(i):
        return Prims[i].Stroke.extract_spatial_dimensions(scale_convert_to_int=True)["diag"]
    def _max_wh(i):
        return Prims[i].Stroke.extract_spatial_dimensions(scale_convert_to_int=True)["max_wh"]

    def _shapeabstract(i):
        # e.g, "line"
        return Prims[i].ShapeNotOriented

    def _shape(i):
        # return string
        if reclassify_shape_using_stroke:
            return Prims[i].label_classify_prim_using_stroke(return_as_string=True)
        else:
            return Prims[i].shape_oriented(include_scale=True)
        # return objects[i][0]

    ################## GRID METHODS

    # Create sequence of tokens
    datsegs = []

    for i in range(len(Prims)):

        # 1) Things that don't depend on grid
        datsegs.append({
            "shapeabstract":_shapeabstract(i),
            "shape":_shape(i),
            "shape_oriented":_shape(i),
            "width":_width(i),
            "height":_height(i),
            "diag":_diag(i),
            "max_wh":_max_wh(i),
            "Prim":Prims[i],
            "ind_taskstroke_orig":inds_taskstrokes[i],
            "center": Prims[i].Stroke.extract_center(), # in pixels
            "gridloc": gridlocs[i] if gridlocs is not None else ("IGN", "IGN"),
            "gridloc_local": gridlocs_local[i] if gridlocs_local is not None else ("IGN", "IGN"),
            "stroke_index":i
            })

    Tk = Tokens(datsegs, version="beh")
    Tk.sequence_context_relations_calc()

    return Tk

class Tokens(object):
    """
    """
    def __init__(self, tokens, version=None):
        """
        PARAMS:
        - version, string, e.g, "beh" or "task"
        """

        assert isinstance(tokens, (list, tuple))

        if len(tokens)>0:
            assert isinstance(tokens[0], dict)

        if version is not None:
            assert version in ["beh", "task"]
        self.Version = version
        self.Tokens = tuple(tokens) # order is immutable

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        else:
            from pythonlib.tools.checktools import check_objects_identical
            return check_objects_identical(self.Tokens, other.Tokens)
            # return self.Tokens == other.Tokens

    def extract_locations_concrete(self, assign_to_tokens=True):
        """ Return the concrete locations (2-tuples, xy) for each
        tokens, in a list of tuples
        PARAMS:
        - assign_to_tokens, then modifies self.Tokens with new key: loc_concrete
        """
        list_prims = [t["Prim"] for t in self.Tokens]
        list_loc = [p.extract_as("loc_concrete") for p in list_prims]

        if assign_to_tokens:
            for loc, tok in zip(list_loc, self.Tokens):
                tok["loc_concrete"] = loc

        return list_loc

    def feature_location(self, i, ver="grid"):
        """ get location of this token
        PARAMS:
        - ver, str, grid or concrete
        RETURNS:
        - xloc, yloc
        """
        
        if ver=="grid":
            xloc, yloc = self.Tokens[i]["gridloc"]
        elif ver=="concrete":
            # Check that concrete coords are present
            if "loc_concrete" not in self.Tokens[0].keys():
                self.extract_locations_concrete()
            xloc, yloc = self.Tokens[i]["loc_concrete"]
        else:
            print(ver)
            assert False

        return xloc, yloc

    def featurepair_posdiff(self, i, j, ver="grid"):
        """ difference in grid position
        returns (x, y)
        """
        pos1 = self.feature_location(i, ver=ver)
        pos2 = self.feature_location(j, ver=ver)
        return pos2[0]-pos1[0], pos2[1] - pos1[1]

    def featurepair_dist(self, i, j, ver="grid"):
        """ Distance in grid, euclidian scalar
        """
        (x,y) = self.featurepair_posdiff(i, j, ver=ver)
        return (x**2 + y**2)**0.5

    def chunks_update_by_shaperank(self, map_shape_to_rank):
        """ Assign each token to a chunk and a ordinal withn the chunk, based
        on rule that shapes are done in order, such as line --> circle.
        E.g, if did [L, L, L, C, C, L] then will return:
        - chunk_rank = [0 0 0 1 1 2]
        - chunk_within_rank = [0 1 2 0 1 0]
        """

        tokens = self.Tokens

        # 1) get cjhunk rank
        chunk_rank = [map_shape_to_rank[t["shape"]] for t in tokens]

        # 2) get ordinal within the chunk
        # e.g [0 0 1 1 1 0] will map to [0 1 0 1 2 0]
        chunk_within_rank = []
        r_prev = None
        for r in chunk_rank:
            if r==r_prev:
                o = o+1
            else:
                # reset
                o = 0
            r_prev = r
            chunk_within_rank.append(o)
        
        # 3) modify tokens
        for t,r,o in zip(tokens, chunk_rank, chunk_within_rank):
            t["chunk_rank"] = r
            t["chunk_within_rank"] = o
        
        return chunk_rank, chunk_within_rank

    def chunks_update_by_chunksobject(self, C, return_n_in_chunk=False):
        """ Assign each token to a chunk and a ordinal withn the chunk, based
        on the Hier defined in C.Hier. E..g, C could define concrete chunks, like lollis
        [[1,2], [0,4]], and this will assigns toeksn with strokeinds 1 and 2 to chunk 0, a
        and with 0 and 4 to chunk 1.
        EG:
        - C.Hier = [[0, 1], [2, 3], [4, 5]]
        --> ([0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1])

        - C.Hier = [[0, 4], [2, 3], [1, 5]]
        --> ([0, 2, 1, 1, 0, 2], [0, 0, 0, 1, 0, 0])

        - C.Hier = [[2, 3], [0, 1], [4, 5]]
        --> ([1, 1, 0, 0, 2, 2], [0, 1, 0, 1, 0, 1])

        NOTE: the numbers index the order the chunks appear in C.Hier. If Fixed order is False,
        then these numbers are not meaningful, but simply are unique identifiers for items in Hier.

        PARAMS:
        - return_n_in_chunk, bool, if True, then returns for each index the number of 
        items in the chunk that index is in. e.g,, if
        chunk_rank = [0, 0, 0, 1, 1, 1, 2] 
        chunk_within_rank = [0, 1, 2, 0, 1, 2, 0]
        Then:
        chunk_n_in_chunk = [3, 3, 3, 3, 3, 3, 1]
        """

        tokens = self.Tokens
        strokeinds = [tok["ind_taskstroke_orig"] for tok in tokens]
        chunk_rank = C.find_hier_for_list_taskstroke(strokeinds) # e.g, [0, 0, 1, 1, 2, 2]

        # 2) get ordinal within the chunk
        # e.g [0 0 1 1 1 0] will map to [0 1 0 1 2 0]
        chunk_within_rank = []
        r_prev = None
        for r in chunk_rank:
            if r==r_prev:
                o = o+1
            else:
                # reset
                o = 0
            r_prev = r
            chunk_within_rank.append(o)
    
        # For each item get the n items in that item's chunk        
        n_current = chunk_within_rank[-1]
        do_reset = True
        chunk_n_in_chunk = []
        for i in chunk_within_rank[::-1]:
            # Travel backwards.
            
            if do_reset:
                # Then you hit 0 on previous item. or this is the start
                n_current = i+1 # n items in the chunk that is starting.
                do_reset = False
            
            if i==0:
                # Reset on next item
                do_reset = True
            
            chunk_n_in_chunk.append(n_current)
        chunk_n_in_chunk = chunk_n_in_chunk[::-1] # flip so is correct order.

        # get rank from end of chunk, where -1 means is end
        # e.g., if chunk_within_rank = [0,1,2,3,0,1,0, 1, 2, 3, 4]
        # then chunk_within_rank_fromlast = [-4, -3, -2, -1, -2, -1, -5, -4, -3, -2, -1]
        reset=True
        out = []
        for i in chunk_within_rank[::-1]:
            if reset:
                ct=0
                reset=False
            ct+=-1
            if i==0:
                reset=True
            out.append(ct)
        chunk_within_rank_fromlast = out[::-1]

        # 3) modify tokens
        for t,r,o,l,n in zip(tokens, chunk_rank, chunk_within_rank, chunk_within_rank_fromlast, chunk_n_in_chunk):
            t["chunk_rank"] = r
            t["chunk_within_rank"] = o
            t["chunk_within_rank_fromlast"] = l
            t["chunk_n_in_chunk"] = n

            # semantic, n within chunk
            if o==0 and o+1==n:
                chunk_within_rank_semantic = "both_fl"
                chunk_within_rank_semantic_v2 = "both_fl"
            elif o==0:
                chunk_within_rank_semantic = "first"
                chunk_within_rank_semantic_v2 = "first"
            elif o+1==n:
                chunk_within_rank_semantic = "last"
                chunk_within_rank_semantic_v2 = "last"
            else:
                chunk_within_rank_semantic = "middle"
                chunk_within_rank_semantic_v2 = f"{o}"

            if chunk_within_rank_semantic == "both_fl":
                # A unique state
                chunk_within_rank_semantic_v3 = -9
            elif chunk_within_rank_semantic == "first":
                chunk_within_rank_semantic_v3 = 0
            elif chunk_within_rank_semantic == "last":
                chunk_within_rank_semantic_v3 = 99
            elif o == 1:
                # 2nd stroke wins
                chunk_within_rank_semantic_v3 = 1
            elif l == -2:
                chunk_within_rank_semantic_v3 = 98
            else:
                # Middle strokes
                chunk_within_rank_semantic_v3 = 50

            t["chunk_within_rank_semantic"] = chunk_within_rank_semantic
            t["chunk_within_rank_semantic_v2"] = chunk_within_rank_semantic_v2
            t["chunk_within_rank_semantic_v3"] = chunk_within_rank_semantic_v3

            # Semantic, v2


        if return_n_in_chunk:
            return chunk_rank, chunk_within_rank, chunk_n_in_chunk
        else:
            return chunk_rank, chunk_within_rank


    def sequence_gridloc_direction(self, GRIDLOC_VER = "gridloc"):
        """ Get each item's griodloc locatio minus previous item's, with first item in seuqence hvaing (IGN, IGN).
        Appends a new key in self.Tokens["gridloc_rel_prev"], which is 2-tuple of gridloc distance from
        previous item.
        """

        # GRIDLOC_VER = "gridloc_local" # better, since considers just the grid of this task.
        tokens = self.Tokens

        grid_ver = "on_grid"
        for dseg in tokens:
            if dseg[GRIDLOC_VER] == ("IGN",) or dseg[GRIDLOC_VER] == ("IGN", "IGN"):
                grid_ver = "on_rel"

        def _location(i):
            if grid_ver == "on_grid":
                xloc, yloc = tokens[i][GRIDLOC_VER]
            elif grid_ver == "on_rel":
                xloc = "IGN"
                yloc = "IGN"
            else:
                assert False
            return xloc, yloc

        def _posdiffs(i, j):
            # return xdiff, ydiff,
            # in grid units, from i --> j
            if grid_ver == "on_grid":
                pos1 = _location(i)
                pos2 = _location(j)
                return pos2[0]-pos1[0], pos2[1] - pos1[1]
            elif grid_ver == "on_rel":
                return ("IGN", "IGN")
            else:
                assert False

        def _gridloc_rel_prev(i):
            if i==0:
                return (0, "START")
            else:
                return _posdiffs(i-1, i)

        for i, dseg in enumerate(tokens):
            dseg["gridloc_rel_prev"] = _gridloc_rel_prev(i)

    def sequence_context_relations_calc(self):
        """ Get each item's sequence context, entering as new features
        for each tok.
        NOTE: derived from taskgeneral._tokens_generate_relations
        """

        # GRIDLOC_VER = "gridloc"
        GRIDLOC_VER = "gridloc_local" # better, since considers just the grid of this task.
        tokens = self.Tokens

        grid_ver = "on_grid"
        for dseg in tokens:
            if dseg[GRIDLOC_VER] == ("IGN",) or dseg[GRIDLOC_VER] == ("IGN", "IGN"):
                grid_ver = "on_rel"

        def _location(i):
            if grid_ver == "on_grid":
                # print(tok[GRIDLOC_VER] for tok in tokens)
                # print(tokens[i])
                # print(tokens[i][GRIDLOC_VER])
                xloc, yloc = tokens[i][GRIDLOC_VER]
            elif grid_ver == "on_rel":
                xloc = "IGN"
                yloc = "IGN"
            else:
                assert False

            return xloc, yloc

        def _posdiffs(i, j):
            # return xdiff, ydiff, 
            # in grid units, from i --> j
            if grid_ver == "on_grid":
                pos1 = _location(i)
                pos2 = _location(j)
                return pos2[0]-pos1[0], pos2[1] - pos1[1]
            elif grid_ver == "on_rel":
                return "IGN", "IGN"
            else:
                assert False

        def _direction(i, j):
            # only if adjacnet on grid.
            if grid_ver == "on_grid":
                xdiff, ydiff = _posdiffs(i,j)
                if np.isclose(xdiff, 0.) and np.isclose(ydiff, 1.):
                    return "up"
                elif np.isclose(xdiff, 0.) and np.isclose(ydiff, -1.):
                    return "down"
                elif xdiff ==-1. and np.isclose(ydiff, 0.):
                    return "left"
                elif xdiff ==1. and np.isclose(ydiff, 0.):
                    return "right"
                else:
                    return "far"
            elif grid_ver == "on_rel":
                return "IGN"
            else:
                assert False


        def _relation_from_previous(i):
            # relation to previous stroke
            if i==0:
                return "start"
            else:
                return _direction(i-1, i)

        def _horizontal_or_vertical(i, j):
            xdiff, ydiff = _posdiffs(i,j)
            if np.isclose(xdiff, 0.) and np.isclose(ydiff, 0.):
                return "same_location"
                # assert False, "strokes are on the same grid location, decide what to call this"
            elif np.isclose(xdiff, 0.):
                return "vertical"
            elif np.isclose(ydiff, 0.):
                return "horizontal"
            else: # xdiff, ydiff are both non-zero
                return "diagonal" 

        def _horiz_vert_move_from_previous(i):
            if i==0:
                return "start"
            else:
                return _horizontal_or_vertical(i-1, i)


        for i, dseg in enumerate(tokens):
            # 2) Things that depend on grid
            if grid_ver=="on_grid":
                # Then this is on grid, so assign grid locations.
                dseg["rel_from_prev"] = _relation_from_previous(i)
                # dseg["rel_to_next"] = _relation_to_following(i)
                dseg["h_v_move_from_prev"] = _horiz_vert_move_from_previous(i)
                # dseg["h_v_move_to_next"] = _horiz_vert_move_to_following(i)
            elif grid_ver=="on_rel":
                # Then this is using relations, not spatial grid.
                # give none for params
                # datsegs[-1]["rel_to_next"] = None
                # datsegs[-1]["h_v_move_to_next"] = None
                # dseg["rel_from_prev"] = None
                # dseg["h_v_move_from_prev"] = None
                dseg["rel_from_prev"] = "IGN"
                dseg["h_v_move_from_prev"] = "IGN"
            else:
                print(grid_ver)
                assert False, "code it"

            #### Context
            if i==0:
                dseg["CTXT_loc_prev"] = (0, "START") # tuple, so that is same type as gridloc
                dseg["CTXT_loc_prev_local"] = (0, "START") # tuple, so that is same type as gridloc
                dseg["CTXT_shape_prev"] = "START"
                if "loc_on_clust" in dseg.keys():
                    dseg["CTXT_loconclust_prev"] = "START"
                if "loc_off_clust" in dseg.keys():
                    dseg["CTXT_locoffclust_prev"] = "START"
            else:
                dseg["CTXT_loc_prev"] = tokens[i-1]["gridloc"]
                dseg["CTXT_loc_prev_local"] = tokens[i-1]["gridloc_local"]
                dseg["CTXT_shape_prev"] = tokens[i-1]["shape"]
                if "loc_on_clust" in dseg.keys():
                    dseg["CTXT_loconclust_prev"] = tokens[i-1]["loc_on_clust"]
                    # assert isinstance(dseg["CTXT_loconclust_prev"], int)
                if "loc_off_clust" in dseg.keys():
                    dseg["CTXT_locoffclust_prev"] = tokens[i-1]["loc_off_clust"]
                    # assert isinstance(dseg["CTXT_locoffclust_prev"], int)

            if i==len(tokens)-1:
                dseg["CTXT_loc_next"] = ("END", 0)
                dseg["CTXT_loc_next_local"] = ("END", 0)
                dseg["CTXT_shape_next"] = "END"
                if "loc_on_clust" in dseg.keys():
                    dseg["CTXT_loconclust_next"] = "END"
                if "loc_off_clust" in dseg.keys():
                    dseg["CTXT_locoffclust_next"] = "END"
                # else:
                #     print(dseg.keys())
                #     assert False

            else:
                dseg["CTXT_loc_next"] = tokens[i+1]["gridloc"]
                dseg["CTXT_loc_next_local"] = tokens[i+1]["gridloc_local"]
                dseg["CTXT_shape_next"] = tokens[i+1]["shape"]
                if "loc_on_clust" in dseg.keys():
                    dseg["CTXT_loconclust_next"] = tokens[i+1]["loc_on_clust"]
                    # if not isinstance(dseg["CTXT_loconclust_next"], int):
                    #     print(dseg["CTXT_loconclust_next"])
                    #     print(type(dseg["CTXT_loconclust_next"]))
                    #     assert False
                if "loc_off_clust" in dseg.keys():
                    dseg["CTXT_locoffclust_next"] = tokens[i+1]["loc_off_clust"]
                    # assert isinstance(dseg["CTXT_locoffclust_next"], int)
                # else:
                #     assert False
                # assert False

    def print_summary(self):
        for i, tok in enumerate(self.Tokens):
            print("--- token: ", i)
            print(tok)

    ########################### FEATURES
    def features_extract_wrapper(self, features_get = None, shape_semantic_regenerate_from_stroke=False,
                                 angle_twind = (0, 0.2), label_as_novel_if_shape_semantic_fails=False):
        """
        WRapper to generate features for each token,
        which are appended to tokens. Where
        features is flexible -- semantic, motor, image, etc.
        Goal is to consoldiate all methods for doing so into here.
        Methods may be from:
            taskgeneral, taskmodel, prim, stroketools, features, ds
        PARAMS:
        - label_as_novel_if_shape_semantic_fails, bool if True, then useful for
        days with novel prims, where wnat to try to label semantic, but if that fails, then
        call is novel. 
        :return: Addpends to self.Tokens
        NOTE: THis is the ONLY place shape_semantic is computed
        """
        from pythonlib.tools.stroketools import angle_of_stroke_segment
        from pythonlib.tools.expttools import deconstruct_filename
        from pythonlib.tools.exceptions import NotEnoughDataException

        if features_get is None:
            features_get = ["shape_semantic", "loc_on", "angle"]

        n_fails = 0
        n_tot = 0
        for tok in self.Tokens:
            strok = tok["Prim"].Stroke()
            for feature in features_get:
                if feature=="shape_semantic":

                    # First, shape_semantic is defined only if there is no tforms extra rotation

                    # assert "tforms_extra_exist" in tok.keys(), "needs this to know whether is novel prim. see Dataset.tokens_preprocess_wrapper_good()"
                    if "tforms_extra_exist" in tok.keys() and tok["tforms_extra_exist"]:
                        # Then ignore, since this is not base prim.
                        tok["shape_semantic"] = "NOVEL-X-X-X"
                        tok["shape_semantic_cat"] = "NOVEL"
                    else:

                        try:
                            # #TODO:
                            # # get this: TaskClass.PlanDat, and from this get whether did extra
                            # # tform, as PlanDat.ParamsSpatialExtra.tforms_each_prim. If did, then
                            # # don't try to get shape semantic (doesnt exist). 
                            # assert False

                            if shape_semantic_regenerate_from_stroke:
                                # Then force rgenerate from raw stroke
                                tok["shape_semantic"] = tok["Prim"].label_classify_prim_using_stroke_semantic()
                            else:
                                # First, try to get mapping between shap string and semantic.
                                if tok["shape"] in MAP_SHAPE_TO_SHAPESEMANTIC:
                                    tok["shape_semantic"] = MAP_SHAPE_TO_SHAPESEMANTIC[tok["shape"]]
                                else:
                                    # If that fails, then try to compute it. But this will often fail for CHAR, since
                                    # these strokes are not task-strokes, so are irregular.
                                    tok["shape_semantic"] = tok["Prim"].label_classify_prim_using_stroke_semantic()

                            tmp = deconstruct_filename(tok["shape_semantic"])
                            tok["shape_semantic_cat"] = tmp["filename_components_hyphened"][0] # e..g, ['test', '1', '2']
                        except NotEnoughDataException as err:
                            # Failed... how deal with this.
                            if label_as_novel_if_shape_semantic_fails:
                                tok["shape_semantic"] = "NOVEL-X-X-X"
                                tok["shape_semantic_cat"] = "NOVEL"
                                n_fails+=1
                                # close the diagnostic plot that is made in label_classify_prim_using_stroke_semantic
                                plt.close("all")
                            else:
                                print("...Failed at determining shape_semantic for this tok:", tok)
                                raise err                               
                        except Exception as err:
                            print("...Failed at determining shape_semantic for this tok:", tok)
                            raise err                               
                elif feature=="loc_on":
                    tok["loc_on"] = strok[0,:2]
                elif feature=="loc_off":
                    tok["loc_off"] = strok[-1,:2]
                elif feature=="angle":
                    # angle of onset.
                    tok["angle"] = angle_of_stroke_segment([strok], twind=angle_twind)[0]
                else:
                    print(feature)
                    assert False, "code it"
            n_tot +=1

        if False: # This doesnt makes ense. it needs to be counting across all Tokens, not just all tokens with this Token.
            MAX_FRAC_FAIL = 0.1
            if n_fails/n_tot>MAX_FRAC_FAIL:
                print(n_fails, n_tot)
                assert False, "Maybe you expect this many novel prims to fail getting shape semantic? If so, comment this out."

    def data_extract_raw(self):
        """ Return as list of dicts, excluding any
        custom classes
        """
        tokens = self.Tokens
        tokens = [{k:v for k, v in tok.items() if not k=="Prim"} for tok in tokens]
        return tokens

