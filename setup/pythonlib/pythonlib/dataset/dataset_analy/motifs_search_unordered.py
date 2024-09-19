""" Methods to find motifs in unordered tokens, e.g., caring only about locations 
and relations, like all possible lollipops, WITHOUT caring about sequence used by
monkey. This previously was in discrete (3/5/23)
"""

import numpy as np
import matplotlib.pyplot as plt


def find_object_groups_new(Task, params):
    """ return list of groups (list of objects/shapes) passing constraints.
    Uses tokens from Task.tokens_generate(), and assumes this is unordered list. 
    So allows for any ordering of tokens, which means this outputs different ways
    of chunking. This differs from BehClass methods for finding motifs, because
    the latter assumes ordered tokens.
    e..g, find list of lollis, where each lolli is a set of 2 inds in objects
    PARAMS:
    - Task, taskclass instnace
    - params, dict holding flexible params
    RETURNS:
    - list_groups, list of list, each inner list being a single group or chunk. Ordering
    will not matter (both across chunks and within chunks)
    - left_over, list of ints, taskstroke indices that were not used in any chunk
    NOTE:
    - a given object can be used more than once, or not at all,
    e.g., if it participates in multiple gourpings.
    """

    GRIDLOC_VER = "gridloc_local" 

    # Generate tokens
    tokens = Task.tokens_generate(assert_computed=True)
    objects = tokens # naming convention change.

    # Methods for defining tokens or relations between tokens in a pair
    def _shape(i):
        """ String name of shape of i"""
        return objects[i]["shape"]
    def _oshape(i):
        """ String name, shape+orientation"""
        return objects[i]["shape_oriented"]
    def _location_grid(i):
        """ Location on grid, in grid units (integers, centered at 0)
        Returns (x,y)
        """
        return objects[i][GRIDLOC_VER]
    def _posdiffs_grid(i, j):
        """ Difference in positions between i and j, in grid units
        Return xdiff, ydiff"""
        pos1 = _location_grid(i)
        pos2 = _location_grid(j)
        return pos2[0]-pos1[0], pos2[1] - pos1[1]
    def _direction_grid(i, j):
        """ String name, cardinal direction, only if adjacnet on grid.
        Uses grid locations. direction from i to j """
        xdiff, ydiff = _posdiffs_grid(i,j)
        if np.isclose(xdiff, 0.) and ydiff ==1.:
            return "U"
        elif np.isclose(xdiff, 0.) and ydiff ==-1.:
            return "D"
        elif xdiff ==-1. and np.isclose(ydiff, 0.):
            return "L"
        elif xdiff ==1. and np.isclose(ydiff, 0.):
            return "R"
        else:
            # FAR
            return "f"
    def _orient(i):
        """ Orientation, in string, {'horiz, 'vert', ...}
        """
        assert False, "first pull in theta to objects in TaskClass"""
        if np.isclose(objects[i][1]["theta"], 0.):
            return "horiz"
        elif np.isclose(objects[i][1]["theta"], pi):
            return "horiz"
        elif np.isclose(objects[i][1]["theta"], pi/2):
            return "vert"
        elif np.isclose(objects[i][1]["theta"], 3*pi/2):
            return "vert"
        else:
            print(objects[i])
            assert False


    if params["rule"]=="concrete_chunk":
        # Replaces lolli (more general version)
        # concrete chunk is grouping of defined shape/locations. spatially defined, (i.e,, any temporal order)
        from pythonlib.tools.stringtools import decompose_string
        shapes_in_order = params["shapes_in_order"]
        orientation = params["orientation"]
        
        # e.g {'rule': 'concrete_chunk', 'shapes_in_order': ['line-8-4-0', 'line-8-3-0'], 'orientation': 'U'}
        if len(shapes_in_order)>2:
            print(shapes_in_order)
            assert False, "# not coded yet, assuming length 2"
        
        sh = shapes_in_order[0]
        n = len(decompose_string(sh)) 
        if n==1:
            # abstract
            def _shape_helper(i):
                """ String name of shape of i"""
                return objects[i]["shapeabstract"]
        elif n==4:
            # orietnted
            def _shape_helper(i):
                """ String name of shape of i"""
                return objects[i]["shape_oriented"]
        else:
            print(sh)
            assert False

        list_groups = [] # 
        for i in range(len(objects)):
            for j in range(len(objects)):
                if _shape_helper(i)==shapes_in_order[0] and _shape_helper(j)==shapes_in_order[1] and _direction_grid(i, j)==orientation:
                    # Found a chunk, include it
                    if False:
                        print("FOUDN THIS")
                        print(i, j)
                        print(_oshape(i), _oshape(j), _direction_grid(i, j))
                    list_groups.append([i, j])

        # Find what objects are left over
        list_groups_flat = [xx for x in list_groups for xx in x]
        left_over = [i for i in range(len(objects)) if i not in list_groups_flat]
    

    elif params["rule"]=="concrete_chunks_set":
        # Like concrete_chunk, but pass in multiple possible chunks, each altenraition of 
        # shape_oritened, direction,shape...

        list_shapes_in_order = params["list_shapes_in_order"]

        # for shapes_in_order in list_shapes_in_order:
        #     if len(shapes_in_order)>2:
        #         print(shapes_in_order)
        #         assert False, "# not coded yet, assuming length 2"
        
        list_groups = [] # 
        for i in range(len(objects)):
            for j in range(len(objects)):
                for this in list_shapes_in_order:
                    # this = shape_oritened, direction,shape...
                    assert len(this)==3, 'only coded for 2 objects, currenlty.'
                    shapes_in_order = [this[0], this[2]]
                    orientation = this[1]
                    if _oshape(i)==shapes_in_order[0] and _oshape(j)==shapes_in_order[1] and _direction_grid(i, j)==orientation:
                        # Found a chunk, include it
                        if False:
                            print("FOUDN THIS")
                            print(i, j)
                            print(_oshape(i), _oshape(j), _direction_grid(i, j))
                        list_groups.append([i, j])

        # Find what objects are left over
        list_groups_flat = [xx for x in list_groups for xx in x]
        left_over = [i for i in range(len(objects)) if i not in list_groups_flat]
    
    elif params["rule"]=="concrete_chunk_any_orient":
        # Replaces lolli (more general version).
        # Here allows any orientation (or a restricted set), with the shapes concrete
        from pythonlib.tools.stringtools import decompose_string

        shapes_in_order = params["shapes_in_order"]
        allowed_orientations = params["allowed_orientations"]
                
        assert False, "still inpropgress?"
        sh = shapes_in_order[0]
        n = len(decompose_string(sh)) 
        if n==1:
            # abstract
            def _shape_helper(i):
                """ String name of shape of i"""
                return objects[i]["shapeabstract"]
        elif n==4:
            # orietnted
            def _shape_helper(i):
                """ String name of shape of i"""
                return objects[i]["shape_oriented"]
        else:
            print(sh)
            assert False

        print(shapes_in_order)
        print(allowed_orientations)
        for o in objects:
            print(o)
        # print(objects)
        assert False
        if len(shapes_in_order)>2:
            print(shapes_in_order)
            assert False, "# not coded yet, assuming length 2"
        
        list_groups = [] # 
        for i in range(len(objects)):
            for j in range(len(objects)):
                if _shape_helper(i)==shapes_in_order[0] and _shape_helper(j)==shapes_in_order[1] and _direction_grid(i, j) in allowed_orientations:
                    # Found a chunk, include it
                    if False:
                        print("FOUDN THIS")
                        print(i, j)
                        print(_oshape(i), _oshape(j), _direction_grid(i, j))
                    list_groups.append([i, j])

        # Find what objects are left over
        list_groups_flat = [xx for x in list_groups for xx in x]
        left_over = [i for i in range(len(objects)) if i not in list_groups_flat]
    else:
        print(params)
        assert False, "code it"

    return list_groups, left_over
