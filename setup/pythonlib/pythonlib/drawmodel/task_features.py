""" Methods that generally take in TaskGeneral object, and returns various features
about it. Doesnt care about behavior, just the task.
Often operates with tokens.

NOTE: doesnt usualyl do things related to sequences/groups, which is instead done in 
motifs_search, or discrete. 

3/5/23
"""

import numpy as np

def shapes_n_each_extract(Task, list_shapes, shape_key="shapeabstract"):
    """ Return list with n times each shape exists in this task
    PARAMS;
    - list_shapes, list of str.
    RETURNS;
    - list_n, list of int, how often each shape occur,
    - n_left_over, int, how many shapes in Task were not detected?
    """

    tokens = Task.tokens_generate(assert_computed=True)
    shapes = [t[shape_key] for t in tokens]
    
    # ignore order. just count how many A, B, C, ... etc
    nshapes = []
    for sh in list_shapes:
        n = sum([sh==x for x in shapes])
        nshapes.append(n)

    # shapes left over?
    n_left_over = len([x for x in shapes if x not in list_shapes])

    return nshapes, n_left_over



# IGNORE, obsolete, replaced by shapes_has_separated_cases_of_shape
# def shapes_has_separated_cases_of_this_shape(Task, shape_of_interest, ploton=False,
#     shape_key = "shapeabstract", dist_ver="concrete"):
#     """ Returns True if mjultiple strokes of this shape exist, and are seprated in space
#     such that at least one of them is closer to a different shape than it is to the same
#     shape. E..g, lines are far apart, with circle in between.
#     PARAMS:
#     - shape_of_interest, the shape for wich to return True if it is separated.
#     - dist_ver, defualt # use the concrete locaitons, not the grid locs
#     RETURNS;
#     - bool.
#     """


#     # def _min_distances(shapes_in_order, distances, shapes):
#     #     distances_in_order = []
#     #     for sh in shapes_in_order:
#     #         distances_shape = [d for d, s in zip(distances, shapes) if s==sh]
#     #         if len(distances_shape)>0:
#     #             dthis = min(distances_shape)
#     #             distances_in_order.append(dthis)
#     #     return distances_in_order

#     # Genreate tokens
#     Tok = Task.tokens_generate(return_as_tokensclass=True)
#     Tok.extract_locations_concrete() # get the concrete locatons.

#     if ploton:
#         Task.plotStrokes(ordinal=True)

#     res = []
#     for i in range(len(Tok.Tokens)):

#         # Collect distances to each other token
#         distances = []
#         shapes = []
#         shape_this = Tok.Tokens[i][shape_key]
#         if not shape_this==shape_of_interest:
#             # then ignore.
#             continue

#         for j in range(len(Tok.Tokens)):
#             # compare this tok to all other toks.
#             if j==i:
#                 continue

#             # get their distnace
#             d = Tok.featurepair_dist(i, j, ver=dist_ver)

#             sh1 = Tok.Tokens[i][shape_key]
#             sh2 = Tok.Tokens[j][shape_key]
#             res.append({
#                 "dist":d,
#                 "sh1":sh1,
#                 "sh2":sh2
#             })

#             distances.append(d)
#             shapes.append(sh2)

#         # Collect distances to this shape and other shapes.
#         dists_shape_this = [d for d, sh in zip(distances, shapes) if sh==shape_this]
#         dists_shape_other = [d for d, sh in zip(distances, shapes) if not sh==shape_this]
#         # if shape_this=="line":
#         #     shape_other = "circle"
#         # else:
#         #     shape_other ="line"

#         if ploton:
#             print("conditioned on token #:", i, shape_this)
#             print("dists to this shape:", dists_shape_this)
#             print("dists to other shapes:", dists_shape_other)

#         if len(dists_shape_this)>0 and len(dists_shape_other)>0:
#             min_dist_this = min(dists_shape_this)
#             min_dist_other = min(dists_shape_other)
#             if min_dist_other<0.99*min_dist_this:
#                 # then distnace to other is closer than to this same... 
#                 # means these two instances of same shape are separted by other shape.
#                 return True

#     return False

def shapes_has_separated_cases_of_shape(Task, shape_same=None, list_shape_diff=None, ploton=False, 
        shape_key = "shape", dist_ver="concrete", DEBUG=False):
    """
    [GOOD] Finds whether any shape has 2 instances separated by instance of a different shape.
    Improves on shapes_has_separated_cases_of_this_shape because solves prolbmes, including that
    the latter can fail if two islands of shape1 are separated by shape2.
    PARAMS:
    - shape_same, str name of shape which you want to check (i.e,, check all cases of pairs of this shape).
    Leave None to consider all shapes
    - list_shape_diff, list of str name of shape to consider as the sahpe that separates the two items of the
    shape_same in the pair of shape_same. Leave None to consider all shapes
    - shape_key, str, either "shape", or "shapeabstract"
    RETURNS:
    - bool.
    """

    from scipy.spatial.distance import pdist
    from scipy.spatial.distance import squareform

    LENIENCY = 0.8 # this means will only call a case True if the "diff shape" is close to the two
    # same shapes by this multiple, so 0.
    
    if ploton:
        Task.plotStrokes(ordinal=True)

    # shape_key = "shape"
    # dist_ver = "concrete"

    Tok = Task.tokens_generate(assert_computed=True, return_as_tokensclass=True)
    Tok.extract_locations_concrete() # get the concrete locatons.

    locs = np.array([Tok.feature_location(i, ver=dist_ver) for i in range(len(Tok.Tokens))])

    distmat = squareform(pdist(locs))
    shapes = [x[shape_key] for x in Tok.Tokens]

    for i in range(len(Tok.Tokens)):
        for j in range(i+1, len(Tok.Tokens)):
            
            shape1 = Tok.Tokens[i][shape_key]
            shape2 = Tok.Tokens[j][shape_key]
            
            if shape1==shape2:
                    
                # Skip if this is not the shape you care about.
                if shape_same is not None and not shape1==shape_same:
                    continue

                dist_same = distmat[i, j]
                assert dist_same == Tok.featurepair_dist(i, j, dist_ver)
                
                inds_same_shape = [k for k, sh in enumerate(shapes) if (sh==shape1 and not k==i and not k==j)]
                # exists other istnace of same shape that is in between i and j. if so, then skip evaluation
                # of i and j
                exists_same_shape_in_between = np.any(np.all(distmat[inds_same_shape, :][:, [i, j]] < dist_same, axis=1))
                if exists_same_shape_in_between:
                    if DEBUG:
                        print("!! exists_same_shape_in_between")
                        print("same shapes: ", i, j)
                        print("dist (same):", dist_same)
                        print("index of other same shapes:", inds_same_shape)
                        print(distmat[inds_same_shape, :][:, [i, j]])
                    continue

                # get indices of all tokens that are DIFF shape
                inds_diff_shape = [k for k, sh in enumerate(shapes) if not sh==shape1]

                # Only consider diff shapes that are in the list
                if list_shape_diff is not None:
                    inds_diff_shape = [i for i in inds_diff_shape if shapes[i] in list_shape_diff]
                
                # Exists a diff shape that is in between i and j. if so, then return True

                # - slice out the part of distmat that matters
                distmat_this = distmat[inds_diff_shape, :][:, [i, j]] # (inds_diff_shape, [i j])

                # - (True, False, True) means 0th and 2nd diff shape are in between these two same shapes
                diff_shapes_that_are_in_between_same_shapes = np.all(distmat_this < 0.8*dist_same, axis=1)

                if np.any(diff_shapes_that_are_in_between_same_shapes):
                    # Found a case of diff shape in between two same shapes...
                    if DEBUG:
                        print("!! Found pair of same shapes separated by a diff shape")
                        print("same shapes: ", i, j)
                        print("dist (same):", dist_same)
                        print("diff shapes:", inds_diff_shape)
                        print("dist (diffshapes x same): ")
                        print(distmat[inds_diff_shape, :][:, [i, j]])
                        print("looking for rows where both items are smaller than dist between same")
                    return True
    
    # Good, no pairs are separated.
    return False
                