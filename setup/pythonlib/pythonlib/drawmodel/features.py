# collect all code that takes in strokes or multipel strokes
# and outputs features
# In general functions will take in a strokes obejct, which is a 
# list of np arrays, each a stroke (N x 2). in general will
# return a list of features, one for each stroke.
# NOTE Confirmed that have moved everything here from the following
# places where I also tend to put feature analyuses:
# drawmonkey.calc , ...



import numpy as np

##################################
# def strokeCenters(strokes, ver="median"):
#   """ get center of mass fo stroke. 
#   ignores timepoints"""
#   if ver=="median"
#   return [np.median(s[:,:2], axis=0) for s in strokes]


def getCentersOfMass(strokes, method="use_median"):
    """ list, which is center for each stroke(i.e., np array) within strokes """
    if method=="use_median":
        return [np.median(s[:,:2], axis=0) for s in strokes]
    elif method=="use_mean":
        return [np.mean(s[:,:2], axis=0) for s in strokes]
    else:
        assert False, "not coded"


def stroke2angle(strokes, stroke_to_use="all_strokes", 
    force_use_two_points=False):
    """ get angles. outputs a list.
    - "first", then takes first stroke and gets vector
    from first to last point, and gets
    angle (in rad, (relative to 1,0)) for that
    - UPDATE: can now output angle for each stroke as 
    a list. note this type will be list, 
    updated to output will always be a list.
    - note, angles will all be relative to a universal (1,0)
    - will use nan wherever there is no movement for a stroke."""
    from ..tools.vectools import get_angle
    def _angle_for_one_stroke(s):
        """ s in np array T x 2(or 3)"""
        v = s[-1,[0,1]] - s[0,[0,1]]
        if np.linalg.norm(v)==0:
            return np.nan
        else:
            a = get_angle(v)
            if np.isnan(a):
                print(v)
                print(s)
                assert False
            return a

    if stroke_to_use=="first":
        # s = strokes[0]
        angles = [_angle_for_one_stroke(strokes[0])]
    elif stroke_to_use=="first_two_points":
        if not force_use_two_points:
            assert False, "have to modify to make sure that the two points are not equal. this leads to nan for the angle."
        s = strokes[0]
        s = s[[0,1],:]
        angles = [_angle_for_one_stroke(s)]
    elif stroke_to_use=="all_strokes":
        angles = [_angle_for_one_stroke(s) for s in strokes]
    else:
        assert False, "have not coded"
    return angles


###################################
def strokeDurations(strokes):
    """ time duration for each strok"""
    return [s[-1,2]-s[0,2] for s in strokes]


def strokeDistances(strokes):
    """ cumulative distance traveled along stroke
    resturns one scalar for each strok in strokes.
    """
    def D(s):
        # s is one np.array, N by 3, N = timepoints.
        # adds up sum of distnace btween adjacent points
        d = [np.linalg.norm(s1-s2) for s1, s2 in zip(s[:-1,:2], s[1:,:2])]
        return sum(d)
    return [D(s) for s in strokes]

def gapDistances(strokes):
    """ list of distances between gaps. assumes straight line
    """
    list_gapdists = []
    for s1, s2 in zip(strokes[:-1], strokes[1:]):
        list_gapdists.append(np.linalg.norm(s2[0,:2] - s1[-1,:2]))
    return list_gapdists

def strokeDisplacements(strokes):
    """displacement from strok onset to offset"""
    return [np.linalg.norm(s[-1,:2] - s[0,:2]) for s in strokes]

def strokeCircularitySigned(strokes, prune_outer_flanks_frac=False):
    """ circulatiry, but signed, so that + means CCW.
    NOTE: method for gettiung sign is rough, but works ok if is clean curvature.
    Mostly works, but this should be improved --> mainly the curvatuyre code is
    old, not really tested...
    PARAMS:
    - prune_outer_flanks_frac, either None (ignre) or fraction, in which case prunes
    that from edges before computing
    """
    from pythonlib.tools.stroketools import strokesCurvature

    circs = strokeCircularity(strokes, prune_outer_flanks_frac=prune_outer_flanks_frac)

    # What direction is the curvature?
    fs = int(1/np.mean(np.diff(strokes[0][:,2])))
    strokes_curv = strokesCurvature(strokes, fs, absval=False)

    list_median_curve = []
    for sc in strokes_curv:

        # Remove the first and last 100ms
        t1 = sc[0, 1]+0.1
        t2 = sc[-1, 1]-0.1
        if t1<t2:
            idx1 = np.argmin(np.abs(sc[:,1]-t1))
            idx2 = np.argmin(np.abs(sc[:,1]-t2))
            sc = sc[idx1:idx2, :]
        list_median_curve.append(np.median(sc[:,0]))

        # plot curvature
        if False:
            plotDatStrokesTimecourse([strok_curv], ax=ax, plotver="speed", label="curv", overlay_stroke_periods=False)
            ax.set_title(f"median curv:, {np.median(strok_curv[:,0]):.4f}")
            YMAX = 1/50
            YMIN = -YMAX

            ax.set_ylim([-0.1, 0.1])
            ax.set_ylabel("1/pix (1/radius)")



    return np.sign(list_median_curve) * np.asarray(circs)

def strokes_prune_outer_flanks_frac(strokes, frac, column_time):
    """ For each strok in strokes, remove the flanking fracs, e.g, if
    frac is 0.1, then keeps the inner 0.8 (by time). ASSUMES that times
    are regularly spaced...
    """
    strokes_pruned = []
    for s in strokes:

        npts = len(s)
        npts_flank = int(np.floor(frac*npts))

        # # Remove the first and last 100ms
        # t1 = s[0, column_time] + npts_flank
        # t2 = s[-1, column_time] - npts_flank
        #
        # idx1 = np.argmin(np.abs(s[:,column_time]-t1))
        # idx2 = np.argmin(np.abs(s[:,column_time]-t2))
        #
        # print(len(s))
        # print(npts_flank, idx1, idx2)
        strokes_pruned.append(s[npts_flank:-npts_flank, :])

    return strokes_pruned

def strokeCircularity(strokes, prune_outer_flanks_frac=None):
    """ 0 for straint line, 1 for full circle.
    is based on ratio of displacement to distance
    PARAMS;
    - prune_outer_flanks_frac, either None (ignre) or fraction, in which case prunes
    that from edges before computing. useful to exclude edge effects.
    """

    if prune_outer_flanks_frac is not None:
        assert not isinstance(prune_outer_flanks_frac, bool), "shouod be number"
        strokes = strokes_prune_outer_flanks_frac(strokes, prune_outer_flanks_frac, 2)

    displace = strokeDisplacements(strokes)
    distance = strokeDistances(strokes)
    return [1-p/t for p,t in zip(displace,distance)]

def strokeCurvature(strokes):
    """ see https://psycnet.apa.org/fulltext/1989-10716-001.pdf
    """
    print("see stroketools")
    assert False, "not done!!"

####### ONE VALUE SUMMARIZING STROKES
def computeDistTraveled(strokes, origin=None, include_lift_periods=True, 
    include_origin_to_first_stroke=True, doneloc=None, include_transition_to_done=False,
    center_to_center=False):
    """ assume start at origin. assumes straight line movements
    between strokes, and travel along stroke during strokes.
    by default includes times when not putting down ink.
    IGNORES third column(time) - i.e., assuems that datpoints are in 
    chron order.
    - include_transition_to_done, then includes gap between end of last stroke and done button. 
    Must pass in doneloc.
    - include_origin_to_first_stroke, then must pass in origin.
    - center_to_center, then instead of endpoint to endpoint, uses stroke centers.
    """
    
    if include_transition_to_done:
        assert doneloc is not None
    if include_origin_to_first_stroke:
        assert origin is not None
    if center_to_center:
        assert include_lift_periods==True, "otherwise pointless, since not considering within-stroke distances"

    if center_to_center:
        # convert strokes so that all pts in a stroke are its center pts
        centers = getCentersOfMass(strokes)
        strokesthis = [np.repeat(c.reshape(1,-1), 3, 0) for c in centers]
    else:
        strokesthis = strokes

    cumdist = 0


    if include_lift_periods:
        # - Get inter-stroke movements
        if not include_origin_to_first_stroke:
            # then say you started at locaiton of first touch
            origin = strokesthis[0][0, :2]

        prev_point = origin
        for S in strokesthis:
            cumdist += np.linalg.norm(S[0,:2] - prev_point)
            prev_point = S[-1, :2]

        # - Include to done button
        if include_transition_to_done:
            cumdist += np.linalg.norm(doneloc - strokesthis[-1][-1,:2])
    
    # - Get stroke distances
    cumdist += np.sum(strokeDistances(strokesthis))
    return cumdist


def strokesAngleOverall(strokes, ver="mean_of_center_jumps"):
    """ get angle reflecting overal sequence of strokes.
    INPUTS:
    - ver, which version
    --- mean_of_center_jumps, get sum of unit vectors between
    stroke centers (median), then get angle of that vector.
    RETURNS:
    - angle, radians, where (1,0) is 0, in range 0 --> 2pi.
    - (if stroke len 1, then np.nan)
    """
    from ..tools.vectools import unit_vector, get_angle
    if len(strokes)==1:
        return np.nan
    
    if ver=="mean_of_center_jumps":
        # get angle of transition between center of strokes
        centers = getCentersOfMass(strokes)
#         jumpangles = []
        uvs = []
        for s1, s2 in zip(centers[:-1], centers[1:]):
            jump = s2 - s1
#                 jumpangles.append(get_angle(jump))
            uvs.append(unit_vector(jump))

        # Get vector mean of angles
        # get the average unit vector
        mean_vec = np.mean(np.stack(uvs), axis=0)

        # Get angle of mean vec
        return get_angle(mean_vec)
    else:
        print(ver)
        assert False, "not coded"



######## GET FEATURES, GIVEN STROKES
def strokeFeatures(strokeslist):
    """ 
    input: list(trials) of list(strokes) of np array(strok)
    retursns: list of list of dict, matching input dimension,
    where each dict corresponds to one strok.
        
    """
    
    # --- Collect features, one for each strok
    strokesfeatures = []
    for strokes in strokeslist:
        tmp = []
        for a, b in zip(
            strokeCircularity(strokes),
            strokeDistances(strokes)):

            tmp.append(
                {
                "circularity":a,
                "distance":b
                }
            )
        strokesfeatures.append(tmp)
        
    return strokesfeatures


