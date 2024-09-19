"""simple derived meausures, e.g. speed, taking as input raw timecourses"""
import numpy as np
import pythonlib.tools.vectools as vt
from pythonlib.tools.stroketools import stroke2angle, smoothStrokes
from pythonlib.tools.timeseriestools import smoothDat

def segmentTouchDat(xyt, return_extras=False):
    """segment touch data.
    quick method just uses raises (nan)
    does not do anything smart, so ignores duration of nan and strokes, even if really short
    outputs heirarchial (strokes --> pts)
    
    takes in a single trial xyt (N x 3)
    
    onsets are indices (0) of first pt in stroke
    offsets are indices of last pt in storkes. 
    always length matched
    
    assumes x and y have nans at same timepounts

    """
    if len(xyt)==0:
        return []

    x = xyt[:,0]
    xx = np.r_[np.nan, x, np.nan] # append nans to edges. assumes onsets and offstes at edges
    b = np.diff(np.isnan(xx).astype(int))
    onsets = (b<0).nonzero()[0]
    offsets = (b>0).nonzero()[0]-1
    
    # print("3")
    # print(onsets)
    # print(xyt[0])
    # --- return heirarchical structure (strokes --> stroke data)
    strokes = []
    for on, of in zip(onsets, offsets):
        strokes.append(xyt[on:of, :])
    strokes = [s for s in strokes if len(s)>0]



    if return_extras:
        strokelens = offsets - onsets
        gaplens = onsets[1:] - offsets[:-1]
        return strokes, (onsets, offsets, strokelens, gaplens)
    else:
        return strokes
    


def dat2velocity(xyt):
    """currently returns speed simply as distance/time for each adjacent time bins
    - from Cisek 2020: The recorded positions of the stylus were interpolated at 
    100Hz using a 2D spline and filtered at 20Hz 252 with a low-pass Butterworth 
    filter (9th order) with zero delay. Velocity was computed using a five-point 
    differentiation routine, and then both position and velocity were up-sampled 
    to 1000Hz with linear interpolation and again low-pass filtered at 20Hz. Using 
    inverse kinematics equations for a planar arm model, we calculated the angular
     position of each joint through time and then passed them through an  inverse 
     dynamics model (SimMechanics) to calculate the muscle torques produced at the 
     shoulder and elbow joints.  
    - Also see : http://www.robots.ox.ac.uk/~sjrob/Teaching/EngComp/ecl6.pdf
    - 19 timepoints (150hz, so ~9ms) in https://www.jneurosci.org/content/jneuro/39/17/3320.full.pdf
    using S-G filter.
    - 10hz filter on pots, then central difference eqations: https://www.sciencedirect.com/science/article/pii/S0042698996001162#aep-section-id12
    - Schwartz 1992; fit spline, then differentiated.
    - also see: https://sci-hub.tw/https://www.tandfonline.com/doi/abs/10.1080/00222895.1994.9941661

    """


    print("velocity calc is too simple (subtracting adjacent bins) use better mthod?")
    speeds = []
    tbins = []
    for p1, p2 in zip(xyt[:-1], xyt[1:]):
        d = np.linalg.norm(p1[:2]-p2[:2])
        t = p2[2] - p1[2]
        tbins.append((p2[2]+p1[2])/2)
        if t==0:
            # happens rarely
            assert False, "why this happen?"
            speeds.append([])
        else:
            speeds.append(d/t)
    return np.concatenate((np.array(speeds).reshape(-1,1), np.array(tbins).reshape(-1,1)), axis=1)






def angles2bins(angles, N):
    """given angles (0,2pi), relative to +x,
    and number of bins, outputs bin number"""
    bins = np.linspace(0, 2*pi, N+1)
    return np.digitize(angles, bins)
#     angles_all_binned = [np.digitize(a, bins) for a in angles_all]


def strokesTransform(strokes, s=1.0, theta=0.0, x=0.0, y=0.0, order='trs'):
    """given strokes, list of T x 3 arrays, outputs the
    same shape, but after affine transformation
    
    strokes = strokes_all_task[0]
    a = angles_all_task[0]
    strokes = strokesTransform(strokes, theta=-a)
    """
    
    import pythonlib.drawmodel.primitives as P

    # - remove time information and transform
    strokes_tform = [s[:,[0,1]] for s in strokes]
    strokes_tform = P.transform(strokes_tform, s=s, theta=theta, x=x, y=y, order=order)
        
    # - put time back
    strokes_out = []
    for snew, sold in zip(strokes_tform, strokes):
        strokes_out.append(np.concatenate((snew, sold[:,2].reshape(-1,1)), axis=1))
    return strokes_out
