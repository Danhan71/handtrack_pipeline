""" Holds a single stroke, usualyl behavior, but could be task as well.
Contrast this with PrimitiveClass, which also holds a single storke (uusally) but is about
a symbolic represnetaiotn, so usually is task.
Here is wrapper for (N,3) np array.
Think of BehaviorClass as holding a sequence of strokeclasses
"""

import numpy as np

class StrokeClass(object):
    """
    """
    def __init__(self, traj=None):
        """
        PARAMS:
        - traj, (N,3) np array. 
        """
        if traj is None:
            self.Stroke = None
        else:
            self.input_data("nparray", traj)

    def __call__(self):
        """ Returns the np array"""
        return self.Stroke

    def input_data(self, ver, data, params=None):
        """ initilaize this stroke.
        PARAMS:
        - ver, string for different methods fo data entyr
        - data, flexible  type, depends on ver.
        - params, dict, holding params.
        """

        if ver=="nparray":
            """ data is (N,3) numpy array, where N is num timestemps, and (x,y,t) are columns
            """
            self.Stroke = data
        else:
            print(ver)
            assert False, "code it"

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        else:
            return np.all(self.Stroke == other.Stroke)

    ############# STROKE FEATURES
    def extract_center(self):
        # Returns [x,y] list
        from pythonlib.tools.stroketools import getCenter
        return getCenter([self.Stroke])

    def extract_single_feature(self, featurename):
        """ Extract this single feature, usually a scalar value.
        Wrapper for various methods written over the years to do this.
        PARAMS:
        - featurename, string name,
        RETURNS:
        - val,
        """
        
        strokes = [self.Stroke]
        from pythonlib.drawmodel.features import strokeCircularity, stroke2angle, strokeDistances, strokeDisplacements

        if featurename=="circularity":
            return strokeCircularity(strokes)[0]
        elif featurename=="distcum":
            return strokeDistances(strokes)[0]
        elif featurename=="displacement":
            return strokeDisplacements(strokes)[0]
        elif featurename=="angle":
            return stroke2angle(strokes)[0]
        else:
            print(featurename)
            assert False, "finish this."

    ############# STROKE SPATIAL DIMENSIONS
    def extract_spatial_dimensions(self, scale_convert_to_int=False):
        """ Extract spatial dimensions for this stroke
        PARAMS;
        - scale_convert_to_int, then helps avoid numericla precision situation
        where everything has slightly different value.
        RETURNS:
        - outdict, k:v are different kinds of spatial dimensions
        """
        from pythonlib.tools.stroketools import getMinMaxVals

        outdict = {}

        # Get width and height of the prim
        strokes = [self.Stroke]
        xyminmax = getMinMaxVals(strokes)
        outdict["width"] = xyminmax[1] - xyminmax[0]
        outdict["height"] = xyminmax[3] - xyminmax[2]

        # diagonal
        outdict["diag"] = (outdict["width"]**2 + outdict["height"]**2)**0.5

        if scale_convert_to_int:
            for k in ["width", "height", "diag"]:
                outdict[k] = int(outdict[k])
        outdict["max_wh"] = np.max([outdict["width"], outdict["height"]])

        return outdict
    
    def plot_stroke(self, ax=None):
        import matplotlib.pyplot as plt
        from pythonlib.drawmodel.strokePlots import plotDatStrokesWrapper
        if ax is None:
            fig, ax = plt.subplots()
        plotDatStrokesWrapper([self.Stroke], ax=ax)



