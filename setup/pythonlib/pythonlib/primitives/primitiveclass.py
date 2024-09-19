""" Represents a single primitive, usually a task stroke.
Especially useful if doing stuff with low-level coordinates (e.g., classifying based on hand-coded rules).
Usually this is a base primitive. Usually corresponds to a single stroke, 
but doesnt have to (i.e. could be a chunk, which is multiple strokes concatenated
into a single new stroke)
3/29/22 - Try to use this for representations of tasks
"""

import numpy as np

def shape_string_convert_to_components(shape_str):
    """
    Decompose shape string..
    "arcdeep-4-1-0" --> ('arcdeep', '4', '1', '0')
    NOTE: sometimes the input shape can have decimels (e..g, Diego, 23/12/04
    Reason is these are non-standard shapes, e.g., perfblocky, and
    they are renamed using their actual params. e/g.,.
    V-83-5.8-0. They do end up being categorical tbhough, so
    is fine to use them as is.
    """
    if shape_str=="IGN" or shape_str is None:
        # this shape will be ignored in other code, so is ok.
        return "IGN", -1, -1, -1

    from pythonlib.tools.expttools import deconstruct_filename
    tmp = deconstruct_filename(shape_str)
    if len(tmp["filename_components_hyphened"])<4:
        print(shape_str)
        print(tmp)
        assert False, "bug upstream?"
    else:
        shape_abstract = tmp["filename_components_hyphened"][0]
        scale = tmp["filename_components_hyphened"][1]
        rotation = tmp["filename_components_hyphened"][2]
        reflect = tmp["filename_components_hyphened"][3]
        return shape_abstract, scale, rotation, reflect

def generate_primitiveclass_from_raw(traj, shape_string):
    """
    Helper to generate a P from input raw data
    :param traj:
    :param shape_string: something like Lcentered-4-2-0
    :return:
    """
    assert shape_string is not None, "this leads to confusions later (calling it IGN). give it a name."

    if shape_string is not None:
        shape_abstract, scale, rotation, reflect = shape_string_convert_to_components(shape_string)
    else:
        shape_abstract, scale, rotation, reflect = None, None, None, None
    P = PrimitiveClass()
    try:
        P.input_prim("prototype_prim_abstract", {
            "shape":shape_abstract,
            "scale":scale,
            "rotation":rotation,
            "reflect":reflect},
            traj = traj)
    except Exception as err:
        print(traj.shape)
        print(shape_string, shape_abstract, scale, rotation, reflect)
        assert False
    return P


class PrimitiveClass(object):
    """
    """

    def __init__(self):
        self.ParamsAbstract = None
        self.ParamsConcrete = None
        self.Stroke = None # strokeclass representation

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        else:
            return np.all(self.Stroke() == other.Stroke())

    def input_prim(self, ver, params, traj=None):
        """ Initialize data, represnting a single primtiive
        PARAMS:
        - traj, optional stroke, np array (N,2 or 3). If input, then creates strokeclass
        TODO:  Give it a hash code (since it is missing prim indices) within PrimitiveClass")
        """


        if ver=="prototype_prim_abstract":
            """ A prototype prim as in matlab ml2 code, where each prim is
            (shape, scale, rotation), where each are abstract categories.
            This doesn't explicitly know the concrete prim params
            - See drawmodel.tasks.planclass_extract_all, where uses planclass prims
            """

            if "shape" in params.keys():
                self.ShapeNotOriented = params["shape"]
            else:
                self.ShapeNotOriented = None

            # self.ParamsAbstract = {
            #     "reflect":int(params["reflect"]),
            #     "scale":int(params["scale"]),
            #     "rotation":int(params["rotation"]),
            # }
            self.ParamsAbstract = {}
            list_keys_abstract = ["reflect", "scale", "rotation"]
            for k in list_keys_abstract:
                if k not in params.keys() or params[k] is None:
                    self.ParamsAbstract[k] = None
                else:
                    try:
                        self.ParamsAbstract[k] = int(params[k])
                    except Exception as err:
                        # One time I had this usquare-86-5.5-0.8 [usquare 86 5.5 0]
                        # This is fine! See shape_string_convert_to_components() for
                        # explanation
                        self.ParamsAbstract[k] = params[k]

            self.ParamsConcrete = {}
            list_keys_concrete = ["x", "y", "theta", "sx", "sy", "order"]
            for k in list_keys_concrete:
                if k not in params.keys() or params[k] is None:
                    self.ParamsConcrete[k] = None
                elif isinstance(params[k], str):
                    self.ParamsConcrete[k] = params[k]
                else:
                    self.ParamsConcrete[k] = np.around(params[k], 3)
            # self.ParamsConcrete = {
            #     "x":np.around(params["x"], 3),
            #     "y":np.around(params["y"], 3)
            # }

            for k in params.keys():
                assert k in ["shape"] + list_keys_concrete + list_keys_abstract, "typo in entry"

            if traj is not None:
                self.strokeclass_generate(traj)

        elif ver=="prototype_prim_concrete":
            """ A prototype prim, but inputing concrete tform variables.
            This used in drawmodel.tasks.objects_extract. Prefer abstract over this,
            since objects_extract has a lot of ad-hoc code for parsing what prim a 
            subprogram corresponds to.
            """
            assert False, "code it, see below"
            # Entry will have params like this
            # T.Objects
            # [{'obj': 'line',
            #   'tform': {'x': array(1.7),
            #    'y': array(1.7),
            #    'th': array(2.35619449),
            #    'sx': array(1.4),
            #    'sy': array(1.4),
            #    'order': 'trs'}},
            #  {'obj': 'L',
            #   'tform': {'x': array(0.1),
            #    'y': array(1.7),
            #    'th': array(6.28318531),
            #    'sx': array(1.9),
            #    'sy': array(1.9),
            #    'order': 'trs'}},
            #  {'obj': 'line',
            #   'tform': {'x': array(1.7),
            #    'y': array(0.1),
            #    'th': array(1.57079633),
            #    'sx': array(1.4),
            #    'sy': array(1.4),
            #    'order': 'trs'}},
            #  {'obj': 'squiggle1',
            #   'tform': {'x': array(0.1),
            #    'y': array(0.1),
            #    'th': array(1.57079633),
            #    'sx': array(1.2),
            #    'sy': array(1.2),
            #    'order': 'trs'}}]
        else:
            print(ver)
            assert False, "code it"

    def print_summary(self):
        """ quickly summarize this prim
        """
        print(f"- Shape: {self.ShapeNotOriented}")
        print("- abstract params:")
        for k, v in self.ParamsAbstract.items():
            print(f"{k}: {v}")
        print("- concrete params:")
        for k, v in self.ParamsConcrete.items():
            print(f"{k}: {v}")


    def _label_stroke_features(self):
        """ Return dict with features of the stroke, treating it
        as an image, and which can be used to define semantic label"""
        from pythonlib.tools.vectools import get_angle, bin_angle_by_direction

        shcat = self.extract_as()[1] # e.g, ('line-11-1-0', 'line', 11, 1, 0, 0.0, -0.614)

        # a marker of scale and angle.
        S = self.Stroke
        cen = S.extract_center()

        # define the endpoint1 to be that more on the lower-left.
        # loc1 = S.Stroke[0, :2]
        # loc2 = S.Stroke[-1, :2]

        npts = S.Stroke.shape[0]

        # make sure loc1 is the one towards the bottom left
        centhis = [cen[0], cen[1]]
        if np.sum(S.Stroke[0, :2]-centhis) > np.sum(S.Stroke[-1, :2]-centhis): # np.sum() does project onto (1,1).
            loc1 = S.Stroke[-1, :2]
            loc2 = S.Stroke[0, :2]

            # use 2/5 and 3/5 so that flatter Z can be similar result as less-flat Z
            loc1_midpt = S.Stroke[int(np.floor(3/5 * npts)), :2]
            loc2_midpt = S.Stroke[int(np.floor(2/5 * npts)), :2]

            loc1_midpt_arm = S.Stroke[int(np.floor(4/5 * npts)), :2]
            loc2_midpt_arm = S.Stroke[int(np.floor(1/5 * npts)), :2]
        else:
            loc1 = S.Stroke[0, :2]
            loc2 = S.Stroke[-1, :2]

            loc1_midpt = S.Stroke[int(np.floor(2/5 * npts)), :2]
            loc2_midpt = S.Stroke[int(np.floor(3/5 * npts)), :2]
            loc1_midpt_arm = S.Stroke[int(np.floor(1/5 * npts)), :2]
            loc2_midpt_arm = S.Stroke[int(np.floor(4/5 * npts)), :2]

        # midpoint
        loc3 = S.Stroke[int(np.floor(npts/2)), :2]

        if False:
            # Old version (before 2/26,24) which failed to disambig L that have same nedpoints but diff orientation
            # - center --> onset
            vec1 = loc1 - cen # vec from cen to onset.
            angle1 = np.round(get_angle(vec1), decimals=1)
            # angle_bin = bin_angle_by_direction([get_angle(vec)], num_angle_bins=16)[0] # bin the angle

            # - center --> offset
            vec2 = loc2 - cen # vec from cen to onset.
            angle2 = np.round(get_angle(vec2), decimals=1)

            # - center --> midpoint of curve (useful to disambiguate L that have same endpoint but diff orientatoin)
            vec3 = loc3 - cen
            angle3 = np.round(get_angle(vec3), decimals=1)
        else:
            # vec1 = loc1 - loc3 # vec from midpt to onset.
            vec1 = loc1 - loc1_midpt # vec from midpt to onset.
            angle1 = get_angle(vec1)

            vec1 = loc1 - loc1_midpt_arm # vec from midpt to onset.
            angle1_arm = get_angle(vec1)

            # vec2 = loc2 - loc3 # vec from midpt to offset.
            vec2 = loc2 - loc2_midpt # vec from midpt to offset.
            angle2 = get_angle(vec2)

            vec2 = loc2 - loc2_midpt_arm # vec from midpt to offset.
            angle2_arm = get_angle(vec2)

        # scale
        vec = loc1 - cen # vec from cen to onset.
        scale = np.linalg.norm(vec)

        # from pythonlib.tools.stroketools import strokes_bounding_box_dimensions
        # strokes_bounding_box_dimensions([])
        tmp = S.extract_spatial_dimensions()
        height_divide_width = tmp["height"]/tmp["width"]

        features = {
            "shape_cat":shcat,
            # "vec_midpt_to_onset":vec1,
            # "vec_midpt_to_offset":vec2,
            "angle_midpt_to_onset":angle1,
            "angle_midpt_to_offset":angle2,
            "angle_midpt_to_onset_arm":angle1_arm,
            "angle_midpt_to_offset_arm":angle2_arm,
            "scale":scale,
            "height_divide_width":height_divide_width
        }
        return features

    def label_classify_prim_using_stroke_semantic(self, return_as_string=True):
        """ Helper to return just the 3 features that define each shape, instead of 4,
        i.e,, exclude scale. as string or 3-tuple (including the shape string).
        """
        return self.label_classify_prim_using_stroke(return_as_string=return_as_string, version="semantic", exclude_scale=True)

    def label_classify_prim_using_stroke(self, return_as_string=False,
                                         shape_rename_perfblockey_decimals_to_defaults=False,
                                         version="default", exclude_scale=False, cluster_by_sim_idx=None):
        """ To classify this prim, qwhich usualy would be
        liek line-10-0-1, but this doesnt generaklkly work,
        becuase somtimes you have extra transfomrs that chagne how this
        is actually angled, etc. Here solves this problem by replacing the
        scale and angle with actual measured, based on stroke.'

        NOTE: this tries to be invariant to the direction of stroke, since this is a task stroke.

        PARAMS:
        - exclude_scale, if True, then returns 3 items, else 4.
        RETURNS:
            - label, tuple that uniquely ids this prim, (sh, scale, angle to on, angle to off)
            e.g., ('line', 60.0, 1.57, 4.71)
        NOTE: reason to have angles to both on and off is to deal with possible cases of
        same on. and/or reflections.
        """
        from pythonlib.tools.exceptions import NotEnoughDataException
        from pythonlib.tools.vectools import get_angle, bin_angle_by_direction
        from math import pi
        features = self._label_stroke_features() 

        def _vec_to_angle_bin(vec):
            a = get_angle(vec)
            a_binned = bin_angle_by_direction([a], starting_angle=-pi/8, num_angle_bins=8)
            return a_binned

        # return as tuple
        if version=="cluster_by_sim":
            # Then you pass in the name directly, becuase this is based on clustering using the entire dataset.
            assert isinstance(cluster_by_sim_idx, int)

            shcat = features["shape_cat"]

            label = (shcat, cluster_by_sim_idx, cluster_by_sim_idx, cluster_by_sim_idx)

        elif version=="hash":
            # This gets a unique string for each stroke, based on motor features, that is more unique thatn the
            # default. HJere takes the features from defualt, and adds also a unique hash based on the strokes
            # pts
            # This is useful if down the line you will define shape features -e.g , psychometric, novel prims
            from pythonlib.tools.stroketools import strokes_to_hash_unique
            
            # Default features
            shcat = features["shape_cat"]
            scale = int(10*features["scale"]) # need this to be whole number.
            angle1 = int(100*features["angle_midpt_to_onset"])
            angle2 = int(100*features["angle_midpt_to_offset"])

            # Hash
            # - First center the stroke
            # Hash shouldnt be larger than 6, since then gets to numerical imprecision. 4 is fine.
            hashnum = strokes_to_hash_unique([self.Stroke()], 4, centerize=True, align_to_onset=False)

            # label = (shcat, str(hashnum)[:3], str(hashnum)[3:6], str(hashnum)[6:])
            # label = (shcat, str(scale), str(angle1)+str(angle2), str(hashnum)[:2], str(hashnum)[2:4], str(hashnum)[4:])
            
            # label = (shcat, f"{scale:.1f}", f"{angle1:.1f}{angle2:.1f}", hashnum)
            label = (shcat, scale, f"{angle1}{angle2}", hashnum)

        elif version=="default":
            # Use the actual numerical values to label this shape.
            shcat = features["shape_cat"]
            scale = features["scale"]
            angle1 = features["angle_midpt_to_onset"]
            angle2 = features["angle_midpt_to_offset"]
            label = (shcat, scale, angle1, angle2)
        elif version=="semantic":
            # Map from stroke properties --> a semantic label.
            # assert False, "first check if this label is in strokes database.. if so, then use that."
            shcat = features["shape_cat"]
            scale = features["scale"]
            # scale = "X" # ignore

            def _raise_error(a1, a2):
                print(shcat)
                print(features)
                print(a1, a2)
                print(self.extract_as())
                fig = self.plot_stroke()
                fig.savefig("/tmp/stroke.png")
                print("Check saved stroke at ", "/tmp/stroke.png")
                print("for code debuggin, see 210506_analy_dataset_summarize.ipynb --> 'DatStrokes, reclassifying prims based on motor (image) (e.g., novel prims)'")
                raise NotEnoughDataException
                # assert False, "for code debuggin, see 210506_analy_dataset_summarize.ipynb --> 'DatStrokes, reclassifying prims based on motor (image) (e.g., novel prims)'"

            if shcat in ["circle"]:
                label = (shcat, scale, "XX" , "XX")
            elif shcat in ["V2"]:
                # is more like 90deg (not sure if it is)
                a1 = features["angle_midpt_to_onset_arm"]
                a2 = features["angle_midpt_to_offset_arm"]
                a1 = bin_angle_by_direction([a1], starting_angle=-pi/8, num_angle_bins=8)[0]
                a2 = bin_angle_by_direction([a2], starting_angle=-pi/8, num_angle_bins=8)[0]

                if a1==4 and a2==2:
                    label = (shcat, scale, "UU" , "UU") # opens to top
                elif a1==6 and a2==8:
                    label = (shcat, scale, "DD" , "DD") #
                else:
                    _raise_error(a1, a2)

            elif shcat in ["V", "arcdeep", "usquare"]:
                a1 = features["angle_midpt_to_onset_arm"]
                a2 = features["angle_midpt_to_offset_arm"]
                a1 = bin_angle_by_direction([a1], starting_angle=-pi/8, num_angle_bins=8)[0]
                a2 = bin_angle_by_direction([a2], starting_angle=-pi/8, num_angle_bins=8)[0]

                if a1==3 and a2==3:
                    label = (shcat, scale, "UU" , "UU") # opens to top
                elif a1==5 and a2==5:
                    label = (shcat, scale, "LL" , "LL") # opens to top
                elif a1==7 and a2==7:
                    label = (shcat, scale, "DD" , "DD") # opens to top
                elif a1==1 and a2==1:
                    label = (shcat, scale, "RR" , "RR") # opens to top
                else:
                    _raise_error(a1, a2)
            # elif shcat in ["arcdeep"]:
            #     a1 = features["angle_midpt_to_onset"]
            #     a2 = features["angle_midpt_to_offset"]
            #     a1 = bin_angle_by_direction([a1], starting_angle=-pi/8, num_angle_bins=8)[0]
            #     a2 = bin_angle_by_direction([a2], starting_angle=-pi/8, num_angle_bins=8)[0]
            #
            #     if a1==4 and a2==3:
            #         label = (shcat, scale, "UU" , "UU")
            #     elif a1==6 and a2==5:
            #         label = (shcat, scale, "LL" , "LL")
            #     elif a1==7 and a2==7:
            #         label = (shcat, scale, "DD" , "DD")
            #     elif a1==6 and a2==1:
            #         label = (shcat, scale, "RR" , "RR")
            #     else:
            #         print(features)
            #         print(a1, a2)
            #         print(self.extract_as())
            #         assert False
            elif shcat in ["Lcentered"]:
                a1 = features["angle_midpt_to_onset"]
                a2 = features["angle_midpt_to_offset"]
                a1 = bin_angle_by_direction([a1], starting_angle=-pi/8, num_angle_bins=8)[0]
                a2 = bin_angle_by_direction([a2], starting_angle=-pi/8, num_angle_bins=8)[0]

                if a1==3 and a2==1:
                    label = (shcat, scale, "UR" , "UR") # opens to top-right
                elif a1==4 and a2==2:
                    label = (shcat, scale, "UU" , "UU") # opens to top
                elif a1==5 and a2==3:
                    label = (shcat, scale, "UL" , "UL") #
                elif a1==6 and a2==4:
                    label = (shcat, scale, "LL" , "LL") #
                elif a1==6 and a2==8:
                    label = (shcat, scale, "DD" , "DD") #
                elif a1==7 and a2==1:
                    label = (shcat, scale, "DR" , "DR") #
                elif a1==8 and a2==2:
                    label = (shcat, scale, "RR" , "RR") #
                elif a1==5 and a2==7:
                    label = (shcat, scale, "DL" , "DL") #
                else:
                    _raise_error(a1, a2)
            elif shcat in ["Lzigzag", "squiggle3", "Lzigzag1", "zigzagSq"]:
                a1 = features["angle_midpt_to_onset"]
                a2 = features["angle_midpt_to_offset"]
                hw = features["height_divide_width"]
                a1 = bin_angle_by_direction([a1], starting_angle=-pi/8, num_angle_bins=8)[0]
                a2 = bin_angle_by_direction([a2], starting_angle=-pi/8, num_angle_bins=8)[0]

                if (a1 in [3, 4] and a2==8) or (a2 in [3, 4] and a1==8):
                    label = (shcat, scale, "LL" , 0) # [2] location of the "top" if you were writing "S". [3] reflected [first reflect, then reflect]
                elif a1 in [6, 7] and a2==2 and hw<1:
                    label = (shcat, scale, "LL" , 1) # [2] location of the "top" if you were writing "S". [3] reflected [first reflect, then reflect]
                elif a1 in [5, 6] and a2==2 and hw>1:
                    label = (shcat, scale, "UU" , 0) # [2] location of the "top" if you were writing "S". [3] reflected [first reflect, then reflect]
                elif a1 in [1, 8] and a2==4:
                    label = (shcat, scale, "UU" , 1) # [2] location of the "top" if you were writing "S". [3] reflected [first reflect, then reflect]
                else:
                    _raise_error(a1, a2)
            elif shcat in ["line"]:
                a1 = features["angle_midpt_to_onset"]
                a2 = features["angle_midpt_to_offset"]
                hw = features["height_divide_width"]
                a1 = bin_angle_by_direction([a1], starting_angle=-pi/8, num_angle_bins=8)[0]
                a2 = bin_angle_by_direction([a2], starting_angle=-pi/8, num_angle_bins=8)[0]

                if (a1==4 and a2==8) or (a1==8 and a2==4):
                    label = (shcat, scale, "UL" , "UL") # direction of line, in top hemisphere
                elif a1==5 and a2==1:
                    label = (shcat, scale, "LL" , "LL") # direction of line, in top hemisphere
                elif a1==6 and a2==2:
                    label = (shcat, scale, "UR" , "UR") # direction of line, in top hemisphere
                elif a1==7 and a2==3:
                    label = (shcat, scale, "UU" , "UU") # direction of line, in top hemisphere

                else:
                    _raise_error(a1, a2)
            elif "novelprim" in shcat:
                shcat = features["shape_cat"]
                scale = features["scale"]
                angle1 = features["angle_midpt_to_onset"]
                angle2 = features["angle_midpt_to_offset"]
                label = (shcat, scale, angle1, angle2)
            else:
                _raise_error(None, None)
        else:
            print(version)
            assert False, "code it"

        assert isinstance(label, tuple) and len(label)==4

        if shape_rename_perfblockey_decimals_to_defaults:
            # e.g., 231204, diego, for Perfblocky, convert shapes that have decimal names to their closest default shape.
            # THis would require by eye writing mapping.
            # Or better is to do it auto by comapring distance to DS saved strokes (which has rtask strokes).
            # Decided to skip this, and instead use motor- ckuster labels for analyuses (which I prev did for char, but here
            # decide to do also for SP and PIG).

            # If do this, also prob should have this done auto in dataset_preprocess, early on, before seq context stuff, etc.

            # Would need to code  mapping:
            # (shcat, scale, angle1, angle2) --> (shcat, scale, angle1, angle2) [with integers, the defaults]

            # ujseful noteobok: 230623_char_STROKES_CLUSTERING --> section "plot examples of all the prims"

            assert False, "code this. and add flag so that dataset_preprocess knows to do this. Or make it default?"

        # Round, so that doesnt suffer from num imprecicion.
        label = list(label)
        for i in range(1, len(label)):
            if not isinstance(label[i], str):
                if i==1:
                    label[i] = np.round(label[i], decimals=0)
                else:
                    label[i] = np.round(label[i], decimals=1)
        label = tuple(label)

        if return_as_string:
            label_str = [label[0]]

            if isinstance(label[1], str):
                label_str.append(label[1])
            else:
                label_str.append(f"{label[1]:.0f}")

            if isinstance(label[2], str):
                label_str.append(label[2])
            else:
                label_str.append(f"{label[2]:.1f}")

            if isinstance(label[3], str):
                label_str.append(label[3])
            else:
                label_str.append(f"{label[3]:.1f}")

            if exclude_scale:
                return "-".join([label_str[0], label_str[2], label_str[3]])
            else:
                # label_str = [label[0], f"{label[1]:.0f}", f"{label[2]:.1f}", f"{label[3]:.1f}"]
                return "-".join([x for x in label_str])
        else:
            if exclude_scale:
                return tuple([label[0], label[2], label[3]])
            else:
                return label

    def shape_oriented(self, include_scale=True):
        """ Returns shape oriented, e.g., line-3-0
        """
        prms = self.ParamsAbstract
        def _sh(x):
            # SHORTHAND
            if x is None:
                return "x"
            else:
                return x
        if include_scale:
            return f"{self.ShapeNotOriented}-{_sh(prms['scale'])}-{_sh(prms['rotation'])}-{_sh(prms['reflect'])}"
        else:
            return f"{self.ShapeNotOriented}-{_sh(prms['rotation'])}-{_sh(prms['reflect'])}"

    def extract_as(self, output="primtuple", include_scale=True):
        """ [Wrapper] Help extract ths primitives in different formats
        PARAMS:
        - output, string name for how to extract
        """

        if output=="label_unique":
            return self.label_classify_prim_using_stroke()
        elif output=="primtuple":
            return self._convert_to_primtuple()
        if output=="primtuple_string":
            return self._convert_to_primtuple(as_string=True)
        elif output=="shape":
            """ TaskGeneral.Shape, which is like ["line", {"x":x, "y":y, ...}]
            """
            par = self._extract_params(include_scale=include_scale)
            return [par["shape_rot"], {
                "x":par["cnr_x"],
                "y":par["cnr_y"],
                "sx":par["cnr_sx"],
                "sy":par["cnr_sy"],
                "theta":par["cnr_theta"],
                "order":par["cnr_order"],
                }
                ] # leave the tform as None, ideally extract this from combining (x,y) in plan and sizes/rotation for this primtive
        elif output=="params":
            """ A dict holding relevant params"""
            return self._extract_params()
        elif output=="loc_concrete":
            # 2-tuple, x and y coords.
            par = self._extract_params(include_scale=include_scale)
            return (par["cnr_x"], par["cnr_y"])
        else:
            print(output)
            assert False, "code it"

    def _extract_params(self, include_scale=True):
        """ Extract params for this prim in a dict format
        RETURNS:
        - params, dict, where keys are things like scale, rotation, etc.
        """

        out = {}
        out["shape"] = self.ShapeNotOriented

        for k, v in self.ParamsAbstract.items():
            out[f"abs_{k}"] = v
        for k, v in self.ParamsConcrete.items():
            out[f"cnr_{k}"] = v
        
        # assert False, 'also add reflection here'
        # print("Also add reflection here")
        out["shape_rot"] = self.shape_oriented(include_scale=include_scale)
        
        return out

    def _convert_to_primtuple(self, use_abstract=True, as_string=False):
        """ A hashable represetnation of this prim, 
        either as tuple (defualt) or string (concatenation)
        - is unique id for this prim
        """

        params = self._extract_params()

        if use_abstract:
            scale = params["abs_scale"]
            rot = params["abs_rotation"]
            refl = params["abs_reflect"]
        else:
            scale = params["cnr_scale"]
            rot = params["cnr_theta"]
            refl = params["abs_reflect"]

        # shape x rotation defines prim
        primtuple = (params["shape_rot"], params["shape"], scale, rot, refl, params["cnr_x"], params["cnr_y"])

        if as_string:
            primtuple = "_".join([f"{x}" for x in primtuple])
        return primtuple

    ##################### REPRESENTING AS STROKECLASS
    def strokeclass_generate(self, traj):
        """ Represent this primitive as a concrete stroke using 
        StrokeClass. Allows using all the methods there.
        PARAMS:
        - traj, np.array, (N,2 or 3).
        MODIFIES:
        - self.StrokeClass is replaced.
        """
        from pythonlib.behavior.strokeclass import StrokeClass
        self.Stroke = StrokeClass(traj)



    def plot_stroke(self):
        """

        :return:
        """
        import  matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(self.Stroke()[:,0], self.Stroke()[:,1], "xk")
        ax.plot(self.Stroke()[0,0], self.Stroke()[0,1], "or")

        return fig
