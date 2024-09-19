""" General TaskClass
(made during drawnn but should be extendable for other purposes
"""
from . import primitives as Prim
import numpy as np
from pythonlib.globals import PATH_ANALYSIS_OUTCOMES

ATOL = 0.05 # toelrace for saying a coord is close to another (for deciding if is on grid)

class TaskClass(object):
    """ Abstract class for all tasks.
    All data exposed will have first already been converted to abstract cooridnates
    (e..g,, self.Strokes, self.Points)
    TODO: general-purpose self.Program and self.Plan
    """

    def __init__(self):
        self.Params = {}
        self._initialized = False
        self.Name = None
        self.Parser = None
        self.Parses = None
        self.ShapesOldCoords = None
        self._TokensLocked = False

    def initialize(self, ver, params, coord_force_same_aspect_ratio=True,
            convert_coords_to_abstract=True, split_large_jumps=False,
            auto_hack_if_detects_is_gridlinecircle_lolli=False):
        """ General purpose, to initialize with params for
        this task. Is flexible, takes in many formats, converts
        each into general abstract format
        INPUT:
        - ver, str, in {"drawnn", "ml2"}, indicates what format
        - params, flexible, what format depends on what ver.
        - coord_force_same_aspect_ratio, then for drawnn and ml2 coords,
        - convert_coords_to_abstract, then self.Strokes, and others will be in abstract system.
        otherwise keep in whatever was input.
        will make sure is asme aspect ratio as asbtract coords. Does this
        by padding the smaller dimension to match abstract (equally on both ends)
        """

        # Save input params
        self.Params["input_ver"] = ver
        self.Params["input_params"] = params
        self.Params["coord_force_same_aspect_ratio"] = coord_force_same_aspect_ratio
        self.Params["convert_coords_to_abstract"] = convert_coords_to_abstract
        self.Params["split_large_jumps"] = split_large_jumps

        # Extract Strokes
        if ver=="drawnn":
            # New version of drawnn
            program = params["program"]
            shapes = params["shapes"]
            chunks = params["chunks"]
            self._initialize_drawnn(program, shapes, chunks)

        # Create things like self.StrokesOldCoords
        # Do not create self.Strokes here. This done next block.
        elif ver=="ml2":
            # Monkeylogic tasks
            taskobj = params
            self._initialize_ml2(taskobj, auto_hack_if_detects_is_gridlinecircle_lolli=auto_hack_if_detects_is_gridlinecircle_lolli) 
        else:
            print(ver)
            assert False, "not coded"

        # Convert strokes to appropriate coordinate system
        # and generate Points
        # convert things like self.StrokesOldCoords to self.Strokes
        if convert_coords_to_abstract:
            self._initialize_convcoords()
        else:
            # use the input coordinates
            # TODO: self.ProgramOldCoords and others.
            self.Strokes = self.StrokesOldCoords
            self.Shapes = self.ShapesOldCoords

        # Convert strokes to points
        self.Points = np.stack([ss for s in self.Strokes for ss in s], axis=0) # flatten Strokes

        # Check that nothing goes out of bounds
        self._check_out_of_bounds()

        # Split into distinct strokes, wherever task pts make dramatic jump.
        if split_large_jumps:
            self._split_strokes_large_jump()

        # Sanity check that all initializatinos are consistente with each other.
        if False:
            self._sanity_check_initializations()

    def _initialize_ml2(self, taskobj, auto_hack_if_detects_is_gridlinecircle_lolli=False):
        """ Pass in a task from monkeylogic.
        - taskobj, task class from drawmodel.tasks

        NOTES:
        - self.Shapes is like list of lists, where lower lists are [shapename, tform params]
            e.g., for ml2 [note: might be different for different task versions]
            [['line',{'x': array(1.5),
               'y': array(0.),
               'th': array(0.),
               'sx': array(2.),
               'sy': array(2.),
               'order': 'trs'}],
             ['circle',
              {'x': array(0.),
               'y': array(1.5),
               'th': array(0.),
               'sx': array(0.7),
               'sy': array(0.7),
               'order': 'trs'}]
        """

        # Extract data
        self.StrokesOldCoords = taskobj.Strokes

        taskobj.program_extract() 
        self.ProgramOldCoords = taskobj.Program
        
        if taskobj.get_tasknew() is not None:
            # Get abstract representaion of objects (primitives)
            taskobj.planclass_extract_all()
            self.PlanDat = taskobj.PlanDat

            if self.PlanDat is None:
                # Then this is old version, before using planclass
                taskobj.objects_extract()
                self.ShapesOldCoords = taskobj.Objects
                self.ShapesOldCoords = [[S["obj"], S["tform"]] for S in self.ShapesOldCoords] # conver to general format
                for i in range(len(self.ShapesOldCoords)):
                    if self.ShapesOldCoords[i][1] is not None:
                        self.ShapesOldCoords[i][1]["theta"] = self.ShapesOldCoords[i][1]["th"]
                        del self.ShapesOldCoords[i][1]["th"]
                
                # [gridlinecirlce, 210821] if call print(self.ShapesOldCoords) returns:
                # [['line', {'x': array(-1.5), 'y': array(-1.5), 'sx': array(1.55), 'sy': array(1.55), 'order': 'trs', 'theta': array(0.)}]]
            else:
                # New version, use planclass representation of prims. Dont have to re-extract
                # here post-hoc. Dont use the "Shapes" thing anymore, instead represent them as
                # PrimitiveClass objects
                # self.Primitives = taskobj.PlanDat["primitives"]

                # Extract Chunks
                self.ChunksListClass = taskobj.PlanDat["ChunksListClass"]

                if False:
                    # SKIP THIS! Motifs should be multipel objects...
                    # Extract primitivechunks (e..g, motif is actually a single object)
                    self.objects_extract_using_planclass()


        else:
            assert False, "must generate self.Primitives (instead of using Shapes...)"

        ############ GOOD, update to final concated objectclass
        self.primitives_extract_final(auto_hack_if_detects_is_gridlinecircle_lolli=auto_hack_if_detects_is_gridlinecircle_lolli)

        # populate self.ShapesOldCoords, for backwards compativbility with other code.
        # print("BEFORE UPDATE: ", self.ShapesOldCoords)
        self.ShapesOldCoords = [p.extract_as("shape") for p in self.Primitives]
        # print(self.Primitives)
        # print("AFTER UPDATE: ", self.ShapesOldCoords)


    def primitives_extract_final(self, return_prims = False, auto_hack_if_detects_is_gridlinecircle_lolli=False):
        """
        [GOOD] Replaces self.Primitives with updated version, which applies for
        all bersions of tasks, including old tasks. Run this at very end.

        For newest tasks: this accounts for
        concatenation (which happens in step between PlanCalss and ObjectClass).
        Must first run both planclass and objectclass extraction.
        This _SHOULD_ match up perfectly with ObectClass. 
        In  practice, self.PRimitives is changed only if there was concatenation.
        
        RETURNS:
        - modifies self.Primitives
        """

        from pythonlib.primitives.primitiveclass import PrimitiveClass

        nprims = len(self.StrokesOldCoords)
        O = self.ml2_objectclass_extract(auto_hack_if_detects_is_gridlinecircle_lolli=auto_hack_if_detects_is_gridlinecircle_lolli)
        P = self.PlanDat
        TT = self.extract_monkeylogic_ml2_task()
        TN = TT.get_tasknew()

        # for p in Prims:
        #     p.print_summary()
        # assert False

        # for i in range(nprims):
                
        #     Prim = PrimitiveClass()
        #     traj = self.StrokesOldCoords[i]

        primitives = []
        if O is not None and P is not None:
            
            ################### V1, latest, if both O and P exist
            assert len(self.PlanDat["ReflectsAfterConcat"])==len(self.StrokesOldCoords)
            assert len(self.PlanDat["CentersAfterConcat"])==len(self.StrokesOldCoords)
            
            PrimsList = self.PlanDat["primitives"]
            Features = O["Features_Active"]
            list_shapes_1 = Features["shapes"] # MATLAB: these are not correct. ie, calls a "V" an "L"...
            list_shapes_2 = P["ShapesAfterConcat"] # These are correct
            # list_shapes_3 = [p.Shape for p in PrimsList] # These should match list_shapes_2
            list_prims_plandat = P["PrimsAfterConcat"]

            for i in range(nprims):
                    
                Prim = PrimitiveClass()

                # Things that are true whether or not concat
                traj = self.StrokesOldCoords[i]
                loc = P["CentersAfterConcat"][i]

                # Shapes signals whtehr this was concated (novel_prim)
                shape = list_shapes_2[i]
                if shape[:9]=="novelprim":
                    # Then this is concatted. Use the new prim name

                    if False:
                        # PREVIOUSLY:
                        shape = list_shapes_1[i] # e.g, "-line-line..."
                        # list_shapes_2[i] # novelprim
                    else:
                        # UPDATED:
                        # shape = list_shapes_1[i] # e.g, "-line-line..."
                        shape = list_shapes_2[i] # novelprim-4603624566784205119

                    # then not defined:
                    scale = None
                    rotation = None
                    reflect = None
                    # scale = O["Features_Active"]["prot_level"][i]
                    # rotation = O["Features_Active"]["prot_rot"][i]

                    assert len(PrimsList)>len(list_shapes_2), "this expected if this is concatted"
                else:
                    # Then use plandat, just like before.
                    scale = int(list_prims_plandat[i][1][1])
                    rotation = int(list_prims_plandat[i][1][2])
                    reflect = P["ReflectsAfterConcat"][i]

                    # Sanity check that O and P match.
                    if "prot_level" in Features.keys():
                        assert scale == Features["prot_level"][i]
                        assert rotation == Features["prot_rot"][i]

                    # Assertion, sanity check that what I previuosly called 
                    # self.ShapesOldCoord (which was generated directly from 
                    # PrimsList in old code) matches the new prims.
                    if False:
                        # incorrect, since PrimsList is potentailyl longer than list_prims_plandat (if concatted)
                        prms = PrimsList[i].extract_as("params")
                        assert prms["shape"]==shape
                        assert prms["abs_reflect"]==reflect
                        assert prms["abs_scale"]==scale
                        assert prms["abs_rotation"]==rotation
                        assert np.isclose(prms["cnr_x"], loc[0], atol=0.001)
                        assert np.isclose(prms["cnr_y"], loc[1], atol=0.001)

                
                Prim.input_prim("prototype_prim_abstract", {
                        "shape":shape,
                        "scale":scale,
                        "rotation":rotation,
                        "reflect":reflect,
                        "x":loc[0],
                        "y":loc[1]}, 
                        traj = traj)

                primitives.append(Prim)

        elif P is None and TN is not None:
            #################### V2, using saved program in TaskNew

            if len(self.ShapesOldCoords)==len(self.StrokesOldCoords):
                # then good
                list_shapes = [x[0] for x in self.ShapesOldCoords]
                list_prms = [x[1] for x in self.ShapesOldCoords]
            else:
                list_shapes = [None for _ in range(len(self.StrokesOldCoords))]
                list_prms = [None for _ in range(len(self.StrokesOldCoords))]

            for i in range(nprims):
                Prim = PrimitiveClass()
                traj = self.StrokesOldCoords[i]

                shape = list_shapes[i]
                prms = list_prms[i]

                if shape is None and prms is None:
                    print("THIS SHOULD ONLY HAPPEN FOR GRIDLINECIRCLE, LOLLI, 210829! if not, then debug.")
                    Prim.input_prim("prototype_prim_abstract", {}, 
                            traj = traj)
                else:
                    Prim.input_prim("prototype_prim_abstract", {
                            "shape":shape,
                            "theta":prms["theta"],
                            "x":prms["x"],
                            "y":prms["y"],
                            "sx":prms["sx"],
                            "sy":prms["sy"],
                            "order":prms["order"]
                            }, 
                            traj = traj)

                primitives.append(Prim)

        elif TN is None:
            #################### v3: oldest version, before even using programs...
            for i in range(nprims):
                Prim = PrimitiveClass()
                traj = self.StrokesOldCoords[i]

                Prim.input_prim("prototype_prim_abstract", 
                    {}, 
                    traj = traj)
                primitives.append(Prim)
        else:
            assert False, "what kind of task is this?"
            
        # #     dat["primitives"].append(Prim)
        # primitives.append(Prim)

        if return_prims:
            return primitives
        else:
            self.Primitives = primitives
            
    def objects_extract_using_planclass(self):
        """ Uses plan in PlanClass version of task to decide on Objcts. Main point is
        the following chunking organization, based on the hand-coded prim categories (matlab):
        Objects are: (i) base prims; (ii) motifs that are touching. Otherwise, Objects are lower 
        level, so the following are split up: (i) characters [dont really have to do anything] (
        (ii) motifs not touchign [e.g. parallele lines]
        RETURNS:
        - modifies self.PrimitivesObjects
        """

        # 1) Get the chunks and prims categories
        CLC = self.ChunksListClass
        primcats = self.PlanDat["PrimsCategories"]
        strokes = self.Strokes

        # 2) For each chunk, decide if it is motif and if is touching
        CLC.print_summary()
        print(primcats)
        print(len(strokes))

        # 3) For any motif, figure out if all strokes are connected to al other strokes.
        assert False, "not coded"



    def _initialize_drawnn(self, program=None, shapes=None, chunks=None):
        """
        To initialize a new task for drawnn training.
        PARAMS:
        - program is list of subprograms, each suprogram is list of 
        lines, where each line evaluates and takes in the accumulation of
        previous lines within the subprogram. e..g,:
            program = []
            program.append(
                [["line"],
                ["repeat", {"p":"$", "N":3, "x":0.5}]])
            program.append(
                [["circle", [None, None, 0.5, 0.5, None, None]],
                 ["line"],
                 ["repeat", {"y":0.5}]]
            )
        Note: each line is like [<function>, <params>], where params can be
        --- dict that will be passed in as kwargs. Note that "$" means take in
        accumulation of previousl lines in this subprogram as this argument. If 
        dont pass in $, then will assume to pass in previous lines into first
        position argument (e..g,, in repeat in 2nd suprogram).
        --- list, this must not miss any arguemnts, but can pass None if to use
        default. Can choose to pass in the $, or not, if not then uses as first arg.

        - shapes, is list of shape (single stroke) primitives. this will be automatcaly
        extracted from program. these are still symbolic. 
        --- can pass in shapes, in which case program will be None. I did this since 
        havent figured out best way to evaluate program.
        - strokes, is like shapes, but in numbers. This is computed automatically.
        - chunks, list of higher level chunking, where each chunk is [chunkname<str>, [shapes in this chunk]]
        e.g., chunks = [["lolli", [0, 1]], ["lolli", [4,3]]]. I added this so can use for 
        hierarhcical RNN.
        """


        a = program is not None
        b = shapes is not None
        assert a!=b, "provide either program or shapes"

        self.ProgramOldCoords = program

        if program is not None:
            self.ShapesOldCoords = self.program2shapes(self.ProgramOldCoords)
        else:
            self.ShapesOldCoords = shapes
            
        self.StrokesOldCoords = self.shapes2strokes(self.ShapesOldCoords)

        # NOT YET DONE - general coords, make these empty. following code will generate these
        # by converting coords.
        self.Program = None
        self.Shapes = None
        self.Strokes = None

        # Chunks
        if chunks is not None:
            self.Chunks = chunks

            # sanity checks
            for c in self.Chunks:
                inds = c[1]
                for i in inds:
                    assert i<len(self.ShapesOldCoords), "chunks are referencing shapes that dont exist"



    def _initialize_convcoords(self, out_="abstract"):
        """ initial conversion of input coordinate system into abstract system
        Modifies: Strokes and Points. 
        Converts things like self.StrokesOldCoords to self.Strokes.
        TODO: have not done for self.Program and self.Strokes
        print("TODO (drawmodel-->taskgeneral): convert coords for self.Shapes. Do this in ml2 taskobject")
        """

        in_ = self.Params["input_ver"]
        # out_ = "abstract"

        # print(in_, out_)
        # print(self.Strokes)

        self.Strokes = [self._convertCoords(s, in_, out_) for s in self.StrokesOldCoords]

        # TODO: self.Shapes = function(self.ShapesOldCoords) # have not coded this before. previously just
        # used the input coordinate system.

    def _preprocess(self):

        # Convert strokes to abstract coordinates (to be more general)
        from .image import convertCoordGeneral  
        pass


    def _split_strokes_large_jump(self):
        from pythonlib.tools.stroketools import split_strokes_large_jumps
        if self.Params["input_ver"]=="ml2":
            thresh = 50
        else:
            assert False, "what is good threshold where if adajncet pts greater, then split storke here?"
        self.Strokes = split_strokes_large_jumps(self.Strokes, thresh)

    def _convertCoords(self, pts, in_, out_):
        """ convert between coordinate systems.
        automatically determines sketchpads, etc.
        INPUTS:
        - in_, {"abstract", "ml2", "drawnn"}
        - out_, {"abstract", "ml2", "drawnn"}
        - pts, N x 2(or3), array
        """

        from pythonlib.drawmodel.image import convCoordGeneral
        # strokes = T.Strokes

        # make sure sketchpads are initialized
        self._make_sketchpads()

        # compute
        edges_in = self._Sketchpads[in_]
        edges_out = self._Sketchpads[out_]
        pts_out = convCoordGeneral(pts, edges_in, edges_out)

        return pts_out

    def get_sketchpad_final(self):
        """ Helps to return the sketchpad associated wtih the final coordinate
        system. i.e,., whgat' sin self.Points, etc.
        RETURNS:
        - skethpad, 2x2 np array, where sketchpad[0,:] are xlims...
        """

        if self.Params["convert_coords_to_abstract"]:
            # Then final pts are in abstract coords
            sketchpad = self._Sketchpads["abstract"]
        else:
            # then are in input coords
            if self.Params["input_ver"] == "ml2":
                sketchpad = self._Sketchpads["ml2"]
            else:
                assert False, "code this"
        return sketchpad

    def _make_sketchpads(self):
        """ get sketchpad coordinates, both hard coded and dynamic.
        for format, see _converCoords for detaisl
        """
        
        if not hasattr(self, "_Sketchpads"):
            self._Sketchpads = {
                "abstract":np.array([[-3.5, 3.5], [-3.5, 3.5]]),
                "drawnn":np.array([[-3.5, 3.5], [-3.5, 3.5]])    
            }

            if self.Params["input_ver"] == "ml2":
                self._Sketchpads["ml2"] = self.Params["input_params"].Task["sketchpad"].T
                # print("Generated self._Sketchpads to:")
            if self.Params["coord_force_same_aspect_ratio"]:
                from .image import coordsMatchAspectRatio

                edgesgood = self._Sketchpads["abstract"]

                # then makes sure each sketchpad is same aspect ratio as abstracat
                sketchpad_new = {}
                for k, edges in self._Sketchpads.items():

                    if k != "abstract":
                        edgesnew = coordsMatchAspectRatio(edgesgood, edges)
                        sketchpad_new[k] = edgesnew
                        sketchpad_new[f"{k}_orig"] = edges
                    else:
                        sketchpad_new[k] = edges

                        # self._Sketchpads[f"{k}_orig"] = edges
                        # self._Sketchpads[k] = edgesnew
                self._Sketchpads = sketchpad_new



    def _check_out_of_bounds(self):
        """
        check no pts out of bounds. would indicate mistake in sketchpad in.
        assumes hav already converted to abstract
        RETURNS:
        - fails (assert False) if pts not in bounds.
        """

        self._make_sketchpads()
        sketchpad = self.get_sketchpad_final()
        xlims = sketchpad[0,:]
        ylims = sketchpad[1,:]
        pts = self.Points

        a = np.all((pts[:,0]>=xlims[0]) & (pts[:,0]<=xlims[1]))
        b = np.all((pts[:,1]>=ylims[0]) & (pts[:,1]<=ylims[1]))
        if a==False or b==False:
            print(sketchpad)
            print(pts)
            print(self.Strokes)
            print(self.Params)
            assert False , "not in bounds"

    def extra_tform_params_extract(self):
        """
        REturn either None (iof doesnt exist) or dict of extra tform params, like this:

        {'tforms': {},
        'tforms_each_prim_p': [{},
            [['th', array(-0.16839016)],
            ['sx', array(1.01434708)],
            ['sy', array(1.01434708)]],
            {}],
        'flips': array([0., 0., 0.]),
        }        
        """
        # print(self.PlanDat["PrimsExtraParams"])
        # assert False

        if "PrimsExtraParams" not in self.PlanDat.keys():
            return None
        elif all([len(x)==0 for x in self.PlanDat["PrimsExtraParams"]]):
            # liek this [{}, {}]
            return None
        else:
            if len(self.PlanDat["PrimsExtraParams"])>1:
                print(self.PlanDat["PrimsExtraParams"])
                assert False, "what is this..."
            
            extra_tforms_dict = self.PlanDat["PrimsExtraParams"][0] # Dict

            # Add default keys if no exist.
            if "tforms" not in extra_tforms_dict:
                extra_tforms_dict["tforms"] = {}
            if "tforms_each_prim_p" not in extra_tforms_dict:
                extra_tforms_dict["tforms_each_prim_p"] = [{} for _ in range(len(self.Strokes))]
            if "flips" not in extra_tforms_dict:
                extra_tforms_dict["flips"] = {}

            assert isinstance(extra_tforms_dict, dict)
            print(extra_tforms_dict)
            assert all([k in ["tforms", "tforms_each_prim_p", "flips"] for k in extra_tforms_dict.keys()])

            return extra_tforms_dict
        

    def check_prims_extra_params_exist(self):
        """
        Check if any extra transformations were applied to the prims in matlab code.
        If so, then the prim label (e.., line-10-1-0) will not be accurate.
        RETURNS:
            - bool, whether tforms were applied.
        EXAMPLES:
        eg self.PlanDat["PrimsExtraParams"] that eixsts:
        [{'tforms': [['th', array(4.71238898)]]},
         {'tforms': [['th', array(0.)]]},
         {'tforms': [['th', array(0.)]]}]

        eg self.PlanDat["PrimsExtraParams"] thjat doesnt exist:
            [{}]
        """
        # a = len(T.PlanDat["PrimsExtraParams"])>
        # return a and b

        extra_tforms_dict = self.extra_tform_params_extract()
        if extra_tforms_dict is None:
            return False

        return any([len(p)>0 for p in extra_tforms_dict.values()])
                   
    ########################

    # def get_task_id(self):
    #     """ get unique id for this task. 
    #     RETUNRS:
    #     tuple (category, number in cat)
    #     """
    #     if self.Params["input_ver"]=="ml2":
    #         taskcat = self.Params["input_params"].Task["stage"]
    #         taskstr = self.Params["input_params"].Task["str"]
    #         idx = taskstr.find(taskcat)
    #         tasknum = int(taskstr[idx+len(taskcat)+1:])
    #     else:
    #         assert False
    #     return taskcat, tasknum

    def get_task_category(self):
        """ [GOOD] Get human-interpretable task category
        This flexible depending ion task
        RETURNS:
        - taskcat, str, 
        """
        if self.Params["input_ver"]=="ml2":
            taskcat = self.Params["input_params"].info_name_this_task_category()   
            return taskcat
        else:
            assert False

    def get_los_id(self):
        """ Get load_old_set information, which means info for this as a fixed task
        that was loaded online during expt
        RETURNS:
        EIther:
        - setname, setnum, taskind OR
        - None
        """
        if self.Params["input_ver"]=="ml2":
            return self.Params["input_params"].info_summarize_task()["los_info"]
        else:
            assert False

    def get_unique_name(self):
        """ Wrapper to get good unique name
        THIS is good version, identical to Dataset class
        RETURNS:
        - name, string name that is unique to this task.
        """
        if self.Params["input_ver"]=="ml2":
            Task = self.Params["input_params"]
            random_task = not Task.info_is_fixed()
            if random_task:
                return self._get_number_hash(ndigs=10, include_taskstrings=False,
                    include_taskcat_only=True, compact=True)
            else:
                return self._get_number_hash(ndigs=6, include_taskstrings=True)
        else:
            assert False

    def _get_number_hash(self, ndigs = 6, include_taskstrings=True, 
        include_taskcat_only=False, compact = False):
        """ Returns number, with digits ndigs, based on 
        coordinates of task. shodl be unique id.
        - include_taskstrings, False, then just number. True, then uses task 
        category, and numberical id (which entered during task creation)
        NOTE: confirmed that the strokes that go in here are identical to strokes in Dataset
        """
        # if include_taskcat_only:
        #     assert include_taskstrings is False, "choose either entier string, or just the cat"
        if self.Params["input_ver"]=="ml2":
            idnum = self.Params["input_params"].info_generate_unique_name(self.Strokes, 
                nhash=ndigs, include_taskstrings=include_taskstrings, include_taskcat_only=include_taskcat_only)

            if compact:
                # remove long "random" in string
                ind = idnum.find("random")
                if ind>0:
                    idnum = idnum[:ind+1] + idnum[ind+6:]

            return idnum
        else:
            assert False

    #############################
    def program2shapes(self, program):
        assert False, "not done, see __init__ for what need to do."
        shapes =[]
        for subprog in program:

            funcstring = evaluateProg(subprog) 
            subshapes = evaluateString(funcstring)
            shapes.extend(subshapes)


    def generatePlan(self):
        """ convert to sequential actions.
        e.g., based on simple sequential biases.
        """
        self.Plan = None
        assert False, "not coded"


    def shapes2strokes(self, shapes):
        def evaluateShape(shape, params):
            """ 
            - shape, is string name
            - params, different options:
            --- list, will be passed into transform in order.
            --- dict, will be passed into transform as kwargs.
            --- params order: [x=0, y=0, sx=1, sy=1, theta=0, order="trs"]
            --- Note: can pass in None to get defaults.
            === NOTES
            - now line is centered at 0,0. Original primitives centered at (0.5,0)
            === RETURNS
            [np.array]
            """
            def transform(p, x=None, y=None, sx=None, sy=None, theta=None, order=None):
                """ outputs stroke """
                T = Prim.makeAffine2(x, y, sx, sy, theta, order)
                return Prim._tform(p, T)

            if shape=="line":
                p = Prim.transform(Prim._line, x=-0.5)
            elif shape=="circle":
                p = Prim._circle
            else:
                print(shape)
                assert False, "not coded"

            if isinstance(params, list):
                return transform(p, *params)
            elif isinstance(params, dict):
                return transform(p, **params)
            else:
                print(params)
                assert False

        strokes =[]
        for sh in shapes:
            s = sh[0]
            if len(sh)==1:
                params = []
            else:
                params = sh[1]
            strokes.extend(evaluateShape(s, params))

        return strokes


    ############# SHAPES
    def _shape_to_string(self, shape):
        """
        Convert shape to a string, which is hashable.
        shape is an item in self.Shapes. 
        - shape, list, [shapename, dict_params], e./.g,"['line',
      {'x': array(-1.7),
       'y': array(1.7),
       'sx': array(1.55),
       'sy': array(1.55),
       'order': 'trs',
       'theta': array(1.57079633)}]
       
       RETURNS:
       - string, e.g., linex-1.7y1.7sx1.55sy1.55ortrsth1.57
       """
        
        import numpy as np
        outstring = ""
        
        # name of object (e..g, line)
        outstring += shape[0]
        
        # affine transofmr params
        prms_in_order = ["x", "y", "sx", "sy", "order", "theta"]
        for key in prms_in_order:
            if key in shape[1].keys():
                this = shape[1][key]
                if isinstance(this, np.ndarray):
                    this = str(this)[:4]
                outstring += f"{key[:2]}{this}"
        
        return outstring

    def get_shapes_hash(self):
        """ convert self.Shapes to a frozenset of strings, each string a unique shape based on 
        category and affine params.
        Operates on self.Shapes.
        OUT:
        - frozenset of strings. (frozen allows the output to be hashable)
        --- e.g.,{'circlex-1.7y0.0sx0.7sy0.7ortrsth0.0', 'linex-1.7y1.7sx1.55sy1.55ortrsth1.57'}
        NOTE:
        - raises error if self.Shapes is empty
        """

        assert len(self.Shapes)>0, "old version of task? no shapes"

        return frozenset((self._shape_to_string(sh) for sh in self.Shapes))




    ############ FILTERS
    def filter_by_shapes(self, F, kind):
        """ general purpose, for figuring whheter this task passes filter criteria. 
        supports different ways of formatting F
        INPUT
        - F, different versions:
        --- list of dicts, where each dict has the shape and range for params, where if fall
        within range for all p[arams then consider this a pass. e..g, 
        F = [{"shape":"circle"}]
        - kind, kind of filter.
        """

        if kind=="any_shape_in_range":
            # return True if any shape has features that all fall within ranges for the
            # features designated in F. Any features not designated will be ignored.
            # e.g., F = {"x":[-0, 0.8], "y":[-1.6, -1.5], "sy":[0, 0.5]}
            # NOTE: can pass in list len 0, if want to get exactly that value: e.g.,:
            # F = {"x":[0], "y":[-1.5]}

            def _check_shape(ind, feature, minval, maxval):
                # returns True if shape ind's feature is between minval and maxval.
                # is inclusive.
                if self.Shapes[ind][1][feature]>=minval and self.Shapes[ind][1][feature]<=maxval:
                    return True
                else:
                    return False

            for ind in range(len(self.Shapes)):
                check_list = []
                for feature, valrange in F.items():
                    assert isinstance(valrange, list)
                    if len(valrange)==1:
                        minval = valrange[0]
                        maxval = valrange[0]
                    else:
                        minval = valrange[0]
                        maxval = valrange[1]
                    check_list.append(_check_shape(ind, feature, minval, maxval))
                if all(check_list):
                    # at least one shape (this) has passed. that is enough.
                    return True

            # none of the shapes passed all checks
            return False

        else:
            print(kind)
            print(F)
            assert False, "not coded"

    ############ REPRESENT AS SEQUENCE OF DISCRETE TOKENS
    def compare_on_same_grid(self, TaskOther):
        """ Returns True if other task is on same grid. False if grid sizies diff 
        ( e.g., 5x5 vs 3x3) or diff spacing.
        PARAMS:
        - TaskOther, taskclass object
        """

        # fail if iether is not on grid
        if not self.get_grid_ver()=="on_grid" or not TaskOther.get_grid_ver()=="on_grid":
            return False

        # Fail if any of these key params are not the same
        gp1 = self.get_grid_params()
        gp2 = TaskOther.get_grid_params()
        # for key in ["grid_x", "cell_width", "cell_height", "center"]:
        for key in ["grid_x_actual_after_rel", "grid_y_actual_after_rel", "cell_width", "cell_height", "center"]:
            if not gp1[key].size==gp2[key].size: # use a.size, not len(a), b/c is np.array, and 1-element was giving error
                # diff num grid spacings
                return False
            if not np.all(np.isclose(gp1[key], gp2[key])):
                # diff valued grid spacings.
                return False
            
        return True


    def compare_prims_on_same_grid_location(self, TaskOther):
        """ Returns True if other task has (i) same grid and (ii) prims
        on identical location. Doesnt care whteher they are in same tempora order,.
        or what those prims are.
        - NOTE: ok if num prims different, as long as locations identical (which means 
        one must have overlapoing prims)
        """

        if not self.compare_on_same_grid(TaskOther):
            return False

        datseg1 = self.tokens_generate(assert_computed=True)
        locations1 = sorted(set([d["gridloc"] for d in datseg1]))

        datseg2 = TaskOther.tokens_generate(assert_computed=True)
        locations2 = sorted(set([d["gridloc"] for d in datseg2]))

        return locations1 == locations2

    def get_grid_ver(self):
        """ Each task is either on a grid or off a grid, based on locations of centers of the primitves.

        RETURNS:
        - grid_ver, string, in 
        --- 'on_grid', meaning that all objects are centered on a global grid location
        --- 'on_rel', meaning that all objects are anchored to other obejcts, not to
        global grid
        NOTE: if old version (before around 3/2022) then returns None, since gridclass is not
        defined
        """

        # Latest version, this is saved as a meta param

        ######################## Old task versions...
        if not self.get_is_new_version():
            return "undefined"
        gridparams = self.get_grid_params() 
        if gridparams is None:
            return "undefined"
        
        ######################## 
        # METHOD 1, TaskSetClass params. use the quick meta params originally used
        # for genreating tsc
        if "grid_or_char" in gridparams.keys() and gridparams["grid_or_char"] is not None:
            if gridparams["grid_or_char"]=="grid":
                return "on_grid"
            elif gridparams["grid_or_char"]=="char":
                return "on_rel"
            # elif gridparams["grid_or_char"] is None:
            #     return "undefined"
            else:
                print(gridparams)
                assert False

        #########################
        # METHOD 2 - look at relations info in gridparams.
        # NOT DONE! 

        ######################## 
        # METHOD 3

        # Older version, infer it from the saved centers.
        # if all centers are aligned with grid centers, then this is on grid.
        grid_x = gridparams["grid_x_actual_after_rel"]
        grid_y = gridparams["grid_y_actual_after_rel"]
        # grid_x = gridparams["grid_x"]
        # grid_y = gridparams["grid_y"]

        if False:
            centers = self.PlanDat["CentersActual"]
        else:
            # Only consider centers for the first prim in each chunk. This is concvention.
            centers = self.PlanDat["CentersAfterConcat_FirstStrokeInChunk"]
            

            # CLC = self.PlanDat["ChunksListClass"]
            # # Use the first chunk, by convnetion
            # Ch = CLC.ListChunksClass[0]
            # centers = self.PlanDat["CentersActual"]

            # centers = [centers[h[0]] for h in Ch.Hier] # take first stroke in each chunk(hier)

            # try:
            #     assert self.PlanDat["CentersAfterConcat"] == centers, 'just a sanity check that what I think shoudl be identical'
            # except Exception as err:

            #     for k, v in self.PlanDat.items():
            #         print("--", k)
            #         print(v)
            #     print(self.PlanDat["CentersAfterConcat"])
            #     print(centers)
            #     print(grid_x, grid_y)
            #     self.plotStrokes()
            #     raise err

        grid_ver = self._get_grid_ver_manual(grid_x, grid_y, centers)

        # grid_centers = []
        # for x in grid_x:
        #     for y in grid_y:
        #         grid_centers.append(np.asarray([x, y]))

        # if all([isin_array(c, grid_centers, atol=ATOL) for c in centers]):
        #     # print("centers are on grid:")
        #     # print(centers)
        #     # print(grid_centers)
        #     # Then each stroke center matches a grid center
        #     grid_ver = "on_grid"
        # else:
        #     # print([isin_array(c, grid_centers, atol=ATOL) for c in centers])
        #     # Then at least one stroke is off grid.
        #     grid_ver = "on_rel"

        return grid_ver


    def _get_grid_ver_manual(self, grid_x, grid_y, centers):
        """ Given arrays for gird_x and y, and arrays of 2-points for
        centers, return string for what grid ver
        PARAMS:
        - centers, list of 2-arrays
        """
        from pythonlib.tools.nptools import isin_array

        grid_centers = []
        for x in grid_x:
            for y in grid_y:
                grid_centers.append(np.asarray([x, y]))

        if all([isin_array(c, grid_centers, atol=ATOL) for c in centers]):
            # print("centers are on grid:")
            # print(centers)
            # print(grid_centers)
            # Then each stroke center matches a grid center
            grid_ver = "on_grid"
        else:
            # print([isin_array(c, grid_centers, atol=ATOL) for c in centers])
            # Then at least one stroke is off grid.
            grid_ver = "on_rel"

        return grid_ver

    def get_grid_xy(self, hack_is_gridlinecircle=False):
        """ [GOOD} Return the x and y values for the grid. 
        Helps deal with outliers
        RETURNS:
        - xgrid, np array of coordinates of each x grid value
        - ygrid
        - grid_ver, string
        """

        # Grid ver
        grid_ver = self.get_grid_ver()
        
        # - grid spatial params 
        get_grid = True
        if hack_is_gridlinecircle:
            xgrid = np.linspace(-1.7, 1.7, 3)
            ygrid = np.linspace(-1.7, 1.7, 3)
            get_grid = False

            centersthis = []
            for p in Prims:
                prms = p.extract_as("shape")[1]
                centersthis.append([prms["x"], prms["y"]])

                if prms["x"] is None:
                    assert False, "this is probably gridlinecirlce lolli? 210829 Need to extract x/y loc. see notes here"
                    # This is becuase concatted, so doesnt extract the xy locations. 
                    # See this code in self.primitives_extract_final():
                        # if shape is None and prms is None:
                        # Prim.input_prim("prototype_prim_abstract", {}, 
                        #         traj = traj)
                    # Do: 

            # grid_ver = "on_grid"
            grid_ver = self._get_grid_ver_manual(xgrid, ygrid, centersthis)          
            # assert grid_ver=="on_grid"      

        if get_grid:
            # Look for this saved
            gridparams = self.get_grid_params()
            # xgrid = gridparams["grid_x"]
            # ygrid = gridparams["grid_y"]
            xgrid = gridparams["grid_x_actual_after_rel"]
            ygrid = gridparams["grid_y_actual_after_rel"]

            # for key, val in gridparams.items():
            #     print('--', key, val)
            # # print(gridparams)
            # assert False

        return xgrid, ygrid, grid_ver

    def get_grid_params(self, also_return_each_kind_of_params=False):
        """ What is the grid structure (if any) for this task?
        This is saved now (3/2022) in PlanClass in matlab.
        - also_return_each_kind_of_params, bool, if true, then terurned nested dict
        dict[params_ver_1, ...], where params_ver_1 is a dict of params. each inner
        dict is different kidn fparams (se ebelow)).
        RETURNS:
        - gridparams, dict holding things like {grid_x, grid_y, ...}, with np array values
        - or None, if this before 3/2022 (grid not defined)
        """
        T = self.extract_as('ml2')
        # print(T.PlanDat.keys())
        # print("---")
        # for k, v in T.PlanDat.items():
        #     print('-', k)
        #     print(v)

        if T.PlanDat is None or "TaskGridClass" not in T.PlanDat.keys() or len(T.PlanDat["TaskGridClass"])==0:
            return None
        else:
            ###############
            # 1. the grid params in gridclass. these are usually default (e.g., gridpts)
            # but sometimes can chose to use this grid but not use the grid pts, in which
            # case the code below is the actual grid pts
            gridparams_background = T.PlanDat["TaskGridClass"]["GridParams"]

            #################
            # 2. grid params based on relations explicitly used (and sampled) to get 
            # location coords.
            TT = self.extract_monkeylogic_ml2_task()
            tsc = TT.tasksetclass_summary()

            # a) the "metaparams" used in Tasksetclass, to autoamticlaly generate relationsa nd grd.
            # tsc["tsc_params"]["quick_sketchpad_params"] =
            # ['grid', ['3.35_all', ['prims', 'grid_indexed_4_by_4', array(0, dtype=uint8), {}]]]
            
            grid_or_char = grid_scale = center_on = rel_kind = None # Defualts.

            if tsc is not None and "tsc_params" in tsc.keys() and tsc["tsc_params"] is not None and "quick_sketchpad_params" in tsc["tsc_params"].keys():
                # print("--------")
                # print(tsc)
                # print("--------")
                # print(tsc["tsc_params"])
                # print("--------")
                # print(tsc["tsc_params"].keys())
                if tsc["tsc_params"]["quick_sketchpad_params"] is not None:
                    grid_or_char = tsc["tsc_params"]["quick_sketchpad_params"][0] # "grid", "char"
                    grid_scale = tsc["tsc_params"]["quick_sketchpad_params"][1][0] # e.g, 3.2
                    center_on = tsc["tsc_params"]["quick_sketchpad_params"][1][1][0] # "prims", "chunks"
                    if len(tsc["tsc_params"]["quick_sketchpad_params"][1][1])>1:
                        rel_kind = tsc["tsc_params"]["quick_sketchpad_params"][1][1][1] # str: "<>...5_by_5"
                    if False:
                        # these may not always be avialble.. (len too short)
                        center_global = tsc["tsc_params"]["quick_sketchpad_params"][1][1][2]
                        tforms_global = tsc["tsc_params"]["quick_sketchpad_params"][1][1][3]
                    assert grid_or_char in ["grid", "char"], "I assumed so, made a mistake.. prob fine, but now im confused. tasksetclass_helper() in dragmonkey defines this."

            gridparams_tsc = {
                "grid_or_char": grid_or_char,
                "grid_scale": grid_scale,
                "center_on": center_on,
                "rel_kind": rel_kind,               
            }


            ##################
            # 3. Extract the actual relations used (across all tasks using this TSC)
            # - if grid, then these are the centers
            # - if char, then these are the delta relations (usualyl [0,0])
            # NOTE: Potential problem, the extracted grid is specific to this TSC, so if multiple TSC
            # in a single dataset, then might have different gridloc-actualloc mapping. This I plan to solved
            # in the wrapper extracting behclass, for it to check that all grids are same across tasks.
            if tsc is not None and "tsc_params" in tsc.keys() and tsc["tsc_params"] is not None and "relations" in tsc["tsc_params"].keys():
                rel_xy_values = []
                rels_list_of_dict = []
                for relation_struct in tsc["tsc_params"]["relations"]:
                    # print("RELATIONS:")
                    # print(relation_struct)
                    if "xy_pairs" in relation_struct.keys():
                        # Then this method was active, list of paris.
                        # relation_struct["xy_pairs"][1] is an array of weights for each pt
                        # # array([1., 1., 1., 1., 1.])
                        for pt in relation_struct["xy_pairs"][0]:
                            rel_xy_values.append(pt)
                        #assert False, "extract each coordinate "
                    else:
                        # xy provides x and y separately, then takes cross-product
                        xs = relation_struct["xy"][0]
                        ys = relation_struct["xy"][1]
                        attachpt1_locations = relation_struct["attachpt1"][0] # list of str, to sampel from,..e.g,  ['center_xylim', ...
                        attachpt1_weights = relation_struct["attachpt1"][1] # list of num, weights during sampleing.
                        attachpt2_locations = relation_struct["attachpt2"][0] # list of str, to sampel from,..e.g,  ['center_xylim', ...
                        attachpt2_weights = relation_struct["attachpt2"][1] # list of num, weights during sampleing.
                        fromprim_indices = relation_struct["fromprim"][0] # list of ints, indices of prims to attach to
                        fromprim_weights = relation_struct["fromprim"][1] # list of num, weights during sampleing.

                        # collect all xy in a bag
                        if len(xs.shape)==0:
                            xs = [xs]
                        if len(ys.shape)==0:
                            ys = [ys]
                        for x in xs:
                            for y in ys:
                                rel_xy_values.append(np.array([x, y]))

                        # save this rel
                        rels_list_of_dict.append(relation_struct)
            

                # compute the actual grid locaitons, which is the produce of the baseline grid 
                # and the relations. usually relations are integers, which puts prims on the grid.
                # but if not integer, then is produce of rel and grid.
                # - get all possible x and y locations (just to define grid, even if
                # takss were only on seubset of loations)
                xyall = np.stack(rel_xy_values) # (npts, 2)
                xs = np.sort(np.unique(xyall[:,0].round(decimals=3))) 
                ys = np.sort(np.unique(xyall[:,1].round(decimals=3)))
                # for k, v in gridparams_background.items():
                #     print('---', k, v)
                if False:
                    # this was mistake. Grid cell sizes are used for
                    # rescaling prim size, NOT for rescaling grid (i.e, relations).
                    # For that, use the grid_diff...
                    # (see PlanClass.interpret_relation())
                    x_mult = gridparams_background["cell_width"] # num
                    y_mult = gridparams_background["cell_height"] # num

                else:
                    x_mult = gridparams_background["grid_diff_x"]
                    y_mult = gridparams_background["grid_diff_y"]

                grid_center = gridparams_background["center"] # 2-list

                grid_x_actual = xs * x_mult + grid_center[0] # (n,) array
                grid_y_actual = ys * y_mult + grid_center[1]

            else:

                # rel_xy_values = rels_list_of_dict = grid_x_actual = grid_y_actual = None
                rel_xy_values = rels_list_of_dict = None

                # older code, usualyl didnt rescale, so these are accurate
                grid_x_actual = gridparams_background["grid_x"]
                grid_y_actual = gridparams_background["grid_y"]
            
            gridparams_rels = {
                "rel_xy_values": rel_xy_values,
                "rels_list_of_dict":rels_list_of_dict,
                "grid_x_actual_after_rel":grid_x_actual,
                "grid_y_actual_after_rel":grid_y_actual
            }

            # print("[GridParams]:")
            # print("===", gridparams_background)
            # print("===", gridparams_tsc)
            # print("===", gridparams_rels)
            # print("===", np.stack(gridparams_rels["rel_xy_values"]))

            ################# COMBINE
            # 1) flat
            gridparams = {}
            for k, v in gridparams_background.items():
                gridparams[k] = v
            for k, v in gridparams_tsc.items():
                assert k not in gridparams.keys()
                gridparams[k] = v
            for k, v in gridparams_rels.items():
                assert k not in gridparams.keys()
                gridparams[k] = v

            # 2) hierarchial
            gridparams_each = {
                "baseline":gridparams_background,
                "tsc_quick":gridparams_tsc,
                "rels":gridparams_rels,
            }

            # for k, v in gridparams_each.items():
            #     print("----")
            #     print(k)
            #     print(v)
            # assert False

            if also_return_each_kind_of_params:
                return gridparams, gridparams_each
            else:
                return gridparams

    def tokens_generate(self, params = None, inds_taskstrokes=None, 
        track_order=True, hack_is_gridlinecircle=False, assert_computed=True,
        include_scale=False, input_grid_xy=None, return_as_tokensclass=False,
                        reclassify_shape_using_stroke=False,
                        reclassify_shape_using_stroke_version="default",
                        tokens_gridloc_snap_to_grid=False,
                        list_cluster_by_sim_idx=None):
        """ [CONFIRMED - this is the ONLY place where self._DatSegs is accessed (across
        all code files).
        Wrapper to eitehr create new or to return cached. see
        _tokens_generate for more
        PARAMS:
        - assert_computed, bool, by default True, so that you explciitly turn this off 
        when you want to recompute it, useful to make sure correct params are used when 
        you do recompute
        - input_grid_xy, see inner
        - reclassify_shape_using_stroke, see Dataset()
        """
        assert isinstance(reclassify_shape_using_stroke_version, str)
        if not hasattr(self, "_TokensLocked"):
            self._TokensLocked = False

        if self._TokensLocked:
            assert False, "tried to generate new toekns, but it is locked..."

        if params is None:
            params = {}

        if assert_computed:
            # Then force that you have alreadey computed. this usefeul for 
            # gridline circle, where correct construction requires hack_is_gridlinecircle=True.
            assert hasattr(self, '_DatSegs') and self._DatSegs is not None and len(self._DatSegs)>0

        assert inds_taskstrokes==None, "only use tokens_generate to genereate in default order. To reorder, use tokens_reorder"
        assert len(params)==0, "do not pass in params. this needs to be default. instead, pass into tokens_reorder"

        # Check if it already exists.
        if hasattr(self, '_DatSegs') and self._DatSegs is not None and len(self._DatSegs)>0:
            tokens = self._DatSegs
        else:
            # Generate from scratch
            datsegs = self._tokens_generate(params, inds_taskstrokes, track_order, 
                hack_is_gridlinecircle=hack_is_gridlinecircle, 
                include_scale=include_scale, input_grid_xy=input_grid_xy,
                                            reclassify_shape_using_stroke=reclassify_shape_using_stroke,
                                            reclassify_shape_using_stroke_version=reclassify_shape_using_stroke_version,
                                            tokens_gridloc_snap_to_grid=tokens_gridloc_snap_to_grid,
                                            list_cluster_by_sim_idx=list_cluster_by_sim_idx)
            self._DatSegs = datsegs
            tokens = self._DatSegs

        if return_as_tokensclass:
            from pythonlib.drawmodel.tokens import Tokens
            return Tokens(tokens)
        else:
            return tokens

    def tokens_concat(self, tokens):
        """ concatenate these toeksn into a singl etoekn with average features
        IN PROGRESS: add mpre things
        PARAMS:
        - tokens, list of dicts
        RETURNS:
        - tok, a single dict
        """

        gridx = np.mean([t["gridloc"][0] for t in tokens])
        gridy = np.mean([t["gridloc"][1] for t in tokens])

        tok = {
            "gridloc":(gridx, gridy)
        }

        return tok

    def tokens_reorder(self, inds_taskstrokes):
        """ Return copy of datsegs in any desired order.
        NOTE: This must regenerate datsegs, becuasse the relational features need to know
        about ordering.
        """
        import copy

        # Shallow copy is good enough, becuase you just want to avoid replacing the
        datsegs_new = self.tokens_generate(assert_computed=True)
        # datsegs_new = [copy.copy(d) for d in self._DatSegs]
        datsegs_new = [datsegs_new[i] for i in inds_taskstrokes]
        return self._tokens_generate_relations(datsegs_new)


    def _tokens_generate_relations(self, datsegs):
        """ Given datsegs already computed from _tokens_generate, now
        compute relations. Only works if on grid (checks that locations are ints)
        PARAMS:
        - datsegs, return it from self._tokens_generate() or tokens_reorder()
        """

        from pythonlib.drawmodel.tokens import Tokens
        Tk = Tokens(datsegs)
        Tk.sequence_context_relations_calc()
        return Tk.Tokens

    def _tokens_delete(self):
        """ Deletes tokens..
        """
        self._DatSegs = None

    def _tokens_generate(self, params = None, inds_taskstrokes=None, 
            track_order=True, hack_is_gridlinecircle=False, include_scale=True,
            input_grid_xy = None, reclassify_shape_using_stroke=False,
            reclassify_shape_using_stroke_version="default",
            tokens_gridloc_snap_to_grid=False,
            list_cluster_by_sim_idx=None):
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
        - tokens_gridloc_snap_to_grid, bool, if True, then assigns each stroke to a grid location even
        if it doesnt perfectly match, by "snapping" to it, with some sanityc checks that is is close.
        RETURNS:
        - datsegs, list of dicts, each a token.
        """
        from math import pi

        if not reclassify_shape_using_stroke_version=="default":
            assert reclassify_shape_using_stroke==True, "mistake, you siganlled you want to reclassify shapes by stroke becuaes you inputed a param that is not default.."
        elif reclassify_shape_using_stroke_version == "cluster_by_shape_sim":
            assert list_cluster_by_sim_idx is not None

        shape_rename_perfblockey_decimals_to_defaults = False
        # Not yet coded, this would require hand=coding the shape labels (e.g., binning angles, etc), which can do in
        # taskgeneral

        if params is None:
            params = {}
        assert track_order==True, "if not, leads to mismatch in keys..."
        assert inds_taskstrokes==None, "see docs"        
        assert len(params)==0, "do not pass in params. this needs to be default. instead, pass into tokens_reorder"

        ############ PREPARE DATA
        # Extract shapes, formatted correctly
        # objects = self.ShapesOldCoords

        # print("TODO: instead of ShapesOldCoords, should first concatenate strokes that are (i) in same motif and (ii) touching")
        # print("see: get_grid_ver")
        # print("should be in self.PrimitivesObjects")
        # DONE: see self.info_...

        # also extract as primitiveclasses (NOTE: this should also be changed to primitivechunk)
        if not hasattr(self, "Primitives"):
            # then old version
            assert False, "must generate self.Primitives (instead of using Shapes...)"
            # Prims = None
        elif self.Primitives is None:
            assert False, "must generate self.Primitives (instead of using Shapes...)"
        else:
            Prims = self.Primitives
            # Rels = self.PlanDat["RelsBeforeRescaleToGrid"]
            # for k, v in self.PlanDat.items():
            #     print(k, '---', v)
            # print(1)
            # for x in Prims:
            #     print(x)
            # print(1)
            # for x in Rels:
            #     print(x)
            assert len(Prims)==len(self.PlanDat["ShapesAfterConcat"]), "just a sanity check, nothing special about ShapesAfterConcat"
            # assert len(Prims)==len(objects), "why mismatch? is one a chunk?"
        # p.Stroke.extract_spatial_dimensions(scale_convert_to_int=True)

        if inds_taskstrokes is None:
            inds_taskstrokes = list(range(len(Prims)))

        try:
            # objects = [objects[i] for i in inds_taskstrokes]
            # if Prims is not None:
            Prims = [Prims[i] for i in inds_taskstrokes]
            # Rels = [Rels[i] for i in inds_taskstrokes]
        except Exception as err:
            # print(objects)
            # print(len(objects))
            print(inds_taskstrokes)
            print(len(self.Strokes))
            print(Prims)
            # print(Rels)
            for p in Prims:
                p.print_summary()
            self.plotStrokes()
            raise err

        ############# Some grid params for this task
        # - was this on grid?
        # grid_ver = self.get_grid_ver()
        if input_grid_xy is None:
            xgrid, ygrid, grid_ver = self.get_grid_xy(hack_is_gridlinecircle=hack_is_gridlinecircle)
        else:
            _, _, grid_ver = self.get_grid_xy(hack_is_gridlinecircle=hack_is_gridlinecircle)
            assert len(input_grid_xy)==2
            assert isinstance(input_grid_xy, (list, tuple))
            xgrid = input_grid_xy[0]
            ygrid = input_grid_xy[1]
            assert np.all(np.diff(xgrid)>0)
            assert np.all(np.diff(ygrid)>0)

        # always get local grid (ie. this task), for releations
        xgrid_thistask, ygrid_thistask, _ = self.get_grid_xy(hack_is_gridlinecircle=hack_is_gridlinecircle)

        ################ METHODS (doesnt matter if on grid)
        def _orient(i):
            if False:
                # Angle, orientation
                # Old version. stopped working..
                this = Prims[i].extract_as("shape")
                if this[1]["theta"] is None:
                    return "undef"
                else:
                    th = this[1]["theta"]
                    # Prims[i].print_summary()
                    # assert False, "did not extract theta... this is the case for newer expts."
                if np.isclose(th, 0.):
                    return "horiz"
                elif np.isclose(th, pi):
                    return "horiz"
                elif np.isclose(th, pi/2):
                    return "vert"
                elif np.isclose(th, 3*pi/2):
                    return "vert"
                else:
                    print(this)
                    assert False
            else:
                if _shape(i)=="line-8-1-0":
                    return "horiz"
                elif _shape(i)=="line-8-2-0":
                    return "vert"
                else:
                    return "undef"
                # elif _shape(i) in ["circle-6-1-0", "arcdeep-4-4-0"]:
                #     return "undef"
                # else:
                #     print(_shape(i))
                #     assert False, "code it"

        # Spatial scales.
        def _width(i):
            if Prims is None:
                return None
            else:
                return Prims[i].Stroke.extract_spatial_dimensions(scale_convert_to_int=True)["width"]
        def _height(i):
            if Prims is None:
                return None
            else:
                return Prims[i].Stroke.extract_spatial_dimensions(scale_convert_to_int=True)["height"]
        def _diag(i):
            if Prims is None:
                return None
            else:
                return Prims[i].Stroke.extract_spatial_dimensions(scale_convert_to_int=True)["diag"]
        def _max_wh(i):
            if Prims is None:
                return None
            else:
                return Prims[i].Stroke.extract_spatial_dimensions(scale_convert_to_int=True)["max_wh"]

        def _shapeabstract(i):
            # e.g, "line"
            return Prims[i].ShapeNotOriented

        def _shape(i):
            # return string
            if reclassify_shape_using_stroke:
                if reclassify_shape_using_stroke_version == "cluster_by_sim":
                    assert len(list_cluster_by_sim_idx) == len(Prims)
                    # assert len(list_shape_name_overwrite) == len(Prims)
                    # shape_name_overwrite = list_shape_name_overwrite[i]
                    cluster_by_sim_idx = list_cluster_by_sim_idx[i]
                else:
                    cluster_by_sim_idx = None
                    # shape_name_overwrite = None
                sh = Prims[i].label_classify_prim_using_stroke(return_as_string=True,
                                               shape_rename_perfblockey_decimals_to_defaults=shape_rename_perfblockey_decimals_to_defaults,
                                               version=reclassify_shape_using_stroke_version,
                                               cluster_by_sim_idx = cluster_by_sim_idx)
                return sh
            else:
                return Prims[i].shape_oriented(include_scale=include_scale)
            # return objects[i][0]
        
        def _shape_oriented(i):
            # different name depedning on orientaion
            if _shape(i)=="line" and _orient(i)=="horiz":
                return "hline"
            elif _shape(i)=="line" and _orient(i)=="vert":
                return "vline"
            elif _shape(i)=="circle":
                return "circle"
            else:
                return _shape(i)




        ################## GRID METHODS
        if grid_ver in ["on_grid"]:
            # Sanity check that the hard coded things are correct.
            from pythonlib.tools.nptools import isin_close
            # for o in objects:
            #     print("------")
            #     print(o[1]["x"], xgrid)
            #     print(o[1]["y"], ygrid)
            #     if not isin_close(o[1]["x"], xgrid, atol=ATOL)[0] or not isin_close(o[1]["y"], ygrid, atol=ATOL)[0]:
            #         self.plotStrokes()
            #         print("---")
            #         print(o[1], xgrid, ygrid)
            #         print(isin_close(o[1]["x"], xgrid, atol=ATOL))
            #         print("---")
            #         assert False

            def _assign_prims_to_grid_locations(Prims, xgrid, ygrid):
                nver = len(ygrid)
                nhor = len(xgrid)
                locations = []
                for i, p in enumerate(Prims):
                    prms = p.extract_as("shape", include_scale=include_scale)[1]
                    xloc = prms["x"]
                    yloc = prms["y"]
                    # print("---", i, xloc, yloc)
                    
                    if not isin_close(xloc, xgrid, atol=ATOL)[0] or not isin_close(yloc, ygrid, atol=ATOL)[0]:
                        # Option 1 -- you state that this day everythign must be "snapped" to grid. Will then snap to closest grid location.
                        # snap_to_grid = True

                        if tokens_gridloc_snap_to_grid:
                            def _snap_to_grid(loc, grid):
                                """
                                - loc, scalar value
                                - grid, array of values (e.g., [0,1,2] represnting x coord.)
                                """
                                _diffs = np.abs(grid - loc)
                                _ind = np.argmin(_diffs)

                                # Sanity check that this is actually close to that grid location
                                assert _diffs[_ind] < 0.8, "is kind of far - are you sure you want to snap it?"
                                if len(grid)>1:
                                    if not _diffs[_ind]/sorted(_diffs)[1]<0.65:
                                        print(_diffs)
                                        print(_ind)
                                        print(_diffs[_ind], " --- ", sorted(_diffs))
                                        assert False, "loc is relatively not that close to the grid.."
                                return _ind 

                            try:
                                _xind = _snap_to_grid(xloc, xgrid)
                                _yind = _snap_to_grid(yloc, ygrid)
                            except Exception as err:
                                print(xloc, xgrid)
                                print(yloc, ygrid)
                                raise err

                            xind = int(_xind) - int((nhor-1)/2)
                            yind = int(_yind) - int((nver-1)/2)

                        else:
                            # POssibility 1- actually on grid, but onsets are offset based on the attachpt (e.g,, onset_panch). This
                            # leads to the x y locations off-grid. Look into the original Relations struct to get the 
                            # grid loc.
                            a = self.PlanDat["RelsBeforeRescaleToGrid"][i][1]==0 # meaning: is relation rel sketchpad.
                            b = self.PlanDat["RelsBeforeRescaleToGrid"][i][2][0] in ["center_xylim", "center_prim_sketchpad"] # meaning: is relation rel sketchpad.
                            c = self.PlanDat["RelsBeforeRescaleToGrid"][i][2][1] in ["onset_pancho", "onset_diego"] # ones where this fails. add to this list.
                            d = self.PlanDat["RelsBeforeRescaleToGrid"][i][2][2][0] == 0 # i.e., (0,0), which means center of page. only coded for this so far, since this was written fro Pancho, and all prims were at center. Assuming this allows me to hard code xind = yind = 0 below.
                            e = self.PlanDat["RelsBeforeRescaleToGrid"][i][2][2][1] == 0 

                            if c:
                                if True:
                                    # Then has no defined gridloc, since is alinged to stroke onset. I previsoly forced it to (0,0) wjhen was doing tasks with all at center,
                                    # but ing eneral this is not right. So now just wait to define later in Dataset (5/16/24)
                                    xind = yind = "IGN"
                                    grid_ver = "on_rel"
                                else:
                                    # Old version, where I forced it to be at (0,0), based on one day's expts with Pancho.
                                    if not d or not e:
                                        print(*self.PlanDat["RelsBeforeRescaleToGrid"][i][2])
                                        print(*self.PlanDat["RelsBeforeRescaleToGrid"][i][2][2])
                                        print(*self.PlanDat["Rels"])
                                        print("THIS IS IT? center before shift to pancho/diego onset...?", self.PlanDat["CentersAfterConcat"])
                                        print(a,b,c,d,e)
                                        print("xgrid, ygrid:", xgrid, ygrid)
                                        print("THSI TASK: xgrid, ygrid:", xgrid_thistask, ygrid_thistask)
                                        # print(self.get_grid_xy())
                                        # print(self._get_grid_ver_manual(xgrid, ygrid, ))
                                        assert False, "are these grid units or continuos? if grid, then use them for the cetner.. if not, then use Relations, not RelsBeforeRescaleToGrid?"
                                    if self.PlanDat["RelsBeforeRescaleToGrid"][i][0]=="translate_xy" and a and b and c and d and e:
                                        # Then is actually on grid! And it is at center (based on d and e being true).
                                        xind = 0
                                        yind = 0
                                        # 'RelsBeforeRescaleToGrid': [['translate_xy',
                                        #   array(0.),
                                        #   ['center_xylim', 'onset_pancho', array([0., 0.])]]],
                            else:
                                # not sure why failed...
                                print(self.PlanDat["RelsBeforeRescaleToGrid"])
                                self.plotStrokes()
                                print("---")
                                print(prms, xgrid, ygrid)
                                print(isin_close(prms["x"], xgrid, atol=ATOL))
                                print("---")
                                print(self.PlanDat["RelsBeforeRescaleToGrid"])
                                assert False, "prob just need to input it."
                    else:
                        # Good, got grid locations.
                        xind = int(isin_close(xloc, xgrid, atol=ATOL)[1][0]) - int((nhor-1)/2)
                        yind = int(isin_close(yloc, ygrid, atol=ATOL)[1][0]) - int((nver-1)/2)
                    
                    # print("locations:", xloc, yloc)
                    # print("locations(grid):", xind, yind)
                    locations.append((xind, yind))
                return locations

            locations = _assign_prims_to_grid_locations(Prims, xgrid, ygrid)
            locations_thistaskgrid = _assign_prims_to_grid_locations(Prims, xgrid_thistask, ygrid_thistask)
        else:
            assert grid_ver in ["on_rel"]


        # Create sequence of tokens
        datsegs = []
        for i in range(len(Prims)):
        # for i in range(len(objects)):

            # 1) Things that don't depend on grid
            datsegs.append({
                "orient_string":_orient(i),
                "shapeabstract":_shapeabstract(i),
                "shape":_shape(i),
                "shape_oriented":_shape_oriented(i),
                "width":_width(i),
                "height":_height(i),
                "diag":_diag(i),
                "max_wh":_max_wh(i),
                "Prim":Prims[i] if Prims is not None else None,
                "ind_taskstroke_orig":inds_taskstrokes[i],
                "center": Prims[i].Stroke.extract_center() # in pixels
                })
            
            # 2) Things that depend on grid
            if grid_ver=="on_grid":
                # Then this is on grid, so assign grid locations.
                datsegs[-1]["gridloc"] = locations[i]
                datsegs[-1]["gridloc_local"] = locations_thistaskgrid[i]
            elif grid_ver=="on_rel":
                # Then this is using relations, not spatial grid.
                # give none for params
                datsegs[-1]["gridloc"] = ("IGN", "IGN")
                datsegs[-1]["gridloc_local"] = ("IGN", "IGN")
            else:
                print(grid_ver)
                assert False, "code it"
        datsegs = self._tokens_generate_relations(datsegs)
        datsegs = tuple(datsegs)

        return datsegs

    #############

    def get_is_new_version(self):
        """ Returns True if this trial; uses the new "PLanclass"
        versipn of tasks
        PARAMS:
        - ind, index into self.Dat
        RETURNS:
        - bool.
        """
        if not hasattr(self, 'PlanDat') or self.PlanDat is None or len(self.PlanDat)==0:
            return False    
        else:
            return True

    def get_task_kind(self, SIMPLE = True):
        """ Get the kind for this task, things like prims_in_grid, chars, etc.
        This only after like early 2022, when created this distinction between 
        tasks. improving the method computing this. Does this by checking whether 
        The prims are all on the grid.
        NOTE: newer versions should do this byu checking the TaskSetClass (matlab 
        class) definitons for reltiaons ands o on.
        PARAMS;
        - SIMPLE< bool, if True, then only lmits to "prims_on_grid" and "character". If false, 
        thena lso include "prims_in_rel", motifs_in_grid. But tod o this, ened to extract and parse
        reelations.
        NOTE: if this is older version (before around 3/2022) then taskkind is not defined, so 
        returns "undefined"
        """

        if SIMPLE:
            grid_ver = self.get_grid_ver()
            if grid_ver=="undefined":
                # Then this is old code, not defined
                return "undefined"
            else:
                if grid_ver=="on_grid":
                    if len(self.Strokes)==1:
                        return "prims_single"
                    else:
                        return "prims_on_grid"
                elif grid_ver=="on_rel":
                    return "character"
                else:
                    print(grid_ver)
                    assert False
        else:
            # self.PlanDat["RelsBeforeRescaleToGrid"] Look in here for thetarelations,
            # if they are small, then this is prims in rel

            # To know if is motif, look at the toikes extracted from self.tokens_generate,
            # based ojn hard coded map between names and motif/oprim
            assert False


    ############ PARSER
    def input_parser(self, Parser):
        """ Parser class, input, is fine even if Parser is already done
        """
        self.Parser = Parser

    def input_parses_strokes(self, parses_list):
        """ Input list of parses, where each parse is a strokes
        """
        self.Parses = parses_list


    ############ Extract in different formats
    def extract_monkeylogic_ml2_task(self):
        return self.extract_as("ml2")

    def extract_as(self, this):
        """ Extract this task in different fromats
        """

        if this=="ml2":
            """ Extract as ml2 task. fails if this is not actually an ml2 task
            """
            if self.Params["input_ver"]=="ml2":
                return self.Params["input_params"]
            else:
                print(self.Params)
                assert False, "this is not an ml2 task"
        else:
            print(this)
            assert False

    ############## ML2 objectclass stuff
    # def ml2_plandat_extract(self):
    #     """ Helper to return, making sure to try to extract if looks
    #     like havent done yet
    #     """
    def ml2_objectclass_extract(self, auto_hack_if_detects_is_gridlinecircle_lolli=False):
        """
         Return Objectclass from  ml2 task. Try extactig if not yet extracted.
        """
        TT = self.extract_monkeylogic_ml2_task()
        if not hasattr(TT, "ObjectClass"):
            TT.objectclass_extract_all(auto_hack_if_detects_is_gridlinecircle_lolli=auto_hack_if_detects_is_gridlinecircle_lolli)
        return TT.ObjectClass

    def ml2_objectclass_extract_active_chunk(self, return_as_inds=False):
        """ Return the Active chukn
        RETURNS:
        - ChunkClass object,
        - OR: list of indices, flattened, in order of chunks
        """
        TT = self.extract_monkeylogic_ml2_task()
        C = TT.objectclass_extract_active_chunk()
        if return_as_inds:
            return C.extract_strokeinds_as("flat")
        else:
            return C

    ############### TASKSEQUENCER
    def ml2_tasksequencer_params_extract(self):
        """ 
        REturtn tasksequencer params used in ml2 to conditions the sequencing for this task.
        RETURNS:
        - sequence_ver, str name of type of sequ, e.g, 'directionv2'
        - seuqence_params, list of params for this ver,e .g, [['left'], 'default', array(1.)]
        """
        O = self.ml2_objectclass_extract()
        
        sequence_ver = O["Grammar"]["Tasksequencer"]["sequence_ver"]
        sequence_params = O["Grammar"]["Tasksequencer"]["sequence_params"]

        return sequence_ver, sequence_params

    ######### PLOT
    def plotStrokes(self, ax= None, ordinal=False):
        """ Quick plot of this task
        RETURNS:
        - ax, 
        """
        import matplotlib.pyplot as plt
        from pythonlib.drawmodel.strokePlots import plotDatStrokes, plotDatStrokesWrapper

        if ax is None:
            fig, ax = plt.subplots(1,1)

        if ordinal:
            # plotDatStrokes(self.Strokes, ax, clean_ordered_ordinal=True, number_from_zero=True)
            plotDatStrokesWrapper(self.Strokes, ax, color="k", mark_stroke_onset=True, 
                add_stroke_number=True, mark_stroke_center=True)
        else:
            plotDatStrokes(self.Strokes, ax, clean_task=True)
        return ax

    ############ SAVING
    def save(self, sdir, suffix):
        """ save all this as pickle
        saves as f"{sdir}/task_{suffix}.pkl"
        """
        import pickle
        
        path = f"{sdir}/task_{suffix}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Saved to: {path}")


    def save_task_for_dragmonkey(self, subdirname, fname, SDIR = f"{PATH_ANALYSIS_OUTCOMES}/main/resaved_tasks_for_matlab"):
        """ Save the task in a format that can be loaded in dragmonkey (matlab)
        to generate tasks for experiments
        PARAMS:
        - subdirname, string
        - fname, string
        --- file will be f"{SDIR}/{subdirname}/{fname}.mat"
        RETURNS:
        - saves at path above.
        """
        from pythonlib.tools.matlabtools import convert_to_npobject
        from scipy.io import savemat
        import os

        # 2) Things to save
        savedict = {}
        if "exptname" in self.PlanDat["Info"]["TaskSetClass"]["params"].keys():
            # older version
            savedict["tsc_name"] = self.PlanDat["Info"]["TaskSetClass"]["params"]["exptname"]
        else:
            savedict["tsc_name"] = self.PlanDat["Info"]["TaskSetClass"]["UniqueID"]

        savedict["tsc_constraints"] = self.PlanDat["Info"]["TaskSetClass"]["params"]["constraints"]
        if "online_updates" in self.PlanDat["Info"]["TaskSetClass"]["params"].keys():
            # older version
            savedict["tsc_online_updates"] = self.PlanDat["Info"]["TaskSetClass"]["params"]["online_updates"]
        else:
            savedict["tsc_online_updates"] = []


        savedict["tsc_params_planclass"] = self.PlanDat["Info"]["TaskSetClass"]["params_planclass"]
        savedict["Plan"] = self.PlanDat["Plan"]
        savedict["PrimsCategories"] = self.PlanDat["PrimsCategories"]

        # 3) convert to format that can be saved into mat file
        for k, v in savedict.items():
            savedict[k] = convert_to_npobject(v)

        # 4) Save
        sdir = f"{SDIR}/{subdirname}"
        os.makedirs(sdir, exist_ok=True)
        fname = f"{sdir}/{fname}.mat"
        savemat(fname, savedict)
        print("Saved to: ", fname)


    def info_shape_category(self):
        """ Good, the latest (12/2022) methods to get shape category for each
        prim, which is like "Lcentered" or "circle", etc.
        This works even with concatenated prims (their categories are concatenated
        strings). Uses info from both ObjectClass and PlanClass
        RETURNS:
        - list of str, each a category for a prim
        """

        return [P.ShapeNotOriented for P in self.Primitives]

    def info_extract_all_prim_versions(self):
        """ Get all versions of prims spanning odl and new task versions.
        This only fully wokrs fo new verisons. for old, modify to skip somet hings.
        Useful for taking notes.
        NOTES
        - how does it deal with motifs? Treats them as individual strokes, but chunked by
        hierarhcy.
        """

        out = {}
        # 1. task program prims:
        taskobj = self.extract_monkeylogic_ml2_task()
        taskobj.program_extract() 
        try:
            taskobj.objects_extract()
        except AssertionError as err:
            pass

        out["task_program"] = taskobj.Objects
                
        # 2. plan prims. These are base prims before concat.
        # (only started after glc)
        if self.PlanDat is not None:
            out["plan_prims"] = self.PlanDat["Prims"]
        else:
            out["plan_prims"] = None

        # 3. objectclass prims. After concat. Post 12/2022, concatting in correct way.
        # Before then, one time did concat (glc, lolli). 
        if "Features_Active" in self.ml2_objectclass_extract().keys():
            out["objectclass_features"] = self.ml2_objectclass_extract()["Features_Active"]
        else:
            out["objectclass_features"] = None

        # 4. Objectlass prims, which are all after concatenation.
        # - gridlinecircle (lolli, that one day): 4 line/circle concatted to 2 lollis.
        # - all other expt: indices correspnd to concated strokes, hier are sequencings.
        if "ChunksListClass" in self.ml2_objectclass_extract().keys():
            out["objectclass_chunks"] = self.ml2_objectclass_extract()["ChunksListClass"]
        else:
            out["objectclass_chunks"] = None

        # Note the location of original prims before concat in that one day for glc.
        out["objectclass_prims_before_concat_glc"] = "this is in original ObjectClass.ChunkList"

        # 4. final, merging plandat and objclass (plandat, except when concat)
        out["final_prims"] = self.Primitives
        out["final_strokes"] = self.StrokesOldCoords

        return out


#################### TO WORK WITH LISTS OF TASKS
def getSketchpad(TaskList, kind):
    """ will fail in any Tasks in Tasklist have different sketchpads
    RETURN:
    - np array, 2,2
    """
    tmp = np.stack([task._Sketchpads[kind] for task in TaskList]) # combine all tasks
    tmp = tmp.reshape(tmp.shape[0], -1) 
    assert np.all(np.diff(tmp, axis=0)==0.) # chekc that all are the same.
    spad = TaskList[0]._Sketchpads[kind]
    return spad


