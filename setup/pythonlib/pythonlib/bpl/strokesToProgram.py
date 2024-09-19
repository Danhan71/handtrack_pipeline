""" generally working from strokes --> programs
Can also score strokes by first infering program, then scoring.
Generally donest work much (if at all) with images)
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from pybpl.model import CharacterModel
from pybpl.library import Library
from .scoring import scoreMPs_factorized, scoreMPs # since legacy code expects it to be here.


# Load Library
USE_HIST = True
LIB_PATH = '/data1/code/python/pyBPL/lib_data/'
lib = Library(LIB_PATH, use_hist=USE_HIST)
model = CharacterModel(lib)

def infer_MPs_from_strokes(strokes_list, indices_list, dataset_preprocess_params, sketchpad_edges, 
    use_fake_images =True, imageWH = 105, k=1, do_rescore=True, save_checkpoints=None):
    """ wrapper to extract best-fitting motor programs
    INPUTS:
    - strokes_list, list of strokes, each of which is a list of np array.
    - indices_list, list of indices (could be string or int), saved , to
    allow linking back to some dataset.
    - use_fake_images, then doesnt render strokes into image.
    Is ok, since the scoring doesnt use image anyway (although) note
    that optimization of continuos params does...
    - k, num programs to return per strokes.
    - sketchpad_edges, [], e.g., D.Metadats[0]["sketchpad_edges"].T
    - do_rescore, then rescores using pyBPL library. 
    - save_checkpoints, either None, or [n, path], where n is how often, and
    path is path dir (will append filename)
    - dataset_preprocess_params, just for saving purposes, doesnt affect anything here.
    RETURNS:
    - MPlist, list (len strokeslist), of lists (num parses per strokes) of MPs 
    (i.e., MP is a character type).
    NOTES:
    - uses matlab engine and Brenden Lake's BPL lbirary
    - Uses also Reuben Feinmans's PyBPL library
    """

    if save_checkpoints is not None:
        dosave = True
        ncheck = save_checkpoints[0]
        path = save_checkpoints[1]
        from pythonlib.tools.expttools import makeTimeStamp
        ts = makeTimeStamp()
        import os
        pathsave = f"{path}/infer_MPs_from_strokes-{ts}"
        os.makedirs(pathsave)
        def _save(out_all, score_all, MPlist=None):
            print("Saving checkpoint")
            import pickle
            with open(f"{pathsave}/out_all.pkl", "wb") as f:
                pickle.dump(out_all, f)
            with open(f"{pathsave}/score_all.pkl", "wb") as f:
                pickle.dump(score_all, f)
            with open(f"{pathsave}/indices_list.pkl", "wb") as f:
                pickle.dump(indices_list, f)
            params = {
                "sketchpad_edges":sketchpad_edges,
                "use_fake_images":use_fake_images,
                "imageWH":imageWH,
                "k":k,
                "do_rescore":do_rescore,
                "dataset_preprocess_params":dataset_preprocess_params
            }
            with open(f"{pathsave}/params.pkl", "wb") as f:
                pickle.dump(params, f)
            if MPlist is not None:
                with open(f"{pathsave}/MPlist.pkl", "wb") as f:
                    pickle.dump(MPlist, f)

    else:
        dosave=False

    def _matlab2py(out):
        """ unpacks matlab output into py format"""
        for i, parse in enumerate(out): # for each program
            for j, stroke in enumerate(parse):
                sparams = stroke[0]
                sparams = [np.asarray(p) for p in sparams]
                rparams = stroke[1]
                rparams = [np.asarray(p) for p in rparams]
                out[i][j] = [sparams, rparams]
        return out

    # Preparematlab
    import matlab.engine
    import os
    
    eng = matlab.engine.start_matlab()
    bplscripts_path = "/home/lucast4/code/matlab/BPLscripts"
    eng.addpath(eng.genpath(bplscripts_path), nargout=0)
    bpl_path = os.environ['BPL_PATH']
    eng.addpath(eng.genpath(bpl_path), nargout=0)
    ls_path = os.environ['LIGHTSPEED_PATH'] # for lighspeed matlab toolbox, used in BPL. see BPL readme.
    eng.addpath(eng.genpath(ls_path), nargout=0)


    # Make fake image
    if use_fake_images:
        I = np.random.rand(imageWH, imageWH)
        I = I>0.7
        I = matlab.logical(I.tolist())
    else:
        assert False, "then you must pass in image. not coded"

    # Run, iterating over all strokes.
    out_all = []
    score_all = []
    for i, strokes in enumerate(strokes_list):
        strokes_matlab = prepStrokes(strokes, sketchpad_edges)
        out, score = eng.strokes_to_MPs(I, strokes_matlab, k, nargout=2)
        score_all.append(score)
        out_all.append(_matlab2py(out))

        if dosave and i%ncheck==0:
            _save(out_all, score_all)

    # Convert each out into a caracter type
    MPlist = []
    for out in out_all:
        tmp = []
        for prog in out:
            ctype = params2ctype(prog)
            tmp.append(ctype)
        MPlist.append(tmp)

    if dosave:
        _save(out_all, score_all, MPlist)

    # Rescore using the pyBPL library
    score_all = [scoreMPs(MPs) for MPs in MPlist]

    if dosave:
        _save(out_all, score_all, MPlist)

    return MPlist, score_all


def prepStrokes(strokes, sketchpad_edges, imageWH = 105):
    """ prep strokes (monkey coords) to correct format
    for inference
    - sketchpad_edges, [], e.g., D.Metadats[0]["sketchpad_edges"].T
    """
    from pythonlib.drawmodel.image import convCoordGeneral
    strokes_matlab = [s[:,:2] for s in strokes]
    # if sketchpad_edges is None:
    #     sketchpad_edges = np.array([[0., 1.], [0., 1.]]) # fuill screen, this is possibel if in matlab didn't specify sketchpad.
    strokes_matlab = [convCoordGeneral(s, sketchpad_edges, np.array([[0, imageWH-1], [-(imageWH-1), 0]])) for s in strokes_matlab]
    strokes_matlab = [[s.tolist() for s in strokes_matlab], [s.tolist() for s in strokes_matlab]]
    return strokes_matlab


def _stroke(sparams):
    """ takes output from _matlab2py, and converts to
    stroketype object
    # TODO:
    - ids is assumed to be length 1, ie.. not using subparts...
    this makes sense since passing in strokes.

    """
    from pybpl.objects.part import StrokeType
    # unpack
    ids, invscales, shapes_type, shapes_token = sparams
    # preprocess
    nsub = 1
    ids = int(ids-1)
    shapes = shapes_token
    assert nsub==1, "I diont htink it accurately does multiple substrokes yet"
    # generate stroke
    return StrokeType(
        nsub=torch.tensor(nsub), 
        ids=torch.tensor(ids).view(1),
#         shapes=lib.shape['mu'][0].view(5, 2, 1),
        shapes=torch.tensor(shapes.astype("float32")).view(5, 2, 1),
        invscales=torch.tensor(invscales).view(1)
    )   
    return s

def _rel(rparams):
    """ takes output from _matlab2py, and converts to
    relation object
    """

    from pybpl.objects.relation import RelationIndependent, RelationAttachAlong, RelationAttach
    
    rtype = rparams[0]
    
    if rtype == "unihist":
        gpos, nprev = rparams[1:]
        
        return RelationIndependent(
            category='unihist',
            gpos=torch.tensor(gpos).squeeze(),
            xlim=lib.Spatial.xlim,
            ylim=lib.Spatial.ylim,
            )
    
    elif rtype =="mid":
        attach_subix, ncpt, _, eval_spot_token, attach_ix, _ = rparams[1:]
        attach_ix = int(attach_ix)-1
        attach_subix = int(attach_subix)-1
        # print(torch.tensor(eval_spot_token.astype("float32")).shape)
        return RelationAttachAlong(
            category=rtype,
            attach_ix=torch.tensor(attach_ix).view(1),
            attach_subix=torch.tensor(attach_subix).view(1),
            eval_spot=torch.tensor(eval_spot_token.astype("float32")),
            ncpt=ncpt
            )
        
    elif rtype in ['start', 'end']:
        attach_ix, _ = rparams[1:]
        attach_ix = int(attach_ix)-1

        return RelationAttach(
            category=rtype,
            attach_ix=torch.tensor(attach_ix).view(1),
            )
        
    else:
        print(rparams)
        assert False, "which rel type is this?"


def params2ctype(prog):
    """ 
    - prog, is a single program, e..g,
    get from matlab, out= strokes_to_MPs,
    and prog = out[0], since out is multiple programs
    potnetialyl
    RETURNS:
    - ctype, a CharacterType
    """
    from pybpl.objects.concept import CharacterType

    # collect part and rel for each strok (p)
    P = [_stroke(p[0]) for p in prog]
    R = [_rel(p[1]) for p in prog]
    k = torch.tensor(len(P))
    # initialize the type
    ctype = CharacterType(k, P, R)
    
    return ctype


def plotMP(ctype, score=None, xlim=(0, 104), ylim=(-104, 0), ax=None):
    """ plot both img and motor, for this program (type level)"""

    def box_only(obj):
        # for better image visualization in matplotlib
        obj.tick_params(
            which='both',
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False
        )
    # Sample token
    ctoken = model.sample_token(ctype)

    # fig, axes = plt.subplots(2, 1, figsize=(10,5))
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5,5))

    # Sample prob image
    if False:
        pimg = model.get_pimg(ctoken)
        ax = axes.flatten()[0]
        ax.imshow(pimg.detach().numpy(), cmap='Greys')
        box_only(plt)
        # ax.set_xlim(xlim)
        # ax.set_ylim(ylim)

    # PLOT MOTOR TRAJECTORY
    motor = [p.motor for p in ctoken.part_tokens] # list of (nsub, ncpt, 2)
    
    # apply affine transformation if needed
    if ctoken.affine is not None:
        motor = apply_warp(motor, ctoken.affine)
    motor_flat = torch.cat(motor) # (nsub_total, ncpt, 2)

    # numpy
    motor_flat = [m.detach().numpy() for m in motor_flat]

    # plot
    from pythonlib.drawmodel.strokePlots import plotDatStrokes
    # ax = axes.flatten()[1]
    plotDatStrokes(motor_flat, ax, each_stroke_separate=True)
    if score is not None:
        ax.set_title(f"score {score}")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax



    