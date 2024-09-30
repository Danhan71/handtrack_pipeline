""" Initalize a new expt
"""
import deeplabcut
from pyvm.utils.experiments import get_params
from pyvm.dlc.utils import find_expt_config_paths
from pyvm.globals import BASEDIR, WINDOWS
import os
import argparse


def downsample_all_videos(list_video_paths, camnames, 
        width_pix=800, suffix="-downsampled", overwrite=False):
    """ downsample all videos, place in same directory, but name has suffix.
    width_pix, new width (long edge), with same aspect ratio.
    PARAMS;
    - list_video_paths, full paths 
    - camnames, list of str names, same len as list_video_paths, for purpose to detecting if outputs fname done already.
    - overwrite, bool, if False, then skips if detects that downsampled video already created.
    RETURNS:
    - list_new_video_paths, list of full paths to downsampled vidoes (works ewven if skipped since already done)
    NOTE:
    - for each video file, creates a downsampled version:
    --- vidfname --> <camname>-vidfname-downsampled.mp4

    """
    from pyvm.utils.cvtools import get_video_wh, calc_downscale_params
    from pyvm.utils.directories import getPaths, get_path_video_downsampled
    from pythonlib.tools.expttools import modPathFname

    assert len(list_video_paths)==len(camnames)

    list_new_video_paths = []
    for video, camname in zip(list_video_paths, camnames):

        # Check if it is already downsampled
        video_downsampled = get_path_video_downsampled(video, camname)
        list_new_video_paths.append(video_downsampled) # always return, whether or not did replace.

        if os.path.exists(video_downsampled) and overwrite==False:
            # Then skip
            print("[downsample] skipping, since already exists: ", video_downsampled)
            pass
        else:

            w, h = get_video_wh(video)
            hnew = calc_downscale_params(w, h, width_pix)
            # downsample it.

            # print(getPaths(video))
            # print(modPathFname(video, "test", suffix))
            # assert False
            # # "{camname}-{fname}-downsampled.mp4"
            path_without_camname_prefix = deeplabcut.DownSampleVideo(video, 
                width = int(width_pix), height =int(hnew), outsuffix=suffix)

            # move file
            # from: fname-downsampled to camname-fname-downsampled
            path_final = modPathFname(path_without_camname_prefix, camname)

            # sanity check
            if path_final !=video_downsampled:
                print(path_final)
                print(path_without_camname_prefix)
                print(camname)
                print(video_downsampled)
                assert False, "prob stupid mistake."


    return list_new_video_paths


def gather_and_preprocess_videos(dirname, condition, list_camnames, inds_cams_to_take, animal,
    do_downsample=True, vidnums=None):
    """
    Collect list of videos (paths), combined across all cameras. Does the following:
    --- Downsample vidoes (maintain a. ratio).
    --- Rename videos with camera name as prefix, so can combine multiple to train a single combined model.
    --- Returns list of new names.
    - inds_cams_to_take, which cams (indexes into list_camnames)
    - do_downsample, then downsamples videos, and takes those. appends suffix to video names. 
    By default downsamples to width of 800.
    OUT:
    - flattened list of full paths to downsampled vides
    """
    from pyvm.utils.directories import get_paths_videos_in_dir, get_path_video_downsampled
    import glob

    video_all = [] # list of all videos
    camnames_all = [] # camera name, so that can mix videos in same dir

    # Collect all videos across cameras
    for ind in inds_cams_to_take:
        camera = list_camnames[ind]

        # get full paths to videos 
        videos = get_paths_videos_in_dir(camera, dirname, condition, animal, inds_cams=vidnums)
        
        video_all.extend(videos)
        camnames_all.extend([camera for _ in range(len(videos))])

    if do_downsample:
        list_new_video_paths = downsample_all_videos(video_all, camnames=camnames_all)

        # move all videos to new filename that includes the name of the camera.
        # list_new_video_paths = [get_path_video_downsampled(vidpath, camname) for zip(list_new_video_paths, camnames_all)]

        # tmp = []
        # for path, camera in zip(list_new_video_paths, camnames_all):
        #     newpath = modPathFname(path, camera)
        #     tmp.append(newpath)
        # list_new_video_paths = tmp
    else:
        assert False, "then must use another method to move to new filename with cameraname as prefix"
        list_new_video_paths = video_all

    if WINDOWS:
        list_new_video_paths = [repr(path) for path in list_new_video_paths]

    return list_new_video_paths


def initialize_expt(exptname_or_date, animal, condition_keep=None):
    """ 
    Initalizes expt and related directories and returns the string path to config file.
    If already initialized, then just returns the path (NOTE: this is quirky, only works if
    you run this on same day, since expts are indexed by date. To get paths, use 
    find_expt_config_paths insetad)
    """
    from pyvm.utils.directories import get_metadata
    experimenter='Lucas' # Enter the name of the experimenter

    # ------------ RUN
    params = get_params(exptname_or_date,animal) # get hand-entered params
    metadat = get_metadata(exptname_or_date, animal) # get auto params

    if "dirname" in params.keys():
        # old version, where dirs where date_expt
        dirname = params["dirname"]
    else:
        # new version, dirname is date.
        dirname = exptname_or_date

    # USE THIS TO GET ALL VIDS IN DIRECTORIES.
    list_pathconfig = [] # to hold

    for i, condition in enumerate(params["list_conditions"]):
        if condition_keep is not None:
            if condition not in condition_keep:
                continue

        bodyparts = params["list_bodyparts"][i]
        skeleton = params["list_skeletons"][i]

        # EXPEIRMENT/CONDITION specific params
        numframes2pick = params["list_numframes2pick"][i]
        vidnums = params["list_vidnums"][i]
        combine_cameras = params["list_combinecams"][i]
        list_camnames = params["list_camnames"]

        # if "list_camnames" in params.keys():
            # list_camnames = params["list_camnames"]
        # else:
        #     list_camnames = metadat["list_camnames"]

        if combine_cameras:
            # 1) Downsamples vidoes...
            list_videos = gather_and_preprocess_videos(dirname, condition, list_camnames, 
                range(len(list_camnames)), do_downsample=True, vidnums=vidnums, animal=animal) 

            working_directory = f"{BASEDIR}/{animal}/{dirname}/{condition}/DLC"
            task="combined-" + "_".join(params['list_camnames'])

            path_config_file=deeplabcut.create_new_project(task,experimenter, list_videos,
                                                                           copy_videos=False,
                                                                           working_directory=working_directory,
                                                                           bodyparts = bodyparts,
                                                                           skeleton = skeleton,
                                                                           return_configpath_always=True,
                                                                           numframes2pick=numframes2pick,
                                                                          )             
            list_pathconfig.append(path_config_file)

        else:
            working_directory = f"{BASEDIR}/{animal}/{dirname}/{condition}/DLC"
            for i in range(len(list_camnames)):
                camera = list_camnames[i]
                task=f"{exptname}_{condition}_{camera}"

                list_videos = gather_and_preprocess_videos(dirname, condition, list_camnames, 
                    [i], do_downsample=True, vidnums=vidnums)

                path_config_file=deeplabcut.create_new_project(task,experimenter,video,
                                                               copy_videos=False,
                                                               working_directory=working_directory,
                                                               bodyparts = bodyparts,
                                                               skeleton = skeleton,
                                                               return_configpath_always=True,
                                                               numframes2pick=numframes2pick
                                                              ) 
                list_pathconfig.append(path_config_file)

    return list_pathconfig



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Description of your script.")
    parser.add_argument("name", type=str, help="Experiment name/date")
    parser.add_argument("algo", type=str, help="Algorithm for frame extraction")
    parser.add_argument("animal", type=str, help="What do you think")
    parser.add_argument("--data", type=str, help="Run in debug mode")
    parser.add_argument("--skip", type=int, help="Skip frame extraction")

    args = parser.parse_args()

    DATA = args.data
    EXTRACT_FRAMES = args.skip
    exptname_or_date = args.name # e.g, 220521
    algo = args.algo
    animal = args.animal
    print ("Data: ", DATA)
    print("Extract Frames: ", EXTRACT_FRAMES)
    print("Name: ", exptname_or_date)
    print("Algorithm: ", algo)
    
    if DATA == "wand":
        # wand has much fewer vidoes...
        condition_keep = ["wand"] # GOOD> dont do checkerboard. it doesnt need dlc.
    elif DATA == "behavior":
        condition_keep=["behavior"]
    else:
        condition_keep = ["behavior", "wand"] # GOOD> dont do checkerboard. it doesnt need dlc.
    ########################## RUN
    list_pathconfig = initialize_expt(exptname_or_date, animal,condition_keep=condition_keep)
    if EXTRACT_FRAMES:
        ######## Extract frames
        #there are other ways to grab frames, such as uniformly; please see the paper:

        #AUTOMATIC:
        for path_config_file in list_pathconfig:
            print("EXTRACTING FRAMES FOR THIS:", path_config_file)
            # deeplabcut.extract_frames(path_config_file, algo="kmeans", userfeedback=False) 
            deeplabcut.extract_frames(path_config_file, algo=algo, userfeedback=False) 
