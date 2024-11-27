""" Preprocess to move camera files to correct directories, etc 
"""

import os
import sys
import glob
import csv

# Update this based on your machine
from pyvm.globals import BASEDIR

MAP_SERIAL_TO_NAME = {
    18061552: "flea",
    21155487: "fly1",
    21495924: "bfs1", # older version, low res
    20142776: "bfs2", # newer version
    22131878: "fly2" #The mythical 5th camera
}
list_conditions_dirs = ["3dgrid", "behavior", "checkerboard", "wand"]

#Custom exception for no cam metadat


def get_metadata(DATE, animal, condition=None, allow_generate_from_scratch=True):
    exptname=DATE
    """ 
    Generate metadata about cameras and paths, given this date
    Assumes your raw data is organized like:
    <BASEDIR>/<DATE>/<DATE>_<exptname>, e..g,
    '/data2/camera/220515/220515_charfinal4/Camera1'
    '/data2/camera/220515/220515_charfinal4/Camera2'
    '/data2/camera/220515/220515_charfinal4/Camera3'
    '/data2/camera/220515/220515_charfinal4/Camera4'
    '/data2/camera/220515/220515_charfinal4_wandgrid/Camera4' and so on...
    PARAMS:
    - allow_generate_from_scratch, bool, if False, then you must have already 
    run this and saved.
    RETURNS:
    - [if run for firs ttime, and allow_generate_from_scratch==True] metadat dict, and saves metadat
    - [if not already run, and allow_generate_from_scratch==False] then None.
    - [if already run] metadat dict
    """
    from pythonlib.tools.expttools import writeDictToYaml, load_yaml_config

    path_metadata = f"{BASEDIR}/{animal}/{DATE}/metadat.yaml"

    if os.path.exists(path_metadata):
        # Then metadat already ran and saved
        metadat = load_yaml_config(path_metadata)
    else:
        if allow_generate_from_scratch:
            # construct it for first time, and save
            basedir = f"{BASEDIR}/{animal}/{DATE}" # basedir for all data/analyses for this day

            # Get the mapping between camera num and name
            data_dirs = glob.glob(f"{basedir}")
            print("-- These dirs holding raw data: ", data_dirs)

            # Check that have expeected data
            # (220515_charfinal4 and 220515_charfinal4_wandgrid)
            if len(data_dirs)==0:
                print(data_dirs)
                assert False, "no data found at all"
            # elif len(data_dirs)>2:
            #     print(data_dirs)
            #     assert False, "found too much data, is this becuase you restarted recording?"
            #     # figure out how to deal with this issue... just make these separate folders entirely?
            # elif len(data_dirs)==1:
            #     print(data_dirs)
            #     print("-- Only found one data dir. Missing checkerboard/wand/grid?")
            #     pass
            else:
                print(f"-- Found {len(data_dirs)} data dirs (ok)!")
                # expect them to be <date>_<exptname>...
                from pythonlib.tools.expttools import extractStrFromFname

                # Get exptname
                # out = []
                list_exptname = []
                map_camnum_to_serial_name = {}
                conditions_dict = {}

                # Try to get one set of videos for each condition (e.g,, beh, wand, ..)

                list_exptname.append(exptname)

                # Get the camera dirs for this data dir
                camera_dirs = glob.glob(f"{basedir}/Camera*")
                map_camname_to_path = {} # specific for this data dir
                for dirthis_cam in camera_dirs:

                    # what is the camnum
                    # - assumes is named Camera<int>
                    indthis = dirthis_cam.find("Camera")
                    camnum = int(dirthis_cam[indthis+6:])

                    # load metadat for the first trial.
                    try:
                        dict_meta = load_campy_matadat_csv(dirthis_cam)
                    except:
                        continue 
                    # dict_meta = load_campy_matadat_csv(f"{dirthis_cam}/metadata-t0.csv")
                    serialnum = int(dict_meta["cameraSerialNo"])
                    camname = MAP_SERIAL_TO_NAME[serialnum]
                    
                    #Outright discrimination against bfs2 if mroe than 4 cams
                    if len(camera_dirs) > 4 and camname == 'bfs2':
                        continue
                     
                    # store
                    if camnum in map_camnum_to_serial_name.keys():
                        # check that the same camera num mappings are used across all data dirs
                        assert map_camnum_to_serial_name[camnum][0]==serialnum
                        assert map_camnum_to_serial_name[camnum][1]==camname
                    else:
                        map_camnum_to_serial_name[camnum] = (serialnum, camname)

                    # map from camname to directory
                    map_camname_to_path[camname] = dirthis_cam

                    # Each condition is linked to specific videos in this directory
                    map_condition_interpreted_to_videonums = interpret_condition(DATE, condition, animal)

                    for condition_interpreted, vidnums in map_condition_interpreted_to_videonums.items():
                        if vidnums is None:
                            # e.g., for checkerboard
                            assert condition_interpreted=="checkerboard", "or else why None?"
                        elif len(vidnums)==0:
                            # get all videos
                            vidnums = [name for name in  os.listdir(dirthis_cam) if name.endswith(",mp4")]
                        else:
                            assert isinstance(vidnums, list), " should be list of ints, vidnums"

                        # assert condition_interpreted not in conditions_dict.keys()
                        conditions_dict[condition_interpreted] = {
                                "path":{basedir},
                                "exptname":exptname,
                                "condition":condition,
                                "condition_interpreted":condition_interpreted,
                                "map_camname_to_path":map_camname_to_path,
                                "vid_nums":vidnums}

                print("-- Found these data dirs (and associated medata):")
                print(conditions_dict)

                # Sanity check that all paths have same exptname
                assert len(set(list_exptname))==1, "found multiple exptnames..."
                EXPTNAME = list_exptname[0]
                print("-- Found exptname: ", EXPTNAME)

                print("--Got this mapping from camera num to (serial, camname): ")
                for k, v in map_camnum_to_serial_name.items():
                    print(k, v)

                # Get list of all camnames
                list_camnames = []
                for v in map_camnum_to_serial_name.values():
                    list_camnames.append(v[1])

                # The final dirs, separated by condition (beh, wand, etc) and camera
                map_condition_cam_to_dir = {}
                for camname in list_camnames:
                    path = f"{basedir}/{condition}/{camname}"
                    map_condition_cam_to_dir[(condition, camname)] = path


                # Collect all into a dict
                metadat = {
                    "path_base":basedir,
                    "paths_data_dirs":data_dirs,
                    # "metadat_data":out,
                    "conditions_dict":conditions_dict,
                    "exptname":EXPTNAME,
                    "map_camnum_to_serial_name":map_camnum_to_serial_name,
                    "list_camnames":list_camnames,
                    "list_conditions_dirs":condition,
                    "map_condition_cam_to_dir": map_condition_cam_to_dir
                }

            # Save it
            writeDictToYaml(metadat, path_metadata)
        else:
            # then reutrn None
            return None
    return metadat

def get_cam_list(date, animal):
    """Gets cam name list w/o using metadata file for reasons"""
    map_camnum_to_serial_name = {}
    basedir = f"{BASEDIR}/{animal}/{date}"
    camera_dirs = glob.glob(f"{basedir}/Camera*")
    map_camname_to_path = {} # specific for this data dir
    for dirthis_cam in camera_dirs:

        # what is the camnum
        # - assumes is named Camera<int>
        indthis = dirthis_cam.find("Camera")
        camnum = int(dirthis_cam[indthis+6:])

        # load metadat for the first trial.
        try:
            dict_meta = load_campy_matadat_csv(dirthis_cam)
        except:
            print("bad camera data")
            continue
        # dict_meta = load_campy_matadat_csv(f"{dirthis_cam}/metadata-t0.csv")
        serialnum = int(dict_meta["cameraSerialNo"])
        camname = MAP_SERIAL_TO_NAME[serialnum]

        # store
        if camnum in map_camnum_to_serial_name.keys():
            # check that the same camera num mappings are used across all data dirs
            assert map_camnum_to_serial_name[camnum][0]==serialnum
            assert map_camnum_to_serial_name[camnum][1]==camname
        else:
            map_camnum_to_serial_name[camnum] = (serialnum, camname)

        # map from camname to directory
        map_camname_to_path[camname] = dirthis_cam

    # Get list of all camnames
    list_camnames = []
    for v in map_camnum_to_serial_name.values():
        list_camnames.append(v[1])


    return list_camnames


def interpret_condition(date, condition, animal):
    """ directroy path string to human interpretable condition
    RETURNS:
    - dict mapping from possible conditions(huuman name) to video nums. vidnums is list of ints
    or empty (get all files or not defined)
    """

    # load the yaml file that tells you the mapping between condition and video nums
    from .experiments import get_params_yaml
    
    
    params = get_params_yaml(DATE=date, animal=animal)
    def _get_vidnums_this_condition(condition_interpreted):
        # Map this condition to video nums
        # for k,v in params.items():
        #     print(k, ' ---- ', v)
        vidnums = params["vidnums_to_copy"][condition_interpreted]

        return vidnums

    out = {}
    if condition == "behavior":
        condition_interpreted = "behavior"
        out[condition_interpreted] = _get_vidnums_this_condition(condition_interpreted)
    else:
        # is this condition wandgrid?
        if condition in ["wandgrid", "gridwand", "wand", "grid"]:
            FOUND_WANDGRID = True
            for condition_interpreted in ["wand", "3dgrid"]:
                # condition_interpreted = "wandgrid"
                out[condition_interpreted] = _get_vidnums_this_condition(condition_interpreted)
        elif condition in ["checker", "checkerboard"]:
            out["checkerboard"] = None # checkerboard vbideos dont have numbers, since done using SpinVIEW Gui.
        else:
            print(condition)
            assert False, "cannot interpret this, waht is?"
    
    # # make sure matches names of dirs
    # assert condition_interpreted in list_conditions_dirs

    return out
            

def load_campy_matadat_csv(path):
    """ Loads files, 
    PARAMS;
    - path, path to the folder holding metdata. Will find any
    metadata (e.g, metadata-t1.csv) and use that
    RETURNS:
    - dict
    """
    import csv


    # Find any metadata in this directory
    paths = glob.glob(f"{path}/metadata-t*.csv")
    print(paths)
    if len(paths)>0:
        print(f"no metadat exists in here? {path}")
        raise ExceptionNoCam("No metadat for this camera")
    path_metadata = sorted(paths)[0] # take any...

    with open(path_metadata, mode='r') as infile:
        reader = csv.reader(infile)
        mydict = {rows[0]:rows[1] for rows in reader}
    return mydict


# def get_path_cond_cam_final(METADAT, condition, camname):
#     """ Get the (final) path for this condition and cmaera
#     PARAMS:
#     - condition, str, e.g., beh, grid...
#     - camname, str, e.g, flea
#     RETURNS:
#     - path
#     """

#     for dat in METADAT["metadat_data"]:
#         dat["path"]
#         dat["path"]

# USELESS
# ^w^
# def move_videos_to_final_path(METADAT):
#     copy_videos_to_final_path(METADAT, do_move=False)

# def copy_videos_to_final_path(METADAT, do_move=False):
#     """ Copy videos from raw directories to standard directory structure that 
#     separates videsos by behavior, checkerboard, wand, grid
#     Does not delete original files... (ACTUALLY DOES if do_move)
#     NOTE: does sanity checks, only moves if all filetypes exist at sources, and no files at
#     targ (so doesnt overwrite)
#     """
#     import shutil
#     conditions_dict = METADAT["conditions_dict"]
#     for condition, dat in conditions_dict.items():

#         # the source dir
#         # path_vids = dat["path"]

#         map_camname_to_path = dat["map_camname_to_path"]
#         vidnums = dat["vid_nums"]
#         print("these videos for condition: ", condition)
#         print(vidnums)
#         for camname, path_vids in map_camname_to_path.items():

#             path_targ = METADAT["map_condition_cam_to_dir"][(condition, camname)]
#             # move all files
#             print(f"[{condition}]", "Copying these videos from: ", path_vids, " to ", path_targ)

#             # Make these directories
#             os.makedirs(path_targ, exist_ok=True)
#             print(condition, camname, "Moving/copying all files to ", path_targ)
#             # iterate over each video
#             for num in vidnums:

#                 # Its files
#                 list_fname_all_filetypes = [f"frametimes-t{num}.npy", f"metadata-t{num}.csv", f"vid-t{num}.mp4"]
#                 list_pathsource_all_filetypes = [f"{path_vids}/{fname}" for fname in list_fname_all_filetypes]
#                 list_pathdest_all_filetypes = [f"{path_targ}/{fname}" for fname in list_fname_all_filetypes]
                
#                 # print(list_fname_all_filetypes)
#                 # print(list_pathsource_all_filetypes)
#                 # print(list_pathdest_all_filetypes)

#                 # Sanity checks:
#                 # 1. Only move this video num if it has all file types (source)
#                 check_all_source_exists = all([os.path.exists(path) for path in list_pathsource_all_filetypes])
#                 # 2). Make sure the destinatesion don't eixst (no overwrite)
#                 check_no_dest_exists = not any([os.path.exists(path) for path in list_pathdest_all_filetypes])

#                 # print(list_fname_all_filetypes)
#                 # print(check_all_source_exists, check_no_dest_exists)
#                 # assert False

#                 if check_all_source_exists==True and check_no_dest_exists==True:
#                     # good, do move/copy
#                     for pathin, pathout in zip(list_pathsource_all_filetypes, list_pathdest_all_filetypes):
#                         # pathin = f"{path_vids}/{fname}"
#                         # pathout = f"{path_targ}/{fname}"
#                         if False:
#                             if do_move:
#                                 print("MOVING: ", pathin, ' to ' , pathout)
#                             else:
#                                 print("COPYING: ", pathin, ' to ' , pathout)
#                         # copy it
#                         # assert False
#                         if do_move:
#                             # Move
#                             os.replace(pathin, pathout)
#                         else:
#                             if False:
#                                 # Copy, including metadata.
#                                 shutil.copy2(pathin, pathout)
#                             else:
#                                 # actually, better to make symlink
#                                 os.symlink(pathin, pathout)
#                 else:
#                     print(f"[ERROR?] skipping {list_fname_all_filetypes}, because either source dont all exist, or some dest exists")


def get_paths_videos_in_dir(camname, date, condition, animal, inds_cams=None,
    exclude_downsampled=True):
    """ Returns list of full paths to all videos.
    PARAMS:
    - camname, e..g, bfs1
    - date, str or int, YYMMDD, could also be date_expt, like 220515_chunksbyshap4 (for older version)
    - condition, str, e.g., wand, behavior, etc..
    - inds_cams, list of ints, vidoe indiices. leave none to get all. only gets those
    that actually exist.
    RETURNS:
    - list of full paths
    """

    vdir = f"{BASEDIR}/{animal}/{date}/{condition}/{camname}"
    print(vdir)
    assert os.path.exists(vdir), "first generate these directories.. and move cam files to them"

    # Then get all vidoes
    videos = glob.glob(f"{vdir}/*.mp4") + glob.glob(f"{vdir}/*.avi")

    if inds_cams:
        # assert condition=="behavior", "filename format not sure if applies to other conditions?"
        def _video_is_in_indscams(v):
            for i in inds_cams:
                if f"vid-t{i}." in v:
                    return True
            return False
        videos = [v for v in videos if _video_is_in_indscams(v)]

    # Exclude videos with downsampled in name, they are not hte originals.
    if exclude_downsampled:
        videos = [v for v in videos if "-downsampled" not in v]

    return videos



def getPaths(path_video):
    """ get useful paths based on path to video
    RETURNS:
    - dict holding paths
    """
    import os

    path_dir = os.path.split(path_video)[0]
    path_vid = os.path.split(path_video)[1]

    path_frames = f"{path_dir}/{path_vid}-frames"

    return {
    "path_dir":path_dir, 
    "path_vid":path_vid, 
    "path_frames":path_frames
    }

def get_path_video_downsampled(video_fullpath, camname):
    """
    REturns teh full path to the downsampled version of this vidoe
    camname-video-downsampled.mp4
    PARAMS:
    - camera, str, e.g., bfs1
    """
    from pythonlib.tools.expttools import modPathFname
    newpath = modPathFname(video_fullpath, camname, "downsampled", do_move=False)
    return newpath



        


