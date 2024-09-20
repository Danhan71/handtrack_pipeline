""" after getting DLC trained wand, then run this to extract frames with common
pts across all cameras
"""
from pyvm.classes.videoclass import Videos
import os
import numpy as np
from pythonlib.tools.expttools import writeStringsToFile
from pyvm.globals import BASEDIR, CB_DIR


def extract_save_easywand_pts(V, list_part =None, indgrp = 0):
    """ Finalize easywand extraction, extract pts for each cam and bodypart
    and save to folder.
    PARAMS:
    - list_part, order matters, e.g., ["red", "blue"]
    --- leave None to get auotmatically.
    - indgrp, assumes that this is 0.
    NOTE: 
    - assumes there is only single video per group, and one trial (N videos)
    """
    # # OLD VERSION:
    # prmlist = [
    #     (0, "red", ["x", "y"]),
    #     (1, "red", ["x", "y"]),
#     (2, "red", ["x", "y"]),
    #     (0, "blue", ["x", "y"]),
    #     (1, "blue", ["x", "y"]),
    #     (2, "blue", ["x", "y"]),
    # ]

    # # assumes framenums are same across vidoes
    # if False:
    #     # 
    #     dat = V.DatVideos[0]
    #     framenums = dat["good_frames"]
    # else:
    #     # This works even if loading V from scratch. find videos saved to disk
    #     _, framenums, _ = V.goodframes_mapping_new_old_index(0)

    # vals = []
    # cameras = []
    # for prm in prmlist:
    #     xy = V.dlc_data_part_feat_mult(prm[0], prm[1], prm[2])
    #     vals.append(xy[framenums,:])
        
    #     datv = V.wrapper_extract_dat_video(prm[0])
    #     cameras.append(datv["camera_name"])

    # vals = np.concatenate(vals, axis=1)

    # assert np.any(np.isnan(vals)) == False, "why nan?"
    # print(vals.shape)
    # print(vals[:5])
    # print(cameras)

    #Get good frames as list
    _, framenums, _ = V.goodframes_mapping_new_old_index(0)
    indtrial = V.inds_trials()[0]
    #Pass list to this function to extract a matrix of pts
    vals, columns = V.dlc_extract_pts_matrix(indtrial, list_part, framenums)
    cameras = [c[1] for c in columns]


    ##### SAVE COORDINATES TO TEXT FIEL

    basedir = V.Params["load_params"]["basedir"]
    SDIR = f"{basedir}/wand_calibration"
    os.makedirs(SDIR, exist_ok = True)
    fname = f"{SDIR}/wandPointsScreen.csv"
    np.savetxt(fname, vals, delimiter=",")

    fname = f"{SDIR}/camera_names_in_order.txt"
    writeStringsToFile(fname, cameras)
    fname = f"{SDIR}/columns.txt"
    writeStringsToFile(fname, columns)
    fname = f"{SDIR}/rows_frames.txt"
    writeStringsToFile(fname, framenums)

def extract_save_checkerboard_calib(V):
    """ First must have done stuff in checkerboard. here extract for each cam, in format
    readable by easyWand
    """
    dict_cam = V.get_cameras()
    cameras_in_order = [dict_cam[i][0] for i in range(len(dict_cam))]
    # cameras_in_order = ["flea", "fly1", "bfs1", "bfs2", "fly2"]

    rows = []
    for i, cam in enumerate(cameras_in_order):
        # load calibration output
        x = [datv for datv in V.DatVideos if datv["camera_name"]==cam]
        
        # assert len(x)==1
        datv = x[0]
        
        assert os.path.isdir(CB_DIR), "CB calib dir not found, please check and update config"
        sdir = f"{CB_DIR}/{cam}/collected_frames/calib_pycv2"

        
        # resolution
        w, h = V.resolutions(datv["index"])["orig"]
        
        # focal point
        mtx = np.load(f"{sdir}/calibration_mtx.npy")
        fp = np.mean([mtx[0,0], mtx[1,1]])
        
        # principal pt
        px = mtx[0, 2]
        py = mtx[1, 2]
        
        # distortions
        dists = np.load(f"{sdir}/calibration_dist.npy")
        
        rows.append([i, fp, w, h, px, py, 1] + (dists).tolist()[0])

    camera_profile = np.stack(rows)

    basedir = V.Params["load_params"]["basedir"]
    SDIR = f"{basedir}/wand_calibration"
    os.makedirs(SDIR, exist_ok = True)
    fname = f"{SDIR}/camera_profiles_checkerboard.txt"
    np.savetxt(fname, camera_profile, delimiter=",")

    fname = f"{SDIR}/camera_profiles_checkerboard_columns.txt"
    writeStringsToFile(fname, cameras_in_order)


# NOTE: old version was to extract common frames by hand:
# ## METHOD 1:  manually aneter frames
# if False:
#     # dict holding good frames

#     ## Camtest3
#     if False:
#         good_frames = {
#             ("cam1flea", 0, "vid-t5DLC_resnet_50_camtest3_wand2_cam1fleaApr19shuffle1_50000_labeled"):
#                 [56, 77, 116, 141, 181, 192, 215, 239, 260, 268, 286, 298, 
#                 335, 367, 491, 516, 532, 547, 579, 619, 648, 682, 718, 748, 762, 
#                  794, 803,
#                 75, 114, 135,  152,  183, 194, 207, 216, 239, 253, 270, 285, 322, 331,
#                 344, 351, 361, 494, 503, 517,  527, 543, 612, 648, 672, 693, 713, 752, 777, 
#                 795, 817, 839],
#             ("cam2blackfly", 0, "vid-t5DLC_resnet_50_camtest3_wand2_cam2blackflyApr19shuffle1_100000_labeled"):
#                 [56, 77, 116, 141, 181, 192, 215, 239, 260, 268, 286, 298, 
#                 335, 367, 491, 516, 532, 547, 579, 619, 648, 682, 718, 748, 762, 
#                  794, 803,
#                 75, 114, 135,  152,  183, 194, 207, 216, 239, 253, 270, 285, 322, 331,
#                 344, 351, 361, 494, 503, 517,  527, 543, 612, 648, 672, 693, 713, 752, 777, 
#                 795, 817, 839],
#         }

#     ## Camtest4
#     if True:
#         good_frames = {
#             ('cam1bflyu3', 0, 'vid-t0DLC_resnet_50_camtest4_wand_cam1bflyu3Aug13shuffle1_50000_labeled'):
#                 [104, 125],
#             ('cam2flea3', 0,'vid-t0DLC_resnet_50_camtest4_wand_cam2flea3Aug13shuffle1_50000_labeled'):
#                 [104, 125],
#             ('cam3bfs', 0, 'vid-t0DLC_resnet_50_camtest4_wand_cam3bfsAug13shuffle1_150000_labeled'):
#                 [104, 125],
#         }

#     # Extract these goodfframes
#     V.input_good_frames(good_frames, True)
#     # == copy good frames across vidoes into a single directory, renaming as frame1, 2 ...

#     # Do this once for each camera
#     V.collect_goodframes_from_videos()

#     # Collect frames with dlc labels
#     V.collect_goodframes_from_videos(vid_kind="dlc_labeled")
# else:
#     # METHOD 2: auto, uniformly sample frames.
#     V.sample_and_extract_auto_good_frames()

if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Description of your script.")
    parser.add_argument("name", type=str, help="Experiment name/date")
    parser.add_argument("animal", type=str,help="Your mother")
    # parser.add_argument("--rmcam", type=str, help="List of cams to remove from the wand points extraction", default = None, required= False )

    args = parser.parse_args()

    name = args.name
    date = name.split("_")[0]
    expt = '_'.join(name.split("_")[1:])
    animal = args.animal
    # if args.rmcam is not None:
        # rmcam = args.rmcam.split(',')

    V = Videos()
    V.load_data_wrapper(date=date,expt=expt,animal=animal, condition="wand")
    V.import_dlc_data()
    

    # 1) Extract uniformly all frames, same across all cameras.
    V.sample_and_extract_auto_good_frames(ntoget=5000)

    # 2) Prune, but only keeping those passing DLC threshold for likeli.
    V.filter_good_frames_dan(screen=True)

    extract_save_easywand_pts(V)
    # extract_save_checkerboard_calib(V)
