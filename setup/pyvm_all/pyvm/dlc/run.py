
from pyvm.globals import BASEDIR
from utils import find_analysis_path

HACK1 = False
HACK2 = False
#Artifacts of Lucas code, d Have to go and chnage a bunch of stuff to make it work without these names defined



# Resources to check for plotting tools, etc.
# https://github.com/DeepLabCut/DLCutils/blob/master/Demo_loadandanalyzeDLCdata.ipynb
# https://github.com/DeepLabCut/DeepLabCut/blob/master/docs/HelperFunctions.md
# https://stackoverflow.com/questions/61412870/how-to-access-multi-index-h5-data

import deeplabcut
import tensorflow as tf
from initialize import find_expt_config_paths
from pythonlib.tools.expttools import load_yaml_config
import os
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "2"

def get_video_list(path_config_file):
    """ return list of videos"""
    # import yaml
    config = load_yaml_config(path_config_file)
    vidlist = list(config["video_sets"].keys())
    return vidlist

def load_config_file(path_config_file):
    """ return dict"""
    # import yaml
    config = load_yaml_config(path_config_file)
    return config



def main(path_config_file, ver, train_ver="new", checkpoint_path=None, maxiters=100000, base_path=None,
        analysis_suffix = "nosuffix"):
    vidlist = get_video_list(path_config_file)
    # ANALYSIS - Get base path
    if ver in ["analyze", "filter_and_plot", "create_labeled_video"]:
        # if doing analyze, then start a new folder. Otherwise take the most recent folder.
        from pythonlib.tools.expttools import findPath, extractStrFromFname, writeDictToYaml, load_yaml_config

        list_path = findPath(base_path, [[analysis_suffix]], None)

        if ver=="analyze":
            analysis_path, _ = find_analysis_path(base_path, analysis_suffix, do_iterate_if_exists=True)

            # Make the directory
            import os
            os.makedirs(analysis_path, exist_ok=True)

            # save the list of analysis videos
            vidlist = sorted(vidlist)
            writeDictToYaml({"list_videos":vidlist}, f"{analysis_path}/list_videos.yaml")

        else:
            # continue an analysis
            analysis_path, _ = find_analysis_path(base_path, analysis_suffix, do_iterate_if_exists=False)

            # load analysis videos. check they identical to input here.
            tmp = load_yaml_config(f"{analysis_path}/list_videos.yaml")
            # print(len(tmp["list_videos"]))
            # print(len(vidlist))
            assert tmp["list_videos"] == sorted(vidlist), "input vids must match analysis vids"

    ############################## RUN
    if ver=="check_labels":
        # Extracts images and overlays labels.
        # deeplabcut.check_labels(path_config_file, visualizeindividuals=False)
        deeplabcut.check_labels(path_config_file)

    elif ver=="train":
        # always initialize a new training dataset (even if continue training), since if dont then will restart iter from 0,..
        # and overwrite previously active training dataset.
        # NOTE: if run this, always starts from iter = 0.
        # checkpoint only says what starting weights to use. 

        if train_ver=="new":
            # Should only use this if is truly new. otherwise will overwrite old training.
            deeplabcut.create_training_dataset(path_config_file)
            deeplabcut.train_network(path_config_file, allow_growth=True, saveiters=10000, maxiters=maxiters)

        elif train_ver=="new_after_refine":
            # Then merge datasets, but start with fresh weights
            deeplabcut.merge_datasets(config=path_config_file) # iterate, so a new training dataset.
            deeplabcut.create_training_dataset(path_config_file) # need to run this to actually construct the dataset
            deeplabcut.train_network(path_config_file, allow_growth=True, saveiters=10000, 
                maxiters=maxiters)

        elif train_ver=="continue":
            # pick up from last checkpoint
            deeplabcut.merge_datasets(config=path_config_file) # iterate, so a new training dataset.
            deeplabcut.create_training_dataset(path_config_file) # need to run this to actually construct the dataset
            assert checkpoint_path is not None
            deeplabcut.train_network(path_config_file, allow_growth=True, saveiters=10000,
                checkpoint = checkpoint_path, maxiters=maxiters)
        else:
            assert False

    elif ver=="evaluate":
        """ Makes folder evaluation_results. 
        Evaluates the network based on the saved models at different stages of the training network.\n
        The evaluation results are stored in the .h5 and .csv file under the subdirectory 'evaluation_results'.
        Change the snapshotindex parameter in the config file to 'all' in order to evaluate all the saved models.

        plotting: bool, optional
            Plots the predictions on the train and test images. The default is ``False``; if provided it must be either ``True`` or ``False``

        show_errors: bool, optional
            Display train and test errors. The default is `True``

        NOTE: turned off plotting, because if plotting=True, then it fails to re-analyze if retrained model.
        RETURNS:
        - /evaluation_results.
        """
        deeplabcut.evaluate_network(path_config_file, plotting=False)

    elif ver =="analyze":
        """
        Makes prediction based on a trained network. The index of the trained network is specified by parameters in the config file (in particular the variable 'snapshotindex')

        Output: The labels are stored as MultiIndex Pandas Array, which contains the name of the network, body part name, (x, y) label position \n
                in pixels, and the likelihood for each frame per body part. These arrays are stored in an efficient Hierarchical Data Format (HDF) \n
                in the same directory, where the video is stored. However, if the flag save_as_csv is set to True, the data can also be exported in \n
                comma-separated values format (.csv), which in turn can be imported in many programs, such as MATLAB, R, Prism, etc.

        destfolder: string, optional
            Specifies the destination folder for analysis data (default is the path of the video). Note that for subsequent analysis this
            folder also needs to be passed.

        RETURNS:
        - 3 files for each video, coords and the below meta.pickle.

        Contents of output files:
        - "...meta.pickle":
            start  --  1636830768.121373
            stop  --  1636830776.2009337
            run_duration  --  8.07956075668335
            Scorer  --  DLC_resnet_50_combined-bfs_flea_fflyNov9shuffle1_100000
            DLC-model-config file  --  {'stride': 8.0, 'weigh_part_predictions': False, 'weigh_negatives': False, 'fg_fraction': 0.25, 'mean_pixel': [123.68, 116.779, 103.939], 'shuffle': True, 'snapshot_prefix': '/data2/camera/211106_cagetest2/behavior/DLC/combined-bfs_flea_ffly-Lucas-2021-11-09/dlc-models/iteration-0/combined-bfs_flea_fflyNov9-trainset95shuffle1/test/snapshot', 'log_dir': 'log', 'global_scale': 0.8, 'location_refinement': True, 'locref_stdev': 7.2801, 'locref_loss_weight': 1.0, 'locref_huber_loss': True, 'optimizer': 'sgd', 'intermediate_supervision': False, 'intermediate_supervision_layer': 12, 'regularize': False, 'weight_decay': 0.0001, 'crop_pad': 0, 'scoremap_dir': 'test', 'batch_size': 8, 'dataset_type': 'imgaug', 'deterministic': False, 'mirror': False, 'pairwise_huber_loss': True, 'weigh_only_present_joints': False, 'partaffinityfield_predict': False, 'pairwise_predict': False, 'all_joints': [[0]], 'all_joints_names': ['fingertip'], 'dataset': 'training-datasets/iteration-0/UnaugmentedDataSet_combined-bfs_flea_fflyNov9/combined-bfs_flea_ffly_Lucas95shuffle1.mat', 'init_weights': '/data2/camera/211106_cagetest2/behavior/DLC/combined-bfs_flea_ffly-Lucas-2021-11-09/dlc-models/iteration-0/combined-bfs_flea_fflyNov9-trainset95shuffle1/train/snapshot-100000', 'net_type': 'resnet_50', 'num_joints': 1, 'num_outputs': 1}
            fps  --  123.47422893810806
            batch_size  --  8
            frame_dimensions  --  (586, 800)
            nframes  --  187
            iteration (active-learning)  --  0
            training set fraction  --  0.95
            cropping  --  False
            cropping_parameters  --  [0, 800, 0, 586]
        - .csv and .h5: coordinates for each frame
        """
        print("** SAVING ANALYSES AT: ", analysis_path)
        # print(vidlist)
        # print(analysis_path)
        # assert False
        deeplabcut.analyze_videos(path_config_file, vidlist, videotype='.mp4', save_as_csv=True, 
            destfolder=analysis_path, batchsize=8)

    elif ver=="filter_and_plot":
        """ Filter over time
        Fits frame-by-frame pose predictions with ARIMA model (filtertype='arima') or median filter (default).

        destfolder: string, optional
            Specifies the destination folder for analysis data (default is the path of the video). Note that for subsequent analysis this
            folder also needs to be passed.

        p_bound: float between 0 and 1, optional
            For filtertype 'arima' this parameter defines the likelihood below,
            below which a body part will be consided as missing data for filtering purposes.
        RETURNS:
        - Same as analyze_videos, but with "filtered" suffix.
        """
        print("** FILTERING ANALYSES AT: ", analysis_path)
        deeplabcut.filterpredictions(path_config_file, video=vidlist, videotype=".mp4", 
            destfolder=analysis_path)
        """
        plot trajectories over time

        filtered: bool, default false
        Boolean variable indicating if filtered output should be plotted rather than frame-by-frame predictions. Filtered version can be calculated with deeplabcut.filterpredictions

        destfolder: string, optional
            Specifies the destination folder that was used for storing analysis data (default is the path of the video).

        RETURNS:
        - plot-posts folder, with quantifications
        """
        print("** PLOTTING TRAJS FOR ANALYSES AT: ", analysis_path)
        deeplabcut.plot_trajectories(path_config_file, videos=vidlist, videotype=".mp4", 
            destfolder=analysis_path, filtered=True)
        deeplabcut.plot_trajectories(path_config_file, videos=vidlist, videotype=".mp4", 
            destfolder=analysis_path, filtered=False)

    elif ver=="create_labeled_video":
        """
        Labels the bodyparts in a video. Make sure the video is already analyzed by the function 'analyze_video'

        filtered: bool, default false
            Boolean variable indicating if filtered output should be plotted rather than frame-by-frame predictions. Filtered version can be calculated with deeplabcut.filterpredictions
        destfolder: string, optional
            Specifies the destination folder that was used for storing analysis data (default is the path of the video).
        """
        print("** create_labeled_video FOR ANALYSES AT: ", analysis_path)
        if HACK2:
            vidlist = [v for v in vidlist if "ffly-" in v]
        # if HACK2:
        #     vidlist = [v for v in vidlist if "bfs1-" in v]
        
        if False:
            deeplabcut.create_labeled_video(path_config_file, vidlist, destfolder=analysis_path, filtered=False)
            if False:
                deeplabcut.create_labeled_video(path_config_file, vidlist, destfolder=analysis_path, filtered=True)
        else:
            # UNCOMMENT THIS if videos not converting:
            # This error:
            # timebase 1000/120893 not supported by MPEG 4 standard, the maximum admitted value for the timebase denominator is 65535
            deeplabcut.create_labeled_video(path_config_file, vidlist, destfolder=analysis_path, filtered=False, fastmode=False, outputframerate=60.0)
            if False:
                # Could do, but less imporant than unfiltered, and takes a bit of time.
                deeplabcut.create_labeled_video(path_config_file, vidlist, destfolder=analysis_path, filtered=True, fastmode=False, outputframerate=60.0)


    elif ver=="extract_outlier_frames":
        # After run analyze_videos, can pull out outlier frames.
        # NOTES:
        # 

        # NOTE on outlier algorithm. Tested:
        # "uncertain": uses likeli. not good. mostly gets low likeli stuff that is correclty low likeli
        # "jumps": better, often in body, probably is jump between frames both with low likeli.
        # "fitting": best, got stuff where missed a finger.
        # Note: this takes a while, so start by taking just N of videos. Another reason is subsequence
        # merge only iterates model iteration if you have completely refined all extracted videos. Can still
        # merge fine, but will not iterate model.
        outlieralgorithm = "fitting"
        nvids = 100
        #numframes2pick = 5 # frames per vid
        # -- RUN
        analysis_path, _ = find_analysis_path(base_path, analysis_suffix, do_iterate_if_exists=False)
        # os.symlink(analysis_path, base_path)
        import random
        vidlist = random.sample(vidlist, nvids)
        deeplabcut.extract_outlier_frames(path_config_file, vidlist, 
            extractionalgorithm="uniform", automatic=True, 
            outlieralgorithm=outlieralgorithm, destfolder=analysis_path) #Here dest folder is for where 
                                                #analyzed videos are, not for any sort of outputs
        ## Messing around
        # deeplabcut.extract_outlier_frames(path_config_file, ['/home/danhan/Documents/hand_track/data/Pancho/220317/behavior/DLC/combined-flea_bfs1_ffly_bfs2-Lucas-2024-07-03/analyze_videos-allvideos-0'], 
        #     extractionalgorithm="uniform", automatic=True, 
            # outlieralgorithm=outlieralgorithm) #pass a specific video
        # NOTE:
        # then run deeplabcut.refine_labels(path_config_file) in notebook (uses gui)
        # Then run: "retrain_after_refinement" here
    elif ver=="extract_outlier_frames_wand":
        # After run analyze_videos, can pull out outlier frames.
        # NOTES:
        # 

        # NOTE on outlier algorithm. Tested:
        # "uncertain": uses likeli. not good. mostly gets low likeli stuff that is correclty low likeli
        # "jumps": better, often in body, probably is jump between frames both with low likeli.
        # "fitting": best, got stuff where missed a finger.
        # Note: this takes a while, so start by taking just N of videos. Another reason is subsequence
        # merge only iterates model iteration if you have completely refined all extracted videos. Can still
        # merge fine, but will not iterate model.
        outlieralgorithm = "fitting"
        nvids = 5
        #numframes2pick = 5 # frames per vid
        # -- RUN
        analysis_path, _ = find_analysis_path(base_path, analysis_suffix, do_iterate_if_exists=False)
        # os.symlink(analysis_path, base_path)
        import random
        deeplabcut.extract_outlier_frames(path_config_file, vidlist, 
            extractionalgorithm="uniform", automatic=True, 
            outlieralgorithm=outlieralgorithm, destfolder=analysis_path) #Here dest folder is for where 
                                                #analyzed videos are, not for any sort of outputs
        ## Messing around
        # deeplabcut.extract_outlier_frames(path_config_file, ['/home/danhan/Documents/hand_track/data/Pancho/220317/behavior/DLC/combined-flea_bfs1_ffly_bfs2-Lucas-2024-07-03/analyze_videos-allvideos-0'], 
        #     extractionalgorithm="uniform", automatic=True, 
            # outlieralgorithm=outlieralgorithm) #pass a specific video
        # NOTE:
        # then run deeplabcut.refine_labels(path_config_file) in notebook (uses gui)
        # Then run: "retrain_after_refinement" here
    elif ver=="retrain_after_refinement":
        # after extract outlier frames, relabel, then this merges new labels with old, increases training iteration
        # by 1, then creates new training set and starts training again.
        # deeplabcut.merge_datasets(path_config_file) # merge

        # Old, when I thought better to continual trainig
        # main(path_config_file, "train", train_ver="continue", checkpoint_path=checkpoint_path) # train

        # But better to start with new weights.
        main(path_config_file, "train", train_ver="new_after_refine") # train
    else:
        assert False

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Description of your script.")
    parser.add_argument("name", type=str, help="Experiment name/date")
    parser.add_argument("--iters", type=int, help="Integer value for train iters", default=100000)
    parser.add_argument("--cond", type=str, help="String formatted like list for conditions", default='empty')
    parser.add_argument("--checkp", type=str, help="String formatted like list for checkpoints", default='empty')
    parser.add_argument("--tver", type=str, help="String formatted like list for training types", default='empty')
    parser.add_argument("--frac", type=int, help="Fraction of GPU to use", default=60)
    parser.add_argument("--step", type=int, help="Run step")

    args = parser.parse_args()

    name = args.name
    print("Name:            ", name)
    iters = args.iters
    print("Iters:           ", iters)
    step=args.step
    #Assign input values if they exist
    if args.cond != 'empty':
        conditionlist = args.cond.split(",")
        print("Condition List:  ", conditionlist)
    if args.checkp != 'empty':
        list_checkpointpath = args.checkp.split(",")
        print("Checkpoint Paths:", list_checkpointpath)
    if args.tver != 'empty':
        list_trainver = args.tver.split(",")
        print("Train Vers:      ", list_trainver)
    if args.frac is not None:
        FRAC = (args.frac/100.0)
        print("GPU Fraction:    ", FRAC)

    #Assign some default values if the user skips the optional paramters. Probably a more efficient way to do this
    if(len(conditionlist) == 1 and args.checkp == 'empty' and args.tver == 'empty'):
        list_checkpointpath = [None]
        list_trainver = [None]
    elif(len(conditionlist) == 2 and args.checkp == 'empty' and args.tver == 'empty'):
        list_checkpointpath = [None, None]
        list_trainver = [None, None]
    # else:
        # assert False, "Condition list options have changed, please revise this script"

    if step == 0:
        verlist = ["train"]
    elif step == 1:
        verlist = ["evaluate", "analyze", "filter_and_plot", "create_labeled_video"]
    elif step == 2 and conditionlist[0] == "behavior":
        verlist = ["extract_outlier_frames"]
    elif step == 2 and conditionlist[0] == "wand":
        verlist = ["extract_outlier_frames_wand"]
    elif step == 4:
        verlist = ["retrain_after_refinement"]
    elif step == 5:
        verlist = ["analyze", "filter_and_plot", "create_labeled_video"]
    else:
        assert False, "Please select a correct step number"

    print("Executing steps:", verlist)

    analysis_suffix="allvideos"



    for ver in verlist:
        for condition, train_ver, checkpoint_path in zip(conditionlist, list_trainver, list_checkpointpath):
            dict_paths, base_paths = find_expt_config_paths(name, condition)
            pcflist = dict_paths.values()
            base_paths = base_paths.values()
            print(pcflist)
            print(base_paths)

            for path_config_file, bp in zip(pcflist, base_paths):
                try:
                    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FRAC)
                except:
                    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=FRAC)

                sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

                main(path_config_file, ver, train_ver, checkpoint_path, iters, base_path = bp, 
                    analysis_suffix=analysis_suffix)