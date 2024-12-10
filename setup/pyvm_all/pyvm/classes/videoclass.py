
from pythonlib.tools.expttools import findPath, extractStrFromFname
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pyvm.globals import BASEDIR
import pandas as pd


class Videos(object):
    """
    process videos
    """
    def __init__(self):
        self.Params = {}
        self.Metadats = {} # e.g., self.Metadats["num_frames"][index_vid]

    def _check_preprocess(self):
        """ sanity checks
        """

        # 1) Each video should have unique grouping (vidgrp, camnam, path)
        list_of_ids = [(D["camera_name"], D["video_group"], D["name"]) for D in self.DatVideos]
        assert len(set(list_of_ids)) == len(list_of_ids), "video files are not uniquely identified! give each more info: camnam or group"

    def input_video_files(self, pathlist, camera_name_list = None, 
        video_group_list = None, min_size_bytes = 5000, list_good_frames = None,
        good_frames_delete_if_none = False, do_extraction_frames = True, 
        trialnums_to_keep=None):
        """ each datapoint is a single video (given a path)
        also save metadat about each video
        INPUT:
        - pathlist, list of str, paths to video name.
        - camera_name_list, same len as pathlist, list of str, to identiy each camera
        - video_group_list, same len as pathlist, list of anything hashable to identify how video sshould be grouped
        e.g., if mult videos should be combined for calibration purposes.
        - list_good_frames, list of list of ints, num lists same length as pathlist.
        - good_frames_delete_if_none, if True, then only keeps videos which have good frames passed in.
        (e.g., useful for checkerboard)
        """
        import os

        if camera_name_list is not None:
            assert len(camera_name_list)==len(pathlist)
        else:
            camera_name_list = [None for _ in range(len(pathlist))]
    
        if video_group_list is not None:
            assert len(video_group_list)==len(pathlist)
        else:
            video_group_list = [None for _ in range(len(pathlist))]

        if list_good_frames is not None:
            assert len(list_good_frames)==len(pathlist)
        else:
            list_good_frames = [[] for _ in range(len(pathlist))]

        # Put into data structure
        self.DatVideos = []
        dict_good_frames = {}
        skipped_vids = {}
        for path, cam, grp, good_frames in zip(pathlist, camera_name_list, 
            video_group_list, list_good_frames):

            # Check filesize, if below threshold, then exclude.
            if min_size_bytes is not None:
                x = os.path.getsize(path)
                y = cv2.VideoCapture(path)
                if x<min_size_bytes:
                    # skip
                    print(f"SKIPPING {path}, since fsize {x} < {min_size_bytes}")
                    skipped_vids[(path.split('/')[-1],cam)] = 'too small'
                    continue
                if not y.isOpened():
                    # skip 
                    print(f"SKIPPING {path}, beacuse video file does not open (probably ungracefully truncated)")
                    skipped_vids[(path.split('/')[-1],cam)] = 'truncated'
                    continue 
                cv2.destroyAllWindows()
                y.release()

            # Each video should have unique index
            name = extractStrFromFname(path, None, None, return_entire_filename=True)
            index = (cam, grp, name)

            self.DatVideos.append(
                {"path_video": path,
                "camera_name":cam,
                "video_group":grp,
                "good_frames":[],
                "name": name,
                "index":index
                }
                )

            # Collect this, so can use helper code to input good frames.
            dict_good_frames[index] = good_frames

        # # Each video should have unique index
        # for D in self.DatVideos:
        #     D["index"] = (D["camera_name"], D["video_group"], D["name"])

        for D in self.DatVideos:
            D["path_base"] = self._paths(D["index"])["path_dir"]

        #Janky
        path_here = '/'.join(D["path_base"].split('/')[:-2])
        with open (f'{path_here}/skipped_vids.txt','w') as f:
            for k,v in skipped_vids.items():
                f.writelines(f"{k},{v}\n")

        # Checks
        self._check_preprocess()

        # If is campy, then also extract trial nums:
        self.compute_trialnums()
        
        # Remove trials you dont want
        if trialnums_to_keep is not None:
            self.DatVideos = [dat for dat in self.DatVideos if dat["trialnum0"] in trialnums_to_keep]

        # Extract all frames. skips if already done
        if do_extraction_frames:
            self.extract_all_frames_from_videos()
        
        # Save metadata, for each access later
        if not do_extraction_frames:
            # Then should extract num frames
            self.metadat_extract_write(kind="num_frames")

        # Input good frames
        if all([len(v)>0 for v in dict_good_frames.values()]):
            self.input_good_frames(dict_good_frames, good_frames_delete_if_none) 
        else:
            print("[input_video_files] skipping input_good_frames since you didnt give frames.")

        # make groups
        self.generate_group_level_data()

        print("Done adding data to self.DatVideos!")


    def load_video_files_helper(self, ver, params):
        """ helkper to find paths for files given common directory
        structure for diff kinds of data. outpute can be passed into 
        "input_video_files" to initialize.
        INPUT:
        - ver, what kind of expt this is.
        - params, dict, relevant items depends on ver.
        --- group_list, by default if dont include this, redundnat with camera name,
        or if you pass in group, then use that. Only need this if 
        """

        def find_videos(basedir, list_vidnums=None):
            """ extract list of paths to each vid
            PARAMS:
            - list_vidnums, only kleep these. None to get all.
            """

            print("SEARCHIG: ", basedir)
            pathlist = findPath(f"{basedir}/Camera{i+1}",
                     [],
                     "vid",
                     ".mp4") + findPath(f"{basedir}/Camera{i+1}",
                     [],
                     "vid",
                     ".avi")
            if len(pathlist)==0:
                # try using the camera name
                namethis = camera_names[i]
                pathlist = findPath(f"{basedir}/{namethis}",
                         [],
                         "vid",
                         ".mp4") +  \
                        findPath(f"{basedir}/{namethis}",
                         [],
                         "vid",
                         ".avi") + \
                         findPath(f"{basedir}/{namethis}",
                             [],
                             namethis,
                             ".avi")
            if len(pathlist)==0:
                print(basedir, namethis)
                assert False, "didnt find any vids.,"
            return pathlist

        # === INTERPRET PARAMS
        if params["condition"] == "checkerboard":
            # checkerboard, only useful if I have given good frames
            good_frames_delete_if_none = True
            good_frames_bycam = params["good_frames_checkerboard"]
            if len(good_frames_bycam)==0:
                good_frames_bycam = [[] for _ in range(params["numcams"])]
            elif len(good_frames_bycam)!=params["numcams"]:
                print(good_frames_bycam)
                print(params["numcams"])
                assert False, "probably make good_frames_checkerboard in yaml file to [[], [], [], []], if 4 cams."
        else:
            # dont enter good frames
            good_frames_delete_if_none = False
            good_frames_bycam = None

        if params["condition"] == "behavior":
            # too many beh files, and not using this anywa.
            do_extraction_frames = False
        else:
            do_extraction_frames = True

        # === RUN
        DAT = []
        if ver in ["campy", "spinview"]:
            for k in ["camera_names", "numcams", "basedir"]:
                assert k in params
            # camera_names = {
            #       0: "flea3",
            #       1: "blackfly"}
            # numcams = 2 
            numcams = params["numcams"]
            basedir = params["basedir"]
            camera_names = params["camera_names"] 

            assert numcams > 0, 'y no cams?'

            for i in range(numcams):

                # Passed in good frames for this camera?
                good_frames = []
                if good_frames_bycam is not None:
                    if len(good_frames_bycam[i])>0:
                        good_frames = good_frames_bycam[i]
                # Find all videos for this camera
                pathlist = find_videos(basedir)

                assert len(pathlist)>0, "didnt find video, check name"
                # Colelct each video
                for path in pathlist:
                    # Only keep if is the original video
                    if "downsampled" in path:
                        continue
                    if "DLC" in path:
                        continue
                    DAT.append(
                        {"path_video":path,
                         "good_frames":good_frames,
                         "id":camera_names[i],
                         "cam_num":i
                        })

        # elif ver=="dlc":
        #     assert False, "not using this anymre - instead they are either spinview or campuy"
        #     # Then is processed DLC videos
        #     numcams = params["numcams"]
        #     basedir = params["basedir"]
        #     camera_names = params["camera_names"]
        #     exptname = params["exptname"]
        #     iternum_list = params["iternum_list"]
        #     for i in range(numcams):
        #         camname = camera_names[i]
        #         assert False, "use find_videos from above"
        #         pathlist = findPath(f"{basedir}",
        #              [[f"{exptname}_{camname}"], ["videos"]],
        #              f"{iternum_list[i]}_labeled",
        #              ".mp4")
        #         if len(pathlist)!=1:
        #             print(pathlist)
        #             assert len(pathlist)==1, "were you trying to get multiple videos?"

        #         for path in pathlist:
        #             DAT.append(
        #                 {"path_video":path,
        #                  "good_frames":[],
        #                  "id":camname,
        #                  "cam_num":i
        #                 })


        else:
            print(ver)
            assert False, "not coded"
        pathlist = [D["path_video"] for D in DAT]
        camera_name_list = [D["id"] for D in DAT]
        list_good_frames = [D["good_frames"] for D in DAT]

        if "group_list" in params.keys():
            video_group_list = params["group_list"]
            if len(video_group_list)!=len(DAT):
                print(video_group_list)
                print(len(DAT))
                print(DAT)
                assert False
        else:
            video_group_list = [D["cam_num"] for D in DAT]

        # save to params
        self.Params["load_ver"] = ver
        self.Params["load_params"] = params
        self.Params["dlc_iternum"] = params["dlc_iternum"]
        trialnums_to_keep = params["vidnums"]

        # print(pathlist)
        # print(video_group_list)
        # print(list_good_frames)
        # print(good_frames_delete_if_none)
        # print(do_extraction_frames)
        # print(trialnums_to_keep)
        self.input_video_files(pathlist, camera_name_list, video_group_list,
            list_good_frames =list_good_frames, 
            good_frames_delete_if_none=good_frames_delete_if_none, 
            do_extraction_frames=do_extraction_frames, 
            trialnums_to_keep = trialnums_to_keep)

        # Pull out input parasm (which are list indexced by coniditon) to main part of params
        # condition = self.Params["load_params"]["condition"]
        # ind = self.Params["load_params"]["list_conditions"].index(condition)
        # list_names = ["bodyparts", "skeletons", "vidnums", "conditions"]
        # for name in list_names:
        #     assert name not in self.Params.keys()
        #     self.Params[name] = self.Params["load_params"][f"list_{name}"][ind]


    def load_data_wrapper(self, date, expt, animal, condition, session=None):
        """ High level wrapper to load expt.
        PARAMS:
        - expt, str, e.g, "cagetest2"
        - condition, str, {"behavior", "wand", "checkerboard"}, saying which set of videos
        to load, and associated params.
        NOTE:
        - must have entered this expt and condition in pyvm/metadata
        """
        from pyvm.utils.experiments import get_params
        if session == 1 or session is None:
            sess_print='Done' #I wanted to make it rhyme
            sess_print=''
        else:
            sess_print=f"_{session}"
        # 1) Load metadata
        p = get_params(f"{date}_{expt}{sess_print}", animal)

        # 2) convert params to videoclass-expected
        p["camera_names"] = {i:name for i, name in enumerate(p["list_camnames"])}
        del p["list_camnames"]
        p["numcams"] = len(p["camera_names"])
        p["basedir"] = f"{BASEDIR}/{animal}/{p['dirname']}/{condition}"
        p["exptname"] = expt
        p["condition"] = condition

        # Things that depend on condition.
        idx = p["list_conditions"].index(condition)
        assert isinstance(idx, int)
        p["dlc_iternum"] = p["list_dlciternum"][idx]
        list_names = ["bodyparts", "skeletons", "vidnums", "conditions"]
        for name in list_names:
            assert name not in p.keys()
            p[name] = p[f"list_{name}"][idx]

        if condition=="behavior":
            ver = "campy"
        elif condition=="wand":
            if expt in ["camtest5", "chunkbyshape4"]:
                ver = "spinview"
            elif expt=="cagetest2":
                ver = "campy"
            else:
                ver = "campy"
                # assert False, "enter this"
            # ver = "spinview"
        elif condition=="checkerboard":
            # This done using spinview
            ver =  "spinview"
        else:
            assert False


            
        # Load data
        self.load_video_files_helper(ver, p)

    def screen_frames(self,indtrial,list_part):
        """
        Function to take frames only that are near screen. Should only be used for 220914 cam setup,
        if other calibration hsould not be hard to alter function to work pretty easily
        INPUTS: 
        - None
        OUTPUTS: 
        - List of frames that are outside certain distance of screen 
        (I did in same half of the frame as the screen as judged by x/y coords from top left)
        """
        _, framenums, _ = self.goodframes_mapping_new_old_index(0)
        indtrial = self.inds_trials()[0]
        #Pass list to this function to extract a matrix of pts
        vals, cols = self.dlc_extract_pts_matrix(indtrial, list_part, framenums)

        dict_cams = self.get_cameras()
        cam_list = [c[1][0] for c in dict_cams.items()]

        bad_frames_dict_cam = {}
        #(0,0.5) means take max*0 for x thresh (i.e. all x's) and take max*.5 for y thresh (everything in lower half)
        cam_to_screen_half = {
        'bfs1': [0,0.75],
        'flea': [0.75,0],
        'fly1': [0,0.75],
        'fly2': [0,0.75],
        }
        df = pd.DataFrame(vals, columns=cols)
        
        xcols = [e for i,e in enumerate(cols) if i%2==0]
        ycols = [e for i,e in enumerate(cols) if i%2==1]
        ymaxs = [500,500,900,1100]
        xmaxs = [900,900,1000,1000]
        for cam, xcol, ycol, xmaxa, ymaxa in zip(cam_list,xcols,ycols,xmaxs,ymaxs):
            if cam == 'bfs2':
                continue
            assert (cam in xcol) and (cam in ycol), "Misaligned, check why"
            this_df = df.filter(regex=cam)
            this_df.loc[:,'frames'] = framenums
            xpart = cam_to_screen_half[cam][0]
            ypart = cam_to_screen_half[cam][1]
            xmax = max(this_df.iloc[:,0])
            ymax = max(this_df.iloc[:,1])

            # assert xmax<=xmaxa and ymax<=ymaxa, f'{xmax},{ymax}'

            xthresh = xmaxa * xpart
            ythresh = ymaxa * ypart


            bad_frames = this_df[(this_df[xcol]<xthresh) | (this_df[ycol]<ythresh)]
            print(bad_frames)
            bad_frames_dict_cam[cam] = list(bad_frames.loc[:,'frames'])
        return bad_frames_dict_cam




    def filter_good_frames_dan(self):
        """ Prune the current good frames, using various methods. e.g., if you extracted
    a lot of frames, can then remove those that fail criteria.
        PARAMS:
        - ver, str, which version.
        - screen, bool, restrict pts to those close to screen
        NOTE:
        At end triggers function that removes bad frames from extracted frames per camera, at frame/data extractuion time will do checks for multi camera
        sharing (at least 2 cams must share frames for easywand to accept them)
        """
        ncams = self.num_cams()
        dict_cams = self.get_cameras()
        cam_list = [c[1][0] for c in dict_cams.items()]
        # indtrial = self.Dat
        # ind_group = 0

        # Sanity checks
        # 1) One video per cam
        assert self.num_cams()==len(self.DatVideos), f"{self.num_cams()} == {len(self.DatVideos)}"
        assert len(self.DatGroups)==len(self.DatVideos)
        assert len(self.inds_trials())==1
        # 2) Each cam has exact same good frames
        list_good_frames = [dat["good_frames"] for dat in self.DatVideos]
        for list1, list2 in zip(list_good_frames[:-1], list_good_frames[1:]):
            assert list1 == list2

        THRESH = 0.95
        list_part, list_feat = self.dlc_get_list_parts_feats()
        bad_frames = {}
        for indtrial in self.inds_trials():
            for cam,camname in zip(range(ncams),cam_list):
                bad_frames[camname] = []
                datv = self.wrapper_extract_dat_video(None, cam, indtrial)
                goodframes = datv["good_frames"]
                print("Good frames: (for this cam):", datv["index"])
                print(goodframes)
                for part in list_part:
                    # LIKELI
                    likeli = self.dlc_data_part_feat(datv["index"], part, "likelihood").values
                    for frame in goodframes:
                        if likeli[frame]<THRESH:
                            bad_frames[camname].append(frame)
                            
                    # NOTE OUT OF BOUNDS 
                    out_of_bounds = self.dlc_data_part_feat(datv["index"], part, "out_of_bounds").values
                    frames_oob = np.where(out_of_bounds==True)[0].tolist()
                    frames_oob = [f for f in frames_oob if f in goodframes] # only keep frames that are in good frames list.
                    bad_frames[camname].extend(frames_oob)
            if False:
                for cam in cam_list:
                    unscreened_frames = self.screen_frames(indtrial,list_part)
                    # print(unscreened_frames[camname])
                    # assert False
                    # print('1',bad_frames[camname])
                    bad_frames[camname].extend(unscreened_frames[camname])
                    # print('2',bad_frames[camname])
                    # assert False
            print(f"BAD FRAMES, trial {indtrial}, across all cams (original frame idx:")
            print(bad_frames)


        # Remove frames bad for each cam

        print("REMOVING these frames")
        self.remove_badframes_from_goodframes_all_dan(bad_frames)

    def filter_good_frames(self, ver):
        assert False, "Outdated function"
        """ Prune the current good frames, using various methods. e.g., if you extracted
        a lot of frames, can then remove those that fail criteria.
        PARAMS:
        - ver, str, which version.
        NOTE:
        """

        ncams = self.num_cams()
        dict_cams = self.get_cameras()

        if ver=="dlc_likeli_wand":
            """ only keep frames where dlc likeli is high for all parts and all cams. 
            Assumes only one video per cam, and all share same good frames 
            Use this after extracting a lot of shared frames."""

            # Sanity checks
            # 1) One video per cam
            assert self.num_cams()==len(self.DatVideos), f"{self.num_cams()} == {len(self.DatVideos)}"
            assert len(self.DatGroups)==len(self.DatVideos)
            assert len(self.inds_trials())==1
            # 2) Each cam has exact same good frames
            list_good_frames = [dat["good_frames"] for dat in self.DatVideos]
            for list1, list2 in zip(list_good_frames[:-1], list_good_frames[1:]):
                assert list1 == list2

            THRESH = 0.99
            list_part, list_feat = self.dlc_get_list_parts_feats()


            for indtrial in self.inds_trials():

                # Collect all bad frames across all cams for this trial.
                bad_frames = []
                for cam in range(ncams):
                    datv = self.wrapper_extract_dat_video(None, cam, indtrial)
                    goodframes = datv["good_frames"]
                    print("GOod freames: (for this cam):", datv["index"])
                    print(goodframes)
                    for part in list_part:
                        
                        # LIKELI
                        likeli = self.dlc_data_part_feat(datv["index"], part, "likelihood").values
                        for frame in goodframes:
                            if likeli[frame]<THRESH:
                                bad_frames.append(frame)
                                
                        # NOTE OUT OF BOUNDS 
                        out_of_bounds = self.dlc_data_part_feat(datv["index"], part, "out_of_bounds").values
                        frames_oob = np.where(out_of_bounds==True)[0].tolist()
                        frames_oob = [f for f in frames_oob if f in goodframes] # only keep frames that are in good frames list.
                        bad_frames.extend(frames_oob)
                print(f"BAD FRAMES, trial {indtrial}, across all cams (original frame idx:")
                print(sorted(set(bad_frames)))

            # Remove any frame that is bad in at least one camera
            print("REMOVING these frames")
            self.remove_badframes_from_goodframes_all(bad_frames, "old_index")
            

            assert False

            # bad_frames = sorted(set(bad_frames))
            # goodframes_updated = {}
            # for cam in range(ncams):
            #     datv = V.datgroup_extract_single_video_data(cam, indtrial, True)
            #     goodframes = datv["good_frames"]
            #     goodframes_updated[datv["index"]] = [f for f in goodframes if f not in bad_frames]
                
            #     print("-- Camera", datv["camera_name"])
            #     print(bad_frames)


    def sample_and_extract_auto_good_frames(self, ntoget=200, skip_if_inputed =True,
        also_extract_dlc=True):
        """ Instead of manually entering good frames (input_good_frames), autoamtically smaple
        uniformly over time. Can then prune to those that pass certain criteria (other code)
        Runs this separately for each video.
        PARAMS:
        - ntoget, how many frames.
        - skip_if_inputed, bool, then skips if finds that already inputed
        - also_extract_dlc, then extracts frames from DLC labeled video as well. Do
        V.import_dlc_data() first.
        RETURNS:
        - each dat video will have inputed good frames
        - extracts frames to new directory.
        NOTE:
        - will not overwrite any inputted good frames. But will always reextract frames, even those
        that did not overwrite inputted good frames. 
        """
        import numpy as np

        # 1. check if already have good frames. if so, then by defalt dont run
        # Get good frames for each video.
        good_frames = {}
        for dat in self.DatVideos:
            ind_video = dat["index"]
            nframes = self.num_frames(ind_video)
            list_framenums = np.linspace(0, nframes-1, ntoget)
            list_framenums = sorted(set([int(x) for x in list_framenums]))
                
            inputed = self.check_goodframes_gotten(ind_video)
            if inputed:
                if skip_if_inputed:
                    print("Skipping inputting for: ", ind_video)
                    continue

            # 1) Save the good frames
            good_frames[dat["index"]] = list_framenums


        # inds_trials = self.inds_trials()
        # ncams = self.num_cams()
        # for trial in inds_trials:
        #     ## Extract good frames automatically
        #     # (currnetly uniform sampling over frames)

        #     # Call these good_frames
        #     for indcam in range(ncams):

        #         nframes = self.num_frames2(indcam, trial) 
        #         list_framenums = np.linspace(0, nframes-1, ntoget)
        #         list_framenums = sorted(set([int(x) for x in list_framenums]))
                
        #         # dat = self.datgroup_extract_single_video_data(indcam, trial)
        #         indvid = self.helper_index_good((indcam, indtrial))["index"]
        #         # indvid = self.get_vidindex_from_cam_trial(indcam, trial) 
        #         inputed = self.check_goodframes_gotten(indvid)

        #         if inputed:
        #             if skip_if_inputed:
        #                 print("Skipping inputting for: ", indcam)
        #                 continue

        #         # 1) Save the good frames
        #         dat = self.wrapper_extract_dat_video(None, indcam, trial)
        #         good_frames[dat["index"]] = list_framenums

        # 1) Input good frames
        self.input_good_frames(good_frames, True)

        # 2) Extract frames
        # Do this once for each camera
        self.collect_goodframes_from_videos()

        # Collect frames with dlc labels
        if also_extract_dlc:
            self.collect_goodframes_from_videos(vid_kind="dlc_labeled")


    def check_goodframes_gotten(self, index_vid):
        """ Check if (1) good frames inputed and (2) if extracted new directory
        PARAMS:
        - index_vid, for a single video
        RETURNS:
        - inputed, bool, whether in dat["good_frames"] is list of ints
        """

        dat = self.wrapper_extract_dat_video(index_vid)
        inputed = len(dat["good_frames"])>0
        return inputed


    def input_good_frames_dan(self, good_frames, delete_videos_without_good_frames = False):
        """
        Pass in dict of good frames for each cam. Will assign good frames to the data structure for the videos.
        INPUT:
        - good frames, dict, with camname as keys and good frames as values (one for each camera)
        NOTE:
        Updated version o the below function, if you want to use this behavior uncomment the assert False
        """
        #Should not be a problem with the new data structure but this will ensure unique sorted good frames
        for D in self.DatVideos:
            for k, v in good_frames.items():
                good_frames[k] = sorted(set(v))
        # print(good_frames)
        # assert False
        for D in self.DatVideos:
            key_camname = D["index"][0]
            if key_camname in good_frames and len(good_frames[key_camname]) > 0:
                D["good_frames"] = good_frames[key_camname]
                print(f"good_frames assigned to video {key_camname}")
            else:
                print(f"No good frames assigned to video {key_camname} since not found in good frames, or entry was emoty (maybe cam was removed?)")
                D["good_frames"] = []

        if delete_videos_without_good_frames:
            print("DELETED these vides froms self.DatVideos since no good frames found:")
            print([D["name"] for D in self.DatVideos if len(D["good_frames"])<0])
            self.DatVideos = [D for D in self.DatVideos if len(D["good_frames"])>0]

        self.generate_group_level_data()
    def input_good_frames(self, good_frames, delete_videos_without_good_frames=False):
        # FUNCTIOPN OUTDATED FOR FILTERIMNG STEP< SEE ABOVE
        """ pass in frame numbers that are "good"
        Will assign into self.DatVideos[i]["good_frames"], a list of 
        ints for each video.
        INPUT:
        - good_frames, dict, with keys (camname, vidgrp, name), which indexes each
        unique vid.
        - delete_videos_without_good_frames, then removes from self.DatVideos
        NOTE:
        also automatically cleans up self.DatGroups
        - alternatively, pass in good frames initially in input_video_files
        """

        for v in good_frames.values():
            if len(v)==0:
                print(good_frames)
                assert False, "cannot run if you dont give me the frames"

        # First, only take unique frames
        # Also sorts.
        for k, v in good_frames.items():
            good_frames[k] = sorted(set(v))

        # input into dat
        for D in self.DatVideos:
        #     (D["cam_num"])
            key = D["index"]
            if key in good_frames:
                D["good_frames"] = good_frames[key]
                print(f"Good! good_frames assigned to video {key}")
            else:
                print(f"No good frames assigned to video {key} since not found in good_frames")
                D["good_frames"] = []

                    
        if delete_videos_without_good_frames:
            print("DELETED these vides froms self.DatVideos since no good frames found:")
            print([D["name"] for D in self.DatVideos if len(D["good_frames"])<0])
            self.DatVideos = [D for D in self.DatVideos if len(D["good_frames"])>0]

        # reconstruct datgroup based on 
        self.generate_group_level_data()



    ################# GROUP VIDEOS INTO GROUP-LEVEL DATASET
    def generate_group_level_data(self):
        """ group videos based on the first two levels (cameraname, videogroup)
        RETURNS:
        - makes self.DatGroups
        NOTE:
        - also does saniyt check to make sure:
        --- num frames match across cameras for each trial

        """
        from pythonlib.tools.expttools import get_common_path

        DAT = self.DatVideos
        self.DatGroups = []

        indexthis_list = set([(D["camera_name"], D["video_group"]) for D in DAT])
        
        print("Made self.DatGroups!, with len:")
        for indexthis in indexthis_list:
            out = {}
            DATTHIS = [D for D in DAT if (D["camera_name"], D["video_group"])==indexthis]

            # get common path
            path_shared = get_common_path([D["path_video"] for D in DATTHIS])
            out["path_shared"] = path_shared

            pvlist = [D["path_video"] for D in DATTHIS]
            gframelist = [D["good_frames"] for D in DATTHIS]

            out["videos_path_list"] = pvlist
            out["videos_goodframes_list"] = gframelist
            out["videos_index_list"] = [D["index"] for D in DATTHIS]

            out["index_grp"] = indexthis
            
            self.DatGroups.append(out)
            print("***", out["index_grp"])

        # If is campy, then also extract trial nums:
        self.compute_trialnums()
        
        # print("This many groups: ", len(self.DatGroups))

        # Consolidate
        if False:
            self.datgroup_consolidate_videos()


        ### SANITY CHECKS
        # chekc that num frames match (actual videos)
        # a version that doesnt require campy (just check num frames)
        if self.Params["load_params"]["condition"] in ["behavior", "wand"]:
            inds_trials = self.inds_trials()
            ncams = len(self.DatGroups)
            for trial in inds_trials:
                x = []
                # x = [self.num_frames2(indcam, trial) for indcam in range(ncams)]
                for indcam in range(ncams):
                    try:
                        x.append(self.num_frames2(indcam,trial))
                    except AssertionError:
                        x.append(-1)

                if len(set(x))>1:
                    skip = True
                else:
                    skip = False

                for indcam in range(ncams):
                    try:
                        datv = self.wrapper_extract_dat_video(None, indcam, trial)
                    except AssertionError:
                        continue
                    # datv = self.datgroup_extract_single_video_data(indcam, trial, True)
                    datv["SKIP"] = skip



    def clean_videos(self):
        """ Removes videos (from self, not the file) if marked as SKIP==True, based on 
        sanity checks (both general checks [see generate_group_level_data] and campy-checks 
        [see that code])
        DOES:
        - if framenums are not similar across cams
        - if explictly tell me ot remove videos (campy)
        RETURN:
        - self, but with some of self.DatVideos removed
        """

        ## Throw out any videos that are not well-aligned across cameras..
        # Kind of hack, but fine if this is not a common issue.
        X = []
        print("REMOVING VIDEOS with SKIP=True. If doenst print, then means all videos good.")
        for dat in self.DatVideos:
            if dat["SKIP"]==False:
                X.append(dat)
            else:
                print("REMOVING VIDEO: ", dat["index"])
        self.DatVideos = X

        # Remove videos based on epxlicit input.
        # condition = self.Params["load_params"]["condition"]
        trials_to_keep = self.Params["load_params"]["vidnums"]
        X = []
        if trials_to_keep is not None:
            # Then remove these trials
            for dat in self.DatVideos:
                if dat["trialnum0"] in trials_to_keep:
                    X.append(dat)
                else:
                    print("REMOVING VIDEO: ", dat["index"], "(not in trialstokeep)")
            self.DatVideos = X

        # Regroup
        self.generate_group_level_data()


    def campy_export_to_ml2(self):
        """ export campy data so can load in ml2 (Export into pandas)
        """
        import pandas as pd
        import os
        from pythonlib.tools.expttools import writeDictToYaml, makeTimeStamp

        out = []

        # for datg in self.DatGroups:
        #     print(datg["index_grp"])
        #     datv = self.helper_index_good(datg["videos_index_list"][0])
        #     print(datv["index_grp"])
        #     assert False
        #     for datv in datg["Dat"]:
                

        #         print(datv.keys())
        #         print(datg["index_grp"])
        #         print(self.helper_index_good(datv["index"]).keys())
        #         assert False
        #         this = {}
        #         for k, v in datv.items():
        #             this[k] = v
                    
        #         this["index_grp"] = datg["index_grp"]
            
        #         out.append(this)
        
        df = pd.DataFrame(self.DatVideos)
            
        # save
        sdir = f"{self.Params['load_params']['basedir']}/extracted_campy_data"
        os.makedirs(sdir, exist_ok=True)
        df.to_pickle(f"{sdir}/dat.pkl")

        self.Params["tstamp"] = makeTimeStamp()
        writeDictToYaml(self.Params, f"{sdir}/params.yaml")

        print("EXTRACTED campy data to:")
        print(sdir)


    ################## VIDEO PROVESSING
    def extract_list_frames_good(self, idx_vid, indframes, vidkind="orig", suffix=None,
        overwrite=True):
        """
        [GOOD] flexible extraction of frames.
        IN:
        - idx_vid, index to video
        - indframes, 0, 1, 2.., list of frames to extract. None, to get all.
        - vidkind, which video to use. 
        --- "orig", the original
        --- "dlc", the dlc, usually downsampled.
        --- "dlc_labeled", labeled with marker predictions
        - suffix, to add to directory, to call this extraction.
        """
        from ..utils.preprocess import collectFrames

        # Find path to video
        paths = self.get_paths_good(idx_vid)
        if vidkind=="orig":
            p = paths["path_vid_full"]
        elif vidkind =="dlc":
            assert False, "add this to getpathsgood"
        elif vidkind=="dlc_labeled":
            p = paths["vid_labeled_predictions"]
        else:
            print(vidkind)
            assert False

        # Make a new directory for these frames
        path_base = paths["path_dir"]
        newdir = f"{path_base}/extracted_frames"
        if suffix is not None:
            newdir += f"-{suffix}"

        # Overwrite if exist.
        if overwrite:
            if os.path.isdir(newdir):
                import shutil
                shutil.rmtree(newdir)
            os.makedirs(newdir, exist_ok=True)

        gframelist = indframes

        collectFrames([p], [gframelist], newdir)


    def extract_all_frames_from_videos(self, ver="orig", skip_if_already_extracted=True):
        """ extract all frames for all videos 
        to subdirectory
        INPUT:
        - ver, which video?
        """
        import os
        from ..utils.preprocess import extractFrames
        if ver=="orig":
            # Use the old code
            for D in self.DatVideos:
                path_frames = self.get_paths_good(D["index"], include_dlc=False)["path_frames"]
                if os.path.exists(path_frames):
                    if skip_if_already_extracted:
                        continue
                path_video = D["path_video"]
                extractFrames(path_video)       
        elif ver=="dlc":
            assert False, "extract this path"
        elif ver=="dlc_labeled":
            # EXTRACT all frames for any bvide.
            for D in self.DatVideos:
                path_video = self.get_paths_good(D["index"])["vid_labeled_predictions"]
                extractFrames(path_video)

    def metadat_get_path(self, kind):
        """
        Get path for metadata dict, previously extracted
        PARAMS;
        - kind, string, like num_frames, each a different file.
        - index_vid, index into vid.
        RETURNS:
        - path string
        - path exists, bool
        """
        D = self.wrapper_extract_dat_video(index_vid = 0)
        paths = self.get_paths_good(D["index"], include_dlc=False)

        # make the metadat dir 
        if not os.path.exists(paths['metadata_dir']):
            os.makedirs(paths['metadata_dir'])
            print("Made metadat dir: ", paths['metadata_dir'])

        paththis = f"{paths['metadata_dir']}/{kind}"
        return paththis, os.path.exists(paththis)


    def metadat_extract_write(self, kind):
        """ extract and save txt file of metadat
        for easy loading later, does this for all videos.
        If previously gotten, then generally will just load that
        and then resave it.
        PARAMS:
        - kind, string for what to extreacat, e.g, "num_frames"
        RETURNS:
        - outdict, list of dicts, index_vid:value
        -- Also saves into e.g., <'/data2/camera/220317_chunkbyshape4/behavior>/metadata 
        """
        from pythonlib.tools.expttools import writeDictToYaml
        
        paththis, _ = self.metadat_get_path(kind)

        print("Extracting metadata for ", kind, " Might take some minutes ... ")

        # Collect this metadat
        outdict = {}
        if kind=="num_frames":
            for D in self.DatVideos:
                nframes = self.num_frames(D["index"])
                outdict[D["index"]] = nframes

        # Save it.
        writeDictToYaml(outdict, paththis)
        # for k,v in outdict.items():
        #     if 'bfs1' in k and 'vid-t121' in k:
        #         print(k,v)
        return outdict

    def metadat_read(self, kind, index_vid, do_reload=False):
        """ Read out metadat for this video
        PARAMS:
        - kind, string 
        - index_vid, tuple index, like ('flea', 0, 'vid-t10')
        RETURNS:
        - either:
        --- val, metadata value
        --- None, if cannot find preextracted metadata.
        NOTE: the first time you run this, will load and cache..
        """
        from pythonlib.tools.expttools import load_yaml_config

        # Extract and cache?
        if kind not in self.Metadats.keys() or do_reload:
            # Then load
            paththis, pathexists = self.metadat_get_path(kind)
            if pathexists:
                metadat = load_yaml_config(paththis)
                self.Metadats[kind] = metadat
            
        # Look for this preextracted data
        if kind in self.Metadats.keys():
            if index_vid in self.Metadats[kind].keys():
                # Then previousl extracted
                return self.Metadats[kind][index_vid]

        # Not yet extracted.
        return None


    def get_paths_good_grp(self, indgrp):
        """ dirctories at group level
        the directory this can
        aggreg across videos and poetnailyl cams, so go by group.
        """ 

        datgrp = self.DatGroups[indgrp]

        paths = {
            "path_shared": datgrp["path_shared"],
            "videos_path_list": datgrp["videos_path_list"],
            "collected_frames": f"{datgrp['path_shared']}/collected_frames"
        }
        return paths

    def goodframes_mapping_new_old_index(self, ind_grp):
        """ for this group, what is mapping for collected frames between
        new and old indices. 
        PARAMS:
        - ind_grp, int, 0,1,2,...
        RETURNS:
        - mapping_old_new, list of tuples, len collected frames, each tuple is (old, new)
        - list_goodframes_oldnum, list of ints
        - list_goodframes_newnum, list of inds
        """
        import yaml
        # The current good frames, in original indinces
        path_to_collected_frames = self.get_paths_good_grp(ind_grp)["collected_frames"]
        fthis = f"{path_to_collected_frames}/framenums_old_new.yaml"
        with open(fthis) as file:
            mapping_old_new = yaml.load(file, Loader=yaml.FullLoader)
        list_goodframes_newnum = [x[1] for x in mapping_old_new]
        list_goodframes_oldnum = [x[0] for x in mapping_old_new]
        return mapping_old_new, list_goodframes_oldnum, list_goodframes_newnum
    
    def goodframes_mapping_new_old_index_dan(self, ind_grp):
        """ for this group, what is mapping for collected frames between
        new and old indices. 
        PARAMS:
        - ind_grp, int, 0,1,2,...
        RETURNS:
        - mapping_old_new, list of tuples, len collected frames, each tuple is (old, new)
        - list_goodframes_oldnum, list of ints
        - list_goodframes_newnum, list of inds
        """
        import yaml
        dict_cams = self.get_cameras()
        cam_list = [c[1][0] for c in dict_cams.items()]
        # The current good frames, in original indinces
        path_to_collected_frames_cam = self.get_paths_good_grp(ind_grp)["collected_frames"]
        path_to_collected_frames = '/'.join(path_to_collected_frames_cam.split('/')[:-2])
        dict_goodframes_oldnum = {}
        dict_goodframes_newnum = {}
        for cam in cam_list:
            fthis = f"{path_to_collected_frames}/{cam}/collected_frames/framenums_old_new.yaml"
            with open(fthis) as file:
                mapping_old_new = yaml.load(file, Loader=yaml.FullLoader)
            dict_goodframes_newnum = [x[1] for x in mapping_old_new]
            dict_goodframes_oldnum[cam] = [x[0] for x in mapping_old_new]
        return mapping_old_new, dict_goodframes_oldnum, dict_goodframes_newnum


    def remove_badframes_from_goodframes_all_dan(self, bad_frames):
        ind_vid = 0
        ind_group = 0
        dict_cams = self.get_cameras()
        cam_list = [c[1][0] for c in dict_cams.items()]
        mapping_old_new, list_goodframes_oldnum, list_goodframes_newnum \
            = self.goodframes_mapping_new_old_index(ind_group)
        good_frames_dict = {}
        for cam in cam_list:
            good_frames_dict[cam] = [frame for frame in list_goodframes_oldnum \
                                     if frame not in bad_frames[cam]]

        #input good frames into datstruct and reoganize the data
        self.input_good_frames_dan(good_frames_dict, True)

        #Extract good frames from videos, this will extract all good frames from each camera, regardless of being shared
        self.collect_goodframes_from_videos()        
        self.collect_goodframes_from_videos(vid_kind="dlc_labeled")


    def remove_badframes_from_goodframes_all(self, frame_nums, ver, do_extraction=True):
        """ Remove list of frames from good frames.
        PARAMS:
        - frame_nums, list of ints, meaning depends on ver
        --- if ver=="old_index", then is the original frame nums in vid.
        --- if ver=="new_index", then is the current index in collected frames
        --- if None, then looks for it in metadat, i.e, self.Params["load_params"]["bad_frames_wand
        - do_extraction, then will re-extract to directories.
        NOTE: 
        - need to have already assigned good frames. 
        - Currently removes from all videos and cameras - assumes their frames are shared, e.g.,
        in wand.
        - asserts that good frames exist.
        """
        import yaml

        # hacky, assumes only one video...
        ind_group = 0
        ind_vid = 0 

        if frame_nums is None:
            # Then use whatever is in metadat.
            frame_nums = sorted(self.Params["load_params"]["bad_frames_wand"])
        else:
            frame_nums = sorted(set(frame_nums))

        mapping_old_new, list_goodframes_oldnum, list_goodframes_newnum \
            = self.goodframes_mapping_new_old_index(ind_group)

        # --- Remove frames
        if ver=="new_index":
            
            # MApping between new and old index.
            # The vcurrent good frames.
            list_goodframes_oldnum1 = \
                self.DatGroups[ind_group]["videos_goodframes_list"][ind_vid]

            # The current good frames, in original indinces
            mapping_old_new, list_goodframes_oldnum1, list_goodframes_newnum \
                = self.goodframes_mapping_new_old_index(ind_group)
            if list_goodframes_oldnum != list_goodframes_oldnum1:
                print(list_goodframes_oldnum1)
                print(list_goodframes_oldnum)
                assert False
            
            # Frames to exclude
            list_badframes_newnum = frame_nums

            # get list of oldnums, given the list of newnums (bad ones)
            list_badframes_oldnum = []
            for numnew in list_badframes_newnum:
                tmp = [x[0] for x in mapping_old_new if x[1]==numnew]
                assert len(tmp)==1, "did not find this newnum"
                list_badframes_oldnum.append(tmp[0])

        elif ver=="old_index":
            # Nothing to do.
            list_badframes_oldnum = frame_nums
            
        else:
            assert False, "not coded"

        # assert that all bad frames you gave me are actually previusly part of good frema
        # otherwise could be sign somethign is wrong.
        assert all([x in list_goodframes_oldnum for x in list_badframes_oldnum]), "likely didnt restrict to good frames in ppreceding analyss."

        # return a new list of frames, excluding the bad ones
        list_framenums_excludingbad = [x for x in list_goodframes_oldnum 
            if x not in list_badframes_oldnum]


        # FINALLY reget all frames
        goodframes_excludingbad = {dat["index"]:list_framenums_excludingbad for dat in self.DatVideos}
        
        # Update goodframes
        for k, v in goodframes_excludingbad.items():
            assert len(v)>0, "at least one cam video without good frames..."
        self.input_good_frames(goodframes_excludingbad, True)

        if do_extraction:
            self.collect_goodframes_from_videos()        
            self.collect_goodframes_from_videos(vid_kind="dlc_labeled")




    def collect_goodframes_from_videos(self, overwrite=True, vid_kind="orig"):
        """ copy good frames across vidoes
         into a single directory, renaming as frame1, 2 ...
         - Flexibly for deciding which videos will hav their frames combined into one
         directory. DEFAULT: currnetly is to group by (cameraname, videogroup), so all vids
         with same values for these will be grouped. 
         - Finds most specific common dir across videos, for saving.
         NOTE:
         - overwrites by deleting if find old dir with old collected frames.
         """

        # if False:
        #     # [separate for each bideo] 
        #     pvlist = [D["path_video"] for D in DAT]
        #     gframelist = [D["good_frames"] for D in DAT]
        #     path_to=None
        #     collectFrames(pvlist, gframelist, path_to)

        # # == [combine multiple videso] copy good frames across vidoes into a single directory, renaming as frame1, 2 ...
        # # Do this once for each camera
        # DAT = self.DatVideos
        
        # indexthis_list = set([(D["camera_name"], D["video_group"]) for D in DAT])
        # path_to_collected_frames = {}

        # for indexthis in indexthis_list:
        #     DATTHIS = [D for D in DAT if (D["camera_name"], D["video_group"])==indexthis]

        #     # get common path
        #     path_shared = os.path.commonpath([D["path_video"] for D in DATTHIS])
            
        #     pvlist = [D["path_video"] for D in DATTHIS]
        #     gframelist = [D["good_frames"] for D in DATTHIS]
        #     print(pvlist)
        #     print(gframelist)

        #     path_to = f"{path_shared}/collected_frames"
        #     print(path_to)
        # #     path_to = None
        #     collectFrames(pvlist, gframelist, path_to)
            
        #     # For each camera, note down path to its frames
        #     path_to_collected_frames[cam_num] = path_to

        from ..utils.preprocess import collectFrames
        if vid_kind=="orig":
            # origianl version
            for DAT in self.DatGroups:
                path_shared = DAT["path_shared"]
                pvlist = DAT["videos_path_list"]
                gframelist = DAT["videos_goodframes_list"]
                        
                path_to = f"{path_shared}/collected_frames"
                DAT["path_to_collected_frames"] = path_to
                if os.path.isdir(path_to):
                    if overwrite:
                        import shutil
                        shutil.rmtree(path_to)
                os.makedirs(path_to, exist_ok=True)
                collectFrames(pvlist, gframelist, path_to)
        else:
            # NEW version, flexibly extract any video, any frames
            for datv in self.DatVideos:
                idx = datv["index"]
                goodframes = datv["good_frames"]
                if len(goodframes)==0:
                    print(idx)
                    assert False
                self.extract_list_frames_good(idx, goodframes, vidkind=vid_kind, suffix=f"collected-{vid_kind}")            



    ################### UTILS [operates on dataset (datgroup)]
    def get_video_sizes(self, patternSize=(9,6)):
        """ hacky"""
        from .calibrate import get_checkerboards
        imsize_list = []
        for DAT in self.DatGroups:
            # figure out savedir
            SDIR = DAT["path_to_collected_frames"]

            # get pathlist
            pathlist_frames = findPath(SDIR, [], "", ".jpg")

            ### Find corner pts (world and camera) and reporject and save fig
            successes, objpts, imgpts, imsize, figlist = get_checkerboards(pathlist_frames[:2], patternSize=patternSize)
            imsize_list.append(imsize)
        return imsize_list


    def compute_trialnums(self, fname_base = "vid"):
        """
        - for each video, get its trial num (0, 1, ...) in chron order, using
        the names of files. 
        - Does separtely for each DatGroup. For campy, this is same as doing for each camera.
        RETURNS:
        - new key in self.DatGroups --> videos_trialnum0
        NOTE: currently only works with campy datasets
        """

        if self.Params["load_ver"]=="campy":
            # Then know the crhon order based on filename

            def _get_trial_num(vname):
                """
                e.g., vidname = "vid-t0" or "vid-t22"
                returns 0, or 22
                """
            #     vname = "vid-t0.mp4"
                ind1 = vname.find(f"{fname_base}-")
                # ind2 = vname.find(".mp4")

                # check if there is a suffix.
                ind2 = vname.find(f"-downsampled")
                if ind2==-1:
                    # then this is not the suffix..
                    ind2 = len(vname)
                    assert ind2>ind1

                indtrial0 = int(vname[ind1+5:ind2])
                return indtrial0

            for datvid in self.DatVideos:
                indtrial = _get_trial_num(datvid["index"][2])
                datvid["trialnum0"] = indtrial

            # for indgrp in range(len(self.DatGroups)):
            #     videos_trialnum0 = []
            #     for idx in self.DatGroups[indgrp]["videos_index_list"]:
            #         indtrial = _get_trial_num(idx[2])
            #         videos_trialnum0.append(indtrial)
                    
            #     self.DatGroups[indgrp]["videos_trialnum0"] = videos_trialnum0
        else:
            # arbitray, give it trialnum based on order of video (wihtin each group)
            # NOTE: This might or might not match up videos correctly.
            if True:
                # Just name all trials 0.
                for datvid in self.DatVideos:
                    # indtrial = _get_trial_num(datvid["index"][2])
                    # print(datvid, indtrial)
                    # assert False
                    datvid["trialnum0"] = 0
            else:
                # Old version. doesnt work anymore since DatGroups not defined until after trialnums assigned..
                for datg in self.DatGroups:
                    for i, idx in enumerate(datg["videos_index_list"]):
                        datv = self.get_video_from_index(idx)
                        datv["trialnum0"] = i
                    
        

    def datgroup_consolidate_videos(self):
        """ instead of dict of lists, convert to list of dicts (each dict one video)
        OUT:
        self.DatGroups[ind]["Dat"], is list of dicts.
        """

        assert False, "dont use this. instaed, directly extract datv using self.helper_index_good"
        for Dat in self.DatGroups:

            # iterate over all vids
            out = []
            for ind in range(len(Dat["videos_path_list"])):
                # Things that need to get from DatVideos
                idx =  Dat["videos_index_list"][ind]
                datvid = self.wrapper_extract_dat_video(idx)
                trialnum = datvid["trialnum0"]
                if "dat_campy" in datvid.keys():
                    campy_framenums = datvid["dat_campy"]["framenums_session"]
                    campy_frametimes = datvid["dat_campy"]["frametimes_sincefirstframe"]
                else:
                    campy_framenums, campy_frametimes = None, None
                path_video = datvid["path_video"]

                # Things that are in DatGroup (generated by generate_group_level_data)
                out.append({
                    "path_video": path_video,
                    "index": idx,
                    "campy_framenums":campy_framenums,
                    "campy_frametimes":campy_frametimes,
                    "trialnum":trialnum
                    })
                Dat["Dat"] = out


    #################### UTILS [operates on videos, i..e, DatVideo]
    def _paths(self, ind_video):
        """ Helper to get dict of useful paths for this video
        OUT:
        - a dict holding paths.
        """
        from ..utils.preprocess import getPaths
        datvid = self.helper_index_good(ind_video)
        # datvid = self._datvideo_from_index(ind_video)
        return getPaths(datvid["path_video"])

    def get_paths_good(self, idx_vid, include_dlc=True):
        """ [GOOD], getter of all paths related to this video.
        - REPLACES _paths
        """
        animal = self.Params["load_params"]["animal"]

        pathdict = {}

        # Origianl video
        x = self._paths(idx_vid)
        for k,v in x.items():
            pathdict[k] = v
        pathdict["path_vid_full"] = pathdict["path_dir"] + "/" + pathdict["path_vid"]
        animal = self.Params["load_params"]["animal"]

        for condition in ["wand", "checkerboard", "behavior"]:
            pathdict[f"base_{condition}"] = f"{BASEDIR}/{animal}/{self.Params['load_params']['dirname']}/{condition}"

        # Base path
        # e.g,, '/data2/camera/220317_chunkbyshape4/behavior
        if self.Params["load_params"]["condition"]=="behavior":
            pathdict["path_base"] = pathdict["base_behavior"]
        elif self.Params["load_params"]["condition"]=="wand":
            pathdict["path_base"] = pathdict["base_wand"]
        elif self.Params["load_params"]["condition"]=="checkerboard":
            pathdict["path_base"] = pathdict["base_checkerboard"]
        else:
            print(pathdict)
            print(self.Params)
            assert False

        pathdict["metadata_dir"] = f"{pathdict['path_base']}/metadata"
        os.makedirs(pathdict["metadata_dir"], exist_ok=True)

        # DLC
        if include_dlc:
            if self.Params["load_params"]["condition"] != "checkerboard":
                # Then get DLC
                x = self.path_get_dlc(idx_vid, animal = animal)
                for k,v in x.items():
                    pathdict[k] = v        

        return pathdict


    def extract_frame_single(self, ind_video, ind_frame, return_path_only=False,
        fail_if_noexist=True):
        """ returns a single frame as a pathname
        IN:
        - ind_video, flexible type, {int, tuple}, will interpret correclt
        - ind_frame = 0,1 , ...
        - return_path_only, then returns string. otherwise returns FrameClass
        - fail_if_noexist, fails if path no exist. otherwise reutrn None
        OUT:
        - depends:
        --- if path doesnt exist: None
        --- if return_path_only, then string
        --- otherwise (defualt) FrameClass instance, representing this frame.
        """

        from .frame_class import FrameClass
        import os

        path_frames = self._paths(ind_video)["path_frames"]
        path_thisframe = f"{path_frames}/frame{ind_frame}.jpg"

        # check if frame eixsts
        if os.path.isfile(path_thisframe):
            if return_path_only:
                return path_thisframe
            else:
                F = FrameClass(path_thisframe)
                return F
        else:
            if fail_if_noexist:
                print(path_thisframe)
                assert False, "doesnt exist"
            else:
                return None


    # ##################### UTILS [operates on frames]
    # # Underscored means takes in framedict, holding data for a single frame, output from self.extract_frame_single()
    # # In general, should wrap these in functions that take in indexes (video, frame), not framedict. 
    # def _frame_to_image(self, framedict):

    def get_key(self):
        """
        Function to get user input for the "GUI"
        """
        from pynput import keyboard
        key_pressed = []
        def on_press(key):
            try:
                # Check if the key is good
                if key.char in ['1','2','9','0']:
                    # Convert the key to an integer and store it
                    key_pressed.append(int(key.char))
                    return key_pressed
            except AttributeError:
                pass
                print("Enter valid key")

        def on_release(key):
            # Stop the listener once a valid key is pressed
            try:
                # Check if the key is good
                if key.char in ['1','2','9','0']:
                    # Convert the key to an integer and store it
                    return False
            except AttributeError:
                pass
                print("Enter valid key")

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
        if key_pressed:
            return key_pressed[0]  
        else:
            print("Label issue please repeat key stroke")
            return self.get_key()
        
        # Return the stored key
    
    def review_reprojected_frames(self,frames_list, add=60):
        '''
        Function to review reprojected frmaes to pick the good ones
        '''
        from PIL import Image
        frames_dict = {}
        i = 0
        good_hits = 0
        print(frames_list)
        while (i < len(frames_list) and good_hits < add):
            print("loop")
            frame = frames_list[i]
            print(frame)
            with Image.open(frame) as im:
                plt.figure(figsize=(20,20))
                plt.imshow(im,aspect='auto')
                plt.axis('off')
                plt.tight_layout()
                plt.show(block=False)
                plt.draw()
                plt.pause(0.5)
                print("""########################## \n
                Swipe left for bad frame (hit 1 key) \n
                Swipe right for good frame (hit 2 key) \n
                Or mv fwd/bkwd(9/0) \n
                btw hitting 9 subtract 1 from selected good frames count \n
                lazy behavior but its to ensure we get at least the num frames wanted \n
                per vid, in the case where you go back and change from good to bad""")
                print(f"Doing image: {i}")
                print(f"Good frames so far: {good_hits}")
                response = self.get_key()
                print(f"You entered: {response}")
                if response == 9:
                    i = i-1
                    good_hits = good_hits-1
                elif response == 0:
                    i = i+1
                else:
                    frames_dict[f"{frame}"] = response
                    i = i+1
                    if response == 2:
                        good_hits = good_hits+1
                plt.close()
        #Get list of just frame names that are good
        good_frames_list = [k.split('/')[-1] for k,v in frames_dict.items() if v == 2]
        return good_frames_list

    ##################### CALIBRATION (SINGLE CAMERA)
    def calibrate_each_camera(self, patternSize=(9,6), ploton=False,
        camnum = None, manual_good_frames = False):
        """ each camera calibrated based on good frames that are
        pre-ectracted. this uses the list of cameras in self.DatGroups,
        so is actuall indexed by (camname, videogroup). Saves plots
        and calibration files in the shared path.
        - Must have run self.collect_goodframes_from_videos first.

        PARAMS:
        - camnum, if not None, then index, to pick out a camera
        - manual_good_frames, bool, flag to indicate if frames have been manually reviewd
        already (True), or if function should automatically review frames and ask for user input
        """
        import random
        import shutil
        from ..utils.calibrate import get_checkerboards, find_good_frames_from_all
        from pythonlib.tools.expttools import writeStringsToFile
        for DAT in self.DatGroups:
            # figure out savedir
            SDIR = DAT["path_to_collected_frames"]
            SDIR_CALIB = f"{SDIR}/calib_pycv2"
            os.makedirs(SDIR_CALIB, exist_ok=True)
            print("Saving calibration results at:")
            print(SDIR_CALIB)

            if manual_good_frames:
                # get pathlist
                pathlist_frames = findPath(SDIR, [], "", ".jpg")
            else:
                #Automatically find good frames from all frames
                all_frames_dir = f"{DAT['videos_path_list'][0]}-frames"
                all_frames = [os.path.join(all_frames_dir,f) for f in os.listdir(all_frames_dir)]
                pathlist_frames = find_good_frames_from_all(all_frames,patternSize)
                pathlist_frames_samp = random.sample(pathlist_frames,150)

            ### Find corner pts (world and camera) and reporject and save fig
            successes, objpts, imgpts, imsize, figlist = get_checkerboards(pathlist_frames_samp, patternSize=patternSize)

            # Save figures
            SDIR_CALIB_FIGS = f"{SDIR_CALIB}/frames_reprojected"
            os.makedirs(SDIR_CALIB_FIGS, exist_ok=True)
            for i, (fig, path) in enumerate(zip(figlist, pathlist_frames_samp)):
                name = extractStrFromFname(path, None, None, return_entire_filename=True)
                fig.savefig(f"{SDIR_CALIB_FIGS}/{name}.jpg")
            plt.close("all")

            if not manual_good_frames:
                #add 60 frames to calib by default
                add=60
                os.makedirs(f"{SDIR_CALIB}/reviewed_reprojs", exist_ok=True)
                existing_frames = os.listdir(f"{SDIR_CALIB}/reviewed_reprojs")
                existing_good_frames = []
    
                if len(existing_frames) > 1:
                    while True:
                        print("""
                        Some reviewed frames already exist
                        would you like to (type number): 
                        (1) Restart calibration
                        (2) Add more frames to the calibration\n""")
                        user_input = self.get_key()
                        if user_input == 1:
                            print("Deleting old frames and restarting calibration...")
                            shutil.rmtree(f"{SDIR_CALIB}/reviewed_reprojs")
                            os.makedirs(f"{SDIR_CALIB}/reviewed_reprojs")
                            existing_good_frames = []
                            existing_frames=[]
                            break
                        elif user_input == 2:
                            add = int(input("How many additional frames would you like to add?\n"))
                            print(f"Adding {add} more frames to calibration...")
                            existing_good_frames = [frame for frame in pathlist_frames if \
                                                    frame.split('/')[-1] in existing_frames]
                            successes_add, objpts_add, imgpts_add, imsize_add, _ = \
                                get_checkerboards(existing_good_frames,patternSize=patternSize,plot_on=False)

                            break
                        else:
                            print("Invalid response, try again")

                reproj_paths = []
                frame_nums_samp = [frame.split('/')[-1] for frame in pathlist_frames_samp]
                for frame in os.listdir(SDIR_CALIB_FIGS):
                    if (frame not in existing_frames) and (frame in frame_nums_samp):
                        reproj_paths.append(os.path.join(SDIR_CALIB_FIGS,frame))
                pathlist_frames_reviewed = self.review_reprojected_frames(reproj_paths, add=add)
                good_inds = [i for i,frame in enumerate(pathlist_frames_samp) \
                              if frame.split('/')[-1] in pathlist_frames_reviewed]
                for frame in pathlist_frames_reviewed:
                    shutil.copyfile(f"{SDIR_CALIB_FIGS}/{frame}",\
                                    f"{SDIR_CALIB}/reviewed_reprojs/{frame}")
                
                objpts = [objpts[i] for i in good_inds]
                imgpts = [imgpts[i] for i in good_inds]
                figlist = [figlist[i] for i in good_inds]
                if len(existing_good_frames)>0:
                    objpts.extend(objpts_add)
                    imgpts.extend(imgpts_add)

            ####### GET CAMERA CALIBRATION

            print(f"{len(figlist)} total frames passed review.")
            # assert len(figlist) >= 30, "Should have at least 30 good frames extracted"

            # 1 Only keep frames sucessfully got checkerboard
            tmp = [[i, o] for i, o in zip(imgpts, objpts) if len(i)>0]
            imgpts = [t[0] for t in tmp]
            objpts = [t[1] for t in tmp]

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, 
                imsize, None, None)

            if False:
                out = cv2.calibrateCamera(objpts, imgpts, imsize, None, None)
                print(out)

            camfiles = {}
            camfiles["calibration_mtx"] = mtx 
            camfiles["calibration_dist"] = dist 
            camfiles["calibration_rvecs"] = rvecs 
            camfiles["calibration_tvecs"] = tvecs 
            camfiles["calibration_error"] = ret 

            print(f"reproj error: {ret}")
            print(f"camera matrix: {mtx}")
            print(f"dsitortion coeffs: {dist}")
            # print(f"camera matrix: {mtx}")

            # SAVE camera calibration results

            for k, v in camfiles.items():
                print("Saving", k)

                if isinstance(v, list):
                    vtext = np.stack(v).squeeze()
                else:
                    vtext = v

            #     if not isinstance(vtext, np.ndarray):
            #         vtext = np.array(vtext)

                # 1) text
                try:
                    np.savetxt(f"{SDIR_CALIB}/{k}.txt", vtext, fmt="%.4f")
                except:
                    pass

                # 2) numpy
                np.save(f"{SDIR_CALIB}/{k}.npy", v)


            # 3) mat (matlb)
            from scipy.io import savemat
            savemat(f"{SDIR_CALIB}/all_files.mat", camfiles)

            # Other calibration notes
            stringsthis = []
            stringsthis.append(f"reprojection error: {ret}")
            stringsthis.append(f"image size: {imsize}")

            writeStringsToFile(f"{SDIR_CALIB}/calib_notes.txt", stringsthis)


            # ========= UNDISTORT
            SDIR_CALIB_UNDISTORT = f"{SDIR_CALIB}/frames_undistorted"
            os.makedirs(SDIR_CALIB_UNDISTORT, exist_ok=True)

            # Run this for all figures
            for path in pathlist_frames:
                img = cv2.imread(path)
                newim = img
                h, w = img.shape[:2]

                newmtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

                dst = cv2.undistort(img, mtx, dist, None, newmtx)
                x,y,w,h = roi
                dstcropped = dst[y:y+h, x:x+w]

                fig, axes = plt.subplots(1,3, figsize=(30, 10))

                # plot
                axes[0].imshow(img, interpolation = 'bicubic')
                axes[0].set_title('original')
                axes[1].imshow(dst, interpolation = 'bicubic')
                axes[1].set_title('undistorted')
                axes[2].imshow(dstcropped, interpolation = 'bicubic')
                axes[2].set_title('undistorted(cropped)')

                # save
                name = extractStrFromFname(path, None, None, True)
                fig.savefig(f"{SDIR_CALIB_UNDISTORT}/{name}.jpg")

                plt.close("all")

        print("**** Post-processings, go to following direction and check (1) corner detection (2) reprojection, and (3) undistorted and (4) whether the values make sense.")
        print(SDIR_CALIB)


    ########################## CAMPY
    # OLD VERSION: uses data in datgroup, not in datvideoes.
    # def campy_get_frametimes(self):
    #     """ For each video, extract frametimes svaed in campy.
    #     - Must have each camera as a single group in self.DatGroups
    #     RETURNS:
    #     - modifies, into self.DatGroups[indgrp]["videos_campy_dat"]
    #     NOTE:
    #     - can also extract metadat saved in csv file, see code for easy mods to do that
    #     """
    #     import csv
    #     assert False, "save these in datvideo, not datgroup"
    #     # Iterate over all cams
    #     for indgrp in range(len(self.DatGroups)):

    #         DIR = self.DatGroups[indgrp]["path_shared"] # dir holding all vids
            
    #         # iterate over all trials.
    #         list_dat_campy = []
    #         for trial in self.DatGroups[indgrp]["videos_trialnum0"]:

    #             # 1, frametimes in numpy
    #             frametimes = np.load(os.path.join(DIR, f"frametimes-t{trial}.npy"))
    #             framenums_session = frametimes[0] # from start of rec session
    #             frametimes_sincefirstframe = frametimes[1] # time since first frame in first session

    #             # 2, metadat, uncomment if want to keep
    #             if False:
    #                 metadat = {}
    #                 metadatf = os.path.join(DIR, f"metadata-t{trial}.csv")
    #                 with open(metadatf, 'r', newline='') as f:
    #                     r = csv.reader(f, delimiter=',')
    #                     for row in r:
    #                         metadat[row[0]] = row[1]
    #                         assert len(row)==2
                            
    #                 # 3 video
    #                 if False:
    #                     for k, v in metadat.items():
    #                         print(k, ":", v)

    #             # Save
    #             dat_campy = {}
    #             dat_campy["framenums_session"] = framenums_session
    #             dat_campy["frametimes_sincefirstframe"] = frametimes_sincefirstframe
    #             dat_campy["trial0"] = trial
                
                
    #             list_dat_campy.append(dat_campy)
                
    #         self.DatGroups[indgrp]["videos_campy_dat"] = list_dat_campy

    def campy_load_frametimes(self):
        """ For each video, extract frametimes svaed in campy.
        - Must have each camera as a single group in self.DatGroups
        RETURNS:
        - modifies, into self.DatVideos[ind]["campy_dat"]
        NOTE:
        - can also extract metadat saved in csv file, see code for easy mods to do that
        """
        import csv
        # Iterate over all cams
        for datvid in self.DatVideos:

            # DIR = self._paths(datvid["index"])["path_dir"]
            DIR = datvid["path_base"] # dir holding all vids
            trial = datvid["trialnum0"]

            # 1, frametimes in numpy
            frametimes = np.load(os.path.join(DIR, f"frametimes-t{trial}.npy"))
            framenums_session = frametimes[0] # from start of rec session
            frametimes_sincefirstframe = frametimes[1] # time since first frame in first session

            # 2, metadat, uncomment if want to keep
            if False:
                metadat = {}
                metadatf = os.path.join(DIR, f"metadata-t{trial}.csv")
                with open(metadatf, 'r', newline='') as f:
                    r = csv.reader(f, delimiter=',')
                    for row in r:
                        metadat[row[0]] = row[1]
                        assert len(row)==2
                        
                # 3 video
                if False:
                    for k, v in metadat.items():
                        print(k, ":", v)

            # Save
            dat_campy = {}
            dat_campy["framenums_session"] = framenums_session
            dat_campy["frametimes_sincefirstframe"] = frametimes_sincefirstframe
            dat_campy["trial0"] = trial
            
            datvid["dat_campy"] = dat_campy
            # list_dat_campy.append(dat_campy)
            
        # self.DatGroups[indgrp]["videos_campy_dat"] = list_dat_campy


    def campy_preprocess_check_frametimes(self, FIX_FRAMETIME_PROBLEMS=False, CHECK_MATCH_VIDEOS=True,
            CHECK_SAME_FRAMES_ACROSS_CAMS=False):
        """ assign correctly all frametimes to videos, and check various 
        sanity checks. 
        PARAMS:
        - FIX_FRAMETIME_PROBLEMS, bool, whether to try fixing issues. Better to not, since ideally 
        want to fix both campy and video frames at samet ime. here would just fix campy.
        - CHECK_MATCH_VIDEOS, bool, important, checks same num frames betwen this and videos.
        aborts if fails.
        - CHECK_SAME_FRAMES_ACROSS_CAMS, bool, checks cams have simialr frames. if fails, then
        doesnt abort, just marks SKIP=True in datv for all n cams for this trial.
        NOTE:
        """
        ### SANITY CHECKES    
        abort_if_fail = False # if true, then aborts. if false, then adds flag: "SKIP" to dat.

        # Load frametimes for each video
        if self.Params["load_ver"]=="campy":
            # 1) Load raw data
            self.campy_load_frametimes()

            if FIX_FRAMETIME_PROBLEMS:
                # 2) Check for any obvious errors in frametimes, and fix here
                for i in range(len(self.DatVideos)):
                    self.campy_fix_frametimes_singlevideo(i)
            
            # 2) consolidate into V.DatGroup[][Dat], a list of dicts
            # puts int he campy data
            if False:
                self.datgroup_consolidate_videos()

            # First make all not skip
            # for datg in self.DatGroups:
            #     for datv in datg["Dat"]:
            #         datv["SKIP"] = False
            for datv in self.DatVideos:
                datv["SKIP"] = False

            list_trials = self.inds_trials()
            list_cams = self.get_cameras().keys()


            if CHECK_MATCH_VIDEOS:
                # 1. num frameimtes matches num frames (i.e., video vs. metadat)
                for datv in self.DatVideos:
                    # dict for this single vid.
                    n1 = self.num_frames(datv["index"]) # from video file
                    times, nums = self.campy_get_frametimes(datv["index"])
                    n2 = len(times)
                    n3 = len(times)
                    assert n1==n2, "probably mismathc between the mp4 file and extracted metadat?"
                    assert n2==n3

            if CHECK_SAME_FRAMES_ACROSS_CAMS:
                # same num videos across cameras
                list_numvids = [len(datg["videos_index_list"]) for datg in self.DatGroups]
                assert len(set(list_numvids))==1
                
                # sanity check, no skipped frames, based on consistency of ifi
                FRAC = 0.1 # max minus min frametime shouldnt be more than this times the mean. 
                for indcam in list_cams:
                    for indtrial in list_trials:
                        out = self.campy_summarize_frametimes((indcam, indtrial))
                        if out["ifi_max"] - out["ifi_mean"]>FRAC*out["ifi_mean"]:
                            if abort_if_fail:
                                print(self.wrapper_extract_dat_video(None, indcam, indtrial))
                                print(out)
                                print(indcam, indtrial)
                                self.campy_plot_frametimes((indcam, indtrial))
                                assert False, "skipped frame?"
                            else:
                                # Skip this trial across all cameras
                                for indcam in list_cams:
                                    dat = self.wrapper_extract_dat_video(None, indcam, indtrial)
                                    dat["SKIP"] = True
                                print(f"NON-UNIFORM FRAMES: Flagged as SKIP: cam {indcam}, trial {indtrial}")



                # 2. num frames matches across videos (for same trial)
                for indtrial in list_trials:

                    list_nframes = [len(self.campy_get_frametimes((indgrp, indtrial))[0])
                                        for indgrp in list_cams]
                    rec_durs = []
                    for indgrp in list_cams:
                        campy_frametimes = self.campy_get_frametimes((indgrp, indtrial))[0]
                        rec_durs.append(campy_frametimes[-1] - campy_frametimes[0])

                    ifis = [r/(n-1) for r, n in zip(rec_durs, list_nframes)]

                    # check that ifis are similar across cameras.
                    if max(ifis)-min(ifis) > 0.01*np.mean(ifis):
                        if abort_if_fail:
                            print(indtrial)
                            for indcam in list_cams:
                                self.campy_plot_frametimes((indcam, indtrial))                        
                            assert False, "cameras dont have same ifis?"
                        else:
                            # Skip this trial across all cameras
                            print(max(ifis)-min(ifis), 0.01*np.mean(ifis))
                            print(max(ifis)-min(ifis) > 0.01*np.mean(ifis))
                            for indcam in list_cams:
                                dat = self.wrapper_extract_dat_video(None, indcam, indtrial)
                                dat["SKIP"] = True
                            print(f"NON-UNIFORM FRAMES ACROSS CAMS: Flagged as SKIP: cam {indcam}, trial {indtrial}")


                    # Check that cameras have same num frames
                    if len(set(list_nframes))>1:
                        # Mark this video as bad:
                        if abort_if_fail:
                            print("PROBLEM: diff num frames. Most likley beucase they skipped frames at start or end,because of assertions passed above")
                            print(list_nframes)
                            print("trioal num:", indtrial)
                            print("rec durations:", rec_durs)
                            print("average interframe intervals:", ifis)

                            for indgrp in list_cams:
                                print(self.helper_index_good((indgrp, indtrial)))

                                # Plot the frametimes for comparison
                                self.campy_plot_frametimes((indgrp, indtrial))

                            assert False, "why num frames dont match across cams?"
                        else:
                            for indgrp in list_cams:
                                dat = self.wrapper_extract_dat_video(None, indgrp, indtrial)
                                dat["SKIP"] = True
                            print(f"DIFF NFRAMES ACROSS CAMS: Flagged as SKIP: cam {indgrp}, trial {indtrial}")
            
            # Summary skips
            print("** OVERALL, SKIPPED THESE VIDEOS due to errors:")
            for D in self.DatVideos:
                if D["SKIP"]:
                    print(D["index"])
        else:
            assert False, "this not campy"



    def campy_get_frametimes(self, index_any):
        """ Helper to get frametime information
        PARAMS:
        - indcam, canoicnical index for cameras. usually similar to indgrp
        - indtrial, 0, 1,...
        RETURNS:
        - frametimes, list of times since first frame of session.
        - framenums, list of ints, num since start of rec sessoin
        """
        dat = self.helper_index_good(index_any)
        # dat = self.wrapper_extract_dat_video(None, indcam, indtrial)
        frametimes = dat["dat_campy"]["frametimes_sincefirstframe"]
        framenums = dat["dat_campy"]["framenums_session"]
        
        return frametimes, framenums

    def campy_summarize_frametimes(self, index_any):
        """ return dict summarizing frametime stats.
        - indcam, canoicnical index for cameras. usually similar to indgrp
        - indtrial, 0, 1,...
        """

        frametimes, framenums = self.campy_get_frametimes(index_any)

        out = {}
        out["nframes"] = len(frametimes)
        out["recdur"] = frametimes[-1] - frametimes[0]
        out["ifi_mean"] =  out["recdur"]/(out["nframes"]-1)
        out["ifi_max"] = np.max(np.diff(frametimes))
        out["ifi_min"] = np.min(np.diff(frametimes))
        out["frametimes"] = frametimes
        out["framenums"] = framenums 
        out["inter_frame_intervals"] = np.diff(frametimes)

        return out

    def campy_plot_frametimes(self, index_any):
        """ series of plots of frametimes.
        Useful for debugging
        """
        from ..utils.plots import plot_frametime_summary

        frametimes, framenums = self.campy_get_frametimes(index_any)

        fig, axes = plot_frametime_summary(frametimes)

        return fig, axes


    def campy_fix_frametimes_singlevideo(self, index):
        """ Fix frametimes issues taht are clearly diagnosable
        PARAMS:
        - index, general index
        RETURNS:
        - modifies self.DatVideo directly.
        """

        ##### Two adjacent ifis that are  half period, this assumed to be an extra
        # frame is passed in somehow. Fix this by removing the middle frame (out of 3)
        # Checks that the pair of frames are adjacent, and both close to half ifi.
        ifis = self.campy_summarize_frametimes(index)["inter_frame_intervals"]
        ifi_mean = np.mean(ifis)
        thresh = 0.75 * ifi_mean
        inds_ifis = np.where(ifis < thresh)[0]

        # find pairs
        if len(inds_ifis)<2:
            # then no pairs..
            return 
        
        ifi_mean = np.mean(ifis[~inds_ifis])
        def close_to_half_ifi(x):
            """ Returns Ture if x is close to 1/2 of the mean ifi
            """
            if (x>0.45*ifi_mean) & (x<0.55*ifi_mean):
                return True
            else:
                return False
        
        frames_to_remove = []
        ifis_used = []
        frametimes_to_add = []
        for i1, i2 in zip(inds_ifis[:-1], inds_ifis[1:]):
            if i1 in ifis_used or i2 in ifis_used:
                # oen of thes ifis has contrinbuted to a succesfuly fixing of 2 adjavent ifis. don't use it.
                continue
            if i1==i2-1:
                # then they are adjacent
                if close_to_half_ifi(ifis[i1]) and close_to_half_ifi(ifis[i2]):
                    # Then they are both close to 1/2 ifi mean over all ifis.
                    ifis_used.extend([i1, i2])
                    frames_to_remove.append(i2) # remove the middle frame
        
        # Remove the bad frames.
        if len(frames_to_remove)>1:
            print(frames_to_remove)
            print(index)
            print(inds_ifis)
            assert False, "first time get >1 case. check it is correct"
            
        if len(frames_to_remove)>0:
            frametimes, framenums = self.campy_get_frametimes(index)

            print("Starting len of frametimes: ", len(frametimes))
            print("Removing these frmaes: ", frames_to_remove)

            frametimes = [f for i,f in enumerate(frametimes) if i not in frames_to_remove]
            framenums = [f for i,f in enumerate(framenums) if i not in frames_to_remove]

            datv = self.helper_index_good(index)
            datv["dat_campy"]["frametimes_sincefirstframe"] = frametimes
            datv["dat_campy"]["framenums_session"] = framenums

            print("New len of frametimes: ", len(frametimes))
            print("Updated datv")

    ########################## DEEP LAB CUT
    def path_get_dlc(self, idx_video, animal, analysis_suffix = "allvideos", videos_in_orig_folder=False):
        """
        Help find path to DLC data for a given video
        PARAMS:
        - videos_in_orig_folder, bool, which folder to look for analyzed data. see below.
        RETURNS:
        - path_dict, path to .h5 file and labeled video.
        NOTE:
        - fails if doesnt fine only one.
        """
        from pyvm.dlc.utils import find_expt_config_paths
        from pyvm.dlc.utils import find_analysis_path

        # dat = self.get_video_from_index(idx_video)
        dat = self.wrapper_extract_dat_video(idx_video)

        path = os.path.split(dat["path_video"])[0]
        name = dat["name"]
        camera_name = dat["camera_name"]
        # iternum = self.Params["load_params"]["dlc_iternum"]
        iternum = self.Params["dlc_iternum"]

        path_dict = {}

        # base dlc path
        dict_path, base_paths = find_expt_config_paths(self.Params["load_params"]["dirname"], self.Params["load_params"]["condition"], animal)
        path_dict["config"] = dict_path["combined"]
        path_dict["dlc_base"] = base_paths["combined"]

        # analysis
        analysis_path, _ = find_analysis_path(path_dict["dlc_base"], 
            analysis_suffix, do_iterate_if_exists=False)
        path_dict["analysis_path"] = analysis_path
        
        # to h5 data
        if videos_in_orig_folder:
            # Then labeled vieods are svaed alongside their origianl vidoes. This is old version, stopped
            # doing becuase overwrote videos whenever redid analsysi.
            path_analyzed_vids = path
        else:
            # New version, where analysis is in dedicxated folder, which iterates each time run
            # analysis.
            # .e.g, "/data2/camera/220317_chunkbyshape4/behavior/DLC/combined-flea_bfs1_bfs2_ffly-Lucas-2022-04-07/analyze_videos-allvideos-0"
            path_analyzed_vids = path_dict["analysis_path"]
        list_path = findPath(path_analyzed_vids, [[camera_name, name+"-", iternum]], 
                             None, "h5", strings_to_exclude_in_path = ["filtered"])
        for p in list_path:
            print("--", p)
        assert len(list_path)==1
        path_dict["coords"] = list_path[0]

        # to labeled video
        list_path = findPath(path_analyzed_vids, [[camera_name, name+"-", iternum, "labeled"]], 
                             None, "mp4", strings_to_exclude_in_path = ["filtered"])
        assert len(list_path)==1
        path_dict["vid_labeled_predictions"] = list_path[0]


        # analysis for this specific video
        list_path = findPath(analysis_path, [[camera_name, name+"-", iternum]], 
                             None, "h5", strings_to_exclude_in_path = ["filtered"])
        assert len(list_path)==1
        path_dict["analysis_path_thisvid"] = list_path[0]
        # - filtered
        list_path = findPath(analysis_path, [[camera_name, name+"-", iternum, "filtered"]], 
                             None, "h5")
        assert len(list_path)==1
        path_dict["analysis_path_thisvid_filtered"] = list_path[0]

        return path_dict


    def import_dlc_data(self, ver="separate_analysis_dir", use_filtered=True, 
            analysis_suffix = "allvideos"):
        """ for each video, imports its DLC data. By defualts, searches for analysis in 
        folder suffixed with "allvideos", and the latest iteration, and filtered.
        PARAMS:
        - ver, str, where to look for analysis. 
        - use_filtered, bool, usually is median filter done using dlc code.
        - analysis_suffix, str, for identiyfing analysis directory.
        SANITY CHECKS
        - n frames match video.
        """
        import pandas as pd

        # For each video find its markers
        animal = self.Params["load_params"]["animal"]

        for dat in self.DatVideos:
                
            for key in ["data_dlc_downscaled", "data_dlc"]:
                if key in dat.keys():
                    del dat[key]

            if ver=="legacy":
                # Then analyses all saved in same folder at videos
                # camtest5 did was last one.
                if False:
                    # Old version
                    path = os.path.split(dat["path_video"])[0]
                    name = dat["name"]
                    name = name[:name.find("_labeled")] # removed labeled from name
                    path = f"{path}/{camera_name}-{name}-downsampled.h5" 
                else:
                    path = self.path_get_dlc(dat["index"])["coords"]
                    print(f"Loading {path}, DLC data")
            elif ver=="separate_analysis_dir":
                # analysis has own folder.
                pathdict = self.path_get_dlc(dat["index"], analysis_suffix=analysis_suffix, animal = animal)
                if use_filtered:
                    path = pathdict["analysis_path_thisvid_filtered"]
                else:
                    path = pathdict["analysis_path_thisvid"]
            else:
                assert False
            df = pd.read_hdf(path)

            # Sanity checks
            # - num frames is correct
            nframes_dlc = len(df)
            nframes_vid = self.num_frames(dat["index"])
            assert nframes_vid == nframes_dlc
            dat["data_dlc"] = df

            # Check if in bounds
            self.dlc_check_in_bounds(dat["index"])

        # Rescale data to orig
        self.dlc_rescale_data()

        print("DONE importing")


    def dlc_rescale_data(self):
        """ Autoamtically figure out coordinates in space of original video
        NOTE:
        - conversion is that top-left corner of video is always (0,0) for orig and 
        dlc.
        """

        # get resolutions of original and dlc
        for datv in self.DatVideos:
            # if already rescaled, then dont do again
            idx = datv["index"]

            if self.dlc_which_version(idx)=="orig":
                assert False, "already done, dont do again"
            
            # get their resolutions
            res = self.resolutions(idx)
            wh_orig = res["orig"]
            wh_dlc = res["dlc"]

            # x (width)
            ratio_x = wh_orig[0]/wh_dlc[0]
            ratio_y = wh_orig[1]/wh_dlc[1]
            
            # upscale all dlc datapoints.
            df = datv["data_dlc"]
            data_dlc_downscaled = df.copy() # save copy of original.
            datv["data_dlc_downscaled"] = data_dlc_downscaled

            # Replace df with original coord.
            for col in df.columns:
                if col[2]=="x":
                    df[col] = df[col]*ratio_x
                elif col[2]=="y":
                    df[col] = df[col]*ratio_y
                elif col[2] in ["likelihood", "out_of_bounds"]:
                    pass
                else:
                    print(col)
                    assert False

    def dlc_which_version(self, idx_vid):
        """ returns "orig" or "dlc" for which coord
        is currently active. 
        """
        datv = self.wrapper_extract_dat_video(idx_vid)
        if "data_dlc_downscaled" in datv.keys():
            # Then already returned to original coords
            return "orig"
        else:
            return "dlc"

    def dlc_check_in_bounds(self, idx_vid):

        # check orig or dlc? (for bounds)
        wh = self.resolutions(idx_vid)[self.dlc_which_version(idx_vid)]

        def _check_bounds(dfthis, feat, col):
            # returns inds for bad rows (out of bounds)
            minval = 0
            if feat=="x":
                maxval = wh[0]
            elif feat=="y":
                maxval = wh[1]
            else:
                assert False
            return (dfthis<minval) | (dfthis>maxval)
            
        datv = self.wrapper_extract_dat_video(idx_vid)
        df = datv["data_dlc"]
        for col in df.columns:
            if col[2] in ["x", "y"]:
                inds = _check_bounds(df[col], col[2], col)
                df[(col[0], col[1], "out_of_bounds")] = False
                df.loc[inds, (col[0], col[1], "out_of_bounds")]= True



    def dlc_get_multiindex(self, part, feat, vid_num = 0 ):
        """ helper to return the multicolumn index, which is
        (model, part, feat)
        INPUTS:
        - vid_num = 0 # choose any. assume you have extracted all.
        - feat, e.g., {x, y, likelihood}
        """
        df = self.DatVideos[vid_num]["data_dlc"]
        col = [col for col in df.columns if col[1]==part and col[2] == feat]
        # print(len(col))
        # df.head()
        assert len(col)==1
        return col[0]


    def dlc_get_list_parts_feats(self):
        """ Return list of unique parts and features across all vidoels.
        OUT:
        - list_parts, list of str
        - list_feats, list of str
        NOTE:
        - assumes first video feats and parts are same all othres.
        """
        indvid = 0
        df = self.DatVideos[0]["data_dlc"]  

        list_part = sorted(set([col[1] for col in df.columns]))
        list_feat = sorted(set([col[2] for col in df.columns]))

        return list_part, list_feat



    def dlc_data_part_feat(self, idx_vid, part, feat):
        """ helper to deal with multi-indexing, to 
        pul out dta for this part (e..g, finger) and feature (e.g., x)
        """

        datv = self.wrapper_extract_dat_video(idx_vid)

        # datv = self.get_video_from_index(idx_vid)
        df = datv["data_dlc"]
        # find the columns name
        col = self.dlc_get_multiindex(part, feat)

        return df[col]


    def dlc_data_part_feat_mult(self, vid_num, part, feat_list):
        """ multiple fetaures, returns in a stacked np array, size N x len(featlist)
        """

        dat = []
        for feat in feat_list:
            dat.append(self.dlc_data_part_feat(vid_num, part, feat).values)

        return np.stack(dat, axis=1)

    def dlc_info_parts(self):
        """ Return list of parts tracked/labeled.
        RETURNS:
        - list_part, list of string
        """
        return self.Params["load_params"]["bodyparts"]
    def dlc_extract_pts_matrix_dan(self, indtrial, frames_get, list_part=None):
        """ 
        Extract matrix of pts across cameras, in formate useful for triangulation, 
        easywand, DLT, etc.
        IN:
        - list_part, list of strings, .e.g, list_part
        --- if None, tries to figure out what you want.
        - ind_trial, int.
        - frames_get, dict, keys are cam and values are good frames for that cam
        RETURN:
        - vals
        --- array, with shape (nframes, nparts x 2(xy))
        columsn, lsit of tuples, each labeling a columns. i.e,:
        --- columns are like: part1_cam1_x, part1_cam1_y, part1_cam2_x, ...
        NOTE:
        - camera order will always be 0, 1, 2,.. where these are canonical indices
        (i.e., see self.get_cameras())
        """

        # Which parts
        dict_cameras = self.get_cameras()
        # print(dict_cameras)
        # print("get_cams_count",GET_CAMS_COUNT)
        # assert False
        if list_part is None:
            if self.Params["load_params"]["condition"]=="wand":
                # list_part = ["red", "blue"]
                list_part = self.dlc_info_parts()
            else:
                assert False, "cant figure out which parts you want"
        #Generate list of good frames across all cameras, to track who needs NaN
        good_frames_all = []
        for _,frames in frames_get.items():
            good_frames_all = list(set(good_frames_all) | set(frames))

        #Make sure that we have alits of ints for frames
        if isinstance(frames_get, dict):
            assert isinstance(frames_get[dict_cameras[0][0]][0], int)
        else:
            assert False

        
        list_feat = ["x", "y"]
        vals = []
        columns = []
        for part in list_part:
            for camname, ind in dict_cameras.values(): # dict
                datv = self.wrapper_extract_dat_video(None, camname, indtrial)
                idxvid = datv["index"]
                xy = self.dlc_data_part_feat_mult(idxvid, part, list_feat)
                # frames_get = [frame for frame in frames_get if frame < len(xy)]
                #Extract all frames for this cam, then fill in the bad frames with NaN values 
                bad_inds = [f for f in good_frames_all if f not in frames_get[camname]]
                xy[bad_inds] = np.nan
                filt_frames = xy[good_frames_all,:]
                #LOWER LEFR COORDINATE SYSTEM
                # _,height = self.resolutions(idxvid)["orig"]
                # filt_frames[:,1] = height-filt_frames[:,1] 
                # print(bad_inds)
                # print(all_frames)
                # assert False
                # all_frames[bad_inds] = np.nan
                vals.append(filt_frames)
                for feat in list_feat:
                    columns.append((part, datv["camera_name"], feat))

        vals = np.concatenate(vals, axis=1)
        # assert np.any(np.isnan(vals)) == False, "why nan?"
        return vals, columns
    def dlc_extract_pts_matrix(self, indtrial, list_part=None, frames_get="all"):
        """ 
        Extract matrix of pts across cameras, in format useful for triangulation, 
        easywand, DLT, etc.
        IN:
        - list_part, list of strings, .e.g, list_part
        --- if None, tries to figure out what you want.
        - ind_trial, int.
        - frames_get, flexible, 
        --- list of ints: range(10)
        --- str, "good_frames"
        --- str, "all"
        RETURN:
        - vals
        --- array, with shape (nframes, nparts x 2(xy))
        columsn, lsit of tuples, each labeling a columns. i.e,:
        --- columns are like: part1_cam1_x, part1_cam1_y, part1_cam2_x, ...
        NOTE:
        - camera order will always be 0, 1, 2,.. where these are canonical indices
        (i.e., see self.get_cameras())
        """

        # part = ""
        # indtrial = 0
        # prmlist = [
        #     (0, "red", ["x", "y"]),
        #     (1, "red", ["x", "y"]),
        #     (2, "red", ["x", "y"]),
        #     (0, "blue", ["x", "y"]),
        #     (1, "blue", ["x", "y"]),
        #     (2, "blue", ["x", "y"]),
        # ]

        # Which parts
        dict_cameras = self.get_cameras()
        if list_part is None:
            if self.Params["load_params"]["condition"]=="wand":
                # list_part = ["red", "blue"]
                list_part = self.dlc_info_parts()
            else:
                assert False, "cant figure out which parts you want"

        # Which frames?
        if isinstance(frames_get, list):
            assert isinstance(frames_get[0], int)
        elif isinstance(frames_get, str):
            if frames_get=="good_frames":
                frames_get = self.datgroup_extract_single_video_data(0, indtrial, True)["good_frames"]
                assert len(frames_get)>0, "need to enter good frames"
            #Find camera with smallest number of frames for this trial and take those ones.
            #This assumes that missed frames are at the end of the trial whihc may not be true....
            elif frames_get=="all":
                frames = []
                for i in range(len(dict_cameras)):
                    try:
                        n = self.num_frames2(i, indtrial)
                    except AssertionError:
                        return [],[]
                    frames.append(n)
                min_frames = min(frames)
                frames_get = range(0,min_frames)
            else:
                assert False
        else:
            assert False

        
        list_feat = ["x", "y"]
        vals = []
        columns = []
        for part in list_part:
            for i in range(len(dict_cameras)): # dict
                camname = dict_cameras[i][0]
                datv = self.wrapper_extract_dat_video(None, camname, indtrial)
                idxvid = datv["index"]
                xy = self.dlc_data_part_feat_mult(idxvid, part, list_feat)
                # frames_get = [frame for frame in frames_get if frame < len(xy)]
                #CHANGE TO LOWER LEFT COORDINATE SYSTEM
                # width,height = self.resolutions(idxvid)["orig"]
                # print(camname, width, height)
                # xy[:,1] = height-xy[:,1] 
                vals.append(xy[frames_get,:])
                for feat in list_feat:
                    columns.append((part, datv["camera_name"], feat))

        vals = np.concatenate(vals, axis=1)
        assert np.any(np.isnan(vals)) == False, "why nan?"
        return vals, columns



    ################## VARIOUS HELPERS (extract things)
    def resolutions(self, idx_vid):
        """ Return dict with resolution for all videos
        """
        from ..utils.cvtools import get_video_wh

        paths = self.get_paths_good(idx_vid)
        vid_orig = paths["path_vid_full"]
        vid_dlc = paths["vid_labeled_predictions"]

        # get their resolutions
        wh_orig = get_video_wh(vid_orig)
        wh_dlc = get_video_wh(vid_dlc)

        return {
            "orig":wh_orig,
            "dlc":wh_dlc
        }

    def num_trials(self):
        indcam = 0
        print(self.DatGroups[0])
        assert False, "do this without using datgroups?"
        return len(self.DatGroups[0]["Dat"])

    def inds_trials(self):
        """ return list of all unique trialnums that exist across all cams
        """
        return sorted(set([D["trialnum0"] for D in self.DatVideos]))

    def get_cameras(self):
        """ get useful list of cameras. 
        each camera associated with unique "cameranum" as decided when you 
        inputed in params.
        RETURNS:
        - dict[camid] = (camname, list_of_datgroup_inds).
        --- where camid is 0, 1, 2...
        --- camname is str
        --- where list_of_datgroup_inds...
        """
        dict_cams = self.Params["load_params"]["camera_names"]
        dict_cams_datgroups = {}
        # For each camera, get list of datgroups for it
        for camnum, camname in dict_cams.items():
            inds = [i for i, datg in enumerate(self.DatGroups) if datg["index_grp"][0]==camname]
            dict_cams_datgroups[camnum] = (camname, inds)
        # print(dict_cams_datgroups)
        # print("get_cams_count",GET_CAMS_COUNT)
        # assert False
        return dict_cams_datgroups


    def num_cams(self):
        return len(self.DatGroups)

    def num_frames2(self, indcam, indtrial):
        """ get num frames for this cam and trial,
        if grouping cameras, i.e., index by camera and trial
        PARAMS:
        - indcam, canonical index
        - intrial, int, 0, 1, ..
        """

        idx = self.wrapper_extract_dat_video(None, indcam, indtrial)["index"]
        # idx = self.datgroup_extract_single_video_data(indcam, indtrial)["index"]
        
        return self.num_frames(idx)


    def num_frames(self, ind_video):
        """ 
        Returns int, num frames existing for this video
        """

        # First method, try to load from saved metadat
        val = self.metadat_read("num_frames", ind_video)
        if val is not None:
            # print("num frames gotten, method 1")
            return val

        # Second, check that frames exist for this vidoe.
        # fails if doesnt.
        out = self.extract_frame_single(ind_video, 0, 
            return_path_only=True, fail_if_noexist=False)
        if out is not None:
            # Then frames have been preextracted.
            # now count frames.
            # print("num frames gotten, method 2")
            framenum =0
            out = True
            while out is not None:
                out = self.extract_frame_single(ind_video, framenum, 
                    return_path_only=True, fail_if_noexist=False)
                framenum+=1
            return framenum-1

        # Third method, loads framesa directly and counts
        # Then frame is not preextracted. Use manual count method
        from ..utils.preprocess import count_frames, get_frames_from_video
        D = self.wrapper_extract_dat_video(index_vid = ind_video)
        # print("num frames gotten, method 3")

        # nframes = None
        # get_frames_from_video(D["path_video"])
        nframes = count_frames(D["path_video"])
        return nframes            



    def helper_index_good(self, index):
        """ [GOOD] flexible index input, and always same output, single video dict
        PARAMS:
        - index:
        --- 2-tuple, then is (indcam, intrial) -- 
        --- 3-tuple, then is video index (camname, group, vidname)
        --- int, then indexes into self.DatVideos
        --- str, fails
        RETURNS:
        - datv, single item in self.DatVideos
        """
        def get_vidindex_from_cam_trial(indcam, indtrial):
            """
            Get index for video for this camera and trial
            PARAMS:
            - indcam, canimcal index either:
            --- str or int
            RETURNS:
            - indvid, tuple
            """

            if isinstance(indcam, int):
                # convert to string
                indcam = self.get_cameras()[indcam][0]
            else:
                assert isinstance(indcam, str)

            # list_indgrp = self.get_indgrp_from_index(indcam)
            # assert len(list_indgrp)==1, "not sure what to do if multiple groups"
            # indgrp = list_indgrp[0]

            x = [datv for datv in self.DatVideos if \
                datv["trialnum0"]==indtrial and datv["camera_name"]==indcam]
            # if indcam == 'fly1':
            #     for datv in self.DatVideos:
            #         print(datv["trialnum0"],indtrial,datv["camera_name"],indcam)


            # for k, v in self.DatGroups[indgrp].items():
            #     print(k, v)
            # for k, v in self.DatVideos[0].items():
            #     print(k, v)
            # assert False
            # x = [dat for dat in self.DatGroups[indgrp]["Dat"] if dat["trialnum"]==indtrial]
            if len(x)==0:
                print("** You are looking for")
                print(indtrial)
                print(indcam)
                assert False,"Didn't find any"
            elif len(x)>1:
                print(x)
                assert False, "found too manby"
            return x[0]["index"]

        def _datvideo_from_index(index):
            """ index into video"""
            if isinstance(index, int):
                return self.DatVideos[index]
            elif isinstance(index, tuple):
                tmp = [D for D in self.DatVideos if D["index"]==index]
                assert len(tmp)==1
                return tmp[0]
            elif isinstance(index, str):
                assert False, "code this - throw error if multiple vidoes have same camname"
            else:
                print(index)
                print(type(index))
                assert False, "not coded"

        # Is this video or (cam, trial) index?
        ver = "video"
        if isinstance(index, tuple):
            if len(index)==2:
                ver = "cam_trial"

        # Get the video data
        if ver=="video":
            return _datvideo_from_index(index)
        elif ver=="cam_trial":
            index_vid = get_vidindex_from_cam_trial(index[0], index[1])
            return _datvideo_from_index(index_vid)
        else:
            print(index)
            assert False


    def get_indgrp_from_index(self, idx_cam):
        """
        Flexible index, gets the list of indgrps that match this camera
        - idx_cam, can be int or string, canoncinal index.
        RETURNS:
        - list of indices into self.DatGroup
        """

        if isinstance(idx_cam, int):
            return self.get_cameras()[idx_cam][1]
        elif isinstance(idx_cam, str):
            dictcam = self.get_cameras()
            x = [v[1] for k, v in dictcam.items() if v[0]==idx_cam]
            assert len(x)==1, "didnt find this camera..."
            return x[0]
        elif isinstance(idx_cam, tuple):
            print(tuple)
            assert False, "figure out how to proceed"
        else:
            print(idx_cam)
            assert False


    def wrapper_extract_dat_video(self, index_vid=None, index_cam=None, index_trial=None):
        """
        Flexible extraction of a single video datapt
        PARAMS:
        Either enter:
        - index_vid, tuple or int (indexes into self.DatVideo)
        or:
        - index_cam, either str or int. canonical indx
        - index_trial. int
        RETURNS:
        - dat
        NOTE:
        - this is entirely obsolete, replaced by helper_index_good
        """
        
        # def get_video_from_index(index):
        #     """ flexible index type
        #     - if int, then assume self.DatVideos[index]
        #     - if tuple, then is (camname, group, vidpath). will assert that there only one.
        #     - if str, then is camname..
        #     """
        #     return self._datvideo_from_index(index)


        # def datgroup_extract_single_video_data2(index_cam, indtrial):
        #     """ Extract video for single trial. Different from v1 in here using indexcam.
        #     INPUT:
        #     - index_cam is canonical index for this camera. multiple types possible:
        #     --- int, which is a fixed order, 0, 1, 2...
        #     --- str, name of camera
        #     - indtrial, trial, 0, 1, ...
        #     NOTE: always get original_vid_data. if want original_vid_data=False, look into
        #     datgroup_extract_single_video_data
        #     """
 
        #     # ind_vid = self.get_vidindex_from_cam_trial(index_cam, indtrial)
        #     # return get_video_from_index(ind_vid)
        #     return self.helper_index_good((index_cam, indtrial))

        #     # # First get the indgrps corresponding to this cam and trial
        #     # list_indgrp = self.get_indgrp_from_index(index_cam)
        #     # assert len(list_indgrp)==1, "this camera has multiple indices, bug."
        #     # return self.datgroup_extract_single_video_data(list_indgrp[0], indtrial, 
        #     #     original_vid_data=original_vid_data)


        # def datgroup_extract_single_video_data(indgrp, indtrial, original_vid_data=False):
        #     """ Helper to get a single trial (defined by chron order, only for
        #     campy so far) for single camera (indgrp)
        #     [Use datgroup_extract_single_video_data2 instead]
        #     INPUT:
        #     - indgrp, 0, 1, ...
        #     - indtrial, 0, 1, ...
        #     - original_vid_data, bool. [avoid this, just make this True]
        #     OUTPUT:
        #     - dict, holding this data.
        #     NOTE:
        #     - must first have run datgroup_consolidate_videos
        #     """
        #     assert False, "dont use this. it doesnt use video canonicial index"
        #     x = [dat for dat in self.DatGroups[indgrp]["Dat"] if dat["trialnum"]==indtrial]
        #     assert len(x)==1
        #     dat = x[0]
        #     if original_vid_data:
        #         return self.helper_index_good(dat["index"])
        #         # return get_video_from_index(dat["index"])
        #     else:
        #         return dat

        if index_vid is not None:
            assert index_cam==None and index_trial==None
            return self.helper_index_good(index_vid)
            # return self.helper_index_good(index_vid)
        else:
            assert index_vid==None
            return self.helper_index_good((index_cam, index_trial))
            # return datgroup_extract_single_video_data2(index_cam, index_trial)
