""" to pull in video hand pose tracking data and integrate with ml2 data.
The main goal is the integration of different kinds of data (ml2 touchscreen,
campy frametimes, and dlc XYZ coords.). Then plan is to extract this data, and
do actual analyses in Dataset code.
"""
assert False, "Dont use"

import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from pythonlib.tools.stroketools import *

# SAVEDIR = "/data2/camera"
SAVEDIR = "/home/danhan/Documents/hand_track/data/Pancho"
NCAMS=4

class HandTrack(object):
    """ 
    NOTE:
    - Generally use a new Handtrack Instance for each filedata (fd).
    """
    def __init__(self, ind1_vid, ind1_ml2, fd, date, expt, regressor = 0):
        """
        PARAMS
        - ind1_vid, the first video ind with data. Should correspnd to
        trial in ind1_ml2. 0-indexed, as per suffixes based on campy.
        - ind1_ml2, trial in filedata, matching first vid trial, 1-indexed.
        - date, int or str, YYMMDD
        - expt str
        """

        self.IndFirst ={}
        self.IndFirst["vid"] = ind1_vid
        self.IndFirst["ml2"] = ind1_ml2
        self.Date = str(date)
        self.Expt = expt
        self.Fd = fd
        self.regressor = regressor

        # trial_map = [1, 6] # if [1,6], then ml2 trial 1 is vid6

    def setRegressor(self, reg):
        self.regressor = reg

    def process_data_singletrial(self, trial_ml2, ploton=False, 
            filter_by_likeli_thresh=True, return_in_meters = True, finger_raise_time=0.05,
            ncams=3):
        """ Does manythings:
        - Converts cam data into strokes formate, for both strokes and gaps.
        - interpoaltes cam so that matches ml2 timings.
        - gets in units of both pixels and m
        PARAMS:
        - runs for a single trial.
        RETURNS:
        - dict, holding data
        (empty {} if there is no data)
        NOTE:
        - skips if this trial no fixation success
        """

        from .utils import getTrialsStrokes, getTrialsFixationSuccess, getTrialsTaskAsStrokes

        datall = {}

        if not getTrialsFixationSuccess(self.Fd, trial_ml2):
            return {}, []
        # # get interpolated data in two forms:
        # # i) flat array
        # tnew = np.concatenate([s[:,2] for s in strokes])
        # pts_cam_interp = f(tnew)
        # # ii) in strokes format
        # strokes_cam_interp = [f(s[:,2]) for s in strokes]
        # DAT["pts_cam_interp"] = pts_cam_interp
        # DAT["strokes_cam_interp"] = strokes_cam_interp

        ####### Convert cam data into strokes and gaps.
        def snap_pts_to_strokes(strokes_template, pts_to_snap, finger_raise_time=finger_raise_time):
            """
            takes datapoints in pts_to_snap, and generates strokes version of this, 
            based on relatiing its timettamps to those fo strokes_template. Assumes
            that pts_to_snap can yield both sstrokes and gaps. 
            PARAMS:
            - strokes_template, strokes type, this determines the onsets and offsets of
            strokes (that's all)
            - pts_to_snap, NxD, where D usually 3 (x,y,t) or 4 (x,y,z,t). 
            - finger_raise_time = 0.1 # num seconds allow for raising and lowering. will ignore this much of the time flanking
            # strokes, for defining what is a gap.
            RETURNS:
            - DAT, dict holding variations of strokes formatted pts_to_snap
            NOTES:
            - assumes that the last column is time. will use this to do snapping
            - naming reflects that wrote this for cam data.
            """
            from scipy.interpolate import interp1d
    
            DAT ={}

            # --- Which are dimensions of time?
            dim_t_1 = strokes_template[0].shape[1]-1
            dim_t_2 = pts_to_snap.shape[1]-1

            # Interpolate video to match touchscreen times
            t = pts_to_snap[:,dim_t_2]
            pts = pts_to_snap[:,:dim_t_2]
            funcinterp = interp1d(t, pts, axis=0)

            # create strokes_cam, using original cam pts, taking within bounds of strokes.
            strokes_cam = []
            strokes_cam_interp = []
            gaps_cam = [] # same format as strokes..
            t0 = 0.
            for i, strok in enumerate(strokes_template):
                
                # find all cam pts within bounds of the times of this ml2 stroke
                t1 = strok[0,dim_t_1]
                t2 = strok[-1,dim_t_1]
                tall = strok[:,dim_t_1]
                
                # - strokes
                inds = (pts_to_snap[:,dim_t_2]>=t1) & (pts_to_snap[:,dim_t_2]<=t2)
                strokes_cam.append(pts_to_snap[inds, :])

                # - strokes, but interpolate to use same timestamps 
                if np.any((tall<=t[0]) | (tall>=t[-1])):
                    assert False, "fix the underying issue."
                tall = tall[(tall>=t[0]) & (tall<=t[-1])] # cannot extrapolate.
                strokes_cam_interp.append(funcinterp(tall))
                
                # - gaps (the one preceding this stroke)
                inds = (pts_to_snap[:,dim_t_2]>t0+finger_raise_time) & (pts_to_snap[:,dim_t_2]<t1-finger_raise_time)
                gaps_cam.append(pts_to_snap[inds, :])
                
                # if this is the last stroke, then the rest of data is a long gap
                if i==len(strokes_template)-1:
                    inds = (pts_to_snap[:,dim_t_2]>t2+finger_raise_time)
                    gaps_cam.append(pts_to_snap[inds, :])
                
                # - update t0 for next gap
                t0 = t2

            # remove empty things
            strokes_cam = [s for s in strokes_cam if len(s)>0]
            gaps_cam = [s for s in gaps_cam if len(s)>0]        

            # get flatteend versions
            pts_cam_interp = np.concatenate(strokes_cam_interp)

            DAT["pts_cam_interp"] = pts_cam_interp
            DAT["strokes_cam_interp"] = strokes_cam_interp
            DAT["strokes_cam"] = strokes_cam
            DAT["gaps_cam"] = gaps_cam
            DAT["pts_time_cam_all"] = pts_to_snap
            DAT["strokes_touch"] = strokes_template

            return DAT


        def convert_pix_to_meters(pts):
            """ 
            PARAMS:
            - pts, assumes all columns are pixels (i..e, removed last time oclumn)
            RETURNS:
            - copy of pts, in meters
            """
            pts = pts.copy()
            conv = self.coordinate_conversion()
            return pts/conv["pix_over_m"]

        # 1) Extract cam and coinvert coords.
        dfall, t, pts, camdict = self.get_trials_all_data(trial_ml2, filter_by_likeli_thresh=filter_by_likeli_thresh, ncams=ncams)
        if dfall is None:   
            # failed becuase no campy data
            return {}, []
        dfall = self.convert_coords(dfall)
        pts_time_cam_all = dfall[["x", "y", "z", "t_trial"]].values # all times, not just those in ml2 strokes

        # get strokes from onset to touch done
        # strokes = getTrialsStrokesByPeanuts(fd, trial_ml2)
        # strokes = getTrialsStrokesClean(fd, trial_ml2)
        strokes = getTrialsStrokes(self.Fd, trial_ml2, window_rel_go_reward = [-0.1, 0.1])
        strokes_task = getTrialsTaskAsStrokes(self.Fd, trial_ml2)

        if len(pts_time_cam_all)==0:
            return {}, []

        if return_in_meters:
            pts_time_cam_all = pts_time_cam_all.copy()
            pts_time_cam_all[:, :3] = convert_pix_to_meters(pts_time_cam_all[:, :3])
            strokes_meters = []
            for strok in strokes:
                x = strok.copy()
                x[:, :2] = convert_pix_to_meters(x[:,:2])
                strokes_meters.append(x)
            strokes = strokes_meters

            strokes_meters = []
            for strok in strokes_task:
                x = strok.copy()
                x[:, :2] = convert_pix_to_meters(x[:,:2])
                strokes_meters.append(x)
            strokes_task = strokes_meters

        dat = snap_pts_to_strokes(strokes, pts_time_cam_all)
        dat["strokes_task"] = strokes_task

        # store this
        for k, v in dat.items():
            datall[k]=v

        # dat = snap_pts_to_strokes(strokes_meters, pts_time_cam_all_meters)
        # for k, v in dat.items():
        #     datall[f"{k}_meters"]=v

        #Applied lienar regression to cam data if regressor is fed into function
        assert False
        if True:
            print("###DOING REGRESSION ON CAM PTS###")
            assert False, "Made it"
            reg_pts_list = []
            for strok_cam in datall["strokes_cam"]:
                reg_cam_pts = self.regressor.predict(strok_cam)
                reg_pts_list.append(np.array(reg_cam_pts))
            datall["reg_strokes_cam"] = reg_pts_list
            

        # Plot to compare extracted strokes
        if ploton:
            # assert False, "plot z coordinates of gaps by color. incorporate this into plotDatStrokes"
            from pythonlib.dataset.dataset  import Dataset
            from pythonlib.drawmodel.strokePlots import plotDatStrokes
            from .utils import getTrialsStrokes

            list_figs = []

            # 1) Plot each coordinate timecourse, seaprate plots
            pts_cam = dat["pts_time_cam_all"]
            nplots = pts_cam.shape[1]
            t = pts_cam[:,3]
            fig, axes = plt.subplots(nplots ,1, sharex=True, figsize=(10, nplots*4))
            list_figs.append(fig)
            for i in range(nplots):
                ax = axes.flatten()[i]
                ax.plot(t, pts_cam[:,i])
                ax.set_ylabel(f"dim{i}")
                ax.set_xlabel("time")

            # 1) Plot each coordinate timecourse, seaprate plots
            pts_cam = dat["pts_time_cam_all"]
            z = pts_cam[:,2]
            fig, axes = plt.subplots(2 ,1, sharex=True, figsize=(10, nplots*4))
            list_figs.append(fig)
            for i in range(2):
                ax = axes.flatten()[i]
                ax.plot(pts_cam[:,i], z)
                ax.set_ylabel("z")
                ax.set_xlabel(f"dim{i}")

            # 2) Histograms of all values
            fig, axes = plt.subplots(nplots ,1, sharex=False, figsize=(7, nplots*2))
            list_figs.append(fig)
            for i in range(nplots):
                ax = axes.flatten()[i]
                ax.hist(pts_cam[:,i], 50)
                ax.set_ylabel(f"dim{i}")

            # 4) Overlay on touchscreen data
            fig, axes = plt.subplots(3, 1, sharex=True, figsize=(15, 10))
            list_figs.append(fig)

            # - touchscreen data
            strokes_touch = dat["strokes_touch"]

            if len(strokes_touch)>0:
                strokes_xyt = np.concatenate(strokes_touch)
                t = strokes_xyt[:,2]
                x = strokes_xyt[:,0]
                y = strokes_xyt[:,1]
                axes[0].plot(t, x, 'xk', label="x")
                axes[0].plot(t, y, 'xr', label="y")
                axes[0].legend()
                axes[2].plot(t, x, 'xk', label="x")
                axes[2].plot(t, y, 'xr', label="y")
                axes[2].legend()

            # - video data
            t = pts_cam[:,3]
            x = pts_cam[:,0]
            y = pts_cam[:,1]
            z = pts_cam[:,2]

            indax = 1
            axes[indax].plot(t, x, 'xk', label="x")
            axes[indax].plot(t, y, 'xr', label="y")
            # axes[indax].plot(t, z, 'xb', label="z")
            axes[indax].plot(t, 5*z, 'xb', label="5*z")
            axes[indax].axhline(0)
            axes[indax].legend()

            indax = 0
            axes[indax].plot(t, x, 'xk', label="x")
            axes[indax].plot(t, y, 'xr', label="y")
            # axes[indax].plot(t, z, 'xb', label="z")
            axes[indax].plot(t, 5*z, 'xb', label="5*z")

            axes[indax].axhline(0)
            axes[indax].legend()

            indax = 2
            # axes[indax].plot(t, x, 'xk', label="x")
            # axes[indax].plot(t, y, 'xr', label="y")
            # # axes[indax].plot(t, z, 'xb', label="z")
            # axes[indax].plot(t, 5*z, 'xb', label="5*z")

            axes[indax].axhline(0)
            axes[indax].legend()


            ########
            titles = ["strokes_touch", "strokes_cam", "gaps_cam"]
            D = Dataset([])

            # Plot all
            keys_to_plot = []
            list_strokes_plot = ['strokes_cam', 'reg_strokes_cam' 'gaps_cam', 'strokes_touch', 'strokes_touch', 'strokes_touch']
            titles =  ['strokes_cam', 'gaps_cam', 'strokes_touch', 'strokes-strokes', 'strokes-gap']
            list_strokes = [datall[k] for k in list_strokes_plot]
            list_strokes.insert(0, [])
            titles =  ['strokes_task', 'strokes_cam', 'gaps_cam', 'strokes_touch', 'strokes-strokes', 'strokes-gap']
            fig, axes = D.plotMultStrokes(list_strokes, titles=titles, SIZE=5, ncols=4)
            list_figs.append(fig)

            ax = axes.flatten()[0]
            plotDatStrokes(dat["strokes_task"], ax, clean_task=True)
            
            # - overlay cam on top of touch
            ax = axes.flatten()[3]
            pts = np.concatenate(datall["strokes_cam"])
            ax.plot(pts[:,0], pts[:,1], 'xk')

            ax = axes.flatten()[4]
            pts = np.concatenate(datall["gaps_cam"])
            ax.plot(pts[:,0], pts[:,1], '.r')
            ax.plot(pts[0,0], pts[0,1], 'or')

            ########## COLORED PLOTS
            list_strokes = []
            list_strokes_values = []

            # video only (time)
            strokes = [dat["pts_time_cam_all"]]
            strokes_values = [s[:,3] for s in strokes]
            list_strokes.append(strokes)
            list_strokes_values.append(strokes_values)

            # video only (z)
            list_strokes.append([dat["pts_time_cam_all"]])
            list_strokes_values.append([s[:,2] for s in strokes])

            # video only (gaps, time)
            list_strokes.append(dat["gaps_cam"])
            list_strokes_values.append([d[:,3] for d in dat["gaps_cam"]])

            # video only (gaps, z)
            list_strokes.append(dat["gaps_cam"])
            list_strokes_values.append([d[:,2] for d in dat["gaps_cam"]])

            titles = ["video_time", "video_z", "gaps_time", "gaps_z"]
            fig, axes = D.plotMultStrokesColorMap(list_strokes, list_strokes_values, 
                titles=titles, SIZE=5, ncols=4)
            list_figs.append(fig)

            strokes = dat["strokes_touch"]
            for ind in [2,3]:
                ax = axes.flatten()[ind]
            #     plotDatStrokes(strokes, ax, clean_ordered_ordinal=True)
                # for s in strokes:
                #     ax.plot(s[:,0], s[:,1], 'r')
                plotDatStrokes(strokes, ax, clean_unordered=True)            

            # z coordinates
            fig, ax = plt.subplots(1,1)
            list_figs.append(fig)

            strokes_cam = datall["strokes_cam"]
            gaps_cam = datall["gaps_cam"]

            pts_strokes_cam = np.concatenate(strokes_cam)
            z_strokes = pts_strokes_cam[:,2]
            pts_gaps_cam = np.concatenate(gaps_cam)
            z_gaps = pts_gaps_cam[:,2]

            vals = np.r_[z_strokes, z_gaps]
            xbins = np.linspace(0, max(vals), 60)
            ax.hist(z_strokes, xbins, label="z_strokes", alpha=0.5)
            ax.hist(z_gaps, xbins, label="z_gaps", alpha=0.5)
            ax.legend()
            ax.set_xlabel("z")
            ax.set_title("z is close to 0 during touch")
        else:
            list_figs = []

        return datall, list_figs

    def convert_trialnums(self, trial_ml2=None, trial_campy=None):
        """ 
        Helper to return the ind for either campy(vid) or ml2, given the other.
        Must pass in one of trial_ml2 or trial_campy.
        PARAMS
        - trial_ml2, int, 1, 2, ...
        - trial_campy, int, 0, 1, ...
        RETURNS
        - the other ind.
        """
        
        trial_diff = self.IndFirst["vid"] - self.IndFirst["ml2"]
        # trial_diff = trial_map[1] - trial_map[0]
        
        if trial_ml2 is None:
            assert trial_campy is not None
            out = trial_campy - trial_diff
        else:
            assert trial_campy is None
            out = trial_ml2 + trial_diff

        return out

    def load_campy_data(self, trial_ml2, fail_if_skip=True, return_empty_if_skip=False):
        """ Helper to load pre-extracted and saved campy data.
        And to pull out this single trial.
        RETURNS:
        - dataframe, each row a single frame.
        """
        sdir = f"{SAVEDIR}/{self.Date}_{self.Expt}/behavior/extracted_campy_data/dat.pkl"
        df_campy = pd.read_pickle(sdir)

        # flatten by extracting campy trametimes
        df_campy = self._getTrialsCampyData(trial_ml2, df_campy, fail_if_skip, 
            return_empty_if_skip=return_empty_if_skip)
        if len(df_campy)==0:
            return []

        # Flatten, extracting campy data
        list_campy = df_campy["dat_campy"].tolist()
        list_frametimes = [x["frametimes_sincefirstframe"] for x in list_campy]
        list_framenums = [x["framenums_session"] for x in list_campy]
        assert [x["trial0"] for x in list_campy] == df_campy["trialnum0"].tolist()

        df_campy["campy_frametimes"] = list_frametimes
        df_campy["campy_framenums"] = list_framenums
        del df_campy["dat_campy"]

        return df_campy


    def _getTrialsCampyData(self, trial_ml2, df_campy, fail_if_skip,
        return_empty_if_skip=False):
        """
        Get campy data for this trial,.
        INPUTS:
        - fail_if_skip, then fails if you ask for a trial that in campy was marked
        as fail (which means it failed sanity tests for frames, etc). Note: this is 
        not the same thing as likelihood being low...    
        - return_empty_if_skip, then overwrites fail_if_skip. returns [] if skip.
        """

        # convert trialnum
        t_vid = self.convert_trialnums(trial_ml2 = trial_ml2)
        
        # get all cameras for this trial
        dfthis = df_campy[df_campy["trialnum0"]==t_vid]
        
        # if any failed, then abort
        if return_empty_if_skip:
            if any(dfthis["SKIP"]==True):
                return []
        elif fail_if_skip:
            if any(dfthis["SKIP"]==True):
                print(dfthis)
                assert False, "skip is true.."
                
        # Return 
        return dfthis


    def get_trials_all_data(self, trial_ml2, bodypart = "fingertip", ncams=3, 
            filter_by_likeli_thresh=False, thresh_likeli=0.8, return_empty_if_skip=True):
        """ [Get all data] Helper, get pre-extracted DLC pts, along with other things,
        like likehood, etc.
        Also gets campy frametimes.
        PARAMS:
        - trial (in ml2, so 1, 2, 3, ..)
        - filter_by_likeli_thresh, then dfall will be filtered to only keep
        frames that have all cameras likelis above thresh_likeli. the other outputs will not be filtered.
        - return_empty_if_skip, then returns Nones for everuthing if dont hvae campy data.
        RETURNS:
        - frametimes_mean, array of times in seconds,
        frametime mean over all cameras, which may have some delay rel each otherl, from campy
        - pts, 3d pts for each time. (N, 3), where columns are x, y, (Triangulated DLC data)
        - camdict, where keys are camera names, and values are df holding each timestamp inforamtion
        (2d pts for each cam, and likelihoods), DLC data.
        NOTE:
        - must have already extracted and saved DLC analyses. see videomonkey stuff.
        """
        from pythonlib.tools.expttools import findPath

        print("TODO: Get accurate frametime after pass in frame extraction to Buttons")
        trial_dlc = self.convert_trialnums(trial_ml2=trial_ml2)
        
        # 1) Load 3d pts.
        path = f"{SAVEDIR}/{self.Date}_{self.Expt}/behavior/extracted_dlc_data/3d-part_{bodypart}-trial_{trial_dlc}-dat.npy"
        pts = np.load(path)
        
        # 2) Load each camera data (e.g.,Likelihood)
        list_path = findPath(f"{SAVEDIR}/{self.Date}_{self.Expt}/behavior/extracted_dlc_data", [["camera", f"trial_{trial_dlc}-", "dat"]], None)
        if len(list_path)!=ncams:
            print(list_path)
            assert False

        def _get_cam(path):
            """ return string, name of this camera, etracted from path"""
            ind1 = path.find("camera_")
            ind2 = path.find("_-trial")
            return path[ind1+7:ind2]

        def _load(path):
            """ load fd data from this path"""
            with open(path, "rb") as f:
                dat = pd.read_pickle(f)
            return dat

        # extract once for each camera
        camdict = {}
        for p in list_path:
            cam = _get_cam(p)
            camdict[cam] = _load(p)

        # 3) Load Campy data (to get frametimes)
        df_campy = self.load_campy_data(trial_ml2, fail_if_skip=True, 
            return_empty_if_skip=return_empty_if_skip)
        if len(df_campy)==0:
            assert return_empty_if_skip==True, "how else explain?)"
            return None, None, None, None

        frametimes_mean = np.mean(np.stack(df_campy["campy_frametimes"]), 0) # mean frametimes over all cameras
            
        # 3b) Get frametimes relative to trial
        t_intrial = getTrialsCameraFrametimes(self.Fd, trial_ml2)[0]
        assert len(t_intrial)==len(frametimes_mean)


        # 4) Sanity checks - confirm that campy and dlc are aligned
        assert len(frametimes_mean)==len(pts)
        for k, v in camdict.items():
            assert len(frametimes_mean)==len(v)

        # 5) Concatenate data into single dataframe
        indat = {"x":pts[:,0], "y":pts[:,1], "z":pts[:,2], "t":frametimes_mean, "t_trial":t_intrial}
        
        # Get DLC data
        def _extract_col_dat(dat, bp, coord):
            """ 
            - dat, dataframe, e.g, camdict[camname]
            - bp, string, bodypart
            - coord, string, col name
            # bp = "fingertip"
            # coord = "likelihood"
            RETURNS:
            - pd series for this col.
            """
            x = [col for col in dat.columns if col[1]==bp and col[2]==coord]
            assert len(x)==1
            col = x[0]
            return dat[col]

        # Same the minimum likelihood across all cams.
        # get list of cams in alphabetical order
        list_cams = sorted(list(camdict.keys()))
        tmp = []
        for cam in list_cams:
            v = camdict[cam]
            l = _extract_col_dat(v, bodypart, "likelihood")
            tmp.append(l)
        indat["likelihood_min"] = np.min(np.stack(tmp).T, axis=1)
        # for k, v in indat.items():
        #     print(k, v.shape)
        dfall = pd.DataFrame(indat)

        # 6) Filter to only keep if have good likeli scores
        if filter_by_likeli_thresh:
            dfall = dfall[dfall["likelihood_min"]>thresh_likeli]

        # if len(dfall)==0:
        #     print(dfall)
        #     print(pts)
        #     assert False, "no data, after filtering by threshodl"
        
        return dfall, frametimes_mean, pts, camdict


    def coordinate_conversion(self):
        """ Returns useful conversion factors
        """

        ### PARAMETERS
        # convert pts to actual size
        screen_width = 0.227
        screen_height = 0.302
        # diam_m = 0.381 # (15 inch)
        diam_m = (screen_height**2 + screen_width**2)**0.5 # (15 inch)
        # print(diam_m)
        diam_pix = (1024**2 + 768**2)**0.5
        # diam_inch = 15
        
        deltat = 0.006 # seconds, delay between trigger and frame

        # conversion_pix_inch = diam_pix / diam_inch
        conversion_pix_m = diam_pix / diam_m


        out = {
            "pix_over_m":conversion_pix_m
        }

        return out


    def convert_coords(self, dfall):
        """ Convert coordinates to pixels, same as touchscreen data.
        input coords: depends on what axes used in calibration (easywand). Usually is
        top-left is 0,0, and x increases right, y increases going down.
        Output coords: center is (0,0).
        RETURNS:
        - modified dfall
        """

        ### PARAMETERS
        # convert pts to actual size
        screen_width = 0.227
        screen_height = 0.302
        # diam_m = 0.381 # (15 inch)
        diam_m = (screen_height**2 + screen_width**2)**0.5 # (15 inch)
        # print(diam_m)
        diam_pix = (1024**2 + 768**2)**0.5
        # diam_inch = 15
        
        deltat = 0.006 # seconds, delay between trigger and frame

        # conversion_pix_inch = diam_pix / diam_inch
        conversion_pix_m = diam_pix / diam_m

        # convert time, since frames are actually like 5-10msec after the onset of the trigger
        dfall["t"] = dfall["t"] + deltat
        dfall["t_trial"] = dfall["t_trial"] + deltat

        # - paired pts in screen space vs. 3d grid space.
        # location of origin, in different coord systems
        # all in m


        if True: # CAGETEST2
            # # xscreen = 0.001 + 0.01 + "distanavec grom top left grid bar to top right" # from left-most edge of lcd to origin (top right bar)
            # xscreen = 0.001 + 0.01 + 0.1778 # from left-most edge of lcd to origin (top right bar)
            # # xscreen = 0.227 - (0.001 + 0.013 - 0.002)
            # yscreen = 0.279 - 0.029 # # from bottom of LCD screen to top right rod (origin)
            # yscreen = 0.043 + 0.254 "duist from bottom to top rods"
            # zscreen = 2.5inch # height of rod, relative to bottom surface of 3dgrid.

            # convert from 3d grid coord systemt (origin at top-right rod), x inxreases to left, z is at bar tip, and increases as go towards creen
            # to frame coord system (see below)
            xdist = 0.202 # x distance from left edge of 3d grid to orign
            ydist = 0.278 # y distance from bottom of 3d grid to origin.
            zdist = 0.0635 # height of rod, relative to bottom surface of 3dgrid.

            xframe = xdist - dfall["x"]
            yframe = ydist - dfall["y"]
            zframe = zdist - dfall["z"] 

            # convert from "frame" coord system (origin at bottom left, x incresaed to right, z 0 is scren surface) to screen
            xscreen = xframe - 0.001
            yscreen = yframe - 0.03
            zscreen = zframe

            # cagetest2
            x = xscreen
            y = yscreen
            z = zscreen

        else: # camtest5
            # z distance between origin and the screen.
            z_dist = 0.0067 # thickness of grid.

            # 0) 
            # 1) Convert to origin is bottom left, and xy going right up
            a = -0.005 # see evernote. distance from edge of grid to edge of black screen. (in m)
            b = 0.03
            x = dfall["x"] + 0.02413 - a
            y = 0.2794 - dfall["y"] - b
            z = dfall["z"] + z_dist

        # 2) convert to units of pixels
        x = x * conversion_pix_m
        y = y * conversion_pix_m
        z = z * conversion_pix_m

        # 2) Centerize
        center = [768/2, 1024/2]
        x = x-center[0]
        y = y-center[1]

        # Replace values
        dfall["x"] = x
        dfall["y"] = y
        dfall["z"] = z
        
        return dfall


    def plot_overview(self, dfall, trial_ml2):
        """
        Plots for this trial
        """
        assert False, "use process_data_singletrial - all here should have already been moved over there"

        from .utils import getTrialsStrokes

        # 1) Plot each coordinate timecourse, seaprate plots
        nplots = len(dfall.columns)
        fig, axes = plt.subplots(nplots ,1, sharex=True, figsize=(10, nplots*4))
        for i, col in enumerate(dfall.columns):
            ax = axes.flatten()[i]
            ax.plot(dfall["t"], dfall[col])
            ax.set_ylabel(col)
            ax.set_xlabel("time")


        # 2) Histograms of all values
        nplots = len(dfall.columns)
        fig, axes = plt.subplots(nplots ,1, sharex=False, figsize=(7, nplots*2))
        for i, col in enumerate(dfall.columns):
            ax = axes.flatten()[i]
            ax.hist(dfall[col], 50)
            ax.set_ylabel(col)

        if False:
            # Replaced by plots in process_data_singletrial
            # 3) Plot in 2d space
            from pythonlib.drawmodel.strokePlots import plotDatStrokes, plotSketchpad
            from .plots import plotTrialSimple, plotGetBlankCanvas
            fig, axes = plt.subplots(2,2, sharex=True, sharey=True, figsize=(12, 20))

            # a) Plot touchscreen alone
            ax = axes.flatten()[0]
            plotTrialSimple(self.Fd, trial_ml2, ax=ax, clean=True)
            ax.set_title("ml2 only")

            # b) Plot video alone
            ax = axes.flatten()[1]
            ax = plotGetBlankCanvas(self.Fd, ax=ax)
            strokes = [np.stack([dfall["x"], dfall["y"], dfall["t_trial"]]).T]
            plotDatStrokes(strokes, ax, "raw", markersize=20)
            ax.set_title("video only")

            # b) Plot video alone (color by z)
            ax = axes.flatten()[2]
            ax = plotGetBlankCanvas(self.Fd, ax=ax)
            strokes = [np.stack([dfall["x"], dfall["y"], dfall["t_trial"]]).T]
            zmax = np.percentile(dfall["z"], [80])[0]
            ax.scatter(dfall["x"], dfall["y"], c=dfall["z"], vmax=zmax)
            ax.set_title(f"video only (color by z, max = {zmax})")

            # c) Plot overlaid
            ax = axes.flatten()[3]
            plotTrialSimple(self.Fd, trial_ml2, ax=ax, clean=True)
            plotDatStrokes(strokes, ax, "raw", markersize=10)
            ax.set_title("overlay ml2 and video")

        # 4) Overlay on touchscreen data
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(20, 10))

        # - touchscreen data
        # strokes = getTrialsStrokes(self.Fd, trial_ml2)
        strokes = getTrialsStrokes(self.Fd, trial_ml2, window_rel_go_reward = [-0.1, 0.1])

        if len(strokes)>0:
            strokes_xyt = np.concatenate(strokes)
            t = strokes_xyt[:,2]
            x = strokes_xyt[:,0]
            y = strokes_xyt[:,1]
            axes[0].plot(t, x, 'xk', label="x")
            axes[0].plot(t, y, 'xr', label="y")
            axes[0].legend()

        # - video data
        t = dfall["t_trial"]
        x = dfall["x"]
        y = dfall["y"]
        z = dfall["z"]

        # t = xyzlt[:,4]
        # plt.figure()
        # plt.plot(t, x, '-k', label="x")
        # plt.plot(t, y, '-r', label="y")
        # # axes[indax].plot(t, pts[:,2], '-b', label="z")

        indax = 1
        axes[indax].plot(t, x, 'xk', label="x")
        axes[indax].plot(t, y, 'xr', label="y")
        axes[indax].plot(t, z, 'xb', label="z")
        axes[indax].plot(t, 5*z, 'xm', label="5*z")
        axes[indax].axhline(0)
        axes[indax].legend()

        indax = 0
        axes[indax].plot(t, x, 'xk', label="x")
        axes[indax].plot(t, y, 'xr', label="y")
        axes[indax].plot(t, z, 'xb', label="z")
        axes[indax].plot(t, 5*z, 'xm', label="5*z")

        axes[indax].axhline(0)
        axes[indax].legend()


    #################### ANALYSES
    def analy_compute_errors(self, trial_ml2, ploton=False):
        """ compute ptwise errors, only for timepoints where touchscreen says there
        is pt. interpolates cam to match timepoints exactly.
        PARAMS:
        - trial_ml2
        RETURNS:
        - list of distances, pt by pt.
        - mean dist.
        - fig
        NOTE:
        - in units of meter
        """

        # from pythonlib.tools.distfunctools import distStrok

        # Compute error
        dat, _ = self.process_data_singletrial(trial_ml2, filter_by_likeli_thresh=True,
            return_in_meters=True, ncams=NCAMS)
        strokes_ml2 = dat["strokes_touch"]
        strokes_cam = dat["strokes_cam_interp"]

        pts_ml2 = np.concatenate(strokes_ml2)[:,:2]
        pts_cam = np.concatenate(strokes_cam)[:,:2]

        assert pts_ml2.shape==pts_cam.shape

        # distStrok(pts_ml2, pts_cam, auto_interpolate_if_needed=False)

        # d = 0
        # for p1, p2 in zip(pts_ml2, pts_cam):
        #     d+=np.linalg.norm(p1-p2)
        # d = d/len(pts_ml2)

        # what about distribution of errors
        list_dists = [np.linalg.norm(p1-p2) for p1, p2 in zip(pts_ml2, pts_cam)]

        if ploton:
            fig, ax = plt.subplots(1,1)
            ax.hist(list_dists)
            ax.set_xlabel('distances, pt by pt (m)')
            ax.axvline(np.mean(list_dists), color='r')
        else:
            fig = None

        return list_dists, np.mean(list_dists), fig


############################## VIDEOCAMERA STUFF
def getTrialsCameraFrametimes(fd, trial, chan="Btn1", thresh = 0.5):
    """ Get times of frames onset and offset, assuming frames 
    entered as Button1, (i..e, the trigger signal sent to camera
    is also routed into Button1)
    RETURNS:
    - ons_sec, offs_sec, times, relative to trial onset.
    ons is time of the first frame that passes thresh. offs is time of 
    first frame that returns to 0.
    NOTE: Generically, can use to extract times for any digital signal 
    """
    from .utils import getTrialsAnalogData

    # frames
    dat_frames = getTrialsAnalogData(fd, trial, chan)

    # take analog input (frametimes) and convert to ons and offs
    # -- Statitsics of frame data
    # get crossings
    v = dat_frames[:,0] # voltage
    t = dat_frames[:,1]

    # shgould already be thresholded, by do this anyway so that no numerical errors.
    
    v[v>=thresh] = 1
    v[v<thresh] = 0
    vdiff = np.diff(v)

    # Get onsets, offsets
    ons = np.where(vdiff==1)[0]+1 # plus 1, sinec want first frame of new state.
    offs = np.where(vdiff==-1)[0]+1 
    assert len(ons)==len(offs), "clipped?"

    # Convert from indices to times.
    ons_sec = t[ons]
    offs_sec = t[offs]
    
    return ons_sec, offs_sec


