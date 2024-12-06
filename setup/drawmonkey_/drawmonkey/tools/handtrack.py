""" to pull in video hand pose tracking data and integrate with ml2 data.
The main goal is the integration of different kinds of data (ml2 touchscreen,
campy frametimes, and dlc XYZ coords.). Then plan is to extract this data, and
do actual analyses in Dataset code.
"""

import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pythonlib.tools.stroketools import strokesInterpolate2
from pyvm.globals import BASEDIR, NCAMS
from .utils import getTrialsIsAbort


ncams = NCAMS

class HandTrack(object):
    """ 
    NOTE:
    - Generally use a new Handtrack Instance for each filedata (fd).
    """
    def __init__(self, ind1_vid, ind1_ml2, fd, date, expt, animal, sess_print, regressor = 0):
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
        self.SessPrint = sess_print
        self.Fd = fd
        self.regressor = regressor
        self.animal = animal
        self.AllDay = {}

        # trial_map = [1, 6] # if [1,6], then ml2 trial 1 is vid6

    def setRegressor(self,reg):
        self.regressor = reg

    def process_data_singletrial(self, trial_ml2, ploton=False, 
            filter_by_likeli_thresh=False, return_in_meters = True, finger_raise_time=0.05,
            ts_cam_offset=0.06, aggregate=False):
        """ Does manythings:
        - Converts cam data into strokes formate, for both strokes and gaps.
        - interpoaltes cam so that matches ml2 timings.
        - gets in units of both pixels and m

        PARAMS:
        - runs for a single trial.
        - ts_cam_offset = Number of seconds the ts lags behind the camera, usually 2-3 frames (0.)
        - aggregate, collect data across whole day for summary plots?

        RETURNS:
        - dict, holding data
        (empty {} if there is no data)
        NOTE:
        - skips if this trial no fixation success
        """

        from .utils import getTrialsStrokes, getTrialsFixationSuccess, getTrialsTaskAsStrokes

        datall = {}

        if not getTrialsFixationSuccess(self.Fd, trial_ml2):
            return {}, [], []
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
             strokes, for defining what is a gap.
             
            RETURNS:
            - DAT, dict holding variations of strokes formatted pts_to_snap
            NOTE:
            - assumes that the last column is time. will use this to do snapping
            - naming reflects that wrote this for cam data.
            - ALSO makes ALlDay attribute whihc tracks data across day. Each row is one trial
                with data in order z-gaps,z-strokes,reg-error
            """
            from scipy.interpolate import interp1d
    
            DAT ={}

            # --- Which are dimensions of time?
            dim_t_1 = strokes_template[0].shape[1]-1
            dim_t_2 = pts_to_snap.shape[1]-1

            # Interpolate video to match touchscreen times
            # t is t from camera reckoning
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
                #tall is from touchscreen reckoning
                t1 = strok[0,dim_t_1]
                t2 = strok[-1,dim_t_1]
                tall = strok[:,dim_t_1]
                
                # - strokes
                inds = (pts_to_snap[:,dim_t_2]>=t1) & (pts_to_snap[:,dim_t_2]<=t2)
                strokes_cam.append(pts_to_snap[inds, :])
                #Hack probalby not safe
                # if tall[-1] > t[-1]:
                #     tall_og = tall
                #     tall=np.array([ta for ta in tall if ta <= t[-1]])
                #     assert (len(tall_og)-len(tall)) < 50, 'too much shit sliced out, previous 3 lines is for cleaning up loose ends not fixing major misalignments' 
                # - strokes, but interpolate to use same timestamps
                if np.any((tall<=t[0]) | (tall>=t[-1])):
                    # print('tall',tall)
                    # print('t',t)
                    # assert False
                    return {}
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
        dfall, t, pts, camdict = self.get_trials_all_data(trial_ml2, filter_by_likeli_thresh=filter_by_likeli_thresh)
        if dfall is None:   
            # failed becuase no campy data
            return {}, [], []

        dfall = self.convert_coords(dfall)
        pts_time_cam_all = dfall[["x", "y", "z", "t_trial"]].values # all times, not just those in ml2 strokes

        # get strokes from onset to touch done
        # strokes = getTrialsStrokesByPeanuts(fd, trial_ml2)
        # strokes = getTrialsStrokesClean(fd, trial_ml2)
        strokes = getTrialsStrokes(self.Fd, trial_ml2, window_rel_go_reward = [-0.1, 0.1])
        strokes_task = getTrialsTaskAsStrokes(self.Fd, trial_ml2)

        if len(pts_time_cam_all)==0:
            return {}, [], []

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
        strokes_lag_adjusted = []
        for strok in strokes:
            lag_adjusted = np.array([[p[0],p[1],p[2]-ts_cam_offset] for p in strok])
            strokes_lag_adjusted.append(lag_adjusted)
        dat = snap_pts_to_strokes(strokes_lag_adjusted, pts_time_cam_all)
        if dat == {}:
            return {},[],[]
        dat["strokes_task"] = strokes_task

        # store this
        for k, v in dat.items():
            datall[k]=v

        # strokes

        # datall["strokes_cam_old"] = datall["strokes_cam"]
        # datall["strokes_cam"] = cam_preds


        # dat = snap_pts_to_strokes(strokes_meters, pts_time_cam_all_meters)
        # for k, v in dat.items():
        #     datall[f"{k}_meters"]=v

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

            # np.save("/home/danhan/pts_cam.npy", pts_cam)

            if len(strokes_touch)>0:
                strokes_xyt = np.concatenate(strokes_touch)
                # np.save("/home/danhan/strokes_xyt.npy", strokes_xyt)
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
            #Hacky way to set cap of 0.3 on ylim so as to not squish too much if outliers exist
            if axes[indax].get_ylim()[1] > 0.3:
                axes[indax].set_ylim((-0.3,0.3))
            axes[indax].legend()

            indax = 0
            axes[indax].plot(t, x, 'xk', label="x")
            axes[indax].plot(t, y, 'xr', label="y")
            # axes[indax].plot(t, z, 'xb', label="z")
            axes[indax].plot(t, 5*z, 'xb', label="5*z")
            if axes[indax].get_ylim()[1] > 0.3:
                axes[indax].set_ylim((-0.3,0.3))
            axes[indax].axhline(0)
            axes[indax].legend()

            indax = 2
            # axes[indax].plot(t, x, 'xk', label="x")
            # axes[indax].plot(t, y, 'xr', label="y")
            # # axes[indax].plot(t, z, 'xb', label="z")
            # axes[indax].plot(t, 5*z, 'xb', label="5*z")

            axes[indax].axhline(0)
            axes[indax].legend()
            # np.save("/home/danhan/Desktop/str_toc1.npy", datall["strokes_touch"][1])
            # np.save("/home/danhan/Desktop/str_cam1.npy", datall["strokes_cam"][1])
            # np.save("/home/danhan/Desktop/str_toc0.npy", datall["strokes_touch"][0])
            # np.save("/home/danhan/Desktop/str_cam0.npy", datall["strokes_cam"][0])


            ########
            titles = ["strokes_touch", "strokes_cam", "gaps_cam"]
            D = Dataset([])

            # Plot all
            keys_to_plot = []
            list_strokes_plot = ['strokes_cam', 'gaps_cam', 'strokes_touch', 'strokes_touch', 'strokes_touch']
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
            # np.savetxt(f'/home/danhan/Documents/hand_track/Pancho/221015_dircolor1/{trial_ml2}strokes_cam.txt',pts)
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
            # assert not np.any(np.isnan(np.concatenate(list_strokes)))
            # video only (z)
            list_strokes.append([dat["pts_time_cam_all"]])
            list_strokes_values.append([s[:,2] for s in strokes])
            # assert not np.any(np.isnan(np.concatenate(list_strokes)))

            # video only (gaps, time)
            list_strokes.append(dat["gaps_cam"])
            list_strokes_values.append([d[:,3] for d in dat["gaps_cam"]])
            # assert not np.any(np.isnan(np.concatenate(list_strokes)))

            # video only (gaps, z)
            list_strokes.append(dat["gaps_cam"])
            list_strokes_values.append([d[:,2] for d in dat["gaps_cam"]])
            # assert not np.any(np.isnan(np.concatenate(list_strokes)))

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

            # self.AllDay[trial_ml2].append(np.array(z_gaps))
            # self.AllDay[trial_ml2].append(np.array(z_strokes))
            # print(self.AllDay[trial_ml2])

            vals = np.r_[z_strokes, z_gaps]
            xbins = np.linspace(min(vals), max(vals), 60)
            ax.hist(z_strokes, xbins, label="z_strokes", alpha=0.5)
            ax.hist(z_gaps, xbins, label="z_gaps", alpha=0.5)
            ax.legend()
            ax.set_xlabel("z")
            ax.set_title("z is close to 0 during touch")
        else:
            list_figs = []
            # self.AllDay[trial_ml2].append([])
            # self.AllDay[trial_ml2].append([])

        list_reg_figs = []

        #Plot the data with the linear reg cam pts

        if self.regressor != 0:
            print("###DOING REGRESSION ON CAM PTS###")
            reg_pts_list = []
            reg_gaps_list = []
            err_list = []
            
            for strok_cam in datall["strokes_cam"]:
                strok_cam_xyz = [(p[0],p[1],p[2]) for p in strok_cam]
                strok_cam_z = [p[2] for p in strok_cam]
                strok_cam_t = [p[3] for p in strok_cam]
                reg_cam_pts = self.regressor.predict(strok_cam_xyz)
                stitch = [(p[0],p[1],z,t) for p,z,t in zip(reg_cam_pts,strok_cam_z,strok_cam_t)]
                reg_pts_list.append(np.array(stitch))

            for gap_cam in datall["gaps_cam"]:
                gap_cam_xyz = [(p[0],p[1],p[2]) for p in gap_cam]
                gap_cam_t = [p[3] for p in gap_cam]
                reg_gap_pts = self.regressor.predict(gap_cam_xyz)
                stitch = [(p[0],p[1],z[2],t) for p,t,z in zip(reg_gap_pts, gap_cam_t,gap_cam_xyz)]
                reg_gaps_list.append(np.array(stitch))

            pts_cam = dat["pts_time_cam_all"]
            pts_cam_xyz = [(p[0],p[1],p[2]) for p in pts_cam]
            pts_cam_t = [p[3] for p in pts_cam]
            reg_pts_cam_xy = self.regressor.predict(pts_cam_xyz)
            reg_pts_cam = np.array([(p[0],p[1],z[2],t) for p,t,z in zip(reg_pts_cam_xy,pts_cam_t,pts_cam_xyz)])



            

            # print("raw pts cam all", dat["pts_time_cam_all"])
            # print("reg pts cam all", reg_pts_cam)

            # print("raw strokes cam", datall["strokes_cam"])
            # print("reg pts list", reg_pts_list)

            # print("raw gap pts", datall["gaps_cam"])
            # print("reg gap pts", reg_gaps_list)
            # assert False

            
            if return_in_meters:
                pts_time_cam_all = reg_pts_cam.copy()
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

            dat["reg_strokes_task"] = strokes_task

            datall["reg_strokes_cam"] = reg_pts_list
            datall["reg_gaps_cam"] = reg_gaps_list
            dat["reg_gaps_cam"] = reg_gaps_list
            dat["reg_pts_time_cam_all"] = reg_pts_cam

            # print(reg_pts_cam)
            # assert False

            if ploton:

                list_reg_figs = []

                # 1) Plot each coordinate timecourse, seaprate plots
                pts_cam = dat["reg_pts_time_cam_all"]
                nplots = pts_cam.shape[1]
                t = pts_cam[:,3]
                fig, axes = plt.subplots(nplots ,1, sharex=True, figsize=(10, nplots*4))
                list_reg_figs.append(fig)
                for i in range(nplots):
                    ax = axes.flatten()[i]
                    ax.plot(t, pts_cam[:,i])
                    ax.set_ylabel(f"dim{i}")
                    ax.set_xlabel("time")

                # 1) Plot each coordinate timecourse, seaprate plots
                pts_cam = dat["reg_pts_time_cam_all"]
                z = pts_cam[:,2]
                fig, axes = plt.subplots(2 ,1, sharex=True, figsize=(10, nplots*4))
                list_reg_figs.append(fig)
                for i in range(2):
                    ax = axes.flatten()[i]
                    ax.plot(pts_cam[:,i], z)
                    ax.set_ylabel("z")
                    ax.set_xlabel(f"dim{i}")

                # 2) Histograms of all values
                fig, axes = plt.subplots(nplots ,1, sharex=False, figsize=(7, nplots*2))
                list_reg_figs.append(fig)
                for i in range(nplots):
                    ax = axes.flatten()[i]
                    ax.hist(pts_cam[:,i], 50)
                    ax.set_ylabel(f"dim{i}")

                # 4) Overlay on touchscreen data
                fig, axes = plt.subplots(3, 1, sharex=True, figsize=(15, 10))
                list_reg_figs.append(fig)

                # - touchscreen data
                strokes_touch = dat["strokes_touch"]

                # np.save("/home/danhan/pts_cam.npy", pts_cam)

                if len(strokes_touch)>0:
                    strokes_xyt = np.concatenate(strokes_touch)
                    # np.save("/home/danhan/strokes_xyt.npy", strokes_xyt)
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
                if axes[indax].get_ylim()[1] > 0.3:
                    axes[indax].set_ylim((-0.3,0.3))
                axes[indax].legend()

                indax = 0
                axes[indax].plot(t, x, 'xk', label="x")
                axes[indax].plot(t, y, 'xr', label="y")
                # axes[indax].plot(t, z, 'xb', label="z")
                axes[indax].plot(t, 5*z, 'xb', label="5*z")
                if axes[indax].get_ylim()[1] > 0.3:
                    axes[indax].set_ylim((-0.3,0.3))


                axes[indax].axhline(0)
                axes[indax].legend()

                indax = 2
                # axes[indax].plot(t, x, 'xk', label="x")
                # axes[indax].plot(t, y, 'xr', label="y")
                # # axes[indax].plot(t, z, 'xb', label="z")
                # axes[indax].plot(t, 5*z, 'xb', label="5*z")

                axes[indax].axhline(0)
                axes[indax].legend()
                # np.save("/home/danhan/Desktop/str_toc1.npy", datall["strokes_touch"][1])
                # np.save("/home/danhan/Desktop/str_cam1.npy", datall["reg_strokes_cam"][1])
                # np.save("/home/danhan/Desktop/str_toc0.npy", datall["strokes_touch"][0])
                # np.save("/home/danhan/Desktop/str_cam0.npy", datall["reg_strokes_cam"][0])


                ########
                titles = ["strokes_touch", "reg_strokes_cam", "reg_gaps_cam"]
                D = Dataset([])

                # Plot all
                keys_to_plot = []
                list_strokes_plot = ['reg_strokes_cam', 'reg_gaps_cam', 'strokes_touch', 'strokes_touch', 'strokes_touch']
                titles =  ['reg_strokes_cam', 'reg_gaps_cam', 'strokes_touch', 'strokes-strokes', 'strokes-gap']
                list_strokes = [datall[k] for k in list_strokes_plot]
                list_strokes.insert(0, [])
                titles =  ['reg_strokes_task', 'reg_strokes_cam', 'reg_gaps_cam', 'strokes_touch', 'strokes-strokes', 'strokes-gap']
                fig, axes = D.plotMultStrokes(list_strokes, titles=titles, SIZE=5, ncols=4)
                list_reg_figs.append(fig)

                ax = axes.flatten()[0]
                plotDatStrokes(dat["reg_strokes_task"], ax, clean_task=True)
                
                # - overlay cam on top of touch
                ax = axes.flatten()[3]
                pts = np.concatenate(datall["reg_strokes_cam"])
                ax.plot(pts[:,0], pts[:,1], 'xk')

                ax = axes.flatten()[4]
                pts = np.concatenate(datall["reg_gaps_cam"])
                ax.plot(pts[:,0], pts[:,1], '.r')
                ax.plot(pts[0,0], pts[0,1], 'or')

                ########## COLORED PLOTS
                list_strokes = []
                list_strokes_values = []

                # video only (time)
                strokes = [dat["reg_pts_time_cam_all"]]
                # print(strokes)
                # assert False
                strokes_values = [s[:,3] for s in strokes]
                list_strokes.append(strokes)
                list_strokes_values.append(strokes_values)

                # video only (z)
                list_strokes.append([dat["pts_time_cam_all"]])
                list_strokes_values.append([s[:,2] for s in strokes])

                # video only (gaps, time)
                list_strokes.append(dat["reg_gaps_cam"])
                list_strokes_values.append([d[:,3] for d in dat["reg_gaps_cam"]])

                # video only (gaps, z)
                list_strokes.append(dat["reg_gaps_cam"])
                list_strokes_values.append([d[:,2] for d in dat["reg_gaps_cam"]])

                titles = ["video_time", "video_z", "gaps_time", "gaps_z"]
                fig, axes = D.plotMultStrokesColorMap(list_strokes, list_strokes_values, 
                    titles=titles, SIZE=5, ncols=4)
                list_reg_figs.append(fig)

                strokes = dat["strokes_touch"]
                for ind in [2,3]:
                    ax = axes.flatten()[ind]
                #     plotDatStrokes(strokes, ax, clean_ordered_ordinal=True)
                    # for s in strokes:
                    #     ax.plot(s[:,0], s[:,1], 'r')
                    plotDatStrokes(strokes, ax, clean_unordered=True)            

                # z coordinates
                fig, ax = plt.subplots(1,1)
                list_reg_figs.append(fig)

                reg_strokes_cam = datall["reg_strokes_cam"]
                reg_gaps_cam = datall["reg_gaps_cam"]

                pts_reg_strokes_cam = np.concatenate(reg_strokes_cam)
                z_strokes = pts_reg_strokes_cam[:,2]
                pts_gaps_cam = np.concatenate(reg_gaps_cam)
                z_gaps = pts_gaps_cam[:,2]

                vals = np.r_[z_strokes, z_gaps]
                xbins = np.linspace(min(vals), max(vals), 60)
                ax.hist(z_strokes, xbins, label="z_strokes", alpha=0.5)
                ax.hist(z_gaps, xbins, label="z_gaps", alpha=0.5)
                ax.legend()
                ax.set_xlabel("z")
                ax.set_title("z is close to 0 during touch")
        else:
            list_reg_figs = []
        assert (list_figs is not None) == (list_reg_figs is not None), "why one none other not?"

        if aggregate:
            if trial_ml2 not in self.AllDay:
                self.AllDay[trial_ml2] = {}
            pts_strokes_cam = np.concatenate(strokes_cam)
            z_strokes = pts_strokes_cam[:,2]
            pts_gaps_cam = np.concatenate(gaps_cam)
            z_gaps = pts_gaps_cam[:,2]
            self.AllDay[trial_ml2]['z_gaps'] = np.array(z_gaps)
            self.AllDay[trial_ml2]['z_strokes'] = np.array(z_strokes)
            if self.regressor != 0:
                cam_all = np.concatenate(datall['strokes_cam'])
                touch_all = np.concatenate(datall['strokes_touch'])
                strok_cam_xyz = [(p[0],p[1],p[2]) for p in cam_all]
                strok_cam_z = [p[2] for p in cam_all]
                strok_cam_t = [p[3] for p in cam_all]
                N = ['input_times']
                N.append(np.array(strok_cam_t))
                strok_touch_annoying = []
                strok_touch_annoying.append(np.array(touch_all))
                touch_interp = strokesInterpolate2(strok_touch_annoying,N)
                touch_interp_xyz = [(p[0],p[1],0) for p in touch_interp[0]]
                stroke_error = self.regressor.score(strok_cam_xyz,touch_interp_xyz)
                self.AllDay[trial_ml2]['reg_errs'] = np.array(stroke_error)

                x_res = [tch[0]-strk[0] for tch,strk in zip(touch_interp,strok_cam_xyz)]
                y_res = [tch[1]-strk[1] for tch,strk in zip(touch_interp,strok_cam_xyz)]
                self.AllDay[trial_ml2]['x_coord'] = np.array([tch[0] for tch in touch_interp])
                self.AllDay[trial_ml2]['y_coord'] = np.array([tch[1] for tch in touch_interp])
                self.AllDay[trial_ml2]['x_res'] = np.array(x_res)
                self.AllDay[trial_ml2]['y_res'] = np.array(y_res)
            else: 
                self.AllDay[trial_ml2]['reg_errs'] = np.array([])

        return (datall, list_figs, list_reg_figs)

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
        sdir = f"{BASEDIR}/{self.animal}/{self.Date}_{self.Expt}{self.SessPrint}/behavior/extracted_campy_data/dat.pkl"
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


    def get_trials_all_data(self, trial_ml2, bodypart = "fingertip", 
            filter_by_likeli_thresh=False, thresh_likeli=0.8, return_empty_if_skip=True, align = True):
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

        EXTRA NOTE:
        Function has been heavily edited to adjust the lengths of the data structres to meet the trial on off times, if you would like to not do have that
        done you can set the align=True flag to false
        """
        from pythonlib.tools.expttools import findPath

        print("TODO: Get accurate frametime after pass in frame extraction to Buttons")
        trial_dlc = self.convert_trialnums(trial_ml2=trial_ml2)
        
        # 1) 3d pts.
        path = f"{BASEDIR}/{self.animal}/{self.Date}_{self.Expt}{self.SessPrint}/behavior/extracted_dlc_data/3d-part_{bodypart}-trial_{trial_dlc}-dat.npy"
        pts_raw = np.load(path)
        
        # 2) Load each camera data (e.g.,Likelihood)
        list_path = findPath(f"{BASEDIR}/{self.animal}/{self.Date}_{self.Expt}{self.SessPrint}/behavior/extracted_dlc_data", [["camera", f"trial_{trial_dlc}-", "dat"]], None)

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

        def align_frametimes(volt_times, cam_times, pts_in, camdict_in):
            """
            Function to align the framteim edata between cameras nand voltage. Volt times are precise to trial on and off 
            while cam time is constant during and after trials. This function slices off the tail end of the cam frames to align.
            Look at notebook called check_frametimes in the /pipeline/setup folder for some plots on what I mean by this.

            PARAMS:
            volt_times, list like of times for voltage onsets (starts at first frame after 0 (~.02s))
            cam_times, frame times for camera (willl be cont. timed through whole trial)

            RETURN:
            volt_align and cam_align times, will not be zero indexed as the code does not necessarily require that. 
            May zero index in the future if there a re problems.
            """

            #make pd.series
            volt_times = pd.Series(volt_times)
            cam_times= pd.Series(cam_times)

            #slice out the garbage at the beginning of the volt times, assuming only major time difference is at beginning
            volt_diffs = volt_times.diff()
            voltdiff_max = max([d for d in volt_diffs if str(d) != "nan"])
            # assert volt_max > 0.03, "very small jump here, maybe bad trial or soemthing else went wrong?"
            max_ind = 0
            for i in range(len(volt_diffs)):
                if str(volt_diffs[i]) != "nan":
                    if volt_diffs[i] == voltdiff_max:
                        max_ind = i
                        break
            from random import sample
            if voltdiff_max > 0.04:
                assert max_ind < 5, "Comment this out if you are aware of why the first 5+ frames are bad, this is just a check as most seem to be a few frames"
                assert len(volt_times[max_ind:]) <= len(cam_times), "Dont think volt time shsould be longer than cam times"
                volt_align = volt_times[max_ind:]
            else:
                volt_align = volt_times
                assert len(volt_times) <= len(cam_times), "Dont think volt time shsould be longer than cam times"
            # rand_list = sample(range(len(cam_times)), len(volt_align))
            # cam_align = cam_times[rand_list]
            # pts_align = pts_in[rand_list]
            last_ind = len(volt_align)
            if (len(pts_in) < last_ind):
                if getTrialsIsAbort:
                    last_ind = len(pts_in)
                    volt_align = volt_align[:len(pts_in)]
                else:
                    assert False, "Why is DLC data shorter than volt/campy data?"
            
            cam_align = cam_times[:last_ind]
            pts_align = pts_in[:last_ind]
            camdict_align = {}
            for k, v in camdict_in.items():
                v_align = v[:last_ind]
                camdict_align[k] = v_align
            
            return volt_align, cam_align, pts_align, camdict_align

        # extract once for each camera
        camdict_raw = {}
        for p in list_path:
            cam = _get_cam(p)
            camdict_raw[cam] = _load(p)

        # 3) Load Campy data (to get frametimes)
        df_campy = self.load_campy_data(trial_ml2, fail_if_skip=True, 
            return_empty_if_skip=return_empty_if_skip)
        if len(df_campy)==0:
            assert return_empty_if_skip==True, "how else explain?)"
            return None, None, None, None

        frametimes_mean_raw = np.mean(np.stack(df_campy["campy_frametimes"]), 0) # mean frametimes over all cameras
            
        # 3b) Get frametimes relative to trial
        t_intrial_raw = getTrialsCameraFrametimes(self.Fd, trial_ml2)[0]
        
        # align = False
        if align == True:
            t_intrial, frametimes_mean, pts, camdict = align_frametimes(volt_times=t_intrial_raw,\
             cam_times=frametimes_mean_raw, pts_in=pts_raw, camdict_in=camdict_raw)
        else:
            t_intrial, frametimes_mean, pts, camdict = (t_intrial_raw, frametimes_mean_raw, pts_raw, camdict_raw)
        

        assert len(t_intrial)==len(frametimes_mean), f"op1={len(t_intrial)}, op2={len(frametimes_mean)}"

        # 4) Sanity checks - confirm that campy and dlc are aligned
        assert len(frametimes_mean)==len(pts), f"op1={len(frametimes_mean)}, op2={len(pts)}"
        
        for k, v in camdict.items():
            assert len(frametimes_mean)==len(v)

        # 5) Concatenate data into single dataframe
        # print(t_intrial)
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
        # assert False

        dfall = pd.DataFrame(indat, index=range(len(t_intrial)))
        """Account for Na values due to slicing out first few times in volt alignment section, 
        hacky (yet overcomplicated) fix. Was necessary if you look at the plots there are a few pts
        recorded before the actualy trial starts (when cam sends first volt out or soemthing)
        but if it doesnt work then maybe theres a bigger problem, hence why this section only replaces up to 5 pts 
        """
        temp_len=len(dfall['t_trial'])
        for i in range(min(5,temp_len-2)):
            if np.isnan(dfall.loc[i,'t_trial']):
                dfall.loc[i,'t_trial'] = dfall.loc[min(temp_len-1,5),'t_trial'] - (min(temp_len,5)-i)*(dfall.loc[min(temp_len,6),'t_trial']-dfall.loc[min(temp_len-1,5),'t_trial'])
        assert not np.any(np.isnan(dfall['t_trial'])),'problem bigger than the above hack can solve (more than first 5 vals na)'

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
            # xdist = .35
            ydist = 0.279 # y distance from bottom of 3d grid to origin.
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
        dat, _, _ = self.process_data_singletrial(trial_ml2, filter_by_likeli_thresh=True,
            return_in_meters=True)
        if not dat:
            print("No touch screen data found here")
            return None, None, None, None, None, None
        strokes_ml2 = dat["strokes_touch"]
        strokes_cam = dat["strokes_cam_interp"]

        pts_ml2 = np.concatenate(strokes_ml2)[:,:2]
        pts_cam = np.concatenate(strokes_cam)[:,:2]
        

        # assert pts_ml2.shape==pts_cam.shape, print(f'op1={pts_ml2.shape}, op2={pts_cam.shape}')

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

        if self.regressor != 0:
            reg_strokes_cam = dat["reg_strokes_cam"]
            reg_pts_cam = np.concatenate(reg_strokes_cam)[:,:2]
            reg_list_dists = [np.linalg.norm(p1-p2) for p1, p2 in zip(pts_ml2, reg_pts_cam)]
            if ploton:
                reg_fig, ax = plt.subplots(1,1)
                ax.hist(list_dists)
                ax.set_xlabel('distances, pt by pt (m)')
                ax.axvline(np.mean(reg_list_dists), color='r')
                mean_reg_list_dists = np.mean(reg_list_dists)
            else:
                reg_fig = None
        else:
            reg_list_dists = None
            reg_fig = None
            mean_reg_list_dists = None


        return list_dists,reg_list_dists, np.mean(list_dists),mean_reg_list_dists, fig, reg_fig
    def plot_data_all_day(self):
        def separate_outliers(data):
            """Function to seprate outliers from reg data for plotting purposes
            Args:
                data (pd df): data, with indices

            Returns:
                good_dat : dict with good dat arrays indexed by trial
                outliers : dict with outliers arrays indexed by trial
            """
            col = data.columns[0]
            if data[col].values[0].size > 1:
                all_d = np.concatenate(data[col].values)
            else:
                 all_d = data[col].values
            mean = np.mean(all_d)
            sd = np.std(all_d)
            good_dat = {}
            outliers = {}
            for trial,row in data.iterrows():
                d = row[col]
                z_scores = np.abs((d-mean / sd))
                outliers[trial] = d[z_scores >= 50]
                good_dat[trial] = d[z_scores < 50]
                
            return good_dat,outliers
        
        df = pd.DataFrame.from_dict(self.AllDay, orient='index')
        df = df.dropna()
        df = df.map(np.atleast_1d)
        # with open('/home/dhanuska/dhanuska/df.pkl','wb') as f:
        #     pickle.dump(df,f)
        b= 20
        
        disps = pd.DataFrame(df['disp'],index=df.index)
        errs = pd.DataFrame(df['reg_errs'],index=df.index)

        disp_good,disp_out = separate_outliers(disps)
        # err_good,err_out = separate_outliers(errs)
        good_disps = np.concatenate(list(disp_good.values()))
        # good_errs = np.concatenate(list(err_good.values()))

        all_strokes = np.concatenate(df['z_strokes'].values)
        all_gaps = np.concatenate(df['z_gaps'].values)
        all_disps = np.concatenate(df['disp'].values)

        gap_vals = np.r_[all_strokes, all_gaps]
        gap_xbins = np.linspace(min(gap_vals), max(gap_vals), b)

        all_xres = np.concatenate(df['x_res'].values)
        all_yres = np.concatenate(df['y_res'].values)
        all_res = np.concatenate((all_xres,all_yres))
        all_xs = np.concatenate(df['x_coord'].values)
        all_ys = np.concatenate(df['y_coord'].values)


        fig,ax = plt.subplots(ncols=3,nrows=3,figsize=(30,30))
        ax[0][0].hist(all_disps,bins=b,color='blue')
        ax[0][1].hist(good_disps,bins=b,color='blue')
        ax[1][0].scatter(errs.index,errs,color='blue')
        ax[1][2].hist(all_gaps,gap_xbins,color='darkblue',alpha=0.6)
        ax[1][2].hist(all_strokes,gap_xbins,color='darkorange',alpha=0.6)
        ax[1][1].hist(all_res,bins=b)
        ax[2][0].scatter(all_xs,all_xres)
        ax[2][1].scatter(all_ys,all_yres)

        ax[0][0].set_title('Displacements, log(counts)')
        ax[0][0].set_yscale('log')
        ax[0][1].set_title('Displacements, no outliers, log(counts)')
        ax[0][1].set_yscale('log')
        ax[1][0].set_title('Stroke Error')
        ax[1][0].set_xlabel('trial')
        ax[1][2].set_title('z coords blue=gaps oran=strokes')
        ax[1][1].set_title('Hist of all Residuals')
        ax[2][0].set_title('x resid vs x coord')
        ax[2][1].set_title('y resid vs y coord')

        # err_vals = np.concatenate(list(err_out.values()))
        # err_xbins = np.linspace(min(err_vals), max(err_vals), b)

        try:
            disp_vals = np.concatenate(list(disp_out.values()))
            disp_xbins = np.linspace(min(disp_vals), max(disp_vals), b)
        except:
            pass
        else:
            counts,edges,bars = ax[0][2].hist(disp_out.values(),label=disp_out.keys(),bins=disp_xbins,stacked=True)
            color_to_label = {bar[0].get_facecolor(): label for bar, label in zip(bars, disp_out.keys())}        
            for bar_group, dataset_counts in zip(bars, counts):
                for rect in bar_group:
                    # Only label bars with non-zero height
                    if rect.get_height() > 0:
                        # Determine the label based on the color
                        label = color_to_label.get(rect.get_facecolor(), "Unknown")
                        # Calculate the position for the text
                        x_pos = rect.get_x() + rect.get_width() / 2
                        y_pos = rect.get_y() + rect.get_height() / 2
                        # Add the label
                        ax[0][2].text(x_pos, y_pos, label, ha='center', va='center', color='white', fontsize=8)
            ax[0][2].set_title('Displacement Outliers')
            # for k,v in err_out.items():
            #     ax[1][1].scatter(k,v,color='black')

        for i in errs.index:
            ax[1][0].annotate(int(i),(int(i),errs.loc[i]))
        return fig
    
        
    


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

    # should already be thresholded, by do this anyway so that no numerical errors.
    
    v[v>=thresh] = 1
    v[v<thresh] = 0
    vdiff = np.diff(v)

    # Get onsets, offsets
    ons = np.where(vdiff==1)[0]+1 # plus 1, sinec want first frame of new state.
    offs = np.where(vdiff==-1)[0]+1

    #If shifted by one assume that there is just an onset or offset error in the beginning
    size_diff = len(offs) - len(ons)
    if size_diff == 1:
        offs = offs[1:len(offs)]
    elif size_diff == -1:
        ons = ons[0:len(ons)-1]
    else: 
        assert len(ons)==len(offs), f"ONS:{len(ons)}, OFFS:{len(offs)},clipped? Wrong day data?"

    # Convert from indices to times.
    ons_sec = t[ons]
    offs_sec = t[offs]
    
    return ons_sec, offs_sec



# CODE GRAVEYARD

#Use this code if you want to do 2d regression
            # for strok_cam in datall["strokes_cam"]:
            #     strok_cam_xy = [(p[0],p[1]) for p in strok_cam]
            #     strok_cam_zt = [(p[2], p[3]) for p in strok_cam]
            #     reg_cam_pts = self.regressor.predict(strok_cam_xy)
            #     stitch = [np.concatenate((p,q)) for p,q in zip(reg_cam_pts, strok_cam_zt)]
            #     reg_pts_list.append(np.array(stitch))

            # for gap_cam in datall["gaps_cam"]:
            #     gap_cam_xy = [(p[0],p[1]) for p in gap_cam]
            #     gap_cam_zt = [(p[2], p[3]) for p in gap_cam]
            #     reg_gap_pts = self.regressor.predict(gap_cam_xy)
            #     stitch = [np.concatenate((p,q)) for p,q in zip(reg_gap_pts, gap_cam_zt)]
            #     reg_gaps_list.append(np.array(stitch))

            # pts_cam = dat["pts_time_cam_all"]
            # pts_cam_xy = [(p[0],p[1]) for p in pts_cam]
            # pts_cam_zt = [(p[2], p[3]) for p in pts_cam]
            # reg_pts_cam_xy = self.regressor.predict(pts_cam_xy)
            # reg_pts_cam = np.array([np.concatenate((p,q)) for p,q in zip(reg_pts_cam_xy,pts_cam_zt)])