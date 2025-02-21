import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pythonlib.tools.stroketools import strokesInterpolate2, strokesFilter, smoothStrokes
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.optimize import minimize_scalar, minimize

plt.style.use('dark_background')

UB = 0.15

def euclidAlign(cam_pts, touch_pts, ploton=False):
    plot_bound_size = 10

    fig, ax = plt.subplots(1,2,figsize=(30,10))
    large_len = len(cam_pts)
    small_len = len(touch_pts)
    cam_pts_xy = cam_pts[:,[0,1]]
    touch_pts_no_time = touch_pts[:,[0,1]]

    min_dist = float('inf')
    best_index = -1

    for i in range(large_len - small_len + 1):
        window = cam_pts_xy[i:i + small_len]
        distances = np.linalg.norm(window - touch_pts_no_time, axis=1)
        total_dist = np.sum(distances)

        if total_dist < min_dist:
            min_dist = total_dist
            best_index = i

    lag = [touch_pts[0,2],cam_pts[best_index,3]]
    lag_adj = lag[0] - lag[1]

    touch_lag_adj = touch_pts[:,2] - lag_adj

    if plot_bound_size <= best_index < len(cam_pts) - plot_bound_size:
        plot_bounds = (best_index-plot_bound_size,best_index+plot_bound_size)
    elif best_index < plot_bound_size and best_index < len(cam_pts) - plot_bound_size:
        plot_bounds = (0, best_index+plot_bound_size)
    elif best_index >= plot_bound_size and best_index > len(cam_pts) - plot_bound_size:
        plot_bounds = (best_index-plot_bound_size,len(cam_pts)-1)
    else:
        plot_bounds = (0,len(cam_pts)-1)

    best_ts = cam_pts[plot_bounds[0]:plot_bounds[1]+small_len,3]

    if ploton:
        ax[0].plot(best_ts,cam_pts[plot_bounds[0]:plot_bounds[1]+small_len,0], '.-', label = 'cam pts')
        # ax[0].plot(cam_pts[:,3],cam_pts[:,0], label='all cam pts')
        ax[0].plot(touch_lag_adj, touch_pts[:,0], '.-', label='touch lag adj')
        ax[0].plot(touch_pts[:,2], touch_pts[:,0], '.-', color='grey', label='raw touch')
        # ax.set_title('Trial:', trial)

        ax[1].plot(best_ts,cam_pts[plot_bounds[0]:plot_bounds[1]+small_len,1], '.-', label='cam pts')
        # ax[1].plot(cam_pts[:,3], cam_pts[:,1], label='all cam pts')
        ax[1].plot(touch_lag_adj,touch_pts[:,1], '.-', label='touch lag adj')
        ax[1].plot(touch_pts[:,2],touch_pts[:,1], '.-', color='grey', label='raw touch')
        plt.legend()

    return lag,fig

def corrAlign(cam_pts, touch_pts, ploton=False):
    plot_bound_size = 0

    fig, ax = plt.subplots(2,2,figsize=(40,20), sharex=True)
    large_len = len(cam_pts)
    small_len = len(touch_pts)
    
    touch_pts_norm = touch_pts[:,[0,1]] - np.mean(touch_pts[:,[0,1]], axis=0)
    # touch_pts_norm = np.divide(touch_pts[:,[0,1]],np.max(touch_pts[:,[0,1]],axis=0))
    
    max_sim = 0
    best_index = -1
    
    sim_course = []
    false_alarms = []
    found_good_sim = False
    for i in range(large_len - small_len + 1):
        window = cam_pts[i:i + small_len]
        window_norm = window[:,[0,1]] - np.mean(window[:,[0,1]], axis=0)
        # window_norm = np.divide(window[:,[0,1]],np.max(window[:,[0,1]],axis=0))
        sim = np.einsum('ij,ij->', window_norm, touch_pts_norm)

        this_lag = touch_pts[0,2] - cam_pts[i,2]
        if sim > max_sim and np.abs(this_lag) < UB:
            max_sim = sim
            best_index = i
            found_good_sim = True
        elif sim > max_sim and max_sim != 0:
            false_alarms.append(cam_pts[i,2])
        sim_course.append((cam_pts[i,2],sim))
    
    #Only save if good peak found
    sim_course = np.array(sim_course)
    if found_good_sim:
        lag = [touch_pts[0,2],cam_pts[best_index,2]]
        lag_adj = lag[0] - lag[1]
    else:
        return None,None
    
    left_peak = False
    right_peak = False
    for i,s in enumerate(sim_course):
        if s[0] == cam_pts[best_index,2]:
            if i > 5:
                left_peak = np.all(sim_course[i-5:i,1] < max_sim)
            else:
                left_peak = np.all(sim_course[:i,1] < max_sim)
            if len(sim_course) > i+5:
                right_peak = np.all(sim_course[i+1:i+6,1] < max_sim)
            else:
                right_peak = np.all(sim_course[i+1:,1] < max_sim)
    if not (left_peak and right_peak):
        return None,None

    touch_lag_adj = touch_pts[:,2] - lag_adj

    if plot_bound_size <= best_index < len(cam_pts) - plot_bound_size:
        plot_bounds = (best_index-plot_bound_size,best_index+plot_bound_size)
    elif best_index < plot_bound_size and best_index < len(cam_pts) - plot_bound_size:
        plot_bounds = (0, best_index+plot_bound_size)
    elif best_index >= plot_bound_size and best_index > len(cam_pts) - plot_bound_size:
        plot_bounds = (best_index-plot_bound_size,len(cam_pts)-1)
    else:
        plot_bounds = (0,len(cam_pts)-1)

    best_ts = cam_pts[plot_bounds[0]:plot_bounds[1]+small_len,2]

    if ploton:
        ax[0,0].plot(best_ts,cam_pts[plot_bounds[0]:plot_bounds[1]+small_len,0], '.-', label = 'cam pts y')
        # ax[0].plot(cam_pts[:,3],cam_pts[:,0], label='all cam pts')
        ax[0,0].plot(touch_lag_adj, touch_pts[:,0], '.-', label='touch lag adj')
        ax[0,0].plot(touch_pts[:,2], touch_pts[:,0], '.-', color='grey', label='raw touch', alpha=0.5)
        ax[0,0].legend()
        # ax.set_title('Trial:', trial)

        ax[0,1].plot(best_ts,cam_pts[plot_bounds[0]:plot_bounds[1]+small_len,1], '.-', label='cam pts x')
        # ax[1].plot(cam_pts[:,3], cam_pts[:,1], label='all cam pts')
        ax[0,1].plot(touch_lag_adj,touch_pts[:,1], '.-', label='touch lag adj')
        ax[0,1].plot(touch_pts[:,2],touch_pts[:,1], '.-', color='grey', label='raw touch',alpha=0.5)
        ax[0,1].legend()

        ax[1,0].plot(best_ts,cam_pts[plot_bounds[0]:plot_bounds[1]+small_len,0], '.-', color = 'indianred')
        # ax[0].plot(cam_pts[:,3],cam_pts[:,0], label='all cam pts')
        # ax[1,0].plot(touch_lag_adj, touch_pts[:,0], '.-', label='touch lag adj')
        ax[1,0].plot(touch_pts[:,2], touch_pts[:,0], '.-', color='indianred', label='raw touch x', alpha=0.5)
        # ax.set_title('Trial:', trial)

        ax[1,0].plot(best_ts,cam_pts[plot_bounds[0]:plot_bounds[1]+small_len,1], '.-', color = 'lightgreen')
        # ax[1].plot(cam_pts[:,3], cam_pts[:,1], label='all cam pts')
        # ax[1,0].plot(touch_lag_adj,touch_pts[:,1], '.-', label='touch lag adj')
        ax[1,0].plot(touch_pts[:,2],touch_pts[:,1], '.-', color='lightgreen', label='raw touch y',alpha=0.5)
        
        ax[1,0].plot(cam_pts[:,2], cam_pts[:,0], label='x coord')
        ax[1,0].plot(cam_pts[:,2], cam_pts[:,1], label = 'y coord')
        for p in false_alarms:
            ax[1,0].axvline(p, color='w', zorder=0, alpha = 0.1)
        ax[1,0].legend()

        ax[1,1].plot(*zip(*sim_course))
        plt.axvline(cam_pts[best_index,2], color ='w', linestyle='--')

    return lag,fig

def get_lags(dfs_func, monkey, run):
    euc_lags = {}
    corr_lags = {}
    corr_lags_index = []
    euc_lags_index = []
    index = 0
    import os
    import shutil
    euc_dir = f'/home/danhan/Documents/align_figs/{monkey}{run}/euc'
    corr_dir = f'/home/danhan/Documents/align_figs/{monkey}{run}/corr'
    if os.path.exists(euc_dir):
        shutil.rmtree(euc_dir)
    if os.path.exists(corr_dir):
        shutil.rmtree(corr_dir)
    os.makedirs(euc_dir, exist_ok=True)
    os.makedirs(corr_dir, exist_ok=True)
    for trial, dat in dfs_func.items():
        corr_lags[trial] = []
        euc_lags[trial] = []
        if len(dat) == 0:
            continue
        dat = dat['220914_f12_dlc']
        if len(dat) == 0:
            continue
        cam_pts = dat['pts_time_cam_all']
        trans_cam_pts = dat['trans_pts_time_cam_all']
        strokes_touch = dat['strokes_touch']

        touch_fs = 1/np.mean(np.diff(strokes_touch[0][:,2]))
        cam_fs = 1/np.mean(np.diff(cam_pts[:,3]))
        trans_cam_fs = 1/np.mean(np.diff(trans_cam_pts[:,3]))

        
        t_onfix_off = strokes_touch[0][-1,2]
        t_offfix_on = strokes_touch[-1][0,2]

        # filter data to be within desired times
        all_cam = cam_pts[(cam_pts[:,3] >= t_onfix_off) & (cam_pts[:,3] <= t_offfix_on)]
        trans_all_cam = trans_cam_pts[(trans_cam_pts[:,3] >= t_onfix_off) & (trans_cam_pts[:,3] <= t_offfix_on)]
        
        if len(all_cam) == 0:
            print('Skipping trial:', trial)
            continue

        cam_interp = strokesInterpolate2([all_cam],kind='linear',N=["fsnew",1000,cam_fs])
        cam_interp_smth = smoothStrokes(cam_interp, 1000, window_type='median')[0]
        cam_interp_smth = cam_interp_smth[:,[0,1,3]]

        trans_cam_interp = strokesInterpolate2([trans_all_cam],kind='linear',N=["fsnew",1000,trans_cam_fs])
        trans_cam_interp_smth = smoothStrokes(trans_cam_interp, 1000, window_type='median')[0]
        # trans_cam_interp_smth = trans_cam_interp_smth[:,[0,1,3]]

        touch_interp = strokesInterpolate2(strokes_touch,kind='linear',N=["fsnew",1000,touch_fs])
        touch_interp_noz = []
        for stroke in touch_interp:
            touch_interp_noz.append(stroke[:,[0,1,3]])
        touch_interp_noz = touch_interp_noz[1:-1]

        for i,touch_stroke in enumerate(touch_interp_noz):
            touch_stroke_filt = touch_stroke
            if len(touch_stroke_filt) == 0:
                continue
            euc_lag, euc_fig = euclidAlign(trans_cam_interp_smth,touch_stroke_filt, ploton=True)
            corr_lag, corr_fig = corrAlign(cam_interp_smth,touch_stroke_filt, ploton=True)
            corr_lags[trial].append(corr_lag)
            corr_lags_index.append(corr_lag)
            euc_lags[trial].append(euc_lag)
            euc_lags_index.append(euc_lag)
            if euc_fig is not None:
                euc_fig.savefig(f'/home/danhan/Documents/align_figs/{monkey}{run}/euc/trial{trial}-{i}_euc.png')
            if corr_fig is not None:
                corr_fig.savefig(f'/home/danhan/Documents/align_figs/{monkey}{run}/corr/trial{trial}-{i}_corr.png')
            index = index + 1
            plt.close('all')
    return corr_lags,corr_lags_index,euc_lags,euc_lags_index

def plot_stuff(corr_lags_index, euc_lags_index, good_inds):
    plt.rcParams["figure.figsize"] = (10, 5)
    all_corr_lags_unfilt = [c[0] - c[1] for c in corr_lags_index if c is not None]
    all_corr_lags = [lag for lag in all_corr_lags_unfilt if np.abs(lag) < UB]
    all_euc_lags_unfilt = [c[0] - c[1] for c in euc_lags_index if c is not None]
    all_euc_lags = [lag for lag in all_euc_lags_unfilt if np.abs(lag) < UB]
    corr_lag_nums = [corr_lags_index[i][0]-corr_lags_index[i][1] for i in good_inds if corr_lags_index[i] is not None]
    euc_lag_nums = [euc_lags_index[i][0]-euc_lags_index[i][1] for i in good_inds if euc_lags_index[i] is not None]
    bins = 30
    plt.hist(all_corr_lags, bins=bins, color='indianred', alpha=0.5, label='corr lag')
    # plt.hist(all_euc_lags, bins=bins, color='lightgreen', alpha=0.5, label='euc lag')
    plt.show()
    plt.hist(corr_lag_nums, bins=bins, color='indianred', alpha=0.5, label='corr lag')
    # plt.hist(euc_lag_nums, bins=bins, color='lightgreen', alpha=0.5, label='euc lag')
    plt.legend()
    plt.show()

    corr_mean = round(np.mean(corr_lag_nums),4)
    # euc_mean = round(np.mean(euc_lag_nums),4)
    # plt.boxplot([corr_lag_nums,euc_lag_nums], label=[f'corr lag {corr_mean}', f'euc lag {euc_mean}'])
    plt.boxplot(corr_lag_nums, label=f'corr lag {corr_mean}')
    plt.legend()
    plt.show()
    plt.plot(all_corr_lags,'.-')
    plt.show()
    # plt.plot(all_euc_lags,'.-')
if __name__ == '__main__':      
    #Data loaded here is the dat object that is returned from the process_data_singletrial function in the handtrack class
    dfs = {}
    with open("/home/danhan/freiwaldDrive/dhanuska/230126_pancho_proc_data.pkl", 'rb') as f:
        dfs['pancho1'] = pickle.load(f)
    with open("/home/danhan/freiwaldDrive/dhanuska/240605_pancho_proc_data.pkl", 'rb') as f:
        dfs['pancho2'] = pickle.load(f)
    with open("/home/danhan/freiwaldDrive/dhanuska/231211_diego_proc_data.pkl", 'rb') as f:
        dfs['diego1'] = pickle.load(f)
    with open("/home/danhan/freiwaldDrive/dhanuska/240605_diego_proc_data.pkl", 'rb') as f:
        dfs['diego2'] = pickle.load(f)

    dfs['pancho2'] = {k:v for k,v in dfs['pancho2'].items() if k in list(range(50,551))}
    dfs['diego2'] = {k:v for k,v in dfs['diego2'].items() if k in list(range(50,551))}

    save_df = {}

    for name, df in dfs.items(): 
        this_df = {}
        this_df['corr_lags'],this_df['corr_lags_index'],this_df['euc_lags'],this_df['euc_lags_index'] = get_lags(df,name[:-1],name[-1])
        save_df[name] = this_df

    with open('/home/danhan/Documents/lag_data.pkl','wb') as f:
        pickle.dump(save_df,f)
