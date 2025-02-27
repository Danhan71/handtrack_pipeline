# functions for checking alignment of camera times/trial/touch screent imes

import os
from pythonlib.tools.stroketools import *
from pyvm.tools.preprocess import *
from pyvm.classes.videoclass import Videos
from pythonlib.tools.expttools import load_yaml_config
from pyvm.globals import BASEDIR
from drawmonkey.tools.handtrack import HandTrack, getTrialsCameraFrametimes
from pyvm.utils.directories import get_metadata
import argparse
import matplotlib.pyplot as plt
import pickle
import argparse
import os
import shutil

def plot_alignment_data(lags, good_inds):
    """ Function to generate plots on alignment/lag data

    Args:
        lags (dict): trial keyed dict with list of lags (one for each stroke) for corr method {trial:[lags],trial:[lags]}
        good_inds (list): trials and strokes with good corr plots. Should be a list fo strings formatted like:
            ['trial-stroke', ... ,'trial-stroke'] e.g. ['10-0','12-1','20-0']
            * Uses the same labelling scheme as the get_lags function in stroketools thay is called below
    """
    plt.style.use('dark_background')
    all_corr_lags = []
    for trial,lags in lags.items():
        all_corr_lags.extend([c[0]-c[1] for c in lags if c is not None])
    corr_lag_nums = []
    for index in good_inds:
        trial = int(index.split('-')[0])
        stroke = int(index.split('-')[1])
        this_lag = lags[trial][stroke]
        if this_lag is None:
            continue
        lag_num = this_lag[0]-this_lag[1]
        corr_lag_nums.append(lag_num)
        print(index,lag_num)
    euc_lag_nums = []
    # for index in good_inds:
    #     trial = int(index.split('-')[0])
    #     stroke = int(index.split('-')[1])
    #     this_lag = euc_lags[trial][stroke]
    #     if this_lag is None:
    #         continue
    #     lag_num = this_lag[0]-this_lag[1]
    #     euc_lag_nums.append(lag_num)
    #     # print(index,lag_num)
    bins = 30
    fig,ax = plt.subplots(1,4,fig_size=(10,15))
    ax[0].hist(all_corr_lags, bins=bins, color='indianred', alpha=0.5, label='corr lag')
    ax[0].set_xlabel('Lag times (pos means touch_t0 > cam_t0)')
    ax[0].set_title('All lags')
    # plt.hist(all_euc_lags, bins=bins, color='lightgreen', alpha=0.5, label='euc lag')
    ax[1].hist(corr_lag_nums, bins=bins, color='indianred', alpha=0.5, label='corr lag')
    ax[1].set_title('Good inds lags')
    # plt.hist(euc_lag_nums, bins=bins, color='lightgreen', alpha=0.5, label='euc lag')

    corr_mean = round(np.mean(corr_lag_nums),4)
    # euc_mean = round(np.mean(euc_lag_nums),4)
    # plt.boxplot([corr_lag_nums,euc_lag_nums], label=[f'corr lag {corr_mean}', f'euc lag {euc_mean}'])
    ax[2].boxplot(corr_lag_nums)
    ax[2].set_title(f'Good inds lags boxplot, mean: {corr_mean}')
    ax[3].plot(all_corr_lags,'.-')
    ax[3].set_title('Lags over trials')
    return fig

if __name__ == "__main__":
    import traceback
      
    parser = argparse.ArgumentParser(description="Final Plots etc")
    parser.add_argument("name", type=str, help="Experiment name/date")
    parser.add_argument("--animal", type=str, help="Animal name", default=100000)
    parser.add_argument("--sdir",type=str, help="sdir for data and plots generated for this step")
    parser.add_argument("--tstart", type=int, default=10)
    parser.add_argument("--tend", type=int, default=110)
    parser.add_argument('--plot',action='store_true', help="Do plot step, if set have data file called good inds filled out")

    args = parser.parse_args()

    sdir = args.sdir
    animal = args.animal
    name = args.name
    expt_info_list = name.split('_')
    date = str(expt_info_list[0])
    expt = []
    for i in range(1,len(expt_info_list)):
        if not expt_info_list[i].isdigit():
            expt.append(expt_info_list[i])
    expt = '_'.join(expt)
    if len(expt_info_list) >= 3:
        sess = expt_info_list[-1]
        sess_print = "_" + sess
    else:
        sess = 1
        sess_print = ""
    expt_with_sess = f"{expt}{sess_print}"
    tstart = int(args.tstart)
    tend = int(args.tend)
    plot = args.plot
    ind1_vid = 0
    ind1_ml2 = 1
    trange = range(tstart,tend)
    os.makedirs(sdir, exist_ok=True)
    if os.path.exists(f'{sdir}/ht_proc_data.pkl'):
        print('Processed data already extracted, loading data file and skipping to number crunching')
        with open(f'{sdir}/ht_proc_data.pkl','rb') as f:
            trial_dats = pickle.load(f)
    else:
        fd = loadSingleDataQuick(animal, date, expt, sess)
        fd['params']['sample_rate'] = fd['params']['sample_rate'][0]
        HT = HandTrack(ind1_vid, ind1_ml2, fd, sess_print=sess_print, animal=animal, date=date, expt=expt)
        HT.load_campy_data(ind1_ml2)

        
        #LOAD AND PROCESS DATA
        trial_dats = {}
        HT.fit_regression(trange)
        for trial_ml2 in trange:
            try:
                dat, dict_figs, dict_reg_figs, all_day_figs = HT.process_data_wrapper(trial_ml2, ploton=False, \
                                                                    finger_raise_time=0.0, ts_cam_offset=0.0, aggregate=False)
            except Exception as e:
                print(traceback.format_exc())
                dat = {}
                print(f'Skipping trial {trial_ml2}') 
            if len(dat) == 0:
                continue 
            if isinstance(fd['params']['sample_rate'],np.ndarray):
                fd['params']['sample_rate'] = fd['params']['sample_rate'][0]
            for coefs in dat.keys():
                dat[coefs]['peanut_strokes'] = getTrialsStrokesByPeanuts(fd,trial_ml2)
            trial_dats[trial_ml2] = dat
        #saves extracted data to save time if need to run the function again
        with open(f'{sdir}/ht_proc_data.pkl', 'wb') as f:
            pickle.dump(trial_dats,f)
        

    #DO CORR ANALYSIS
    assert len(trial_dats) > 0, 'No trial dat in the dict'
    coefs_list = []
    for dat in trial_dats.values():
        if len(dat) == 0:
            continue
        if len(dat.keys()) > 0:
            coefs_list = dat.keys()
            break
    assert len(coefs_list) > 0, 'No coeffs found'
    for coefs in coefs_list:
        print(f'Doing lags for coefs {coefs}')
        this_sdir = f'{sdir}/{coefs}'
        if os.path.exists(f'{this_sdir}/lag_data.pkl'):
            print(f'Lag data already exists, plotting = {plot}')
            with open(f'{this_sdir}/lag_data.pkl','rb') as f:
                lags = pickle.load(f)
        else:
            lags={}
            lags['corr_lags'],lags['euc_lags'] = get_lags(trial_dats,this_sdir,coefs,True)
            with open(f'{this_sdir}/lag_data.pkl','wb') as f:
                pickle.dump(lags,f)
        if plot:
            with open(f'{this_sdir}/good_inds.txt','rb') as f:
                good_inds = f.readline().replace(' ','').split(',')
            if len(good_inds) < 10:
                print(f'{len(good_inds)} not enough good inds, please enter more in file {this_sdir}/good_inds.txt')
            else:
                import re
                def is_valid_format(s):
                    return bool(re.fullmatch(r"\d+-\d+", s))
                def validate_list(lst):
                    return all(is_valid_format(item) for item in lst)
                assert validate_list(good_inds), f'At least some elements in {this_sdir}/good_inds are formatted improperly (should be comma sep list of trial-stroke as strs). Check this you fool'
            fig = plot_alignment_data(lags,good_inds)
            fig.savefig(f'{this_sdir}/lag_fig.png')
