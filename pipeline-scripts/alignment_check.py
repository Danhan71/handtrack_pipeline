# functions for checking alignment of camera times/trial/touch screent imes

import os
from pythonlib.tools.camtools import get_lags, finalize_alignment_data
from drawmonkey.tools.preprocess import *
from pyvm.globals import BASEDIR
import argparse
import pickle
import os

if __name__ == "__main__":
    import traceback
      
    parser = argparse.ArgumentParser(description="Final Plots etc")
    parser.add_argument("name", type=str, help="Experiment name/date")
    parser.add_argument("--animal", type=str, help="Animal name", default=100000)
    parser.add_argument("--tstart", type=int, default=10)
    parser.add_argument("--tend", type=int, default=110)
    parser.add_argument('--plot',action='store_true', help="Do plot step, if set have data file called good inds filled out")

    args = parser.parse_args()

    sdir = BASEDIR
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
    if os.path.exists(f'{sdir}/processed_data.pkl'):
        print('Processed data already extracted, loading data file and skipping to number crunching')
        with open(f'{sdir}/processed_data.pkl','rb') as f:
            trial_dats = pickle.load(f)
        #Filter to subset of trials to make later steps go faster
        trial_dat_filtered = {} 
        for trial,dat in trial_dats.items():
            if trial in trange:
                trial_dat_filtered[trial] = dat
        trial_dat = trial_dat_filtered
                
    else:
        assert False, 'Extract data first using wand step 5, using the --noreg flag'

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
    sdir_lag = f'{sdir}/lag'
    for coefs in coefs_list:
        print(f'Doing lags for coefs {coefs}')
        this_sdir = f'{sdir_lag}/{coefs}'
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
            with open(f'{sdir}/good_inds.txt','r') as f:
                good_inds = [item.strip().strip("'") for item in f.readline().replace(' ', '').split(',')]      
            if len(good_inds) < 10:
                print(f'{len(good_inds)} not enough good inds, please enter more in file {this_sdir}/good_inds.txt')
            else:
                import re
                def is_valid_format(s):
                    return bool(re.fullmatch(r"\d+-\d+", s))
                def validate_list(lst):
                    return all(is_valid_format(item) for item in lst)
                assert validate_list(good_inds), f'At least some elements in {this_sdir}/good_inds are formatted improperly (should be comma sep list of trial-stroke as strs). Check this you fool'
            fig,corr_lag, lags = finalize_alignment_data(lags,good_inds)
            fig.savefig(f'{this_sdir}/lag_fig.png')
            if True:
                with open(f'/home/danhan/code/test_data/lags/{expt_with_sess}_lags.pkl') as f:
                    pickle.dump(lags,f)
