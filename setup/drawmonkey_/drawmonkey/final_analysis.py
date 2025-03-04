from pythonlib.tools.stroketools import *
from pyvm.tools.preprocess import *
from pyvm.classes.videoclass import Videos
from pythonlib.tools.expttools import load_yaml_config
from pyvm.globals import BASEDIR
from drawmonkey.tools.handtrack import HandTrack, getTrialsCameraFrametimes
from pyvm.utils.directories import get_metadata
import argparse
import matplotlib.pyplot as plt
import os
import pickle


if __name__ == "__main__":

	import traceback

	parser = argparse.ArgumentParser(description="Final Plots etc")
	parser.add_argument("name", type=str, help="Experiment name/date")
	parser.add_argument("--animal", type=str, help="Animal name", default=100000)
	parser.add_argument("--reg", type=int, help="Do linear regression on cam data")
	parser.add_argument("--pipe", type=str, help="Path to pipeline dir")
	parser.add_argument("--data", type=str, help="Base dir for where the data is stored (up to but not includong animal name)")
	parser.add_argument("--supp", type=int,help="Do you have supplemental data saved in supp_reg_pts folder to use for regression?")
	# parser.add_argument("--out", type=str, help="Output dirtectory (auto-formatted)")

	args = parser.parse_args()

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

	animal = args.animal
	ind1_ml2 = 1
	reg = args.reg
	pipe_path = args.pipe
	data_dir = args.data
	supp = args.supp
	

	#Get range of trials to analyze data from
	config = load_yaml_config(f"{pipe_path}/metadata/{animal}/{name}.yaml")
	vid_inds = config["list_vidnums"][0]
	trange = range(vid_inds[0]+ind1_ml2,vid_inds[1]+ind1_ml2)
	# ind1_vid = vid_inds[0]global

	#Sort of a vestige but we'll keep it
	if expt=="chunkbyshape4":
		ind1_vid = 4
	else:
		ind1_vid=0

	# print(ind1_vid)
	# assert False
	print("Vid", trange)

	fd = loadSingleDataQuick(animal, date, expt, sess)
	fd['params']['sample_rate'] = fd['params']['sample_rate'][0]

	HT = HandTrack(ind1_vid, ind1_ml2, fd, sess_print=sess_print, animal=animal, date=date, expt=expt)
	HT.load_campy_data(ind1_ml2)
	trials_no_ts_data = []

	# fd = loadSingleDataQuick("Pancho", "220317", "chunkbyshape4", 1)
	# HT = HandTrack(4, 1, fd, 220317, "chunkbyshape4")

	SAVEDIR = f"{data_dir}/{animal}/{date}_{expt}{sess_print}"
	

	if reg:
		#Include out=SAVEDIR if you want to save the regression for later use
		HT.fit_regression(trange, out=f'{SAVEDIR}/transforms')

	for coefs in HT.Coefs:
		SAVEDIR = f"{data_dir}/{animal}/{date}_{expt}{sess_print}/{coefs}_figures"
		if os.path.exists(SAVEDIR):
			import shutil
			shutil.rmtree(SAVEDIR)
		os.makedirs(SAVEDIR)

	

	fails = {}
	dat_trials = {}
	for trial_ml2 in trange:
		dat_trials[trial_ml2] = {}
		finger_raise_time = 0.0
		try:
			dat, dict_figs, dict_reg_figs, all_day_figs = HT.process_data_wrapper(trial_ml2, ploton=True, \
															  finger_raise_time=finger_raise_time, aggregate=True,ts_cam_offset=0.043)
			dat_trials[trial_ml2] = dat
		except Exception as e:
			print(traceback.format_exc())
			fails[trial_ml2] = traceback.format_exc()
			continue

		# save all figs
		


		for coefs, list_figs in dict_figs.items():
			SAVEDIR = f"{data_dir}/{animal}/{date}_{expt}{sess_print}/{coefs}_figures/trialml2_{trial_ml2}"
			os.makedirs(SAVEDIR, exist_ok=True)
			for i, fig in enumerate(list_figs):
				os.makedirs(f"{SAVEDIR}/no_regression", exist_ok=True)
				if fig is not None:
					fig.savefig(f"{SAVEDIR}/no_regression/overview_{i}.pdf")
				else: 
					print("*****No data found for", trial_ml2, "so no figure saved!!")
					trials_no_ts_data.append(trial_ml2)
				
		for coefs, list_reg_figs in dict_reg_figs.items():
			SAVEDIR = f"{data_dir}/{animal}/{date}_{expt}{sess_print}/{coefs}_figures/trialml2_{trial_ml2}"
			for i, fig in enumerate(list_reg_figs):
				os.makedirs(f"{SAVEDIR}/regression", exist_ok=True)
				if fig is not None:
					fig.savefig(f"{SAVEDIR}/regression/overview_{i}.pdf")
		plt.close('all')

	print("No touch screen data or bad cam data found for the following trials, no figures were saved:")
	print(trials_no_ts_data)
	print(list(fails.keys()))


	with open (f'{data_dir}/{animal}/{date}_{expt}{sess_print}/skipped_trials.txt','w') as f:
		for trial in trials_no_ts_data:
			f.write(f"{trial}\n")
		f.write("Trial_Failures,Reasons\n")
		for k,v in fails.items():
			f.write(f"{k},{v}\n")

	_,_,_,all_day_figs = HT.process_data_wrapper(all_day=True)
	if all_day_figs is not None:
		for coefs, figs in all_day_figs.items():
			SAVEDIR = f"{data_dir}/{animal}/{date}_{expt}{sess_print}/{coefs}_figures"
			figs.savefig(f"{SAVEDIR}/all_day_summary.png")
	with open(f'{data_dir}/{animal}/{date}_{expt}{sess_print}/processed_data.pkl','wb') as f:
		pickle.dump(dat_trials,f)






#code graveyard

#Jump quant figs
#Doesn't actually add much information, see all day figs dispalcement hist
#Problem with this function is it incorporates data that is outside of the fixation period.
#In theory it would be possible to restrict it, but I think that was too much work so I just 
#do the jump quant in HT where that is already figured out. Only downside is HT displacements
# are after coords have been triangulated, so if you want pre transform displacements (i.e. displacements)
#for each camera then you will have to do it here with video class

# _ = jump_quant(date, expt, animal, HT=HT, vid_inds=vid_inds, sess=sess, sess_print=sess_print)
# jump_quant_figs = [None]

# if jump_quant_figs is not None:
# 	os.makedirs(f'{SAVEDIR}/jump_quants',exist_ok=True)
# 	[fig.savefig(f"{SAVEDIR}/jump_quants/trialml2_{trial}-jump_quant.pdf") for trial,fig in jump_quant_figs.items() if fig is not None]
# else: 
# 	print("*****No data found so no figures saved!!")
# 	assert False

# def jump_quant(date, expt, animal, HT, vid_inds, sess, sess_print, condition="behavior", ploton=False):
	# """
	# Plots fiugres for looking at jumps in data. Mian figure cumulative displacement vs frame, with line saturation based on likelihood of DLC label
	# PARAMS:
	# expt = experiment name
	# condition = condition (behavior)
	# RETURNS:
	# Big fig with all trials and cameras from this day
	# """
	# name = f"{date}_{expt}{sess_print}"
	# list_trials=list(range(vid_inds[0],vid_inds[1]))
	# data_dir = f"{BASEDIR}/{animal}/{name}/{condition}/extracted_dlc_data"

	# V = Videos()
	# V.load_data_wrapper(date=date, expt=expt, condition=condition, animal=animal, session=sess)

	# V.import_dlc_data()
	# sdir = f"{V.Params['load_params']['basedir']}/extracted_dlc_data"
	# cams = list(V.Params['load_params']['camera_names'].values())

	# pkl_list = []
	# for file in os.listdir(data_dir):
	# 	if file.endswith(".pkl"):
	# 		pkl_list.append(f"{data_dir}/{file}")
	# #After running loop this will contain one df for each trial, containing
	# #the (x,y) coord, displacement, and likelhiood for each camera
	# df_list = []
	# # for cam in cams:
	# #     columns.append(f"{cam}_x")
	# #     columns.append(f"{cam}_y")
	# #     columns.append(f"{cam}_disp")
	# #     columns.append(f"{cam}_like")
	# good_cams = {}
	# skipped_trials = []
	# for trial in list_trials:
	# 	good_cams[trial] = []
	# 	skip = False
	# 	df = pd.DataFrame()
	# 	for cam in cams:
	# 		suffix = f"camera_{cam}_-trial_{trial}-dat.pkl"
	# 		this_file = [file for file in pkl_list if file.endswith(suffix)]
	# 		assert len(this_file) <= 1, f"{len(this_file)} many files found. Extra copy of pkl file for this cam/trial? Or new naming convention (uh oh). PKL list = {pkl_list[0:2]}..."
	# 		if len(this_file) == 0:
	# 			continue
	# 		good_cams[trial].append(cam)
	# 		with open(this_file[0], 'rb') as f:
	# 			data = pd.read_pickle(f)
	# 			data = pd.DataFrame(data)
	# 			data = data.droplevel([0,1],axis=1)
	# 			data["disp"] = displacement(data['x'],data['y']).fillna(0)
	# 			n = data.shape[1] // 2
	# 			d = {
	# 				f"{cam}_x": data['x'],
	# 				f"{cam}_y": data['y'],
	# 				f"{cam}_disp": data["disp"],
	# 				f"{cam}_like": data["likelihood"]
	# 			}
	# 			add_df = pd.DataFrame(data=d)
	# 			df = pd.concat([df,add_df], axis=1)
	# 	#catch trials with not data
	# 	if len(df) > 0:
	# 		df_list.append(df)
	# 	else:
	# 		skipped_trials.append(trial)
	
	# fig_dict = {}

	
	# list_trials_good = [t for t in list_trials if t not in skipped_trials]
	# for df,t in zip(df_list,list_trials_good):
	# 	#t+1 because t here is vid trial num (0 ind) and t in handtrack is matlab trial (1 ind)
	# 	m = len(good_cams[t]) + 1
	# 	if t+1 not in HT.AllDay:
	# 		HT.AllDay[t+1] = {}

		
	# 	disp_list = []
	# 	if ploton:
	# 		fig, axes = plt.subplots(nrows=m, ncols = 1, figsize=(18,10*m))
	# 	for cam in good_cams[t]:
	# 		if ploton:
	# 			axes[0].scatter(x=df.index, y=df[f"{cam}_disp"].cumsum().fillna(0),s=df[f"{cam}_like"], label=cam)
	# 		disp_list.extend(df[f"{cam}_disp"])
	# 	HT.AllDay[t+1]['disp'] = np.array(disp_list)
	# 	if ploton:
	# 		axes[0].set_xlabel('Frame')
	# 		axes[0].set_ylabel('Displacement')
	# 		axes[0].set_title(f'Frame by Frame Displacements for Each Camera Trial {t}')
	# 		axes[0].legend()
	# 		axes[0].grid(True)
	# 		rng = range(1,len(good_cams[t])+2)
	# 		for i,cam in zip(rng,good_cams[t]):
	# 			n_points = len(df)
	# 			indices=np.arange(n_points)
	# 			colors = plt.cm.viridis(indices/max(indices))
	# 			axes[i].scatter(df[f"{cam}_x"], df[f"{cam}_y"], c=colors, cmap='viridis', label=cam)
	# 			axes[i].set_xlabel('x coord')
	# 			axes[i].set_ylabel('y coord')
	# 			axes[i].set_title(f'xy trajectory over time for trial {t}, {cam}')
	# 			axes[i].legend()
	# 			axes[i].grid(True)
	# 		fig_dict[t+1] = fig
	# #Put empty arrays in skipped trials
	# for t in skipped_trials:
	# 	if t+1 not in HT.AllDay:
	# 		HT.AllDay[t+1] = {}
	# 	HT.AllDay[t+1]['disp'] = np.array([])

	# return fig_dict

	# For posterity's sake
	# def displacement(x,y):
		# x_diff = x.diff()
		# y_diff = y.diff()  
		# return np.sqrt(x_diff**2 + y_diff**2)