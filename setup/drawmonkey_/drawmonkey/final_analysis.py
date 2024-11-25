from pythonlib.tools.stroketools import *
from tools.preprocess import *
from sklearn.linear_model import LinearRegression
from pyvm.classes.videoclass import Videos
from tools.handtrack import HandTrack, getTrialsCameraFrametimes
from pyvm.utils.directories import get_metadata
from pythonlib.tools.expttools import load_yaml_config
from pyvm.globals import BASEDIR
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle

def displacement(x,y):
	x_diff = x.diff()
	y_diff = y.diff()  
	return np.sqrt(x_diff**2 + y_diff**2)

def fit_regression_cam(HT, trange, supp=None, reg_type='basic'):
	"""
	Fits linear regression for this HT object and trial list
 
	PARAMS:
	- HT: hand track object
	- trange: range of trials
	- supp : Supplement regression with other points to make more robust? Use directory in pipeline to store supp files
		- Should be passed in as list with dlt pts at 0 and ground truth pts at 1
	- reg_type : regression type, basic or lasso
 
	RETURNS: 
	- scikit linear regression object trained on given reg type
	"""
	strokes_cam_all = []
	strokes_cam_allz = []
	strokes_touch_all = []
	touch_interpz = []

	outcomes = {}
	for trial in trange:

		try:
			dat, _, _ = HT.process_data_singletrial(trial, ploton=False, finger_raise_time=0.0)
		except:
			print("Not regressing this trial bc failure")
			dat = {}
		#Skips trials nbo data
		if dat == {}:
			continue

		for strok_cam, strok_touch in zip(dat["strokes_cam"], dat["strokes_touch"]):
			strokes_cam_allz.append(np.array(strok_cam))
			strokes_touch_all.append(np.array(strok_touch))

	assert (strokes_touch_all != {}), "No data for this expt"

	N = ["input_times"]
	for strok in strokes_cam_allz:
		N.append(np.array([p[3] for p in strok]))

	touch_interp = strokesInterpolate2(strokes_touch_all, N)

	
	for strok in touch_interp:
		add_z = [[p[0],p[1],0] for p in strok]
		touch_interpz.append(np.array(add_z))

	#take out z for regression
	for strok in strokes_cam_allz:
		strokes_cam_all.append(np.array([[p[0],p[1], p[2]] for p in strok]))

	cam_one_list = np.array([])
	touch_one_list = np.array([])

	#Makes one list of pts out of the middle 50% of strokes (the tail ends tend to be bad cam data and mess up regression)
	for strok_cam, strok_touch in zip(strokes_cam_all, touch_interpz):
		assert len(strok_cam)==len(strok_touch), "Stroke lens misaligned, maybe can just remove this assert idk"
		n = len(strok_touch)
		q1 = int(n/4)
		q3 = int(3*n/4)
		if len(cam_one_list) == 0 and len(touch_one_list) == 0:
			cam_one_list = np.array(strok_cam[q1:q3])
			touch_one_list = np.array(strok_touch[q1:q3])
		else:
			cam_one_list = np.concatenate((cam_one_list,strok_cam[q1:q3]))
			touch_one_list = np.concatenate((touch_one_list,strok_touch[q1:q3]))
   
	if supp is not None:
		assert (len(supp) == 2) & (len(supp[0]) == len(supp[1]))
		list(cam_one_list).extend(supp[0])
		list(touch_one_list).extend(supp[1])
    #Touhc list is 'ground truth' cam list is data to fit
	assert len(cam_one_list) == len(touch_one_list), "cam, touch different lengths"
 
	if reg_type == 'basic':
		reg = LinearRegression().fit(cam_one_list, touch_one_list)
	#Sucks
	elif reg_type == 'lasso':
		from sklearn.linear_model import Lasso
		from sklearn.preprocessing import StandardScaler
		from sklearn.model_selection import train_test_split
  
		z_weight = 1

		scaler = StandardScaler()
		X_scaled = scaler.fit_transform(cam_one_list)
  
		Y_weighted = touch_one_list.copy()
		Y_weighted[:, 2] *= z_weight
  
		#Make train and test sets, but not really necessary since we don't care baout it generalizing to all data
		# X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_weighted, test_size=0.2, random_state=42)
  
		#Do this if doing train/test split
  		# #Append the supplemental data to the train set so it gets more visibility
		# if supp is not None:
		# 	X_train.extend(supp[0])
		# 	Y_train.extend(supp[1])

		#Fit lasso model
		reg = Lasso(alpha=0.0001)
		reg.fit(X_scaled,Y_weighted)
	else:
		assert False, "Code ur own model then, fool"
	
	# with open ('/home/danhan/Documents/reg.pkl','wb') as f:
	# 	pickle.dump(reg,f)
	# with open ('/home/danhan/Documents/dat.pkl','wb') as f:
	# 	pickle.dump([cam_one_list,touch_one_list],f)
	return reg

def jump_quant(date, expt, animal, HT, vid_inds, condition="behavior"):
	"""
	Plots fiugres for looking at jumps in data. Mian figure cumulative displacement vs frame, with line saturation based on likelihood of DLC label
	PARAMS:
	expt = experiment name
	condition = condition (behavior)
	RETURNS:
	Big fig with all trials and cameras from this day
	"""
	list_trials=list(range(vid_inds[0],vid_inds[1]))
	data_dir = f"{BASEDIR}/{animal}/{date}_{expt}/{condition}/extracted_dlc_data"

	V = Videos()
	V.load_data_wrapper(date=date, expt=expt, condition=condition, animal=animal)

	V.import_dlc_data()
	sdir = f"{V.Params['load_params']['basedir']}/extracted_dlc_data"
	cams = list(V.Params['load_params']['camera_names'].values())

	pkl_list = []
	for file in os.listdir(data_dir):
		if file.endswith(".pkl"):
			pkl_list.append(f"{data_dir}/{file}")
	#After running loop this will contain one df for each trial, containing
	#the (x,y) coord, displacement, and likelhiood for each camera
	df_list = []
	# for cam in cams:
	#     columns.append(f"{cam}_x")
	#     columns.append(f"{cam}_y")
	#     columns.append(f"{cam}_disp")
	#     columns.append(f"{cam}_like")
	good_cams = []
	for trial in list_trials:
		df = pd.DataFrame()
		for cam in cams:
			suffix = f"camera_{cam}_-trial_{trial}-dat.pkl"
			this_file = [file for file in pkl_list if file.endswith(suffix)]
			assert len(this_file) <= 1, f"{len(this_file)} many files found. Extra copy of pkl file for this cam/trial? Or new naming convention (uh oh). PKL list = {pkl_list[0:2]}..."
			if len(this_file) == 0:
				continue
			good_cams.append(cam)
			with open(this_file[0], 'rb') as f:
				data = pd.read_pickle(f)
				data = pd.DataFrame(data)
				data = data.droplevel([0,1],axis=1)
				data["disp"] = displacement(data['x'],data['y']).fillna(0)
				n = data.shape[1] // 2
				d = {
					f"{cam}_x": data['x'],
					f"{cam}_y": data['y'],
					f"{cam}_disp": data["disp"],
					f"{cam}_like": data["likelihood"]
				}
				add_df = pd.DataFrame(data=d)
				df = pd.concat([df,add_df], axis=1)
		df_list.append(df)
	
	fig_dict = {}

	m = len(good_cams) + 1
	for df,t in zip(df_list,list_trials):
		#t+1 because t here is vid trial num (0 ind) and t in handtrack is matlab trial (1 ind)
		if t+1 not in HT.AllDay:
			HT.AllDay[t+1] = {}

		fig, axes = plt.subplots(nrows=m, ncols = 1, figsize=(18,10*m))
		disp_list = []
		for cam in good_cams:
			axes[0].scatter(x=df.index, y=df[f"{cam}_disp"].cumsum().fillna(0),s=df[f"{cam}_like"], label=cam)
			disp_list.extend(df[f"{cam}_disp"])
		HT.AllDay[t+1]['disp'] = np.array(disp_list)
		axes[0].set_xlabel('Frame')
		axes[0].set_ylabel('Displacement')
		axes[0].set_title(f'Frame by Frame Displacements for Each Camera Trial {t}')
		axes[0].legend()
		axes[0].grid(True)
		rng = range(1,len(good_cams)+2)
		for i,cam in zip(rng,good_cams):
			n_points = len(df)
			indices=np.arange(n_points)
			colors = plt.cm.viridis(indices/max(indices))
			axes[i].scatter(df[f"{cam}_x"], df[f"{cam}_y"], c=colors, cmap='viridis', label=cam)
			axes[i].set_xlabel('x coord')
			axes[i].set_ylabel('y coord')
			axes[i].set_title(f'xy trajectory over time for trial {t}, {cam}')
			axes[i].legend()
			axes[i].grid(True)
		fig_dict[t+1] = fig

	return fig_dict


if __name__ == "__main__":

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
	expt = ''.join(expt_info_list[1:])
	if len(expt_info_list) == 3:
		sess = expt_info_list[2]
		sess_print = "_" + sess
	else:
		sess = 1
		sess_print = ""

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
	HT = HandTrack(ind1_vid, ind1_ml2, fd, animal=animal, date=date, expt=expt)
	HT.load_campy_data(ind1_ml2, sess=sess_print)
	trials_no_ts_data = []

	# fd = loadSingleDataQuick("Pancho", "220317", "chunkbyshape4", 1)
	# HT = HandTrack(4, 1, fd, 220317, "chunkbyshape4")

	if reg:
		# assert False
		if supp:
			assert os.path.exists(f'{pipe_path}/supp_reg_pts/xyz_pts_dlt.csv') & os.path.exists(f'{pipe_path}/supp_reg_pts/xyz_pts_gt.csv')
			supp_dlt = np.genfromtxt(f'{pipe_path}/supp_reg_pts/xyz_pts_dlt.csv')
			supp_gt = np.genfromtxt(f'{pipe_path}/supp_reg_pts/xyz_pts_gt.csv')
			supp=[supp_dlt,supp_gt]
		else:
			supp = None
		regression = fit_regression_cam(HT, trange, supp=supp)
		HT.regressor = regression
	else:
		regression = None

	#Returns big figure for all trials and cameras
	jump_quant_figs = jump_quant(date, expt, animal, HT=HT, vid_inds=vid_inds)
	# jump_quant_figs = [None]

	SAVEDIR = f"{data_dir}/{animal}/{date}_{expt}{sess_print}/figures"
	os.makedirs(SAVEDIR, exist_ok=True)

	if jump_quant_figs is not None:
		os.makedirs(f'{SAVEDIR}/jump_quants',exist_ok=True)
		[fig.savefig(f"{SAVEDIR}/jump_quants/trialml2_{trial}-jump_quant.pdf") for trial,fig in jump_quant_figs.items() if fig is not None]
	else: 
		print("*****No data found so no figures saved!!")

	# assert False

	fails = {}
	for trial_ml2 in trange:
		finger_raise_time = 0.0
		try:
			dat, list_figs, list_reg_figs = HT.process_data_singletrial(trial_ml2, ploton=True, \
															  finger_raise_time=finger_raise_time, aggregate=True)
		except Exception as e:
			fails[trial_ml2] = e
		# with open('/home/danhan/Documents/dat.pkl','wb') as f:
		# 	pickle.dump(dat,f)
		# Get errors
		list_dists, reg_list_dists,_, _, fig_error, reg_fig_error  = HT.analy_compute_errors(trial_ml2, ploton=True)
		dat["errors_ptwise"] = list_dists

		#Make figs for all day data


		list_figs.append(fig_error)
		list_reg_figs.append(reg_fig_error)

		# save all figs
		SAVEDIR = f"{data_dir}/{animal}/{date}_{expt}{sess_print}/figures/trialml2_{trial_ml2}"
		print(SAVEDIR)
		os.makedirs(SAVEDIR, exist_ok=True)


		for i, fig in enumerate(list_figs):
			os.makedirs(f"{SAVEDIR}/no_regression", exist_ok=True)
			if fig is not None:
				fig.savefig(f"{SAVEDIR}/no_regression/overview_{i}.pdf")
			else: 
				print("*****No data found for", trial_ml2, "so no figure saved!!")
				trials_no_ts_data.append(trial_ml2)
		for i, fig in enumerate(list_reg_figs):
			os.makedirs(f"{SAVEDIR}/regression", exist_ok=True)
			if fig is not None:
				fig.savefig(f"{SAVEDIR}/regression/overview_{i}.pdf")

	print("No touch screen data or bad cam data found for the following trials, no figures were saved:")
	print(trials_no_ts_data)

	all_day_figs = HT.plot_data_all_day()
	if all_day_figs is not None:
		SAVEDIR = f"{data_dir}/{animal}/{date}_{expt}{sess_print}/figures"
		all_day_figs.savefig(f"{SAVEDIR}/all_day_summary.pdf")

	with open (f'{data_dir}/{animal}/{date}_{expt}{sess_print}/skipped_trials.txt','w') as f:
		for trial in trials_no_ts_data:
			f.write(f"{trial}\n")
		f.write("Trial_Failures,Reasons")
		for k,v in fails.items():
			f.write(f"{k},{v}")