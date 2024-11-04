""" Calibrate single camera to checkerboards.
"""


from pyvm.classes.videoclass import Videos
import argparse




if __name__=="__main__":

	parser = argparse.ArgumentParser(description="Description of your script.")
	parser.add_argument("name", type=str, help="Experiment name/date")
	parser.add_argument("--runtype", type=int, help="Run option (0=calibrate or 1=extract_all_frames)")
	parser.add_argument("--dim1", type=int, help="Checkerb size (nrow vert). Default: (10)", default=10)
	parser.add_argument("--dim2", type=int, help="Checkerb size (ncols). Default: (7)", default=7)
	parser.add_argument("--animal", type=str, help="Aminmal name")


	args = parser.parse_args()
	# RUN = 0 = "extract_all_frames"
	# RUN = 1 = "calibrate"
	RUN = args.runtype
	PATTERNSIZE = (args.dim1, args.dim2) # (nrow vertices,ncols)
	animal = args.animal
	name = args.name
	date = name.split('_')[0]
	expt = name.split('_',1)[1]
	condition = "checkerboard"

	V = Videos()
	V.load_data_wrapper(date,expt,animal,condition)
	print("Experiment: ", expt)
	print("Pattern Size", PATTERNSIZE)
	print("Run #", RUN)
	# 2. By eye, find good frames for each cam. enter these into metadata.

	if RUN==1:
		# 3. Extract good frames and calibrate
		# V = Videos()
		# V.load_data_wrapper(expt, condition)
		V.collect_goodframes_from_videos(overwrite=False)
		V.calibrate_each_camera(patternSize=PATTERNSIZE)