#Quick and dirty little script to reorder the columns in wandpts files to match the order of cameras of the checkerboard day

import os
import pandas as pd
import shutil

#Put here the order cams should be from the date checkerboard was done for (can be found in metadat file)
checkerboard_cam_order = ["flea", "fly1", "bfs1", "fly2"]

#Put here a list of the days which have wand points. That means you have run everything in pipeline up to wand step 1
wand_pts_dates = ["221015_dircolor2_wandgrid","221015_dirshapecolor1b_wandgrid","221015_dircolor3b_wandgrid"]

#Base dir that the data is stored in 
data_dir = "/data3/hand_track/Pancho"
#List of wand point file names

#Name of want pts file, maybe need list if all diff names
# names_list = {"wandPointsNoScreen.csv","wandPointsNoScreen.csv","wandPointsScreen.csv"}
name = "241014_wandPoints_99thresh.csv"

for date in wand_pts_dates:
    this_dir = f"{data_dir}/{date}/wand/wand_calibration"
    col_list = open(f"{this_dir}/columns.txt").read().splitlines()
    # if os.path.exists(f"{this_dir}/{date}-wandPoints.csv"):
        # os.remove(f"{this_dir}/{date}-wandPoints.csv")
    # wand_pts = [c for c in os.listdir(this_dir) if c.endswith('.csv')][0]
    wand_pts = f"{this_dir}/{name}"
    df = pd.read_csv(f"{wand_pts}", header=None)
    df.columns = col_list
    align_list = []
    for cam in checkerboard_cam_order:
        for col in list(df.columns):
            if cam in col and 'green' in col:
                align_list.append(col)
    for cam in checkerboard_cam_order:
        for col in list(df.columns):
            if cam in col and 'red' in col:
                align_list.append(col)
    df = df.reindex(columns=align_list)
    df.to_csv(f"{this_dir}/{date}-{name}", index = False, header = False)
    shutil.copy(f"{this_dir}/{date}-{name}", f"{data_dir}/220914_wandall/wand/wand_calibration/241014_{date}_99.csv")






# Maybe also add a section to do the reodering of the dlt coefficients after the wand cal has been done

