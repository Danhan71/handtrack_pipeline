import pandas as pd

data_dir = "/data3/hand_track/Pancho/220317_chunkbyshape4/wand/wand_calibration"
df = pd.read_csv(f"{data_dir}/wandPoints.csv", header = None)

rng = list(range(0,len(df)))

inds = [ind for ind in rng if (ind % 4) == 0  ]
print(inds)


sub_df = df.iloc[inds, :]

sub_df.to_csv(f"{data_dir}/subWandPoints.csv", index = False, header = False)

