Overview of code for multi-day analysis.


FD = loadMultDat [loadMultDataForExpt is shortcut], gives array, one for each session.
-->

	PROBEDAT = PROBEDATfromFD, gives one item per trial, all flattened.
	-->
		strokfeats = function(PROBEDAT), gives one item per strokes.
sessdict = loadMultDat.



Where to get code for various analyses:
1) Most raw summary: Summary/overview plots of entire experiment, focusing on plotting trials (drawings):
- analysis_modelexpt_multsession_100420

2) Learning-trajectory for model-based scores (e.g,, nstrokes, model score)
- analysis_modelexpt_multsession_taskmodelscore_100620

3) [stroke as datapoint]
Processing and summary plots for strokmodel:
- analysis_line2_090720_generativeStrokModel (notebook).

Summary/overview plots for stroke features (many plots)
- analysis_modelexpt_multsession_strokmodel_100420

2d histogram of strok-model scores (data level = strokes, generalization only), 
comparing epochs:
- analysis_modelexpt_multsession_strokmodel_100420

Replotting 1d histogram sumamrizing distance/circulatiry, flattened over all strokes
- analysis_modelexpt_multsession_strokmodel_100420



============ GENERAL MAP OF ANALYSES (KEEP UPDATED)
Scalars over time/days
	General: analysis_TEMPLATE [ALL PORTED TO dataset analysis simple]
	Grouping by block: analysis_TEMPLATE [NOT PORTED]
	Including model score: analysis_TEMPLATE [PORTED]
	Including arbitrary score: analysis_TEMPLATE (like first_stroke_horiz_pos) [NOT PORTED]

Plot raw behavior over all trials
	General: analysis_TEMPLATE

Fitting task model
	Without fitting model, just a parse-based model: analysis_TEMPLATE
	Bayesian model fitting: devo_taskmodel_motorcost
	Descriptive statistics, used for prior: (Not yet done)

Stroke-level model fitting and plots:
	(See notes above)

Old analyses:
	End-state comfort:
	Coarticulation:
	Copy strategy:

Various plotting code:
	CRCNS, plotting trials as grids: (see notebook) [now in analysis_template]
	Waterfall: (see notebook)
	Stroke num of first touch (biasdir): (see analysis_biasdir23456)



################## USEFUL CODE
Rename files, replacing string with another.
This renames "test" to "1" in all files.
rename 's/test/1/' *

https://stackoverflow.com/questions/1392768/renaming-part-of-a-filename
