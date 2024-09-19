"""
Linear mixed effects modeling

See docs:
https://www.statsmodels.org/stable/mixed_linear.html
[formulas for design matrix] https://patsy.readthedocs.io/en/latest/index.html
[tutorials] https://www.pythonfordatascience.org/mixed-effects-regression-python/
[tutorials] https://vasishth.github.io/Freq_CogSci/checking-model-assumptions.html
"""
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
from pythonlib.tools.statstools import plotmod_pvalues
from pythonlib.tools.pandastools import grouping_append_and_return_inner_items
from pythonlib.tools.plottools import rotate_x_labels

def lme_categorical_fit_plot(df, y, fixed_treat, lev_treat_default=None,
    rand_grp_list=None, rand_grp=None, PLOT = False):
    """
    Model continuos response (y) regressed on categorical vairable (fixed_treat),
    with contrasts relative to a default level of that variable (lev_treat_default).
    Also have random grouping variable/intercepts (rand_grp)
    PARAMS:
        (see docs above)
        - lev_treat_default, if None, then takes the first in alpha
    EXAMPLE:
        fixed_treat = "microstim_epoch_code"
        lev_treat_default = "off"
        rand_grp = "lme_grp"
    """

    if lev_treat_default is None:
        lev_treat_default = df[fixed_treat].unique().tolist()[0]
        
    if rand_grp is None:
        assert not rand_grp_list is None
    else:
        assert rand_grp_list is None

    if rand_grp_list is not None:
        _, df = grouping_append_and_return_inner_items(df, rand_grp_list, 
            new_col_name="dummytmp", return_df=True)
        rand_grp = "dummytmp"
    else:
        assert rand_grp in df.columns

    str_treat = f"C({fixed_treat}, Treatment('{lev_treat_default}'))"
    formula = f"{y} ~ {str_treat}"
    md = smf.mixedlm(formula, df, groups=df[rand_grp])
    mdf = md.fit()

    if False:
        # OLD code, before I figured out how to change the default level
        # Here does each level one by one (comparison to default)
        # Test effect of stim on each pair against "off"
        mc_default = "off"
        y = "dist_beh_task_strok"
        list_microstim_code = DS.Dat["microstim_epoch_code"].unique().tolist()
        for mc in list_microstim_code:
            if not mc==mc_default:
                dfthis = DS.Dat[DS.Dat["microstim_epoch_code"].isin([mc_default, mc])]
                
                md = smf.mixedlm(f"{y} ~ microstim_epoch_code", dfthis, groups=dfthis["lme_grp"])
                mdf = md.fit()
                print("---------------------", mc)
                print(mdf.summary())
        
    
    print("MODEL SUMMARY: ")
    print(mdf.summary())

    # Map from exog name (in model) to interpretable name
    map_exogname_semaname = {}
    for name in md.exog_names:
        if name=="Intercept":
            map_exogname_semaname[name] = lev_treat_default
        else:
            map_exogname_semaname[name] = name[len(str_treat)+3:-1]
    for k, v in map_exogname_semaname.items():
        print(k, ":", v)

    # Extract condifence intervals.
    res_conf_int = {}
    for key, name in map_exogname_semaname.items():
        res_conf_int[name] = np.asarray([mdf.conf_int().loc[key][0], mdf.conf_int().loc[key][1]]) # (2,)

    # Extract fixed effects.
    res_fixed_effects = {}
    for key, name in map_exogname_semaname.items():
        res_fixed_effects[name] = np.asarray([mdf.fe_params.loc[key]]) # (2,)

    # Extract p values
    res_pvalue = {}
    for key, name in map_exogname_semaname.items():
        res_pvalue[name] = np.asarray([mdf.pvalues.loc[key]]) # (2,)

    # Plot summary
    if PLOT:

        w = len(res_conf_int)*1.5
        if w>20:
            w = 20

        fig, axes = plt.subplots(1,2, figsize=(12,5))
        # ax.plot(res_pvalue.keys(), res_pvalue.values())
        # ax.plot(res_conf_int.keys(), res_conf_int.values())

        ####### ALL
        ax = axes.flatten()[0]
        x = [k for k,v in res_conf_int.items()]
        ci_min = [v[0] for k,v in res_conf_int.items()]
        ci_max = [v[1] for k,v in res_conf_int.items()]
        pvals = [v[0] for k,v in res_pvalue.items()]
        fixed_effects = [v[0] for k,v in res_fixed_effects.items()]

        for xthis, c1, c2 in zip(x, ci_min, ci_max):
            ax.plot([xthis, xthis], [c1, c2], "-k")
        # ax.plot(x, ci_min, "ok")
        # ax.plot(x, ci_max, "ok")
        ax.plot(x, fixed_effects, "ok")
        ax.axhline(0)
        ax.set_ylabel(f"{y}")
        ax.set_title("CI")

        # overlay p values
        plotmod_pvalues(ax, x, pvals)  
        rotate_x_labels(ax)  

        ######## zoom
        ax = axes.flatten()[1]
        x = [k for k,v in res_conf_int.items() if not k==lev_treat_default]
        ci_min = [v[0] for k,v in res_conf_int.items() if not k==lev_treat_default]
        ci_max = [v[1] for k,v in res_conf_int.items() if not k==lev_treat_default]
        pvals = [v[0] for k,v in res_pvalue.items() if not k==lev_treat_default]
        fixed_effects = [v[0] for k,v in res_fixed_effects.items() if not k==lev_treat_default]

        for xthis, c1, c2 in zip(x, ci_min, ci_max):
            ax.plot([xthis, xthis], [c1, c2], "-k")
        # ax.plot(x, ci_min, "ok")
        # ax.plot(x, ci_max, "ok")
        ax.plot(x, fixed_effects, "ok")
        ax.axhline(0)
        ax.set_ylabel(f"{y} vs {lev_treat_default}")
        ax.set_title("CI")

        # overlay p values
        plotmod_pvalues(ax, x, pvals)    
        rotate_x_labels(ax)     

    else:
        fig, ax = None, None

    RES = {
        "model":md,
        "model_results":mdf,
        "map_exogname_semaname":map_exogname_semaname,
        "res_conf_int":res_conf_int,
        "res_fixed_effects":res_fixed_effects,
        "res_pvalue":res_pvalue}

    return RES, fig, ax


        # # Plot residuals
        # fig, axes = plt.subplots(2,1, figsize=(10, 10))

        # # Histogram of residuals
        # ax = axes.flatten()[0]
        # ax.hist(mdf.resid, 50);
        # ax.axvline(0, color="r")
        # ax.set_xlabel("residual")

        # # Resid vs. fitted values
        # ax = axes.flatten()[1]
        # ax.plot(mdf.fittedvalues, mdf.resid, 'xk', alpha=0.5)
        # ax.axhline(0)
        # ax.set_xlabel("fitted value")
        # ax.set_ylabel("residual")

