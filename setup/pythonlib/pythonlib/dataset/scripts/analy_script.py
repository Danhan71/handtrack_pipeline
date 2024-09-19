    


if __name__=="__main__":
    # To run various analyses here:

    assert False, "consolidated in dataset_analy/primitives.py. Is fine here, but dont derive things from here"
    from pythonlib.dataset.dataset import Dataset
    # import pickle
    # import os
    import numpy as np
    import matplotlib.pyplot as plt

    for a in ["Pancho", "Diego"]:
        #     a = "Diego"
        e = "primcat12"
        r = "null"

        D = Dataset([])
        D.load_dataset_helper(a, e, rule=r)

        # Extract pre-saved tasks
        D.load_tasks_helper()


        SAVEDIR = f"/data2/analyses/main/primitives/{a}-{e}-{r}/FIGS/drawfig_by_date"
        from pythonlib.tools.expttools import writeDictToYaml, makeTimeStamp
        notes = {
            "tstamp":makeTimeStamp(),
            "code":"dataset/analy_dataset_drawfigplots_210623",
            "goal":"visualize progression during learning"
        }

        import os 
        os.makedirs(SAVEDIR, exist_ok=True)
        writeDictToYaml(notes, f"{SAVEDIR}/notes.yaml")



        # Preprocess, expose each tasks params
        from pythonlib.tools.pandastools import applyFunctionToAllRows
        def F(x, ver):
            """ expose the task params"""
            T = x["Task"]
            
            if "circle" in x["character"] and ver=="theta":
                # circles are symmetric circluar.
                return 0.
            else:
            #     sx = T.Shapes[0][1]["sx"]
            #     sy = T.Shapes[0][1]["sy"]
            #     xpos = T.Shapes[0][1]["x"]
            #     ypos = T.Shapes[0][1]["y"]
            #     th = T.Shapes[0][1]["theta"]
                return T.Shapes[0][1][ver]
            
        for ver in ["sx", "sy", "x", "y", "theta"]:
            D.Dat = applyFunctionToAllRows(D.Dat, lambda x:F(x, ver), ver)



        only_keep_random_tasks = True # True, since in general for prims I made them random.
        if only_keep_random_tasks:
            prim_list = D.Dat[D.Dat["random_task"]==True]["character"].unique()
        else:
            ##### Plot, randomly interleaved trials over all time.
            # 1) Find all trials for a given primitive
            prim_list = D.Dat["character"].unique()



        ###### PLOT
        from pythonlib.dataset.plots import plot_beh_grid_singletask_alltrials

        Nmin_toplot = 5 # only ploit if > this many trials across all days.
        max_cols = 40 # max to plot in single day

        # pick out all trials for a combo of (scale, orientation)
        for PRIM in prim_list:

            SAVEDIR_THIS = f"{SAVEDIR}/{PRIM}"
            os.makedirs(SAVEDIR_THIS, exist_ok=True)
            
            list_sx = np.unique(D.Dat[D.Dat["character"]==PRIM]["sx"].to_list())
            list_sy = np.unique(D.Dat[D.Dat["character"]==PRIM]["sy"].to_list())
            list_theta = np.unique(D.Dat[D.Dat["character"]==PRIM]["theta"].to_list())
            
        #     Dprim = D.filterPandas({"character":[PRIM]}, "dataset")
        #     list_sx = np.unique(Dprim.Dat["sx"])
        #     list_sy = np.unique(Dprim.Dat["sy"])
        #     list_theta = np.unique(Dprim.Dat["theta"])
            print(list_sx)
            print(list_sy)
            print(list_theta)

            # For each combo of params, plot sample size
            from itertools import product
            for sx, sy, th in product(list_sx, list_sy, list_theta):
                DprimThis = D.filterPandas({"character":[PRIM], "sx":[sx], "sy":[sy], "theta":[th]}, "dataset")
                n =  len(DprimThis.Dat)
                print("**", sy, sy, th, "N: ", n)

                if n>Nmin_toplot:
                    figb, figt = plot_beh_grid_singletask_alltrials(DprimThis, PRIM, "date", plotkwargs={"max_cols":max_cols})

                    fname = f"sx{sx:.2f}_sy{sy:.2f}_th{th:.2f}"

                    figb.savefig(f"{SAVEDIR_THIS}/{fname}-BEH.pdf")
                    figt.savefig(f"{SAVEDIR_THIS}/{fname}-TASK.pdf")
                plt.close("all")