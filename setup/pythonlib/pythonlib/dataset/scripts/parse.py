""" script, extracting parses, using the Parser (good) code.
This runs and saves thru Dataset code.

NOTE: for pipeline, see CHUNK_EXTRACTION below
"""
from pythonlib.dataset.dataset import Dataset

QUICK = True # quick means fast parsing.
INITIAL_EXTRACTION = False
BEH_ALIGNED_EXTRACTION = False
CHUNK_EXTRACTION = True
# CHUNK_EXTRACTION_FORCE_REDO=True
reload_parser_each_time = True # if multiprocessing, ensures have latest version
DEBUG = False # focus on just one task.

def get_dataset(a,e,r, FIXED):
    D = Dataset([])
    D.load_dataset_helper(a, e, ver="mult", rule=r)
    D.load_tasks_helper()

    # Keep only the fixed tasks?
    if FIXED:
        D = D.filterPandas({"random_task":[False]}, "dataset")
    return D

def run(a,e,r,v, FIXED):
    SDIR = f"/data2/analyses/database/PARSES_GENERAL/{e}"
    D = get_dataset(a,e,r, FIXED)
    
    parse_params = {"quick":QUICK, "ver":v, "savenote":""}

    if INITIAL_EXTRACTION:
        # D.parser_extract_and_save_parses(ver=v, quick=QUICK, savenote=f"fixed_{FIXED}",     
        #     SDIR=SDIR, save_using_trialcode=False)
        D.parser_extract_and_save_parses(ver=v, quick=QUICK, savenote="",     
            SDIR=SDIR, save_using_trialcode=False)

    if BEH_ALIGNED_EXTRACTION:
        # extract all all the beh-aligned parses too
        list_parse_params = [{"quick":QUICK, "ver":v, "savenote":""}]
        list_suffixes = [x["ver"] for x in list_parse_params]
        pathbase = SDIR
        name_ver = "unique_task_name"
        EXTRACT_BEH_ALIGNED_PARSES = True
        D.parser_load_presaved_parses(list_parse_params, 
            list_suffixes, pathbase=pathbase, name_ver=name_ver,
            ensure_extracted_beh_aligned_parses=EXTRACT_BEH_ALIGNED_PARSES)
    
    if CHUNK_EXTRACTION:
        # NOTE: This does all the steps starting from after tou have gotten initial parses (INITIAL_EXTRACTION)
        # 1) Beh aligned extraction --> {taskname}-behalignedperms.pkl (all in self.Parses so far)
        # --- (parser_load_presaved_parses)
        # 2a) Extract chunks --> into self.BaseParses
        # 2b) Extract perms for each chunk --> into self.Parses
        # 2b) Get best-fit for each chunk --> into self.BaseParses
        # --- parser_extract_chunkparses

        list_parse_params = [{"quick":QUICK, "ver":v, "savenote":""}]
        list_suffixes = [x["ver"] for x in list_parse_params]
        pathbase = SDIR
        name_ver = "unique_task_name"
        EXTRACT_BEH_ALIGNED_PARSES = True # will still load them, but will not try to etract
        D.parser_load_presaved_parses(list_parse_params, 
            list_suffixes, pathbase=pathbase, name_ver=name_ver,
            ensure_extracted_beh_aligned_parses=EXTRACT_BEH_ALIGNED_PARSES)
        
        if reload_parser_each_time:
            # delete from D, save memory
            del D.Dat["parser_graphmod"]

        for indtrial in range(len(D.Dat)):

            # -- print before
            # P = D.parser_get_parser_helper(indtrial)
            # print("=== P, before getting best-fit parses")
            # print("---", D.trial_tuple(indtrial))
            # print(len(P.ParsesBase))
            # for i, p in enumerate(P.ParsesBase):
            #     print(i, p["rule"])
            #     tmp = list(p["best_fit_perms"].keys())
            #     for t in tmp:
            #         print(t)

            D.parser_extract_chunkparses(indtrial, parse_params, saveon=True, 
                reload_parser_each_time=reload_parser_each_time, DEBUG=DEBUG)
            
            # -- print after
            # print("=== P, after getting best-fit parses")
            # print("---", D.trial_tuple(indtrial))
            # P = D.parser_get_parser_helper(indtrial)
            # print(len(P.ParsesBase))
            # for i, p in enumerate(P.ParsesBase):
            #     print(i, p["rule"])
            #     tmp = list(p["best_fit_perms"].keys())
            #     for t in tmp:
            #         print(t)



if __name__=="__main__":
    MULTI= False
    FIXED = True # only do fixed tasks.

    if MULTI:
        # Multiprocessing.
        args1, args2, args3, args4, args5 = [], [], [], [], []
        # animal_list = ["Red", "Pancho"]
        # animal_list = ["Pancho"]
        # ver_list = ["graphmod"]
        expt_list = ["gridlinecircle"]
        rule_list = ["baseline", "circletoline", "linetocircle", "lolli"]
        animal_list = ["Pancho"]
        ver_list = ["graphmod"]
        for a in animal_list:
            for e in expt_list:
                for r in rule_list:
                    for v in ver_list:
                        args1.append(a)
                        args2.append(e)
                        args3.append(r)
                        args4.append(v)
                        args5.append(FIXED)
                        # D.parser_extract_and_save_parses(ver=v, quick=QUICK, savenote=f"fixed_{FIXED}")

        from multiprocessing import Pool
        ncores = 4
        with Pool(4) as pool:
            pool.starmap(run, zip(args1, args2, args3, args4, args5))
    else:
        rule_list = ["baseline", "circletoline", "linetocircle", "lolli"]
        # rule_list = ["linetocircle"]
        for a in ["Pancho", "Diego"]:
            e = "gridlinecircle"
            v = "graphmod"
            for r in rule_list:
                run(a,e,r,v, FIXED)

        # rule_list = ["baseline", "circletoline", "linetocircle", "lolli"]
        # # rule_list = ["linetocircle"]
        # for a in ["Pancho"]:
        #     e = "gridlinecircle"
        #     v = "graphmod"
        #     for r in rule_list:
        #         run(a,e,r,v, FIXED)


