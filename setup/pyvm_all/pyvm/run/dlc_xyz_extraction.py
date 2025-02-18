# """
# final processed DLC (x,y,z) coordinates --> export for use in ml2
# """

from pyvm.classes.videoclass import Videos
import os
import numpy as np
import pandas as pd
from pythonlib.tools.expttools import writeStringsToFile
import cv2
import shutil
from pyvm.globals import BASEDIR
from pythonlib.tools.expttools import load_yaml_config



#dlt reconstruct adapted by An Chi Chen from DLTdv5 by Tyson Hedrick
# https://github.com/MMathisLab/3D-DeepLabCut-Cheetah/blob/master/3D_Evaluation/dlt_reconstruct.py
    

import numpy as np
import csv

def align_dlt_coefs(dlt_coefs, cols, V):
    import pandas as pd
    """
    Takes dlt coefs, cols, and videoclass object as argument and aligns the dlt coefs with this days ordering
    PARAMS:
    dlt_coefs, usual dlt coefs file loaded as array
    cols, cols as list like object
    V, videoclass object
    RETURN:
    reorganized dlt coefs for this day's data
    """

    dict_cams = V.get_cameras()
    cam_list = [c[1][0] for c in dict_cams.items()]

    df = pd.DataFrame(data=dlt_coefs, columns=cols)
    org_array = []

    print("Organizing dlt coeffs ~_~")
    for cam in cam_list:
        if cam not in cols:
            continue
        org_array.append(list(df[cam]))
    org_array = np.array(org_array).T

    return org_array
def extract_P_from_dlt(dltcoefs):
    '''function which recreates the Projection matrix'''
    dltcoefs = np.append(dltcoefs, 0)
    norm = np.linalg.norm(dltcoefs)
    dltcoefs = dltcoefs/norm

    P = dltcoefs.reshape(3,4)
    return P
def triangulate_cv2(dltcoefs,pts):
    """
    Use cv2 to triangulate points, must be done for each pair of cameras and then averaged accross all cams
    INPUTS: 
    dltcoeffs, np array with shape (11,ncams)
    pts, np array with a row for each point, columns are (cam1x,cam1y,cam2x,cam2y...)

    cv2 does normalization so inputting the raw pts and dlt coefs is good 

    RETURNS:
    3d array of xyz pts, one matrix of pts for each frame (1 xyz per frame) 
    (d1 is cam pairs; d2 is number of pts/frames; d3 should be 3 for xyz)
    List telling which cam pair each d1 layer represents
    """
    assert len(dltcoefs[0]) == len(pts[0])/2, f"coefs {len(dltcoefs)} anbd pts {len(pts[0]/2)} not aligned"
    ncams = len(dltcoefs[0])
    cam_pairs = []
    for i in range(ncams):
        for j in range(i+1,ncams):
            cam_pairs.append([i,j])
    #Dataframe to hold the xyz prjections from each cam pair
    pairwise_xyz = []
    for cam_pair in cam_pairs:
        #get coefs and pts for the two cams in question to pass into the internal function
        pair_xyz = triangulate_cv2internal(dltcoefs,pts, cam_pair)
        #There must be a better way to do this
        pairwise_xyz.append(pair_xyz)
    return np.array(pairwise_xyz), cam_pairs
            
def triangulate_cv2internal(dltcoeffs,pts,cams=[0,1]):
    """
    Internal function to call cv2 function for 2 cams. Any size >2 cam pts and dlt coefs can be inputted, 
    will take first two cameras in columns by default. Wanted it to be stand alone ish. 
    See non-internal fxn for inputting format
    """

    #In .mat file with same name as dlt coefs
    cam1_mat = np.array([
        [794,0,452],
        [0,794,508],
        [0,0,1]
    ])
    cam2_mat = np.array([
        [743,0,370],
        [0,743,268],
        [0,0,1]
    ])
    #In checkboard, calibrate_dist.txt
    cam1_distc = np.array([-0.4224, 0.2622, 0.0003, 0.0013, -0.0984])
    cam2_distc = np.array([-0.4122, 0.3531, -0.0001, 0.0010, -0.3494])
    cam1 = cams[0]
    cam2 = cams[1]
    assert cam1 != cam2 and cam1 < cam2, "meowow, bad cams inds"

    cam1_pts = np.array([[p[cam1*2],p[cam1*2+1]] for p in pts])
    cam2_pts = np.array([[p[cam2*2],p[cam2*2+1]] for p in pts])
    cam1_coefs = np.array([p[0] for p in dltcoeffs])
    cam2_coefs = np.array([p[1] for p in dltcoeffs])

    pcam1 = extract_P_from_dlt(cam1_coefs)
    pcam2 = extract_P_from_dlt(cam2_coefs) 

    cam1_undist = cv2.undistortPoints(cam1_pts,cam1_mat,cam1_distc,P=pcam1)
    cam2_undist = cv2.undistortPoints(cam2_pts,cam2_mat,cam2_distc,P=pcam2)

    xyz = []
    for pt1,pt2 in zip(cam1_pts,cam2_pts):
        pt1_homo, pt2_homo = (X.reshape(1,1,2) for X in [pt1,pt2])
        position = cv2.triangulatePoints(pcam1,pcam2,pt1_homo,pt2_homo)
        xyz_pts = cv2.convertPointsFromHomogeneous(position.T).flatten()
        xyz.append(xyz_pts)
    return xyz
def mean_of_stack(stack):
    """
    Function that gets mean of each value of stack of np arrays. (like if you stick a needle at any point in matr)
    INPUT:
    3d stack of np.arrays you want to take average of
    RETURN:
    one 2d array with means for each position in the stack 
    """
    out = np.zeros((stack.shape[1],stack.shape[2]))
    for i in range(stack.shape[2]):
        for j in range(stack.shape[1]):
            elements_this_ij = []
            for k in range(stack.shape[0]):
                elements_this_ij.append(stack[k,j,i])
            out[j,i] = np.mean(np.array(elements_this_ij))
    return out       


def dlt_reconstruct(c,camPts,method='np'):
    # assert False, "only use high likelihood points for reconstruction"

    #number of frames
    nFrames=len(camPts)
    #number of cameras
    nCams=len(camPts[0])/2

    #setup output variables
    xyz = np.empty((nFrames, 3))
    rmse = np.empty((nFrames, 1))

    #process each frame
    for i in range(nFrames):
  
        #get a list of cameras with non-NaN [u,v]
        cdx_size = 0
        cdx_temp=np.where(np.isnan(camPts[i-1,0:int(nCams*2)-1:2])==False, 1, 0)
        for x in range(len(cdx_temp)):
            if cdx_temp[x-1] == 1:
                cdx_size = cdx_size + 1
        cdx = np.empty((1, cdx_size))
        for y in range(cdx_size):
            cdx[0][y] = y+1
        
        #print(cdx_size)

        #if we have 2+ cameras, begin reconstructing
        if cdx_size>=2:
    
            #initialize least-square solution matrices
            m1=np.empty((cdx_size*2, 3))
            m2=np.empty((cdx_size*2, 1))

            temp1 = 0
            temp2 = 0
            #^^ make these 1 if you want to use the 1 index code 
            for z in range(cdx_size*2):
                #virgin 1 indexed code that lived here before I got here, leaving it for memories sake, but I think this is an issue of copying the code from matlab
                # if z%2==0:
                #     m1[z,0]=camPts[i-1,(temp1*2)-2]*c[8,(temp1-1)]-c[0,(temp1-1)]
                #     m1[z,1]=camPts[i-1,(temp1*2)-2]*c[9,(temp1-1)]-c[1,(temp1-1)]
                #     m1[z,2]=camPts[i-1,(temp1*2)-2]*c[10,(temp1-1)]-c[2,(temp1-1)]
                #     m2[z,0]=c[3,temp1-1]-camPts[i-1,(temp1*2)-2]
                #     temp1 = temp1+1
                # else:
                #     m1[z,0]=camPts[i-1,(temp2*2)-1]*c[8,temp2-1]-c[4,temp2-1]
                #     m1[z,1]=camPts[i-1,(temp2*2)-1]*c[9,temp2-1]-c[5,temp2-1]
                #     m1[z,2]=camPts[i-1,(temp2*2)-1]*c[10,temp2-1]-c[6,temp2-1]
                #     m2[z,0]=c[7, temp2-1]-camPts[i-1,(temp2*2)-1]
                #     temp2 = temp2+1

                #Chad 0 index code
                if z%2==0:
                    m1[z,0]=camPts[i,(temp1*2)]*c[8,(temp1)]-c[0,(temp1)]
                    m1[z,1]=camPts[i,(temp1*2)]*c[9,(temp1)]-c[1,(temp1)]
                    m1[z,2]=camPts[i,(temp1*2)]*c[10,(temp1)]-c[2,(temp1)]
                    m2[z,0]=c[3,temp1]-camPts[i,(temp1*2)]
                    temp1 = temp1+1
                else:
                    m1[z,0]=camPts[i,(temp2*2)]*c[8,temp2]-c[4,temp2]
                    m1[z,1]=camPts[i,(temp2*2)]*c[9,temp2]-c[5,temp2]
                    m1[z,2]=camPts[i,(temp2*2)]*c[10,temp2]-c[6,temp2]
                    m2[z,0]=c[7, temp2]-camPts[i,(temp2*2)]
                    temp2 = temp2+1

                # if z%2==0:
                #     R=c[8,temp1]+c[9,temp1]+c[10,temp1]+1
                #     m1[z,0]=(camPts[i,(temp1*2)]*c[8,(temp1)]-c[0,(temp1)])/R
                #     m1[z,1]=(camPts[i,(temp1*2)]*c[9,(temp1)]-c[1,(temp1)])/R
                #     m1[z,2]=(camPts[i,(temp1*2)]*c[10,(temp1)]-c[2,(temp1)])/R
                #     m2[z,0]=(c[3,temp1]-camPts[i,(temp1*2)])/R
                #     temp1 = temp1+1
                # else:
                #     R=c[8,temp2]+c[9,temp2]+c[10,temp2]+1
                #     m1[z,0]=(camPts[i,(temp2*2)]*c[8,temp2]-c[4,temp2])/R
                #     m1[z,1]=(camPts[i,(temp2*2)]*c[9,temp2]-c[5,temp2])/R
                #     m1[z,2]=(camPts[i,(temp2*2)]*c[10,temp2]-c[6,temp2])/R
                #     m2[z,0]=(c[7, temp2]-camPts[i,(temp2*2)])/R
                #     temp2 = temp2+1
            
            #print(temp1)
            #print(temp2)  
            #get the least squares solution to the reconstruction
            if method == "qr_decomp":
                Q, R = np.linalg.qr(m1) # QR decomposition with qr function 
                y = np.dot(Q.T, m2) # Let y=Q'.B using matrix multiplication 
                x = np.linalg.solve(R, y) # Solve Rx=y
                xyz_pts = x.transpose()
            elif method == "np":
                xyz_pts = np.linalg.lstsq(m1,m2,rcond=None)[0].T #let np decides whats best
            else:
                assert False, "You wanna code your own? You think you are better than the thousands that came before you?"

            xyz[i,0:3]=xyz_pts
            #print(xyz)
            #compute ideal [u,v] for each camera
            #uv=m1*xyz[i-1,0:2].transpose
    
            #compute the number of degrees of freedom in the reconstruction
            #dof=m2.size-3
    
            #estimate the root mean square reconstruction error
            #rmse[i,1]=(sum((m2-uv)**2)/dof)^0.5
    
    return xyz

def easyWand_triangulate(pts,calib_dir,pipe_path,cam_list):
    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.genpath(f"{pipe_path}/setup/easyWand")
    xyz = eng.triangulate_middle_man(calib_dir,matlab.double(pts),cam_list)
    return np.array(xyz);

def determineDLTAuto(date,cams):
    '''
    Function to autkoamtically determine DLT coeff prefix.
    Rewrite this function if you have different coeffs than used circa 2024
    Do lists, so that code cna handle cases where there are multiple camera pairs to use
    '''
    prefix = None
    if int(date) < 220914:
        if "fly1" in cams and "bfs2" in cams:
            prefix = ["220412_no_f1bf2"]
        else:
            assert False, f"Need cams fly1 and bfs2 only have {', '.join(cams)}"
    if int(date) >= 220914:
        prefix = []
        if "fly1" in cams and "fly2" in cams:
            prefix.append("220914_f12_dlc")
        if "flea" in cams and "bfs1" in cams:
            prefix.append("220914_flea_bfs1_dlc")
        if len(prefix) == 0:
            assert False, f"Need cams fly1 & fly2 and/or cams bfs1 & flea, only have {', '.join(cams)}"
    return prefix



if __name__=="__main__":

    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Description of your script.")
    parser.add_argument("name", type=str, help="Experiment name/date")
    parser.add_argument("animal", type=str, help="Aminmal")
    parser.add_argument("--cond", type=str, help="Experiment condition")
    parser.add_argument("--pipe", type=str, help="Pipeline path")
    parser.add_argument("--step", type=int, help="Prematlab data extraction 1, post matlab 2")
    parser.add_argument("--coeff", type=list, help="Name of coef dir eg 220914_f12_dlc", default=None)

    args = parser.parse_args()

    ################ PARAMS
    step = args.step
    name = args.name
    expt_info_list = name.split('_')
    date = str(expt_info_list[0])
    expt = '_'.join(expt_info_list[1:])
    animal = args.animal
    condition = args.cond
    pipe = args.pipe
    prefixes = args.coeff

    #quick cam_dirs to make sure coefs and cams agree (e.g. cam used in coefs broke)
    metadat = load_yaml_config(f"{BASEDIR}/{animal}/{name}/metadat.yaml")
    cams = metadat["conditions_dict"][condition]["map_camname_to_path"].keys()

    if prefixes is None:
        #Rewrite this fxn with your own dates if needed
        prefixes = determineDLTAuto(date, cams)
    print (date, expt, condition, animal)

    ################# RUN
    V = Videos()
    V.load_data_wrapper(date=date, expt=expt, condition=condition, animal=animal)
    #Save originla cam dict so changes arent perm
    og_cam_dict = V.Params['load_params']['camera_names']

    #OLD DLT coeff bs

        # Load DLT coefficients
        # assert os.path.ispath(path)
    for prefix in prefixes:
        V.Params['load_params']['camera_names'] = og_cam_dict
        coef_path = f"{pipe}/dlt_coeffs/{prefix}"
        dlt_coefs = np.loadtxt(f"{coef_path}/dltCoefs.csv", delimiter=",")
        with open(f"{coef_path}/columns.csv", 'r') as file:
            coef_cols = file.read().splitlines()

        align_coefs = align_dlt_coefs(dlt_coefs=dlt_coefs, cols=coef_cols, V=V)
        assert dlt_coefs.shape == align_coefs.shape, "coefs not aligned, some formatting error mprobaly"
        print("LOADED DLT Coefs, with shape:")
        print(align_coefs.shape)


        ## Extract/import dlc data for each trial, in appropriate data structure.
        V.import_dlc_data()

        ## compute 3d pts and then save
        sdir = f"{V.Params['load_params']['basedir']}/{prefix}_extracted_dlc_data"
        import os
        os.makedirs(sdir, exist_ok=True)
        from pythonlib.tools.expttools import writeStringsToFile

        list_trials = V.inds_trials()
        list_part, _ = V.dlc_get_list_parts_feats()
        cams = V.Params['load_params']['camera_names']

        #Restrict cameras to just those foind in dlt coeffs, but in right order
        rest_cam_dict = {k:cam for k,cam in cams.items() if cam in coef_cols}
        #Change params so only searchging for cams we are using
        V.Params['load_params']['camera_names'] = 'rest_cam_dict'
        cam_list = list(rest_cam_dict.values())


        temp_dir_base=f"{pipe}/temp_matlab_files/"
        temp_dir = f"{temp_dir_base}/{animal}/{date}_{expt}/{prefix}"
        #Make relevant dirs in step 1, temp_dir should not already exists to avoid overwriting already extracted data
        if step == 1:
            os.makedirs(temp_dir_base, exist_ok=True)
            if os.path.exists(temp_dir):
                print(f"meowmeow temp_dir {temp_dir} already exists. Deleting")
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
            with open(f"{temp_dir}/cams.txt", 'w') as f:
                for cam in cam_list:
                    f.write(f"{cam}\n")
            np.savetxt(f"{temp_dir}/dltCoefs.txt",align_coefs,delimiter=',')
        skipped_trials =[]
        for trial in list_trials:
            for part in list_part:
                #Save only the columns with pts relevent to the dlt calibration
                pts, columns = V.dlc_extract_pts_matrix(trial, [part])
                #If no data skip
                if len(pts)+len(columns) == 0:
                    skipped_trials.append(trial)
                    continue
                pts_df = pd.DataFrame(pts)
                pts_df.columns = columns
                new_pts = []
                for col in columns:
                    for cam in cam_list:
                        if cam in col:
                            new_pts.append(pts_df[col].tolist())
                pts = new_pts
                if step == 1:
                    np.savetxt(f"{temp_dir}/pts_t{trial}.txt", np.array(pts))

                    # pts3 = easyWand_triangulate(pts=pts,calib_dir=calib,pipe_path=pipe,cam_list=cam_list)
                    # # pts3 = mean_of_stack(pts3_all)

                    # # pts3 = dlt_reconstruct(align_coefs, pts)
                if step == 2:
                    #Import triangulated data from matlab (could just have matlab save it but I think its easier to do all that here)
                    pts3 = pd.read_csv(f"{temp_dir}/xyz_pts_t{trial}.txt")
                    # export as finalize dataframes.
                    np.save(f"{sdir}/3d-part_{part}-trial_{trial}-dat.npy", pts3)
                    np.savetxt(f"{sdir}/3d-part_{part}-trial_{trial}-dat.txt", pts3, delimiter=",")
            #         np.savetxt(f"{sdir}/part_{part}-trial_{trial}-columns.csv", columns, delimiter=",")
                    writeStringsToFile(f"{sdir}/3d-part_{part}-trial_{trial}-columns.csv", columns)
        
                
                print("Extracted:", trial, part, "to", f"{sdir}/3d-part_{part}-trial_{trial}-dat.npy")


        # also save original DLC and delete temp_dir.
        # run campy extraction as
        if step == 2:
            list_trials_good = [t for t in list_trials if t not in skipped_trials]
            for trial in list_trials_good:
                
                #save cam list in data dir for future use (also in proper order)
                with open(f"{sdir}/cams.txt", 'w') as f:
                    for cam in cam_list:
                        f.write(f"{cam}\n")

                for i, cam in enumerate(cam_list):
                    datv = V.helper_index_good((cam, trial))
            #         datv = V.datgroup_extract_single_video_data2(i, trial, True)
                    dfthis = datv["data_dlc"]
                    dfthis.to_pickle(f"{sdir}/camera_{cam}_-trial_{trial}-dat.pkl")

            #         from pythonlib.tools.expttools import writeDictToYaml, makeTimeStamp
            #         V.Params["tstamp"] = makeTimeStamp()
            #         writeDictToYaml(V.Params, f"{sdir}/params.yaml")


            #         # export as finalize dataframes.
            #         np.save(f"{sdir}/part_{part}-trial_{trial}-dat.npy", pts)
            #         np.savetxt(f"{sdir}/part_{part}-trial_{trial}-dat.npy", pts, delimiter=",")
            # #         np.savetxt(f"{sdir}/part_{part}-trial_{trial}-columns.csv", columns, delimiter=",")
            #         writeStringsToFile(f"{sdir}/part_{part}-trial_{trial}-columns.csv", columns)
                    
                    print("Extracted original dlc data:", trial, "to", f"{sdir}/camera_{cam}_-trial_{trial}-dat.pkl")

            shutil.rmtree(temp_dir)
            V.campy_preprocess_check_frametimes()
            V.campy_export_to_ml2()
     