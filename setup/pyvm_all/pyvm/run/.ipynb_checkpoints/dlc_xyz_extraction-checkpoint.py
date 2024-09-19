# """
# final processed DLC (x,y,z) coordinates --> export for use in ml2
# """

from pyvm.classes.videoclass import Videos
import os
import numpy as np
from pythonlib.tools.expttools import writeStringsToFile



#dlt reconstruct adapted by An Chi Chen from DLTdv5 by Tyson Hedrick
# https://github.com/MMathisLab/3D-DeepLabCut-Cheetah/blob/master/3D_Evaluation/dlt_reconstruct.py
    

import numpy as np
import csv

def dlt_reconstruct(c,camPts):
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

            temp1 = 1
            temp2 = 1
            for z in range(cdx_size*2):
                if z%2==0:
                    m1[z,0]=camPts[i-1,(temp1*2)-2]*c[8,(temp1-1)]-c[0,(temp1-1)]
                    m1[z,1]=camPts[i-1,(temp1*2)-2]*c[9,(temp1-1)]-c[1,(temp1-1)]
                    m1[z,2]=camPts[i-1,(temp1*2)-2]*c[10,(temp1-1)]-c[2,(temp1-1)]
                    m2[z,0]=c[3,temp1-1]-camPts[i-1,(temp1*2)-2]
                    temp1 = temp1+1
                else:
                    m1[z,0]=camPts[i-1,(temp2*2)-1]*c[8,temp2-1]-c[4,temp2-1]
                    m1[z,1]=camPts[i-1,(temp2*2)-1]*c[9,temp2-1]-c[5,temp2-1]
                    m1[z,2]=camPts[i-1,(temp2*2)-1]*c[10,temp2-1]-c[6,temp2-1]
                    m2[z,0]=c[7, temp2-1]-camPts[i-1,(temp2*2)-1]
                    temp2 = temp2+1
            
            #print(temp1)
            #print(temp2)  
            #get the least squares solution to the reconstruction
            Q, R = np.linalg.qr(m1) # QR decomposition with qr function 
            y = np.dot(Q.T, m2) # Let y=Q'.B using matrix multiplication 
            x = np.linalg.solve(R, y) # Solve Rx=y
            xyz_pts = x.transpose()
            
            xyz[i,0:3]=xyz_pts
            #print(xyz)
            #compute ideal [u,v] for each camera
            #uv=m1*xyz[i-1,0:2].transpose
    
            #compute the number of degrees of freedom in the reconstruction
            #dof=m2.size-3
    
            #estimate the root mean square reconstruction error
            #rmse[i,1]=(sum((m2-uv)**2)/dof)^0.5
    
    return xyz


if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Description of your script.")
    parser.add_argument("name", type=str, help="Experiment name/date")
    parser.add_argument("--cond", type=str, help="Experiment condition")
    parser.add_argument("--coeff", type=str, help="Dir of dlt coeffs, from pipeline config file")

    args = parser.parse_args()

    ################ PARAMS
    expt = args.name
    condition = args.cond
    path = args.coeff # File in /wand/wand_calibration.
    print (expt, condition, path)

    ################# RUN
    V = Videos()
    V.load_data_wrapper(expt, condition)

    ## Load DLT coefficients
    # assert os.path.ispath(path)
    dlt_coeffs = np.loadtxt(path, delimiter=",")
    print("LOADED DLT Coeffs, with shape:")
    print(dlt_coeffs.shape)

    ## Extract/import dlc data for each trial, in appropriate data structure.
    V.import_dlc_data()

    ## compute 3d pts and then save
    sdir = f"{V.Params['load_params']['basedir']}/extracted_dlc_data"
    import os
    os.makedirs(sdir, exist_ok=True)
    from pythonlib.tools.expttools import writeStringsToFile

    list_trials = V.inds_trials()
    list_part, _ = V.dlc_get_list_parts_feats()

    for trial in list_trials:
        for part in list_part:
            pts, columns = V.dlc_extract_pts_matrix(trial, [part])
            
            pts3 = dlt_reconstruct(dlt_coeffs, pts)

            # export as finalize dataframes.
            np.save(f"{sdir}/3d-part_{part}-trial_{trial}-dat.npy", pts3)
            np.savetxt(f"{sdir}/3d-part_{part}-trial_{trial}-dat.txt", pts3, delimiter=",")
    #         np.savetxt(f"{sdir}/part_{part}-trial_{trial}-columns.csv", columns, delimiter=",")
            writeStringsToFile(f"{sdir}/3d-part_{part}-trial_{trial}-columns.csv", columns)
            
            print("Extracted:", trial, part, "to", f"{sdir}/3d-part_{part}-trial_{trial}-dat.npy")


    # also save original DLC.
    for trial in list_trials:
        
        for i, cam in V.get_cameras().items():
            datv = V.helper_index_good((cam[0], trial))
    #         datv = V.datgroup_extract_single_video_data2(i, trial, True)
            dfthis = datv["data_dlc"]
            dfthis.to_pickle(f"{sdir}/camera_{cam[0]}_-trial_{trial}-dat.pkl")

    #         from pythonlib.tools.expttools import writeDictToYaml, makeTimeStamp
    #         V.Params["tstamp"] = makeTimeStamp()
    #         writeDictToYaml(V.Params, f"{sdir}/params.yaml")


    #         # export as finalize dataframes.
    #         np.save(f"{sdir}/part_{part}-trial_{trial}-dat.npy", pts)
    #         np.savetxt(f"{sdir}/part_{part}-trial_{trial}-dat.npy", pts, delimiter=",")
    # #         np.savetxt(f"{sdir}/part_{part}-trial_{trial}-columns.csv", columns, delimiter=",")
    #         writeStringsToFile(f"{sdir}/part_{part}-trial_{trial}-columns.csv", columns)
            
            print("Extracted original dlc data:", trial, "to", f"{sdir}/camera_{cam[0]}_-trial_{trial}-dat.pkl")        