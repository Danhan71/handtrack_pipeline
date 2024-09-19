import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_checkerboards(images, patternSize=(9,6)):
    """
    INPUTS:
    - images, list of paths to images
    - patternSize, 
    """
    
    assert len(images)>0, "probably you need to enter frames in metadata, in field good_frames_checkerboard"

    # --- object points (3d)
    objp = np.zeros((patternSize[0]*patternSize[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:patternSize[0],0:patternSize[1]].T.reshape(-1,2)
    
    FLAGS = cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_LARGER
    
    # ---
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objpts = []
    imgpts = []
    successes = []
    figlist = []
    for fname in images:
        img = cv2.imread(fname)
        assert img is not None, f"cant find {fname}"
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # === Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, patternSize, flags = FLAGS)
        successes.append(ret)

        # === PLOT
        fig, axes = plt.subplots(1,2, figsize=(30, 15))
        figlist.append(fig)

        axes[0].imshow(gray, cmap = 'gray', interpolation = 'bicubic')
        axes[0].set_title(fname)

        objpts.append(objp)

        if ret == True:

            corners2 = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), criteria)
            imgpts.append(corners2)

            # Draw and display the corners
            img1 = cv2.drawChessboardCorners(img, patternSize, corners, ret)
            img2 = cv2.drawChessboardCorners(img, patternSize, corners2, ret)
            # axes[1].imshow(img, cmap = 'gray', interpolation = 'bicubic')
            axes[1].imshow(img2, cmap = 'gray', interpolation = 'bicubic')

#             plt.title(fname)
#             plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
#             plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

    #         plt.show()
    #         cv2.imshow('img',img)
    #         cv2.waitKey(500)
        else:
            imgpts.append(np.empty(0))

    return successes, objpts, imgpts, gray.shape[::-1], figlist

