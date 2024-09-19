""" plotting, for cv stuff"""
import cv2
from .preprocess import getPaths
import matplotlib.pyplot as plt
import numpy as np

def plotImageByFrameNum(path_video, frame_num, ax=None):
    """ 
    single plot of this frame for this video.
    """

    # find and load image
    pf = getPaths(path_video)["path_frames"]
    assert False, "continue here. goal is to try stereo calib, but first have to get pairs of frames."

    if not ax:
        fig, ax = plt.subplots(1,1, figsize=(8,8))
    else:
        plotImage()
        ax.imshow(gray, cmap = 'gray', interpolation = 'bicubic')


def plotImage(img, ax=None):
    """
     img is output of cv2.imread(path)
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if not ax:
        fig, ax = plt.subplots(1,1, figsize=(8,8))
    else:
        ax.imshow(gray, cmap = 'gray', interpolation = 'bicubic')
        # axes[0].set_title(fname)


def plot_frametime_summary(frametimes):
    """ 
    INPUT:
    - frametimes, array of times, assumes sec, but doesnt matter.
    """

    nrows = 3
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4))

    # ons and offs
    ax = axes.flatten()[0]
    ax.plot(frametimes, np.ones_like(frametimes), 'xr')
    ax.set_xlabel("frametimes")

    # ax.plot(nums, times, 'o-k')

    # interframe
    ifi = np.diff(frametimes)
    ax = axes.flatten()[1]
    ax.plot(range(len(ifi)), ifi, 'xr')
    ax.set_title("inter frame intervals in chron order")

    return fig, axes
    


