""" tools, using opencv
"""
import cv2
import numpy as np


def get_video_wh(video):
    """
    Return width and height of video in pix
    INPUT:
    - video, path to...
    e.g., path = "/data2/camera/210826_camtest5/behavior/cam1bfu/vid-t1.mp4"
    OUT:
    - w, h
    """

    vcap = cv2.VideoCapture(video)

    if vcap.isOpened(): 
        # get vcap property 
        width  = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    else:
        print(video)
        assert False

    return width, height


def calc_downscale_params(worig, horig, wnew=None, hnew=None):
    """
    Determine new w or h that maintains aspect ratio, but is divisible by 2.
    """

    def _round_to_nearest_mult_two(x):
        if np.ceil(x)%2==0:
            return np.ceil(x)
        else:
            return np.floor(x)
    
    ratio_wh = worig/horig

    if wnew is None:
        assert hnew is not None, "Can only give me one or other"
        wnew = ratio_wh * hnew
        wnew = _round_to_nearest_mult_two(wnew)
        return wnew
    else:
        assert hnew is None
        hnew = (1/ratio_wh) * wnew
        hnew = _round_to_nearest_mult_two(hnew)
        return hnew

