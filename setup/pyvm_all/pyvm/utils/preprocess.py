""" preprocessing videos"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pyvm.utils.directories import getPaths


def count_frames(path_video):
    """ Count frames manually by iterating over extracted frames
    RETURN:
    - currentframe, number of successful frames.
    """
    # open video
    cap = cv2.VideoCapture(path_video)

    # collect list of frames
    currentframe=0
    while True:
        success, _ = cap.read()

        if not success:
            break
        # increasing counter so that it will 
        # show how many frames are created 
        currentframe += 1
    return currentframe

def get_frames_from_video(path_video, list_frames=None, savedir_frames = None):
    """ get list of frames (cv2 objects).
    INPUT:
    - list_frames,
    --- list of ints will return frames, but SORTED!!!
    --- None, will return all frames.
    - savedir_frames, then will save, and return None.
    NOTE:
    - frames will always be in chron order in video, EVEN IF lsit_frames is not in order
    """
    if list_frames is not None:
        if sorted(list_frames)!=list_frames:
            assert False

    # open video
    cap = cv2.VideoCapture(path_video)

    if list_frames is not None:
        if not isinstance(list_frames[0], int):
            print(list_frames)
            assert False

    print("This video loading: ", path_video)
    # collect list of frames
    success = True
    currentframe=0
    list_gotten_frames = []
    while success:
        success, frame = cap.read()
        if success: 
            # print("--")
            # print(currentframe)
            if list_frames is None:
                # then take this frame
                list_gotten_frames.append(frame)
            elif currentframe in list_frames:
                # then take this frame
                list_gotten_frames.append(frame)


            # increasing counter so that it will 
            # show how many frames are created 
            currentframe += 1
        else: 
            print(currentframe, "Video end")
            break

    if list_frames is not None:
        if len(list_gotten_frames)!=len(list_frames):
            print(currentframe)
            print(list_gotten_frames)
            print(list_frames)
            print(len(list_gotten_frames))
            print(len(list_frames))

            assert False, "why didnt get all frames, is becasue not enough frames in vid? check currentframe"

    return list_gotten_frames


def extractFrames(path_video, suffix=None, skip_if_already_extracted=True):
    """ extracts frames, saved in a subdir
    - path_video, full path to video file.
    RETURNS:
    - saves all frames in <dir of video>/<video name>-frames/frame1.jpg and so on.
    - returns None
    """
    import os
    path_frames = getPaths(path_video)["path_frames"]
    if suffix:
        path_frames += f"-{suffix}"
    if os.path.exists(path_frames) and skip_if_already_extracted:
        # Then skip
        return
    os.makedirs(path_frames, exist_ok=True)
    print("Extracting to:", path_frames)
    # collect list of frames
    list_gotten_frames = get_frames_from_video(path_video)
    for i, frame in enumerate(list_gotten_frames):
        name = f"{path_frames}/frame{i}.jpg"
        print ('Creating...' + name) 

        # writing the extracted images 
        cv2.imwrite(name, frame) 

def collectFrames(path_video_list, frames_list, output_path=None):
    """ collect specific hand-provided frames cross different videos
    and put all into a single folder.
    - path_video_list, each element is full path to a video. 
    - frames_list, list of list, where each sub-list holds frame numbers, 
    indices start at 0.
    - output_path, dir to copy frames. leave as None if want to use
    based dir for path_video. NOTE: This would make separate fir for each
    video, (with defualt dir name = "collected_frames")
    NOTE: you must have already run extractFrames, putting frames into
    a subdirectory
    """
    from shutil import copyfile
    from pythonlib.tools.expttools import writeStringsToFile, writeDictToYaml
    assert len(path_video_list)==len(frames_list)


    def _finalize(path_collected_frames, fnames_in_order_new, fnames_in_order, framenums):
        # save text file mapping between old and new names
        stringlist = [f"{new}, {old}" for new, old in zip(fnames_in_order_new, fnames_in_order)]
        writeStringsToFile(f"{path_collected_frames}/fnames.txt", stringlist)            

        # stringlist = [f"{num[0]}, {num[1]}" for num in framenums]
        # writeStringsToFile(f"{path_collected_frames}/framenums_old_new.txt", stringlist)            
        writeDictToYaml(framenums, f"{path_collected_frames}/framenums_old_new.yaml")            

        # reset
        fnames_in_order = []
        fnames_in_order_new = []
        ct = 0

        return ct, fnames_in_order, fnames_in_order_new

    # number frames anew.
    ct = 0
    framenums = [] # list of tuples, (old, new) numbers
    fnames_in_order = []
    fnames_in_order_new = []
    for pv, fs in zip(path_video_list, frames_list):
        
        # path to hold frames.
        if output_path is not None:
            # then you have given me one path to combine all the videos
            path_collected_frames = output_path
            os.makedirs(path_collected_frames, exist_ok=True)
        else:
            # separate path for each video.
            path_dir = getPaths(pv)["path_dir"]
            path_collected_frames = f"{path_dir}/collected_frames"
            os.makedirs(path_collected_frames, exist_ok=False)
            print("- Saving collected frames to:")
            print(path_collected_frames)

        # where frames are stored
        path_frames = getPaths(pv)["path_frames"]

        # copy frames
        print(path_frames)
        if os.path.isdir(path_frames):
            for framenum in fs:
                path = f"{path_frames}/frame{framenum}.jpg"
                path_new = f"{path_collected_frames}/frame-{ct}.jpg"
                # Then just copy over previouslty extract frames
                copyfile(path, path_new)

                fnames_in_order.append(path)
                fnames_in_order_new.append(f"frame-{ct}.jpg")

                framenums.append((framenum, ct))
                ct += 1
        else:        
            # reextract frames
            # list_frames = get_frames_from_video(pv, list_frames=[fs])
            list_frames = get_frames_from_video(pv, list_frames=fs)

            # writing the extracted images 
            for i, (frame, oldnum) in enumerate(zip(list_frames, fs)):
                path_new = f"{path_collected_frames}/frame-{i}.jpg"
                print ('Creating...' + path_new) 

                # writing the extracted images 
                # cv2.imwrite(name, frame) 
                cv2.imwrite(path_new, frame) 

                fnames_in_order.append(f"frame_{oldnum}")
                fnames_in_order_new.append(f"frame-{i}.jpg")

                framenums.append((oldnum, i))
                ct = len(list_frames)

        # Final things to do
        if output_path is None:
            ct, fnames_in_order, fnames_in_order_new = _finalize(path_collected_frames, 
                fnames_in_order_new, fnames_in_order, framenums)


    if output_path is not None:
       ct, fnames_in_order, fnames_in_order_new = _finalize(path_collected_frames, 
        fnames_in_order_new, fnames_in_order, framenums)


   

