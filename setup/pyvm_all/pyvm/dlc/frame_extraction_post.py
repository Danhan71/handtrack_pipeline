"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

def extract_frames(
    config,
    numvids,
    numframestot,
    videos_list,
    algo="uniform",
    crop=False,
    userfeedback=False,
    cluster_step=1,
    cluster_resizewidth=30,
    cluster_color=False,
    opencv=True,
    slider_width=25,
    config3d=None,
    extracted_cam=0,
):
    import os
    import sys
    import re
    import glob
    import numpy as np
    import random
    from pathlib import Path
    from skimage import io
    from skimage.util import img_as_ubyte
    from deeplabcut.utils import frameselectiontools
    from deeplabcut.utils import auxiliaryfunctions

    config_file = Path(config).resolve()
    cfg = auxiliaryfunctions.read_config(config_file)
    print("Config file read successfully.")

    start = cfg["start"]
    stop = cfg["stop"]

    # Check for variable correctness
    if start > 1 or stop > 1 or start < 0 or stop < 0 or start >= stop:
        raise Exception(
            "Erroneous start or stop values. Please correct it in the config file."
        )
    if numframestot < 1 and not int(numframestot):
        raise Exception(
            "Perhaps consider extracting more, or a natural number of frames."
        )
    videos_all = videos_list

    if numvids > len(videos_all):
        print("######### Actual number of videos is less than the number of videos this function was told to pick, defaulting to actual...")
        numvids = len(videos_all)

    #This ensures we always a number of frames equal to the indictated total
    numframes2pick = int(np.ceil(numframestot/numvids))

    #Sample subset of videos to extaact frames from
    videos = random.sample(videos_all, numvids)

        
    if opencv:
        from deeplabcut.utils.auxfun_videos import VideoReader
    else:
        from moviepy.editor import VideoFileClip

    has_failed = []
    for video in videos:
         
        cap = VideoReader(video)
        nframes = len(cap)
        if not nframes:
            print("Video could not be opened. Skipping...")
            continue


        indexlength = int(np.ceil(np.log10(nframes)))

        fname = Path(video)
        output_path = Path(config).parents[0] / "check-labeled-data"
        if not output_path.exists():
            os.mkdir(output_path)

        if crop and not opencv:
            clip = clip.crop(
                y1=int(coords[2]),
                y2=int(coords[3]),
                x1=int(coords[0]),
                x2=int(coords[1]),
            )
        elif not crop:
            coords = None

        print("Extracting frames based on %s ..." % algo)
        if algo == "uniform":
            if opencv:
                frames2pick = frameselectiontools.UniformFramescv2(
                    cap, numframes2pick, start, stop
                )
            else:
                frames2pick = frameselectiontools.UniformFrames(
                    clip, numframes2pick, start, stop
                )
        elif algo == "kmeans":
            if opencv:
                frames2pick = frameselectiontools.KmeansbasedFrameselectioncv2(
                    cap,
                    numframes2pick,
                    start,
                    stop,
                    crop,
                    coords,
                    step=cluster_step,
                    resizewidth=cluster_resizewidth,
                    color=cluster_color,
                )
            else:
                frames2pick = frameselectiontools.KmeansbasedFrameselection(
                    clip,
                    numframes2pick,
                    start,
                    stop,
                    step=cluster_step,
                    resizewidth=cluster_resizewidth,
                    color=cluster_color,
                )

        if not len(frames2pick):
            print("Frame selection failed...")
            return

        is_valid = []
        if opencv:
            for index in frames2pick:
                cap.set_to_frame(index)  # extract a particular frame
                frame = cap.read_frame()
                if frame is not None:
                    image = img_as_ubyte(frame)
                    img_name = (
                        str(output_path)
                        + "/img"
                        + str(index).zfill(indexlength)
                        + ".png"
                    )
                    if crop:
                        io.imsave(
                            img_name,
                            image[
                                int(coords[2]) : int(coords[3]),
                                int(coords[0]) : int(coords[1]),
                                :,
                            ],
                        )  # y1 = int(coords[2]),y2 = int(coords[3]),x1 = int(coords[0]), x2 = int(coords[1]
                    else:
                        io.imsave(img_name, image)
                    is_valid.append(True)
                else:
                    print("Frame", index, " not found!")
                    is_valid.append(False)
            cap.close()
        else:
            for index in frames2pick:
                try:
                    image = img_as_ubyte(clip.get_frame(index * 1.0 / clip.fps))
                    img_name = (
                        str(output_path)
                        + "/img"
                        + str(index).zfill(indexlength)
                        + ".png"
                    )
                    io.imsave(img_name, image)
                    if np.var(image) == 0:  # constant image
                        print(
                            "Seems like black/constant images are extracted from your video. Perhaps consider using opencv under the hood, by setting: opencv=True"
                        )
                    is_valid.append(True)
                except FileNotFoundError:
                    print("Frame # ", index, " does not exist.")
                    is_valid.append(False)
            clip.close()
            del clip

        if not any(is_valid):
            has_failed.append(True)
        else:
            has_failed.append(False)

    if all(has_failed):
        print("Frame extraction failed. Video files must be corrupted.")
        return
    elif any(has_failed):
        print("Although most frames were extracted, some were invalid.")
    else:
        print(
            "Frames were successfully extracted for a random sample of the labelled videos."
        )


if __name__=="__main__":
    import argparse
    from pythonlib.tools.expttools import findPath
    from pyvm.globals import BASEDIR
    from initialize import find_expt_config_paths

    parser = argparse.ArgumentParser(description="Description of your script.")
    parser.add_argument("name", type=str, help="Experiment name/date")
    parser.add_argument("animal", type=str, help="Would you like to know pony boy")
    parser.add_argument("--numvids", type=str, help="Number of videos to pick", default=20, required=False)
    parser.add_argument("--numframes", type=str, default=200, required=False, help="Number of frames to pick TOTAL (i.e. numvids*numframes/vid this is to ensure that this many frames are selected if there are fewer videos than indicated as this is mos timportat)")

    args = parser.parse_args()

    name = args.name
    animal = args.animal

    videos_list = findPath(f"{BASEDIR}/{animal}/{name}/behavior/DLC", [["combined"],["allvideos"]], ext = ".mp4")
    numvids = args.numvids
    numframes = args.numframes

    dict_paths, _ = find_expt_config_paths(name, "behavior")
    pcf = list(dict_paths.values())[0]

    extract_frames(videos_list=videos_list, config = pcf, numvids=numvids, numframestot=numframes)


