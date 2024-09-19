
def plotTrialMovie(filedata, trial, savedir="/tmp", 
                  savename = None):
    """ save movie for this trial.
    - savename, if None, then uses trial as name, 
    - NOTE: currently timing is correct for within
    stroke, but gaps are instantaneosu. 
    - NOTE: not showing, byut must save
    """
    from .utils import getTrialsStrokesByPeanuts
    from pythonlib.tools.stroketools import strokesInterpolate2
#     %matplotlib inline
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    strokes = getTrialsStrokesByPeanuts(filedata, trial)
    fsnew = 125
    fsold = 125
    print("REPLACE FS WITH OPARAMS")
          
    # OPTION 1 - KEEP STROKE STRUCTURE, INTERPOLATE THEN FLATTEN
    # problem is ignores gaps.
    if True:

        # print(strokes)
        strokes = strokesInterpolate2(strokes, N= ["fsnew", fsnew, fsold])

        # print(strokes)
        # x = strokes[1][:,0]
        # y = strokes[1][:,1]

        x = [ss[0] for s in strokes for ss in s]
        y = [ss[1] for s in strokes for ss in s]
        # y = [s[:,1] for s in strokes]

    # == first flatten before interpolate, in order to keep gaps correct timing as well.
    # Problem is it interpolates pts in gaps, which arent really drawn.
    else:
        strokesflat = [ss for s in strokes for ss in s]
        strokesflat = [np.array(strokesflat)]
        strokesflat
        strokesflat = strokesInterpolate2(strokesflat, N= ["fsnew", fsnew, fsold])
        x = [ss[0] for s in strokesflat for ss in s]
        y = [ss[1] for s in strokesflat for ss in s]


    fig, ax = plt.subplots()
    line, = ax.plot(x, y, "ok")
    # line = ax.scatter(x, y, color='k')
    
    if False:
        # use entire sketchpad
        lims = [-768/2, 768/2, -1024/2, 1024/2]
    else:
        lims = [min(x), max(x), min(y), max(y)]
    
    def update(num, x, y, line):
        line.set_data(x[:num], y[:num])
        line.axes.axis(lims)
        return line

    ani = animation.FuncAnimation(fig, update, len(x), fargs=[x, y, line],
                                  interval=1000/fsnew, blit=False)
    
    if savename is None:
        savename = f"trial{trial}"
    print("Saving at:")
    print(f"{savedir}/{savename}.mp4")
    ani.save(f"{savedir}/{savename}.mp4", writer="ffmpeg")
    # plt.show()
    