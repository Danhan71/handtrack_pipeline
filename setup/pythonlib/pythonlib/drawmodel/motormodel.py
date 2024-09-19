""" for modeling motor-based beahviroal measurese.g., timecourse, velocities,
curvature, etc. Other model, e.,g in program, is based on segmented, categories
strokes, not on their moment-by-moemnt statstics."""

import numpy as np
import matplotlib.pyplot as plt
from pythonlib.tools.stroketools import *




class strokModel(object):
    """model for single strok. 
    a list of strok make a single stroke
    - a strok made of a sequence of substroks.
    each substrok modeled as a line, with bell-shaped
    velocity. multiple substroks overlaied to
    make a strok. overlay can be diff amount of overlap,
    which can be like concatenation."""

    def __init__(self, fs, vec_over_spatial_ratio=1):
        # self.program = program 
        """ for vec_over_spatial_ratio, see below. 
        this is to multiply spatial score by this amoount to
        put on same sacle as vel. around 3-5 is good, I think"""
        self.fs = fs
        self.vec_over_spatial_ratio = vec_over_spatial_ratio
        self.lowpass_freq_vel = 5

    def getTimecourseGrounded(self, ontime, offtime, totaltime, fs,
                             sm_win = 0.15):
        """ gets sampled timeecourse, in contrast
        getTimecourse gets continuous function 
        - will smooth, mainly to remove discontinuites at
        on and off time, which leads to spikes in veloitye.s
        - e.g.
        ontime = 0.1
        offtime = 0.9
        totaltime = 1.0
        fs = 125
        """
        from ..tools.timeseriestools import getSampleTimesteps, smoothDat
        # get continous function (0,1 valued)
        U = self.getTimecourse(ontime, offtime, totaltime)

        # get discretized timeorouse
        t = getSampleTimesteps(totaltime, fs, force_T=True)[0]
        pos = np.array([U(tt) for tt in t])
        # print(len(pos))
        # print(len(t))

        # === smooth
        window_len = int(np.round(sm_win*fs))
        # window_len = sm_win*fs
        # print(len(pos))
        pos = smoothDat(pos, window_len=window_len)
        # print(len(pos))

        if False:
            plt.figure()
            plt.plot(pos)

        return pos, t

    def getTimecourse(self, ontime, offtime, totaltime,
        enforce_on_off_inbounds=True):
        """ generic "speed" function, which takes in onset time and offset time, and
        then given a particular timepoint, outputs a value from 0 to 1 with a 
        velocity profile that is bell shaped.it will have sustained 0 and 1 at all times
        before onset and offset. 
        - the actual time it takes (i.e, num samples, given some sample rate) depends on
        T, the total time
        - the derivative of this function will look like a bell-shaped velocity profile.
        - returns the function u(t), which takes in real-valued t. to discretize,
        pass in t at sample locations. see below other functions./
        """
        if enforce_on_off_inbounds:
            # then dont allow on and off outside of(0, totaltime). in 
            # principle code should allow outside.
            assert ontime<offtime and ontime>=0 and offtime<=totaltime

        # rescale input so times are between (0,1)
        on = ontime/totaltime
        off = offtime/totaltime
        
        def U(t):
            i = t/totaltime # value between 0 and 1 (index)
            if i<on:
                return 0.
            elif i>off:
                return 1.
            else:
                u = (i-on)/(off-on)
                
                # if u is 0.5, then is in middle, use lgistics.
                # if at edges then is 0 or 1. use quadratic.
                # smothyl morph between. 
                w = 2*(0.5 - np.abs(u-0.5))
                # w = 0

                # 1) logistic function (more at center)
                x = u-0.5
                f1 = 1/(1+np.exp(-10*(x)))
                f1*=w
                # tmp = tmp + (1-w)*(u + 0.5)
    #             print(u)
                
                # 2) quadratic functions at edges.
                # f2 = (1-w)*u
                a = 1
                if u<=0.5:
                    f2 = a*u**2.5
                else:
                    f2 = 1 - a*(1-u)**2.5
                f2*=(1-w)

                return f1+f2
        if False:
            # for debugging:
            ton = 0.1
            toff = 0.9
            T = 1.0
            fs = 100

            pos, t = self.getTimecourseGrounded(ton, toff, T, fs,
                                     sm_win = 0.15)
            plt.figure()
            plt.plot(pos)

        return U


    def synthesize(self, program, ploton=False):
        """ given program of substrokes, create
         a concrete strok (a lsit of strok is a stroke).
         e.g, program:
         program = {
        "substroks":[
            (0, 0.65, 0, 100),    
            (0.35, 1, pi/2, 200)],# for each stroke in the program, specific a few things (onset, offset, rotation, length), in units (sec, sec, rad(from 1,0), pix)
        "totaltime":1., # total duration of stroke 
        "fs" = 125}
        - returns a single strok, np array T x 3
        """
        from ..drawmodel import primitives as P
        self.program = program
        # program = self.program
        
        substroklist =[]
        T = program["totaltime"]
        fs = program["fs"]
        for prog in program["substroks"]:

            # 1) get timecourse for each substroke (unitless)
            pos, t = self.getTimecourseGrounded(prog[0], prog[1], T, fs)

            # 2) get timecourse * position for each substroke
            s = P.transform(P._line, s=prog[3], theta=prog[2])[0][1,:] # coord of endpoint of line

            # 3) multiply timecourse by vector position to get stroke.
            substrok = pos[:, None] * s[None, :]
        #     substrok = np.concatenate((substrok, t[:,None]), axis=1)

            substroklist.append(substrok)

        # 3) overlay substrokes
        strok = np.sum(np.stack(substroklist, axis=2), axis=2)
        # append timesteps
        strok = np.concatenate((strok, t[:, None]), axis=1)
        
        if ploton:
            from ..drawmodel.strokePlots import plotDatStrokes
            fig, axes = plt.subplots(1,1)
            plotDatStrokes([strok], axes)
            strokesVelocity([strok], fs=125, ploton=True)
            # plotDatStrokes([strok_beh], axes[0])
            # plotDatStrokes([strok_mod], axes[0])
            # plotDatStrokesTimecourse([strok_beh_vel], axes[1])
            # plt.ylabel("beh vel")
            # plotDatStrokesTimecourse([strok_mod_vel], axes[2])
            # plt.ylabel("mod vel")

        self.strok = strok
        return strok

    def scoreVsBeh(self, strok_beh, ploton=False):
        """ score that, by default, compares poitn by point
        vs. beahvior, against both spatial and vel distance 
        """
        from pythonlib.tools.distfunctools import _distStrokTimeptsMatched
        
        assert strok_beh.shape == self.strok.shape
        
        dist = _distStrokTimeptsMatched(strok_beh, self.strok, self.fs,
            vec_over_spatial_ratio =self.vec_over_spatial_ratio, 
            lowpass_freq = self.lowpass_freq_vel, ploton=ploton)
        
        return dist


    # ========== FOR FITTING TO BEHAVIOR
    def getCostFunc(self, strok_beh, program_func):
        """ strok_beh doesn't hvae to be preprocessed.
        will do it all here. stroke onset to page center.
        and to t=0
        - program_func is a function that takes in params and
        returns a program(dict)
        e.g,
        """

        # shift strok so start at time 0 (checked, this only mods a copy)
        # and start at 0,0
        strok_beh -= strok_beh[0,:]
        T = strok_beh[-1,2] - strok_beh[0,2]

        def func(params, ploton=False):

            program = program_func(params)
            # program = {
            # "substroks":[
            #     (0, t1*T, theta1, l1),    
            #     (t2*T, T, theta2, l2)],
            # "totaltime":T,
            # "fs":fs}

            # print(program)
            strok_mod = self.synthesize(program)
            cost = self.scoreVsBeh(strok_beh, ploton=ploton)

            return cost

        return func






