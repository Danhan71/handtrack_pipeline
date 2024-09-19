""" useful code to take strokes (or trialstrokes in a datflat object) and segment into useful things.
used extensively for human drawgood - cogsci paper."""
import numpy as np
import math
from ..tools.stroketools import getOnOff

def getxgrid(datseg_single, appendExtreme=False):
    # -- finds all the LL, and uses those as proxies for the x positions. outputs those positions as xgrids.
    # only outputs unique positions
    
    # 1) get x location of columns by looking at the fours LLs
    centerpos = [d["centerpos"] for d in datseg_single]
    codes = [d["codes"] for d in datseg_single]
    
    centers = [c[0] for c, d in zip(centerpos, codes) if d=="LL"] # xvals, just for the LL primitives.
    centers = np.array(sorted(centers)) # sort by orfer from left to right
    TOL = 0.1
    xgrid = np.unique(np.ceil(centers/TOL).astype(int))*TOL # to take unique, must first round away precision errors.
    
    if appendExtreme:
        xgrid = np.concatenate((np.array([-1000]), xgrid)) # add, so that x_extremes will always be between two grid edges. 

    return xgrid

def getgridspacing(grid):
    return np.mean(np.diff(np.array(grid)))


# --- go thru each primtive by code. for each code type, get unique code.
def getUniqueCode(datseg_single):
    """give one sequence. will give the primitives unique codes (e..g, not just C for circle, but C11, C10"""
    # does this by using x and y positions in entire drawing compostion.
    # looks for LL (vertical lines) - this establishes the grid.
    # x and y positions are used to name pritiives, e.g. C12 is x1, y2.
    # IMPORTANT: this only makes sense for stimuli that have vertical structure and have LL. E>g,, sets S8 and S9.
    # would work for other things that have at least one LL, but may not be interpretable.
    # NOTE: mutates in place

    if isinstance(datseg_single, dict):
        datseg_single = [datseg_single]

    # print("GETTING UNIQUE CODES")
            
    def getunique(datseg_single_subsample, xgrid):
        # use this on subset of strokes that have same code (e.g, all circles...)
        # for the set of primtives entered, will output their x and y position within an imaginary grid 
        # x_extremes is set of x (l and r) extremes
        # xgrid are x positions of grid coordinates. 
        # for each case in x_extremes, will use the left position to determine which grid it is overlaying on (i.e. skewered)
        # then, for each skewer, gets order along y axis.    

        # --- get data for these strokes
        assert isinstance(datseg_single_subsample, (list, tuple))
        # print(datseg_single_subsample[0])
        # print("this is number of strokes for this type")
        # print(len(datseg_single_subsample))
        x_extremes = [d["x_extremes"] for d in datseg_single_subsample]
        y_centers = np.array([d["centerpos"][1] for d in datseg_single_subsample])                         
        xgrid = np.concatenate((np.array([-1000]), xgrid)) # add, so that x_extremes will always be between two grid edges. 

        xpos_all = []
        # print("====x extremes")
        # print(x_extremes)
        # print("datseg single subsample")
        # print(datseg_single_subsample)
        # print("---x extremes")
        # for xx in x_extremes:
        #     print("--")
        #     print(xx)

        for xx in x_extremes:
            # --- first define x grid position based on position of leftmost point of primitive
            # xx[0] + 0.025*(xx[1]-xx[0])
            # xx[1] - 0.025*(xx[1]-xx[0])
            # A = xgrid[1:] >= xx[0]
            # B = xgrid[:-1] < xx[0]
            # xpos = np.argwhere(A & B)
            # print(A)
            # print(B)
            # print(xpos)
            # assert len(xpos)==1
            # xpos = xpos[0]
            # print("--")
            # print(xx)
            xpos_all.append(getGridLocation(xgrid, xx[0], addtogrid=0.20))

        # --- figure out the y position, for all the ones that are in this xpos, get its y position
        ypos_all = np.empty(len(xpos_all))
        # print(numxpos):
        # print("these are all the xpositions for this stroke type")
        # print(xpos_all)

        for x in np.unique(xpos_all):
            # -- for each xposition, find all its members and their y positions
            idx = xpos_all==x
            # print("--- sdafsdf")
            # # print(x)
            # print("these are indices {} that share the same xposition {}".format(idx, x))
            # print("these are the ycenters for these inds")
            # print(y_centers[idx])
            # print("these are the sorted indea for those y centers")
            # print(np.argsort(np.argsort(-y_centers[idx])))
            ypos_all[idx] = np.argsort(np.argsort(-y_centers[idx]))
            # print(ypos_all[idx])
            # print("---")
        # print("this is ypos for each xpos")
        # print(ypos_all)
        ypos_all = [int(y) for y in ypos_all]
        return xpos_all, ypos_all

    
    # ============ RUN
    codes = [d["codes"] for d in datseg_single]
    codelist = set([d["codes"] for d in datseg_single])

    # -- 1) get the grid for this program
    xgrid = getxgrid(datseg_single)
    codes_unique = [None]*len(codes)
    for code in codelist:

        # find the strokes 
        inds = [i for i, x in enumerate(codes) if x==code]
        datseg_single_subsample = [datseg_single[i] for i in inds]
        # datseg_single_subsample = itemgetter(*inds)(datseg_single)
        if isinstance(datseg_single_subsample, dict):
            datseg_single_subsample = (datseg_single_subsample)
        # find their unique code
        # print("ASASDASD")
        # print(type(datseg_single_subsample))
        # print(inds)
        # assert isinstance(datseg_single_subsample, list)
        xpos, ypos = getunique(datseg_single_subsample, xgrid)

        # create new unique code name
        for i, x, y in zip(inds, xpos, ypos):
            codes_unique[i] = code + str(x) + str(y)

    # --- put back into datseg_single_subsample
    for c, d in zip(codes_unique, datseg_single):
        d["codes_unique"] = c

    # OUTPUT: adds new field to  datseg_single

def getGridLocation(grid, pt, addtogrid=0):
    # --- given a grid (e.g. x values) and one point, then tells you which discrte bin that point is in
    # assumes that leftmost point determines the bin
    # --- grid must have large extreme negative value on the ver left and be sorted. [i.e., pt cannot land to left of leftmost grid line]
    # print("---")
    # print(grid)
    if addtogrid!=0:
        # shift grid over to right by addtogrid*gridspacing
        grid_spacing = getgridspacing(grid[1:])
        grid = [g+addtogrid*grid_spacing for g in grid]
        # print("NEWGRID")
        # print(grid)
    # print("---")
    # print(grid)
    # print(pt)
    # print(grid)

    if pt>max(grid):
        pos = len(grid)-1
    else:
        A = grid[1:] >= pt
        B = grid[:-1] < pt
        # print(grid)
        # print(A)
        # print(B)
        pos = np.argwhere(A & B)
        # print(A)
        # print(B)
        # print(grid)
        # print(pt)
        if len(pos)>1:
            pos = int(pos[0])
        elif len(pos)==1:
            pos = int(pos[0])
        elif len(pos)==0:
            # then is likely because it is too much to the right
            pos = len(grid)-1
    return pos

def splitLines(datseg):
    # print("DOING SPLIT")
    """Split any long horizontal lines into two"""
    # looks for any horizontal lines that have left extreme in one location and right extreme in 2 grids over.
    # Modified datseg in place

    for i, datseg_single in enumerate(datseg):
        # print("ASDASDASDASDA")
        # find all the horizontal lines
        codes = [d["codes"] for d in datseg_single]
        xextremes = [d["x_extremes"] for d in datseg_single]
        # yextremes = [d["y_extremes"] for d in datseg_single]

        # print(codes)
        # print(i)
        xgrid = getxgrid(datseg_single, appendExtreme=True)

        strokestosplit = []
        midpts = []
        nsplits = []
        for i, (c, xx) in enumerate(zip(codes, xextremes)):
            # print(c)
            if c=="Lh":
                # print("YES")
                # -- get grid locations for each point
                pos1 = getGridLocation(xgrid, xx[0], -0.1)
                pos2 = getGridLocation(xgrid, xx[1], 0.25)
                gridspace = getgridspacing(xgrid[1:])
                xlength = xx[1]-xx[0]

                # --- get boundary between strokes
                # print(xgrid)
                # print(xx)
                # print(pos2)
                # print(pos1)
                # print(xlength)
                # print(gridspace*1.2)
                if pos2==pos1+2 and xlength>gridspace*1.15:
                    # print("doing split (1-->2)")
                    # then split this
                    strokestosplit.append(i)
                    midpts.append([np.mean(xx)])
                    nsplits.append(2)

                elif pos2==pos1+3 and xlength>gridspace*2.15:
                    # print("doing split (1-->3)")
                    strokestosplit.append(i)
                    tmp = (xx[1]-xx[0])/3
                    midpts.append([xx[0]+tmp, xx[0]+tmp*2])
                    nsplits.append(3)  

                elif pos2==pos1+4 and xlength>gridspace*3.15:     
                    # print("doing split (1-->4)")
                    strokestosplit.append(i)
                    tmp = (xx[1]-xx[0])/4
                    midpts.append([xx[0]+tmp, xx[0]+tmp*2, xx[0]+tmp*3])
                    nsplits.append(4)  
                # else:
                #     print("NO SPLIT")
                #     print(c)

        # === do split
        import copy
        datseg_single_orig = copy.deepcopy(datseg_single)
        numremoved = 0
        numadded = 0
        # print("ORIG SEQUENCE")
        # print([d["codes"] for d in datseg_single])

        # print("HERE")
        # print(midpts)
        # print(nsplits)
        for i, (strokenum, midp, nspl) in enumerate(zip(strokestosplit, midpts, nsplits)):
            # --- get the stroke info
            d = datseg_single_orig[strokenum]

            # left to right?
            # if d["on_xy"][0] > d["off_xy"][0]:
            #     onpos = "right"
            # else: 
            #     onpos = "left"

            # --- first get the x-extreme values ready
            # print("midpojnts")
            # print(midp)
            xedges = [d["x_extremes"][0], *midp, d["x_extremes"][1]]
            if d["on_xy"][0] > d["off_xy"][0]:
                xedges.reverse()

            # print("===== x edges (from split code)")
            # # print(nspl)
            # # print(midp)
            # print(xedges)

            # TODO: currently just making a fake stroke, inheriting some features. Should update so actually has real coordiantes and so forth.
            newstrokes = []
            # trim = (xedges[1]-xedges[0])*0.01 # amount to trim off edges to ensure they stay within bounds of a column.
            for x1, x2 in zip(xedges[:-1], xedges[1:]):
                ns = {}
                for key in d.keys():
                    ns[key]=[]
                for key in ["codes", "circleness", "badstroke"]:
                    ns[key] = d[key]

                if x2>x1:
                    ns["x_extremes"]=[x1, x2]
                else:
                    ns["x_extremes"]=[x2, x1]
                ns["y_extremes"]= d["y_extremes"] #TODO: this is not aaccurate, since could change afer segmenting
                ns["centerpos"] = [np.mean([x1, x2]), d["centerpos"][1]]
                ns["isfake"]=True
                newstrokes.append(ns)

            # print("Adding strokes:")
            # print(newstrokes)
            # print("Deleting:")
            # print(d)

            # --- delete this stroke
            # print("DELETING STROKE AT POS")
            # print(strokenum - numremoved + numadded)
            datseg_single.pop(strokenum - numremoved + numadded)
            numremoved += 1

            # --- add strokes in its place
            for jj, ns in enumerate(newstrokes):
                # print("ADDING STROKE AT POS")
                # print(strokenum + 1 - numremoved + numadded)
                datseg_single.insert(strokenum + 1 - numremoved + numadded, ns)
                numadded += 1

        # print("NEW SEQUENCE")
        # print([d["codes"] for d in datseg_single])


        # for any that have x extreme and y extreme too far apart...



# ----- segmentation
def getSegmentation(datflat, doplot=False, unique_codes=False, dosplits=False, removebadstrokes=True, 
    removeLongVertLine=False, updatestrokenum=True, do_unrotate=False, include_npstrokes=False, DEBUG=False,
    do_not_modify_strokes = False):
    """ 
    do_not_modify_strokes, maek True if want to take datflat[0]["trialstrokes"] and not modify it at all. otehriwse will do
    centering, etc.
    """
    def seg(datthis, do_unrotate=False):
        # do for single trial
        # print(datthis)
        # print(datthis[0])

        if do_not_modify_strokes:
            strokearray = datthis["trialstrokes"]
        else:
            from .program import strokes2nparray
            strokearray = strokes2nparray(datthis["trialstrokes"], recenter_and_flip=True, combinestrokes=False, 
                sec_rel_first_stroke=False, sec_rel_task_onset=True)
            if DEBUG:
                strokearray = strokes2nparray(datthis["trialstrokes"], recenter_and_flip=True, combinestrokes=False, 
                    sec_rel_first_stroke=True, sec_rel_task_onset=False)

        if do_unrotate:
            # if the name has "rotate" in it, then unrotate it (original rotation is pi/2 CCW)
            if "rotate" in datthis["stimname"]:
                strokearray = [np.array([[s[1], -s[0], s[2]] for s in strokes]) for strokes in strokearray]
                # print("did rotate")


        # 1) duration
        durs = [s[-1,2]-s[0,2] for s in strokearray]

        # 2) height (relative total height)
        height_all = max([max(s[:,1]) for s in strokearray]) - min([min(s[:,1]) for s in strokearray])
        height = [(max(s[:,1])-min(s[:,1]))/height_all for s in strokearray]

        # print(height_all)

        # extreme positions 
        xmax = [max(s[:,0]) for s in strokearray]
        xmin = [min(s[:,0]) for s in strokearray]
        # print(xmax)
        x_extremes = [[lo, hi] for lo, hi in zip(xmin, xmax)]

        ymax = [max(s[:,1]) for s in strokearray]
        ymin = [min(s[:,1]) for s in strokearray]
        y_extremes = [[lo, hi] for lo, hi in zip(ymin, ymax)]

        # 3) width (rel total width)
        w_all = max([max(s[:,0]) for s in strokearray]) - min([min(s[:,0]) for s in strokearray])
        width = [(max(s[:,0])-min(s[:,0]))/w_all for s in strokearray]
        # print(width)

        # 4) h/(h+w)
        h_rel_hplusw = [h/(w+h) for h,w in zip(height, width)]

#         print(h_rel_hplusw)
        
        assert min(x[0] for x in x_extremes)>-1000, "I assume that -1000 is smaller than anything, for appendig to left of grid values, for both row and column getting"
        assert min(y[0] for y in y_extremes)>-1000, "I assume that -1000 is smaller than anything, for appendig to left of grid values, for both row and column getting"
    
        # 5) distance vs. displacement
        def D(s):
            # s is one np.array, N by 3, N = timepoints.
            # adds up sum of distnace btween adjacent points
            d = [np.linalg.norm(s1-s2) for s1, s2 in zip(s[:-1,:2], s[1:,:2])]
            return sum(d)
        
        displace = [np.linalg.norm(s[-1,:2] - s[0,:2]) for s in strokearray]
        distance = [D(s) for s in strokearray]
        circularity = [1-p/t for p,t in zip(displace,distance)]
        
        # 6) start and end posotion
        onpos = [s[0,:2] for s in strokearray]
        offpos = [s[-1,:2] for s in strokearray]

        # 7) center position
        centerpos = [np.median(s[:,:2], axis=0) for s in strokearray]


        # 8) --- is it a bad stroke?
        def badstroke(durs, onsets, offsets, x_extremes, y_extremes):
            # automatically throw out if stroke is too short (and other potnetealy flags could add)
            # print(onsets)
            durthresh=0.08 #
            def can_throw_out(i, thresh = 0.2):
                # true if allowed to throw out (i.e., if adjancent gaps are both long, then not allowed)
                if i>0:
                    a = onsets[i]-offsets[i-1] < thresh
                else:
                    a = False

                if i<len(onsets)-1:
                    b = onsets[i+1]-offsets[i] <thresh
                else:
                    b = False            
                if a or b:
                    return True
                else:
                    return False
            # xmax = [xx[1]-xx[0] for xx in x_extremes]
            # ymax = [yy[1]-yy[0] for yy in y_extremes]

            bad = [True if (d<durthresh and can_throw_out(i)) else False for i, d in enumerate(durs)]
            bad2 = [True if xx[0]==xx[1] and yy[0]==yy[1] else False for xx, yy in zip(x_extremes, y_extremes)] # points don't move...
            # print(bad)
            # print(bad2)

            bad = [b1 or b2 for b1, b2 in zip(bad, bad2)]
            # print(bad)
            # TODO, add another filter, only throw out if it is adjacent to something else.
            return bad

        # ----- get vector representation of strokes
        if False:
            stroke_vectors = np.array([[a,b,c,d] for a,b,c,d in zip(height, width, h_rel_hplusw, circularity)])

#         # ----- make a template. will take dot product
#         templates = (
#             [], # circle
#             []
#         )

        # ===================================
        # ----- for each stroke get its scores
        on, off = getOnOff(datthis["trialstrokes"], relativetimesec=True)
        scores = []
        badstrokes = badstroke(durs, on, off, x_extremes, y_extremes)
        assert len(strokearray)==len(circularity)
        for i in range(len(circularity)):
            if removebadstrokes:
                if badstrokes[i]:
                    # print("SKIPPIG STROKE, is bad...")
                    continue
            if np.isnan(circularity[i]) or np.isnan(h_rel_hplusw[i]):
                # print("ISNAN (i.e., due to circulairyt or h_rel_hplusq, likley becuase of no distance moved, so divide by 0!!!")
                # print(strokearray[i])
                # print("bad?")
                # print(badstrokes[i])
                assert badstrokes[i]==True, "ISNAN (i.e., due to circulairyt or h_rel_hplusq, likley becuase of no distance moved, so divide by 0, expected badstrokes=True..."

            scores.append(
                {
            "strokenum_orig":i,
            "circleness":circularity[i],
            "badstroke":badstrokes[i],
            "height":height[i],
            "width":width[i],
            "h_rel_hplusw":h_rel_hplusw[i],
            "on_xy": strokearray[i][0,:2],
            "off_xy": strokearray[i][-1,:2],
            "centerpos": np.median(strokearray[i][:,:2], axis=0),
            "x_extremes": x_extremes[i],
            "y_extremes": y_extremes[i]
                })

            if include_npstrokes:
                scores[-1]["stroke"]=strokearray[i]

        

        
        # --- convert to codes
        def C(s,i):
            # --- give an output code for this stroke
            # print(s)
            # print(s["badstroke"])
            if s["badstroke"]:
                return "X"
            elif s["circleness"]>0.49:
                return "C"
            elif s["height"] > 0.5:
                return "LL"
            elif s["h_rel_hplusw"] < 0.5:
                return "Lh"
            elif s["h_rel_hplusw"] >= 0.5:
                return "Lv"

            
        # --- get spatial locations and direction
        # onset, offset, center

        # codes = [C(scores,i) for i in range(len(scores["circleness"]))]
        for s in scores:
            s["codes"] = C(s,i=0)
        # codes = [C(s,i=0) for s in scores]


        # ====== quick code to put things into rows.

        if doplot:
            plt.figure(figsize=(20,10))

            plt.subplot(3,1,1)
            x = list(range(len(durs)))
            plt.plot(x, durs, "-ok", label="durs")
            plt.legend()
            plt.ylim((0,0.5))

            plt.subplot(3,1,2)
            plt.plot(x, height, "-ob", label="height")
            plt.plot(x, width, "-or", label="wid")
            plt.plot(x, h_rel_hplusw, "-og", label="h/(h+w)")
            plt.legend()

            plt.subplot(3,1,3)
            # plt.plot(x, distance, "-ob", label="distance")
            # plt.plot(x, displace, "-or", label="displace")
            plt.plot(x, circularity, "-og", label="circleness")
            # plt.plot(x, h_rel_hplusw, "-og", label="h/(h+w)")
            plt.legend()
        return scores
    

    datsegs = []
    for dat in datflat:
        scores = seg(dat, do_unrotate=do_unrotate)

        datsegs.append(scores)

    if dosplits:
        # print("DOING SPLIT (will report each split)")
        splitLines(datsegs)


    if unique_codes:
        # print("GETTING UNIQUE CODES")
        [getUniqueCode(d) for d in datsegs]


    def getRow(scores, method="quick"):
        if method=="quick":
            """segments y into 3 rows, baesd on extrema derived from the long lines (vertical)"""
            # not perfect, but surprisingly works well.
            # TODO: do optimization - vary the vertical positions and get min within-row variation.

            # -- get y position extrema
            ymax = np.mean([d["y_extremes"][1] for d in scores if d["codes"]=="LL"])
            ymin = np.mean([d["y_extremes"][0] for d in scores if d["codes"]=="LL"])
            centers = [d["centerpos"][1] for d in scores]

            # -- trim from edge of ydist
            ydist = ymax-ymin
            TRIM=0.1
            ytrim = TRIM*ydist
            ymin = ymin+ytrim/2
            ymax = ymax-ytrim/2

            # --- shift center the grid to be closer to center of mass
            currentcenter = np.mean([ymin, ymax])
            centerofmass = np.mean([currentcenter for _ in range(len([d for d in scores if d=="LL"]))] + \
                [d["centerpos"][1] for d in scores if d["codes"]!="LL"]) # add center of mass of 4 sticks (LL) and all other primitives
            # centerofmass = np.mean([d["centerpos"][1] for d in scores if d["codes"]!="LL"]) # add center of mass of 4 sticks (LL) and all other primitives
            # print(centerofmass)
            # print(currentcenter)
            # print((ymin, ymax))
            sh = centerofmass - currentcenter # do shift.
            ymin=ymin+sh
            ymax=ymax+sh
            # print((ymin, ymax))

            # -- get all centers
            yshift = (ymax-ymin)/3
            ygrid = [ymin+yshift*i for i in range(4)]

            # --- assign to grid positions
            # print((ymin, ymax))
            # print(centers)
            # print(ygrid)
            rows = [getGridLocation([-1000] + ygrid, pt) for pt in centers]
            # print(rows)
            return rows

    for scores in datsegs:
        rows = getRow(scores, method="quick")
        for s, r in zip(scores, rows):
            s["row"]=r    


    # === remove long vertical line?
    if removeLongVertLine:
        datsegs = filterCodes(datsegs, "LL") # -- for each stim/subject clean up data

    # === add current order
    if updatestrokenum:
        updateStrokeNumNew(datsegs)

    return datsegs


def updateStrokeNumNew(datsegs):
    # makes strokenum_new indicate correct order (useful if removed strokes, etc)
    for dat in datsegs:
        for i, s in enumerate(dat):
            s["strokenum_new"]=i



def segmentedScores2NumpyStrokes(datsegs, add_time_column=False):
    # takes segmentation output and makes list of np arrays, like program, so can use plotting code
        
    def S(score_single):
        strokes = []
        for s in score_single:
            if s["circleness"]>0.5:
                center = np.array([np.mean(s["x_extremes"]), np.mean(s["y_extremes"])])
                radius = np.mean([np.diff(s["x_extremes"])/2, np.diff(s["y_extremes"])/2])
                C = center + np.array([np.array([(radius*math.cos(theta), radius*math.sin(theta)) for theta in np.linspace(0., 2.*math.pi, num=30)])])
                strokes.append(C[0])        
            else:
                # print(s)
            
                strokes.append(np.array([[s["x_extremes"][0], s["y_extremes"][0]], [s["x_extremes"][1], s["y_extremes"][1]]]))
        if add_time_column:
            strokes = program2strokes(strokes) 
        return strokes 

    strokes_all = []
    for score in datsegs:
        strokes = S(score)
        strokes_all.append(strokes)
    return strokes_all


def filterCodes(datsegs, codetoremove):
    # e.g., if codetoremove is "LL" then will remove any stroke that has things like "LL10", etc.
    for i, dat in enumerate(datsegs):
        datnew = [s for s in dat if codetoremove not in s["codes_unique"]]
        datsegs[i] = datnew

    return datsegs



def codeUniqueFeatures(code):
    # -- given a code (e.g., C01), tell me prim(C), col(0), row(1))
    prim = ""
    rowcol = []
    for i, c in enumerate(code):
        if c.isdigit():
            rowcol.append(int(c))
        else:
            prim = prim + c
    assert len(rowcol)==2, "shoud only be one dig for row and col, unless etting into 10s?"
    col = rowcol[0]
    row = rowcol[1]
    return prim, col, row