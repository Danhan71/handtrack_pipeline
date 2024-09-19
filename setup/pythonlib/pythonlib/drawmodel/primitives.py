
# =========== [NEWER VERSION, NOT USING MATPLOTLIB]
import sys
# sys.path.append("/Users/lucastian/tenen/ec")
import math
import numpy as np
import matplotlib
# import torch
from scipy.ndimage import gaussian_filter as gf
#import cairo
from math import tan
from math import pi
from itertools import permutations
# from dreamcoder.program import Primitive, Application, Abstraction, Index
# from dreamcoder.type import t0,arrow,baseType
from matplotlib import pyplot as plt
import imageio
import random
if False:
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        matplotlib.use('TkAgg')

XYLIM = 3. # i.e., -3 to 3.

PRIMVERSION="full"
assert PRIMVERSION=="full" # do not change , since ocaml code requires names to match.

if PRIMVERSION=="full":
        SCALES = [0.5, 1., 1.25, 1.5, 2., 2.5, 3., 4.]
        NPOLY = range(3,7) # range of regular polyogns allowed.
        DISTS = [-2.5, -2., -1.5, -1., -0.5, -0.25, 0, 0.25, 0.5, 1., 1.5, 2., 2.5, 3.] + [-1.75, -0.65, 0.45, 1.55, 1.1] + [0.5/tan(pi/n) for n in range(3, 7)]# for making regular polygons
        THETAS = [j*(2*pi/8) for j in range(8)] + [-2*pi/6] + [-2*pi/12]
        ORDERS = ["trs", "tsr", "rts", "rst", "srt", "str"]
elif PRIMVERSION=="S8_9":
        SCALES = [2., 4.]
        NPOLY = range(3,7) # range of regular polyogns allowed.
        DISTS = [-2.5, -2., -1.5, -1., -0.5, -0.25, 0, 0.25, 0.5, 1., 1.5, 2., 2.5, 3.] 
        # THETAS = [pi/2]
        THETAS = [j*(2*pi/8) for j in range(8)]
        ORDERS = ["rts", "trs"]

PARSE_EVALUATOR = False
# def set_parsing(p):
#         global PARSE_EVALUATOR
#         PARSE_EVALUATOR = p
#         if p:
#                 Primitive.GLOBALS["circle"].value = {Parse(_circle)}
#                 Primitive.GLOBALS["line"].value = {Parse(_line)}
#         else:
#                 Primitive.GLOBALS["circle"].value = _circle
#                 Primitive.GLOBALS["line"].value = _line

# class Chunk():
#         def __init__(self, l):
#                 assert isinstance(l, list)
#                 for x in l: assert isinstance(x, (np.ndarray, np.generic, Chunk))
#                 self.l = l
#                 self._h = None

#         def applyMatrix(self, m): # applies to everything within chunk, including recursive.
#                 return Chunk([x.applyMatrix(m) for x in self.l ])
        
#         def __eq__(self, other): return hash(self) == str(other) and str(self) == str(other)
#         def __ne__(self, other): return not (self == other)
#         def __hash__(self):
#                 if self._h is None:
#                        self._h = hash(tuple(hash(x) if isinstance(x,Chunk) else hash(x.tostring()) for x in self.l))
#                 return self._h
#         def __str__(self): return f"Chunk({self.l})"
#         def __repr__(self): return str(self)
#         def flatten(self): # returns [[a], [b], ...], where each element is a single "stroke"
#                 return [y
#                         for x in self.l
#                         for y in ([x] if isinstance(x, (np.ndarray, np.generic)) else x.flatten()) ]

#         @staticmethod
#         def invention(parses):
#                 # print("---")
#                 # print(parses)
#                 # for p in parses:
#                 #     print(p)
#                 # print("-----")
#                 return {Parse([Chunk(parse.l)])
#                         for parse in parses}
# # This should *never* the added to the library!
# _chunky_primitive = Primitive("CHUNK_INVENTION", arrow(baseType("tstroke"),baseType("tstroke")), Chunk.invention)


# class Parse():
#         def __init__(self, l):
#                 assert isinstance(l, list)
#                 for x in l: assert isinstance(x, (np.ndarray, np.generic, Chunk))
#                 self.l = l
#                 self._h = None

#         def applyMatrix(self, m):
#                 newList = []
#                 for x in self.l:
#                         if isinstance(x,Chunk):
#                                 newList.append(x.applyMatrix(m))
#                         else:
#                                 newList.append(_tform(x,m))
#                 return Parse(newList)

#         def __eq__(self, other): return hash(self) == hash(other) and str(self) == str(other)
#         def __ne__(self, other): return not (self == other)
#         def __hash__(self):
#                 if self._h is None:
#                         self._h = hash(tuple(hash(x) if isinstance(x,Chunk) else hash(x.tostring()) for x in self.l ))
#                 return self._h
#         def __repr__(self): return str(self)
#         def __str__(self): return f"Parse({self.l})"

#         def flatten(self):
#                 return [y
#                         for x in self.l
#                         for y in ([x] if isinstance(x, (np.ndarray, np.generic)) else x.flatten()) ]

#         def animate(self, fn):
#                 import scipy.misc
#                 trace = self.flatten()
#                 images = [prog2pxl(trace[:n])
#                           for n in range(1, len(trace)+1)]
#                 image = np.concatenate(images,1)
#                 if fn is None:
#                         return image
#                 else:
#                         # scipy.misc.imsave(fn, image)
#                         imageio.imwrite(fn, image)
#         @staticmethod
#         def animate_all(parses, fn):
#                 import scipy.misc
#                 images = [parse.animate(None) for parse in parses ]
#                 # scipy.misc.imsave(fn, np.concatenate(images,0))
#                 imageio.imwrite(fn, np.concatenate(images,0))


#         @staticmethod
#         def ofProgram(p):
#             """Takes a program and returns its set-of-parses"""
#             from datetime import datetime
#             # now = datetime.now()
#             # if p.isApplication:
#             #     import pdb
#             #     pdb.set_trace()
#             def chunky(q):
#                     # print(datetime.now())
#                     # counter+=1
#                     # print(counter)
#                     if q.isApplication or q.isInvented:
#                             f,xs = q.applicationParse()
#                             chunky_arguments = [chunky(x) for x in xs ]
#                             if f.isInvented and str(f.tp.returns()) == "tstroke":
#                                     numberExpands = len(f.tp.functionArguments()) - len(xs)
#                                     return_value = chunky(f.body)
#                                     for x in chunky_arguments:
#                                             if numberExpands > 0: x = x.shift(numberExpands)
#                                             return_value = Application(return_value,x)
#                                     for i in range(numberExpands - 1,-1,-1):
#                                             return_value = Application(return_value, Index(i))
#                                     return_value = Application(_chunky_primitive,return_value)
#                                     for _ in range(numberExpands):
#                                             return_value = Abstraction(return_value)
#                                     # print(return_value)
#                                     # print("this is our type")
#                                     # try:
#                                     #         print(return_value.infer())
#                                     # except:
#                                     #         print("total failure to get a type")
#                                     #         print(q)
#                                     #         print(q.infer())
#                                     #         assert False
#                                     return return_value
#                             elif f.isInvented:
#                                     return_value = chunky(f.body)
#                                     for x in chunky_arguments:
#                                             return_value = Application(return_value,x)
#                                     return return_value
#                             else:
#                                     # import pdb
#                                     # pdb.set_trace()
#                                     return_value = chunky(f)
#                                     for x in chunky_arguments:
#                                             return_value = Application(return_value,x)
#                                     return return_value
#                     if q.isAbstraction:
#                             return Abstraction(chunky(q.body))
#                     if q.isIndex or q.isPrimitive: return q                                        
#             set_parsing(True)
#             # counter=0
#             p = chunky(p)
#             print(p)
#             parses = p.evaluate([])
#             set_parsing(False)
#             return parses
                
                

                
                

# ============= TRANSFORMATIONS
def makeAffine2(x=None, y=None, sx=None, sy=None, theta=None, order=None):
        """General use, e..g, drawnn, and allowing x and y scaling to differ"""
        if sx is None:
                sx=1.
        if sy is None:
                sy=1.
        if theta is None:
                theta=0.
        if x is None:
                x=0.
        if y is None:
                y=0
        if order is None:
                order="trs"
                
        def R(theta):
                T = np.array([[math.cos(theta), -math.sin(theta), 0.], [math.sin(theta), math.cos(theta), 0.], [0.,0.,1.]])
                return T

        def S(sx, sy):
                T = np.array([[sx, 0., 0.], [0., sy, 0.], [0., 0., 1.]])
                return T
        
        def T(x,y):
                T = np.array([[1., 0., x], [0., 1., y], [0., 0., 1.]])
                return T
                
        if order == "trs":
                return T(x,y)@(R(theta)@S(sx, sy))
        elif order == "tsr":
                return T(x,y)@(S(sx, sy)@R(theta))
        elif order == "rts":
                return R(theta)@(T(x,y)@S(sx, sy))
        elif order == "rst":
                return R(theta)@(S(sx, sy)@T(x,y))
        elif order == "srt":
                return S(sx, sy)@(R(theta)@T(x,y))
        elif order == "str":
                return S(sx, sy)@(T(x,y)@R(theta))



def _makeAffine(s=1., theta=0., x=0., y=0., order="trs"):
        
        if s is None:
                s=1.
        if theta is None:
                theta=0.
        if x is None:
                x=0.
        if y is None:
                y=0
        if order is None:
                order="trs"
                
        def R(theta):
                T = np.array([[math.cos(theta), -math.sin(theta), 0.], [math.sin(theta), math.cos(theta), 0.], [0.,0.,1.]])
                return T

        def S(s):
                T = np.array([[s, 0., 0.], [0., s, 0.], [0., 0., 1.]])
                return T
        
        def T(x,y):
                T = np.array([[1., 0., x], [0., 1., y], [0., 0., 1.]])
                return T
                
        if order == "trs":
                return T(x,y)@(R(theta)@S(s))
        elif order == "tsr":
                return T(x,y)@(S(s)@R(theta))
        elif order == "rts":
                return R(theta)@(T(x,y)@S(s))
        elif order == "rst":
                return R(theta)@(S(s)@T(x,y))
        elif order == "srt":
                return S(s)@(R(theta)@T(x,y))
        elif order == "str":
                return S(s)@(T(x,y)@R(theta))


# def _tform_wrapper(p, T):
#         global PARSE_EVALUATOR
#         if PARSE_EVALUATOR:
#                 return {Parse([Chunk(_tform(parse.l,T))]) for parse in p}
#         else:
#                 return _tform(p,T)
                
def _tform(p, T, i=1):
        """Applies the transformation T to the object p for the number of iterations i.
        p can be a list, numpy, Parse, or Chunk."""
        # if isinstance(p, Chunk):
        #         return Chunk([_tform(x, T, i) for x in p.l ])
        # if isinstance(p, Parse):
        #         return Parse([_tform(x, T, i) for x in p.l ])
        if isinstance(p, list):
                return [_tform(x, T, i) for x in p]
        # given prim and affine matrix (T), otuput new p.
        
        # append column of ones to matrix.
        p = np.concatenate((p, np.ones((p.shape[0],1))), axis=1)
        for _ in range(i):
                p = (T@p.transpose()).transpose() # apply affine transfomatiton.
        p = np.delete(p, 2, axis=1)     # --- remove third dimension
        return p


# def _reflect_wrapper(p, theta):
#         global PARSE_EVALUATOR
#         if PARSE_EVALUATOR:
#                 return {_reflect(parse, theta) for parse in p}
#         else:
#                 return _reflect(p, theta)
def _reflect(p, theta=math.pi/2): # TODO: reflect should also be usable with repeat.
        # if isinstance(p, Chunk):
        #         return Chunk([_reflect(x, theta) for x in p.l ])
        # if isinstance(p, Parse):
        #         return Parse([_reflect(x, theta) for x in p.l ])
        if isinstance(p, (np.ndarray, np.generic)):
                return _reflect([p], theta)[0]
        
        # reflection over line thru origin
        # first rotate p by -theta, then reflect across y axis, then unrotate (by +theta)
        # y axis would be theta = pi/2
        th = theta - math.pi/2

        p = transform(p, theta=-th)
        T = np.array([[-1., 0.], [0., 1.]])
        p = [np.matmul(T, pp.transpose()).transpose() for pp in p]
        p = transform(p, theta=th)
        
        return p

# =========== FUNCTIONS ON PRIMTIVES.
def repeat2(p, N, x=0., y=0., sx=1., sy=1., theta=0., order="trs"):
        """ General purpose, and allowing scale differ for x and y"""
        T = makeAffine2(x, y, sx, sy, theta, order)
        p_out = []
        for i in range(N):
                if i>0:
                        p = _tform(p, T) # apply transformation
                pthis = [np.copy(pp) for pp in p] # copy current state, and append
                p_out.extend(pthis)
        return p_out



def _repeat(p, N, T):
        # global PARSE_EVALUATOR
        # if PARSE_EVALUATOR:
        #         all_permutations = []
        #         for parse in p:
        #                 for iteration_range in [list(range(N)),list(range(N-1,-1,-1))]:
        #                         # permutation_of_child should be a list of strokes and chunks
        #                         # we need to apply the operator T to every element of this list N times
        #                         new_child = [ _tform(element_of_child, T, i)
        #                                       for i in iteration_range
        #                                       for element_of_child in parse.l ]
        #                         all_permutations.append(Parse([Chunk(new_child)]))
        #         return set(all_permutations)
                        

        p_out = []
        for i in range(N):
                if i>0:
                        p = _tform(p, T) # apply transformation
                pthis = [np.copy(pp) for pp in p] # copy current state, and append
                p_out.extend(pthis)
        return p_out


def _connect(p1, p2):
        # global PARSE_EVALUATOR
        # if PARSE_EVALUATOR:
        #         return {Parse(list(p))
        #                 for a in p1
        #                 for b in p2 
        #                 for p in permutations(a.l + b.l)}
                
        #  takes two primitives and makes a new one
        return p1 + p2


# ========== STROKES 
_line = [np.array([(0., 0.), (1., 0.)])] # --- unit line, from 0 to 1
_circle = [np.array([(0.5*math.cos(theta), 0.5*math.sin(theta)) for theta in np.linspace(0., 2.*math.pi, num=30)])] # --- circle, centered at 0, diameter 1
_arc = [np.array([(0.5*math.cos(theta), 0.5*math.sin(theta)) for theta in np.linspace(0., math.pi, num=15)])] # --- circle, centered at 0, diameter 1
# _emptystroke = [np.array([(0., 0.)])] # ---
_emptystroke = [] # ---

# --- regular polygons
def polygon(N=3):
    # e.g, if N 3, then this is shortcut to make triangle. could be done entirely with rest of primitives in library. 
    # N = range(3,7)
    y = 0.5/tan(pi/N)
    return _repeat(transform(_line, x=-0.5, y=y), N, _makeAffine(theta=2*pi/N))



## ============================= NEW VERSION, USING CONTINUATION INSTEAD OF CONNECT
def _lineC(k):
    """takes something (k) and connects line to it"""
    return _connect(_line, k)
    # return lambda k: _connect(_line, k)

    # return lambda k: lambda s: k(_connect(s, _line))

def _circleC(k):
    """takes something (k) and connects circle to it"""
    return _connect(_circle, k)
    # return lambda k: _connect(_circle, k)

def _emptystrokeC(k):  
    return _connect(_emptystroke, k)
    # return lambda k: _connect(_emptystroke, k)

def _finishC():
    """attaches empty stroke to finalize"""
    return lambda k: k(_emptystroke)

def _repeatC(s, N, T):
    """as intermediate step grounds s to numbers, does repeat to it, 
    then returns a function that connects the repeated thing to the 
    next thing"""
    # return lambda k: _connect(_repeat(s(_emptystroke), N, T), k)
    return lambda k: _connect(_repeat(s(_emptystroke), N, T), k)

def _transformC(s,T):
    return lambda k: _connect(_tform_wrapper(s(_emptystroke), T), k)

def _reflectC(s, theta):
    return lambda k: _connect(_reflect(s(_emptystroke), theta), k)    


# def _repeatC(N, T):
#     return lambda k: lambda s: k(_repeat(s(_emptystroke), N, T))

# def _repeatC(k, N, T):
#     """returns a function that can take in the next in continuation"""
#     return lambda s: _connect(_repeat(k(_emptystroke), N, T), s)





# ============= NOT PRIMITIVES.
def transform(p, s=1., theta=0., x=0., y=0., order="trs"):
                        # order is one of the 6 ways you can permutate the three transformation primitives. 
                        # write as a string (e.g. "trs" means scale, then rotate, then tranlate.)
                        # input and output types guarantees a primitive will only be transformed once.

        T = _makeAffine(s, theta, x, y, order) # get affine matrix.
        p = _tform(p, T)
        return p
                

def savefig(p, fname="tmp.png"):
        ax = plot(p)
        ax.get_figure().savefig(fname)
        print("saved: {}".format(fname))

def plot(p, color="k", LIMITS=XYLIM):
        fig = plt.figure(figsize=(LIMITS,LIMITS))
        ax = fig.add_axes([-0.03, -0.03, 1.06, 1.06])
        ax.set_xlim(-LIMITS,LIMITS)
        ax.set_ylim(-LIMITS,LIMITS)
        [ax.plot(x[:,0], x[:,1], "-", color=color) for x in p]
        return ax


def plotOnAxes(p, ax, color="k", LIMITS=XYLIM):
        ax.set_xlim(-LIMITS, LIMITS)
        ax.set_ylim(-LIMITS, LIMITS)
        # ax.axis("equal")
        [ax.plot(x[:,0], x[:,1], "-", color=color) for x in p]
        return ax

if False:
    # REPLACED BY prog2pxl
    def __fig2pixel(p, plotPxl=False, smoothing=0., LIMITS=XYLIM):
    #       smoothing is std of gaussian 2d filter. set to 0 to not smooth.
    #       https://stackoverflow.com/questions/43363388/how-to-save-a-greyscale-matplotlib-plot-to-numpy-array
            from skimage import color
            ax = plot(p, LIMITS=LIMITS)
            fig = ax.get_figure()
            fig.canvas.draw()
            ax.axis("off")

            width, height = fig.get_size_inches() * fig.get_dpi()
            # print("dpi: {}".format(fig.get_dpi()))
            # import pdb
            # pdb.set_trace()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            img = color.rgb2gray(img)

            if smoothing>0:
                    img = gf(img, smoothing, truncate=5)
                    
            if plotPxl:
                    # - show the figure
                    plt.figure()
                    # plt.imshow(img, vmin=0, vmax=1, cmap="gray", interpolation="bicubic")
                    plt.imshow(img, vmin=0, vmax=1, cmap="gray")

            return img

def __loss(p1, p2, plotPxl=False, smoothing=2):
        # loss function (compare two images)

        img1 = __fig2pixel(p1, plotPxl=plotPxl, smoothing=smoothing)
        img2 = __fig2pixel(p2, plotPxl=plotPxl, smoothing=smoothing)

        return np.linalg.norm(img2-img1)

def drawLine(p0, p1, img, pytorch = False):
        """Generate line pixels.
        Parameters
        ----------
        p0, p1 : uint8
             batchsize x 2
        img    : uint8
             batchsize x WH x WH
        Returns
        -------
        img    : uint8
             batchsize x WH x WH
        """

        if len(p0)==0 or len(p1)==0:
                return img
        p = p0
        dp = np.int8(p1 - p0)
        sp = np.int8(dp > 0)
        dp = np.abs(dp)

        steep = dp[:, 0] > dp[:, 1]
        nsteep = np.logical_not(steep)
        p[steep, :] = np.flip(p[steep, :], axis=1)
        dp[steep, :] = np.flip(dp[steep, :], axis=1)
        sp[steep, :] = np.flip(sp[steep, :], axis=1)

        d = np.int8((2 * dp[:, 0]) - dp[:, 1])
        n = d.copy()

        for i in range(np.max(dp[:, 1])):
                curr = np.logical_and((i < dp[:, 1]), steep)
                ncurr = np.logical_and((i < dp[:, 1]), nsteep)
                img[np.squeeze(np.argwhere(curr)), p[curr,1], p[curr,0]] = 1
                img[np.squeeze(np.argwhere(ncurr)), p[ncurr,0], p[ncurr,1]] = 1


                n[:] = 0
                cond1 = dp[:,1]>0
                cond2 = np.logical_and(cond1, d > 0)
                n[cond2] = np.int8(np.ceil(np.float32(d[cond2])/(2.0*np.float32(dp[cond2,1]))))
                p[:,0] = p[:,0] + n*(2*sp[:,0]-1)
                d = d - 2*n*dp[:,1]

                n[:] = 0
                cond3 = np.logical_and(cond1, d == 0)
                n[cond3] = 1
                p[:,0] = p[:,0] + n*(2*sp[:,0]-1)
                d = d - 2*n*dp[:,1]

                p[:,1] = p[:,1] + (2*sp[:,1]-1)

                d = d + 2*dp[:,0]

        img[np.arange(img.shape[0]), p1[:,0], p1[:,1]] = 1

        return img


def drawLine_torch(p0, p1, img, mx, inkval):
        """Generate line pixels.
        Parameters
        ----------
        p0, p1 : uint8
             batchsize x 2
        img    : uint8
             batchsize x WH x WH
        Returns
        -------
        img    : uint8
             batchsize x WH x WH
        """
        import torch

        p = p0
        dp = (p1 - p0)
        sp = (dp > 0)
        dp = torch.abs(dp)

        steep = dp[:, 0] > dp[:, 1]
        nsteep = torch.logical_not(steep)
        p[steep, :] = torch.flip(p[steep, :], [1])
        dp[steep, :] = torch.flip(dp[steep, :], [1])
        sp[steep, :] = torch.flip(sp[steep, :], [1])

        d = ((2 * dp[:, 0]) - dp[:, 1])
        n = d.detach().clone()

        for i in range(torch.max(dp[:, 1])):
                nob = torch.logical_and(torch.logical_and(p[:, 0] >= 0, p[:, 0] <= mx), torch.logical_and(p[:, 1] >= 0, p[:, 1] <= mx))

                curr = torch.logical_and(torch.logical_and((i < dp[:, 1]), steep), nob)
                ncurr = torch.logical_and(torch.logical_and((i < dp[:, 1]), nsteep), nob)
                # img[torch.nonzero(curr), p[curr, 1], p[curr, 0]] = inkval
                ind = torch.nonzero(curr)
                if ind.nelement() > 0:
                        xx = p[curr, 1]
                        yy = p[curr, 0]
                        ind = torch.squeeze(ind,1).to(torch.long)
                        img[ind, xx, yy] = inkval
                # img[torch.nonzero(ncurr), p[ncurr, 0], p[ncurr, 1]] = inkval
                ind = torch.nonzero(ncurr)
                if ind.nelement() > 0:
                        xx = p[ncurr, 0]
                        yy = p[ncurr, 1]
                        ind = torch.squeeze(ind,1).to(torch.long)
                        img[ind, xx, yy] = inkval

                n[:] = 0
                cond1 = dp[:, 1] > 0
                cond2 = torch.logical_and(cond1, d > 0)
                n[cond2] = (torch.ceil((d[cond2]).to(torch.float32) / (2.0 * (dp[cond2, 1]).to(torch.float32)))).to(torch.long)
                p[:, 0] = p[:, 0] + n * (2 * sp[:, 0] - 1)
                d = d - 2 * n * dp[:, 1]

                n[:] = 0
                cond3 = torch.logical_and(cond1, d == 0)
                n[cond3] = 1
                p[:, 0] = p[:, 0] + n * (2 * sp[:, 0] - 1)
                d = d - 2 * n * dp[:, 1]


                p[:, 1] = p[:, 1] + (2 * sp[:, 1] - 1)

                d = d + 2 * dp[:, 0]

        nob = torch.logical_and(torch.logical_and(p1[:, 0] >= 0, p1[:, 0] <= mx),
                                torch.logical_and(p1[:, 1] >= 0, p1[:, 1] <= mx))
        ind = torch.nonzero(nob)
        if ind.nelement() > 0:
                xx = p1[nob, 0]
                yy = p1[nob, 1]
                ind = torch.squeeze(ind, 1).to(torch.long)
                img[ind, xx, yy] = inkval

        return img

def prog2pxl_vectorized(allStrokes, WHdraw = 2*XYLIM, WH=128, smoothing=0):
        """ takes an np array and outputs one pixel image per batch element
                - WHdraw, the size of drawing canvas (e.g. 6, if is xlim -3 to 3),
                for now assumes the canvas is a square.
                - WH, WH x WH for output image.
                - smoothing, gaussian filter, this is sigma in pix. set to 0 to skip.
                RETURN:
                I, pixel images. where 0 is no ink, and max ink is 1.
                NOTE: shape is such that I[10,:,0] will be the 11th row from the top of the
                image, while I[:, 10,0] will be 11th column from the left.
                """

        # 1) create canvas
        # WH = 128

        # assert WH%4==0, "empirically if not mod 4 then looks weird.."
        if WH % 4 != 0:
                # then drop to next lowest multiple of 4. afterwards will add 0s to ends
                # get back to 4.
                rows_to_add = WH % 4
                cols_to_add = rows_to_add
                WH = WH - rows_to_add
                # print(WH)
                # print(rows_to_add)
        else:
                rows_to_add = None

        scale = (WH-1) / WHdraw
        # data = np.zeros((WH, WH, allStrokes.shape[3]), dtype=np.uint8)
        data = np.zeros((allStrokes.shape[3], WH, WH), dtype=np.uint8)

        for i in range(allStrokes.shape[2]):
                for j in range(allStrokes.shape[0]-1):
                        p0 = allStrokes[j,:,i,:]
                        p1 = allStrokes[j+1,:,i,:]
                        inds =  ~np.isnan(p1[0,:])
                        p0 = np.uint8(np.round((p0[:, inds] + WHdraw / 2) * scale))
                        p1 = np.uint8(np.round((p1[:, inds] + WHdraw / 2) * scale))
                        # data[:,:,inds] = drawLine(p0, p1, data[:,:,inds])
                        data[inds, :,:] = drawLine(p0.T, p1.T, data[inds, :,:]) 

        data = np.moveaxis(data, 0, -1)

        if smoothing > 0:
                # data = gf(np.flip(data, 0), smoothing, truncate=5)
                data = gf(data, [smoothing, smoothing, 0], truncate=5)
        # else:
        #         data = np.flip(data, 0)

        # Append columns or rows to data to get back to the desired size.
        # append r and c columns of 0s, in order to match a size

        if rows_to_add is not None:
                # rows
                rup = int(np.ceil(rows_to_add / 2))
                rlow = int(np.floor(rows_to_add / 2))
                h, w, b = data.shape
                data = np.concatenate((np.zeros((rup, w, b)), data, np.zeros((rlow, w, b))), axis = 0)

                # columns
                cleft = int(np.ceil(cols_to_add / 2))
                cright = int(np.floor(cols_to_add / 2))
                h, w, b = data.shape
                data = np.concatenate((np.zeros((h, cleft, b)), data, np.zeros((h, cright, b))), axis = 1)

        return data


def prog2pxl(p, WHdraw = 2*XYLIM, WH=128, smoothing=0):
        """ takes a list of np array and outputs one pixel image
        - WHdraw, the size of drawing canvas (e.g. 6, if is xlim -3 to 3),
        for now assumes the canvas is a square. 
        - WH, WH x WH for output image.
        - smoothing, gaussian filter, this is sigma in pix. set to 0 to skip.
        RETURN:
        I, pixel image. where 0 is no ink, and max ink is 1.
        NOTE: shape is such that I[10,:] will be the 11th row from the top of the
        image, while I[:, 10] will be 11th column from the left. if use
        plt.imshow(I) will be same orientation as input p.
        NOTE: there will be inherent smoothing due to downsampling/interp.
        """
        import cairo
        # 1) create canvas
        # WH = 128

        # assert WH%4==0, "empirically if not mod 4 then looks weird.."
        if WH%4!=0:
            # then drop to next lowest multiple of 4. afterwards will add 0s to ends
            # get back to 4.
            rows_to_add =  WH%4
            cols_to_add = rows_to_add
            WH = WH - rows_to_add
            # print(WH)
            # print(rows_to_add)
        else:
            rows_to_add = None
        
        scale = WH/WHdraw
        data = np.zeros((WH, WH), dtype=np.uint8)
        surface = cairo.ImageSurface.create_for_data(data, cairo.Format.A8, WH-2, WH-2)

        # 2) create context
        context = cairo.Context(surface)
        # context.set_line_width(STROKESIZE)
        context.set_source_rgb(256,256,256)

        # 3) add each primitive
        for pp in p:
                pthis = pp
                
                pthis= pthis + WHdraw/2 # -- center
                pthis = pthis*scale # -- scale so that coordinates match

                for ppp in pthis:
                        context.line_to(ppp[0], ppp[1])
                context.stroke() # this draws and also clears context.


        # 4) render
#     data = np.flip(data, 0)/255.0
        # data = data/255.0
        # surface.write_to_png("/tmp/test.png")
#     from matplotlib import pyplot as plt
#     plt.figure(figsize=(3,3))
#     # plt.xlim(-3,3)
#     # plt.ylim(-3,3)
#     plt.imshow(data, vmin=0, vmax=1, cmap="gray")
#     plt.savefig("/tmp/test.svg")

        if smoothing>0:
            # return np.flip(data, 0)/255.0
            data =  gf(np.flip(data, 0)/255.0, smoothing, truncate=5)
        else:
            data =  np.flip(data, 0)/255.0

        # Append columns or rows to data to get back to the desired size.
        # append r and c columns of 0s, in order to match a size

        if rows_to_add is not None:
            # rows
            rup = int(np.ceil(rows_to_add/2))
            rlow = int(np.floor(rows_to_add/2))
            h, w = data.shape
            data = np.r_[np.zeros((rup, w)), data, np.zeros((rlow, w))]

            # columns
            cleft = int(np.ceil(cols_to_add/2))
            cright = int(np.floor(cols_to_add/2))
            h, w = data.shape
            data = np.c_[np.zeros((h, cleft)), data, np.zeros((h, cright))]

        return data



def loss_pxl(img1, img2):
        return np.linalg.norm(img2-img1)

def program_ink(p):
        # takes a list of np array and outputs the amount of ink used
        # import pdb
        # pdb.set_trace()
        cost = 0
        for a in p:
                for n in range(a.shape[0] - 1):
                        u = a[n]
                        v = a[n + 1]
                        cost += ((u - v)*(u - v)).sum()**0.5
        return cost
                        
