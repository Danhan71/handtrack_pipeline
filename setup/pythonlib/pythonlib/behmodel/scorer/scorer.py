

import numpy as np

class Scorer(object):
    
    def __init__(self):

        self.ScoreFunction = None
        self.NormFunction = None
        self.Params = None # dict["score"] = (thetas)
        self.Objects = None
        self._do_score_with_params = False
        self._do_norm_with_params = False

    def input_score_function(self, func):
        """ 
        IN:
        - func, you decide what its arugments
        --- e.g, if Prior, wnat to (strokes, TaskObject=None)
        --- if Likeli, wnat ...
        OUT:
        - assingns to self.ScoreFunction.
        """

        self.ScoreFunction = func

    def input_norm_function(self, func):
        """ 
        Normalizes raw scores. 
        """
        self.NormFunction = func

    def score(self, *args):
        if self._do_score_with_params:
            return self.ScoreFunction(*args, params=self.Params["score"])
        else:
            return self.ScoreFunction(*args)

    def norm(self, scores):
        """ scores is np array (N,) shape
        """
        if self._do_norm_with_params:
            probs = self.NormFunction(scores, params=self.Params["norm"])
        else:
            probs = self.NormFunction(scores)
        return probs

    def score_and_norm(self, *args):
        scores = self.score(*args)
        probs = self.norm(scores)
        return probs

    def params_print(self):
        # 1) motor cost
        if self.Objects is not None:
            if "MotorCost" in self.Objects:
                print(self.Objects["MotorCost"].Params)

        # 2) norm
        print(self.Params)


    def params_ravel(self, return_reg_mus = False):
        """ return flattened params
        - return_reg_mus, then returns array asme size as th,
        which are means for l2 rgulaizations.
        """

        th = np.empty(0)
        reg_mu = []
        # 1) motor cost
        if self.Objects is not None:
            if "MotorCost" in self.Objects:
                this = self.Objects["MotorCost"].params_ravel()
                th = np.append(th, this)
                if return_reg_mus:
                    # reg_mu.append(np.zeros_like(th))
                    reg_mu.extend([0. for _ in range(len(this))])

        # 2) norm
        th = np.append(th, self.Params["norm"])
        if return_reg_mus:
            reg_mu.append(1.)   
        out = th
        assert len(out.shape)==1

        if return_reg_mus:
            reg_mu = np.stack(reg_mu)
            assert len(reg_mu.shape)==1

            return out, reg_mu
        else:
            return out

    def params_unravel(self, th):
        """ return back
        """
        # 1) motor cost
        if self.Objects is not None:
            if "MotorCost" in self.Objects:
                th = self.Objects["MotorCost"].params_unravel(th, return_leftover=True)

        # 2) norm
        self.Params["norm"] = th[:1]
        # th = np.delete(th, [0])


        assert len(th)==1, "inputed too many params"
