""" Handles a single model, works with scorers that work with parses (graph representation of
continuous tasks)
Used for gridlinecircle
"""

import numpy as np
import matplotlib.pyplot as plt

class BehModel(object):
    """
    """

    def __init__(self):
        """
        IN:
        - parse
        """
        self.Params = {}

        self.Prior = None
        self.Likeli = None
        self.Poster = None

        self.Dat = None 
        self.IndTrial = None

        self.UniqueID = None

        self._prior_scores = None
        self._prior_probs = None
        self._likeli_scores = None
        self._behavior = None
        self._task = None
        self._posterior_score = None

        self._list_input_args = ("dat", "trial") # args in order.

    def input_model_components(self, prior_scorer, likeli_scorer, posterior_scorer,
        list_input_args_likeli, list_input_args_prior, 
        poster_use_log_likeli, poster_use_log_prior,
        poster_scorer_test=None):
        """ 
        IN:
        - all scorers are instances of the Scorer class.
        - signatures:
        --- prior_scorer.score(parse_single)
        --- likeli_scorer.score(beh_single, parse_single)
        --- posterior_scorer.score(prior_scores, likeli_scores)
        --- where parse_single and beh_single are strokes types.
        OUT:
        - assigns to fields
        """

        self.Prior = prior_scorer
        self.Likeli = likeli_scorer
        self.Poster = posterior_scorer

        if poster_scorer_test is not None:
            # Useful if want something other that weighted sum.
            self.PosterTest = poster_scorer_test
        else:
            self.PosterTest = None

        # self._list_input_args_likeli = ("dat", "trial")
        # self._list_input_args_prior = ("parsesflat", "trialcode")
        # self._poster_use_log_likeli = True
        # self._poster_use_log_prior = True

        self._list_input_args_likeli = list_input_args_likeli
        self._list_input_args_prior = list_input_args_prior
        self._poster_use_log_likeli = poster_use_log_likeli
        self._poster_use_log_prior = poster_use_log_prior

    def unique_id(self, set_id_to_this=None, ver="random_num"):
        """ a string that identifies this model
        - if set_id_to_this, then will set the ID and not change. will only
        allow to run once.
        - ver,
        --- "timestamp", then sets as this.
        NOTE:
        - if not yet set, then will set.
        - always returns string.
        """
        if self.UniqueID is not None:
            return self.UniqueID
        else:
            if set_id_to_this is not None:
                self.UniqueID = set_id_to_this
                assert isinstance(set_id_to_this, str)
            else:
                if ver=="timestamp":
                    from pythonlib.tools.expttools import makeTimeStamp
                    ts = makeTimeStamp()
                    self.UniqueID = ts
                elif ver=="random_num":
                    import random
                    self.UniqueID = str(random.randint(10e8, 10e9))[:6]
                else:
                    assert False, "not sure how"

        return self.UniqueID

        



    def score_single_trial(self, behavior, task):
        """ scores this behavior on this task, under model
        INPUT:
        - behavior, Behavior() class
        - task, Task() class
        OUT:
        - score, scalar posteiro score.
        """

        self._behavior = behavior
        self._task = task

        parses_list = [p["strokes"] for p in task.Parser.Parses]

        # 1) prior score for all parses
        # parses_list = task.Parses
        # parser = task.Parser
        priors = []
        for parse in parses_list:
            priors.append(self.Prior.score(parse))
        self._prior_scores = priors

        # 2) Likelis (one behavior vs. all parses)
        likelis = []
        strokes_beh = behavior.Strokes
        for parse in parses_list:
            likelis.append(self.Likeli.score(strokes_beh, parse))
        self._likeli_scores = likelis

        # 3) Posterior
        self._posterior_score = self.Poster.score(likelis, priors)
        return self._posterior_score


    def score_single_trial_dataset(self, D, ind):
        """ 
        Requires:
        - priorscorer and likeliscorer both take in a D and ind
        INPUT:
        - D, Dataset,
        - ind, trial
        OUTPUT:
        - score.
        """

        probs = self.Prior.score_and_norm(D, ind)
        likelis = self.Likeli.score(D, ind)

        self._prior_probs = probs
        self._likeli_scores = likelis

        self._posterior_score = self.Poster.score(likelis, probs)

        # save here
        self.Dat = D 
        self.IndTrial = ind

        return self._posterior_score

    def score_likelis_trial(self, D, ind):
        """
        get list of likelis for all parses for this single line of dataset D
        OUT:
        - np array
        NOTE:
        - does not save in model mmeory.
        """

        likelis = self.Likeli.score(D, ind)
        return likelis






    #################### FOR PLOTTING

    def inds_sorted_increasing(self, sort_by):
        """ return inds, 
        sorted by either:
        "prior", "likeli"
        OUT:
        - inds, scores
        """

        if sort_by=="prior":
            scores = self._prior_probs
        elif sort_by=="likeli":
            scores = self._likeli_scores
        elif sort_by=="post":
            assert False, "not done"
        inds = list(np.argsort(scores))
        scores = [scores[i] for i in inds]
        return inds, scores

    def plot_sorted_by(self, sort_by, N=20):

        def make_title(ind):
            pri = self._prior_probs[ind]*100
            likeli = self._likeli_scores[ind]*100

            return f"pri:{pri:.2f}, li{likeli:.2f}"
        
        list_of_parsestrokes = self.Dat.parser_list_of_parses(self.IndTrial, kind="strokes")

        ## -- Plot the actual trial
        self.Dat.plotMultTrials([self.IndTrial], which_strokes="strokes_beh")
        self.Dat.plotMultTrials([self.IndTrial], which_strokes="strokes_task")

        ## -- Plot trials, sorted by their scores
        inds, scores = self.inds_sorted_increasing(sort_by)
        liststrokes = [list_of_parsestrokes[i] for i in inds]
        assert len(inds)==len(liststrokes)

        # Plot bottom N
        # scores = list(np.log(scores)) # logprobs
        indsthis = inds[:N]
        scoresthis = scores[:N]
        strokeslist = liststrokes[:N]
        # titles = [f"{sort_by[:3]}:{s:.2f}" for s in scoresthis]
        titles = [make_title(ind) for ind in indsthis]
        self.Dat.plotMultStrokes(strokeslist, titles=titles);
            
        indsthis = inds[::-1][:N]
        scoresthis = scores[::-1][:N]
        strokeslist = liststrokes[::-1][:N]
        titles = [make_title(ind) for ind in indsthis]
        self.Dat.plotMultStrokes(strokeslist, titles=titles);

    def plot_scatter_likeli_prior(self, log_probs=True):
        fig, ax = plt.subplots(1,1, figsize=(6,6))
        if log_probs:
            p = np.log(self._prior_probs)
        else:
            p = self._prior_probs
        ax.plot(p, self._likeli_scores, "xk")
        ax.set_xlabel('prior probs')
        ax.set_ylabel('likeli scores')
        return fig








