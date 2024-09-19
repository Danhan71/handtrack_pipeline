
DEBUG = False # Jax
if DEBUG:
    import jax.numpy as np
else:
    import numpy as np

def normscore(scores_all, ver, params=None):
    """given lsit of scalars (scores_all)  
    across parses, and a method (ver), output probabilsitys in 
    a list.
    TODO: dont use divide. 
    (it doesnt work well if there exist different signs.)
    - params, tuple of params, flexible depending on ver
    """

    if len(scores_all)==0:
        print(ver)
        print(params)
        assert False


    # scores_all = np.asarray(scores_all, dtype=np.float32)
    scores_all = np.asarray(scores_all)

    if ver=="divide":
        # simple, just divide by sum of alls cores
        # if any is negative, then subtracts it.
        # is very hacky!!!
        if np.any(scores_all>0) and np.any(scores_all<0):
            scores_all = scores_all - np.min(scores_all)
        s_sum = np.sum(scores_all)
        return scores_all/s_sum
        # return (score)/np.sum(scores_all)
    elif ver=="softmax":
        # softmax. first normalize scores so that within similar range
        # dividing by mean of absolute values of scores.. this is hacky...
        from scipy.special import softmax
        # scores_all = np.array([-5, 10, 25])
        # sumabs = np.sum(np.absolute(scores_all))

        # standardize scores. first subtract mean score. then divide by MAD
        invtemp = 1
        if params is None:
            scores_all = scores_all - np.mean(scores_all)
            meanabs = np.mean(np.absolute(scores_all))
            if meanabs>0:
                scores_all = scores_all/meanabs
        elif params=="raw":
            # do nothing to raw scores
            pass
        else:
            invtemp = params[0]
            subtrmean = params[1]

            if subtrmean==True:
                # then subtract mean. this useful if large variation across trials for mean
                scores_all = scores_all - np.mean(scores_all)
        scores_all = invtemp*scores_all
        probs = softmax(scores_all) 
        return probs
    elif ver=="log_softmax":
        # softmax. first normalize scores so that within similar range
        # dividing by mean of absolute values of scores.. this is hacky...
        # from scipy.special import log_softmax
        # scores_all = np.array([-5, 10, 25])
        # sumabs = np.sum(np.absolute(scores_all))

        # standardize scores. first subtract mean score. then divide by MAD
        invtemp = 1
        if params is None:
            scores_all = scores_all - np.mean(scores_all)
            meanabs = np.mean(np.absolute(scores_all))
            if meanabs>0:
                scores_all = scores_all/meanabs
        elif params=="raw":
            # do nothing to raw scores
            pass
        else:
            invtemp = params[0]
            subtrmean = params[1]

            if subtrmean==True:
                # then subtract mean. this useful if large variation across trials for mean
                scores_all = scores_all - np.mean(scores_all)
        invtemp = np.asarray(invtemp, dtype=np.float32)
        scores_all = invtemp*scores_all
        if DEBUG:
            from jax.nn import log_softmax
        else:
            from scipy.special import log_softmax

        log_probs = log_softmax(scores_all) 
        return log_probs
    elif ver=="log":
        # then assumes that scores are probs, and returns log probs
        # must be positive. can be >1 (e.g., likelihoods)
        MINPROB = 0.00001
        scores_all[scores_all==0] == MINPROB
        assert np.all(scores_all>0)
        return np.log(scores_all)

    else:
        assert False, "not coded!"


def posterior_score(likelis, priors, ver):
    if ver=="top1":
        c = np.random.choice(np.flatnonzero(priors == priors.max())) # this randomly chooses, if there is tiebreaker.
        post = likelis[c]
    elif ver=="maxlikeli":
        # is positive control, take the maximum likeli parse
        c = np.random.choice(np.flatnonzero(likelis == likelis.max()))
        post = likelis[c]
    elif ver=="prob_of_max_likeli":
        # Return the prior prob for ind that is max likeli
        c = np.random.choice(np.flatnonzero(likelis == likelis.max()))
        post = priors[c]
    elif ver=="weighted":
        # baseline - weighted sum of likelihoods by prior probabilities
        post = np.average(likelis, weights=priors)
    elif ver=="likeli_weighted":
        # uses likelihoods as weights.. this is like apositive control..
        # 1) convert likelis to probabilities by softmax
        probs = self.normscore(likelis, ver="softmax")
        post = np.average(likelis, weights=probs)
    elif ver=="logsumexp":
        # if likelis and priors are both logprobs, then goal is to get 
        # p(d|M) = SUM[ p(d|parse)p(parse|M) ] where sum is over parses and d
        # is a single datapoint (trial).
        # If these two inner terms are given as log probs, then log(p(d|M)) is logsumexp(likeli+prior)
        # So output will be log posterior prob.
        if DEBUG:
            from jax.scipy.special import logsumexp
        else:
            from scipy.special import logsumexp
        return logsumexp(likelis + priors)
    elif ver=="maxlikeli_for_permissible_traj":
        # Consider only parses whose priors are above 0 (probs). Take max likeli over thise
        # useful for chunks, etc, where prior assigns 0 or 1 to perms.
        # Returns the max likeli. priors must be in probs
        MINPROB = 0.01
        inds = priors>MINPROB
        return likelis[inds].max()
    else:
        print(ver)
        assert False, "not coded"
    return post

