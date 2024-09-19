""" Lots of these tools are copied from Reuben Feinman's GNS-Modeling repo
"""
import numpy as np

import warnings
import math
import itertools
import torch
from pybpl.data import unif_space
# from pybpl.matlab.bottomup import generate_random_parses

# from ...omniglot.minimal_splines import fit_minimal_spline
from .utils import sample_from_generator


def apply_config(parse, config):
    from pybpl.bottomup.initialize.walker_stroke import WalkerStroke
    from .parser_stroke import ParserStroke

    ns = len(parse)
    order, direction = config
    # print(order, direction)

    if isinstance(parse[0], ParserStroke):
        # make copies
        parse_ = [parse[order[i]].copy_object() for i in range(ns)]
    else:
        parse_ = [parse[order[i]] for i in range(ns)]

    if isinstance(parse_[0], list):
        # if each traj is a sequence of graph nodes.
        parse_ = [parse_[i][::-1] if direction[i] else parse_[i] for i in range(ns)]
    elif isinstance(parse_[0], WalkerStroke):
        assert False, "not tested"
        for i in range(ns):
            if direction[i]:
                parse_[i].flipped = not parse_[i].flipped
    elif isinstance(parse_[0], dict):
        assert False, "not tested"
        for i in range(ns):
            if direction[i]:
                parse_[i]["flipped"] = not parse_[i]["flipped"]
    elif isinstance(parse_[0], ParserStroke):
        for i in range(ns):
            if direction[i]:
                parse_[i].do_flip()
    else:
        parse_ = [parse_[i].flip(dims=[0]) if direction[i] else parse_[i] for i in range(ns)]
    # print("---")
    # [p.print_elements() for p in parse_]
    return parse_

def search_parse(parse, score_fn=None, configs_per=100, trials_per=800, max_configs=1e6,
    allow_hack=False):
    """
    LT modifications:
    - score_fn, leave as None to take random
    """
    if trials_per<configs_per:
        trials_per = configs_per
    # assert trials_per >= configs_per
    ns = len(parse)
    
    # if ns > 9:
    #     if False:
    #         warnings.warn('parse searching not yet implemented for '
    #                       'large characters with ns > 9.')
    #         return [], []
    if ns > 26:
        warnings.warn('parse searching not yet implemented for '
                      'large characters with ns > 26.')
        return [], []        
    elif ns >=12:
        if allow_hack:
            # then just take flips
            ordering_configs = [range(ns)]
            direction_configs = itertools.product([False, True], repeat=ns)
            configs = itertools.product(ordering_configs, direction_configs)

            nconfigs = 2**ns
        else:
            return [], []

    elif ns >= 9:
        if allow_hack:
            # then just get reorderings, but not flippings.
            # get all ordering & direction configurations (as generators)
            ordering_configs = itertools.permutations(range(ns))
            direction_configs = itertools.product([False], repeat=ns)
            configs = itertools.product(ordering_configs, direction_configs)

            nconfigs = math.factorial(ns)
        else:
            return [], []
    else:
        # get all ordering & direction configurations (as generators)
        ordering_configs = itertools.permutations(range(ns))
        direction_configs = itertools.product([False,True], repeat=ns)
        configs = itertools.product(ordering_configs, direction_configs)

        nconfigs = math.factorial(ns) * 2**ns

    # if nelt is too large (>10 mil) then limit to the first 1 mil indices
    # This is hack to allow parsing of large characters (instead of warning above)
    # (LT)
    if nconfigs>max_configs:
        nconfigs = max_configs

    # if we have too many configurations, sample subset
    if nconfigs > trials_per:
        configs = sample_from_generator(
            configs, nelt=nconfigs, nsamp=trials_per, replace=ns>7)

    # score configurations and take top-(configs_per)
    parses = [apply_config(parse, c) for c in configs]
    if score_fn is None:
        if len(parses)>configs_per:
            import random
            parses = random.sample(parses, configs_per)
        log_probs = torch.ones((len(parses)))
        return parses, log_probs
    else:
        log_probs = score_fn(parses)
        log_probs, idx = torch.sort(log_probs, descending=True)
        parses = [parses[i] for i in idx]
        return parses[:configs_per], log_probs[:configs_per]


