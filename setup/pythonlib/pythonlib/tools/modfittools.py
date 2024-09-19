""" generic tools for model fitting, optimization, etc
"""

def minimize(fun, params0, bounds=None):
    """ general purpose code for optimiization.
    - fun takes in tuple (params) and returns
    cost, where more negatrive is better
    - params0 is initial params
    - bounds is list, where each element defines upper and lower
    leave empty to not have bounds (indepednly for each element).
    shoudl be same length as params0.
    e.g.,: bounds = [[], (-2,2), [], []]. or
    bounds = None, then all will be empty list.
    """

    from scipy.optimize import minimize as minim
#         params0=np.random.uniform(-8, 8, size=numparams)
    res = minim(fun, params0, method="L-BFGS-B", bounds=bounds)
    return res

