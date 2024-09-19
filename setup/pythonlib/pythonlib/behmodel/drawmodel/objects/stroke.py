"""
Parts for sampling part tokens. Parts, together with relations between parts,
make up concepts.
"""

import torch

from pybpl import splines


class StrokeType(object):
    """
    Holds all type-level parameters of the stroke.

    Parameters
    ----------
    nsub : tensor
        scalar; number of sub-strokes
    ids : (nsub,) tensor
        sub-stroke ID sequence
    shapes : (ncpt, 2, nsub) tensor
        shapes types
    invscales : (nsub,) tensor
        invscales types
    """
    def __init__(self, ids, shapes, invscales, rotation, xlim, ylim):
        self.ids = ids
        self.invscales = invscales
        self.shapes = shapes
        self.Rotation = rotation
        self.Position = None

        # for image bounds
        self.xlim = xlim
        self.ylim = ylim

    # def __init__(self, shapes, invscales, xlim, ylim):
    #     super(StrokeToken, self).__init__()
    #     self.shapes = shapes
    #     self.invscales = invscales
    #     self.position = None

    #     # for image bounds
    #     self.xlim = xlim
    #     self.ylim = ylim


    @property
    def motor(self):
        """
        TODO
        """
        assert self.position is not None
        motor, _ = vanilla_to_motor(
            self.shapes, self.invscales, self.position
        )

        return motor

    @property
    def motor_spline(self):
        """
        TODO
        """
        assert self.position is not None
        _, motor_spline = vanilla_to_motor(
            self.shapes, self.invscales, self.position
        )

        return motor_spline

    def parameters(self):
        """
        Returns a list of parameters that can be optimized via gradient descent.

        Returns
        -------
        parameters : list
            optimizable parameters
        """
        parameters = [self.shapes, self.invscales]

        return parameters

    def lbs(self, eps=1e-4):
        """
        Returns a list of lower bounds for each of the optimizable parameters.

        Parameters
        ----------
        eps : float
            tolerance for constrained optimization

        Returns
        -------
        lbs : list
            lower bound for each parameter
        """
        lbs = [None, torch.full(self.invscales.shape, eps)]

        return lbs

    def ubs(self, eps=1e-4):
        """
        Returns a list of upper bounds for each of the optimizable parameters.

        Parameters
        ----------
        eps : float
            tolerance for constrained optimization

        Returns
        -------
        ubs : list
            upper bound for each parameter
        """
        ubs = [None, None]

        return ubs



def vanilla_to_motor(shapes, invscales, first_pos, neval=200):
    """
    Create the fine-motor trajectory of a stroke (denoted 'f()' in pseudocode)
    with 'nsub' sub-strokes.
    Reference: BPL/classes/Stroke.m (lines 203-238)

    :param shapes: [(ncpt,2,nsub) tensor] spline points in normalized space
    :param invscales: [(nsub,) tensor] inverse scales for each sub-stroke
    :param first_pos: [(2,) tensor] starting location of stroke
    :param neval: [int] number of evaluations to use for each motor
                    trajectory
    :return:
        motor: [(nsub,neval,2) tensor] fine motor sequence
        motor_spline: [(ncpt,2,nsub) tensor] fine motor sequence in spline space
    """
    for elt in [shapes, invscales, first_pos]:
        assert elt is not None
        assert isinstance(elt, torch.Tensor)
    assert len(shapes.shape) == 3
    assert shapes.shape[1] == 2
    assert len(invscales.shape) == 1
    assert first_pos.shape == torch.Size([2])
    ncpt, _, nsub = shapes.shape
    motor = torch.zeros(nsub, neval, 2, dtype=torch.float)
    motor_spline = torch.zeros_like(shapes, dtype=torch.float)
    previous_pos = first_pos
    for i in range(nsub):
        # re-scale the control points
        shapes_scaled = invscales[i]*shapes[:,:,i]
        # get trajectories from b-spline
        traj = splines.get_stk_from_bspline(shapes_scaled, neval)
        # reposition; shift by offset
        offset = traj[0] - previous_pos
        motor[i] = traj - offset
        motor_spline[:,:,i] = shapes_scaled - offset
        # update previous_pos to be last position of current traj
        previous_pos = motor[i,-1]

    return motor, motor_spline