"""
Concept type distributions for sampling concept types from pre-specified
type distributions.
"""
import torch
import torch.distributions as dist

from ..parameters import Parameters
from ..library.library import Library
from ..objects import (RelationType, RelationIndependent, RelationAttach,
                       RelationAttachAlong)
from ..objects import StrokeType, DrawingType
# from ..objects import ConceptType, CharacterType
from pybpl.splines import bspline_gen_s

from .stroke_dist import StrokeTypeDist
from .relation_dist import RelationTypeDist

# list of acceptable dtypes for 'k' parameter
int_types = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]

# from ..objects import DrawingType
# from ..objects import RelationType, RelationIndependent, RelationAttach, RelationAttachAlong, StrokeType, DrawingType



class DrawingDist(object):
    """
    Abstract base class for concept type distributions. Defines the prior
    distribution P(Type) for concept types
    """

    def __init__(self, lib):
        self.pdist = StrokeTypeDist(lib)
        self.rdist = RelationTypeDist(lib)
        assert len(lib.pkappa.shape) == 1
        self.kappa = dist.Categorical(probs=lib.pkappa)

        self.ps = Parameters()

        # token-level position distribution parameters
        means = torch.zeros(2)
        scales = torch.stack([lib.rel['sigma_x'], lib.rel['sigma_y']])
        self.loc_dist = dist.Independent(dist.Normal(means, scales), 1)

        # affine scale dist
        mu_scale = lib.affine['mu_scale']
        Cov_scale = lib.affine['Sigma_scale']
        self.A_scale_dist = dist.MultivariateNormal(mu_scale, Cov_scale)
        # affine translation dist
        mu_trans = torch.stack([lib.affine['mu_xtranslate'], lib.affine['mu_ytranslate']])
        scale_trans = torch.stack([lib.affine['sigma_xtranslate'], lib.affine['sigma_ytranslate']])
        self.A_trans_dist = dist.Independent(dist.Normal(mu_trans, scale_trans), 1)


    def sample_k(self):
        """
        Sample a stroke count from the prior

        Returns
        -------
        k : tensor
            scalar; stroke count
        """
        # sample from kappa
        # NOTE: add 1 to 0-indexed samples
        k = self.kappa.sample() + 1

        return k

    def score_k(self, k):
        """
        # Compute the log-probability of the stroke count under the prior

        # Parameters
        # ----------
        # k : tensor
        #     scalar; stroke count to score

        # Returns
        # -------
        # ll : tensor
        #     scalar; log-probability of the stroke count
        # """
        # # check if any values are out of bounds
        # if k > len(self.kappa.probs):
        #     ll = torch.tensor(-float('Inf'))
        # else:
        #     # score points using kappa
        #     # NOTE: subtract 1 to get 0-indexed samples
        #     ll = self.kappa.log_prob(k-1)

        return ll

    def sample_type(self, k=None):
        """
        Sample a character type

        Parameters
        ----------
        k : int
            optional; number of strokes for the type. If 'None' this will be
            sampled

        Returns
        -------
        c : CharacterType
            character type

        """


        """
        Sample a concept type from the prior

        Parameters
        ----------
        k : int or tensor
            scalar; the number of parts to use

        Returns
        -------
        ctype : ConceptType
            concept type sample
        """
        if k is None:
            # sample the number of parts 'k'
            k = self.sample_k()
        elif isinstance(k, int):
            k = torch.tensor(k)
        else:
            assert isinstance(k, torch.Tensor)
            assert k.shape == torch.Size([])
            assert k.dtype in int_types

        # initialize part and relation type lists
        P = []
        R = []
        # for each part, sample part parameters
        for _ in range(k):
            # sample the part type
            p = self.pdist.sample_part_type(k) # samples a stroke (including spatial coords)
            # sample the relation type
            r = self.rdist.sample_relation_type(P) 
            # append to the lists
            P.append(p)
            R.append(r)

        # create the concept type, i.e. a motor program for sampling
        # concept tokens

        # sample affine warp
        #A = self.sample_affine()
        A = None

        # sample image noise
        #epsilon = self.sample_image_noise()
        epsilon = self.ps.min_epsilon

        # sample image blur
        #blur_sigma = self.sample_image_blur()
        blur_sigma = self.ps.min_blur_sigma

        ctype = DrawingType(P, R, A,
            epsilon, blur_sigma)
        return ctype
