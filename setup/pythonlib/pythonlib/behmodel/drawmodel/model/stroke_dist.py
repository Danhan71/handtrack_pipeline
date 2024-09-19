import torch
import torch.distributions as dist
from ..objects import StrokeType

class StrokeTypeDist(object):
    def __init__(self, lib):
        # is uniform
        self.isunif = lib.isunif
        # number of control points
        self.ncpt = lib.ncpt
        # sub-stroke count distribution
        self.pmat_nsub = lib.pmat_nsub
        # sub-stroke id distribution
        self.logStart = lib.logStart
        self.pT = lib.pT
        # shapes distribution
        self.shapes_mu = lib.shape['mu']
        self.shapes_Cov = lib.shape['Sigma']
        # invscales distribution
        scales_theta = lib.scale['theta']
        assert len(scales_theta.shape) == 2
        self.scales_con = scales_theta[:,0]  # gamma concentration
        # NOTE: PyTorch gamma dist uses rate parameter, which is inv of scale
        self.scales_rate = 1 / scales_theta[:,1]  # gamma rate

        self.logpThetas = lib.logpThetas
        self.Thetas = lib.Thetas

        self.xlim = lib.Spatial.xlim
        self.ylim = lib.Spatial.ylim

    def sample_subIDs(self):
        """
        Sample a sequence of sub-stroke IDs from the prior

        Parameters
        ----------
        nsub : tensor
            scalar; sub-stroke count

        Returns
        -------
        subid : (nsub,) tensor
            sub-stroke ID sequence
        """
        # set initial transition probabilities
        pT = torch.exp(self.logStart)
        # sub-stroke sequence is a list
        subid = dist.Categorical(probs=pT).sample().reshape(-1,)
        return subid

    # def score_subIDs(self, subid):
    #     """
    #     Compute the log-probability of a sub-stroke ID sequence under the prior

    #     Parameters
    #     ----------
    #     subid : (nsub,) tensor
    #         sub-stroke ID sequence

    #     Returns
    #     -------
    #     ll : (nsub,) tensor
    #         scalar; log-probability of the sub-stroke ID sequence
    #     """
    #     # set initial transition probabilities
    #     pT = torch.exp(self.logStart)
    #     # initialize log-prob vector
    #     ll = torch.zeros(len(subid), dtype=torch.float)
    #     # step through sub-stroke IDs
    #     for i, ss in enumerate(subid):
    #         # add to log-prob accumulator
    #         ll[i] = dist.Categorical(probs=pT).log_prob(ss)
    #         # update transition probabilities; condition on previous sub-stroke
    #         pT = self.pT(ss)

    #     return ll


    def sample_shapes_type(self, subid):
        """
        Sample the control points for each sub-stroke ID in a given sequence

        Parameters
        ----------
        subid : (nsub,) tensor
            sub-stroke ID sequence

        Returns
        -------
        shapes : (ncpt, 2, nsub) tensor
            sampled shapes of bsplines
        """
        if self.isunif:
            raise NotImplementedError
        # check that subid is a vector
        assert len(subid.shape) == 1
        # record vector length
        nsub = len(subid)
        # create multivariate normal distribution
        mvn = dist.MultivariateNormal(
            self.shapes_mu[subid], self.shapes_Cov[subid]
        )
        # sample points from the multivariate normal distribution
        shapes = mvn.sample()
        # transpose axes (nsub, ncpt*2) -> (ncpt*2, nsub)
        shapes = shapes.transpose(0,1)
        # reshape tensor (ncpt*2, nsub) -> (ncpt, 2, nsub)
        shapes = shapes.view(self.ncpt,2,nsub)

        return shapes

    # def score_shapes_type(self, subid, shapes):
    #     """
    #     Compute the log-probability of the control points for each sub-stroke
    #     under the prior

    #     Parameters
    #     ----------
    #     subid : (nsub,) tensor
    #         sub-stroke ID sequence
    #     shapes : (ncpt, 2, nsub) tensor
    #         shapes of bsplines

    #     Returns
    #     -------
    #     ll : (nsub,) tensor
    #         vector of log-likelihood scores
    #     """
    #     if self.isunif:
    #         raise NotImplementedError
    #     # check that subid is a vector
    #     assert len(subid.shape) == 1
    #     # record vector length
    #     nsub = len(subid)
    #     assert shapes.shape[-1] == nsub
    #     # reshape tensor (ncpt, 2, nsub) -> (ncpt*2, nsub)
    #     shapes = shapes.view(self.ncpt*2,nsub)
    #     # transpose axes (ncpt*2, nsub) -> (nsub, ncpt*2)
    #     shapes = shapes.transpose(0,1)
    #     # create multivariate normal distribution
    #     mvn = dist.MultivariateNormal(
    #         self.shapes_mu[subid], self.shapes_Cov[subid]
    #     )
    #     # score points using the multivariate normal distribution
    #     ll = mvn.log_prob(shapes)

    #     return ll

    def sample_invscales_type(self, subid):
        """
        Sample the scale parameters for each sub-stroke

        Parameters
        ----------
        subid : (nsub,) tensor
            sub-stroke ID sequence

        Returns
        -------
        invscales : (nsub,) tensor
            scale values for each sub-stroke
        """
        if self.isunif:
            raise NotImplementedError
        # check that it is a vector
        assert len(subid.shape) == 1
        # create gamma distribution
        gamma = dist.Gamma(self.scales_con[subid], self.scales_rate[subid])
        # sample from the gamma distribution
        invscales = gamma.sample()

        return invscales

    def score_invscales_type(self, subid, invscales):
        """
        Compute the log-probability of each sub-stroke's scale parameter
        under the prior

        Parameters
        ----------
        subid : (nsub,) tensor
            sub-stroke ID sequence
        invscales : (nsub,) tensor
            scale values for each sub-stroke

        Returns
        -------
        ll : (nsub,) tensor
            vector of log-likelihood scores
        """
        if self.isunif:
            raise NotImplementedError
        # make sure these are vectors
        assert len(invscales.shape) == 1
        assert len(subid.shape) == 1
        assert len(invscales) == len(subid)
        # create gamma distribution
        gamma = dist.Gamma(self.scales_con[subid], self.scales_rate[subid])
        # score points using the gamma distribution
        ll = gamma.log_prob(invscales)

        # mean = self.scales_con[subid]/self.scales_rate[subid]
        # variance = self.scales_con[subid]/(self.scales_rate[subid]**2)
        # print("inv scales")
        # print(self.scales_con[subid], self.scales_rate[subid])
        # print(mean, variance)
        # print(invscales)
        # print(ll)

        return ll

    def sample_part_type(self, k):
        """
        Sample a stroke type from the prior, conditioned on a stroke count

        Parameters
        ----------
        k : tensor
            scalar; stroke count

        Returns
        -------
        p : StrokeType
            part type sample
        """
        # sample the number of sub-strokes
        # sample the sequence of sub-stroke IDs
        ids = self.sample_subIDs() # list of ids
        # sample control points for each sub-stroke in the sequence
        shapes = self.sample_shapes_type(ids) # shape (5, 2, len(ids))
        # sample scales for each sub-stroke in the sequence
        invscales = self.sample_invscales_type(ids) # shape len(ids)
        # initialize the stroke type
        rotation = self.sample_rotation(ids)

        p = StrokeType(ids, shapes, invscales, rotation, self.xlim, self.ylim)

        return p

    def sample_rotation(self, ids):
        pThetas = torch.exp(self.logpThetas)
        rot = dist.Categorical(probs=pThetas).sample()
        return self.Thetas[rot].reshape(-1,)
        # return rot

    def score_part_type_monkey(self, k, ptype):
        """ same as score_part_type, but ignore subparts
        """
        subIDs_scores = self.score_subIDs(ptype.ids)
        # shapes_scores = self.score_shapes_type(ptype.ids, ptype.shapes)
        invscales_scores = self.score_invscales_type(ptype.ids, ptype.invscales)
        # print(subIDs_scores, shapes_scores, invscales_scores)
        # assert False
        # ll = torch.sum(subIDs_scores) + torch.sum(shapes_scores) \
        #      + torch.sum(invscales_scores)
        ll = torch.sum(subIDs_scores) + torch.sum(invscales_scores)

        return ll

    # def score_part_type(self, k, ptype):
    #     """
    #     Compute the log-probability of the stroke type, conditioned on a
    #     stroke count, under the prior

    #     Parameters
    #     ----------
    #     k : tensor
    #         scalar; stroke count
    #     ptype : StrokeType
    #         part type to score

    #     Returns
    #     -------
    #     ll : tensor
    #         scalar; log-probability of the stroke type
    #     """
    #     nsub_score = self.score_nsub(k, ptype.nsub)
    #     subIDs_scores = self.score_subIDs(ptype.ids)
    #     shapes_scores = self.score_shapes_type(ptype.ids, ptype.shapes)
    #     invscales_scores = self.score_invscales_type(ptype.ids, ptype.invscales)
    #     ll = nsub_score + torch.sum(subIDs_scores) + torch.sum(shapes_scores) \
    #          + torch.sum(invscales_scores)

    #     return ll