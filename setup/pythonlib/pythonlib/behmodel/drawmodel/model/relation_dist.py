import torch
import torch.distributions as dist
from ..objects import RelationTransition, RelationToken, RelationType, RelationIndependent, RelationAttach, RelationAttachAlong, StrokeType

class RelationTypeDist(object):
    __relation_categories = ['unihist', 'start', 'end', 'mid']
    def __init__(self, lib):
        self.ncpt = lib.ncpt
        self.Spatial = lib.Spatial
        # distribution of relation categories
        self.rel_mixdist = dist.Categorical(probs=lib.rel['mixprob'])

    def sample_relation_type(self, prev_parts, DEBUG=True):
        """
        Sample a relation type from the prior for the current stroke,
        conditioned on the previous strokes

        Parameters
        ----------
        prev_parts : list of StrokeType
            previous part types

        Returns
        -------
        r : RelationType
            relation type sample
        """

        if DEBUG:
            return RelationTransition()
            # return None

        for p in prev_parts:
            assert isinstance(p, StrokeType)
        nprev = len(prev_parts)
        stroke_ix = nprev
        # first sample the relation category
        if nprev == 0:
            category = 'unihist'
        else:
            indx = self.rel_mixdist.sample()
            category = self.__relation_categories[indx]

        # now sample the category-specific type-level parameters
        if category == 'unihist':
            data_id = torch.tensor([stroke_ix])
            gpos = self.Spatial.sample(data_id)
            # convert (1,2) tensor to (2,) tensor
            gpos = torch.squeeze(gpos)
            r = RelationIndependent(
                category, gpos, self.Spatial.xlim, self.Spatial.ylim
            )
        elif category in ['start', 'end', 'mid']:
            # sample random stroke uniformly from previous strokes. this is the
            # stroke we will attach to
            probs = torch.ones(nprev)
            attach_ix = dist.Categorical(probs=probs).sample()
            if category == 'mid':
                # sample random sub-stroke uniformly from the selected stroke
                nsub = prev_parts[attach_ix].nsub
                probs = torch.ones(nsub)
                attach_subix = dist.Categorical(probs=probs).sample()
                # sample random type-level spline coordinate
                _, lb, ub = bspline_gen_s(self.ncpt, 1)
                eval_spot = dist.Uniform(lb, ub).sample()
                r = RelationAttachAlong(
                    category, attach_ix, attach_subix, eval_spot, self.ncpt
                )
            else:
                r = RelationAttach(category, attach_ix)
        else:
            raise TypeError('invalid relation')

        return r

    def score_relation_type(self, prev_parts, r):
        """
        Compute the log-probability of the relation type of the current stroke
        under the prior

        Parameters
        ----------
        prev_parts : list of StrokeType
            previous stroke types
        r : RelationType
            relation type to score

        Returns
        -------
        ll : tensor
            scalar; log-probability of the relation type
        """
        assert isinstance(r, RelationType)
        for p in prev_parts:
            assert isinstance(p, StrokeType)
        nprev = len(prev_parts)
        stroke_ix = nprev
        # first score the relation category
        if nprev == 0:
            ll = 0.
        else:
            ix = self.__relation_categories.index(r.category)
            ix = torch.tensor(ix, dtype=torch.long)
            ll = self.rel_mixdist.log_prob(ix)

        # now score the category-specific type-level parameters
        if r.category == 'unihist':
            data_id = torch.tensor([stroke_ix])
            # convert (2,) tensor to (1,2) tensor
            gpos = r.gpos.view(1,2)
            # score the type-level location
            ll = ll + torch.squeeze(self.Spatial.score(gpos, data_id))
        elif r.category in ['start', 'end', 'mid']:
            # score the stroke attachment index
            probs = torch.ones(nprev)
            ll = ll + dist.Categorical(probs=probs).log_prob(r.attach_ix)
            if r.category == 'mid':
                # score the sub-stroke attachment index
                nsub = prev_parts[r.attach_ix].nsub
                probs = torch.ones(nsub)
                ll = ll + dist.Categorical(probs=probs).log_prob(r.attach_subix)
                # score the type-level spline coordinate
                _, lb, ub = bspline_gen_s(self.ncpt, 1)
                ll = ll + dist.Uniform(lb, ub).log_prob(r.eval_spot)
        else:
            raise TypeError('invalid relation')

        return ll