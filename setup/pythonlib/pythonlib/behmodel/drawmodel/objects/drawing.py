from ..objects import (RelationType, RelationIndependent, RelationAttach,
                       RelationAttachAlong, RelationToken)
from ..objects import StrokeType, DrawingType

# from .part import PartType, StrokeType, PartToken, StrokeToken
# from .relation import RelationType, RelationToken


class DrawingType(object):
    """
    Abstract base class for concept tokens. Concept tokens consist of a list
    of PartTokens and a list of RelationTokens.

    Parameters
    ----------
    P : list of PartToken
        part tokens
    R : list of RelationToken
        relation tokens
    """

    def __init__(self, P, R, affine, epsilon, blur_sigma):
        assert isinstance(P, list)
        assert isinstance(R, list)
        assert len(P) == len(R)
        for ptoken, rtoken in zip(P, R):
            assert isinstance(ptoken, StrokeType)
            assert isinstance(rtoken, RelationToken)
        self.part_tokens = P
        self.relation_tokens = R

        self.affine = affine
        self.epsilon = epsilon
        self.blur_sigma = blur_sigma


    # def __init__(self, P, R, affine, epsilon, blur_sigma):
    #     super(CharacterToken, self).__init__(P, R)
    #     for ptoken in P:
    #         assert isinstance(ptoken, StrokeToken)
    #     self.affine = affine
    #     self.epsilon = epsilon
    #     self.blur_sigma = blur_sigma

    def eval(self):
        """
        makes params require no grad
        """
        for param in self.parameters():
            param.requires_grad_(False)

    def to(self, device):
        """
        moves parameters to device
        TODO
        """
        pass
