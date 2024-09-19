""" derived from Rueben feinman: 
https://github.com/rfeinman/pyBPL

"""
import torch


def sub2ind(shape, rows, cols):
    """
    A PyTorch implementation of MATLAB's "sub2ind" function

    Parameters
    ----------
    shape : torch.Size or list or tuple
    rows : (n,) tensor
    cols : (n,) tensor

    Returns
    -------
    index : (n,) tensor
    """
    # checks
    assert isinstance(shape, tuple) or isinstance(shape, list)
    assert isinstance(rows, torch.Tensor) and len(rows.shape) == 1
    assert isinstance(cols, torch.Tensor) and len(cols.shape) == 1
    assert len(rows) == len(cols)
    assert torch.all(rows < shape[0]) and torch.all(cols < shape[1])
    if not len(shape) == 2:
        raise NotImplementedError('only implemented for 2D case.')
    # compute inds
    ind_mat = torch.arange(shape[0]*shape[1]).view(shape)
    index = ind_mat[rows.long(), cols.long()]

    return index


def check_bounds(myt, imsize):
    """
    Given a list of 2D points (x-y coordinates) and an image size, return
    a boolean vector indicating which points are out of the image boundary

    Parameters
    ----------
    myt : (k,2) tensor
        list of 2D points
    imsize : (2,) tensor
        image size; H x W

    Returns
    -------
    out : (k,) Byte tensor
        vector indicating which points are out of bounds
    """
    xt = myt[:,0]
    yt = myt[:,1]
    x_out = (torch.floor(xt) < 0) | (torch.ceil(xt) >= imsize[0])
    y_out = (torch.floor(yt) < 0) | (torch.ceil(yt) >= imsize[1])
    out = x_out | y_out

    return out


def seqadd(D, lind_x, lind_y, inkval):
    """
    Add ink to an image at the indicated locations

    Parameters
    ----------
    D : (m,n) tensor
        image that we'll be adding to
    lind_x : (k,) tensor
        x-coordinate for each adding point
    lind_y : (k,) tensor
        y-coordinate for each adding point
    inkval : (k,) tensor
        amount of ink to add for each adding point

    Returns
    -------
    D : (m,n) tensor
        image with ink added to it
    """
    assert len(lind_x) == len(lind_y) == len(inkval)
    imsize = D.shape

    # keep only the adding points that are in bounds
    lind_stack = torch.stack([lind_x, lind_y], dim=-1)
    out = check_bounds(lind_stack, imsize=imsize)
    lind_x = lind_x[~out].long()
    lind_y = lind_y[~out].long()
    inkval = inkval[~out]

    # return D if all adding points are out of bounds
    if len(lind_x) == 0:
        return D

    # flatten x-y indices
    lind = sub2ind(imsize, lind_x, lind_y).to(inkval.device)

    # add to image
    D = D.view(-1)
    D = D.scatter_add(0, lind, inkval)
    D = D.view(imsize)

    return D

def prepPoints(previous_points, new_points, resid_disp, mx, place_ink):
    p0 = previous_points
    # p0[p0[:,0] < 0, 0] = 0
    # p0[p0[:,0] > mx, 0] = mx
    # p0[p0[:,1] < 0, 1] = 0
    # p0[p0[:,1] > mx, 1] = mx

    tmp = new_points + resid_disp
    p1 = torch.floor(tmp)
    resid_disp = tmp - p1

    start_ob = torch.logical_or(torch.logical_or(p0[:,0] < 0, p0[:,0] > mx), torch.logical_or(p0[:,1] < 0, p0[:,1] > mx))
    end_ob = torch.logical_or(torch.logical_or(p1[:,0] < 0, p1[:,0] > mx), torch.logical_or(p1[:,1] < 0, p1[:,1] > mx))
    ob = torch.logical_and(start_ob, end_ob)
    place_ink[ob] = False

    # resid_disp[p1[:,0] < 0,:] = 0
    # resid_disp[p1[:,0] > mx,:] = 0
    # resid_disp[p1[:,1] < 0,:] = 0
    # resid_disp[p1[:,1] > mx,:] = 0
    #
    # p1[p1[:,0] < 0, 0] = 0
    # p1[p1[:,0] > mx, 0] = mx
    # p1[p1[:,1] < 0, 1] = 0
    # p1[p1[:,1] > mx, 1] = mx

    return p0, p1, resid_disp, place_ink

