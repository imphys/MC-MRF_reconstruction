# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 09:36:24 2018

@author: Martijn Nagtegaal

Functions to make matrix multiplications able to deal with b1rel dictionaries
"""
import numpy as np
import scipy.sparse as scsp


def tensordot(red_dict, image, axes=1, calc_comp=None, fixed_par_img_masks=None, fixed_par_dict_masks=None):
    if calc_comp is None:
        _, nx, ny = image.shape
        calc_comp = np.ones((nx, ny)).astype(bool)
    if fixed_par_img_masks is None:
        r = image[:, calc_comp]
        A = np.tensordot(red_dict, r, axes=axes)
    else:
        r = image[:, calc_comp]
        A = np.empty((sum(fixed_par_dict_masks[0]), len(fixed_par_img_masks[0][calc_comp])))
        for d_sel, s_sel in zip(fixed_par_dict_masks, fixed_par_img_masks):
            A[:, s_sel[calc_comp]] = np.tensordot(red_dict[d_sel], r[:, s_sel[calc_comp]], axes=axes)
    return A


def tensordot1(red_dict, r, axes=1, fixed_par_img_masks=None, fixed_par_dict_masks=None, SC=None):
    if fixed_par_img_masks is None:
        R = red_dict
        if SC is not None:
            R = R[SC]
        A = np.tensordot(R, r, axes=axes)
    else:
        s_sel = fixed_par_img_masks
        if SC is not None:
            A = np.empty((len(SC), len(s_sel)))
        else:
            A = np.empty((sum(fixed_par_dict_masks[0]), len(s_sel)))
        for d_sel, s_sel in zip(fixed_par_dict_masks, fixed_par_img_masks):
            R = red_dict[d_sel]
            if SC is not None:
                R = R[SC]
            A[:, s_sel] = np.tensordot(R, r[:, s_sel], axes=axes)
    return A


# @profile
def multiply(b, D, fixed_par_img_masks=None, fixed_par_dict_masks=None, S=None):
    """Perform b*D where D is sliced as fixed_par_dict_masks"""
    if fixed_par_img_masks is None:
        try:
            Y = b * D
        except ValueError:
            Y = b * D[S]
    else:
        Y = np.zeros(b.shape, dtype=b.dtype)
        for d_sel, s_sel in zip(fixed_par_dict_masks, fixed_par_img_masks):
            Dsel = D[d_sel]
            if S is not None:
                Dsel = Dsel[S]
            Y[s_sel] = b[s_sel] * Dsel
    return Y


def vecmat(b, D, fixed_par_img_masks=None, fixed_par_dict_masks=None, S=None):
    """Perform b@D where D is sliced as fixed_par_dict_masks"""
    if fixed_par_img_masks is None:
        try:
            Y = b @ D
        except ValueError:
            Y = b @ D[S]
    else:
        if D.ndim > 1:
            Y = np.zeros(b.shape[:-1] + (D.shape[1],), dtype=b.dtype)
        else:
            Y = np.zeros(b.shape[:-1], dtype=b.dtype)
        for d_sel, s_sel in zip(fixed_par_dict_masks, fixed_par_img_masks):
            Dsel = D[d_sel]
            if S is not None:
                Dsel = Dsel[S]
            Y[s_sel] = b[s_sel] @ Dsel
    return Y


def matmul(D, C, fixed_par_img_masks=None, fixed_par_dict_masks=None, S=None):
    """Perform b@D where D is sliced as fixed_par_dict_masks"""
    if fixed_par_img_masks is None:
        try:
            Y = D @ C
        except ValueError:
            Y = D[S] @ C
    else:
        Y = np.zeros((D.shape[0], C.shape[1]))
        for d_sel, s_sel in zip(fixed_par_dict_masks, fixed_par_img_masks):
            Dsel = D[:, d_sel]
            if S is not None:
                Dsel = Dsel[S]
            Y[:, s_sel] = Dsel @ C[:, s_sel]
    return Y


def zsort(z, w, S, fixed_par_img_masks=None, fixed_par_dict_masks=None):
    if fixed_par_img_masks is None:
        A = np.einsum('ij,j->j', z, w[S])
    else:
        A = np.zeros(len(S))
        for d_sel, s_sel in zip(fixed_par_dict_masks, fixed_par_img_masks):
            A += np.einsum('ij,j->j', z[s_sel], (w[d_sel])[S])
    return A


# @profile
def JJNLS_weights(red_dict, C, w, eps, M, fixed_par_img_masks=None, fixed_par_dict_masks=None, S=None):
    """
    Function to replace 
    DWD = red_dict1@scsp.diags(w)@red_dict1.T
    w = np.linalg.norm(C,2,axis=1)**2+w-w**2*np.einsum('ij,ij->j',red_dict1,np.linalg.solve(eps*np.eye(M)+DWD,red_dict1))
    Probably not working!
                   
    """
    if fixed_par_dict_masks is None:
        if S is not None:
            red_dict = red_dict[:, S]
        DWD = red_dict @ scsp.diags(w) @ red_dict.T
        if scsp.issparse(C):
            w = scsp.linalg.norm(C, 2, axis=1) ** 2 + w - w ** 2 * np.einsum('ij,ij->j', red_dict,
                                                                             np.linalg.solve(eps * np.eye(M) + DWD,
                                                                                             red_dict))
        else:
            w = np.linalg.norm(C, 2, axis=1) ** 2 + w - w ** 2 * np.einsum('ij,ij->j', red_dict,
                                                                           np.linalg.solve(eps * np.eye(M) + DWD,
                                                                                           red_dict))
    else:
        w0 = w.copy()
        w += np.linalg.norm(C, 2, axis=1) ** 2
        DWD = np.zeros(2 * (red_dict.shape[0],))
        for k in range(len(fixed_par_dict_masks)):
            R = red_dict[:, fixed_par_dict_masks[k]]
            if S is not None:
                R = R[:, S]
            DWD = R @ scsp.diags(w0 ** -1) @ R.T
            w -= w0 ** 2 * np.einsum('ij,ij->j', R, np.linalg.solve(eps * np.eye(M) + DWD, R))

    return w
