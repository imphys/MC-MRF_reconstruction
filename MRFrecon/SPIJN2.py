# -*- coding: utf-8 -*-
"""

@author: Martijn Nagtegaal
SPIJN with support for fixed parameters added
 
"""
import os
import signal
import time
import warnings

import numpy as np
import scipy as sp
import scipy.optimize
from joblib import Parallel, delayed
from scipy import sparse as scsp

from . import d_sel_operations as dso


# import multiprocessing
# @profile
def nnls(A, x, sparse=False):
    res = scipy.optimize.nnls(A, x)[0]
    if sparse:
        # Better perform the transformation to sparsity for more columns at once.
        return scsp.csr_matrix(res).T
    else:
        return res


def multiply(x, y):
    if len(y.shape) == 1:
        try:
            y = np.broadcast_to(y[:, None], x.shape)
        except ValueError:
            y = np.broadcast_to(y[None, :], x.shape)
    if scsp.issparse(x):
        return x.multiply(y).tocsc()
    else:
        x *= y
        return x


def norm(X, ord=None, axis=None):
    if scsp.issparse(X):
        return scsp.linalg.norm(X, ord=ord, axis=axis)
    else:
        return np.linalg.norm(X, ord=ord, axis=axis)


def __get_n_jobs(n_jobs=None):
    if isinstance(n_jobs, int):
        return n_jobs
    else:
        warnings.warn('n_jobs was not set explicitly, maximum is used, not always an improvement!')
        try:  # Only works on some linux versions, better for cluster
            return len(os.sched_getaffinity(0))
        except AttributeError:
            return os.cpu_count() - 1  # Leave 1 unused, leave some space


def _get_n_jobs(n_jobs=None):
    n_jobs = __get_n_jobs(n_jobs)
    if n_jobs == 2:
        n_jobs = 1
    return n_jobs


def maxcomps_repr(C, num_comp):
    """Return the main components for each signal with indices"""
    if scsp.issparse(C):
        if not scsp.isspmatrix_csc(C):
            C = C.tocsc()
        Sfull = np.empty((C.shape[1], num_comp)) + np.nan
        Cr = np.zeros_like(Sfull) + np.nan
        for c in range(C.shape[1]):
            cslice = slice(*C.indptr[c:c + 2])
            cind = C.indices[cslice]
            if len(cind) == 0:
                continue
            cvals = C.data[cslice]
            sind = np.argsort(-cvals)
            nci = min([len(sind), num_comp])
            Sfull[c, :nci] = cind[sind][:nci]
            Cr[c, :nci] = cvals[sind][:nci]
    else:
        Sfull = np.argsort(-C, axis=0)[:num_comp].T
        Cr = np.empty_like(Sfull, dtype=float)
        for k, (s, Cc) in enumerate(zip(Sfull, C.T)):
            Cr[k] = Cc[s]

    return Cr, Sfull


# @profile
def lsqnonneg2(Y, red_dict, C0=None, out_z=None, out_rel_err=None,
               fixed_par_img_masks=None, fixed_par_dict_masks=None, return_r=False,
               L=0, S=None, n_jobs=5, sparse=False, yblock=1e5, verbose=False):
    yblock = int(yblock)
    red_dict = red_dict.T
    n_jobs = _get_n_jobs(n_jobs)
    print(f'number of cores {n_jobs}')

    if S is not None:
        diclen = len(S)
    elif fixed_par_dict_masks is None:
        diclen = red_dict.shape[0]
    else:
        diclen = np.count_nonzero(fixed_par_dict_masks[0])
    z_shape = (diclen, Y.shape[1])
    if out_z is None and not sparse:
        out_z = np.empty(z_shape)
    elif sparse:
        out_z = scsp.lil_matrix(z_shape)
    #        out_z_2 = np.empty((diclen,Y.shape[1]))
    if out_rel_err is None:
        out_rel_err = np.empty(Y.shape[1])
    with Parallel(n_jobs=n_jobs, ) as parallel:

        if fixed_par_dict_masks is not None:

            for i, (dictionary_mask, measurement_mask) in enumerate(zip(fixed_par_dict_masks, fixed_par_img_masks)):
                s_sel = measurement_mask
                R = red_dict[dictionary_mask]
                if S is not None:
                    R = R[S]
                s_indx = np.arange(len(s_sel))[s_sel]
                #            R1 = R.T
                nsig = len(s_indx)
                nblocks = int((nsig - 1) / yblock) + 1
                for k in range(nblocks):

                    st = k * yblock
                    end = np.min([(k + 1) * yblock, nsig])
                    if verbose == 2 and end / yblock > .1:
                        print('d_selection {}/{} nblocks {}/{}'.format(
                            i + 1, len(fixed_par_dict_masks), k + 1, nblocks))
                    sl = slice(st, end)
                    if nsig > 0:
                        Ysub = Y[:, s_indx[sl]]
                        if n_jobs > 2:
                            res = np.asarray(parallel(
                                delayed(lambda y: sp.optimize.nnls(R.T, y)[0])(Ysub[:, i]) for i in
                                range(Ysub.shape[1])))
                        else:
                            res = np.apply_along_axis(lambda x: nnls(R.T, x), 1, Ysub.T)
                        if sparse:
                            res[res < 1e-15] = 0
                        res = res.T
                        out_z[:, s_indx[sl]] = res
        #            for i in s_indx:
        #                out_z[:,i],_ = sp.optimize.nnls(R1,Y[:,i])
        else:
            R = red_dict
            if S is not None:
                R = R[S]
            nsig = Y.shape[1]
            nblocks = int((nsig - 1) / yblock) + 1
            for k in range(nblocks):

                st = k * yblock
                end = np.min([(k + 1) * yblock, Y.shape[1]])
                sl = slice(st, end)
                if verbose == 2 and end / yblock > .1:
                    print('nblocks {}/{}'.format(k + 1, nblocks))
                Ysub = Y[:, sl]
                if n_jobs > 1:
                    res = np.asarray(parallel(
                        delayed(lambda y: scipy.optimize.nnls(R.T, y)[0])(Ysub[:, i]) for i in range(Ysub.shape[1])))
                else:
                    res = np.apply_along_axis(lambda x: nnls(R.T, x, sparse=False), 1, Ysub.T)
                if sparse:
                    res[res < 1e-15] = 0
                res = res.T
                if st == 0 and end == Y.shape[1]:
                    out_z[:, sl] = res
                else:
                    out_z[:, sl] = res

        #        for i in range(Y.shape[1]):
        #            out_z[:,i],_ = sp.optimize.nnls(R1,Y[:,i])

    if sparse:
        if not scsp.isspmatrix_csc(out_z):
            out_z = scsp.csc_matrix(out_z)

    Ycalc = dso.vecmat(out_z.T, red_dict, fixed_par_img_masks=fixed_par_img_masks,
                       fixed_par_dict_masks=fixed_par_dict_masks, S=S)
    r = Y - Ycalc.transpose()
    out_rel_err = np.linalg.norm(r, axis=0) / np.linalg.norm(Y, axis=0)
    #    efficiency()
    if not return_r:
        return out_z, out_rel_err
    else:
        return out_z, out_rel_err, r


# @profile
def rewlsqnonneg2(Y, red_dict, w, C0=None, out_z=None,
                  out_rel_err=None, fixed_par_img_masks=None,
                  fixed_par_dict_masks=None,
                  return_r=False, L=0, S=None, debug=False,
                  n_jobs=5, sparse=False, yblock=1e5, verbose=False, norms=None):
    """Function to replace:
    W = scsp.diags(w**.5)
    D = red_dict1 @ W
    if L>0:
        D = np.vstack((D,L**2*np.ones(D.shape[1])))
        Y1 = np.vstack((Y,np.zeros(Y.shape[1])))
    else:
        Y1 = Y
    Q,rel_err,r = lsqnonneg2(Y1,D,C0=C0,L=L,return_r = True,fixed_par_img_masks = fixed_par_img_masks,fixed_par_dict_masks=fixed_par_dict_masks,S=S)
    C = W @ Q
    
    Performs a reweighted NN least squares signal-wise
    """
    yblock = int(yblock)
    red_dict = red_dict.T
    n_jobs = _get_n_jobs(n_jobs)
    print(f'number of cores {n_jobs}')

    #    clear_pseudo_hist()
    if S is not None:
        diclen = len(S)
    elif fixed_par_dict_masks is None:
        diclen = red_dict.shape[0]
    else:
        diclen = np.count_nonzero(fixed_par_dict_masks[0])

    z_shape = (diclen, Y.shape[1])
    if out_z is None and not sparse:
        out_z = np.empty(z_shape)
    elif sparse:
        out_z = scsp.lil_matrix(z_shape)
    if out_rel_err is None:
        out_rel_err = np.empty(Y.shape[1])
    if w is None:
        w = np.ones(diclen)

    #    W = scsp.diags(w**.5)
    Y1 = np.vstack((Y, np.zeros(Y.shape[1])))
    regfac = np.ones(len(w))
    with Parallel(n_jobs=n_jobs, ) as parallel:
        if fixed_par_dict_masks is not None:
            for i, (dictionary_mask, measurement_mask) in enumerate(zip(fixed_par_dict_masks, fixed_par_img_masks)):
                s_sel = measurement_mask
                R = red_dict[dictionary_mask]
                regfac2 = regfac.copy()
                if norms is not None:
                    regfac2 = norms[dictionary_mask]
                    if S is not None:
                        regfac2 = regfac2[S]

                if S is not None:
                    R = R[S]

                R = R.T * w ** .5
                R = np.vstack((R, L / regfac2))

                s_indx = np.arange(len(s_sel))[s_sel]
                nsig = len(s_indx)
                nblocks = int((nsig - 1) / yblock) + 1
                for k in range(nblocks):
                    st = k * yblock
                    end = np.min([(k + 1) * yblock, nsig])
                    if verbose == 2 and end / yblock > .1:
                        print('d_selection {}/{} nblocks {}/{}'.format(
                            i + 1, len(fixed_par_dict_masks), k + 1, nblocks))
                    sl = slice(st, end)
                    if len(s_indx) > 0:
                        Ysub = Y1[:, s_indx[sl]]
                        if n_jobs >= 2:
                            res = np.asarray(parallel(
                                delayed(lambda y: sp.optimize.nnls(R, y)[0])(Ysub[:, i]) for i in
                                range(Ysub.shape[1]))).T
                        else:
                            res = np.apply_along_axis(lambda x: sp.optimize.nnls(R, x)[0], 0, Ysub)
                        if sparse:
                            res[res < 1e-15] = 0
                        out_z[:, s_indx[sl]] = res

        #            if len(s_indx)>0:
        #                out_z[:,s_indx] = np.apply_along_axis(lambda x:sp.optimize.nnls(R,x)[0],0,Y1[:,s_indx])

        else:
            R = red_dict

            if norms is not None:
                regfac = norms
            if S is not None:
                R = R[S]
                regfac = regfac[S]
            R = R.T * w ** .5
            R = np.vstack((R, L / regfac))
            #        out_z = np.apply_along_axis(lambda x:sp.optimize.nnls(R,x)[0],0,Y1)
            nsig = Y.shape[1]
            nblocks = int((nsig - 1) / yblock) + 1
            for k in range(nblocks):

                st = k * yblock
                end = np.min([(k + 1) * yblock, Y.shape[1]])
                if verbose == 2 and end / yblock > .1:
                    print('nblocks {}/{}'.format(k + 1, nblocks))
                sl = slice(st, end)
                Ysub = Y1[:, sl]
                if n_jobs > 2:
                    res = np.asarray(parallel(
                        delayed(lambda y: sp.optimize.nnls(R, y)[0])(Ysub[:, i]) for i in range(Ysub.shape[1])))
                else:
                    res = np.apply_along_axis(lambda x: sp.optimize.nnls(R, x)[0], 0, Ysub).T
                if sparse:
                    res[res < 1e-15] = 0
                res = res.T
                out_z[:, sl] = res
        #        else:
        #            if n_jobs>2:
        #                out_z = np.asarray(Parallel(n_jobs=n_jobs)(delayed(lambda y: sp.optimize.nnls(R, y)[0])(Y1[:,i]) for i in range(Y1.shape[1])) ).T
        #            else:
        #                out_z = np.apply_along_axis(lambda x:sp.optimize.nnls(R,x)[0],0,Y1)
    if sparse:
        if not scsp.isspmatrix_csc(out_z):
            out_z = scsp.csc_matrix(out_z)
    out_z = multiply(out_z, w ** .5)
    Ycalc = dso.vecmat(out_z.T, red_dict, fixed_par_img_masks=fixed_par_img_masks,
                       fixed_par_dict_masks=fixed_par_dict_masks, S=S)
    r = Y - Ycalc.transpose()
    out_rel_err = np.linalg.norm(r, axis=0) / np.linalg.norm(Y, axis=0)
    if not return_r:
        return out_z, out_rel_err
    else:
        return out_z, out_rel_err, r


def SPIJN(Y, red_dict, weight_type=1, num_comp=None, p=0,
          max_iter=20, correct_im_size=False,
          verbose=True, norm_sig=True, tol=1e-4, L=0, prun=2,
          fixed_par_img_masks=None, fixed_par_dict_masks=None, bd_adj=False,
          C1=None, n_jobs=None, minimal_comp=0, final_it=False,
          sparse=False, yblock=1e4, debug=False, norms=None, pool=False):
    """Perform the SPIJN algorithm
    - weight_type: as given in the thesis
    - num_comp: number of components
    - p: value as used in the reweighting scheme 1. Default 0 works fine
    - max_iter: maximum number of iterations, 20 is normally fine
    - verbose: more output
    - norm_sig: Normalise the signal or not. False is fine
    - tol: Tolerance used, when are the iterations stopped, based on 
        ||C^k-C^{k+1}||_F^2/||C^k||_F^2<tol
    - L: the regularisation parameter used. Better use with correct_im_size
    - prun: if False no pruning, otherwise the number of iterations afterwards 
            the pruning of unused atoms in the dictionary takes place
    """
    if not pool:
        signal.signal(signal.SIGINT, signal.default_int_handler)  # To stop the iteration
    M, J = Y.shape
    if fixed_par_dict_masks is not None:
        N = np.sum(fixed_par_dict_masks[0])
    else:
        N = red_dict.shape[1]
    n_jobs = _get_n_jobs(n_jobs)
    #    C = np.zeros((N,J))
    if num_comp is None:
        num_comp = [0, 10]
    elif type(num_comp) == int:
        num_comp = [0, num_comp]
    if minimal_comp > 0 and num_comp[0] == 0:
        num_comp[0] = minimal_comp
    if norms is None:
        norms = np.ones(red_dict.shape[1])
    eps = 1e-4
    if norm_sig:
        norm_Y = np.linalg.norm(Y, ord=2, axis=0)
        Y = Y / norm_Y
    if correct_im_size:  # Correct regularisation for number of voxels
        L = L * np.log10(J)
    w = np.ones(N)
    t0 = time.process_time()
    if C1 is None:
        C1, r = lsqnonneg2(Y, red_dict, fixed_par_img_masks=fixed_par_img_masks,
                           fixed_par_dict_masks=fixed_par_dict_masks, n_jobs=n_jobs,
                           sparse=sparse, yblock=yblock, verbose=verbose)
        print('matching time: {0:.5f}s'.format(time.process_time() - t0))
    else:
        print('Reused old JNNLS solution')
    C = C1.copy()
    if prun == True:
        prun = 1
    used_indices = np.arange(N)
    WW = [w.copy()]
    k = 0

    prun_thres = 1e-15
    if weight_type > 0:
        if fixed_par_dict_masks is not None and weight_type != 1:
            print('Untested behaviour for weight_type {} with fixed parameters.'.format(weight_type))

        #        red_dict1 = red_dict
        N1 = N
        try:
            for k in range(1, max_iter):
                if prun and (str(prun).startswith('cont') or prun == k):
                    prune_comp = np.asarray(C.sum(axis=1) / J > prun_thres).flatten()
                    used_indices = used_indices[prune_comp]
                    C = C[prune_comp]
                    N1 = sum(prune_comp)
                    if verbose and np.any(~prune_comp):
                        print('Pruned percentage {},rest:{}.'.format(100 - N1 / N * 100, N1))
                    if N1 < 1000:
                        n_jobs = 1
                    w = w[prune_comp]

                    if verbose == 2:
                        verbose = True
                if weight_type == 1:
                    w = (norm(C, 2, axis=1) + eps) ** (1 - p / 2)
                elif weight_type == 2:
                    w = (norm(C, 2, axis=1) ** 2 + eps * norm(C, 2, axis=1))
                if bd_adj and w[0] > 1:
                    w[0] = 1
                w[w < eps] = eps
                if debug:
                    v = np.zeros(N)
                    v[used_indices] = w
                    v.append(w.copy())
                C0 = C.copy()
                t0 = time.process_time()
                C, rel_err, r = rewlsqnonneg2(Y, red_dict, w, C0=C0, out_z=C if not sparse else None, L=L,
                                              return_r=True, fixed_par_img_masks=fixed_par_img_masks,
                                              fixed_par_dict_masks=fixed_par_dict_masks,
                                              S=used_indices, n_jobs=n_jobs, sparse=sparse, yblock=yblock,
                                              verbose=verbose, norms=norms)
                if verbose: print('matching time: {0:.5f}s'.format(time.process_time() - t0))

                rel = norm(C - C0, ord='fro') / norm(C, ord='fro')
                if verbose:
                    print('k: {} relative difference between iterations {},elements: {}'.format(k, rel, np.sum(
                        C.sum(1) > 1e-4)))
                if rel < tol or np.isnan(rel) or np.sum(C.sum(1) > 1e-4) <= num_comp[0]:
                    if np.isnan(rel) or np.sum(C.sum(1) > 1e-4) < num_comp[0]:
                        C = C0
                    if verbose:
                        print('Stopped after iteration {}'.format(k))
                    break

        except KeyboardInterrupt:
            C = C0
        if str(prun).startswith('cont') or (prun and k >= prun):
            w0 = w
            w = np.zeros(N)
            w[used_indices] = w0
            C0 = C
            if sparse:
                C = scsp.lil_matrix((N, J))
            else:
                C = np.zeros((N, J))
            C[used_indices] = C0
            if debug:
                WW_ = WW
                for ww in WW[len(WW[0]):]:
                    ww_ = np.zeros(N)
                    ww_[used_indices] = ww
                    WW_.append(ww_)
                WW = WW_
        # Final solve:
    if k > 0 and final_it:
        prune_comp = np.asarray(C.sum(axis=1) > prun_thres).flatten()
        used_indices = np.arange(len(prune_comp))[prune_comp]  # Determine indices
        C0, r = lsqnonneg2(Y, red_dict, S=used_indices, sparse=sparse, yblock=yblock)
        rel_err = np.linalg.norm(r, axis=0) / np.linalg.norm(Y, axis=0)
        if sparse:
            C = scsp.lil_matrix((N, J))
        else:
            C = np.zeros((N, J))
        C[prune_comp] = C0
    if sparse:
        C = C.tocsc()
    # Ycalc = dso.vecmat(C.T, red_dict.T, fixed_par_img_masks=fixed_par_img_masks, fixed_par_dict_masks=fixed_par_dict_masks).T

    # r = Y - Ycalc  # red_dict@C
    if norm_sig:
        Ydiag = scsp.spdiags(norm_Y, 0, len(norm_Y), len(norm_Y))
        C = C @ Ydiag
    Cr, Sfull = maxcomps_repr(C, num_comp[1])

    # Components, indices, relative error, used weights and result of first iteration are returned.

    return Cr, Sfull, rel_err, WW, C1
