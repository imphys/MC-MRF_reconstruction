import itertools
import math
import os
import signal
import sys
import time
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import sigpy as sp
import sigpy.mri as spmri
import sigpy.plot as pl
from PIL import Image
from tqdm import tqdm

try:
    import nibabel as nib
except ImportError:
    print('Nibabel not found')

from . import d_sel_operations as dso

try:
    # from SPIJN import SPIJN
    from . import SPIJN2

    SPIJN = SPIJN2.SPIJN
except ImportError:
    raise ImportError('SPIJN was not found, retrieve it yourself from "https://github.com/MNagtegaal/SPIJN"')


def get_mask(mask_file):
    maskimage = Image.open(mask_file).convert('1')
    mask = np.array(maskimage)

    imagesize = mask.shape[0]
    numslice = int(mask.shape[1] // imagesize)
    if imagesize * numslice != mask.shape[1]:
        print("Mask image of invalid shape, width is not an integer multiple of height")
        sys.exit(1)
    mask = mask.reshape((imagesize, imagesize, numslice), order='F')
    return mask


class MrfData:
    """
    * ksp: numpy array of shape (#slices, #coils, #dynamics, #spirals_per_image, #point_on_spiral)
    * coord: numpy array of shape (#dynamics, #spirals_per_image, #point_on_spiral, #spiral_coordinates)
    * dictmat: numpy array of shape (#atoms, #dynamics)
    * t1list: list of t1 values of shape (#atoms)
    * t2list: list of t2 values of shape (#atoms)

    """

    def __init__(self, ksp: np.ndarray = None,
                 coord: np.ndarray = None,
                 dictmat: np.ndarray = None,
                 dictt1: np.ndarray = None,
                 dictt2: np.ndarray = None,

                 dictb1: np.ndarray = None,
                 b1_map: np.ndarray = None,
                 norms: np.ndarray = None,

                 spijn_mask: np.ndarray = None,

                 maps: np.ndarray = None,
                 mask: np.ndarray = None,
                 phasecorrection: np.ndarray = None,
                 comp_mat: np.ndarray = None,
                 compress_dict: np.ndarray = None,
                 imgseq: np.ndarray = None,
                 comp: np.ndarray = None,
                 index: np.ndarray = None,
                 ):
        self.ksp = ksp
        self.coord = coord
        self.dictmat = dictmat
        self.dictt1 = dictt1
        self.dictt2 = dictt2
        self.maps = maps
        self.mask = mask
        self.phasecorrection = phasecorrection
        self.comp_mat = comp_mat
        self.compress_dict = compress_dict
        self.imgseq = imgseq
        self.comp = comp
        self.spijn_mask = spijn_mask
        self.dictb1 = dictb1
        self.set_b1_map(b1_map=b1_map)
        # For handling fixed parameter maps
        self.fixed_par_dict_masks = None
        self.fixed_par_img_masks = None
        self.index = index
        self.norms = norms
        self._S = None  # Used for subselections in pruned subdictionaries

    def __len__(self):
        if self.compress_dict is None:
            return self.dictmat.shape[1]
        return self.compress_dict.shape[1]

    def set_b1_map(self, b1_map):
        self.b1_map = b1_map
        self.fixed_b1 = self.dictb1 is not None and b1_map is not None

    @property
    def numslice(self):
        if self.ksp is not None:
            return self.ksp.shape[0]
        elif self.imgseq is not None:
            return self.imgseq.shape[0]

    @property
    def numcoil(self):
        if self.ksp is not None:
            return self.ksp.shape[1]
        else:
            return len(self.maps)

    @property
    def numdyn(self):
        return self.ksp.shape[2]

    @property
    def rank(self):
        if self.compress_dict is None:
            return self.dictmat.shape[0]
        return self.compress_dict.shape[0]

    @property
    def imagesize(self):

        if self.maps is not None:
            return self.maps.shape[-1]
        elif self.mask is not None:
            return self.mask.shape[-1]
        return None

    @property
    def num_dict_atoms(self):
        """

        Returns: Find number of dictionary atoms that are used in matching (excludes fixed par)

        """
        try:
            if self.index is not None:
                return len(self.index)
        except AttributeError:
            pass
        if self.fixed_b1:
            self.fixed_par_processing()
            return self.fixed_par_dict_masks[0].sum()
        elif self.compress_dict is not None:
            return self.compress_dict.shape[-1]
        else:
            return self.dictmat.shape[-1]

    def to_image(self, vec: np.ndarray):
        """
        Reshape back to image from flattened array, only works for square images.
        Args:
            vec:

        Returns:

        """
        return vec.reshape(self.imagesize, self.imagesize)

    @staticmethod
    def to_vec(img: np.ndarray):
        """
        Function to flatten an array, but then slightly different
        Args:
            img:

        Returns:
            flattened array

        """
        return img.reshape(-1, 1).squeeze()

    def get_proxg(self, regtype: str = None, imshape: int = None, lam: float = None, lamscale: float = 1,
                  lambda_scales: np.ndarray = None, comp_device: int = None,
                  lambda_ksp: bool = False):
        """
        Function to define obtain proxg object as used for soft thresholding in lr-inversion
        Args:
            regtype: (None, findif, wavelet, wavelet_sc) type of regularization to use
            imshape:
            lam: used lambda, convergence parameter
            lamscale: lambda scaling option as:
                    lambda_scales = lamscale ** np.arange(imshape[1])
            lambda_scales: predefined lambda scales
            comp_device:
            lambda_ksp: preferred option for lambda scaling based on centre of ksp data

        Returns:
            dictionary with proxg object. only g is not None, otherwise errors occured...

        """
        if regtype is not None and (imshape is None or lam is None or comp_device is None):
            raise IOError(
                f'Regularisation input values are wrong, imshape={imshape}, lam={lam}, comp_device={comp_device}')

        if lam == 0:
            regtype = None

        if regtype == 'findif':
            warnings.warn('regtype findif not overly tested')
            G = sp.linop.FiniteDifference(imshape, axes=(-1, -2))
            proxg = sp.prox.L1Reg(G.oshape, lam)

            def g(x):
                device = sp.get_device(x)
                xp = device.xp
                with device:
                    return lam * xp.sum(xp.abs(x)).item()
        elif regtype == 'wavelet':
            warnings.warn('regtype wavelet not preferred')
            wave_name = 'db4'
            W = sp.linop.Wavelet(imshape[-2:], wave_name=wave_name)
            G = sp.linop.Wavelet(imshape[1:], wave_name=wave_name, axes=(-2, -1))
            G = sp.linop.Reshape((1,) + tuple(G.oshape), tuple(G.oshape)) * G * sp.linop.Reshape(G.ishape, imshape)
            proxg = sp.prox.UnitaryTransform(sp.prox.L1Reg(G.oshape, lam), G)

            # proxg = sp.prox.Stack([proxg]*imshape[0])
            def g(input):
                device = sp.get_device(input)
                xp = device.xp
                with device:
                    return lam * xp.sum(xp.abs(G(input))).item()

            g = None
        elif regtype == 'wavelet_sc':
            if lambda_ksp:
                ksp_centr = self.ksp[0, 0, :, 0, 0]  # np.linalg.norm(data2.ksp, axis=(1,4)).flatten()
                lambda_scales = np.abs(self.comp_mat.T @ ksp_centr)
            elif lambda_scales is None:
                lambda_scales = lamscale ** np.arange(imshape[1])
            lambda_scales = lambda_scales.reshape(imshape[:2] + (1, 1))
            wave_name = 'db4'
            W = sp.linop.Wavelet(imshape[-2:], wave_name=wave_name)
            G = sp.linop.Wavelet(imshape[1:], wave_name=wave_name, axes=(-2, -1))
            G = sp.linop.Reshape((1,) + tuple(G.oshape), tuple(G.oshape)) * G * sp.linop.Reshape(G.ishape, imshape)
            lambdas = sp.to_device(sp.util.resize(lam * lambda_scales, G.oshape[:2] + [1, 1]), comp_device)

            proxg = sp.prox.UnitaryTransform(sp.prox.L1Reg(G.oshape, lambdas), G)

            # proxg = sp.prox.Stack([proxg]*imshape[0])
            def g(input):
                device = sp.get_device(input)
                xp = device.xp
                with device:
                    return xp.sum(lambdas * xp.abs(G(input))).item()

            g = None
            G = None
        else:
            G = None
            g = None
            proxg = None
        return {'G': G, 'g': g, 'proxg': proxg}

    def restr_par_list(self, par='t1', index=None):
        """
        Restrict parameter list with respect to used indices
        Args:
            par:
            index:

        Returns:

        """
        if self.index is None:
            return eval('self.dict' + par)
        else:
            plist = eval('self.dict' + par)
            return plist[self.index]

    @property
    def restr_dictionary(self):
        """
        Obtain a restricted dictionary when self.index is not None
        """
        if self.index is None:
            return self.dictmat
        else:
            if self.fixed_b1 and '_S' in self.__dir__() and self._S is not None:
                S = self._S
                fixed_par_dict_masks = self.fixed_par_dict_masks
                indi = np.arange(self.dictmat.shape[1])
                m_indi = []
                for m in fixed_par_dict_masks:
                    m_indi.append(indi[m][S])
                m_indi = np.concatenate(m_indi)
            else:
                m_indi = self.index
            return self.dictmat[:, m_indi]

    def dict_svd(self, rank: int, verbose=0, comp_device=-1):
        """ Perform PCA on dictionary

        Args:
            rank (int): Rank or the approximation
            verbose (int): (optional) 0: works silently, 1: show plot
        Output:
            (array): low rank dicationary matrix of shape (R, timelength)
        """
        if comp_device > -1:
            try:
                import cupy
                gpu_dic = sp.to_device(self.restr_dictionary, comp_device)
                u, _, _ = cupy.linalg.svd(gpu_dic, full_matrices=False)
                u = sp.to_device(u, -1)
            except ImportError:
                u, s, vh = np.linalg.svd(self.restr_dictionary, full_matrices=False)
        else:
            u, s, vh = np.linalg.svd(self.restr_dictionary, full_matrices=False)
        if verbose:
            plt.figure()
            plt.semilogy(s)
            plt.title("Singular values")
            plt.show()
        # Cut svd matrix to rank R
        ur = np.zeros((self.dictmat.shape[0], rank), dtype=u.dtype)
        ur[:, 0:rank] = u[:, 0:rank]

        return ur

    @staticmethod
    def _discretize_fixed_parameter_map(fixed_parameter_map: np.ndarray, fixed_parameter_val: np.ndarray,
                                        ) -> object:
        """
        Discretizes the fixed parameter map to available values in dictionary

        Args:
            fixed_parameter_map (np.ndarray): array with values to discretize
            fixed_parameter_val (np.ndarray): array with values to discretize to


        Returns:
            fixed_parameter_map2: fixed parameter map with discretized values
            fixed_parameter_map2_ind: indices of discretized values with respect to fixed_parameter_val
        """
        vals_avail = np.unique(fixed_parameter_val)
        bins_p = (vals_avail[1:] + vals_avail[:-1]) / 2
        fixed_parameter_map2_ind = res_dig = np.searchsorted(bins_p, fixed_parameter_map)
        fixed_parameter_map2 = vals_avail[res_dig]
        fixed_parameter_map2[np.isnan(fixed_parameter_map)] = np.nan
        return fixed_parameter_map2, fixed_parameter_map2_ind

    def fixed_par_processing(self, redo: bool = False, flatten: bool = True, mask: bool = False):
        """
        Define fixed_par_dict_masks and fixed_par_img_masks, to know which part of the image is related to which
        pixels.

        Both are lists consisting of boolean arrays, meant to select dictionary atoms or pixels for seperated processing

        Args:
            redo (bool): redo on request
            flatten (bool): flatten images
            mask (bool): mask b1==0
        """
        try:
            if not redo and isinstance(self.fixed_par_dict_masks, list) and isinstance(self.fixed_par_img_masks, list):
                return
        except AttributeError:
            pass
        if self.fixed_b1:
            dictionary_masks, measurement_masks = self._fixed_par_processing(self.dictb1, self.b1_map, flatten=flatten,
                                                                             mask=mask)
        else:
            # Create boring lists if not necessary
            dictionary_masks = [np.ones(self.__len__(), dtype=bool)]
            if self.imgseq is not None:
                measurement_masks = [np.ones(self.imgseq.shape[:-1], dtype=bool)]
            elif self.maps is not None:
                measurement_masks = [np.ones(self.maps[:, 0].shape, dtype=bool)]
            elif self.mask is not None:
                measurement_masks = [np.ones(self.mask.shape, dtype=bool)]
            elif self.imagesize is not None:
                measurement_masks = [np.ones((1, self.imagesize, self.imagesize), dtype=bool)]
            else:
                measurement_masks = [slice(None)]
            if flatten:
                try:
                    measurement_masks = [m.flatten() for m in measurement_masks]
                except:
                    pass
        self.fixed_par_dict_masks = dictionary_masks
        self.fixed_par_img_masks = measurement_masks

    @staticmethod
    def _fixed_par_processing(dictb1: np.ndarray,
                              b1_map: np.ndarray, flatten: bool = True, mask: bool = False,
                              ) -> object:
        """

        Args:
            dictb1 (np.ndarray): list of b1 values as used in dictionary
            b1_map (np.ndarray): b1_map as scanned
            flatten (bool): whether or not to flatten image
            mask (bool): mask b1-values==0

        Returns:
            dictionary_masks (list): masks which part of the dictionary to use when
            measurement_masks (list): masks which voxel belong to which part of the dictionary

        """
        b1_map = b1_map.copy()
        if mask:
            b1_map[b1_map == 0] = np.nan
        else:
            b1_map[np.isnan(b1_map)] = 1.0
            b1_map[b1_map == 0] = 1.0
        fixed_parameter_map, fixed_parameter_map_ind = \
            MrfData._discretize_fixed_parameter_map(
                fixed_parameter_map=b1_map, fixed_parameter_val=dictb1)
        if flatten:
            fixed_parameter_map = fixed_parameter_map.flatten()
        # Now we need to create two lists with masks
        # to know which part of the dictionary should be used when
        dictionary_masks = []
        measurement_masks = []
        par_list = dictb1
        for v in np.unique(par_list):  # check for each fixed parameter value where it is represented
            dictionary_masks.append(np.isclose(par_list, v))
            measurement_masks.append(np.isclose(fixed_parameter_map, v))
        # Check whether parameter can be used as fixed parameter...
        for m in dictionary_masks:
            if len(m) != len(dictionary_masks[0]):
                raise ValueError
        return dictionary_masks, measurement_masks

    @staticmethod
    def density_comp_fact(coord: np.ndarray):
        """ Calculate density compensation factor

        Args:
            coord (array): coord array of shape (n_timeseries, n_samples,  2)
        Output:
            (array): Density compensation factors, array of shape (n_timeseries, n_samples)
        Sources from:
        Hoge, R. D., Kwan, R. K. S., & Pike, G. B. (1997). Density compensation functions for spiral MRI. Magnetic
        Resonance in Medicine, 38(1), 117–128. https://doi.org/10.1002/mrm.1910380117
        """
        out = np.zeros(list(coord.shape)[:2])
        for i in range(coord.shape[0]):
            k_mag = np.abs((coord[0, :, 0] ** 2 + coord[0, :, 1] ** 2) ** 0.5)
            kdiff = np.append([0], np.diff(k_mag))
            out[i, :] = k_mag * np.abs(kdiff)
            out[i, :] = k_mag * np.abs(kdiff)
        return out

    def compress_dictionary(self, rank, comp_device=-1):
        """ Compress_dictionary to low rank

        Compress the dictionary to low rank. Sets the compression matrix (comp_mat) and the compressed dictionary
        (compress_dict) in the class attributes
        self._S is used to restrict the dictionary according to dict-masks

        Args:
            rank (int): rank of the low rank dictionary

        Output:
            None
        """
        self.comp_mat = self.dict_svd(rank, comp_device=comp_device)
        self.compress_dict = np.matmul(self.comp_mat.T, self.dictmat)
        return

    @staticmethod
    def lr_sense_op(coord: np.ndarray, ur: np.ndarray,
                    imagesize: int, oversamp: float, maps: np.ndarray = None,
                    batchsize: int = None):
        """Low Rank sense operator, returns sigpy operator for a single slice.

        Make Low rank sense operator from sigpy linear operators
        Note:
            only acts on a signe slice,
            If maps are not supplied, only a LR single coil nufft is returned; else the full sense operator is returned

        Args:
            coord (array):  coordinate array of shape (timelength, coordlength, 2)
            ur (array): low rank compression matrix of shape (rank, timelength)
            imagesize (int): size of the image
            oversamp (float): oversampling ratio
            maps (array): (optional) sensitivity maps of birdcage coil of shape (n_coils, imgsize[0], imgsize[1])
            batchsize (int): (optional) batchsize for computation

        Output:
            (Linop): low rank sense operator of shape
                    <[n_coils, timelength, coordlength] x [rank, imgsize[0], imgsize[1]]>

        Makes low rank sense operator from standard Sigpy linops
        source: https://sigpy.readthedocs.io/
        """
        if batchsize is None:
            batchsize = coord.shape[0]
        rank = ur.shape[0]
        imgseq_shape = [1, rank, imagesize, imagesize]
        reshape_oshape = [imgseq_shape[0]] + [1] + imgseq_shape[1:]
        to_coil_img = sp.linop.Reshape(ishape=imgseq_shape, oshape=reshape_oshape)
        if maps is not None:
            # Make multiply linop to multiply image sequence with sense maps
            maps_op = sp.linop.Multiply(ishape=to_coil_img.oshape,
                                        mult=maps[:, :, None, :, :]) * to_coil_img
        else:
            # Make empty multiply linop
            maps_op = to_coil_img

        # Low rank interpolation linear operator
        list_interp_linop = []
        # batchsize = settings.getint('lr_sense_batchsize') # overwrite batchsize
        for i in range(math.ceil(coord.shape[0] / batchsize)):
            start = i * batchsize
            end = np.min([(i + 1) * batchsize, coord.shape[0]])
            sl = slice(start, end)

            interp_op = sp.linop.NUFFT(ishape=maps_op.oshape, coord=coord[sl, ...],
                                       oversamp=oversamp)
            fr_proj_op = sp.linop.Multiply(ishape=interp_op.oshape,
                                           mult=ur[None, None, :, sl, None, None])
            sum_op = sp.linop.Sum(ishape=fr_proj_op.oshape, axes=(2,))
            single_op = sum_op * fr_proj_op * interp_op
            list_interp_linop.append(single_op)
        batch_lr_interp_op = sp.linop.Vstack(list_interp_linop, axis=2)
        totalop = batch_lr_interp_op * maps_op
        totalop.repr_str = 'LR Sense'
        return totalop

    def mrf_espirit(self, imagesize, coil_batchsize=None, slice_batchsize=1, max_iter=50,
                    oversamp=1.25,
                    compute_device=-1, mask_thresh=0.0, verbose=1,
                    tol_fac=0):
        """ Calculate sensitivity maps using ESPIRiT.

        Calculate the sensitivity maps from the kspace data.
        First calculates the rank 1 image sequence per coil, than use ESPIRiT for sensitivity maps calculations
        Updates sensitivity maps (maps) in class
        ESPIRiT maps are masked by threshold, if a mask is supplied, that mask will be used instead

        Args:
            imagesize (int): size of the reconstruction matrix
            coil_batchsize (int): (optional) number of coils calculated simultaneously
                                    (defaults to number of coils in kspace)
            slice_batchsize (int): (optional) number of slices calculated simultaneously (defaults to 1)
            max_iter (int): (optional) number of conjugate gradient iterations
            oversamp (float): (optional) oversampling ratio
            compute_device (int): (optional) -1 for cpu, 0 for gpu
            mask_thresh (float): (optional) Threshold for masking the sensitivity maps, relative to max img amplitude
            verbose (int): (optional) 0: work silently; 1: show progressbar; 2: show progressbar and plots
            tol_fac (Float): (optional): tolerance to stop cg iterations

        Output:

        """
        print("Espirit Calibration")
        if coil_batchsize is None:
            coil_batchsize = self.numcoil

        if self.comp_mat is None:
            # if uncompressed, compress to rank 1
            self.compress_dictionary(1, comp_device=compute_device)

        # Load additional data
        img_oshape = (self.numslice, self.numcoil, imagesize, imagesize)

        # determine compute device
        device = sp.Device(compute_device)
        cpu = sp.Device(-1)
        xp = device.xp

        # preallocate data arrays
        comp_mat_gpu = sp.to_device(self.comp_mat[:, 0, None].T, device=device)
        coord_gpu = sp.to_device(self.coord, device=device)
        self.maps = np.zeros(img_oshape, dtype=self.ksp.dtype)

        # create linear operator
        linop = self.lr_sense_op(coord_gpu, comp_mat_gpu, imagesize, oversamp)

        # Calculate rank 1 image sequence per coil
        image = xp.zeros(img_oshape, dtype=np.complex)
        loops = itertools.product(range(math.ceil(img_oshape[0] / slice_batchsize)),
                                  range(math.ceil(img_oshape[1] / coil_batchsize)))
        for sl, cl in list(loops):
            # batch slices
            start = sl * slice_batchsize
            end = np.min([(sl + 1) * slice_batchsize, img_oshape[0]])
            slice_sl = slice(start, end)

            batch_linop = sp.linop.Diag([linop for _ in range(end - start)], iaxis=0, oaxis=0)

            # batch coils
            start = cl * coil_batchsize
            end = np.min([(cl + 1) * coil_batchsize, img_oshape[1]])
            coil_sl = slice(start, end)
            # Construct batched linop
            batch_linop = sp.linop.Diag([batch_linop for _ in range(end - start)], iaxis=1, oaxis=1)

            # load data 1 slice at a time to preserve memory
            ksp_gpu_slice = sp.to_device(self.ksp[slice_sl, coil_sl, ...], device=device)
            tol_sl = tol_fac * np.linalg.norm(self.ksp[slice_sl, coil_sl, ...])

            # Calculation 1 slice at a time because spmri.app.EspiritCalib can only handle 1 slice at the same time
            image[slice_sl, coil_sl, ...] = sp.app.LinearLeastSquares(batch_linop, ksp_gpu_slice,
                                                                      max_iter=max_iter,
                                                                      show_pbar=True, tol=tol_sl, ).run()

        # calculate sensitivity maps from coil images
        for sl in range(self.numslice):
            kspgrid_gpu = sp.fourier.fft(image[sl, ...], axes=(-1, -2))
            maps_gpu = spmri.app.EspiritCalib(kspgrid_gpu, device=device, show_pbar=True).run()

            # store all data in RAM
            self.maps[sl, ...] = sp.to_device(maps_gpu, device=cpu)

        # Apply mask
        if self.mask is None:
            self.mask = np.empty((self.numslice, imagesize, imagesize), dtype=bool)
            for sl in range(self.numslice):
                mult = sp.linop.Multiply((imagesize, imagesize), self.maps[sl, ...])
                sumimg = np.abs(mult.H(sp.to_device(image[sl, ...], device=cpu)))
                self.mask[sl, ...] = sumimg > (mask_thresh * np.max(sumimg))
                scipy.ndimage.binary_closing(self.mask[sl], iterations=3, output=self.mask[sl])

        self.maps *= self.mask[:, None, ...]

        if verbose == 2:
            pl.ImagePlot(np.abs(self.maps[0, ...]), z=0, title='Magnitude sensitivity maps estimated by ESPIRiT')
            pl.ImagePlot(np.angle(self.maps[0, ...]), z=0, title='phase Sensitivity Maps Estimated by ESPIRiT')
            pl.ImagePlot(np.abs(sp.to_device(image[0, ...], device=cpu)), z=0, title='Magnitude calibration image')
            pl.ImagePlot(np.angle(sp.to_device(image[0, ...], device=cpu)), z=0, title='phase Calibration image')
            plt.imshow(sp.to_device(self.mask[0, ...], device=cpu))
            plt.show()

        return

    def coil_compression(self, numvirtcoil):
        """ Apply PCA coil compression

        Perform PCA on sensitivity maps and compress sensitivity maps and k-space
        Updates ksp and maps in class

        Args:
            numvirtcoil (int): Number of virtual coils to simulate

        Output:
            None
        """

        if numvirtcoil is None:
            return

        # Calculate LR maps
        lr_maps_oshape = list(self.maps.shape)
        lr_maps_oshape[1] = numvirtcoil
        lr_maps = np.zeros(lr_maps_oshape, dtype=self.maps.dtype)

        lr_ksp_shape = list(self.ksp.shape)
        lr_ksp_shape[1] = numvirtcoil
        compress_ksp = np.zeros(lr_ksp_shape, dtype=self.maps.dtype)
        for i in range(self.maps.shape[0]):
            maps_reshape = self.maps[i, ...].reshape(self.maps.shape[1], -1).T
            u, s, vh = np.linalg.svd(maps_reshape, full_matrices=False)
            us = np.matmul(u, np.diag(s))
            lr_maps[i, ...] = us.T.reshape(self.maps.shape[1:])[0:numvirtcoil, ...]

            # compress kspace data
            vhr = vh[0:numvirtcoil, :].conj()
            kspdata_reshape = self.ksp[i, ...].reshape(self.ksp.shape[1], -1)
            compress_ksp[i, ...] = np.matmul(vhr, kspdata_reshape).reshape(lr_ksp_shape[1:])

        self.maps = lr_maps
        self.ksp = compress_ksp
        return

    def rotate2real(self):
        """ Rotate a complex vlaued images sequence to a real real valued image sequence.

         Takes a low rank image sequence and removes the phase of the first image to make the image sequence close to a
         real valued image sequence, then discard the imaginary part.
         Takes data from imgseq and updates imgseq to a real valued image sequence
         Data type is changed from complex to float.

         Whatch out! Overwrites imgseq

         Args:
            None

         Output:
            None
        """
        phasecorrection = np.angle(self.imgseq[:, :, 0]) - np.pi
        self.imgseq *= np.exp(1j * -phasecorrection)[:, :, np.newaxis]
        self.phasecorrection = phasecorrection

        # discard left over imaginary part
        self.imgseq = self.imgseq.real
        return

    def filt_back_proj(self, imagesize, compute_device: int = -1, oversamp=1.25, verbose=1):
        """ filtered back projection image reconstruction

         Calculate filtered back projection for each dynamic, updates the image sequence in class

         Args:
            imagesize (int): shape of the image
            compute_device (int): (optional) -1 for cpu, 0 for gpu
            oversamp (float): (optional) oversampling ratio (default 1.25)
            verbose (int): (optional) 0: work silently; 1: show progressbar; 2: show progressbar and plots

         Output:
            None
        """
        # sense recon every image, Single slice only
        device = sp.Device(compute_device)
        cpu = sp.Device(-1)

        coord_gpu = sp.to_device(self.coord, device=device)
        self.imgseq = np.zeros((self.numslice, self.numcoil, imagesize ** 2, self.numdyn), dtype=self.ksp.dtype)

        loops = itertools.product(range(self.numslice), range(self.numcoil), range(self.numdyn))
        for sl, cl, dyn in tqdm(list(loops), disable=verbose == 0):
            dcf = self.density_comp_fact(self.coord[dyn, ...])
            ksp_gpu = sp.to_device(self.ksp[sl, cl, dyn, ...] * dcf, device=device)
            img = sp.fourier.nufft_adjoint(ksp_gpu, coord_gpu[dyn, ...], oshape=(imagesize, imagesize),
                                           oversamp=oversamp)
            self.imgseq[sl, cl, :, dyn] = sp.to_device(self.to_vec(img), device=cpu)
        print('done')

    def senserecon(self, l2reg: float = 0, compute_device: int = -1,
                   verbose=1, tol_fac: float = 0):
        """ Sense image reconstruction

         Calculate iterative sense reconstruction for each dynamic, updates the image sequence in class

         Args:
            l2reg (float): (optional) regularization parameter for L2 regularization
            compute_device (int): (optional) -1 for cpu, 0 for gpu
            verbose (int): (optional) 0: work silently; 1: show progressbar; 2: show progressbar and plots
            tol_fac (float): (optional) tolerance factor as used in least squares solve.
         Output:
            None
        """
        # sense recon every image, Single slice only
        device = sp.Device(compute_device)
        cpu = sp.Device(-1)

        coord_gpu = sp.to_device(self.coord, device=device)
        maps_gpu = sp.to_device(self.maps, device=device)
        imgseq = sp.to_device(np.zeros((self.numslice, self.imagesize ** 2, self.numdyn), dtype=self.ksp.dtype),
                              device=device)

        loops = itertools.product(range(self.numslice), range(self.numdyn))
        for sl, dyn in tqdm(list(loops), desc="Sense reconstruction", disable=not verbose):
            if tol_fac is not None:
                tol_sl = tol_fac * np.linalg.norm(self.ksp[sl, :, dyn])
            ksp_gpu = sp.to_device(self.ksp[sl, :, dyn, ...], device=device)
            imgseq[sl, :, dyn] = self.to_vec(
                spmri.app.SenseRecon(ksp_gpu, maps_gpu[sl, ...],
                                     coord=coord_gpu[dyn, ...],
                                     lamda=l2reg, show_pbar=False,
                                     device=device, tol=tol_sl).run())
        self.imgseq = sp.to_device(imgseq, device=cpu)
        return

    def waveletrecon(self, lamda: float = 0, compute_device: int = -1, verbose=1, norm_ksp=True, tol=None,
                     tol_fac=1e-5):
        """ Wavelet image reconstruction

         Calculate iterative wavelet reconstruction for each dynamic, updates the image sequence in class

         Args:
            lamda (float): (optional) regularization parameter for wavelet regularization
            compute_device (int): (optional) -1 for cpu, 0 for gpu
            verbose (int): (optional) 0: work silently; 1: show progressbar; 2: show progressbar and plots
            tol_fac (float): (optional) tolerance factor as used in least squares solve.

         Output:
            None
        """
        # sense recon every image, Single slice only
        if norm_ksp:
            ksp_norm_fact = np.linalg.norm(self.ksp[:, 0, :, 0, 0], axis=1)
            self.ksp /= ksp_norm_fact[:, None, None, None, None]

        device = sp.Device(compute_device)
        cpu = sp.Device(-1)

        coord_gpu = sp.to_device(self.coord, device=device)
        maps_gpu = sp.to_device(self.maps, device=device)
        imgseq = sp.to_device(np.zeros((self.numslice, self.imagesize ** 2, self.numdyn), dtype=self.ksp.dtype),
                              device=device)

        loops = itertools.product(range(self.numslice), range(self.numdyn))
        for sl, dyn in tqdm(list(loops), desc="Wavelet reconstruction", disable=not verbose):
            if tol is None:  # Set tol scaled with norm of the input data
                tol_sl = tol_fac * np.linalg.norm(self.ksp[sl, :, dyn])
            else:
                tol_sl = tol
            ksp_gpu = sp.to_device(self.ksp[sl, :, dyn, ...], device=device)
            imgseq[sl, :, dyn] = self.to_vec(
                spmri.app.L1WaveletRecon(ksp_gpu, maps_gpu[sl, ...], coord=coord_gpu[dyn, ...],
                                         lamda=lamda, show_pbar=False, device=device, tol=tol_sl).run())
        self.imgseq = sp.to_device(imgseq, device=cpu)

        if norm_ksp:
            self.ksp *= ksp_norm_fact[:, None, None, None, None]
            self.imgseq *= ksp_norm_fact[:, None, None]

        return

    def calc_PDHG_sigma(self, lamda=.01, sl=None, device=-1, maps=None):

        sigmas = []
        if sl is None:
            sls = range(self.numslice)
        elif isinstance(sl, list):
            sls = sl
        elif isinstance(sl, int):
            sls = [sl]
        else:
            raise IOError(f'sl {sl} unknown')

        if maps is None:
            maps = self.maps

        for sl_ in sls:
            if sp.get_device(maps) != device and maps.ndim == 4:
                if kmaps.shape[0] == 1:
                    kmaps = sp.to_device(maps[0], device=device)
                else:
                    kmaps = sp.to_device(maps[sl_], device=device)
            elif sp.get_device(maps) != device:
                kmaps = sp.to_device(maps, device=device)
            elif maps.ndim == 4:
                if maps.shape[0] == 1:
                    kmaps = maps[0]
                else:
                    kmaps = maps[sl_]
            else:
                kmaps = maps
            sig = sp.mri.kspace_precond(kmaps, coord=sp.to_device(self.coord, device), lamda=lamda, device=device)
            if sp.get_device(sig) != sp.get_device(maps):
                sig = sp.to_device(sig, sp.get_device(maps))
            sigmas.append(sig)
        if isinstance(sl, int):
            sigmas = sigmas[0]
        return sigmas

    def lr_inversion(self, batchsize=None, oversamp=1.25, warmstart=False,
                     lam: float = 0, bias: np.ndarray = None,
                     max_iter: int = 150, compute_device=-1,
                     verbose=1, lstsq_solver=None, sigmas=None, tol=None,
                     tol_fac=0.001, reg_kwargs=None, norm_ksp=True, **kwargs):
        """" Low rank inversion image reconstruction

        Use the LR sense linop constructed in self.lr_sense_op() to reconstruct the signal with a linear least squares
        solver for Low rank inversion of the signal to the Low rank image sequence.
        updates the imgseq attribute in class

        Args:
            batchsize (int): (optional) number of dynamics calculated simultaneously defaults to mridata.numdyn
            oversamp (float): (optional) oversampling ratio (default 1.25)
            warmstart (bool): (optional) use previous solution as a warm start, only works if previous solution exists.
            lam (float): (optional) L2 regularization parameter
            bias(array): (optional) Bias for the L2 regularization
            max_iter(int): (optional) Amount of conjugate gradient iterations, default 150
            compute_device (int): (optional) -1 for cpu, 0 for gpu
            verbose (int): 0: (optional) work silently; 1: show progressbar; 2: show progressbar and plots
            tol: (optional) Stopping tolerance for LR inversion iterations
            tol_fac: (optional, float): Scaling factor which is used for calculation of criterium for LR inversion
            reg_kwargs (optional): Kwargs passed to get_proxg for (wavelet) regularization
            **kwargs are pushed to linearleastsquares app
        Output:
            (array): image sequence of shape(imgseq[0] * imgseq[1], rank)
        """
        print("imgseq by Low rank Inversion")
        if batchsize is None:
            batchsize = self.numdyn
        if norm_ksp:
            ksp_norm_fact = np.linalg.norm(self.ksp[:, 0, :, 0, 0], axis=1)
            self.ksp /= ksp_norm_fact[:, None, None, None, None]
        # Determine compute device and batch size
        device = sp.Device(compute_device)
        cpu = sp.Device(-1)

        # Move static data to GPU
        coord_array_gpu = sp.to_device(self.coord, device=device)
        ur_gpu = sp.to_device(self.comp_mat.conj().T, device=device)

        # preallocate output array
        if self.imgseq is None or warmstart is False:
            self.imgseq = np.zeros((self.numslice, self.imagesize ** 2, self.rank), dtype=self.ksp.dtype)
        if reg_kwargs is not None:
            regz = self.get_proxg(imshape=(1, self.rank, self.imagesize, self.imagesize), **reg_kwargs,
                                  comp_device=device)
        else:
            regz = {}
        # Loop over slices
        slice_batchsize = 1
        for sl in range(math.ceil(self.numslice / slice_batchsize)):
            # batch slices
            start = sl * slice_batchsize
            end = np.min([(sl + 1) * slice_batchsize, self.numslice])
            slice_sl = slice(start, end)

            if tol is None:  # Set tol scaled with norm of the input data
                tol_sl = tol_fac * np.linalg.norm(self.ksp[slice_sl])
            else:
                tol_sl = tol

            # load slice specific data in VRAM
            kspacedata_gpu = sp.to_device(self.ksp[slice_sl, ...], device=device)
            if self.maps is not None:
                maps_gpu = sp.to_device(self.maps[slice_sl, ...], device=device)
            total_op = self.lr_sense_op(coord_array_gpu, ur_gpu, self.imagesize, oversamp, maps_gpu, batchsize)

            if bias is None:
                gpu_bias = sp.to_device(np.zeros(total_op.ishape), device=device)
            else:
                gpu_bias = sp.to_device(bias[slice_sl, ...].T.reshape(total_op.ishape), device=device)
            x_init_gpu = sp.to_device(self.imgseq[slice_sl, ...].T.reshape(total_op.ishape), device=device)

            # For PDHG define or get sigma and tau values, as preconditioner and saved other calculations respectivily.
            if lstsq_solver == 'PrimalDualHybridGradient' and sigmas is None:
                sigma = self.calc_PDHG_sigma(sl=sl, maps=maps_gpu, device=device)
            elif lstsq_solver == 'PrimalDualHybridGradient' and sigmas is not None:
                sigma = sigmas[sl]
            else:
                sigma = None
            try:
                tau = self.taus[sl]
            except AttributeError:
                if lstsq_solver == 'PrimalDualHybridGradient':
                    print('(re)Calculate tau for PDHG')
                tau = None
            # Load in VRAM
            if lstsq_solver == 'PrimalDualHybridGradient' and sigma is not None and sp.get_device(sigma) != device:
                sigma = sp.to_device(sigma, device=device)
            if lstsq_solver == 'PrimalDualHybridGradient' and tau is not None and sp.get_device(tau) != device:
                tau = sp.to_device(tau, device=device)

            # Solve linear least squared
            lls_app = sp.app.LinearLeastSquares(total_op, kspacedata_gpu, x=x_init_gpu,
                                                lamda=lam, z=gpu_bias,
                                                max_iter=max_iter, save_objective_values=False,
                                                show_pbar=verbose > 0,
                                                solver=lstsq_solver, sigma=sigma, tol=tol_sl, tau=tau, **regz,
                                                **kwargs)

            sliceresult = lls_app.run()
            if lstsq_solver == 'PrimalDualHybridGradient' and hasattr(self, 'taus'):  # Save tau
                self.taus[sl] = sp.to_device(lls_app.tau, device=cpu)
            # objval = lls_app.objective_values

            self.imgseq[slice_sl, ...] = sp.to_device(sliceresult, device=cpu).reshape(self.rank, -1).T
        if norm_ksp:
            self.ksp *= ksp_norm_fact[:, None, None, None, None]
            self.imgseq *= ksp_norm_fact[:, None, None]
        return

    def nnls(self, regparam: float = None, weights: np.ndarray = None,
             overwrite_imgseq: np.ndarray = None, mask: np.ndarray = None, n_jobs: int = None,
             verbose: bool = False,
             norm_correction: bool = False):
        """Solve Non-Negative Least Squares components from image sequence

        Calculate the non negative components from the image sequence.
        If regparam and weights are supplied the nnls is solved with the reweighted norm
        By default the image sequence stored in class attribute imgseq is used, unless overwrite_imgseq

        Args:
            regparam (float): (optional) joint sparicty regularization parameter
            weights (np.ndarray): (optional) weights for reweighing the dictionary
            overwrite_imgseq (np.ndarray): (optional) Image sequence to use in stead of self.imgseq
            mask (np.ndarray): (optional) mask as used to mask the image sequence (True are kept)
            n_jobs (boolean): (optional) number of jobs created by joblibs, not always an improvement
            norm_correction (boolean): (optional) correct for normalisation of the dictionary

        Output:
            None
        """
        mod_imgseq = self.imgseq.copy()
        if overwrite_imgseq is not None:
            mod_imgseq = overwrite_imgseq
        self.fixed_par_processing(redo=True, flatten=True, mask=False)
        fixed_par_dict_masks = self.fixed_par_dict_masks
        fixed_par_img_masks = self.fixed_par_img_masks

        # modify dictionary
        dictmat = self.compress_dict.copy()

        if mask is None:  # Really only mask zeros
            mask = np.abs(mod_imgseq[:, :, 0]) > 0  # essentially follows mask from espirit calibration

        # reshape into vector form
        mod_imgseq_res = mod_imgseq.reshape(-1, mod_imgseq.shape[-1])
        mask_res = mask.reshape(-1)
        fixed_par_img_masks = [m.flatten()[mask_res] for m in fixed_par_img_masks]
        # preallocate array and calc mask
        nnls_solve = np.zeros([np.prod(list(mod_imgseq.shape)[:-1]),
                               self.num_dict_atoms])

        if weights is None and regparam is None:
            nnls_solve[mask_res, :] = SPIJN2.lsqnonneg2(
                mod_imgseq_res[mask_res, :].T, dictmat,
                out_z=None,
                fixed_par_img_masks=fixed_par_img_masks,
                fixed_par_dict_masks=fixed_par_dict_masks,
                n_jobs=n_jobs, S=self._S, verbose=verbose > 0
            )[0].T
        else:
            nnls_solve[mask_res, :] = SPIJN2.rewlsqnonneg2(
                mod_imgseq_res[mask_res, :].T, dictmat,
                weights, out_z=None,
                fixed_par_img_masks=fixed_par_img_masks,
                fixed_par_dict_masks=fixed_par_dict_masks,
                L=regparam, n_jobs=n_jobs, S=self._S,
                verbose=verbose > 0
            )[0].T
        if self.norms is not None and norm_correction:
            nnls_solve = dso.multiply(nnls_solve, 1 / self.norms, fixed_par_dict_masks=fixed_par_dict_masks,
                                      fixed_par_img_masks=fixed_par_img_masks)
        self.comp = nnls_solve.reshape(list(mod_imgseq.shape)[:-1] + [self.num_dict_atoms])
        return

    def lr_admm(self, admm_param: float, batchsize: int = None, oversamp: float = 1.25,
                max_iter: int = 10, max_cg_iter: int = 20, compute_device=-1,
                outpath=None, regparam=None, weights=None,
                verbose=1, lstsq_solver=None, tol=None, tol_fac=0.001,
                norm_ksp=True, n_jobs=None, norm_correction=False, **kwargs):
        """" Low rank MC-ADMM image and component reconstruction

            Use the ADMM method to solve the constrained the LR inversion image reconstruction
            Updates image sequence attribute in class to the real valued image sequence as calculated.
            Updates components attribute in class

            Args:
                admm_param (float): ADMM coupling parameter
                batchsize (int): (optional) number of dynamics calculated simultaneously defaults to mridata.numdyn
                oversamp (float): (optional) oversampling ratio (default 1.25)
                max_iter (int): (optional) Amount of admm iterations, default 10
                max_cg_iter(int): (optional) Amount of conjugate gradient iterations, default 20
                compute_device (int): (optional) -1 for cpu, 0 for gpu
                outpath (string): (optional) path to file location to store intermediate results
                regparam (float): (optional) joint sparsity regularization parameter
                weights (np.ndarray): (optional) weights for reweighing the dictionary
                verbose (int): 0: (optional) work silently; 1: show progressbar; 2: show progressbar and plots
                lstsq_solver (str) : (optional) Least squares solver to use, default CG
                                                PrimalDualHybridGradient tries to
                                                use a preconditioner
                tol: (optional) Stopping tolerance for LR inversion iterations
                tol_fac: (optional, float): Scaling factor which is used for calculation of criterion for LR inversion


            Output:
                (array): image sequence of shape(imgseq[0] * imgseq[1], rank)
            """
        # Initialize ADMM variables
        converged = False
        it = 0
        imgseq_oshape = (self.numslice, self.imagesize ** 2, self.rank)
        lag_mult = np.zeros(imgseq_oshape, dtype=np.float64)
        residuals = np.zeros((4, max_iter), dtype=np.float64)
        # obj_val_arr = np.zeros((max_iter, max_cg_iter + 1))
        local_admm_param = admm_param

        if norm_ksp:
            ksp_norm_fact = np.linalg.norm(self.ksp[:, 0, :, 0, 0], axis=1)
            self.ksp /= ksp_norm_fact[:, None, None, None, None]

        if lstsq_solver == 'PrimalDualHybridGradient':
            sigmas = self.calc_PDHG_sigma(device=compute_device)
            self.taus = [None] * self.numslice
        else:
            sigmas = None

        comp0 = None

        while it < max_iter and not converged:
            print('Outer loop iteration {} of {}'.format(it + 1, max_iter))

            # %% Step 1: Solve for phase rotated image sequence
            if it == 0:
                self.lr_inversion(batchsize=batchsize, oversamp=oversamp,
                                  compute_device=compute_device,
                                  max_iter=max_cg_iter,
                                  lstsq_solver=lstsq_solver, sigmas=sigmas, tol=tol, tol_fac=tol_fac, norm_ksp=False,
                                  **kwargs)
            else:
                l2bias = (dc - lag_mult) * np.exp(1j * self.phasecorrection)[:, :, np.newaxis]  # DC-v
                self.imgseq = self.imgseq.astype(np.complex) * np.exp(1j * self.phasecorrection)[:, :, np.newaxis]
                self.lr_inversion(batchsize=batchsize, oversamp=oversamp, warmstart=True, lam=local_admm_param,
                                  bias=l2bias, max_iter=max_cg_iter, compute_device=compute_device,
                                  lstsq_solver=lstsq_solver, sigmas=sigmas, tol=tol, tol_fac=tol_fac, norm_ksp=False,
                                  **kwargs)

            # rotate image sequence to real axis
            self.rotate2real()

            # %% Step 2: Solve for components without joint sparsity
            mod_imgseq = self.imgseq + lag_mult
            if verbose:
                print('NNLS solve')
            self.nnls(regparam=regparam, weights=weights,
                      overwrite_imgseq=mod_imgseq, n_jobs=n_jobs, verbose=verbose)  # Don't supply a mask here,
            # that leads to zeros in unwanted places, leading to exploding errors in other places.
            if comp0 is not None:
                rel = np.linalg.norm(self.comp.flatten() - comp0.flatten()) / np.linalg.norm(self.comp.flatten())
                print(f'Iteration: {it}, nnls convergence:{rel}')
                converged = rel < tol_fac / 10
            comp0 = self.comp.astype(np.float32).copy()
            # %% Step 3: Update lagrange multiplier
            dc = dso.vecmat(self.comp, self.compress_dict.T,
                            fixed_par_img_masks=[m.reshape(imgseq_oshape[:-1]) for m in self.fixed_par_img_masks],
                            fixed_par_dict_masks=self.fixed_par_dict_masks,
                            S=self._S).reshape(self.imgseq.shape)
            # dc = (self.comp @ self.compress_dict.T).reshape(self.imgseq.shape)
            lag_mult += self.imgseq - dc  # v+x-dc
            if verbose > 1:
                for i in range(lag_mult.shape[2]):
                    plt.subplot(1, 3, 1)
                    plt.imshow(self.imgseq[0, :, i].reshape(224, 224))
                    plt.colorbar()
                    plt.subplot(1, 3, 2)
                    plt.imshow((self.imgseq[0, :, i] - dc[0, :, i]).reshape(224, 224))
                    plt.colorbar()
                    plt.subplot(1, 3, 3)
                    plt.imshow(dc[0, :, i].reshape(224, 224))
                    plt.colorbar()
                    plt.show()

            # %% end of ADMM loop, the rest is for calculating residuals
            # Forward linear operator

            if verbose:
                list_linop = []
                for i in range(self.numslice):
                    singleslice_linop = self.lr_sense_op(self.coord, self.comp_mat.T, imagesize=self.imagesize,
                                                         oversamp=oversamp, maps=self.maps[i, None, ...],
                                                         batchsize=batchsize)
                    list_linop.append(singleslice_linop)
                linop = sp.linop.Diag(list_linop, 0, 0)
                # norm_y = np.linalg.norm(ksp)

                # re-apply phase offset correction to dc
                dc_phasecorr = dc * np.exp(1j * self.phasecorrection)[:, :, np.newaxis]

                # |GFSDc-y|
                dc_img = np.transpose(dc_phasecorr, (0, 2, 1)).reshape((self.numslice, self.rank, self.imagesize,
                                                                        self.imagesize)).astype(np.complex128)
                obj_val1 = np.linalg.norm(linop(dc_img) - self.ksp)

                print('Objective function value: |GFSDc-y| = {}'.format(obj_val1))
                residuals[0, it] = obj_val1

                imgseq = self.imgseq.astype(np.complex) * np.exp(1j * self.phasecorrection)[:, :, np.newaxis]

                # |GFSx-y|
                lr_imgseq_reshape = np.transpose(imgseq, (0, 2, 1)).reshape((self.numslice, self.rank,
                                                                             self.imagesize, self.imagesize)).astype(
                    np.complex128)
                obj_val2 = np.linalg.norm(linop(lr_imgseq_reshape) - self.ksp)
                print('Obj_val, |GFSx-y| = {}'.format(obj_val2))
                residuals[1, it] = obj_val2

                # |x-Dc|
                obj_val3 = np.linalg.norm(imgseq - dc)
                print('Objective function value: |x-Dc| = {}'.format(obj_val3))
                residuals[2, it] = obj_val3

                # |GFSx-y| + mu1|c| + mu2|x-Dc-v|
                residuals[3, it] = np.linalg.norm(linop(lr_imgseq_reshape) - self.ksp) ** 2 + \
                                   local_admm_param * np.linalg.norm(imgseq - dc + lag_mult) ** 2 + \
                                   local_admm_param * np.linalg.norm(lag_mult) ** 2
                print('Objective function value: |GFSx-y| + mu2|x-Dc+v| = {}'.format(residuals[3, it]))

                # convergence rate
                if it > 0:
                    num = np.linalg.norm(imgseq_old - self.imgseq)
                    rate = num / np.linalg.norm(self.imgseq)
                    print('Convergence rate = {}'.format(rate))
                imgseq_old = self.imgseq

            if outpath is not None:
                self.residuals = residuals
                self.to_h5(outpath, 'admm{}.h5'.format(it))
                delattr(self, 'residuals')

            if verbose == 2:
                imagesequence = np.transpose(self.imgseq, (0, 2, 1)).reshape((self.numslice, self.rank,
                                                                              self.imagesize, self.imagesize))
                for i in range(imagesequence.shape[1]):
                    pl.ImagePlot(imagesequence[:, i, ...], z=0, title='Image sequence, rank {}'.format(i + 1))
            it += 1
        if lstsq_solver == 'PrimalDualHybridGradient':
            del self.taus
        if norm_ksp:
            self.ksp *= ksp_norm_fact[:, None, None, None, None]
            self.imgseq *= ksp_norm_fact[:, None, None]
            self.comp *= ksp_norm_fact[:, None, None]
        if norm_correction and self.norms is not None:
            self.comp = dso.multiply(self.comp, 1 / self.norms,
                                     fixed_par_img_masks=[m.reshape(imgseq_oshape[:-1]) for m in
                                                          self.fixed_par_img_masks],
                                     fixed_par_dict_masks=self.fixed_par_dict_masks,
                                     S=self._S)
        return residuals[3, -1]

    def single_component_match(self, stepsize: int = 1000, verbose: int = 1,
                               absdict=False, calc_nrmse=True):
        """ Single component matching

        Single component matching by way of largest inner product, the largest inner product of the signal with all the
        dictionary atoms is assigned to the pixel in the final image.
        Component are vectorirzed and stored in the comp attribute in order (PD, T1, T2, NRMSE)

        Args:
            stepsize (int): (optional) stepsize states amount of pixels calculated simultaneously
            verbose (int): (optional) 0: work silently; 1: show progressbar; 2: show progressbar and plots
            absdict (bool): (optional) absolute value of dictionary, must use when image sequence is calculated with rss
        Output:

        """
        print("Single component matching")
        if stepsize is None:
            stepsize = self.imagesize
        self.fixed_par_processing(flatten=True)
        imgseq = self.imgseq.reshape((-1, self.rank))
        # normalize dictionary
        x_norm = np.linalg.norm(self.compress_dict, axis=0, keepdims=True)
        dictmatnorm = self.compress_dict / x_norm
        if absdict:
            dictmatnorm = np.abs(dictmatnorm)

        # preallocate solution maps
        pdimg = np.zeros(imgseq.shape[0]) + np.nan
        t1img = np.zeros(imgseq.shape[0]) + np.nan
        t2img = np.zeros(imgseq.shape[0]) + np.nan
        nrmseimg = np.zeros(imgseq.shape[0]) + np.nan
        # %%
        # Create result matrices
        indices = np.zeros(imgseq.shape[0], dtype=np.int32)
        max_vals = np.zeros(imgseq.shape[0], dtype=np.float64)
        phaseimg = phase_vals = np.zeros(imgseq.shape[0], dtype=np.float64)

        for dictionary_mask, measurement_mask in tqdm(zip(self.fixed_par_dict_masks, self.fixed_par_img_masks),
                                                      disable=verbose == 0 or not self.fixed_b1, desc='Matching'):
            if measurement_mask.sum() == 0:
                continue
            dd = dictmatnorm[:, dictionary_mask]
            fixed_ind = np.arange(len(dictmatnorm.T))[dictionary_mask]
            n_meas = np.sum(measurement_mask)
            n_chunks = int(np.ceil(n_meas / float(stepsize)))
            # loop over chunks
            for chunk in tqdm(np.array_split(np.arange(n_meas), n_chunks), desc='Matching',
                              disable=verbose == 0 or self.fixed_b1):
                mask_fl = np.zeros(n_meas, dtype=bool)
                mask_fl[chunk] = True

                measurement_mask_chunk = measurement_mask.copy()
                measurement_mask_chunk[measurement_mask_chunk] = mask_fl

                mask = measurement_mask_chunk

                maxes = np.tensordot(dd, imgseq[mask, :], axes=(0, 1))
                ind_match = np.argmax(np.abs(maxes), axis=0)
                indices[mask] = ind_dict = fixed_ind[ind_match]

                mv = np.take_along_axis(maxes, np.expand_dims(ind_match, axis=0), axis=0)
                mv /= x_norm[:, ind_dict]
                max_vals[mask] = mv.flatten()

                phases = np.angle(maxes)
                phase_vals[mask] = np.take_along_axis(phases, np.expand_dims(ind_match, axis=0), axis=0).flatten()
            if calc_nrmse:
                for k in chunk:
                    # Calculate NRMSE
                    nrmseimg[k] = np.sqrt(
                        np.sum((np.abs(dictmatnorm[:, indices[k]] * pdimg[k] - imgseq[k, :])) ** 2) /
                        dictmatnorm.shape[0])
        # return indices, max_vals, phase_vals
        t1img = self.dictt1[indices]
        t2img = self.dictt2[indices]
        pdimg = max_vals
        # %%

        # mask values
        # t1img[pdimg < 0.3] = 0
        # t2img[pdimg < 0.3] = 0

        if verbose > 1:
            # plot results
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.imshow(self.to_image(t1img), origin="lower")
            plt.title("T1 image")
            plt.colorbar()
            plt.subplot(2, 2, 2)
            plt.imshow(self.to_image(t2img), origin="lower")
            plt.title("T2 image")
            plt.colorbar()
            plt.subplot(2, 2, 3)
            plt.imshow(self.to_image(pdimg), origin="lower")
            plt.title("PD image")
            plt.colorbar()
            plt.subplot(2, 2, 4)
            plt.imshow(self.to_image(np.divide(nrmseimg, pdimg, out=np.zeros_like(nrmseimg), where=pdimg != 0)),
                       origin="lower")
            plt.title("RMSE image")
            plt.colorbar()
            plt.show()
            print("max T1 value")
            print(np.max(t1img))
            print("max T2 value")
            print(np.max(t2img))

        oshape = list(self.imgseq.shape)[:-1] + [5]
        self.single_comp_names = np.asarray(['M0', 'T1', 'T2', 'NRMSE', 'Phase'], dtype='S')
        self.single_comp = np.zeros(oshape, dtype=t1img.dtype)
        self.single_comp[..., 0] = np.abs(pdimg.reshape(oshape[:-1]))
        self.single_comp[..., 1] = t1img.reshape(oshape[:-1])
        self.single_comp[..., 2] = t2img.reshape(oshape[:-1])
        self.single_comp[..., 3] = nrmseimg.reshape(oshape[:-1])
        self.single_comp[..., 4] = np.angle(pdimg.reshape(oshape[:-1]))
        return

    def save_single_comp_nii(self, output_path, aff=None, ):
        if aff is None:
            aff = np.zeros((4, 4))
            aff[0, 1] = 1
            aff[1, 2] = -1
            aff[2, 0] = 1
            aff[3, 3] = 1
        for i, name in enumerate(self.single_comp_names):
            data = self.single_comp[..., i]
            v = np.asarray([self.to_image(ii) for ii in data])
            v[(1 - self.mask).astype(bool)] = np.nan
            img = nib.Nifti1Image(v, aff)
            if isinstance(name, bytes):
                name = name.decode()
            nib.save(img, os.path.join(output_path, f'single_comp_{name}.nii.gz'))
        return

    def plot_comp(self, normalize=False):
        figs = []
        for j in range(self.numslice):
            for i in range(len(self.index)):
                figs.append(plt.figure())
                if normalize:
                    plt.imshow(self.to_image(self.comp[j, :, i] / self.comp[j].sum(axis=-1)))
                else:
                    plt.imshow(self.to_image(self.comp[j, :, i]))
                plt.colorbar()
                if self.index[0] >= 0:
                    plt.title('$T_1={:.1f}, T_2={:.1f}$'.format(self.dictt1[self.index[i]],
                                                                self.dictt2[self.index[i]]))
                plt.show()
        return figs

    def spijn_solve(self, regparam, max_iter=20, verbose=1,
                    n_jobs=None, norm_correction=False):
        """ Solve components using SPIJN algorithm

        Use the SPIJN algorithm to solve for the multi-component component maps from the image sequence
        updates component maps in comp attribute and stores the component indices in index attribute

        Args:
            regparam (float): joint sparsity regularization parameter
            max_iter (int): (optional) maximum number of SPIJN iterations
            verbose (int): (optional) 0: work silently; 1: show info in terminal; 2: show info in terminal and plots

        Output:

        """
        self.fixed_par_processing(True, True, False)
        fixed_par_dict_masks = self.fixed_par_dict_masks
        fixed_par_img_masks = self.fixed_par_img_masks
        lr_imgseq = self.imgseq
        dictmat = self.compress_dict

        # Transpose so input shape matches other functions in  this file
        lr_imgseq_flat = lr_imgseq.reshape(-1, self.rank).T

        # normalize dictionary
        x_norm = np.linalg.norm(dictmat, axis=0, keepdims=True)
        dictmat = dictmat / x_norm

        # Calculate mask
        # avgimg = np.abs(np.sum(lr_imgseq, axis=0) / lr_imgseq.shape[0])
        # mask = avgimg > np.sum(avgimg) / (2*lr_imgseq.shape[1])

        mask = np.abs(lr_imgseq_flat[0, :]) > 0.1
        try:
            mask_im = self.spijn_mask.flatten()  # np.abs(lr_imgseq_flat[0, :]) > 0.1
            print('SPIJN masking difference:', np.sum(mask) - np.sum(mask_im))
        except:
            mask_im = mask
        fixed_par_img_masks = [m[mask_im] for m in fixed_par_img_masks]

        if verbose == 2:
            maskimg = mask.reshape(self.numslice, self.imagesize, self.imagesize)
            pl.ImagePlot(maskimg, z=0, title='SPIJN mask')

        if regparam != 0:
            print("Start SPIJN solve")
            cr, sfull, rel_err, c1, _ = SPIJN(
                lr_imgseq_flat[:, mask_im], dictmat, L=regparam,
                correct_im_size=True,
                max_iter=max_iter, tol=5e-4,
                fixed_par_img_masks=fixed_par_img_masks,
                fixed_par_dict_masks=fixed_par_dict_masks, norms=x_norm[0])

            # fill data to image format to forget about a mask
            cr_long = np.zeros((lr_imgseq_flat.shape[1], cr.shape[1]), dtype=cr.dtype)
            sfull_long = np.zeros((lr_imgseq_flat.shape[1], cr.shape[1]), dtype=cr.dtype)
            cr_long[mask_im, :] = cr
            sfull_long[mask_im, :] = sfull

            # process data
            self._S = comp_indices = np.unique(sfull[cr > .01])  # Determine the used indices
        else:  # no restrictions on components
            print("Directly start NNLS solve")
            comp_indices = range(self.num_dict_atoms)

        print(comp_indices)
        num_comp = len(comp_indices)

        # cut down dictionary
        fixed_par_img_masks = [m[mask] for m in self.fixed_par_img_masks]

        # solve for least squares solution
        nnls_solve, ore = SPIJN2.lsqnonneg2(
            lr_imgseq_flat[:, mask], dictmat,
            S=comp_indices, fixed_par_img_masks=fixed_par_img_masks,
            fixed_par_dict_masks=fixed_par_dict_masks, n_jobs=n_jobs,
            verbose=verbose > 0
        )
        nnls_solve = dso.multiply(nnls_solve.T, 1 / x_norm.flatten(),
                                  fixed_par_dict_masks=fixed_par_dict_masks,
                                  fixed_par_img_masks=fixed_par_img_masks,
                                  S=comp_indices)
        if norm_correction and self.norms is not None:
            nnls_solve = dso.multiply(nnls_solve, 1 / self.norms,
                                      fixed_par_dict_masks=fixed_par_dict_masks,
                                      fixed_par_img_masks=fixed_par_img_masks,
                                      S=comp_indices)
        # nnls_solve = np.apply_along_axis(lambda x: optimize.nnls(dictmat, x)[0], 0, lr_imgseq_flat[:, mask])
        components = np.zeros((lr_imgseq_flat.shape[1], num_comp), dtype=nnls_solve.dtype)
        components[mask, :] = nnls_solve

        self.comp = components.reshape((self.numslice, lr_imgseq.shape[1], num_comp))

        self.index = np.arange(self.compress_dict.shape[1])[self.fixed_par_dict_masks[0]][comp_indices]
        # Plot solutions
        if verbose == 2:
            self.plot_comp()
        return

    def spijn_from_ksp(self, admm_param, oversamp=1.25, max_admm_iter=10, max_cg_iter=20, p=0, max_iter=20,
                       verbose=True, norm_ksp=True, tol=1e-4, reg_param=0.0, correct_im_size=True, prun=2, init_rank=10,
                       compute_device=-1, min_iter=3, lstsq_solver='PrimalDualHybridGradient', tol_fac=0.001,
                       tol_admm=None, norm_correction=True, **kwargs):
        """Direct reconstruction of joint sparse components from kspace data.

        Args:
            admm_param (float): ADMM coupling parameter
            oversamp (float): (optional) oversampling ratio (default 1.25)
            max_admm_iter (int): (optional) Amount of admm iterations, default 10
            max_cg_iter(int): (optional) Amount of conjugate gradient iterations, default 20
            p (float): (optional) value as used in the reweighting scheme 1. Default 0 works fine
            max_iter (int): (optional) maximum number of iterations, 20 is normally fine
            verbose (int): (optional) more output
            norm_ksp (bool): (optional) Normalise the signal or not. False is fine
            tol (float): (optional) Tolerance used, when are the iterations stopped, based on
                ||C^k-C^{k+1}||_F^2/||C^k||_F^2<tol
            reg_param (float): (optional) the regularisation parameter used.
            correct_im_size (bool): (optional) Adjust regularization parameter for number of slices (experimental feature)
            prun (int): (optional) The number of iterations afterwards the pruning of unused atoms in the dictionary
                takes place
            init_rank (int): (optional) Initial reconstruction rank for image reconstruction, only used dictionary has
                not been compressed.
            compute_device (int): (optional) -1 for cpu, 0 for gpu
            min_iter (int): (optional) force the solver to do a certain number of iterations.
            **kwargs are forwarded to lr_admm
        Output:
            (float): residual of ||GFSDc-y||^2_2

            self.comp and self.imgseq is updated afterwards

        """
        signal.signal(signal.SIGINT, signal.default_int_handler)  # To stop during the iterations
        eps = 1e-4
        rel_err_old = 0
        comp_old = None
        if self.fixed_par_dict_masks is not None:
            self.index = np.arange(len(self.dictt1))[self.fixed_par_dict_masks[0]]
        else:
            self.index = np.arange(self.num_dict_atoms)
        self._S = np.arange(self.num_dict_atoms)

        if norm_ksp:
            ksp_norm_fact = np.linalg.norm(self.ksp[:, 0, :, 0, 0], axis=1)
            self.ksp /= ksp_norm_fact[:, None, None, None, None]

        if correct_im_size:  # Correct regularisation for number of voxels
            reg_param *= self.numslice
            # reg_param *= np.log10(self.imagesize*self.imagesize*self.numslice)

        try:  # SPIJN mask
            mask_spijn = self.spijn_mask.flatten()  # np.abs(lr_imgseq_flat[0, :]) > 0.1
        except:
            mask_spijn = ...

        if self.comp_mat is None:
            self.compress_dictionary(init_rank, comp_device=compute_device)

        t0 = time.clock()
        admm_settings = dict(admm_param=admm_param, oversamp=oversamp, max_iter=max_admm_iter,
                             max_cg_iter=max_cg_iter, compute_device=compute_device,
                             verbose=verbose, lstsq_solver=lstsq_solver, tol=tol_admm,
                             tol_fac=tol_fac, norm_ksp=False, )
        if self.comp is None:  # First iteration
            rel_err = rel_err_old = self.lr_admm(
                **admm_settings, **kwargs
            )  # Perform real calculations
            print('matching time it 1: {0:.5f}s'.format(time.clock() - t0))
            print('Relative output error = {}'.format(rel_err_old))
            prunn_comp = None
        else:
            print('Reused old first iteration solution')

        try:  # Try-except is to stop iterations when it takes to long, making it possible to return the latest result
            for k in range(1, max_iter):
                # Calc weights
                sel_comp = self.comp.reshape(-1, self.num_dict_atoms)[mask_spijn]
                w = (np.linalg.norm(sel_comp, 2, axis=0) + eps) ** (1 - p / 2)
                w[w < eps] = eps  # prevent 0-weighting
                if k >= prun:
                    # determine components to prune
                    prunn_comp = np.sum(sel_comp, axis=0) > 1e-12

                    # prune dictionary
                    # self.dictmat = self.dictmat[:, prunn_comp]
                    w = w[prunn_comp]  # Weights are saved in a pruned version
                    self.index = self.index[prunn_comp]  # This takes care of further pruning
                    self._S = self._S[prunn_comp]
                    print('index:', self.index)
                    print('S:', self._S)

                # compress
                self.compress_dictionary(min(self.rank, len(w)), comp_device=compute_device)

                comp_old = self.comp.copy()
                t0 = time.clock()
                print('regparam = {}'.format(reg_param))
                rel_err = self.lr_admm(**admm_settings,
                                       regparam=reg_param,
                                       weights=w,
                                       **kwargs)  # Perform real calculations
                if verbose:
                    print('matching time: {0:.5f}s'.format(time.clock() - t0))

                # Calc residuals
                # Determine relative convergence
                rel_conv = np.abs(rel_err_old - rel_err) / rel_err

                if verbose:
                    print('k: {} relative difference between iterations {},elements: {}'.format(k, rel_conv, np.sum(
                        np.sum(self.comp, 1) > 1e-4)))
                    print('Relative output error = {}'.format(rel_err))

                if verbose:
                    print("Number of components left: {}".format(self.num_dict_atoms))
                    if verbose > 1:
                        if self.num_dict_atoms < 21:
                            value = np.sum(self.comp.reshape(-1, self.num_dict_atoms), 0)
                            for i in range(self.num_dict_atoms):
                                plt.imshow(np.abs(self.comp[0, :, i].reshape(self.imagesize, self.imagesize)),
                                           origin='lower')
                                plt.colorbar()
                                plt.title(
                                    "component {}, from SPIJN iteration {}, intensity value = {}".format(i, k,
                                                                                                         value[i]))
                                plt.show()

                # Check termination conditions
                if (rel_conv < tol and k >= min_iter) or np.isnan(rel_conv):  # or np.sum(np.sum(C,1)>1e-4)<num_comp:
                    if verbose:
                        print('Stopped after iteration {}'.format(k))
                    break

                # start next iteration
                rel_err_old = rel_err

        except KeyboardInterrupt:
            self.comp = comp_old
        if norm_correction and self.norms is not None:
            self.comp = dso.multiply(
                self.comp, 1 / self.norms,
                fixed_par_dict_masks=self.fixed_par_dict_masks,
                fixed_par_img_masks=[m.reshape(self.comp.shape[:-1]) for m in self.fixed_par_img_masks],
                S=self._S
            )

        if norm_ksp:
            self.ksp *= ksp_norm_fact[:, None, None, None, None]
            self.imgseq *= ksp_norm_fact[:, None, None]
            self.comp *= ksp_norm_fact[:, None, None]
        return rel_err

    def copy(self):
        """Defines copy functionality"""
        import copy
        return copy.copy(self)
        # c = self.__class__(ksp = self.ksp, coord = self.coord, dictmat = self.dictmat, dictt1 = self.dictt1, dictt2 = self.dictt2, maps = self.maps, 
        #                       mask = self.mask,
        #                       phasecorrection = self.phasecorrection, comp_mat = self.comp_mat, 
        #                       compress_dict = self.compress_dict, imgseq =  self.imgseq, comp = self.comp, index=self.index
        #                      b1_map = self.b1_map, dictb1 = self.dictb1, )
        # c.fixed_par_dict_masks =self.fixed_par_dict_masks
        # c.fixed_par_img_masks = self.fixed_par_img_masks

    def to_h5(self, path, filename='data.h5', save_raw=True, save_dict=True):
        """Write data to .h5 file

        Args:
            path (string): path to file location, must end in '/'
            filename (string): (optional) Filename of the output file (must end in .h5). Default to 'data.h5'
            save_raw (bool): (optional) If True, output file will also contain ksp and coord information.
        Output:
            None
        """
        if not os.path.exists(path):
            os.makedirs(path)

        keys = self.__dict__.keys()
        with h5py.File(os.path.join(path, filename), 'w') as hf:
            for key in keys:
                if save_raw is False and key in ('ksp', 'coord'):
                    continue
                elif save_dict is False and key in ('dictmat', 'compress_dict'):
                    continue
                value = self.__getattribute__(key)
                if isinstance(value, str):
                    value = value.encode()
                elif isinstance(value, list) and isinstance(value[0], str):
                    value = [v.encode() for v in value]

                if value is not None:
                    try:
                        hf.create_dataset(key, data=value, compression='gzip')
                    except TypeError:
                        print(key + ' could not be compressed while saving')
                        hf.create_dataset(key, data=value)

    @classmethod
    def from_h5(cls, path):
        """Load data from h5

        Args:
            path (string): path to the .h5 file

        Output:
            class instance with data from .h5 file
        """
        instance = cls()
        keys = instance.__dict__.keys()
        with h5py.File(path, 'r') as hf:
            for key in keys:
                v = hf.get(key)
                if isinstance(v, h5py.Dataset):
                    v = v[()]
                instance.__setattr__(key, v)
        return instance
