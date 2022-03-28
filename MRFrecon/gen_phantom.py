"""
    Filename: gen_numerical_phantom.py
    Author: Emiel Hartsema
    Date last modified: 9-10-2020
    Python version: 2.8
    Describtion: generates MRF image sequence from brainweb database ground truth
"""
import os

import numpy as np

try:
    import nibabel as nib
except ImportError:
    print('Nibabel not found')
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import sigpy as sp
import sigpy.mri as spmri
import math

import sys


def load_dict(settings):
    """Load dictionary and manipulate with mrf functions

    Args:
        -

    Output:
        (array): dictmat rotated to the real axis. array of shape (numdyn, numatoms)
        (list): list of T1 values corresponding to dictionary atoms
        (list): list of T2 values corresponding to dictionary atoms
    """
    if not (os.path.exists(settings['dictionary_path'])):
        print('Dictionary file not found at: {}'.format(settings['dictionary_path']))
        sys.exit(-1)
    if settings.getboolean('dict_corne_struct'):
        raise NotImplementedError('This option is not available anymore')

    else:
        with h5py.File(settings['dictionary_path'], 'r') as hf:
            dictmat = hf.get('dictmat')[:].T.imag
            t1list = hf.get('t1list')[:]
            t2list = hf.get('t2list')[:]
            if 'b1list' in hf:
                b1list = hf.get('b1list')[:]
            else:
                b1list = None
    return dictmat, t1list, t2list, b1list


def to_image(vec: np.ndarray):
    return vec.reshape((np.sqrt(len(vec)).astype(int), np.sqrt(len(vec)).astype(int)))


def to_vec(img: np.ndarray):
    return img.reshape(-1, 1).squeeze()


def load_components(settings):
    """ Load components wrapper function

    Args:
        settings file
    Output:
        Components (array): Array of component images of shape (n_components, imgshape[0], imgshape[1])
    """
    if settings['phantom_type'] == 'brainweb':
        components = load_brainweb_database(settings, )
    elif settings['phantom_type'] == 'artificial':
        components = load_artificial_phantom(settings, )
    else:
        raise Exception("Unknown source in load_components(), check spelling")

    return components


def load_metadata(settings, ):
    """ Load metadata wrapper funcion for for the multicomponent sources

    Args:
        settings: config object used to retrieve options

    Output:
        (several): component metadata
    """
    if settings['phantom_type'] == 'brainweb':
        return brainweb_metadata(settings)
    elif settings['phantom_type'] == 'artificial':
        return artificial_phantom_metadata(settings)
    else:
        raise Exception("Unknown source in load_metadata(), check spelling")


def brainweb_metadata(settings):
    """ Brainweb metadata
    Args:
        -

    Output:
        imgtype (list of strings): identifiers for brainweb files
        titles (list of strings): titles for brainweb components
        t1 (list): T1 values of components
        t2 (list): T2 values of components
        pd (list): Proton density values of components
    """
    if settings.getboolean('noskull'):
        imgtype = ['bck', 'csf', 'gm', 'wm']
        titles = ['Background', 'CSF', 'Grey matter', 'White matter']
        t1 = [0, 2569, 833, 500]
        t2 = [0, 329, 83, 70]
        # t2star = [0,   58,   69,   61]
        pd = [0, 1, 0.86, 0.77]
    else:
        imgtype = ['bck', 'csf', 'gm', 'wm', 'fat', 'muscles', 'muscles_skin', 'skull', 'vessels', 'fat2', 'dura',
                   'marrow']
        titles = ['Background', 'CSF', 'Grey matter', 'White matter', 'Fat', 'Muscle', 'Muscle skin', 'Skull',
                  'Vessels',
                  'Fat2', 'Dura', 'Marrow']
        t1 = [0, 2569, 833, 500, 350, 900, 2569, 0, 0, 500, 2569, 500]
        t2 = [0, 329, 83, 70, 70, 47, 329, 0, 0, 70, 329, 70]
        # t2star = [0,   58,   69,   61,  58,  30,   58, 0, 0,   61,   58,   61]
        pd = [0, 1, 0.86, 0.77, 1, 1, 1, 0, 0, 0.77, 1, 0.77]

    return imgtype, titles, t1, t2, pd


def artificial_phantom_metadata(settings):
    """ Artificial phantom metadata
        Args:
            settings: config object used to retrieve options

        Output:
            imgtype (list of strings): identifiers for brainweb files, None for artificial phantom
            titles (list of strings): titles for brainweb components
            t1 (list): T1 values of components
            t2 (list): T2 values of components
            pd (list): Proton density values of components
        """
    # load artificial image properties
    imgtype = None
    titles = ["background", "Component1", "Component2"]
    t1 = [0, 500, 1500]
    t2 = [0, 70, 210]
    pd = [0, 1, 1]
    return imgtype, titles, t1, t2, pd


def find_in_dict(dt1: np.ndarray, dt2: np.ndarray, t1: np.ndarray, t2: np.ndarray):
    """

    Args:
        dt1 (list of float): T1 values in dictionary
        dt2 (list of float): T2 values in dictionary
        t1 (list of float): T1 values to be found in dictionary
        t2 (list of float): T2 values to be found in dictionary

    Output
        (list of int): dictionary indecies that best match the t1 and t2 values
    """
    # find the dictionary entry that most closely matches the t1 and t2 values
    index = np.zeros(len(t1), dtype=int)
    for i in range(len(t1)):
        t1diff = dt1 - t1[i]
        t2diff = dt2 - t2[i]
        error = np.square(t1diff) + np.square(t2diff)
        index[i] = np.argmin(error)
        # print("T1 values, provided, matched")
        # print(t1[i])
        # print(dt1[index[i]])
        # print("T2 values, provided, matched")
        # print(t2[i])
        # print(dt2[index[i]])
    print(index)
    return index


def pad_image(data: np.ndarray, settings):
    """ Pad image with zeros to match the desired imag shape, can also crop

    Args:
        data (array): data array of shape (anything, anything)
        settings: config object used to retrieve options
    Output:
        (array): data array padded to shape (imgsize[0], imgsize[1])
    """
    output = np.zeros((settings.getint('imagesize'), settings.getint('imagesize')))

    # special exception for background image
    if data[0, 0] > 0.9:
        output = np.ones((settings.getint('imagesize'), settings.getint('imagesize')))

    dv = (settings.getint('imagesize') - data.shape[0]) / 2
    dh = (settings.getint('imagesize') - data.shape[1]) / 2
    if dv < 0:
        dv = -dv
        dh = -dh
        output = data[math.ceil(dv):-math.floor(dv), math.ceil(dh):-math.floor(dh)]
    else:
        output[math.ceil(dv):-math.floor(dv), math.ceil(dh):-math.floor(dh)] = data
    return output


def load_brainweb_database(settings, ):
    """ Load brainweb data components

    Args:
        settings: config object used to retrieve options

    Output:
        (array): component maps reshaped into vectors and stacked, array of shape (imgshape[0], imageshapep[1], n_maps)
    """
    print("Loading images from Brainweb database")
    # Identify slicenumbers
    slicenums = list(map(int, settings['brainweb_slice_num'].split(",")))

    # Load metadata from bainweb database
    imgtype, titles, t1, t2, pd = brainweb_metadata(settings)
    groundtruth = np.zeros((len(slicenums), settings.getint('imagesize') * settings.getint('imagesize'), len(imgtype)))
    for j in range(len(slicenums)):
        for i in tqdm(range(len(imgtype))):
            # assemble string
            pathstring = settings['brainweb_path'] + settings['brainweb_subject'] + '_' + imgtype[i] + '.nii.gz'
            nibfile = nib.load(pathstring)

            # get data from .nii file
            data = nibfile.get_fdata()

            # get slice from data
            imgslice = data[slicenums[j], :, :]

            # add padding to image
            imgslice = pad_image(imgslice, settings)

            # store output
            groundtruth[j, :, i] = to_vec(imgslice)

            # plot ground truth in subplot
            if settings.getboolean('showplot'):
                plt.subplot(3, 4, i + 1)
                plt.imshow(imgslice, cmap='gray', origin="lower")
                plt.title(titles[i])

                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
        if settings.getboolean('showplot'):
            plt.show()

    return groundtruth


def load_artificial_phantom(settings):
    """ Load artificial data components

    Args:
        settings: config object used to retrieve options

    Output:
        (array): component maps reshaped into vectors and stacked, array of shape (imgshape[0], imageshapep[1], n_maps)
    """
    imgtype, titles, t1, t2, pd = artificial_phantom_metadata(settings)
    # make circular mask
    y, x = np.ogrid[-128:128, -128:128]
    mask = x * x + y * y <= 90 * 90

    # load components
    blk = np.zeros((256, 256))
    cp1 = np.zeros((256, 256))
    cp2 = np.zeros((256, 256))
    cp1[:, 0:86] = np.ones((256, 86))
    cp1[:, 86:170] = np.ones((256, 84)) * 0.5
    cp2[:, 86:170] = np.ones((256, 84)) * 0.5
    cp2[:, 170:256] = np.ones((256, 86))

    # mask components
    blk[np.invert(mask)] = 1
    cp1[np.invert(mask)] = 0
    cp2[np.invert(mask)] = 0

    components = np.zeros((256 * 256, 3))
    components[:, 0] = to_vec(blk)
    components[:, 1] = to_vec(cp1)
    components[:, 2] = to_vec(cp2)

    # plot components
    if settings.getboolean('showplot'):
        plt.figure()
        for i in range(components.shape[1]):
            plt.subplot(1, components.shape[1] + 1, i + 1)
            plt.imshow(to_image(components[:, i]), cmap='gray', origin="lower")
            plt.title(titles[i])
        plt.show()
    return components


def gen_imgseq(groundtruth: np.ndarray, dictmat: np.ndarray,
               dictt1: np.ndarray, dictt2: np.ndarray, settings):
    """ Generate image sequences
    Loop over components and apply dictionary to generate image sequences. These component specific image sequences are
    added together to for the final image sequence.

    Args:
        groundtruth (array): component maps reshaped into vectors and stacked,
                                array of shape (imgshape[0], imageshapep[1], n_maps)
        dictmat (array): dictmat rotated to the real axis. array of shape
        dictt1 (list): list of T1 values corresponding to dictionary atoms
        dictt2 (list): list of T2 values corresponding to dictionary atoms
        settings: config object used to retrieve options
    Output:
        (array): image sequence, images reshaped into vectors and stacked
    """
    imgtype, titles, t1, t2, pd = load_metadata(settings['phantom_type'])
    image_sequence = np.zeros((groundtruth.shape[0], groundtruth.shape[1], dictmat.shape[0]), dtype='float64')

    index = find_in_dict(dictt1, dictt2, t1, t2)
    # print(dictt1[index])
    # print(dictt2[index])
    print("Generating image sequence")
    # Make transformation matrix
    trans_mat = np.matmul(dictmat[:, index], np.diag(pd)).T

    # calculate image sequence
    oshape = image_sequence.shape[:2] + trans_mat.shape[1:]
    image_sequence = (groundtruth.reshape(-1, groundtruth.shape[2]) @ trans_mat).reshape(oshape)

    return image_sequence


def subsample_imseq(imgseq, settings):
    """ Subsample image sequence
    Use the sense linop from the sigpy library to sample the image sequene.

    Args:
        imgseq: image sequence, images reshaped into vectors and stacked
        settings: config object used to retrieve options
    Output:
        kspace_array (array): k-space data array of shape (n_coils, timelength, coordlength)
        coord_array (array): coord array of shape (timelength, coordlength, 2)
    """
    print('Sample image sequence')
    # Calculate birdcage maps
    birdcagemaps = gen_birdcage(settings)
    spiral_coord = gen_coord(settings, 0)
    if settings.getboolean('showplot'):
        plt.figure()
        plt.plot(spiral_coord[:, 0], spiral_coord[:, 1])
        plt.show()

    ksp_oshape = list((imgseq.shape[0], birdcagemaps.shape[0], imgseq.shape[2], 1, spiral_coord.shape[0]))
    kspace_array = np.zeros(ksp_oshape, dtype='complex128')
    coord_array = np.zeros(ksp_oshape[2:] + [2])
    # loop over all images and timepoints
    for i in tqdm(range(imgseq.shape[0] * imgseq.shape[2])):
        im_slice = int(np.floor(i / imgseq.shape[2]))
        timepoint = i % imgseq.shape[2]

        # generate spiral
        spiral_coord = gen_coord(settings, timepoint)
        coord_array[timepoint, 0, :, :] = spiral_coord

        # Sense operator split to apply noise over single coil images
        ishape = birdcagemaps.shape[1:]
        mapslinop = sp.linop.Multiply(ishape, birdcagemaps)
        nufftlinop = sp.linop.NUFFT(mapslinop.oshape, coord_array[timepoint, ...])

        # Calculate single coil images  apply noise and calculate ksp trajectory
        single_coil_images = mapslinop * to_image(imgseq[im_slice, :, timepoint])
        noisy_single_coil_img = add_noise_img(single_coil_images, settings.getint('phantom_snr'))
        ksp_traj_linop = nufftlinop * noisy_single_coil_img
        kspace_array[im_slice, :, timepoint, ...] = ksp_traj_linop

    return kspace_array, coord_array


def gen_coord(settings, index=0, interleaf=0, ):
    """ Generate coordinates

    Args:
        index (int): Defines how many times coord are rotated by the golden angle.
        interleaf(int): Defines how many times another interleave has to be added, used as input and recursion.
        settings: config object used to retrieve options
    Output:
        (array): coord array of shape (coordlength, 2)

    """
    if settings['samp_pattern'] == 'spiral':
        # Generate spiral trajectory
        # spiral(fov, N, f_sampling, R, ninterleaves, alpha, max_grad_amp, max_slew_rate, gamma=2.678e8):
        # coord = spmri.spiral(0.24, 120, 1/4, 36, 1, 6, 0.03, 150)
        coord = spmri.spiral(settings.getfloat('spir_fov'), settings.getint('spir_mat_size'),
                             settings.getfloat('spir_undersamp_freq'), settings.getfloat('spir_undersamp_phase'), 1,
                             settings.getfloat('spir_var_dens_fac'), settings.getfloat('spir_grad_amp'),
                             settings.getfloat('spir_max_slew_rate'))
    elif settings['samp_pattern'] == 'radial':
        rad = spmri.radial((32, 512, 2), (200, 200), golden=False)
        rad = spmri.radial((settings.getint('rad_num_spokes'), settings.getint('rad_num_read_out'), 2),
                           (settings.getint('imagesize'), settings.getint('imagesize')), golden=False)
        coord = np.zeros((rad.shape[0] * rad.shape[1], 2))
        coord[:, 0] = rad[:, :, 0].reshape(-1)
        coord[:, 1] = rad[:, :, 1].reshape(-1)
    else:
        raise Exception("unknown sampling pattern, check spelling!")

    # Scale coordinages
    # TDO: gen_coord(): this should not be required
    coord = coord * (settings.getint('imagesize') / 256)

    # rotate by index * golden-angle
    gold_ang = -np.pi * (3 - np.sqrt(5))
    comp_coord = (coord[:, 0] + 1j * coord[:, 1]) * np.exp(1j * gold_ang * index)
    realcoord = np.real(comp_coord)
    imgcoord = np.imag(comp_coord)
    output = np.stack((realcoord, imgcoord), axis=1)

    if interleaf:
        output = np.vstack((output, gen_coord(settings, index=index + 1, interleaf=interleaf - 1)))
    return output


def gen_phantom(settings):
    """ Generate numerical phantom Call this function to generate the phantom, first it loads the component maps
    either from the brainweb databse or the artificial phantom hardcocded in the program. Afterwards it loads the
    dictionary and calculates the image sequene. Finally the image sequence is sampled to the spiral coordinates.

    Args:
        -

    Output:
        kspace_array (array): k-space data array of shape (n_coils, timelength, coordlength)
        coord_array (array): coord array of shape (timelength, coordlength, 2)
        imgseq (array): Image sequence of numerical phantom, images reshaped into vectors and stacked
    """

    component_groundtruth = load_components(settings)
    component_groundtruth[:, :, 0] = np.ones_like(component_groundtruth[:, :, 0])

    dictmat, dictt1, dictt2, _ = load_dict(settings)

    imgseq = gen_imgseq(component_groundtruth, dictmat, dictt1, dictt2, settings=settings)

    # subsample the image
    kspace_array, coord_array = subsample_imseq(imgseq, settings)

    # store to .h5 file
    print("Saving result to .h5 file")
    if not os.path.exists(settings['phantom_output_path']):
        os.makedirs(settings['phantom_output_path'])
    with h5py.File(settings['phantom_output_path'] + 'phantom.h5', 'w') as hf:
        hf.create_dataset('groundtruth', data=component_groundtruth)
        hf.create_dataset('kspace', data=kspace_array)
        hf.create_dataset('coord', data=coord_array)
        hf.create_dataset('imgseq', data=imgseq)

    return kspace_array, coord_array, imgseq


def gen_birdcage(settings):
    """Generate sensitivity maps for birdcage coils

    Args:
        -

    Output:
        (array): Sensitivity maps of shape (n_coils, imagesize[0], imagesize[1])
    """
    cage_diam = settings.getfloat('cage_diam')  # in meters
    fov = settings.getfloat('coil_fov')  # field of view in meters, 1mm voxel size
    maps = spmri.birdcage_maps((settings.getint('num_sense_coils'), settings.getint('imagesize'),
                                settings.getint('imagesize')), cage_diam / fov, settings.getint('coils_per_ring'))
    return maps


def add_noise_img(imgseq, snr):
    if snr is not None:
        for k in range(imgseq.shape[0]):
            i = imgseq[k, ...]
            i_max = np.abs(i).max()
            imgseq[k, ...] += np.random.normal(0, i_max / snr, i.shape) + \
                              1j * np.random.normal(0, i_max / snr, i.shape)
    return imgseq


if __name__ == "__main__":
    from config import load_settings

    settings_ = load_settings()
    gen_phantom(settings_)
