"""
Script to load data, was originally in main.py
"""
import os
import warnings

import h5py
import numpy as np
from PIL import Image
import nibabel as nib
try:
    from DICOM_files_Philips import read_enh_dicom
except ImportError:
    read_enh_dicom = False

from .backend import MrfData
from .gen_phantom import gen_phantom, load_dict


# from preprocess_MRI_data import preprocess_mri_data


def load_b1(settings, ):
    try:
        if 'b1_map' in settings:
            file = settings['b1_map']
    except FileNotFoundError:
        print(f'File {file} not found during loading B1')
    except BaseException as e:
        print(f'{e} during loading B1')

    if file is None or not os.path.exists(file):
        print(f'B1 file {file} not found')
        return None
    if file.endswith('dcm') and read_enh_dicom is not False:
        b1_map = read_enh_dicom(file, slices='all', MP=True)['images'] / 100
    elif file.endswith('.npy'):
        b1_map = np.load(file)
    b1_map[b1_map == 0] = np.nan
    b1_map = np.abs(b1_map[:, 0].transpose((0, 2, 1)))
    return b1_map


def load_data(settings):
    # test if we need to calculate kspace and coordinates
    if settings['datasource'] == 'phantom':
        # calculate kspace and coord from phantom
        ksp, coord, groundtruth = gen_phantom(settings)
    elif settings['datasource'] == 'h5':
        if not (os.path.exists(settings['mri_data_path'])):
            raise FileNotFoundError('Data file not found at: {}'.format(settings['mri_data_path']))
        with h5py.File(settings['mri_data_path'], 'r') as hf:
            ksp = hf.get('kspace')[:]
            coord = hf.get('coord')[:]
    else:
        warnings.warn("Skip to next run; MRI data source not identified in [{}]. Got [{}] expected phantom, "
                      "mri_data or h5".format(settings['path_extension'], settings['datasource']))

    # load dictioary
    dictmat, dictt1, dictt2, dictb1 = load_dict(settings)

    # Load B1 map
    if 'b1_map' in settings:
        b1_map = load_b1(settings)
    else:
        b1_map = None

    if dictb1 is not None and b1_map is None:
        if len(np.unique(dictb1)) > 1:
            warnings.warn('Dictionary contains b1 values, but no B1 map was provided...')
    elif dictb1 is None and b1_map is not None:
        raise IOError('Dictionary does not contain b1, but a B1map was provided.')

    if dictmat.shape[0] != coord.shape[0]:
        if dictmat.shape[1] == coord.shape[0]:
            dictmat = dictmat.T
            warnings.warn(
                'Dictionary signal length and coord length did not match!, I transposed the dictionary matrix myself.')
        else:
            raise IOError('Dictionary signal length and coord length did not match!')
    # reduce dictionary
    reducedict = settings.getint('reduce_dict_length')
    if reducedict is not None:
        dictmat = dictmat[:reducedict, :]
        ksp = ksp[:, :, :reducedict, :, :]
        coord = coord[:reducedict, ...]

    norms = np.linalg.norm(dictmat, axis=0)
    dictmat /= norms[None, :]

    retrospective_undersampling = settings.get('retrospective_undersampling', fallback=None)
    if retrospective_undersampling is not None:
        retrospective_undersampling = eval(retrospective_undersampling)
        if retrospective_undersampling is not False:
            if isinstance(retrospective_undersampling, int):
                retrospective_undersampling = [retrospective_undersampling]
            ksp = ksp[..., retrospective_undersampling, :]
            coord = coord[:, retrospective_undersampling]

    # Setup the data class
    data = MrfData()
    data.ksp = ksp
    data.coord = coord
    data.dictmat = dictmat
    data.dictt1 = dictt1
    data.dictt2 = dictt2
    data.dictb1 = dictb1
    data.set_b1_map(b1_map)
    data.norms = norms

    def load_mask(mask_path):
        if '.nii' in mask_path:
            mask = nib.load(mask_path).get_fdata().astype(bool)
        else:
            maskimage = Image.open(mask_path).convert('1')
            mask = np.array(maskimage)

            imagesize = mask.shape[0]
            numslice = int(mask.shape[1] // imagesize)
            if imagesize * numslice != mask.shape[1]:
                raise IOError("Mask image of invalid shape, width is not an integer multiple of height")
            mask = mask.reshape((imagesize, imagesize, numslice), order='F')
            mask = mask.transpose(2, 0, 1)
        return mask				   

    if settings.get('mask_path') is not None:
        data.mask = load_mask(settings['mask_path'])

    if settings.get('mask_spijn_path') is not None:
        data.spijn_mask = load_mask(settings['mask_spijn_path'])

    return data
