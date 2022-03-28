"""
    Filename: postprocessing.py
    Author: Emiel Hartsema
    Date last modified: 25-11-2020
    Python version: 2.7
    Describtion: reconstruction algoritms to reconstruct mrf maps
"""

import os
import os.path
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
from tqdm import tqdm

from .gen_phantom import to_image, load_dict

try:
    import nibabel as nib
except ImportError:
    nib = False

# Load data from configuration file
from .config import load_config_file

from .backend import MrfData


def showcomponents(recondatapath, showplot=True):
    # load dataset
    with h5py.File(recondatapath + 'data.h5', 'r') as hf:
        components = hf.get('comp')[0, ...]
        dictindex = hf.get('index')[:]
        dictt1 = hf.get('dictt1')[:]
        dictt2 = hf.get('dictt2')[:]
        img = -hf.get('imgseq')[0, :, 0]
        dictmat = hf.get('dictmat')[:]
    """
    # load components
    #with h5py.File(MRI_DATA_PATH, 'r') as hf:
        groundtruth = hf.get('groundtruth')[:]
    mixed_gt = np.zeros((groundtruth.shape[0], 3))
    # WM component
    # mixed_gt[:, 0] = 0.77 * np.sum(groundtruth[:, [3, 9, 11]], axis=1)
    mixed_gt[:, 0] = 0.77 * np.sum(groundtruth[:, [3]], axis=1)
    # GM component
    # mixed_gt[:, 1] = 0.86 * np.sum(groundtruth[:, [2, 5]], axis=1)
    mixed_gt[:, 1] = 0.86 * np.sum(groundtruth[:, [2]], axis=1)
    # CSF component
    # mixed_gt[:, 2] = 1.00 * np.sum(groundtruth[:,[1, 6, 10]], axis=1)
    mixed_gt[:, 2] = 1.00 * np.sum(groundtruth[:, [1]], axis=1)
    """
    if not os.path.exists(recondatapath + 'Components/'):
        os.makedirs(recondatapath + 'Components/')
    if not os.path.exists(recondatapath + 'Components_norm/'):
        os.makedirs(recondatapath + 'Components_norm/')

    # Show 1st 3 components together
    """
    for i in range(3):
        plt.figure()
        fig, ax = plt.subplots(1, 3)
        image = to_image(components[:, i])
        gtimage = to_image(mixed_gt[:, i])
        ax[0].imshow(image)
        ax[0].set_title('Reconstructed image')
        ax[0].tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
        im1 = ax[1].imshow(image - gtimage)
        ax[1].set_title('Difference')
        ax[1].tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        plt.colorbar(im1, ax=ax[1], cax=cax, orientation='horizontal')
        ax[2].imshow(gtimage)
        ax[2].set_title('Ground truth')
        ax[2].tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
        fig.suptitle(
            '$T_1={:.1f}, T_2={:.1f}, Index={:.1f}$'.format(dictt1[dictindex[i]], dictt2[dictindex[i]], dictindex[i]))
        plt.savefig(recondatapath + 'Components/component{index}'.format(index=i) + '.png')
        if showplot:
            plt.show()
    """
    mask = np.zeros_like(img)
    mask[img > 0.05 * max(img)] = 1
    mask = to_image(mask)
    if showplot:
        # plot rank 1 image
        plt.imshow(to_image(img))
        plt.colorbar()
        plt.show()

        plt.imshow(mask)
        plt.show()
    # compensate for reduction in dictionary length
    dictlen = dictmat.shape[0]
    og_dictmat, _, _ = load_dict()
    og_dictmat = og_dictmat[:dictlen, :]
    inv_scalefact = np.linalg.norm(og_dictmat, axis=0)[:]
    inv_scalefact = inv_scalefact[dictindex]
    """
    # normalize components
    norm = np.linalg.norm(components, axis=1)
    norm[norm<0.00001*max(norm)] = 1
    components /= norm[:, None]
    """
    # show components seperately
    for i in range(components.shape[1]):
        plt.figure()
        image = to_image(components[:, i]) / inv_scalefact[i]
        plt.imshow(image)
        plt.title('$T_1={:.1f}, T_2={:.1f}, Index={:.1f}$'.format(dictt1[dictindex[i]],
                                                                  dictt2[dictindex[i]], dictindex[i]))
        plt.colorbar()
        plt.savefig(recondatapath + 'Components/component{index}'.format(index=i) + '.png')
        if showplot:
            plt.show()
        else:
            plt.close('all')

    # show components but normalized
    sum_comp = to_image(np.sum(components, axis=1))
    for i in range(components.shape[1]):
        plt.figure()
        image = to_image(components[:, i]) / inv_scalefact[i]
        plt.imshow(np.divide(image, sum_comp, out=np.zeros_like(image), where=mask.astype(np.bool)))
        plt.title('$T_1={:.1f}, T_2={:.1f}, Index={:.1f}$'.format(dictt1[dictindex[i]],
                                                                  dictt2[dictindex[i]], dictindex[i]))
        plt.colorbar()
        plt.savefig(recondatapath + 'Components_norm/component{index}'.format(index=i) + '.png')
        if showplot:
            plt.show()
        else:
            plt.close('all')


def showimgseq(recondatapath, showplot=True):
    # load dataset
    with h5py.File(recondatapath + 'data.h5', 'r') as hf:
        lrimgseq = hf.get('imgseq')[0, ...]

    if not os.path.exists(recondatapath + 'Imgseq/'):
        os.makedirs(recondatapath + 'Imgseq/')

    for i in range(lrimgseq.shape[1]):
        plt.figure()
        # plt.subplot(1, 2, 1)
        plt.imshow(to_image(np.abs(lrimgseq[:, i])), origin='lower')
        plt.axis('off')
        # plt.title("Magnitude")
        # plt.colorbar()
        # plt.subplot(1, 2, 2)
        # plt.imshow(to_image(np.angle(lrimgseq[:, i])), cmap='twilight')
        # plt.title("Phase")
        # plt.colorbar()
        plt.savefig(recondatapath + 'Imgseq/image{index}'.format(index=i) + '.png')
        if showplot:
            plt.show()
        else:
            plt.close('all')


def showlagrange(recondatapath, showplot=True):
    # load dataset
    with h5py.File(recondatapath + 'data.h5', 'r') as hf:
        lag_mult = hf.get('lagrange_multipliers')[:]
        lrimgseq = hf.get('imgseq')[:]

    if not os.path.exists(recondatapath + 'Lagrange/'):
        os.makedirs(recondatapath + 'Lagrange/')

    for i in range(lag_mult.shape[1]):
        # Plot scaled lagrange multipliers
        plt.figure()
        plt.subplot(2, 3, 1)
        plt.imshow(to_image(lag_mult[:, i].real))
        plt.title("Lagrange multiplier \n real part")
        plt.colorbar()

        plt.subplot(2, 3, 4)
        plt.imshow(to_image(lag_mult[:, i].imag))
        plt.title("imag part")
        plt.colorbar()

        # plot image sequence
        plt.subplot(2, 3, 2)
        plt.imshow(to_image(lrimgseq[:, i].real))
        plt.title("Image \n real part")
        plt.colorbar()

        plt.subplot(2, 3, 5)
        plt.imshow(to_image(lrimgseq[:, i].imag))
        plt.title("imag part")
        plt.colorbar()

        # Plot SPIJN input/discarded part
        plt.subplot(2, 3, 3)
        plt.imshow(to_image((lag_mult[:, i] + lrimgseq[:, i]).real))
        plt.title("SPIJN \n input")
        plt.colorbar()

        plt.subplot(2, 3, 6)
        plt.imshow(to_image((lag_mult[:, i] + lrimgseq[:, i]).imag))
        plt.title("discarded")
        plt.colorbar()

        plt.savefig(recondatapath + 'Lagrange/lagrange{index}'.format(index=i) + '.png')
        if showplot:
            plt.show()
        else:
            plt.close('all')


def showerrortypes(recondatapath, phantomdatapath, showplot=True):
    with h5py.File(phantomdatapath + 'phantom.h5', 'r') as hf:
        originalimgseq = hf.get('imgseq')[:]

    with h5py.File(recondatapath + 'data.h5', 'r') as hf:
        components = hf.get('comp')[:]
        comp_index = hf.get('index')[:]
        # reconimgseq = hf.get('frimgseq')[:]

    dictmat, dictt1, dictt2 = load_dict()
    reconimgseq = np.matmul(components, dictmat[:, comp_index].T)

    diffimgseq = originalimgseq - reconimgseq
    rssimg = np.sqrt(np.sum(np.square(diffimgseq), axis=2) / diffimgseq.shape[2])
    plt.figure()
    plt.imshow(to_image(rssimg[0, :]))
    plt.colorbar()
    plt.savefig(recondatapath + 'error.png')
    if showplot:
        plt.show()
    else:
        plt.close('all')


def show_convergence(recondatapath, showplot=True):
    with h5py.File(recondatapath + 'data.h5', 'r') as hf:
        objvals = hf.get('objective_value')[:]
    for i in range(objvals.shape[0]):
        plt.semilogy(objvals[i, :], label='Iteration {}'.format(i + 1))
    plt.legend()
    if showplot:
        plt.show()
    else:
        plt.close('all')


def showresiduals(recondatapath, showplot=True):
    with h5py.File(recondatapath + 'data.h5', 'r') as hf:
        residual = hf.get('residuals')[:]
    plt.plot(residual[0, :], label='|GFSDc-y|_F')
    plt.plot(residual[1, :], label='|GFSx-y|_F')
    plt.plot(residual[2, :], label='|x-Dc|_F')
    plt.plot(residual[3, :], label='total')
    plt.plot(residual[4, :], label='|GFSDc-y|_F+mu|c|')
    plt.legend()
    plt.savefig(recondatapath + 'Residuals' + '.png')
    if showplot:
        plt.show()
    else:
        plt.close('all')


def show_single_comp_match(recondatapath, showplot=True):
    with h5py.File(recondatapath + 'data.h5', 'r') as hf:
        t1img = hf.get('t1img')[:]
        t2img = hf.get('t2img')[:]
        pdimg = hf.get('pdimg')[:]
        nrmseimg = hf.get('nrmseimg')[:]
    # plot results
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(to_image(t1img), origin="lower")
    plt.title("T1 image")
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.imshow(to_image(t2img), origin="lower")
    plt.title("T2 image")
    plt.colorbar()
    plt.subplot(2, 2, 3)
    plt.imshow(to_image(pdimg), origin="lower")
    plt.title("PD image")
    plt.scatter(130, 105, color='r', marker='x')  # big blob
    plt.scatter(95, 175, color='r', marker='x')  # TL blob
    plt.scatter(115, 175, color='r', marker='x')  # TR blob
    plt.scatter(95, 160, color='r', marker='x')  # BL blob
    plt.scatter(115, 160, color='r', marker='x')  # BR blob
    plt.colorbar()
    plt.subplot(2, 2, 4)
    plt.imshow(to_image(np.divide(nrmseimg, pdimg, out=np.zeros_like(nrmseimg), where=pdimg != 0)), origin="lower")
    plt.title("RMSE image")
    plt.colorbar()
    plt.savefig(recondatapath + 'single_component_match' + '.png')
    if showplot:
        plt.show()
    else:
        plt.close('all')


def postpro_admm(recondatapath, phantom_output_path, max_admm_iter):
    for i in tqdm(range(max_admm_iter)):
        try:
            showcomponents(recondatapath + '{}/'.format(i), False)
        except TypeError:
            print("no components")
        showimgseq(recondatapath + '{}/'.format(i), False)
        showlagrange(recondatapath + '{}/'.format(i), False)
        try:
            showerrortypes(recondatapath + '{}/'.format(i), phantom_output_path, False)
        except:
            pass
            # print("No phantom data found, no error types calculated for admm iter: {}".format(i))
    showresiduals(recondatapath, False)


def bulk_postpro(path, only_first_run=False):
    # load config file
    config = load_config_file(path)
    runs = list(config.sections())[1:]
    if only_first_run:
        runs = runs[0]
    for run in runs:
        print('Processing run: {}'.format(run))
        config['DEFAULT']['path_extension'] = run
        settings = config[run]

        # check for missing files
        if not os.path.exists(settings['reconstruction_output_path']):
            print("Run {} not found, continuing...".format(settings['path_extension']))
            continue

        if settings['recon_alg'] in ('filt_back_proj', 'lr_filt_back_proj', 'invert'):
            showimgseq(settings['reconstruction_output_path'], False)
            show_single_comp_match(settings['reconstruction_output_path'], False)
        elif settings['recon_alg'] in ('invert_into_spijn', 'admm_into_spijn', 'direct_spijn', 'sense_into_spijn'):
            try:
                showimgseq(settings['reconstruction_output_path'], False)
                showcomponents(settings['reconstruction_output_path'], False)
            except FileNotFoundError as e:
                print(f'Saving figures {run} ' + e)
            try:
                showerrortypes(settings['reconstruction_output_path'], settings['phantom_output_path'], False)
            except OSError:
                pass
                # print("No phantom data found, no error types calculated for run: {}".format(run))
        # elif settings['recon_alg'] in ('admm_into_spijn', 'admm_spijn'):
        #    postpro_admm(settings['reconstruction_output_path'], settings['phantom_output_path'], settings.getint('max_admm_iter'))


def multirun_convergence(configpath, only_first_run=False):
    config = load_config_file(configpath)
    runs = list(config.sections())[2:]
    param = np.zeros(len(runs))
    i = 0
    for run in runs:
        config['DEFAULT']['path_extension'] = run
        settings = config[run]
        combined_residuals = np.zeros((4, len(runs), settings.getint('max_admm_iter')))
        with h5py.File(settings['reconstruction_output_path'] + 'data.h5', 'r') as hf:
            residual = hf.get('residuals')[:]

        combined_residuals[0, i, :] = residual[0, :]  # |GFSDc-y|
        combined_residuals[1, i, :] = residual[1, :]  # |GFSx-y|
        combined_residuals[2, i, :] = residual[2, :]  # |Dc-x|
        combined_residuals[3, i, :] = residual[3, :]  # |Dc-x|
        param[i] = settings['reconstruction_rank']

        i += 1
    upperlim = 0
    plt.figure()
    for i in range(combined_residuals.shape[1] - upperlim):
        plt.semilogy(combined_residuals[0, i, :], label='rank = {}'.format(round(param[i], 2)))
    plt.legend()
    plt.title('|GFSDc-y|')
    plt.show()

    plt.figure()
    for i in range(combined_residuals.shape[1] - upperlim):
        plt.semilogy(combined_residuals[1, i, :], label='rank = {}'.format(round(param[i], 2)))
    plt.legend()
    plt.title('|GFSx-y|')
    plt.show()

    plt.figure()
    for i in range(combined_residuals.shape[1] - upperlim):
        plt.semilogy(combined_residuals[2, i, :], label='rank = {}'.format(round(param[i], 2)))
    plt.legend()
    plt.title('|Dc-x|')
    plt.show()

    plt.figure()
    for i in range(combined_residuals.shape[1] - upperlim):
        plt.semilogy(combined_residuals[3, i, :], label='rank = {}'.format(round(param[i], 2)))
    plt.legend()
    plt.title('|GFSx-y|+mu1|c|+mu2|Dc-x+v|')
    plt.show()


def imgseq_as_nifti(recondatapath):
    with h5py.File(recondatapath + 'data.h5', 'r') as hf:
        lrimgseq = hf.get('imgseq')[:]
    imsize = int(np.sqrt(lrimgseq.shape[1]))
    lrimgseq = lrimgseq.reshape(lrimgseq.shape[0], imsize, imsize, lrimgseq.shape[2])
    img = nib.Nifti1Image(np.transpose(lrimgseq, (2, 1, 0, 3)), np.diag([1, 1, 5, 1]))
    nib.save(img, recondatapath + 'imgseq.nii.gz')
    print("Done")


def components_as_nifti(recondatapath):
    with h5py.File(recondatapath + 'data.h5', 'r') as hf:
        components = hf.get('comp')[:]
        dictindex = hf.get('index')[:]
        dictmat = hf.get('dictmat')[:]

    # compensate for reduction in dictionary length
    dictlen = dictmat.shape[0]
    og_dictmat, _, _ = load_dict()
    og_dictmat = og_dictmat[:dictlen, :]
    inv_scalefact = np.linalg.norm(og_dictmat, axis=0)[:]
    inv_scalefact = inv_scalefact[dictindex]
    components /= inv_scalefact[None, None, :]

    imsize = int(np.sqrt(components.shape[1]))
    lrimgseq = np.log(components.reshape(components.shape[0], imsize, imsize, components.shape[2]))
    img = nib.Nifti1Image(np.transpose(lrimgseq, (2, 1, 0, 3)), np.diag([1, 1, 5, 1]))
    nib.save(img, recondatapath + 'log_components.nii.gz')
    print("Done")


def load_settings(configpath, run):
    # Try to read path to config.ini file
    try:
        path = sys.argv[1]
    except IndexError:
        path = configpath
    if path is None:
        pass
    elif not (os.path.exists(os.path.join(path, 'config.ini'))):
        print('Config file not found at {}'.format(os.path.join(path, 'config.ini')))
        raise FileNotFoundError('Config file not found')

    config = load_config_file(path)
    # get settings for runs
    run_name = config['RUN'][str(run + 1)]
    config['DEFAULT']['path_extension'] = run_name
    return config[run_name]


def admm_convergence(configpath, num_runs=10, admm_iter=20,
                     save=True, comp_device=None, calc_ksp=False):
    def comp2t12(t12list, comp, imagesize, t1or2, admm_it, mask=None):
        # To calculate geometric mean of t1 or t2
        logt12 = np.log(t12list)
        t12map = np.exp(np.dot(comp, logt12) / np.sum(comp, axis=-1))
        if mask is not None:
            t12map[~mask] = 0
        # plt.imshow(t12map.reshape(imagesize, imagesize))
        # plt.title('T{} map from multicomponent from admm iter {}'.format(t1or2, admm_it))
        # plt.colorbar()
        # plt.show()
        return t12map

    settings = load_settings(configpath, 0)
    meanresult = np.zeros((num_runs, admm_iter, 4))
    relmeanresult = np.zeros((num_runs, admm_iter, 4))
    stdresult = np.zeros((num_runs, admm_iter, 4))

    if comp_device is None:
        comp_device = settings.getint('compute_device')
    # load dictionary
    with h5py.File(settings['dictionary_path'], 'r') as hf:
        t1list = hf.get('t1list')[:]
        t2list = hf.get('t2list')[:]
    # load ground truth
    with h5py.File(settings['mri_data_path'], 'r') as hf:
        groundtruth = np.squeeze(hf.get('groundtruth')[:])
    pd = [0, 1, 0.86, 0.77, 1, 1, 1, 0, 0, 0.77, 1, 0.77]
    groundtruth = groundtruth @ np.diag(pd)
    dict_index = [0, 3151, 1794, 1232, 899, 1852, 3151, 0, 0, 1232, 3151, 1232]
    # Create Phantom reference values
    phantomt1 = comp2t12(t1list[dict_index], groundtruth, 256, '1', 'phantom')
    phantomt2 = comp2t12(t2list[dict_index], groundtruth, 256, '2', 'phantom')
    phantompd = np.sum(groundtruth, axis=1)
    mask = np.logical_and(phantomt1 > 0, phantompd > .1)[None]
    admm_params = []
    for j in range(num_runs):
        settings = load_settings(configpath, j)
        admm_params.append(settings.getfloat('admm_param'))
        for i in range(admm_iter):
            # load dataset

            with h5py.File(settings.get('admm_outpath') + 'admm{}.h5'.format(i), 'r') as data:
                imsize = settings.getint('imagesize')
                numslice = data['imgseq'].shape[0]
                comp = data['comp'][()]

                measpd = np.sum(comp / data['norms'][()], axis=-1)

                pdcorr = np.nanpercentile(measpd, 99)
                comp /= pdcorr
                measpd /= pdcorr
                pderror = (measpd - phantompd)
                pderrorrel = pderror / phantompd
                # calc t1t2 maps with geometric mean
                t1map = comp2t12(data['dictt1'][()], comp,
                                 imsize, '1', i, mask)
                t2map = comp2t12(data['dictt2'][()], comp,
                                 imsize, '2', i, mask)
                # calc linear operator
                if calc_ksp:
                    if i == 0:
                        list_linop = []
                        for k in range(numslice):
                            coord = sp.to_device(data['coord'][()], comp_device)
                            comp_mat = sp.to_device(data['comp_mat'][()].T, comp_device)
                            maps = sp.to_device(data['maps'][()], comp_device)
                            singleslice_linop = MrfData.lr_sense_op(coord,
                                                                    comp_mat, imagesize=imsize,
                                                                    oversamp=settings.getfloat('oversamp_ratio'),
                                                                    maps=maps,
                                                                    batchsize=settings.getint('lr_sense_batchsize'))
                            list_linop.append(singleslice_linop)
                        linop = sp.linop.Diag(list_linop, 0, 0)
                    # calc residual
                    # |GFSDc-y|
                    dc = (comp @ data['compress_dict'][()].T).reshape(data['imgseq'].shape)

                    dc_phasecorr = dc * np.exp(1j * data['phasecorrection'][()])[:, :, None]
                    dc_img = np.transpose(dc_phasecorr, (0, 2, 1)).reshape(numslice,
                                                                           settings.getint('reconstruction_rank'),
                                                                           imsize,
                                                                           imsize).astype(np.complex64)
                    dc_img_gpu = sp.to_device(dc_img, comp_device)
                    ldc_img_gpu = linop(dc_img_gpu)
                    kspgpu = sp.to_device(data['ksp'][()], comp_device)
                    xp = sp.get_array_module(kspgpu)
                    kdiff = ldc_img_gpu - kspgpu
                    ksperror = sp.to_device(xp.linalg.norm(kdiff), -1)
                    kspnorm = sp.to_device(xp.linalg.norm(kspgpu), -1)
                else:
                    ksperror = np.nan
                    kspnorm = 1

            t1error = t1map - phantomt1
            t1errorrel = t1error / phantomt1
            t2error = (t2map - phantomt2)
            t2errorrel = t2error / phantomt2

            meanresult[j, i, 0] = np.sqrt(np.nanmean(pderror[mask] ** 2))
            meanresult[j, i, 1] = np.sqrt(np.nanmean(t1error[mask] ** 2))
            meanresult[j, i, 2] = np.sqrt(np.nanmean(t2error[mask] ** 2))
            meanresult[j, i, 3] = ksperror
            relmeanresult[j, i, 0] = np.sqrt(np.nanmean(pderrorrel[mask] ** 2))
            relmeanresult[j, i, 1] = np.sqrt(np.nanmean(t1errorrel[mask] ** 2))
            relmeanresult[j, i, 2] = np.sqrt(np.nanmean(t2errorrel[mask] ** 2))
            relmeanresult[j, i, 3] = ksperror / kspnorm
            stdresult[j, i, 0] = np.abs(np.nanstd(pderrorrel[mask]))
            stdresult[j, i, 1] = np.abs(np.nanstd(t1errorrel[mask]))
            stdresult[j, i, 2] = np.abs(np.nanstd(t2errorrel[mask]))
            # stdresult[j, i, 3] = np.abs(np.std(ksperror))/kspnorm
            print('Finished run {}, admm iter {}'.format(j, i))
            if save:
                with h5py.File(settings['phantom_output_path'] + "run{}".format(j + 1) + "result.h5", 'w') as hf:
                    hf.create_dataset('relmean', data=relmeanresult)
                    hf.create_dataset('mean', data=meanresult)
                    hf.create_dataset('std', data=stdresult)
    if save:
        with h5py.File(settings['phantom_output_path'] + "result.h5", 'w') as hf:
            hf.create_dataset('relmean', data=relmeanresult)
            hf.create_dataset('mean', data=meanresult)
            hf.create_dataset('std', data=stdresult)
    return meanresult
    """
    plt.subplot(2, 2, 3)
    plt.plot(meanresult[..., 0].T)
    plt.legend(admm_params)
    plt.title("proton density")
    plt.subplot(2, 2, 1)
    plt.plot(meanresult[..., 1].T)
    plt.legend(admm_params)
    plt.title("T1")
    plt.subplot(2, 2, 2)
    plt.plot(meanresult[..., 2].T)
    plt.legend(admm_params)
    plt.title("T2")
    plt.subplot(2, 2, 4)
    plt.plot(meanresult[..., 3].T)
    plt.legend(admm_params)
    plt.title("||GFSDc-y||")
    plt.show()
    """


if __name__ == "__main__":
    # data_loc = '/tudelft.net/'
    # configpath = 'U:/mrflumc/Results_Emiel/admm_tuning_5mm/'
    # settings = load_settings(configpath, 0)
    admm_convergence('')
    # showcomponents(r'output/in_vivo2/extend_dict/1sl_1_32/test_T400_rp25/')
    # showimgseq(r'output/in_vivo2/extend_dict/1sl_1_32/test_T400_rp25/')
    # bulk_postpro(r'output/in_vivo2/extend_dict_pdhg/1sl_1_32/')
    # bulk_postpro(r'output/in_vivo2/extend_dict_pdhg/1sl_5_32/')
    # bulk_postpro(r'output/in_vivo/extend_dict_pdhg/1sl_1_32/')
    # bulk_postpro(r'output/in_vivo/extend_dict_pdhg/1sl_5_32/')
    # bulk_postpro(r'output/in_vivo2/extend2/1sl_5_32/')
