"""
    Filename: mrf_recon.py
    Author: Emiel Hartsema, Martijn Nagtegaal
    Date last modified: 21-10-2021
    Python version: 3.7
    Description: reconstruction algorithms to reconstruct mrf maps
"""
import numpy as np

from .backend import MrfData

max_power_iter_def = 15


def reg_params(settings):
    res = {'regtype': settings.get('regtype', None), 'lam': settings.getfloat('regularization_lambda', None),
           'lamscale': settings.getfloat('regularization_lambda_scale', 1),
           'lambda_ksp': settings.getboolean('regularization_lambda_ksp', False)}
    return res


def espirit_settings(settings):
    return dict(imagesize=settings.getint('imagesize'), coil_batchsize=settings.getint('espirit_coil_batchsize'),
                max_iter=settings.getint('max_espirit_cg_iter'), oversamp=settings.getfloat('oversamp_ratio'),
                compute_device=settings.getint('compute_device'),
                mask_thresh=settings.getfloat('espirit_mask_threshhold'), verbose=settings.getint('verbose'),
                tol_fac=settings.getfloat('tol_fac'),
                )


def recon_settings(settings):
    return dict(batchsize=settings.getint('lr_sense_batchsize'),
                oversamp=settings.getfloat('oversamp_ratio'),
                compute_device=settings.getint('compute_device'),
                verbose=settings.getint('verbose'),
                lstsq_solver=settings.get('lstsq_solver'),
                tol_fac=settings.getfloat('tol_fac'),
                norm_ksp=settings.getboolean('norm_kspace', True),
                max_power_iter=settings.getint('max_power_iter', max_power_iter_def),
                reg_kwargs=reg_params(settings)
                )


def recon_fbp(data, settings):
    print("Start filtered back projection recon with single component matching")
    data.filt_back_proj(settings.getint('imagesize'), compute_device=settings.getint('compute_device'),
                        oversamp=settings.getfloat('oversamp_ratio'), verbose=settings.getint('verbose'))

    # rrs image sequence
    data.imgseq = np.sum(np.abs(data.imgseq) ** 2, axis=1) ** 0.5
    data.compress_dict = data.dictmat
    data.maps = np.zeros((1, 224, 224))  # fake sensitivity maps in order to have the image size

    # Solve for components
    data.single_component_match(verbose=settings.getint('verbose'), absdict=True)

    # Save to .nii.gz
    data.save_single_comp_nii(settings['reconstruction_output_path'])

    # Save to .h5
    data.to_h5(settings['reconstruction_output_path'], save_raw=False, save_dict=False)


def recon_lr_invert(data, settings):
    print("Start Low rank inversion reconstruction with single component matching")
    # compress dictionary
    data.compress_dictionary(settings.getint('reconstruction_rank'), comp_device=settings.getint('compute_device'))

    # Espirit calc for sensitivity maps
    data.mrf_espirit(**espirit_settings(settings))

    # compress coils
    data.coil_compression(settings.getint('num_virt_coil'))

    # Low rank inversion image reconstruction
    data.lr_inversion(**recon_settings(settings),
                      max_iter=settings.getint('max_cg_iter'))

    # # rotate image sequence to real axis
    # data.rotate2real()

    # Solve for components
    data.single_component_match(verbose=settings.getint('verbose'), calc_nrmse=True)

    # Save to .nii.gz
    data.save_single_comp_nii(settings['reconstruction_output_path'])

    # Save to .h5
    data.to_h5(settings['reconstruction_output_path'], save_raw=False, save_dict=False)


def recon_lr_admm(data, settings):
    print("Start Low rank ADMM reconstruction with single component matching")
    # compress dictionary
    data.compress_dictionary(settings.getint('reconstruction_rank'), comp_device=settings.getint('compute_device'))

    # Espirit calc for sensitivity maps
    data.mrf_espirit(**espirit_settings(settings), )

    # compress coils
    data.coil_compression(settings.getint('num_virt_coil'))

    # Step for b1map processing, doesn't hurt if there is no b1map
    data.fixed_par_processing(redo=True, flatten=True)

    # calc admm image sequence
    data.lr_admm(admm_param=settings.getfloat('admm_param'),
                 max_iter=settings.getint('max_admm_iter'),
                 max_cg_iter=settings.getint('max_cg_iter'),
                 outpath=settings.get('admm_outpath'),
                 **recon_settings(settings))

    # Solve for components
    data.single_component_match(verbose=settings.getint('verbose'))

    # Save to .nii.gz
    data.save_single_comp_nii(settings['reconstruction_output_path'])

    # Save to .h5
    data.to_h5(settings['reconstruction_output_path'], save_raw=False, save_dict=False)


def recon_lr_invert_spijn(data, settings):
    print("Start lr inversion with SPIJN reconstruction")
    # compress dictionary
    data.compress_dictionary(settings.getint('reconstruction_rank'), comp_device=settings.getint('compute_device'))

    # Espirit calc for sensitivity maps
    data.mrf_espirit(**espirit_settings(settings))
    # compress coils
    data.coil_compression(settings.getint('num_virt_coil'))

    # Low rank inversion image reconstruction
    data.lr_inversion(
        max_iter=settings.getint('max_cg_iter'),
        **recon_settings(settings))
    # rotate image sequence to real axis
    data.rotate2real()

    # Step for b1map processing, doesn't hurt if there is no b1map
    data.fixed_par_processing(redo=True, flatten=True)

    # Solve for components
    data.spijn_solve(settings.getfloat('spijn_param'),
                     max_iter=settings.getint('max_spijn_iter'),
                     verbose=settings.getint('verbose'), norm_correction=False)

    # Save to .h5
    data.to_h5(settings['reconstruction_output_path'], save_raw=False, save_dict=False)


def recon_admm_into_spijn(data, settings):
    print("Start ADMM into SPIJN reconstruction")
    # compress dictionary
    data.compress_dictionary(settings.getint('reconstruction_rank'))

    # Espirit calc for sensitivity maps
    data.mrf_espirit(**espirit_settings(settings))

    # compress coils
    data.coil_compression(settings.getint('num_virt_coil'))

    # Step for b1map processing, doesn't hurt if there is no b1map
    data.fixed_par_processing(redo=True, flatten=True)

    # calc admm image sequence
    data.lr_admm(admm_param=settings.getfloat('admm_param'),
                 max_iter=settings.getint('max_admm_iter'),
                 max_cg_iter=settings.getint('max_cg_iter'),
                 outpath=settings.get('admm_outpath'),
                 **recon_settings(settings))

    # calc spijn after convergence of admm
    data.spijn_solve(settings.getfloat('spijn_param'), max_iter=settings.getint('max_spijn_iter'),
                     verbose=settings.getint('verbose'), norm_correction=False)

    # Save to .h5
    data.to_h5(settings['reconstruction_output_path'], save_raw=False, save_dict=False)


def recon_sense(data, settings):
    print("Start Sense recon with single component matching")

    # Espirit calc for sensitivity maps
    data.mrf_espirit(**espirit_settings(settings))

    # sense image reconstruction
    data.senserecon(compute_device=settings.getint('compute_device'),
                    verbose=settings.getint('verbose'),
                    tol_fac=settings.getfloat('tol_fac'))

    # store as lr image sequence for further processing
    data.compress_dictionary(settings.getint('reconstruction_rank'))
    data.imgseq_or = data.imgseq.copy()
    data.imgseq = data.imgseq @ data.comp_mat

    # data.rotate2real()

    # Step for b1map processing, doesn't hurt if there is no b1map
    data.fixed_par_processing(redo=True, flatten=True)

    # Solve for components
    data.single_component_match(verbose=settings.getint('verbose'))

    # Save to .nii.gz
    data.save_single_comp_nii(settings['reconstruction_output_path'])

    # Save to .h5
    data.to_h5(settings['reconstruction_output_path'], save_raw=False, save_dict=False)


def recon_sense_into_spijn(data, settings):
    print("Start sense with SPIJN reconstruction")

    # Espirit calc for sensitivity maps
    data.mrf_espirit(**espirit_settings(settings))

    # sense image reconstruction
    data.senserecon(compute_device=settings.getint('compute_device'),
                    verbose=settings.getint('verbose'),
                    tol_fac=settings.getfloat('tol_fac'))

    # store as lr image sequence for further processing
    data.compress_dictionary(settings.getint('reconstruction_rank'))
    data.imgseq = data.imgseq @ data.comp_mat

    data.rotate2real()

    # Step for b1map processing, doesn't hurt if there is no b1map
    data.fixed_par_processing(redo=True, flatten=True)

    # Solve for components
    data.spijn_solve(settings.getfloat('spijn_param'), max_iter=settings.getint('max_spijn_iter'),
                     verbose=settings.getint('verbose'), norm_correction=False)

    # Save to .h5
    data.to_h5(settings['reconstruction_output_path'], save_raw=False, save_dict=False)


def recon_wavelet(data, settings):
    print("Start sense-wavelet with single component matching")

    # Espirit calc for sensitivity maps
    data.mrf_espirit(**espirit_settings(settings))

    data.coil_compression(settings.getint('num_virt_coil'))

    # sense image reconstruction
    data.waveletrecon(lamda=settings.getfloat('regularization_lambda'),
                      compute_device=settings.getint('compute_device'), verbose=settings.getint('verbose'),
                      tol_fac=settings.getfloat('tol_fac'),
                      norm_ksp=settings.getboolean('norm_kspace', True),
                      )
    data.imgseq_or = data.imgseq.copy()

    # store as lr image sequence for further processing
    data.compress_dictionary(settings.getint('reconstruction_rank'))
    data.imgseq_or = data.imgseq.copy()
    data.imgseq = data.imgseq @ data.comp_mat

    # Step for b1map processing, doesn't hurt if there is no b1map
    data.fixed_par_processing(redo=True, flatten=True)

    # Solve for components
    data.single_component_match(verbose=settings.getint('verbose'))

    # Save to .nii.gz
    data.save_single_comp_nii(settings['reconstruction_output_path'])

    # Save to .h5
    data.to_h5(settings['reconstruction_output_path'], save_raw=False, save_dict=False)


def recon_wavelet_into_spijn(data, settings):
    print("Start sense-wavelet with SPIJN reconstruction")

    # Espirit calc for sensitivity maps
    data.mrf_espirit(**espirit_settings(settings))

    data.coil_compression(settings.getint('num_virt_coil'))

    # sense image reconstruction
    data.waveletrecon(lamda=settings.getfloat('regularization_lambda'),
                      compute_device=settings.getint('compute_device'), verbose=settings.getint('verbose'),
                      tol_fac=settings.getfloat('tol_fac'),
                      norm_ksp=settings.getboolean('norm_kspace', True),
                      )
    data.imgseq_or = data.imgseq.copy()

    # store as lr image sequence for further processing
    data.compress_dictionary(settings.getint('reconstruction_rank'))
    data.imgseq = data.imgseq @ data.comp_mat

    data.rotate2real()

    # Step for b1map processing, doesn't hurt if there is no b1map
    data.fixed_par_processing(redo=True, flatten=True)

    # Solve for components
    data.spijn_solve(settings.getfloat('spijn_param'), max_iter=settings.getint('max_spijn_iter'),
                     verbose=settings.getint('verbose'), norm_correction=False)

    # Save to .h5
    data.to_h5(settings['reconstruction_output_path'], save_raw=False, save_dict=False)


def recon_direct_spijn(data, settings):
    # compress dictionary
    data.compress_dictionary(settings.getint('reconstruction_rank'))

    # Espirit calc for sensitivity maps
    data.mrf_espirit(**espirit_settings(settings))

    # compress coils
    data.coil_compression(settings.getint('num_virt_coil'))

    # Step for b1map processing, doesn't hurt if there is no b1map
    data.fixed_par_processing(redo=True, flatten=True)

    data.spijn_from_ksp(admm_param=settings.getfloat('admm_param'),
                        max_admm_iter=settings.getint('max_admm_iter'),
                        max_cg_iter=settings.getint('max_cg_iter'),
                        max_iter=settings.getint('max_spijn_iter'),
                        reg_param=settings.getfloat('spijn_param'), norm_correction=False,
                        **recon_settings(settings)
                        )

    # Save to .h5
    data.to_h5(settings['reconstruction_output_path'], save_raw=False, save_dict=False)


def recon_test(data, settings):
    print("Recon direct recon followed by LR inv image reconstruction")
    # compress dictionary
    data.compress_dictionary(settings.getint('reconstruction_rank'))

    # Espirit calc for sensitivity maps
    data.mrf_espirit(**espirit_settings(settings))

    # compress coils
    data.coil_compression(settings.getint('num_virt_coil'))

    # Step for b1map processing, doesn't hurt if there is no b1map
    data.fixed_par_processing(redo=True, flatten=True)

    # Direct reconstruction
    data.spijn_from_ksp(admm_param=settings.getfloat('admm_param'), oversamp=settings.getfloat('oversamp_ratio'),
                        max_admm_iter=settings.getint('max_admm_iter'), max_cg_iter=settings.getint('max_cg_iter'),
                        max_iter=settings.getint('max_spijn_iter'), verbose=settings.getint('verbose'),
                        reg_param=settings.getfloat('spijn_param'), compute_device=settings.getint('compute_device'),
                        lstsq_solver=settings.get('lstsq_solver'),
                        tol_fac=settings.getfloat('tol_fac'),
                        max_power_iter=settings.getint('max_power_iter', max_power_iter_def))

    # Low rank inversion image reconstruction
    data.lr_inversion(batchsize=settings.getint('lr_sense_batchsize'), oversamp=settings.getfloat('oversamp_ratio'),
                      max_iter=settings.getint('max_cg_iter') + 50, compute_device=settings.getint('compute_device'),
                      verbose=settings.getint('verbose'), lstsq_solver=settings.get('lstsq_solver'),
                      tol_fac=settings.getfloat('tol_fac'), max_power_iter=settings.getint('max_power_iter',
                                                                                           max_power_iter_def))

    # rotate image sequence to real axis
    data.rotate2real()

    # Solve for components
    data.spijn_solve(0, max_iter=settings.getint('max_spijn_iter'),
                     verbose=settings.getint('verbose'), norm_correction=False)

    # Save to .h5
    data.to_h5(settings['reconstruction_output_path'], save_raw=False, save_dict=False)


if __name__ == "__main__":
    from config import settings_

    print("Loading data...")
    data_ = MrfData.from_h5(settings_['mri_data_path'])

    # recon_lr_invert(data, settings)
    recon_lr_invert_spijn(data_, settings_)
    # recon_admm_into_spijn(data, settings)
    # sense_into_spijn(data, settings)
    # recon_direct_spijn(data, settings)
