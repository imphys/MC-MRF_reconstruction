"""
    Author: Emiel Hartsema, Martijn Nagtegaal
    Institution: TU Delft
    Python version: 3.7
"""
import os
import time
import warnings

from MRFrecon import load_data
from MRFrecon import load_settings
from MRFrecon.mrf_recon import recon_fbp, recon_lr_invert, recon_lr_admm, recon_sense, \
    recon_lr_invert_spijn, recon_admm_into_spijn, recon_sense_into_spijn, recon_direct_spijn, recon_test, \
    recon_wavelet_into_spijn, recon_wavelet


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name, )
        print('Elapsed time total calculation, poeh poeh: %.2f seconds' % (time.time() - self.tstart))


if __name__ == "__main__":
    settings = load_settings()
    data = load_data(settings)

    os.makedirs(settings['reconstruction_output_path'], exist_ok=True)

    # Run reconstruction algorithm
    if settings['recon_alg'] == 'fbp':
        with Timer('fbp'):
            recon_fbp(data, settings)
    elif settings['recon_alg'] == 'invert':
        with Timer(settings['recon_alg']):
            recon_lr_invert(data, settings)
    elif settings['recon_alg'] == 'admm':
        with Timer(settings['recon_alg']):
            recon_lr_admm(data, settings)
    elif settings['recon_alg'] == 'sense':
        with Timer(settings['recon_alg']):
            recon_sense(data, settings)
    elif settings['recon_alg'] == 'invert_into_spijn':
        with Timer(settings['recon_alg']):
            recon_lr_invert_spijn(data, settings)
    elif settings['recon_alg'] == 'admm_into_spijn':
        with Timer(settings['recon_alg']):
            recon_admm_into_spijn(data, settings)
    elif settings['recon_alg'] == 'sense_into_spijn':
        with Timer(settings['recon_alg']):
            recon_sense_into_spijn(data, settings)
    elif settings['recon_alg'] == 'wavelet':
        with Timer(settings['recon_alg']):
            recon_wavelet(data, settings)
    elif settings['recon_alg'] == 'wavelet_into_spijn':
        with Timer(settings['recon_alg']):
            recon_wavelet_into_spijn(data, settings)
    elif settings['recon_alg'] == 'direct_spijn':
        with Timer(settings['recon_alg']):
            recon_direct_spijn(data, settings)
    elif settings['recon_alg'] == 'test':
        with Timer(settings['recon_alg']):
            recon_test(data, settings)
    else:
        warnings.warn("Skip to next run; Reconstruction algorithm not identified in [{}]. Got [{}] excpected "
                      "filt_back_proj, lr_filt_back_proj, invert, invert_spijn or "
                      "admm_spijn".format(settings['path_extension'], settings['recon_alg']))
    print('Done')
