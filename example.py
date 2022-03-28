"""
Example script to show how to use the
"""

import h5py

from MRFrecon import MrfData

# %% Load data
# Load data from numerical phantom
with h5py.File(r'example_data/num_phantom.h5', 'r') as hf:
    ksp = hf.get('kspace')[:]
    coord = hf.get('coord')[:]

# Load dictionary
with h5py.File(r'example_data/dictionary.h5', 'r') as hf:
    dictmat = hf.get('dictmat')[:].T.imag
    dictt1 = hf.get('t1list')[:]
    dictt2 = hf.get('t2list')[:]

# %% Create the data class
data = MrfData()
data.ksp = ksp
data.coord = coord
data.dictmat = dictmat
data.dictt1 = dictt1
data.dictt2 = dictt2

# equivalent to:
# data = MrfData(ksp, coord, dictmat, dictt1, dictt2)

# make 2 copies
data2 = data.copy()
data3 = data.copy()

# %% Solve LR inversion with single component matching
# very similar to (slightly different settings)
#       recon_lr_invert(data, settings=load_settings('./example_data/config.ini',0))

# compress dictionary to rank 10
data.compress_dictionary(10)

compute_device = -1
# Espirit calc for sensitivity maps, with reconstruction matrix of size 256
data.mrf_espirit(256, compute_device=compute_device, tol_fac=.005, )

# compress data to 5 virtual coils
data.coil_compression(5)

# Low rank inversion image reconstruction
data.lr_inversion(compute_device=compute_device, tol_fac=.005)

# rotate image sequence to real axis
data.rotate2real()

# Solve for single component MRF
data.single_component_match(verbose=2)

# Save to .h5
data.to_h5('./example_data/', 'lr_inv_single.h5')

# %% Solve LR ADMM with SPIJN component reconstruction
# very similar to (slightly different settings)
# recon_admm_into_spijn(data, settings=load_settings('./example_data/config.ini',2))

# compress dictionary to rank 10
data2.compress_dictionary(10)

# Espirit calc for sensitivity maps, with reconstruction matrix of size 256
data2.mrf_espirit(256, compute_device=compute_device)

# compress data to 5 virtual coils
data2.coil_compression(5)

# Low rank admm image reconstruction
data2.lr_admm(2e-3, compute_device=compute_device)
# Note: low rank admm does not need rotate2real()!

# Solve for components with a joint sparse regularization parameter of 0.15
data2.spijn_solve(0.15, verbose=2)

# Save to .h5
data2.to_h5('./example_data/', 'lr_admm_spijn.h5')

# %% Solve direct reconstruction
# very similar to (slightly different settings)
# recon_admm_into_spijn(data, settings=load_settings('./example_data/config.ini',1))
# compress dictionary to rank 10
data3.compress_dictionary(10)

# Espirit calc for sensitivity maps, with reconstruction matrix of size 256
data3.mrf_espirit(256, compute_device=compute_device)

# compress data to 5 virtual coils
data3.coil_compression(5)

# Direct component reconstruction
data3.spijn_from_ksp(admm_param=2e-3, reg_param=0.25, compute_device=compute_device, verbose=2)

# Save to .h5
data3.to_h5('./example_data/', 'direct.h5')
