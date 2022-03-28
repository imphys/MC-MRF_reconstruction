This repository can be used to reconstruct MR Fingerprinting data and perform single and multi-component matching.

This code heavily relies on `SigPy <https://sigpy.readthedocs.io/en/latest/>`_, so many thanks to the developers of this great package!  

============
MrfData
============
``MRFrecon\Backend.py`` contains a python class ``MrfData`` to reconstruct MRF data and calculate multi-component maps using joint sparsity [1]_ resulting in a small number of tissues over the complete region of interest.

The class can:

* reconstruct MRF image sequence from kspace data using:
    * Filtered back projection
    * Low rank Filtered back projection comparable to [2]_
    * Parallel image reconstruction
    * Low rank inversion image reconstruction
    * Low rank ADMM image reconstruction

* calculate quantitative data
    * Single component matching
    * SPIJN (joint sparse components from an image sequence)
    * Direct joint sparse components from kspace

------------
Installation
------------
The package requires several packages

* Numpy
* Sigpy
* Matplotlib
* h5py
* Scipy
* Cupy (optional)
* Nibabel (optional)

``environment.yml`` contains these packages.


-----
Usage
-----
Load the data class from the module

.. code-block:: Python

    from MRFrecon import MrfData

Create an empty instance of the data class and add the raw mri data

.. code-block:: Python

    data = MrfData()
    data.ksp = ksp
    data.coord = coord
    data.dictmat = dictmat
    data.dictt1 = dictt1
    data.dictt2 = dictt2

    # equivalently
    data = MrfData(ksp, coord, dictmat, dictt1, dictt2)

Note the structure of the variables:

* ksp: numpy array of shape (#slices, #coils, #dynamics, #spirals_per_image, #point_on_spiral)
* coord: numpy array of shape (#dynamics, #spirals_per_image, #point_on_spiral, #spiral_coordinates)
* dictmat: numpy array of shape (#atoms, #dynamics)
* t1list: list of :math:`T_1` values of shape (#atoms)
* t2list: list of :math:`T_2` values of shape (#atoms)

With the data initialized, the class can solve for the image sequence and components.
The following code solves for the joint sparse components after low rank inversion image reconstruction.

.. code-block:: Python

    # compress dictionary
    data.compress_dictionary(10)

    # Espirit calc for sensitivity maps
    data.mrf_espirit(256)

    # compress coils
    data.coil_compression(5)

    # Low rank inversion image reconstruction
    data.lr_inversion()

    # rotate image sequence to real axis
    data.rotate2real()

    # Solve for components
    data.spijn_solve(0.25)

    # Save to .h5
    data.to_h5('./')


A working version of this is contained in ``example.py``.
First download the data from location as specified in ``example_data/readme.txt``.


-----
Usage with config.ini files
-----
To smoothen this procedure and have more reusable methods or perform reconstruction with different settings,
a configparser has been used to use different config files (see example data) together with ``main_from_config_files.py`` this
allows to run the reconstruction from the terminal with predefined reconstruction steps as contained in ``mrf_recon.py``.
This is especially usefull on a cluster with a slurm job manager and the array command.

-------
References
-------
.. [1] Nagtegaal, M, Koken, P, Amthor, T, et al. Fast multi-component analysis using a joint sparsity constraint for MR fingerprinting. Magn Reson Med. 2020; 83: 521– 534. https://doi.org/10.1002/mrm.27947 
.. [2] Assländer, J., Cloos, M.A., Knoll, F., Sodickson, D.K., Hennig, J. and Lattanzi, R. (2018), Low rank alternating direction method of multipliers reconstruction for MR fingerprinting. Magn. Reson. Med., 79: 83-96. https://doi.org/10.1002/mrm.26639

-------
Contact
-------
\(c\) Emiel Hartsema, July 2021

\(c\) Martijn Nagtegaal, March 2022

Technical University of Delft, Faculty of Applied Sciences

emiel@hartsema.com

M.A.Nagtegaal@tudelft.nl

.. image:: https://camo.githubusercontent.com/4fde808ab45b0f7ec5763d9daf2e96192c9ca859792fd4531f86ace05da08230/68747470733a2f2f6431726b616237746c71793566312e636c6f756466726f6e742e6e65742f5f70726f6365737365645f2f362f312f63736d5f496d506879732d6c6f676f5f6d657425323074656b73745f643037366135636437362e706e67
    :alt: Imphys

\

.. image:: https://seeklogo.com/images/T/TU_Delft-logo-D6086E1A70-seeklogo.com.png
    :alt: TUDelft
