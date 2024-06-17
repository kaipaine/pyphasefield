Installation Instructions
=========================

Pyphasefield is a pure-python package and thus may be installed through `PyPI`_ using the command ``pip install pyphasefield``. 
However, it is **strongly** recommended to install pyphasefield into an Anaconda_ installation, as this permits the easy use of the package `numba`_ 
which enables running accelerated phase field simulations using General Purpose GPU computing. 

Anaconda Installation Instructions
----------------------------------
This tutorial assumes you have installed Anaconda_ and (optionally) created and activated an `Anaconda environment`_ to contain these 
packages, separate from the base installation. After opening the Anaconda terminal ("Anaconda Prompt" on Windows) and activating your 
environment if necessary, run the following commands to install pyphasefield, as well as recommended (but not required) packages:

* ``pip install pyphasefield``: Installs pyphasefield alone with no dependencies
* ``python -m pyphasefield``: Runs the pyphasefield installation script, to install all dependencies

These dependencies include (among others):

* Numba/Cudatoolkit: enables python to interface with the GPU
* mpi4py: enables parallel simulations using multiple GPUs (if multiple GPUs are available - like on supercomputer clusters)
* pycalphad: enables interfacing with TDB (Thermodynamic DataBase) files for arbitrary thermodynamics to be used in phase field
* h5py: enables saving/loading HDF5 files. Required, but optionally builds against a parallel hdf5 distribution for parallel sims

.. warning::
	If you are using an Anaconda environment, Jupyter notebook may not detect packages installed in the environment by default. To ensure 
	this does not lead to "Module not found" errors, run the following commands to install the packages in the current environment into 
	a new kernel for Jupyter notebook (`Source <https://stackoverflow.com/questions/33960051/unable-to-import-a-module-from-python-notebook-in-jupyter>`_):
	
	* ``conda install notebook ipykernel``
	* ``ipython kernel install --user``
	
	Running the installation script can create a jupyter kernel for you automatically, which may negate the need to do this.





.. _PyPI: https://pypi.org/
.. _Anaconda: https://www.anaconda.com/products/individual
.. _pycalphad: https://pycalphad.org/docs/latest/
.. _numba: http://numba.pydata.org/
.. _`Anaconda environment`: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html