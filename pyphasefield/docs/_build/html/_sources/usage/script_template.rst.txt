How to use pyphasefield: Script Template
========================================

Pyphasefield is structured around Engine python files. These files contain a subclass of ``simulation`` along with code to run 
a single simulation step. This subclass is then utilized in a python script by calling [Simulation].simulate(number_of_steps). In 
this document, we will go through all the commands required to utilize these engines through an example script file, which may be 
run either as a standalone python script, or in code cells within a jupyter notebook. The code, in both .py and .ipynb formats, may 
be downloaded :download:`here <script_template.zip>`.

.. warning::
	The below script is, as written, non-functional. It is merely an illustration of all the functions which are typically called 
	to run a simulation, and must be edited from its present form to run a simulation. In particular, the script references a path 
	to a TDB file, however no TDB file actually exists at that location in the :download:`downloadable code <script_template.zip>`. 
	Later tutorial pages have functional scripts which may be run as-is with no modification.

The Code
~~~~~~~~

.. code-block:: python

	import pyphasefield as ppf
	import pyphasefield.Engines as engines

	tdbc = ppf.TDBContainer("Ni-Cu_Ideal.tdb", ["FCC_A1", "LIQUID"], ["CU", "NI"])

	sim = engines.Diffusion(dimensions=[200, 200, 200])

	#initialize non-array parameters
	sim.set_framework("CPU_SERIAL")
	sim.set_dx(1.)
	sim.set_dt(0.1)
	sim.set_time_step_counter(0)
	sim.set_temperature_type("ISOTHERMAL")
	sim.set_temperature_initial_T(1584.)
	sim.set_temperature_dTdx(100000.)
	sim.set_temperature_dTdy(0.)
	sim.set_temperature_dTdz(0.)
	sim.set_temperature_dTdt(-3000000.)
	sim.set_temperature_dTdt(0.)
	sim.set_temperature_path("T.hdf5")
	sim.set_t_file_offset_cells([4000, 8000, 10000]) #this or below
	sim.set_t_file_offset_units([0.00025, 0.00008, 0.00043]) #this or above
	sim.set_t_file_min(1500.)
	sim.set_t_file_max(2800.)
	sim.set_temperature_units("K")
	sim.set_tdb_container(tdbc)
	sim.set_tdb_path("Ni-Cu_Ideal.tdb")
	sim.set_tdb_phases(["FCC_A1", "LIQUID"])
	sim.set_tdb_components(["CU", "NI"])
	sim.set_save_path("data/test_simulation")
	sim.set_autosave_flag(True)
	sim.set_autosave_save_images_flag(False)
	sim.set_autosave_rate(40000)
	sim.set_boundary_conditions("NEUMANN")
	sim.set_ghost_rows(32)

	data = {
		
	}
	sim.set_user_data(data)

	#initialize simulation arrays, all parameter changes should be BEFORE this point!
	sim.initialize_engine()

	#change array data here, for custom simulations

	#run simulation
	for i in range(1):
	    sim.simulate(1000)
	    sim.plot_simulation()


Description of Functions
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

	tdbc = ppf.TDBContainer("Ni-Cu_Ideal.tdb", ["FCC_A1", "LIQUID"], ["CU", "NI"])

TDBContainer is an object which stores ufuncs which phase field code uses to extract thermodynamic information about a simulation. 
This is done so that multiple simulations can re-use the same thermodynamic conditions, as for extremely large TDB files in can take 
upwards of a minute to load all the information into pycalphad. The parameters are:
* relative path to the TDB file, with the script being the current directory
* list of phases to use from the TDB file. 
* List of components to use from the TDB file.

.. code-block:: python

	sim = engines.Diffusion(dimensions=[200, 200, 200])

This creates a Simulation object, sim (in this example, it is the subclass Diffusion). The required parameter, dimensions, is a list of 
integer values dictating the dimensionality (number of values in the list) and size of the simulation box. Follows C-style array conventions (z, y, x ordering).

.. code-block:: python

	sim.set_framework("CPU_SERIAL")

This sets what compute framework the simulation will use. Currently implemented values are "CPU_SERIAL", for code running on a CPU like normal, 
and "GPU_SERIAL", for code running on a single GPU (requires numba!)

.. code-block:: python

	sim.set_dx(1.)

This sets the size of a single cell in the simulation. Pyphasefield uses a square regular grid, with the length of each edge of a cell equal to dx. 
Typically, this is in units of meters or centimeters (depends on the model!).
	
.. code-block:: python

	sim.set_dt(0.1)

This sets the length of a time step in the simulation. Some engines will set this value automatically, to prevent instability due to a large timestep 
in an explicit scheme simulation. Typically, this is in units of seconds.
	
.. code-block:: python

	sim.set_time_step_counter(0)

This sets what time step the simulation will have reached at the start of running the simulation. Typically 0, unless running from the middle 
of a thermal file (discussed later).
	
.. code-block:: python

	sim.set_temperature_type("ISOTHERMAL")

This sets what type of thermal conditions are present in the simulation. Simulations have a special field called sim.temperature, and the data 
contained within (numpy ndarray) may be accessed using sim.temperature.data. Possible values include:

* None: Used if a simulation does not use temperature, or if you wish to have temperature evolve using phase field equations (in which case, 
  temperature would just be another field in sim.fields)
* "ISOTHERMAL": Used if the overall temperature does not change during a simulation. Only sim.set_temperature_initial_T() need be used
* "LINEAR_GRADIENT": Used if the simulation has a simple linear gradient in either space or time or both. sim.set_temperature_initial_T() will set the 
  temperature of the cell [0, 0, 0, ...], while sim.set_temperature_dTdx(), sim.set_temperature_dTdy(), sim.set_temperature_dTdz() will set the thermal 
  gradient in the x, y, and z dimensions as necessary, while sim.set_temperature_dTdt() will set how all cells change in temperature over time.
* "THERMAL_HISTORY_FILE": Used if you wish to pull thermal data from a file (for example, if you generate the thermal field using a different simulation scheme, 
  like FEM!). Only sim.set_temperature_path() has to be used for this case.
	
.. code-block:: python

	sim.set_temperature_initial_T(1584.)
	
Sets the temperature of the entire simulation (isothermal) or the initial temperature of the origin (linear gradient). Typically in units of K.
	
.. code-block:: python

	sim.set_temperature_dTdx(100000.)
	sim.set_temperature_dTdy(0.)
	sim.set_temperature_dTdz(0.)

Sets the thermal gradient along each dimension. This is in units of temperature/length, based on units of dx (e.g. K/m or K/cm). 

.. code-block:: python

	sim.set_temperature_dTdt(-3000000.)
	
Sets the change in temperature over time, often called the cooling rate. Note that while cooling is more commonly used, dTdt must be negative 
to result in cooling over time! Typically in units of K/s.

.. code-block:: python

	sim.set_temperature_path("T.hdf5")
	
Sets the relative path to an HDF5 file, to be used to pull thermal data from. The current location of the script (or jupyter notebook) is the 
current directory in this case. XDMF files might also work, but may not function properly with parallelism. We strongly recommend converting XDMF 
files into HDF5 files for optimal compatibility.
	
.. code-block:: python

	sim.set_t_file_offset_cells([4000, 8000, 10000])

Sets the location in the thermal history file to use as an origin, in units of number of cells. 

.. code-block:: python

	sim.set_t_file_offset_units([0.00025, 0.00008, 0.00043])

Sets the location in the thermal history file to use as an origin, in the same units as dx. This function requires that dx has already been set!

.. code-block:: python

	sim.set_t_file_min(1500.)
    
Clamps values of the thermal history file to be no smaller than this value. (Engines may not behave correctly with regions far below melting). 
In the same units as the thermal history file, which typically should be K.

.. code-block:: python

	sim.set_t_file_max(1500.)
    
Clamps values of the thermal history file to be no larger than this value. (Engines may not behave correctly with regions far above melting either). 
In the same units as the thermal history file, which typically should be K.

.. code-block:: python

	sim.set_temperature_units("K")

Mostly unused at present, this modifies the name of the temperature field, so if you wish to print out an image of the thermal field using 
matplotlib, it will label the field automatically.
	
.. code-block:: python

	sim.set_tdb_container(tdbc)
	
One possible method for setting the TDB thermodynamics for a simulation, loading the information from a previously initialized TDBContainer object.
	
.. code-block:: python

	sim.set_tdb_path("Ni-Cu_Ideal.tdb")
	sim.set_tdb_phases(["FCC_A1", "LIQUID"])
	sim.set_tdb_components(["CU", "NI"])
	
Another possible method, giving the path, phases, and components directly to the simulation.
	
.. code-block:: python

	sim.set_save_path("data/test_simulation")
	
Sets the path to where the simulation will save checkpoint files (and images if desired). Path is relative, with the script as the current directory
	
.. code-block:: python

	sim.set_autosave_flag(True)
	
Flag that sets whether a simulation will automatically save a checkpoint file every few steps
	
.. code-block:: python

	sim.set_autosave_save_images_flag(False)
	
Flag that sets whether a simulation will automatically save images of the fields every few steps


.. code-block:: python

	sim.set_autosave_rate(40000)
	
Sets how often a simulation will automatically save checkpoints/images, if the respective flags are set. (Saves every X timesteps).
	
.. code-block:: python

	sim.set_boundary_conditions("NEUMANN")
	
Sets the boundary conditions for a given sim. Options are as follows:

* "PERIODIC": Periodic boundary conditions
* "NEUMANN": Neumann (defined gradient at boundary) boundary conditions
* "DIRICHLET": Dirichlet (defined value at boundary) boundary conditions
* ["PERIODIC", "NEUMANN", "DIRICHLET"]: A list defines different boundary conditions along each dimensions. In this case, it would have periodic boundary conditions along 
  the z axis, Neumann boundary conditions along the y axis, and Dirichlet boundary conditions along the x axis. (C-style array convention is z, y, x).
  Other permutations of the previous three values are also permitted. The list should have length equal to the number of dimensions of the simulation.
* [["DIRICHLET", "NEUMANN"], ["DIRICHLET", "NEUMANN"], ["DIRICHLET", "NEUMANN"]]: A 2-D list defines the boundary condition on the left and right side of each dimension 
  separately. In this case, the left (low-valued index) side would have Dirichlet boundary conditions, while the right (high-valued index) side would have Neumann BCs.
  The list should have shape equal to [D, 2], where D is the number of dimensions of the simulation.
  
The values for Neumann and Dirichlet boundary conditions are located in sim.boundary_fields. Separate documentation will describe how to use these boundaries to their 
fullest extent. By default, these boundaries have a value of zero, corresponding to Neumann conditions being no-flux (zero gradient across the boundary), and to 
Dirichlet conditions enforcing a value of zero on the border (zero value at the boundary).
  
.. code-block:: python

	sim.set_ghost_rows(32)

Sets the number of ghost rows to be used for a parallel simulation. This increases the number of calculations that must be done, but also means that processes 
only have to communicate with each other every X timesteps, due to these redundant calculations. For most practical simulation domain sizes run on multiple GPUs, 
a value of 32 is close to optimal, but feel free to experiment!

.. code-block:: python

	data = {
		
	}
	sim.set_user_data(data)
	
Sets engine-specific parameters, using a dictionary. For example, the Diffusion engine has one engine-specific parameter, D, which may be seen in the 
diffusion example code
	
.. code-block:: python

	sim.initialize_engine()
	
This function initializes the size and content of fields based on the previously defined parameters, as well as initializes boundary conditions, 
loads data from TDB files, and creates and initializes the temperature field. Any changes you wish to make to field data must be done after 
calling this function!
	
.. code-block:: python

	for i in range(1):
	    sim.simulate(1000)
	    sim.plot_simulation()
		
Runs a simulation for 1000 time steps, then plots the fields using matplotlib to show the progress.
