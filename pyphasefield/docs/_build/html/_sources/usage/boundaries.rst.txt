How to use pyphasefield: Boundary Conditions
============================================

In this example, we will show how to fully utilize boundary conditions. Let us begin with the previous Diffusion example, 
but with the field equal to one everywhere. In the absence of boundary conditions, nothing should happen. 

The Starting Code (don't use this!)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

	import pyphasefield.Engines as engines

	sim = engines.Diffusion(dimensions=[200, 200])

	#initialize non-array parameters
	sim.set_framework("CPU_SERIAL") #"CPU_SERIAL", "GPU_SERIAL" (GPU_SERIAL requires numba)
	sim.set_dx(1.)
	sim.set_dt(0.1)
	sim.set_save_path("data/boundary_conditions_test")
	sim.set_autosave_flag(False)
	sim.set_boundary_conditions("PERIODIC")

	data = {
		"D":1.
	}
	sim.set_user_data(data)

	#initialize simulation arrays, all parameter changes should be BEFORE this point!
	sim.initialize_engine()

	#change array data here, for custom simulations
	
	#set the field values manually using numpy slicing, setting it to one everywhere
	sim.fields[0].data[:] = 1.
	


	#initial conditions
	sim.plot_simulation()

	#run simulation
	sim.simulate(2000)

	#final conditions
	sim.plot_simulation()
	

In order to set what type of boundary conditions, we can modify the function sim.set_boundary_conditions(). As previously 
mentioned in the script template example, some options are:

* "PERIODIC": Periodic boundary conditions
* "NEUMANN": Neumann (defined gradient at boundary) boundary conditions
* "DIRICHLET": Dirichlet (defined value at boundary) boundary conditions
* ["PERIODIC", "NEUMANN"]: A list defines different boundary conditions along each dimensions. In this case, it would have periodic boundary conditions along 
  the z axis, Neumann boundary conditions along the y axis, and Dirichlet boundary conditions along the x axis. (C-style array convention is y, x).
  Other permutations of the previous three values are also permitted. The list should have length equal to the number of dimensions of the simulation.
* [["DIRICHLET", "NEUMANN"], ["DIRICHLET", "NEUMANN"]]: A 2-D list defines the boundary condition on the left and right side of each dimension 
  separately. In this case, the left (low-valued index) side would have Dirichlet boundary conditions, while the right (high-valued index) side would have Neumann BCs.
  The list should have shape equal to [D, 2], where D is the number of dimensions of the simulation.
  
For this example, let's use that last case, [["DIRICHLET", "NEUMANN"], ["DIRICHLET", "NEUMANN"]].

Unlike periodic boundary conditions, Dirichlet and Neumann boundary conditions are parameterized: there is some value that is set at the interface for dirichlet boundaries, 
and some value that is set as the gradient at the interface for neumann boundaries. These values can be set in pyphasefield using the sim.boundary_fields object, which is a 
list of Field instances. Similarly to sim.fields, values of the boundary field may be accessed using sim.boundary_fields[0].data, replacing 0 with the index of the Field you 
wish to access.

Of note: only the values on the surface of the boundary field will matter, as those values say what the value of the parameter (Dirichlet: value or Neumann: gradient) is at 
that point on the boundary. Additionally, values at the "corners" of the boundary fields are not independent! 

For this example, we use changes to this interface parameter to illustrate what it does. We specify the boundary field edges in the following way:

* Dirichlet on left-side of X-axis: split into four pieces, two unchanged (default behavior: boundary value 0), two with boundary value 2
* Dirichlet on left-side of Y-axis: split into three pieces, two with boundary value 1, with the center being boundary value 4
* Neumann on right-side of X-axis: no change to the boundary field, remains at its default behavior (zero gradient = no flux boundary condition)
* Neumann on right-side of Y-axis: split into five pieces, three with a gradient of 0.05, three with a gradient of -0.05.

We also show a far future snapshot to illustrate the nature of these simulations. Simulations are perfectly ok with values being negative, even if negative concentration is 
unphysical in real life!

The Final Code (use this!)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

	import pyphasefield.Engines as engines

	sim = engines.Diffusion(dimensions=[200, 200])

	#initialize non-array parameters
	sim.set_framework("CPU_SERIAL") #"CPU_SERIAL", "GPU_SERIAL" (GPU_SERIAL requires numba)
	sim.set_dx(1.)
	sim.set_dt(0.2)
	sim.set_save_path("data/boundary_conditions_test")
	sim.set_autosave_flag(False)
	sim.set_boundary_conditions([["DIRICHLET", "NEUMANN"], ["DIRICHLET", "NEUMANN"]])

	data = {
		"D":1.
	}
	sim.set_user_data(data)

	#initialize simulation arrays, all parameter changes should be BEFORE this point!
	sim.initialize_engine()

	#change array data here, for custom simulations

	#set the field values manually using numpy slicing, setting it to one everywhere
	sim.fields[0].data[:] = 1.

	#set the boundary fields!
	#X-axis Dirichlet, commented lines are default behavior
	#sim.boundary_fields[0].data[:50, 0] = 0.
	sim.boundary_fields[0].data[50:100, 0] = 2.
	#sim.boundary_fields[0].data[100:150, 0] = 0.
	sim.boundary_fields[0].data[150:, 0] = 2.

	#Y-axis Dirichlet
	sim.boundary_fields[0].data[0, :66] = 1.
	sim.boundary_fields[0].data[0, 133:] = 1.
	sim.boundary_fields[0].data[0, 66:133] = 4.

	#Neumann X-axis has no change, equivalent to the bottom line
	#sim.boundary_fields[0].data[:, -1] = 0.

	#Neumann Y-axis
	sim.boundary_fields[0].data[-1, :40] = 0.05
	sim.boundary_fields[0].data[-1, 40:80] = -0.05
	sim.boundary_fields[0].data[-1, 80:120] = 0.05
	sim.boundary_fields[0].data[-1, 120:160] = -0.05
	sim.boundary_fields[0].data[-1, 160:] = 0.05

	#initial conditions
	sim.plot_simulation()

	#run simulation
	sim.simulate(2000)

	#final conditions
	sim.plot_simulation()

	#run simulation
	sim.simulate(48000)

	#final conditions
	sim.plot_simulation()

Results
~~~~~~~

.. image:: boundary0.png

.. image:: boundary2000.png

.. image:: boundary50000.png

Downloads
~~~~~~~~~
:download:`Jupyter Notebook <boundaries_notebook.ipynb>`