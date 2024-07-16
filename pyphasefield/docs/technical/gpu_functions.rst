Primer on GPU Integration
=========================

The Graphics Processing Unit (GPU) represents a significant fraction of the past 20 years of computer hardware improvements. 
While clock speed has remained plateaued at a few gigahertz, computer performance has still improved by creating many separate CPU 
cores which can each work at a few gigahertz. At a very high level, GPUs can be thought of as a few thousand CPU cores in a pretty 
box. It's not an exact analogy, but it's pretty close. In order to utilize the power of the GPU (around a thousand times more 
processing power than a single CPU core), we use the python library Numba. Another similar library is CuPy, which eventually will 
replace Numba in pyphasefield. Here's the information you need to know in order to make the jump from CPU to GPU, culminating 
with making our MyDiffusion class work on the GPU. 

Host vs. Device vs. Global
--------------------------

One very important takeaway is that we use a hybrid architecture for these engines. The high-performance number crunching takes 
place on the GPU, and the more high-level code, where speed is not as much of a priority, still takes place on the CPU like normal. 
So, it's important to keep track of what code is running where.

There are three main types of functions that are used in this architecture:

* Host functions: these are your normal, run of the mill python functions, which run on the CPU.
* Device functions: these are the equivalent of host functions, but instead run on the GPU. The CPU isn't even allowed to call these!
* Global functions: these are functions which are called from the CPU, but are executed on the GPU. Inside this global function is 
  where device functions may be called. One special restriction on global functions: they cannot "return" a value, any outputs from 
  a global function must be retrieved through a parameter (like an array) which is passed to the function. Global functions are 
  sometimes referred to as "kernels".
  
In addition, care must be paid to where memory (variables, arrays, and the like) exist. There are two categories:

* Host memory: these are your normal variables and arrays which exist on the CPU.
* Device memory: this memory only exists on the GPU, and cannot be accessed by the CPU! And similarly, any value you pass to either 
  a global or a device function must exist in device memory: the GPU does not have access to host memory!

Here is an example of all three types of functions, along with an example of transferring memory between CPU, GPU, and back.

.. code-block:: python

	import numba
	from numba import cuda
	import numpy as np
	
	#no decorator = host function. Normal python function which runs on the CPU
	def host_func():
	    #create a numpy array of integers, x_host
	    x_host = np.zeros(100, dtype=int)
	
	    #create a copy of x_host on the GPU. This copy is in device memory, and cannot be directly accessed by the CPU
	    x_device = cuda.to_device(x_host)
	
	    #here, we call the global function, from the CPU, to be executed on the GPU. We pass the device array, x_device, to this function running on the GPU
	    #note the bracket notation, where we tell the GPU how many threads, and how many blocks of threads, to use
	    thread_blocks = 256
	    threads_per_block = 256
	    global_func[thread_blocks, threads_per_block](x_device)
	
	    #finally, copy the results, which have ben stored in x_device, back to x_host
	    x_device.copy_to_host(x_host)
	
	    #print the results (on the CPU!)
	    print(x_host)
	
	#this decorator tells numba to rewrite this python function as a global function to be run on the GPU, whenever it's called by the CPU
	@cuda.jit()
	def global_func(x):
	    thread_id = cuda.grid(1)
	    num_threads = cuda.gridsize(1)
	
	    #this for loop ensures each thread only works on part of the array, thread 0 works on array index 0, thread 1 works on array index 1, etc.
	    #extra threads may go unused, which is completely ok other than being a little inefficient
	    #if the array was larger than num_threads, threads would run on more than one index, due to the stride being equal to num_threads
	    #e.g. if num_threads = 16, thread 0 would work on array indices 0, 16, 32, 48, 64, 80, and 96 for this array of size 100
	    for i in range(thread_id, x.shape[0], num_threads):
	        x[i] += device_func(i)
	
	#this decorator, slightly different than the previous one, defines this function as a device function,
	#    which may only be called on the GPU (by a global function, or another device function)
	@cuda.jit(device=True)
	def device_func(value):
	    return value*value 

Here is the same code, but written in CuPy instead of Numba:

.. code-block:: python

	import cupy as cp
	from cupyx import jit
	import numpy as np
	
	#no decorator = host function. Normal python function which runs on the CPU
	def host_func():
	    #create a numpy array of integers, x_host
	    x_host = np.zeros(100, dtype=int)
	
	    #create a copy of x_host on the GPU. This copy is in device memory, and cannot be directly accessed by the CPU
	    x_device = cp.array(x_host)
	
	    #here, we call the global function, from the CPU, to be executed on the GPU. We pass the device array, x_device, to this function running on the GPU
	    #note the bracket notation, where we tell the GPU how many threads, and how many blocks of threads, to use
	    thread_blocks = 256
	    threads_per_block = 256
	    global_func[thread_blocks, threads_per_block](x_device)
	
	    #finally, copy the results, which have ben stored in x_device, back to x_host
	    x_host = x_device.get()
	
	    #print the results (on the CPU!)
	    print(x_host)
	
	#this decorator tells numba to rewrite this python function as a global function to be run on the GPU, whenever it's called by the CPU
	@jit.rawkernel()
	def global_func(x):
	    thread_id = jit.grid(1)
	    num_threads = jit.gridsize(1)
	
	    #this for loop ensures each thread only works on part of the array, thread 0 works on array index 0, thread 1 works on array index 1, etc.
	    #extra threads may go unused, which is completely ok other than being a little inefficient
	    #if the array was larger than num_threads, threads would run on more than one index, due to the stride being equal to num_threads
	    #e.g. if num_threads = 16, thread 0 would work on array indices 0, 16, 32, 48, 64, 80, and 96 for this array of size 100
	    for i in range(thread_id, x.shape[0], num_threads):
	        x[i] += device_func(i)
	
	#this decorator, slightly different than the previous one, defines this function as a device function,
	#    which may only be called on the GPU (by a global function, or another device function)
	@jit.rawkernel(device=True)
	def device_func(value):
	    return value*value

Going forwards, all GPU examples will only show the Numba example. Almost all of these can be converted to CuPy using different imports, check the above examples 
for the specific differences.

Detailed breakdown of the for loop - parallelizing the problem
--------------------------------------------------------------

There are a few comments on what the for loop is doing in the above example, but to ensure you are absolutely clear about what it is doing, here is 
a detailed explanation.

* If you have 8 independent tasks, and only 1 worker to do those tasks, that worker must do all 8 tasks. This is similar to what a CPU does: a single computer 
  core sequentially does all instructions one after the other.
* If you have 8 independent tasks, and 4 workers to do those tasks, each worker only has to do 2 tasks each, in order for all 8 tasks to get done. This is the 
  fundamental essence of parallelism - doing independent tasks with multiple workers reduces the completion time proportional to the number of workers, up to a point.
* If you have 8 independent tasks, and 100 workers to do those tasks, 8 of those workers will do 1 task, and 92 of them will have nothing to do. Parallelism 
  cannot reduce the time further than the time it takes one worker to do one task!

In order to actually distribute the tasks to the workers, we use a strided for loop, and use two functions from Numba to do so: numba.cuda.grid, and numba.cuda.gridsize:

* numba.cuda.grid(x) returns an integer if x==1 (1D arrangement of threads) or tuple of integers if x > 1 (multidimensional arrangement of threads). These values are unique to the 
  thread which made the call. In 1D, this integer is called the thread ID. In two or more dimensions, each integer gives the "index" of the thread in that dimension. These values may 
  combined to give a single integer representing the thread ID (for example, in an 8x8 grid of threads, thread (2, 3) has thread ID equal to 2*8+3 = 19. As long as you are consistent, 
  the specific order doesn't really matter).
* numba.cuda.gridsize(x) returns an integer if x==1, or a tuple of integers if x > 1. This value (or values) represent the overall number of threads. In the 1D case, with 64 total 
  threads, this will return a value of 64 for every thread which calls this function, so every thread can know how many threads there are. In a 2D case, with an 8x8 grid of threads, 
  this function will return (8,8), so that every thread knows that there are 8 indices of threads in the y direction, and 8 indices of threads in the x direction. Note that this uses 
  C-array order convention: y is the first number, and x is the second number.

We use these values to define our strided for loop. A strided for loop in python has syntax "for i in range(start, end, stride):"

* start is the value that the variable i begins at.
* end is the condition that defines if values of the variable i are "valid". Assuming that "stride" has a positive value, if the value of i is less than "end", that value will be 
  considered valid. Importantly in this case, if "start" is equal to or larger than "end", the for loop will not return any values.
* stride is how far to increment the value of the variable i. For example, if one value of i is 16, and stride is 6, the next value of i will be 22.

For these strided for loops, we pass the result from numba.cuda.grid as the value for start, the overall number of tasks (or the number of tasks in that dimension) 
as the value for end, and the result from numba.cuda.gridsize as the value for stride. Here are three example kernels (global functions) showing this strided behavior in one and two 
dimensions, along with an image that illustrates how the strided for loop splits the problem into pieces for each thread. One of these examples shows what happens when the number of threads 
is much larger than the problem size, resulting in idle threads.

.. code-block:: python

	import numba
	from numba import cuda
	import numpy as np
	
	#all these kernels define an incrementation kernel, where the value at each array location is increased by 1
	#note that case1 and case3 are identical, the only difference comes from defining the number of threads!
	
	arr_host = np.ones(8, dtype=int)
	arr_device = cuda.to_device(arr_host)
	
	@cuda.jit()
	def case1(arr):
	    thread_id = cuda.grid(1)
	    num_threads = cuda.gridsize(1)
	    num_tasks = arr.shape[0]
	
	    for i in range(thread_id, num_tasks, num_threads):
	        arr[i] += 1
	
	case1[1, 4](arr_device) # 1x4 = 4 total threads
	
	arr_host = arr_device.copy_to_host()
	print(arr_host)

[2 2 2 2 2 2 2 2]

.. code-block:: python

	import numba
	from numba import cuda
	import numpy as np
	
	#all these kernels define an incrementation kernel, where the value at each array location is increased by 1
	
	arr_host = np.ones([4, 4], dtype=int)
	arr_device = cuda.to_device(arr_host)
	
	@cuda.jit()
	def case2(arr):
	    thread_id_y, thread_id_x = cuda.grid(2)
	    num_threads_y, num_threads_x = cuda.gridsize(2)
	    num_tasks_y = arr.shape[0]
	    num_tasks_x = arr.shape[1]
	    # above two lines can alternatively be written as the line below:
	    #num_tasks_y, num_tasks_x = arr.shape
	
	    for i in range(thread_id_y, num_tasks_y, num_threads_y):
	        for j in range(thread_id_x, num_tasks_x, num_threads_x):
	            arr[i][j] += 1
	
	case2[1, 4](arr_device) # 1x4 = 4 total threads, implicitly reshaped into a 2x2 array of threads in the kernel
	
	arr_host = arr_device.copy_to_host()
	print(arr_host)

|  [[2 2 2 2]
|   [2 2 2 2]
|   [2 2 2 2]
|   [2 2 2 2]]

.. code-block:: python

	import numba
	from numba import cuda
	import numpy as np
	
	#all these kernels define an incrementation kernel, where the value at each array location is increased by 1
	#note that case1 and case3 are identical, the only difference comes from defining the number of threads!
	
	arr_host = np.ones(8, dtype=int)
	arr_device = cuda.to_device(arr_host)
	
	@cuda.jit()
	def case3(arr):
	    thread_id = cuda.grid(1)
	    num_threads = cuda.gridsize(1)
	    num_tasks = arr.shape[0]
	
	    for i in range(thread_id, num_tasks, num_threads):
	        arr[i] += 1
	
	case3[1, 100](arr_device) # 1x100 = 100 total threads
	
	arr_host = arr_device.copy_to_host()
	print(arr_host)

[2 2 2 2 2 2 2 2]
	

.. image:: ForLoop.png

MyDiffusion Class working on the GPU
------------------------------------

Using the above techniques, we can create a global function (kernel) which runs the diffusion stencil on every internal array location

Some important notes:

* Pyphasefield has a "jit_placeholder" python file, which can be loaded in place of numba and numba.cuda in case GPU capabilities are not installed. 
  Most GPU code shouldn't run unless desired, but the decorators will run upon loading the Engine .py file. This placeholder replaces the decorators 
  with harmless ones that do nothing, which allows the Engine to be loaded on a CPU-only installation, so long as the GPU code isn't otherwise touched.
* Functions for the simulation loop are moved outside the class for clarity. This is best for GPU code, which cannot access the simulation object to 
  begin with
* As the GPU_loop is a global function (kernel), it can only access memory that is explicitly on that GPU. Pyphasefield will automatically store the 
  fields into device memory if running on the GPU, located as sim._fields_gpu_device. There is also a complementary array of fields, sim.fields_out_gpu_device, 
  which stores the results for the next timestep. This is to avoid race conditions - situations where one thread rewrites a value in fields before another thread 
  tries to read that value. 
* Other model-specific parameters, like D, dx, and dt, also by default only exist in host memory. To move these over, we create a custom "params" object in device 
  memory which holds these values. It's generally best to create this object in the "just_before_simulating" function, which is guaranteed to run immediately before 
  running the first simulation step, but after all other parameter changes which could be made.
* Unlike the above examples of splitting the problem in 2D, we do not actually want to evaluate the diffusion equation along the boundaries, hence a +1 offset to start, 
  and -1 offset to end, in each of the strided for loops. Boundary conditions will be applied after each timestep to evaluate these cells, so even if it did not lead 
  to illegal memory accesses, it would still be a waste of time!
* Pyphasefield has built-in expressions for (decent enough...) numbers of threads to use in GPU kernels, located in sim._gpu_blocks_per_grid_2D and 
  sim._gpu_threads_per_block_2D. Similar expressions also exist for one and three dimensional cases. If you would like to try optimizing the number of threads to use 
  for your specific engine, feel free! Pyphasefield by default uses 256 threads per block and 256 blocks per grid, split across multiple dimensions if called for. 
  I believe a value of 512 is compatible with most GPUs, while 1024 is the maximum permitted by CUDA. These values may be inaccurate and subject to change.

.. code-block:: python

	import numpy as np
	try:
	    #import from within Engines folder
	    from ..field import Field
	    from ..simulation import Simulation
	    from ..ppf_utils import COLORMAP_OTHER, COLORMAP_PHASE
	except:
	    try:
	        #import classes from pyphasefield library
	        from pyphasefield.field import Field
	        from pyphasefield.simulation import Simulation
	        from pyphasefield.ppf_utils import COLORMAP_OTHER, COLORMAP_PHASE
	    except:
	        raise ImportError("Cannot import from pyphasefield library!")
	
	try:
	    from numba import cuda
	    import numba
	    from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
	except:
	    import pyphasefield.jit_placeholder as numba
	    import pyphasefield.jit_placeholder as cuda
	
	def CPU_loop(sim):
	    c = sim.fields[0].data
	    D = sim.user_data["D"]
	    dx = sim.dx
	    dt = sim.dt
	
	    #define offset arrays, remember the sign of roll is opposite the direction of the cell of interest
	    #also, x is dimension 1, y is dimension 0 (C style arrays...)
	    c_p0 = np.roll(c, -1, 1) #x+1, y. 
	    c_m0 = np.roll(c, 1, 1) #x-1, y. 
	    c_0p = np.roll(c, -1, 0) #x, y+1. 
	    c_0m = np.roll(c, 1, 0) #x, y-1. 
	
	    #apply change from a single step
	    c += D*dt*(c_p0 + c_m0 + c_0p + c_0m - 4*c)/(dx**2)
	
	@cuda.jit()
	def GPU_loop(fields, fields_out, params):
	    D = params[0]
	    dx = params[1]
	    dt = params[2]
	    c = fields[0]
	    c_out = fields_out[0]
	    thread_id_y, thread_id_x = cuda.grid(2)
	    num_threads_y, num_threads_x = cuda.gridsize(2)
	    num_tasks_y, num_tasks_x = c.shape
	
	    coeff = D*dt/(dx*dx)
	
	    for i in range(thread_id_y+1, num_tasks_y-1, num_threads_y):
	        for j in range(thread_id_x+1, num_tasks_x-1, num_threads_x):
	            c_out[i][j] = c[i][j] + coeff*(c[i+1][j]+c[i-1][j]+c[i][j+1]+c[i][j-1]-4*c[i][j])
	
	class MyDiffusionClass(Simulation):
	    def __init__(self, **kwargs):
	        super().__init__(**kwargs)
	        #additional initialization code goes below
	        #runs *before* tdb, thermal, fields, and boundary conditions are loaded/initialized
	
	    def init_tdb_params(self):
	        super().init_tdb_params()
	        #additional tdb-related code goes below
	        #runs *after* tdb file is loaded, tdb_phases and tdb_components are initialized
	        #runs *before* thermal, fields, and boundary conditions are loaded/initialized
	
	    def init_fields(self):
	        #initialization of fields code goes here
	        #runs *after* tdb and thermal data is loaded/initialized
	        #runs *before* boundary conditions are initialized
	        if not ("D" in self.user_data):
	            self.user_data["D"] = 0.1
	        dim = self.dimensions
	        c = np.zeros(dim)
	        length = dim[0]
	        width = dim[1]
	        c[length // 4:3 * length // 4, width // 4:3 * width // 4] = 1
	        self.add_field(c, "c")
	
	    def initialize_engine(self):
	        super().initialize_engine()
	        #final initialization of the engine goes below
	        #runs *after* tdb, thermal, fields, and boundary conditions are loaded/initialized
	
	    def just_before_simulating(self):
	        super().just_before_simulating()
	        #additional code to run just before beginning the simulation goes below
	        #runs immediately before simulating, no manual changes permitted to changes implemented here
	        if(self._uses_gpu):
	            params = []
	            params.append(self.user_data["D"])
	            params.append(self.dx)
	            params.append(self.dt)
	            self.user_data["params"] = cuda.to_device(np.array(params))
	
	    def simulation_loop(self):
	        #code to run each simulation step goes here
	        if(self._uses_gpu):
	            GPU_loop[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device, self._fields_out_gpu_device, self.user_data["params"])
	        else:
	            CPU_loop(self)

Running this code with the framework "CPU_SERIAL" retains the original behavior, while the framework "GPU_SERIAL" runs on the GPU, and is a bit faster. Diffusion is 
a relatively simple computation so there is not a significantly large speedup, but more complex models can demonstrate speedups of around 500x.

.. code-block:: python

	from MyDiffusionGPU import MyDiffusionClass
	
	sim = MyDiffusionClass(dimensions=[500, 500])
	
	#initialize non-array parameters
	sim.set_framework("GPU_SERIAL") #"CPU_SERIAL" or "GPU_SERIAL"
	sim.set_dx(1.)
	sim.set_dt(0.1)
	sim.set_save_path("data/diffusion_test")
	sim.set_autosave_flag(True)
	sim.set_autosave_save_images_flag(True)
	sim.set_autosave_rate(2000)
	sim.set_boundary_conditions("PERIODIC")
	
	data = {
	    "D":1.
	}
	sim.set_user_data(data)
	
	#initialize simulation arrays, all parameter changes should be BEFORE this point!
	sim.initialize_engine()
	
	#change array data here, for custom simulations
	"""
	sim.fields[0].data[:] = 1.
	length = sim.dimensions[0]
	width = sim.dimensions[1]
	sim.fields[0].data[length // 4:3 * length // 4, width // 4:3 * width // 4] = 0.
	"""
	
	
	#initial conditions
	sim.plot_simulation()
	
	#run simulation
	sim.simulate(2000)
	
	#final conditions
	sim.plot_simulation()

.. image:: ../usage/diffusion1.png

.. image:: ../usage/diffusion2.png