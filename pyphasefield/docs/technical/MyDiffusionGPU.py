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