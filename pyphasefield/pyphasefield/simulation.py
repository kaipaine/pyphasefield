import numpy as np
import meshio as mio
from .field import Field
from pathlib import Path
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm
from . import ppf_utils
try:
    import pycalphad as pyc
    from . import ppf_gpu_utils
except:
    pass

class Simulation:
    def __init__(self, dimensions, framework="CPU_SERIAL", dx=None, dt=None, initial_time_step=0, 
                 temperature_type="NONE", initial_T=None, dTdx=0, dTdy=0, dTdz=0, dTdt=0, temperature_path=None, temperature_units="K",
                 tdb_path=None, tdb_components=[], tdb_phases=[], save_path=None, user_data={},
                 autosave=False, save_images=False, autosave_rate=None, boundary_conditions=None):
        """
        Class used by pyphasefield to store data related to a given simulation
        Methods of Simulation are used to:
            * Run simulations
            * Set what engine is used to evolve the fields each timestep
            * Save and load checkpoints
            * Plot fields
            * Set and evolve the thermal profile of the simulation
            * Load TDB files used by certain engines (requires pycalphad!)
        
        Data specific to a particular field is stored within the Field class
        """
        #framework (cpu, gpu, parallel) specific variables
        self._framework = framework
        self._uses_gpu = False
        self._gpu_blocks_per_grid_1D = (256)
        self._gpu_blocks_per_grid_2D = (16, 16)
        self._gpu_blocks_per_grid_3D = (8, 8, 8)
        self._gpu_threads_per_block_1D = (256)
        self._gpu_threads_per_block_2D = (16, 16)
        self._gpu_threads_per_block_3D = (8, 8, 8)
        
        #variable for determining if class needs to be re-initialized before running simulation steps
        self._requires_initialization = True
        
        #core variables: fields, length of space/time steps, dimensions of simulation region
        self.fields = []
        self._fields_gpu_device = None
        self.dimensions = dimensions
        self.dx = dx
        self.dt = dt
        self.time_step_counter = initial_time_step
        
        #temperature related variables
        self.temperature = None
        self._temperature_gpu_device = None
        self._temperature_type = temperature_type
        self._temperature_path = temperature_path
        self._temperature_units = temperature_units
        self._initial_T = initial_T
        self._dTdx = dTdx
        self._dTdy = dTdy
        self._dTdz = dTdz
        self._dTdt = dTdt
        self._t_file_index = None
        self._t_file_bounds = [0, 0]
        self._t_file_arrays = [None, None]
        self._t_file_gpu_devices = [None, None]
        
        #tdb related variables
        self._tdb = None
        self._tdb_path = tdb_path
        self._tdb_components = tdb_components
        self._tdb_phases = tdb_phases
        self._tdb_ufuncs = []
        
        #progress saving related variables
        self._save_path = save_path
        self._autosave_flag = autosave
        self._autosave_rate = autosave_rate
        self._autosave_save_images_flag = save_images
        
        #boundary condition related variables
        self._boundary_conditions_type = None
        self._boundary_conditions_array = None
        self._boundary_conditions_gpu_device = None
        
        #arbitrary subclass-specific data container (dictionary)
        self.user_data = user_data
        
    def init_fields(framework, num_fields):
        #exclusively used by the subclass, base Simulation class does not initialize fields!
        pass
    
    def init_boundary_conditions(boundary_conditions):
        #self._boundary_conditions_type has two different formats, and three different types
        #formats:
        #    single parameter: same type of boundary conditions are applied to all dimensions in the simulation
        #    list: one parameter per dimension, allowing for different boundary conditions on each axis
        #types:
        #    PERIODIC: values from one side of the field wrap around to the other side
        #    DIRCHLET: values on the boundary have a constant value (defaults to initial values of edges!)
        #    NEUMANN: values on the boundary have a constant derivative across the boundary (defaults to zero!)
        
        dim = self.dimensions.copy()
        for i in range(len(dim)):
            dim[i] += 2
        dim.insert(0, len(self.fields)) #requires initialization of fields first
        self._boundary_conditions_array = np.zeros(dim)
        self.apply_boundary_conditions() 
        #note, this applies default boundary conditions to arrays, run this again after initialization to apply changes to bcs
        
        
    def init_temperature_field(temperature_type):
        if(self._temperature_type == "NONE"):
            pass
        elif(self._temperature_type == "ISOTHERMAL"):
            array = np.zeros(self.dimensions)
            array += self._initial_T
            t_field = Field(data=array, simulation=self, colormap="jet", name="Temperature ("+self._temperature_units+")")
            self.temperature = t_field
        elif(self._temperature_type == "LINEAR_GRADIENT"):
            array = np.zeros(self.dimensions)
            array += self._initial_T
            x_length = self.dimensions[len(self.dimensions)-1]
            x_t_array = np.arange(0, x_length*self.dx*self.dTdx, self.dTdx)
            array += x_t_array
            if(len(self.dimensions) > 1):
                y_length = self.dimensions[len(self.dimensions)-2]
                y_t_array = np.arange(0, y_length*self.dx*self.dTdy, self.dTdy)
                y_t_array = np.expand_dims(y_t_array, axis=1)
                array += y_t_array
            if(len(self.dimensions) > 2):
                z_length = self.dimensions[len(self.dimensions)-3]
                z_t_array = np.arange(0, z_length*self.dx*self.dTdz, self.dTdz)
                z_t_array = np.expand_dims(z_t_array, axis=1)
                z_t_array = np.expand_dims(z_t_array, axis=2)
                array += z_t_array
            array += self.time_step_counter*dTdt
            t_field = Field(data=array, simulation=self, colormap="jet", name="Temperature ("+self._temperature_units+")")
            self.temperature = t_field
        elif(self._temperature_type == "XDMF_FILE"):
            self._t_file_index = 1
            with mio.xdmf.TimeSeriesReader(self._save_path+"/T.xdmf") as reader:
                dt = self.dt
                step = self.time_step_counter
                points, cells = reader.read_points_cells()
                self._t_file_bounds[0], point_data0, cell_data0 = reader.read_data(0)
                self._t_file_arrays[0] = point_data0['T']
                self._t_file_bounds[1], point_data1, cell_data0 = reader.read_data(self._t_file_index)
                self._t_file_arrays[1] = point_data1['T']
                while(dt*step > self._t_file_end):
                    self._t_file_bounds[0] = self._t_file_bounds[1]
                    self._t_file_arrays[0] = self._t_file_arrays[1]
                    self._t_file_index += 1
                    self._t_file_bounds[1], point_data1, cell_data0 = reader.read_data(self.t_file_index)
                    self._t_file_arrays[1] = point_data1['T']
                array = self._t_file_arrays[0]*(self._t_file_bounds[1] - dt*step)/(self._t_file_bounds[1]-self._t_file_bounds[0]) + self._t_file_arrays[1]*(dt*step-self._t_file_bounds[0])/(self._t_file_bounds[1]-self._t_file_bounds[0])
                t_field = Field(data=array, simulation=self, colormap="jet", name="Temperature ("+self._temperature_units+")")
                self.temperature = t_field
        
    def init_tdb_params(self):
        if(self._tdb_path is None):
            return
        #if tdb_path is specified, pycalphad *must* be installed
        if not ppf_utils.successfully_imported_pycalphad():
            raise ImportError
        import pycalphad as pyc
        self._tdb = pyc.Database(tdb_path)
        if self._tdb_phases is None:
            self._tdb_phases = list(self._tdb.phases)
        if self._tdb_components is None:
            self._tdb_components = list(self._tdb.elements)
        self._tdb_phases.sort()
        self._tdb_components.sort()
        param_search = sim._tdb.search
        for k in range(len(self._tdb_phases)):
            phase_id = self._tdb_phases[k]
            phase = self._tdb.phases[phase_id]
            g_param_query = (
                (where('phase_name') == phase.name) & \
                ((where('parameter_type') == 'G') | \
                (where('parameter_type') == 'L'))
            )
            model = pyc.Model(self._tdb, self._tdb_components, phase_id)
            sympyexpr = model.redlich_kister_sum(phase, param_search, g_param_query)
            ime = model.ideal_mixing_energy(sim._tdb)

            sympysyms_list = []
            T = None
            for i in list(sim._tdb.elements):
                for j in sympyexpr_solid.free_symbols:
                    if j.name == phase_id+"0"+i: 
                        #this may need additional work for phases with sublattices...
                        sympysyms_list_solid.append(j)
                    if j.name == "T":
                        T = j
            sympysyms_list = sorted(sympysyms_list, key=lambda t:t.name)
            sympysyms_list.append(T)
        
            for i in self._tdb.symbols:
                d = self._tdb.symbols[i]
                g = sp.Symbol(i)
                sympyexpr = sympyexpr.subs(g, d)
            
            if(self._framework == "CPU_SERIAL" or self._framework == "CPU_PARALLEL"):
                #use numpy for CPUs
                self._tdb_ufuncs.append(sp.lambdify(tuple(sympysyms_list), sympyexpr+ime, 'numpy'))
            else: 
                #use numba for GPUs
                self._tdb_ufuncs.append(numba.jit(sp.lambdify(tuple(sympysyms_list), sympyexpr+ime, 'math')))

    def simulate(self, number_of_timesteps, reset=False):
        """
        Evolves the simulation for a specified number of timesteps
        If a length of timestep is not specified, uses the timestep length stored within the Simulation instance
        For each timestep, the method:
            * Increments the timestep counter by 1
            * Runs the engine function (a function which only takes the Simulation instance as a 
                parameter, and evolves the fields contained within the instance by one time step
            * Updates the thermal field of the simulation depending on which thermal type the simulation is:
                - Isothermal: Do nothing
                - Gradient: Add dT/dt, multiplied by the timestep length, to the thermal field
                - File: Use linear interpolation to find the thermal field of the new timestep
            * If the timestep counter is a multiple of time_steps_per_checkpoint, save a checkpoint of the simulation
        """
        if(self._requires_initialization):
            self.initialize_simulation()
            self._requires_initialization = False
        for i in range(number_of_timesteps):
            self.time_step_counter += 1
            self.simulation_loop()
            self.update_temperature_field()
            self.apply_boundary_conditions()
            if(self._autosave_flag):
                if self.time_step_counter % self._autosave_rate == 0:
                    if(self._uses_gpu):
                        ppf_gpu_utils.retrieve_fields_from_GPU(self)
                    self.save_simulation()
        if(reset):
            self._requires_initialization = True
                
    def initialize_simulation(self):
        if(framework == "GPU_SERIAL" or framework == "GPU_PARALLEL"): 
            """parallel not yet implemented!"""
            self._uses_gpu = True
        self.init_boundary_conditions()
        self.init_temperature_field()
        self.init_tdb_params()
        if(self._uses_gpu):
            self.send_fields_to_GPU()
                
    def simulation_loop(self):
        pass
    
    def add_field(self, array, array_name, colormap="GnBu"):
        field = Field(data=array, name=array_name, simulation=self, colormap=colormap)
        self.fields.append(field)

    def load_tdb(self, tdb_path, phases=None, components=None):
        """
        Loads the tdb file using pycalphad. (Needless to say, this requires pycalphad!)
        The format for phases and components attributes of Simulation are a list of strings 
            that correspond to the terms within the tdb file
        Examples:
            * phases=["FCC_A1", "LIQUID"]
            * components=["CU", "NI"]
        Unless specified, method will load all phases and components contained within the tdb file.
        phases and components lists are always in alphabetical order, and will be automatically 
            sorted if not already done by the user
        """
        if not ppf_utils.successfully_imported_pycalphad():
            return
        import pycalphad as pyc
        self._tdb_path = tdb_path
        self._tdb = pyc.Database(tdb_path)
        if phases is None:
            self._phases = list(self._tdb.phases)
        else:
            self._phases = phases
        if components is None:
            self._components = list(self._tdb.elements)
        else:
            self._components = components
        self._phases.sort()
        self._components.sort()

    def get_time_step_length(self):
        """Returns the length of a single timestep"""
        return self.dt

    def get_time_step_counter(self):
        """Returns the number of timesteps that have passed for a given simulation"""
        return self.time_step_counter

    def set_time_step_length(self, time_step):
        """Sets the length of a single timestep for a simulation instance"""
        self.dt = time_step
        return

    def update_temperature_field(self):
        """Updates the thermal field, method assumes only one timestep has passed"""
        if(self._uses_gpu):
            ppf_gpu_utils.update_temperature_field(self)
            return
        elif(self._temperature_type == "ISOTHERMAL"):
            return
        elif(self._temperature_type == "LINEAR_GRADIENT"):
            self.temperature.data += self._dTdt*self.dt
            return
        elif(self._temperature_type == "XDMF_FILE"):
            dt = self.get_time_step_length()
            step = self.get_time_step_counter()
            while(dt*step > self._t_file_bounds[1]):
                with mio.xdmf.TimeSeriesReader(self._temperature_path) as reader:
                    reader.cells=[]
                    self._t_file_bounds[0] = self._t_file_bounds[1]
                    self._t_file_arrays[0] = self._t_file_arrays[1]
                    self._t_file_index += 1
                    self._t_file_bounds[1], point_data1, cell_data0 = reader.read_data(self._t_file_index)
                    self._t_file_arrays[1] = point_data1['T']
            self.temperature.get_cells() = self._t_file_arrays[0]*(self._t_file_bounds[1] - dt*step)/(self._t_file_bounds[1]-self._t_file_bounds[0]) + self._t_file_arrays[1]*(dt*step-self._t_file_bounds[0])/(self._t_file_bounds[1]-self._t_file_bounds[0])
            return
        if self._temperature_type not in ["ISOTHERMAL", "LINEAR_GRADIENT", "XDMF_FILE", "NONE"]:
            raise ValueError("Unknown temperature profile.")

    def load_simulation(self, file_path=None, step=-1):
        """
        Loads a simulation from a .npz file. Either a filename, step, or both must be provided.
            If no step is specified, checks filename for step #.
            If no filename is specified, a file with the specified step number is loaded from
            the _save_path.
        """
        # Clear fields list
        self.fields = []
        
        if file_path is None:
            file_path = self._save_path
            if self._save_path is None:
                raise ValueError("Simulation needs a path to load data from!")
                
        # Check for file path inside cwd
        if Path.cwd().joinpath(file_path).exists():
            file_path = Path.cwd().joinpath(file_path)
        # Check if file exists in the save directory
        elif Path.cwd().joinpath(self._save_path).joinpath(file_path).exists():
            file_path = Path.cwd().joinpath(self._save_path).joinpath(file_path)
        else:
            file_path = Path(file_path)
            
        if file_path.is_dir():
            if step > -1:
                file_path = file_path.joinpath("step_"+str(step)+".npz")
            else:
                raise ValueError("Given path is a folder, must specify a timestep!")

        #propagate new path to the save path, the parent folder is the save path
        #only does so if the save path for the simulation is not set!
        if(self._save_path is None):
            self._save_path = str(file_path.parent)

        # Load array
        fields_dict = np.load(file_path, allow_pickle=True)

        # Add arrays self.fields as Field objects
        for key, value in fields_dict.items():
            tmp = Field(value, self, key)
            self.fields.append(tmp)
            
        # Set dimensions of simulation
        self._dimensions_of_simulation_region = self.fields[0].data.shape

        # Time step set from parsing file name or manually --> defaults to 0
        if step < 0:
            filename = file_path.stem
            step_start_index = filename.find('step_') + len('step_')
            if step_start_index == -1:
                self._time_step_counter = 0
            else:
                i = step_start_index
                while i < len(filename) and filename[i].isdigit():
                    i += 1
                self._time_step_counter = int(filename[step_start_index:i])
        else:
            self._time_step_counter = int(step)
        if(self._uses_gpu):
            self.send_fields_to_GPU()
        if(self._temperature_type == "LINEAR_GRADIENT"):
            self.temperature.data += self._time_step_counter*self._dTdt*self.dt
        if(self._temperature_type == "XDMF_FILE"):
            self.update_temperature_field()
        return 0
    
    def save_simulation(self):
        """
        Saves all fields in a .npz in either the user-specified save path or a default path. Step number is saved
        in the file name.
        TODO: save data for simulation instance in header file
        """
        save_dict = dict()
        for i in range(len(self.fields)):
            tmp = self.fields[i]
            save_dict[tmp.name] = tmp.data

        # Save array with path
        if not self._save_path:
            engine_name = self._engine.__name__
            print("Simulation.save_path not specified, saving to /data/"+engine_name)
            save_loc = Path.cwd().joinpath("data/", engine_name)
        else:
            save_loc = Path(self._save_path)
        save_loc.mkdir(parents=True, exist_ok=True)

        np.savez(str(save_loc) + "/step_" + str(self._time_step_counter), **save_dict)
        return 0
    
    def plot_simulation(self, fields=None, interpolation="bicubic", units="cells", save_images="False", size=None, norm=False):
        if(self._uses_gpu):
            ppf_gpu_utils.retrieve_fields_from_GPU(self)
        if fields is None:
            fields = range(len(self.fields))
        if(len(self.fields[0].data.shape) == 1): #1d plot
            x = np.arange(0, len(self.fields[0].data))
            for i in fields:
                plt.plot(x, self.fields[i].data)
            plt.show()
        elif((len(self.fields[0].data) == 1) or (len(self.fields[0].data[0]) == 1)): #effective 1d plot, 2d but one dimension = 1
            x = np.arange(0, len(self.fields[0].data.flatten()))
            legend = []
            for i in fields:
                plt.plot(x, self.fields[i].data.flatten())
                legend.append(self.fields[i].name)
            plt.legend(legend)
            plt.show()
        else:
            for i in fields:
                if(units == "cells"):
                    extent = [0, self.fields[i].data.shape[1], 0, self.fields[i].data.shape[0]]
                elif(units == "cm"):
                    extent = [0, self.fields[i].data.shape[1]*self.get_cell_spacing(), 0, self.fields[i].data.shape[0]*self.get_cell_spacing()]
                elif(units == "m"):
                    extent = [0, self.fields[i].data.shape[1]*self.get_cell_spacing()/100., 0, self.fields[i].data.shape[0]*self.get_cell_spacing()/100.]
                if not (size is None):
                    plt.figure(figsize=size)
                if(norm):
                    if(i == 0):
                        plt.imshow(self.fields[i].data, interpolation=interpolation, cmap=self.fields[i].colormap, extent=extent, norm=PowerNorm(10, vmin=0, vmax=1))
                    else:
                        plt.imshow(self.fields[i].data, interpolation=interpolation, cmap=self.fields[i].colormap, extent=extent)
                else:
                    plt.imshow(self.fields[i].data, interpolation=interpolation, cmap=self.fields[i].colormap, extent=extent)
                plt.title(self.fields[i].name)
                plt.colorbar()
                if(units == "cm"):
                    plt.xlabel("cm")
                    plt.ylabel("cm")
                elif(units == "m"):
                    plt.xlabel("m")
                    plt.ylabel("m")
                if(save_images):
                    plt.savefig(self._save_path+"/"+self.fields[i].name+"_"+str(self.get_time_step_counter())+".png")
                plt.show()
                

    def set_dimensions(self, dimensions_of_simulation_region):
        self._dimensions_of_simulation_region = dimensions_of_simulation_region
        return
    
    def get_dimensions(self):
        return self._dimensions_of_simulation_region

    def set_cell_spacing(self, cell_spacing):
        self.dx = cell_spacing
        return

    def get_cell_spacing(self):
        return self.dx

    def set_checkpoint_rate(self, time_steps_per_checkpoint):
        self._time_steps_per_checkpoint = time_steps_per_checkpoint
        return

    def set_automatic_plot_generation(self, plot_simulation_flag):
        self._save_images_at_each_checkpoint = plot_simulation_flag
        return

    def set_debug_mode(self, debug_mode_flag):
        self._debug_mode_flag = debug_mode_flag
        return

    def set_boundary_conditions(self, boundary_conditions_type):
        self._boundary_conditions_type = boundary_conditions_type

    def increment_time_step_counter(self):
        self.time_step_counter += 1
        return

    def apply_boundary_conditions(self):
        neumann_slices_1 = [[(0), (None, 0), (None, None, 0)], [(1), (None, 1), (None, None, 1)]]
        neumann_slices_2 = [[(-1), (None, -1), (None, None, -1)], [(-2), (None, -2), (None, None, -2)]]
        periodic_slices_1 = [[(0), (None, 0), (None, None, 0)], [(-2), (None, -2), (None, None, -2)]]
        periodic_slices_2 = [[(-1), (None, -1), (None, None, -1)], [(1), (None, 1), (None, None, 1)]]
        dirchlet_slices_1 = [(0), (None, 0), (None, None, 0)]
        dirchlet_slices_2 = [(-1), (None, -1), (None, None, -1)]
        if(self._uses_gpu):
            ppf_gpu_utils.apply_boundary_conditions(self)
            return
        if(self._boundary_conditions_type == "PERIODIC"):
            dims = len(self.fields[0].data.shape)
            for i in range(dims):
                if not(self.temperature is None):
                    self.temperature.data[periodic_slices_1[0][i]] = self.temperature.data[periodic_slices_1[1][i]]
                for j in range(len(self.fields)):
                    self.fields[j].data[periodic_slices_1[0][i]] = self.fields[j].data[periodic_slices_1[1][i]]
                    self.fields[j].data[periodic_slices_2[0][i]] = self.fields[j].data[periodic_slices_2[1][i]]
        elif(self._boundary_conditions_type == "NEUMANN"):
            dims = len(self.fields[0].data.shape)
            _slice = []
            for i in range(dims):
                if not(self.temperature is None):
                    self.temperature.data[neumann_slices_1[0][i]] = self.temperature.data[neumann_slices_1[1][i]]
                for j in range(len(self.fields)):
                    self.fields[j].data[neumann_slices_1[0][i]] = self.fields[j].data[neumann_slices_1[1][i]] - self.dx*self._boundary_conditions_array[j][neumann_slices_1[0][i]]
                    self.fields[j].data[neumann_slices_2[0][i]] = self.fields[j].data[neumann_slices_2[1][i]] + self.dx*self._boundary_conditions_array[j][neumann_slices_2[0][i]]
        elif(self._boundary_conditions_type == "DIRCHLET"):
            dims = len(self.fields[0].data.shape)
            _slice = []
            for i in range(dims):
                if not(self.temperature is None):
                    #use neumann boundary conditions for temperature field if using dirchlet boundary conditions
                    self.temperature.data[neumann_slices_1[0][i]] = self.temperature.data[neumann_slices_1[1][i]]
                for j in range(len(self.fields)):
                    self.fields[j].data[dirchlet_slices_1[i]] = self._boundary_conditions_array[j][dirchlet_slices_1[i]]
                    self.fields[j].data[dirchlet_slices_2[i]] = self._boundary_conditions_array[j][dirchlet_slices_2[i]]
        else: #is array
            for i in range(len(self._boundary_conditions_type)):
                if(self._boundary_conditions_type[i] == "PERIODIC"):
                    if not(self.temperature is None):
                        self.temperature.data[periodic_slices_1[0][i]] = self.temperature.data[periodic_slices_1[1][i]]
                    for j in range(len(self.fields)):
                        self.fields[j].data[periodic_slices_1[0][i]] = self.fields[j].data[periodic_slices_1[1][i]]
                        self.fields[j].data[periodic_slices_2[0][i]] = self.fields[j].data[periodic_slices_2[1][i]]
                elif(self._boundary_conditions_type[i] == "NEUMANN"):
                    if not(self.temperature is None):
                        self.temperature.data[neumann_slices_1[0][i]] = self.temperature.data[neumann_slices_1[1][i]]
                    for j in range(len(self.fields)):
                        self.fields[j].data[neumann_slices_1[0][i]] = self.fields[j].data[neumann_slices_1[1][i]] - self.dx*self._boundary_conditions_array[j][neumann_slices_1[0][i]]
                        self.fields[j].data[neumann_slices_2[0][i]] = self.fields[j].data[neumann_slices_2[1][i]] + self.dx*self._boundary_conditions_array[j][neumann_slices_2[0][i]]
                elif(self._boundary_conditions_type[i] == "DIRCHLET"):
                    if not(self.temperature is None):
                        #use neumann boundary conditions for temperature field if using dirchlet boundary conditions
                        self.temperature.data[neumann_slices_1[0][i]] = self.temperature.data[neumann_slices_1[1][i]]
                    for j in range(len(self.fields)):
                        self.fields[j].data[dirchlet_slices_1[i]] = self._boundary_conditions_array[j][dirchlet_slices_1[i]]
                        self.fields[j].data[dirchlet_slices_2[i]] = self._boundary_conditions_array[j][dirchlet_slices_2[i]]
        return
    
    def send_fields_to_GPU(self):
        if(ppf_utils.successfully_imported_numba()):
            ppf_gpu_utils.send_fields_to_GPU(self)
        return
    
    def retrieve_fields_from_GPU(self):
        if(ppf_utils.successfully_imported_numba()):
            ppf_gpu_utils.retrieve_fields_from_GPU(self)
        return

    def plot_all_fields(self):
        """
        Plots each field in self.fields and saves them to the save_path in a separate dir
        Recommended for when the number of fields used would clutter the data folder
        """
        image_folder = "images_step_" + str(self._time_step_counter) + "/"
        save_path = Path(self._save_path).joinpath(image_folder)
        save_path.mkdir(parents=True, exist_ok=True)
        for i in range(len(self.fields)):
            self.plot_field(self.fields[i], save_path)
        return 0

    def plot_field(self, f, save_path=None):
        """
        Plots each field as a matplotlib 2d image. Takes in a field object as arg and saves
        the image to the data folder as namePlot_step_n.png
        """
        if(self._uses_gpu):
            ppf_gpu_utils.retrieve_fields_from_GPU(self)
        if save_path is None:
            save_path = self._save_path
        fig, ax = plt.subplots()
        c = plt.imshow(f.data, interpolation='nearest', cmap=f.colormap)

        title = "Field: " + f.name + ", Step: " + str(self._time_step_counter)
        plt.title(title)
        fig.colorbar(c, ticks=np.linspace(np.min(f.data), np.max(f.data), 5))
        # Save image to save_path dir
        filename = f.name + "Plot_step_" + str(self._time_step_counter) + ".png"
        plt.savefig(Path(save_path).joinpath(filename))
        return 0

    def progress_bar(self):
        return

    def generate_python_script(self):
        return
    
    #import statements, specific to built-in Engines *TO BE REMOVED*

    def init_sim_Diffusion(self, dim=[200], solver="explicit", gmres=False, adi=False):
        Engines.init_Diffusion(self, dim, solver=solver, gmres=gmres, adi=adi)
        return
    
    def init_sim_DiffusionGPU(self, dim=[200, 200], cuda_blocks=(16,16), cuda_threads_per_block=(256,1)):
        if not ppf_utils.successfully_imported_numba():
            return
        Engines.init_DiffusionGPU(self, dim=dim, cuda_blocks=cuda_blocks, cuda_threads_per_block=cuda_threads_per_block)
        return
    
    def init_sim_CahnAllen(self, dim=[200], solver="explicit", gmres=False, adi=False):
        Engines.init_CahnAllen(self, dim, solver=solver, gmres=gmres, adi=adi)
        return
    
    def init_sim_CahnHilliard(self, dim=[200], solver="explicit", gmres=False, adi=False):
        Engines.init_CahnHilliard(self, dim, solver=solver, gmres=gmres, adi=adi)
        return

    def init_sim_Warren1995(self, dim=[200, 200], diamond_size=15):
        Engines.init_Warren1995(self, dim=dim, diamond_size=diamond_size)
        return

    def init_sim_NComponent(self, dim=[200, 200], sim_type="seed", number_of_seeds=1, tdb_path="Ni-Cu_Ideal.tdb",
                            temperature_type="isothermal",
                            initial_temperature=1574, temperature_gradient=0, cooling_rate=0, temperature_file_path="T.xdmf",
                            initial_concentration_array=[0.40831], cell_spacing=0.0000046, d_ratio=1/0.94, solver="explicit", 
                            nbc=["periodic", "periodic"]):
        #initializes a Multicomponent simulation, using the NComponent model
        if not ppf_utils.successfully_imported_pycalphad():
            return
        Engines.init_NComponent(self, dim=dim, sim_type=sim_type, number_of_seeds=number_of_seeds, 
                                tdb_path=tdb_path, temperature_type=temperature_type, 
                                initial_temperature=initial_temperature, temperature_gradient=temperature_gradient, 
                                cooling_rate=cooling_rate, temperature_file_path=temperature_file_path, 
                                cell_spacing=cell_spacing, d_ratio=d_ratio, initial_concentration_array=initial_concentration_array, 
                                solver=solver, nbc=nbc)
        return
    
    def init_sim_NCGPU(self, dim=[200, 200], sim_type="seed", number_of_seeds=1, tdb_path="Ni-Cu_Ideal.tdb",
                            temperature_type="isothermal",
                            initial_temperature=1574, temperature_gradient=0, cooling_rate=0, temperature_file_path="T.xdmf",
                            initial_concentration_array=[0.40831], cell_spacing=0.0000046, d_ratio=1/0.94, solver="explicit", 
                            nbc=["periodic", "periodic"], cuda_blocks = (16,16), cuda_threads_per_block = (256,1)):
        if not ppf_utils.successfully_imported_pycalphad():
            return
        if not ppf_utils.successfully_imported_numba():
            return
        
        Engines.init_NCGPU(self, dim=dim, sim_type=sim_type, number_of_seeds=number_of_seeds, 
                                tdb_path=tdb_path, temperature_type=temperature_type, 
                                initial_temperature=initial_temperature, temperature_gradient=temperature_gradient, 
                                cooling_rate=cooling_rate, temperature_file_path=temperature_file_path, 
                                cell_spacing=cell_spacing, d_ratio=d_ratio, initial_concentration_array=initial_concentration_array, 
                                solver=solver, nbc=nbc, cuda_blocks=cuda_blocks, cuda_threads_per_block=cuda_threads_per_block)
