import numpy as np
from .field import Field
from . import Engines

def successfully_imported_pycalphad():
        try:
            import pycalphad as pyc
        except:
            print("The feature you are trying to use requires pycalphad")
            print("In Anaconda, use \'conda install -c pycalphad -c conda-forge pycalphad\' to install it")
            return False
        return True

class Simulation:
    def __init__(self, save_path):
        self.fields = []
        self.temperature = None
        self._dimensions_of_simulation_region = [200,200]
        self._cell_spacing_in_meters = 1.
        self._time_step_in_seconds = 1.
        self._simulation_time_step_reached = 0
        self._temperature_type = "isothermal"
        self._initial_temperature_left_side = 1574.
        self._thermal_gradient_Kelvin_per_meter = 0.
        self._cooling_rate_Kelvin_per_second = 0. #cooling is a negative number! this is dT/dt
        self._tdb = None
        self._tdb_path = ""
        self._components = []
        self._phases = []
        self._engine = None
        self._save_path = save_path
        self._steps_per_checkpoint = 500
        self._save_images_at_each_checkpoint = False
        self._boundary_conditions_type = ["periodic", "periodic"]
    
    def simulate(self, number_of_timesteps, dt=None):
        if dt is None:
            dt=self._time_step_in_seconds
        self._time_step_in_seconds = dt
        for i in range(number_of_timesteps):
            self.engine(self) #run engine on Simulation instance
            self.update_thermal_field()
            if(self._simulation_time_step_reached%self._steps_per_checkpoint == 0):
                self.save_simulation()
                
    def load_tdb(self, tdb_path, phases=None, components=None):
        #loads the tdb file using pycalphad
        #format for phases and components are a list of strings that correspond to the terms within the tdb file
        #examples:
        #phases=[FCC_A1, LIQUID]
        #components=[CU, NI]
        #unless specified, will load all phases and components contained within the tdb file.
        #phases and components lists are always in alphabetical order
        if not successfully_imported_pycalphad():
            return
        import pycalphad as pyc
        self._tdb_path = tdb_path
        self._tdb = pyc.Database(tdb_path)
        if(phases==None):
            self._phases = list(self._tdb.phases)
        else:
            self._phases = phases
        if(components==None):
            self._components = list(self._tdb.elements)
        else:
            self._components = components
        self._phases.sort()
        self._components.sort()
        
    def get_time_step_length(self):
        return self._time_step_in_seconds
    
    def get_time_step_reached(self):
        return self._simulation_time_step_reached
    
    def set_time_step_length(self, time_step):
        self._time_step_in_seconds = time_step
        return
    
    def set_thermal_isothermal(self, temperature):
        array = np.zeros(self._dimensions_of_simulation_region)
        array += temperature
        self.temperature = array
        return
    
    def set_thermal_gradient(self, initial_T_left_side, dTdx, dTdt):
        array = np.zeros(self._dimensions_of_simulation_region)
        array += temperature
        array += np.linspace(0, dTdx*shape[1]*self._cell_spacing_in_m, shape[1])
        array += self.get_time_step_reached()*self.get_time_step_length()*dTdt
        self.temperature = array
        return
    
    def set_thermal_file(self, thermal_file_path):
        return
    
    def update_thermal_field(self):
        if(self._temperature_type == "isothermal"):
            return
        elif(self._temperature_type == "gradient"):
            return
        elif(self._temperature_type == "file"):
            return
        #if it gets this far, warn user about unexpected temperature_type
        return
    
    def load_simulation(self):
        return
    
    def save_simulation(self):
        return
    
    def set_dimensions(self, dimensions_of_simulation_region):
        self._dimensions_of_simulation_region = dimensions_of_simulation_region
        return
    
    def set_cell_spacing(self, cell_spacing):
        self._cell_spacing_in_meters = cell_spacing
        return
    
    def get_cell_spacing(self):
        return self._cell_spacing_in_meters
    
    def add_field(self, field):
        #warn if field dimensions dont match simulation dimensions
        self.fields.append(field)
        return
    
    def set_engine(self, engine_function):
        self.engine = engine_function
        return
    
    def set_checkpoint_rate(self, steps_per_checkpoint):
        self._steps_per_checkpoint = steps_per_checkpoint
        return
    
    def set_automatic_plot_generation(self, plot_simulation_flag):
        self._save_images_at_each_checkpoint = plot_simulation_flag
        return
    
    def set_debug_mode(self, debug_mode_flag):
        self._debug_mode_flag = debug_mode_flag
        return
    
    def set_boundary_conditions(self, boundary_conditions_type):
        self._boundary_conditions_type = boundary_conditions_type
    
    def increment_step_counter(self):
        self._simulation_time_step_reached += 1
        return
    
    def apply_boundary_conditions(self):
        return
    
    def renormalize_quaternions(self):
        return
    
    def cutoff_order_values(self):
        return
    
    def plot_fields(self):
        return
    
    def progress_bar(self):
        return
    
    def generate_python_script(self):
        return
    
    def init_sim_Diffusion(self, dim=[200]):
        Engines.init_Diffusion(self, dim)
        return
    
    def init_sim_Warren1995(self, dim=[200, 200]):
        Engines.init_Warren1995(self, dim)
        return
    
    def init_sim_NComponent(self, dim=[200,200], sim_type="seed", tdb_path="Ni-Cu_Ideal.tdb", thermal_type="isothermal", 
                           initial_temperature=1574, thermal_gradient=0, cooling_rate=0, thermal_file_path="T.xdmf", 
                           initial_concentration_array=[0.40831]):
        if not successfully_imported_pycalphad():
            return
        Engines.init_NComponent(self, dim, sim_type, tdb_path, thermal_type, initial_temperature, 
                                thermal_gradient, cooling_rate, thermal_file_path)
        return