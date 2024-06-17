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