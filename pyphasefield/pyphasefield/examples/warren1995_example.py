import sys
sys.path.insert(0,"../..")
import pyphasefield as ppf

sim = ppf.Simulation(save_path="data/warren1995_test")
sim.load_simulation(5000)
sim.init_sim_Warren1995([20, 20], diamond_size=10)
print(sim.fields[0])
print(sim.fields[1])
sim.simulate(100)
print(sim.fields[0])
print(sim.fields[1])