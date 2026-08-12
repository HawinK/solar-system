import matplotlib.pyplot as plt
from solar_simulation import Simulation, ensure_json


JSON_FILE = ensure_json("parameters_solar.json")
DT = 1.0 / 600
YEARS = 20
N_STEPS = int(YEARS / DT)
WRITE_EVERY = 60 # write energy every 60 steps

print("Experiment 2 - Energy Conservation")
print(f"  Time step : {DT:.5f} yr")
print(f"  Duration  : {YEARS} yr  ({N_STEPS} steps)\n")

def run_method(method_name, energy_file):
    print(f"Running {method_name} ...")
    sim = Simulation(
        timestep = DT,
        num_iterations = N_STEPS,
        method = method_name,
        energy_output_file = energy_file,
    )
    sim.read_input_data(JSON_FILE)
    sim.run_simulation(energy_write_interval=WRITE_EVERY)
    return sim.time_history, sim.energy_history

times_bm, energy_bm  = run_method("beeman", "energy_beeman.csv")
times_ec, energy_ec  = run_method("euler_cromer", "energy_euler_cromer.csv")
times_de, energy_de  = run_method("direct_euler", "energy_direct_euler.csv")

def normalise(energy_list):
    E0 = energy_list[0]
    return [(E - E0) / abs(E0) for E in energy_list]

norm_bm = normalise(energy_bm)
norm_ec = normalise(energy_ec)
norm_de = normalise(energy_de)

# plot 1: all three methods on the same axes, showing their normalised fractional changes in energy
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(times_bm, norm_bm, label="Beeman",lw=1.0, color="steelblue")
ax1.plot(times_ec, norm_ec, label="Euler-Cromer",lw=1.0, color="seagreen", alpha=0.85)
ax1.plot(times_de, norm_de, label="Direct Euler",lw=1.0, color="tomato",alpha=0.85)
ax1.set_xlabel("Time (Earth years)")
ax1.set_ylabel("Fractional energy change  (E - E0) / |E0|")
ax1.set_title("Experiment 2 - Energy Conservation: All Three Methods")
ax1.legend()
ax1.grid(linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("exp2_all_methods_energy.png", dpi=150)
plt.show()

# plot 2: only beeman, zoomed in to show smaller oscillations
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(times_bm, norm_bm, lw=0.8, color="steelblue")
ax2.set_xlabel("Time (Earth years)")
ax2.set_ylabel("Fractional energy change  (E - E0) / |E0|")
ax2.set_title("Experiment 2 - Beeman Energy Conservation (detail)")
ax2.grid(linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("exp2_beeman_energy_detail.png", dpi=150)
plt.show()

print("\n-- Energy drift summary --") # printing a summary
print(f"{'Method':<15}  {'Max |fractional change|':>25}")
print("-" * 44)
for label, norm_e in [("Beeman", norm_bm),
                       ("Euler-Cromer", norm_ec),
                       ("Direct Euler", norm_de)]:
    max_drift = max(abs(v) for v in norm_e)
    print(f"{label:<15}  {max_drift:>25.6e}")

print("\nPlots saved as exp2_all_methods_energy.png and "
      "exp2_beeman_energy_detail.png")
