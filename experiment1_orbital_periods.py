import matplotlib.pyplot as plt
from solar_simulation import Simulation, ensure_json

NASA_PERIODS = {
    "Mercury": 0.2409,
    "Venus": 0.6152,
    "Earth": 1.0000,
    "Mars": 1.8809,
    "Jupiter": 11.862,
} # given in Earth years


JSON_FILE = ensure_json("parameters_solar.json")
DT = 1.0 / 600
YEARS = 13
N_STEPS = int(YEARS / DT)

sim = Simulation(
    timestep = DT,
    num_iterations = N_STEPS,
    method = "beeman",
    energy_output_file = "energy_exp1.csv",
)
sim.read_input_data(JSON_FILE)


print("Running Experiment 1 – Orbital Periods")
print(f"  Time step : {DT:.5f} yr  ({1/DT:.0f} steps per Earth year)")
print(f"  Duration  : {YEARS} yr  ({N_STEPS} steps total)\n")

sim.run_simulation(energy_write_interval=100)

#compare simulated vs NASA periods
print("\n-- Comparison with NASA values --")
print(f"{'Planet':<10}  {'Simulated (yr)':>15}  {'NASA (yr)':>10}  "
      f"{'Error (%)':>10}")
print("-" * 52)

simulated = {}
for body in sim.body_list:
    if body.is_sun or body.is_satellite:
        continue
    sim_period = body.orbital_period
    nasa_period = NASA_PERIODS.get(body.name)

    if sim_period is not None and nasa_period is not None:
        error_pct = 100.0 * abs(sim_period - nasa_period) / nasa_period # calculating percentage error in result
        print(f"{body.name:<10}  {sim_period:>15.4f}  {nasa_period:>10.4f}  "
              f"{error_pct:>9.2f}%")
        simulated[body.name] = sim_period
    elif sim_period is None:
        print(f"{body.name:<10}  {'not detected':>15}  "
              f"{nasa_period:>10.4f}  {'--':>10}")


# Bar chart showing simulated result next to NASA data
planets = [p for p in NASA_PERIODS if p in simulated]
nasa_vals = [NASA_PERIODS[p] for p in planets]
sim_vals  = [simulated[p] for p in planets]

x = range(len(planets))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar([i - width/2 for i in x], nasa_vals, width, label="NASA", color="steelblue", alpha=0.8)
bars2 = ax.bar([i + width/2 for i in x], sim_vals, width, label="Simulated", color="tomato", alpha=0.8)

ax.set_xticks(list(x))
ax.set_xticklabels(planets)
ax.set_ylabel("Orbital Period (Earth years)")
ax.set_title("Experiment 1 – Orbital Periods: Simulated vs NASA")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("exp1_orbital_periods.png", dpi=150)
plt.show()

print("\nPlot saved as exp1_orbital_periods.png")
