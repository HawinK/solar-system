import math
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from solar_simulation import Simulation, ensure_json, G
import csv


JSON_FILE = ensure_json("parameters_solar.json")
DT = 1.0 / 1000
YEARS = 3 
N_STEPS = int(YEARS / DT)

def earth_circular_speed():
    """Earth's circular orbit speed in AU/yr (around M_sun = 1)."""
    r_earth = 1.0
    return math.sqrt(G / r_earth)

def find_body(sim, name):
    """Return the Body object with the given name."""
    for body in sim.body_list:
        if body.name == name:
            return body
    raise ValueError(f"Body '{name}' not found in simulation.")

v_earth = earth_circular_speed()
print(f"Earth circular speed: {v_earth:.4f} AU/yr\n")

# Hohmann estimate for reference
r_earth = 1.0
r_mars = 1.524
v_hohmann = v_earth * math.sqrt(2 * r_mars / (r_earth + r_mars))
print(f"Hohmann transfer estimate: {v_hohmann:.4f} AU/yr  "
      f"(factor {v_hohmann/v_earth:.4f} x Earth speed)\n")

boost_factors = [round(1.05 + i * 0.01, 3) for i in range(16)]   # 1.05 to 1.20

best_distance = float("inf")
best_factor = None
best_time_yr = None
best_sim = None

print(f"{'Boost':>7}  {'v_launch (AU/yr)':>18}  "
      f"{'Min dist to Mars (AU)':>23}  {'Time of closest (yr)':>22}")
print("-" * 76)

for factor in boost_factors:
    v_launch = factor * v_earth

    sim = Simulation(
        timestep = DT,
        num_iterations = N_STEPS,
        method = "beeman",
        energy_output_file = "energy_exp3_temp.csv",
    )
    sim.read_input_data(JSON_FILE)

    earth_start = np.array([r_earth + 0.001, 0.0])

    sat_velocity = np.array([0.0, v_launch])

    sim.add_satellite(
        name = "Probe",
        colour = "white",
        mass = 1e-25,
        position = earth_start.tolist(),
        velocity = sat_velocity.tolist(),
    )

    sim.initialise_accelerations()

    mars = find_body(sim, "Mars")
    probe = find_body(sim, "Probe")
    min_dist = float("inf")
    time_at_min = 0.0

    with open("energy_exp3_temp.csv", "w", newline="") as f:
        pass 

    for i in range(N_STEPS):
        sim.step_forward()
        d = norm(probe.position - mars.position)
        if d < min_dist:
            min_dist = d
            time_at_min = sim.time

    print(f"{factor:>7.3f}  {v_launch:>18.4f}  {min_dist:>23.4f}  "
          f"{time_at_min:>22.4f}")

    if min_dist < best_distance:
        best_distance = min_dist
        best_factor = factor
        best_time_yr = time_at_min
        best_sim = (factor, v_launch)

print(f"\nBest boost factor : {best_factor}")
print(f"Best launch speed : {best_sim[1]:.4f} AU/yr")
print(f"Closest approach  : {best_distance:.4f} AU  "
      f"({best_distance * 1.496e8:.3e} km)")
print(f"Time of closest   : {best_time_yr:.4f} yr  "
      f"({best_time_yr * 365.25:.1f} days)")

# NASA Perseverance took about 203 days = 0.556 yr
perseverance_yr = 203 / 365.25
print(f"\nNASA Perseverance journey time: {perseverance_yr:.3f} yr (203 days)")
print(f"Our satellite journey time    : {best_time_yr:.3f} yr "
      f"({best_time_yr * 365.25:.0f} days)")

print(f"\nRe-running best case (factor = {best_factor}) for trajectory plot ...")

sim_best = Simulation(
    timestep = DT,
    num_iterations = N_STEPS,
    method = "beeman",
    energy_output_file = "energy_exp3_best.csv",
)
sim_best.read_input_data(JSON_FILE)

best_v_launch = best_factor * v_earth
sim_best.add_satellite(
    name = "Probe",
    colour = "white",
    mass = 1e-25,
    position = [r_earth + 0.001, 0.0],
    velocity = [0.0, best_v_launch],
)

sim_best.initialise_accelerations()
mars_best = find_body(sim_best, "Mars")
probe_best = find_body(sim_best, "Probe")
earth_best = find_body(sim_best, "Earth")

traj = {b.name: {"x": [], "y": []} for b in sim_best.body_list} # trajectories

for i in range(N_STEPS):
    sim_best.step_forward()
    for body in sim_best.body_list:
        traj[body.name]["x"].append(body.position[0])
        traj[body.name]["y"].append(body.position[1])

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_facecolor("black")
fig.patch.set_facecolor("black")
ax.set_aspect("equal")
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)
ax.set_title(f"Experiment 3 – Satellite to Mars (boost = {best_factor}x)",
             color="white")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("white")

ax.plot(0, 0, "o", color="yellow", markersize=12, label="Sun")

colours = {"Earth": "deepskyblue", "Mars": "red", "Probe": "white",
           "Venus": "orange", "Mercury": "gray", "Jupiter": "sandybrown"}

for name, data in traj.items():
    if name == "Sun":
        continue
    colour = colours.get(name, "white")
    alpha = 1.0 if name in ("Earth", "Mars", "Probe") else 0.3
    lw = 1.2 if name in ("Earth", "Mars", "Probe") else 0.5
    ax.plot(data["x"], data["y"], color=colour, lw=lw, alpha=alpha,
            label=name)

ax.legend(facecolor="#222", labelcolor="white", fontsize=8)
plt.tight_layout()
plt.savefig("exp3_trajectory.png", dpi=150)
plt.show()
print("Trajectory plot saved as exp3_trajectory.png")
