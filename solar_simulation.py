import math
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random as random
import json
from matplotlib.animation import FuncAnimation
import csv
import os

G = 4 * (math.pi ** 2) # in terms of AU, yr, M_sun units

class Body:
    """A single celestial body"""
    def __init__(self, name, colour, mass, orbital_radius, is_sun=False, is_satellite=False, sat_position=None, sat_velocity=None):
        """Initialising the name, mass, colour, and motion of the body according to the information in the json file
        The sun has a fixed position at the origin with velocity = zero"""

        self.name = str(name)
        self.mass = float(mass)
        self.colour = str(colour)
        self.is_sun = is_sun
        self.is_satellite = is_satellite
        self.orbital_radius = float(orbital_radius)

        if self.is_sun:
            self.position = np.array([0.0, 0.0])
            self.velocity = np.array([0.0, 0.0])
        elif self.is_satellite:
            if sat_position is None or sat_velocity is None:
                raise ValueError("Satellite position and velocity must be provided for each satellite body")
            self.position = np.array(sat_position, dtype=float)
            self.velocity = np.array(sat_velocity, dtype=float)
        else:
            self.position = np.array([self.orbital_radius, 0.0]) # placing on positive x-axis
            self.velocity = np.array([0.0, math.sqrt(G / self.orbital_radius)]) # using v = sqrt(G/r)
        
        self.acc_current = np.array([0.0, 0.0])
        self.acc_prev = np.array([0.0, 0.0])

        # Orbital period detection when y crossesfrom negative to positive on the positive x axis
        self._prev_y = None
        self._period_detected = False # set to True once the first orbit is found
        self.orbital_period = None # will be set once the orbit is detected
        
    def update_position_beeman(self, dt):
        """Updating the position once using the Beeman method, using acc_current and acc_prev"""
        self.position = (self.position + self.velocity * dt + (dt ** 2 / 6) * (4 * self.acc_current - self.acc_prev))

    def update_velocity_beeman(self, new_acc, dt):
        """Beeman velocity update using new, current, and previous accelerations"""
        self.velocity = (self.velocity + (dt / 6) * (2 * new_acc + 5 * self.acc_current - self.acc_prev))
        self.acc_prev = self.acc_current.copy()
        self.acc_current = new_acc.copy()

    def update_position_euler_cromer(self, dt):
        """Euler-Cromer: position updates after velocity"""
        self.position = self.position + self.velocity * dt

    def update_velocity_euler_cromer(self, new_acc, dt):
        """Euler-Cromer: velocity updated first"""
        self.velocity = self.velocity + new_acc * dt
        self.acc_current = new_acc.copy()

    def update_position_direct_euler(self, dt):
        """Direct Euler: position updated using previous velocity"""
        self.position = self.position + self.velocity * dt

    def update_velocity_direct_euler(self, new_acc, dt):
        """Direct Euler: velocity updates after position"""
        self.velocity = self.velocity + new_acc * dt
        self.acc_current = new_acc.copy()

    def calc_KE(self):
        """Using kinetic energy = 1/2mv**2"""
        return 0.5 * self.mass * norm(self.velocity) ** 2

    def check_orbital_period(self, current_time):
        """
        This method should be called every step. It detects when the body crosses y = 0 moving upward on the positive x axis, meaning it has completed a full orbit by passing its starting point.
        Returns True the step the orbit is first completed, False otherwise.
        """
        # Sun and satellites do not have a simple orbital period to detect
        if self.is_sun or self.is_satellite or self._period_detected:
            return False

        current_y = self.position[1]

        if self._prev_y is None:
            self._prev_y = current_y
            return False

        crossed_upward = (self._prev_y < 0.0) and (current_y >= 0.0)
        on_positive_side = self.position[0] > 0.0

        if crossed_upward and on_positive_side:
            self.orbital_period = current_time
            self._period_detected = True
            self._prev_y = current_y
            return True

        self._prev_y = current_y
        return False

    
class Simulation:
    """Manages the n-body solar system simulation"""
    def __init__(self, timestep, num_iterations, method="beeman",
                 energy_output_file="energy_output.csv", animate=True):
        
        self.timestep = float(timestep)
        self.num_iterations = int(num_iterations)
        self.method = method.lower().strip() # making sure the input works
        self.energy_output_file = energy_output_file
        self.animate = animate
        self.body_list = []
        self.time = 0.0 # current simulation time in years

        self.time_history = []
        self.energy_history = []

    def read_input_data(self, filename="parameters_solar.json"):
        """Read planet/sun data from a json file and create body objects"""
        with open(filename) as f:
            data = json.load(f)

        for entry in data["bodies"]:
            is_sun = entry.get("is_sun", False)
            body = Body(
                name=entry["name"],
                colour=entry["colour"],
                mass=entry["mass"],
                orbital_radius=entry.get("orbital_radius", 0.0),
                is_sun=is_sun,
            )
            self.body_list.append(body)
        
        print(f"Loaded {len(self.body_list)} bodies: "
              f"{[b.name for b in self.body_list]}")
    
    def add_satellite(self, name, colour, mass, position, velocity):
        """Adds a satellite with a position and velocity vector to be used for experiment 3"""
        sat = Body(name=name, colour=colour, mass=mass, orbital_radius=norm(position), sat_position=position, sat_velocity=velocity, is_satellite=True)
        self.body_list.append(sat)
        print(f"Added satellite '{name}' at position {position} with velocity {velocity}")

    def calc_acceleration(self, body):
        """Calculates the acceleration due to gravity on the body due to all other bodies in the simulation (many-body sum)"""
        acc = np.array([0.0, 0.0])
        for other in self.body_list:
            if other is body:
                continue
            r_vec = other.position - body.position # vector from body to other
            r_mag = norm(r_vec)
            if r_mag == 0:
                continue
            acc += G * other.mass / r_mag ** 2 * (r_vec / r_mag)
        return acc

    def calc_PE(self):
        """Total gravitational potential energy of the system"""
        pe = 0.0
        for i, body_i in enumerate(self.body_list):
            for body_j in self.body_list[i + 1:]:
                r = norm(body_j.position - body_i.position)
                if r > 0:
                    pe -= G * body_i.mass * body_j.mass / r
        return pe

    def calc_tot_energy(self):
        """Total energy = KE + PE"""
        ke = sum(b.calc_KE() for b in self.body_list)
        pe = self.calc_PE()
        return ke + pe
    
    def initialise_accelerations(self):
        """Set acc_current for all bodies before the first step. For Beeman you also need acc_prev which is approximated as equal to acc_current at time t=0"""
        for body in self.body_list:
            body.acc_current = self.calc_acceleration(body)
            body.acc_prev = body.acc_current.copy()

    def step_forward(self):
        """Advance the simulation by one time step using the chosen method."""
        dt = self.timestep

        if self.method == "beeman":
            for body in self.body_list:
                body.update_position_beeman(dt)

            new_accs = [self.calc_acceleration(b) for b in self.body_list]

            for body, new_acc in zip(self.body_list, new_accs):
                body.update_velocity_beeman(new_acc, dt)

        elif self.method == "euler_cromer":
            new_accs = [self.calc_acceleration(b) for b in self.body_list]

            for body, new_acc in zip(self.body_list, new_accs):
                body.update_velocity_euler_cromer(new_acc, dt)
            
            for body in self.body_list:
                body.update_position_euler_cromer(dt)
            
        elif self.method == "direct_euler":
            new_accs = [self.calc_acceleration(b) for b in self.body_list]

            for body in self.body_list:
                body.update_position_direct_euler(dt)

            for body, new_acc in zip(self.body_list, new_accs):
                body.update_velocity_direct_euler(new_acc, dt)
        
        else:
            raise ValueError(f"Unknown integration method: {self.method}")
        
        self.time += dt

    def run_simulation(self, energy_write_interval=10):
        """Run the full simulation without animation. Useful for experiments that only need data."""
        self.initialise_accelerations()

        # Record the initial energy at t=0
        E0 = self.calc_tot_energy()
        self.time_history.append(self.time)
        self.energy_history.append(E0)

        with open(self.energy_output_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["time_years", "total_energy"])
            writer.writerow([f"{self.time:.6f}", f"{E0:.6f}"])

            for step in range(1, self.num_iterations + 1):
                self.step_forward()

                for body in self.body_list:
                    if body.check_orbital_period(self.time):
                        print(f"  {body.name:10s} orbital period = "
                        f"{body.orbital_period:.4f} Earth years")

                if step % energy_write_interval == 0:
                    E = self.calc_tot_energy()
                    self.time_history.append(self.time)
                    self.energy_history.append(E)
                    writer.writerow([f"{self.time:.6f}", f"{E:.6f}"])

        print("\nOrbital period summary:")
        for body in self.body_list:
            if body.is_sun or body.is_satellite:
                continue
            if body.orbital_period is not None:
                print(f"  {body.name:10s}: {body.orbital_period:.4f} yr")
            else:
                print(f"  {body.name:10s}: period not detected in simulation")
    
    def run_animated(self, energy_write_interval=10):
        """Run the simulation with a matplotlib animation"""
        self.initialise_accelerations()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")
        ax.set_aspect("equal")
        limit = 6.0 # in AU, should be adjusted if outer planets are added
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_title("Solar System Simulation", color="white", fontsize=12)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")

        # patches represent each body as a filled circle
        patch_list = []
        for body in self.body_list:
            size = 0.18 if body.is_sun else 0.07
            patch = plt.Circle(tuple(body.position), size, color=body.colour, zorder=3)
            ax.add_patch(patch)
            patch_list.append(patch)

        trail_x = {b.name: [] for b in self.body_list}
        trail_y = {b.name: [] for b in self.body_list}
        trail_lines = {}
        for body in self.body_list:
            line, = ax.plot([], [], color=body.colour, lw=0.5, alpha=0.4, zorder=2)
            trail_lines[body.name] = line

        for body in self.body_list:
            if body.check_orbital_period(self.time):
                print(f"  {body.name:10s} orbital period = "
                f"{body.orbital_period:.4f} Earth years")
                        
        time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, color="white", fontsize=9)

        csvfile = open(self.energy_output_file, "w", newline="")
        writer = csv.writer(csvfile)
        writer.writerow(["time_years", "total_energy"])

        E0 = self.calc_tot_energy()
        self.time_history.append(self.time)
        self.energy_history.append(E0)
        writer.writerow([f"{self.time:.6f}", f"{E0:.6f}"])

        step_counter = [0]

        # _frame is the frame number passed in by FuncAnimation
        def update(_frame):
            self.step_forward()
            step_counter[0] += 1

            # Update each body's circle position and trail
            for body, patch in zip(self.body_list, patch_list):
                patch.center = tuple(body.position)

                trail_x[body.name].append(body.position[0])
                trail_y[body.name].append(body.position[1])
                max_trail = 500
                trail_x[body.name] = trail_x[body.name][-max_trail:]
                trail_y[body.name] = trail_y[body.name][-max_trail:]
                trail_lines[body.name].set_data(trail_x[body.name], trail_y[body.name])
                    
            if step_counter[0] % energy_write_interval == 0:
                E = self.calc_tot_energy()
                self.time_history.append(self.time)
                self.energy_history.append(E)
                writer.writerow([f"{self.time:.6f}", f"{E:.6f}"])

            time_text.set_text(f"t = {self.time:.2f} yr")
            return patch_list + list(trail_lines.values()) + [time_text]
        
        # Store as self._anim so Python does not get rid of it mid run
        self._anim = FuncAnimation(fig, update, frames=self.num_iterations, interval=1, blit=True, repeat=False)

        # event is passed in by matplotlib when the window is closed
        def on_close(event):
            csvfile.close()

        fig.canvas.mpl_connect("close_event", on_close)
        plt.show()
        csvfile.close()

        print("\nOrbital period summary:")
        for body in self.body_list:
            if body.is_sun or body.is_satellite:
                continue
            if body.orbital_period is not None:
                print(f"  {body.name:10s}: {body.orbital_period:.4f} yr")
            else:
                print(f"  {body.name:10s}: period not detected in simulation window")
                
    def plot_energy(self, title=None, show=True):
        """Plot total energy vs time from the stored history"""
        if not self.time_history: 
            print("No energy data to plot.")
            return

        plt.figure()
        plt.plot(self.time_history, self.energy_history, lw=0.8)
        plt.xlabel("Time (Earth years)")
        plt.ylabel("Total Energy (M_solar AU^2 yr^-2)")
        plt.title(title or f"Total Energy vs Time [{self.method}]")
        plt.tight_layout()
        if show:
            plt.show()


DEFAULT_JSON = """{
  "bodies": [
    {"name": "Sun",     "colour": "yellow",     "mass": 1.0,       "orbital_radius": 0.0,   "is_sun": true},
    {"name": "Mercury", "colour": "gray",        "mass": 1.652e-7,  "orbital_radius": 0.387},
    {"name": "Venus",   "colour": "orange",      "mass": 2.447e-6,  "orbital_radius": 0.723},
    {"name": "Earth",   "colour": "deepskyblue", "mass": 3.003e-6,  "orbital_radius": 1.0},
    {"name": "Mars",    "colour": "red",         "mass": 3.213e-7,  "orbital_radius": 1.524},
    {"name": "Jupiter", "colour": "sandybrown",  "mass": 9.543e-4,  "orbital_radius": 5.203}
  ]
}
"""

def ensure_json(filename="parameters_solar.json"):
    """Write the default JSON file to disk if it does not already exist. Called at the top of every run/experiment script."""
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write(DEFAULT_JSON)
        print(f"Created {filename}")
    return filename

if __name__ == "__main__":
    json_file = "parameters_solar.json"
    if not os.path.exists(json_file):
        ensure_json()
        print(f"Created {json_file}")
    
    dt = 1 / 600 # 600 steps a year
    years = 12 # run for 12 years
    n_steps = int(years / dt)

    sim = Simulation(
        timestep=dt,
        num_iterations=n_steps,
        method="beeman",
        energy_output_file="energy_beeman.csv",
        animate=True,
    )

    sim.read_input_data(json_file)

    if sim.animate:
        sim.run_animated(energy_write_interval=50)
    else:
        sim.run_simulation(energy_write_interval=50)

    sim.plot_energy()
