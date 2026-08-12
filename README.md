# solar-system
My final project for Computer Simulation. An N-Body simulation testing different integration types for a model of the solar system.

### Main file: solar_simulation.py
This should be run directly in order to produce an animation of planets orbiting the sun. It will also produce a graph of total energy against time using the Beeman method of integration.
The Beeman method is set as the default method of integration so, if you'd like to try any other methods, please specify this as the method attribute when creating a simulation object.

### Experiment 1, Orbital Periods: experiment1_orbital_periods.py
Run this file directly to use the Beeman simulation and compare the detected orbital periods against NASA's published values.

### Experiment 2, Energy Conservation and Alternative Integration Methods: experiment2_energy_conservation.py
Run this file directly to run the simulation three times (Beeman, Euler-Cromer, Direct Euler) and compare how well each method conserves total energy over time.

### Experiment 3, Satellite to Mars: experiment3_satellite_to_mars.py
Run this file directly to launch a small probe from just outside Earth and search for an initial velocity that brings it close to Mars.
The satellite is launched at a tangent to the Earth in the positive y direction.
A range of speed multipliers are tested to find which one gets the probe closest to Mars.
