import numpy as np

# Initialize Values
SDD = 500 #mm source to detector
SOD = 250 #mm source to object
T = 0.1 # time step across frames (1 sec per one time step)
theta = np.deg2rad(T*5) # radians per time step (this is actually delta theta) [40 degrees per time step]

# Number of frames
projections = 5
assert projections >= 1

# Number of particles
num_p = 5
assert num_p >= 1