import numpy as np
import pandas as pd

# Goal: Create a pandas dataframe that holds projection coordinates such that in the global frame, it satisfies stationary case
# Start with one particle case

initial_pos = (1,1,1) #(x,y,z) in the global frame, this stays constant throughout

# 5 projections needed to solve unknowns
projections = 5

# Initialize Values
SDD = 500 #mm source to detector
SOD = 250 #mm source to object
T = 0.1 # time step across frames (0.1 sec per one time step)
theta = np.deg2rad(T*0.5) # radians per time step (this is actually delta theta)

# Initialize a numpy array that stores x coord in projection frame
# Note that y coord in projection frame does not exist
x_p = np.zeros(5)
x_p[0] = (SDD/(SOD+initial_pos[1]))*initial_pos[0] # Some initial position for x on the projection

z_p = np.full(5,[initial_pos[2]]) # constant z across projections and object frames (rotation is about z axis)
angle = theta # track the angle (radians)

for i in range(projections-1):
    x_p[i+1] = (SDD*initial_pos[0])/(SOD + ((np.cos(angle)-1)/np.sin(angle))*initial_pos[0])
    angle += theta

#print(x_p, z_p)

# Convert to pandas DataFrame

# Create frames column
frames = np.arange(5) # 5 projections
#print(frames)

particle = np.zeros(5) # only one particle (particle id = 0)
#print(particle)

y_p = np.full(5, np.nan) # y coord in projection doesn't exist (2D)
#print(y_p)

# Construct the DataFrame
data_array = np.array((x_p, y_p, z_p, frames, particle)).T
#print(data_array)
coords_stationary = pd.DataFrame(data_array, columns = ['x','y','z','frame','particle'])
print(coords_stationary)