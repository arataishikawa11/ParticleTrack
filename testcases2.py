import numpy as np
import pandas as pd
from initial_vals import *

# Goal: Create a pandas dataframe that holds projection coordinates such that in the global frame, it satisfies stationary case
# Start with one particle case


# Initial values
# pos = np.array([[0.5,0.5,0.5],
#                 [0.0,0.0,0.0]]) #(x,y,z) in the global frame (mm)
# vel = np.array([[1.0,1.0,1.0],
#                [0.0,0.0,0.0]]) # u=1, v=w=0 vector field (mm/s)
# acc = np.array([[0.0,0.0,0.0],
#                [0.0,0.0,0.0]]) # a_x = 1, a_y=a_z=0 acceleration (mm/s^2)


# Generalize to multiple particles with random initial positions/velocities/accelerations
# Set seed
np.random.seed(11)

pos = np.random.uniform(-1, 1, (num_p,3))
vel = np.random.uniform(-1, 1, (num_p,3))
acc = np.random.uniform(-0.5 , 0.5, (num_p,3))
#vel = np.zeros((num_p,3))
# pos = np.full((num_p,3), 0.5)
# vel = np.full((num_p,3), 1.0)
# acc = np.zeros((num_p,3))



# Generate synthetic data

# Initialize x_p, z_p, x_o, y_o, z_o
x_p = np.zeros((num_p,projections))
z_p = np.zeros((num_p,projections))




pos_initial = pos
# decouple all pos --> x, y, z so I can use in algorithm
x_global = np.zeros((num_p, projections)) # (particle id, frame number)
y_global = np.zeros((num_p, projections))
z_global = np.zeros((num_p, projections))

for p in range(num_p):
    for i in range(projections):

        if i == 0:
            x_global[p,i] = pos_initial[p,0]
            y_global[p,i] = pos_initial[p,1]
            z_global[p,i] = pos_initial[p,2]

        # Calculate x_pi (projection coord in ith frame)
        x_p[p,i] = (SDD/(SOD+y_global[p,i]))*x_global[p,i]
        z_p[p,i] = (SDD/(SOD+y_global[p,i]))*z_global[p,i]

        # Find next position
        dx = (x_global[p,i] + vel[p,0]*T + 0.5*acc[p,0]*T**2)*np.cos(theta) - (y_global[p,i] + vel[p,1]*T + 0.5*acc[p,1]*T**2)*np.sin(theta) #x_o
        dy = (x_global[p,i] + vel[p,0]*T + 0.5*acc[p,0]*T**2)*np.sin(theta) + (y_global[p,i] + vel[p,1]*T + 0.5*acc[p,1]*T**2)*np.cos(theta) #y_o
        dz = z_global[p,i] + vel[p,2]*T + 0.5*acc[p,2]*T**2 #z_o

        vel[p,0] += acc[p,0]*T
        vel[p,1] += acc[p,1]*T
        vel[p,2] += acc[p,2]*T
        

        # Update current position
        if i < projections-1:
            x_global[p,i+1] = dx
            y_global[p,i+1] = dy    
            z_global[p,i+1] = dz




# Convert to pandas DataFrame

# Create frames column
frames = np.arange(projections)
frames = np.tile(frames,num_p)

# Construct the DataFrame

#For testing with trackpy, strip away particle id
# Reformat into DataFrame with columns [y,x,frames] (2D)

#data_array = np.array((x_p, z_p, frames, particle)).T
data_array = np.array((x_p.flatten(), z_p.flatten(), frames)).T

#coords_test = pd.DataFrame(data_array, columns = ['x','z','frame','particle'])
coords_test = pd.DataFrame(data_array, columns = ['x','z','frame'])
print(coords_test)
