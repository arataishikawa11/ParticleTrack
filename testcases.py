import numpy as np
import pandas as pd
from initial_vals import *

# Goal: Create a pandas dataframe that holds projection coordinates such that in the global frame, it satisfies stationary case
# Start with one particle case


# Initial values
pos = np.array([[0.5,0.5,0.5],
                [0.0,0.0,0.0]]) #(x,y,z) in the global frame (mm)
vel = np.array([[1.0,1.0,1.0],
               [0.0,0.0,0.0]]) # u=1, v=w=0 vector field (mm/s)
acc = np.array([[0.0,0.0,0.0],
               [0.0,0.0,0.0]]) # a_x = 1, a_y=a_z=0 acceleration (mm/s^2)


# Generalize to multiple particles with random initial positions/velocities/accelerations
# Set seed
np.random.seed(11)

pos = np.random.uniform(-3, 3, (num_p,3))
vel = np.random.uniform(-2, 2, (num_p,3))
acc = np.random.uniform(-1, 1, (num_p,3))



# Initialize a numpy array that stores x coord in projection frame
# Note that y coord in projection frame does not exist
#x_p = np.zeros(projections)
#x_p[0] = (SDD/(SOD+pos[1]))*pos[0] # Some initial position for x on the projection
#
#z_p = np.full(projections,[pos[2]]) # constant z across projections and object frames (rotation is about z axis)
#
#for i in range(projections-1):
#    if const_acc:
#        pos[0] = (pos[0] + vel[0]*T + acc[0]*T**2)*np.cos(theta) - (pos[1] + vel[1]*T + acc[1]*T**2)*np.sin(theta)
#        pos[1] = (pos[0] + vel[0]*T + acc[0]*T**2)*np.cos(theta) + (pos[1] + vel[1]*T + acc[1]*T**2)*np.sin(theta)
#        pos[2] = pos[2] + vel[2]*T + acc[2]*T**2
#    elif const_vel:
#        pos[0] = (pos[0] + vel[0]*T)*np.cos(theta) - (pos[1] + vel[1]*T)*np.sin(theta)
#        pos[1] = (pos[0] + vel[0]*T)*np.cos(theta) + (pos[1] + vel[1]*T)*np.sin(theta)
#        pos[2] = pos[2] + vel[2]*T
#    x_p[i+1] = (SDD*pos[0])/(SOD + ((np.cos(theta)-1)/np.sin(theta))*pos[0])

#print(x_p, z_p)
#print(x_p[1]-x_p[0])
#print(x_p[2]-x_p[1])
#print(x_p[3]-x_p[2])
#print(x_p[4]-x_p[3])
#print(x_p[5]-x_p[4])

# Generate synthetic data

# Initialize x_p, z_p, x_o, y_o, z_o
x_p = np.zeros((num_p,projections))
z_p = np.zeros((num_p,projections))

for p in range(num_p):
    for i in range(projections):

        #print('x= %2.4f, y=%2.4f, z=%2.4f' %(pos[0], pos[1], pos[2]))
        # Calculate x_pi (projection coord in ith frame)
        x_p[p][i] = (SDD/(SOD+pos[p][1]))*pos[p][0]
        z_p[p][i] = (SDD/(SOD+pos[p][1]))*pos[p][2]

        
        #print('x_p= %2.4f, z_p=%2.4f' %(x_p[i],z_p[i]))
        # Find next position
        x_rot = (pos[p][0] + vel[p][0]*T + 0.5*acc[p][0]*T**2)*np.cos(theta) - (pos[p][1] + vel[p][0]*T + 0.5*acc[p][1]*T**2)*np.sin(theta) #x_o
        y_rot = (pos[p][0] + vel[p][0]*T + 0.5*acc[p][0]*T**2)*np.sin(theta) + (pos[p][1] + vel[p][1]*T + 0.5*acc[p][1]*T**2)*np.cos(theta) #y_o
        z_rot = pos[p][2] + vel[p][2]*T + 0.5*acc[p][2]*T**2 #z_o

        # Update current position
        pos[p][0], pos[p][1], pos[p][2] = x_rot, y_rot, z_rot
        print('x= %2.4f, y=%2.4f, z=%2.4f' %(pos[p][0], pos[p][1], pos[p][2]))

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






