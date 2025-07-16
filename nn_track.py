import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix


# Generate Synthetic Data
np.random.seed(1111)
n_particles = 3
f1 = np.random.rand(n_particles, 3) * 5 # Frame 1, n_particle rows and 3 columns (x,y,z)


# Construct data structures. Dictionary vs pandas DataFrame
particles = {} # Dictionary with key-value pairs ("p#: [x,y,z]")
for i in range(len(f1)):
    print(i)
    particles["p"+str(i+1)] = f1[i]

print(particles)


# Pandas Dataframe to store particle information
df_particles = pd.DataFrame({
    'frame': [str(0) for i in range(n_particles)],
    'particle_id': [str(i) for i in range(n_particles)],
    'coords': [coord for coord in f1]
})
print("Original Dataframe:")
print(df_particles)


#Displacement Methods:

u = 2 # constant velocity
dt = 5 #time step
a = 3 # constant acceleration

# Stationary Case
stationary = False # For stationary case, we only need one frame, with no displacement of particles
if stationary:
    f2 = f1
    f3 = f2

# Displacements only in x direction
# Linear Case: no acceleration
linear = True
displacement = [u*5, 0, 0]
if linear:
    f2 = f1 + displacement
    f3 = f2 + displacement

# Constant acceleration case:
const_acc = False
displacement= [u*5 + 0.5*a*dt**2, 0, 0]
if const_acc:
    f2 = f1 + displacement
    f3 = f2 + displacement

frames = [f1,f2,f3]


# Perform Matching (sketch for stationary example)
dist = distance_matrix(f1, f2) # Ex: Row 1 corresponds to particle 1 euclid. distance to particle 1,2,3 
matches = np.argmin(dist, axis = 1) # Gives a vector of values that picks out best indices for lowest distances. Ex: [0 1 2] --> first coord in frame x (index 0) is closest to first coord in frame y [0]


# 3D Visualization 
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

for i in range(len(frames)):
    print(i)
    ax.scatter(*frames[i].T, label = "Frame " + str(i))

print(type(frames[0][0]))


# Algorithm
for i in range(len(frames) - 1): # Loop over frames

    # Determine matches
    dist = distance_matrix(frames[i],frames[i+1])
    matches = np.argmin(dist, axis = 1)

    # Plot trajectories
    for j, k in enumerate(matches): 
        
        # jth index particle coords in current ith frame to kth index particle in next frame i+1
        cur_x, cur_y, cur_z = frames[i][j,0], frames[i][j,1], frames[i][j,2]
        new_x, new_y, new_z = frames[i+1][k,0], frames[i+1][k,1], frames[i+1][k,2]

        ax.plot([cur_x, new_x],
                [cur_y, new_y],
                [cur_z, new_z],
                'k--', linewidth = 1)
        
        # Update DataFrame
        new_row = [
            {'frame': str(i+1), 'particle_id': str(k), 'coords': [new_x, new_y, new_z]}
        ]
        df_particles = pd.concat([df_particles, pd.DataFrame(new_row)], ignore_index=True)

print("New Dataframe:")
print(df_particles)


# Set a title
ax.set_title('3D Plot of Stationary Particles')

# Generate Legend
ax.legend()

# Display the plot
plt.show()