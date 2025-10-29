import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import lsq_linear, least_squares
from tabulate import tabulate
from testcases import pos,vel,acc,flags # Import known initial positions and velocities for cross-referencing
from trackpy_test import coords_test # Import data from preprocessing
from initial_vals import * # Import initial values

coords = coords_test

### BLOCKS ###

# We note by pattern-matching that there are 3 blocks that occur every time
# Yellow
block1 = np.array([[T*np.cos(theta), -T*np.sin(theta), 0, 0.5*np.cos(theta)*T**2, -0.5*np.sin(theta)*T**2, 0],
                  [T*np.sin(theta), T*np.cos(theta), 0, 0.5*np.sin(theta)*T**2, 0.5*np.cos(theta)*T**2, 0],
                  [0, 0, T, 0, 0, 0.5*T**2]])
print("block 1:\n" + tabulate(block1))

# Green
def block2(frame, p_id): # input is the projection number (int) and the particle id (int), output is the block for that projection

    frame_coords = coords[coords['frame']==frame] # all particle coords in frame
    x_pi = frame_coords.iloc[p_id,0] # x coord indexed by particle id in frame
    z_pi = frame_coords.iloc[p_id,1] # z coord indexed by particle id in frame

    block = np.array([[1, -x_pi/SDD, 0],
                       [0, -z_pi/SDD, 1]])

    return block
print("block2: \n" + str(block2(0, 0)))

# Blue
block3 = np.array([[np.cos(theta), -np.sin(theta), 0, -1.0, 0, 0],
                   [np.sin(theta), np.cos(theta), 0, 0, -1.0, 0],
                   [0, 0, 1.0, 0, 0, -1.0]])
print("block 3:\n" + tabulate(block3))

### END BLOCKS ###

### b VECTOR ASSEMBLY ###

# Define a function for extending our vector of constants (known values)
def extend(frame, p_id): # input is the projection number (int) and the particle id (int), output is a vector of constants

    frame_coords = coords[coords['frame']==frame] # all particle coords in frame
    x_pi = frame_coords.iloc[p_id,0] # x coord indexed by particle id in frame
    z_pi = frame_coords.iloc[p_id,1] # z coord indexed by particle id in frame


    return np.array([0,0,0, SOD/SDD*x_pi, SOD/SDD*z_pi])
print("vector of constants:\n" + str(extend(1, 0))) # First particle has id p_id = 0

### END b VECTOR ASSEMBLY ###

### DECOUPLED Z MATRIX and CONSTANTS ###

# Magnification Equation for z
def mag_z(frame, p_id):
    frame_coords = coords[coords['frame']==frame] # all particle coords in frame
    z_pi = frame_coords.iloc[p_id,1] # z coord indexed by particle id in frame

    return np.array([-z_pi/SDD, 1])

# Dynamics block for z
z_dynamics = np.array([T, 0.5*T**2, 0, 1, 0, -1]) # dynamics for z

# Define a function for extending our vector of constants (known values)
def extend_z(frame, p_id): # input is the projection number (int) and the particle id (int), output is a vector of constants

    frame_coords = coords[coords['frame']==frame] # all particle coords in frame
    z_pi = frame_coords.iloc[p_id,1] # z coord indexed by particle id in frame

    return np.array([0,(SOD/SDD)*z_pi])

print("initial vector of constants:\n" + str(extend(0, 0))) # First particle has id p_id = 0

### END DECOUPLED Z MATRIX and CONSTANTS ###

### Algorithm ###


# Store results from Decoupled Z Calculation
z_results = []

z_p1 = coords.iloc[:,1].to_numpy() # Grab all z coords in first frame (frame 0)
for p in range(num_p):
    rows = 1
    cols = 4

    M_z = np.zeros((rows,cols)) # initialize/reset the matrix

    M_z[:,-2:] = mag_z(0, p) # Initial Block for the pth particle

    b_z = np.array([0]) # Initial vector for z calculations

    for i in range(projections-1):

        # add 2 rows and 2 rows each time
        new_rows = 2
        new_cols = 2
    
        rows += new_rows
        cols += new_cols
    
    
        # Enlarge matrix by new_rows down, new_cols right. fill these w/ 0
        M_z = np.pad(M_z, ((0,new_rows),(0,new_cols)), mode = 'constant', constant_values=0)

        # Insert blocks
        M_z[rows-new_rows:-1, :6] = z_dynamics
        M_z[-1:, -2:] = mag_z(i+1, p) # i+1 because we already built the first initial block 2
        
        # Extend our vector of constants
        b_z = np.concatenate((b_z,extend_z(i+1, p))) # i+1 because we already initilized b for 1 projection


    # Non-linear least_squares
    def func(x, A, b):
        residuals = (A @ x - b)
        return residuals

    x0 = np.zeros(cols) # Initial guess

    # Solve using least squares
    res_lsq = least_squares(func, x0, args=(M_z, b_z))
    z_results.append(res_lsq.x)

# # We will store our results in a list. Each entry will be a numpy array of unknowns with index corresponding to the particle id. (result[0] corresponds to particle_id = 0)
# result = []


labels = ['w', 'a_z']
for i in range(projections):
    labels.append('y_' + str(i))
    labels.append('z_' + str(i))

df = pd.DataFrame(z_results, columns=labels)
print("Final DataFrame of results: (each row corresponds to a particle)")
print(df)

comparison = pd.DataFrame({
    'Known Position Y': pos[:,1],
    'Known Position Z': pos[:,2],
    'Estimated Position Y': df['y_' + str(projections-1)],
    'Estimated Position Z': df['z_' + str(projections-1)],
    'Error in Position Y': pos[:,1] - df['y_'+str(projections-1)],
    'Error in Position Z': pos[:,2] - df['z_'+str(projections-1)],
    'Known Velocity W': vel[:,2],
    'Estimated Velocity W': df['w'],
    'Error in Velocity W': vel[:,2] - df['w'],
    'Known Acceleration a_z': acc[:,2],
    'Estimated Acceleration a_z': df['a_z'],
    'Error in Acceleration Z': acc[:,2] - df['a_z']
})

print(comparison[['Known Position Y', 'Estimated Position Y', 'Error in Position Y']])
print(comparison[['Known Position Z', 'Estimated Position Z', 'Error in Position Z']])
print(comparison[['Known Velocity W', 'Estimated Velocity W', 'Error in Velocity W']])
print(comparison[['Known Acceleration a_z', 'Estimated Acceleration a_z']])


### END for now ###


# x_p1 = coords.iloc[:,0].to_numpy() # Grab all x coords in first frame (frame 0)
# z_p1 = coords.iloc[:,1].to_numpy() # Grab all z coords in first frame
# for p in range(num_p):

#     # starts with 2 rows and 9 cols every time
#     rows = 2
#     cols = 9

#     M = np.zeros((rows,cols)) # reset the matrix

#     M[0:, -3:] = block2(0, p) # Initial Block for the pth particle

    
#     # Initialize vector
#     # vector of known constant values (2 for one projection)
#     b = np.zeros(2)
#     b[0], b[1] = (SOD/SDD)*x_p1[p], (SOD/SDD)*z_p1[p] 
    
#     for i in range(projections-1):
#         # add 5 rows and 3 cols each time
#         new_rows = 5
#         new_cols = 3
    
#         rows += new_rows
#         cols += new_cols
    
    
#         # Enlarge matrix by new_rows down, new_cols right. fill these w/ 0
#         M = np.pad(M, ((0,new_rows),(0,new_cols)), mode = 'constant', constant_values=0)

#         # Insert blocks
#         M[rows-new_rows:-2, :6] = block1
#         M[-2:, -3:] = block2(i+1, p) # i+1 because we already built the first initial block 2
#         M[rows-new_rows:-2, 6+i*3:] = block3
        
#         # Extend our vector of constants
#         b = np.concatenate((b,extend(i+1, p))) # i+1 because we already initilized b for 1 projection

#     # Implementing one validation check
#     validation = np.zeros(cols)
#     validation[7] = 1.0
#     validation[1] = (projections-1)*T
#     validation[4] = 0.5*(projections-1)*T**2
#     validation[-2] = -1.0


#     print("M @ x shape: " + str((M @ np.zeros(cols)).shape))
#     # Non-linear least_squares
#     def func(x, A, b):
#         residuals = np.linalg.norm((A @ x - b))
#         return residuals

#     x0 = np.zeros(cols) # Initial guess

#     # Solve using least squares
#     res_lsq = least_squares(func, x0, args=(M, b))
#     result.append(res_lsq.x)



# labels = ['u', 'v', 'w', 'a_x', 'a_y', 'a_z']
# for i in range(projections):
#     labels.append('x_' + str(i))
#     labels.append('y_' + str(i))
#     labels.append('z_' + str(i))

# df = pd.DataFrame(result, columns=labels)
# print("Final DataFrame of results: (each row corresponds to a particle)")
# print(df)

# # Cross reference results with known initial positions and velocities
# # Note that we only have access to initial positions and velocities, not accelerations
# print("\n Cross-referencing results with known initial positions and velocities:")
# comparison = pd.DataFrame({
#     'Known Position X': pos[:,0],
#     'Known Position Y': pos[:,1],
#     'Known Position Z': pos[:,2],
#     'Estimated Position X': df['x_' + str(projections-1)],
#     'Estimated Position Y': df['y_' + str(projections-1)],
#     'Estimated Position Z': df['z_' + str(projections-1)],
#     'Error in Position X': pos[:,0] - df['x_'+str(projections-1)],
#     'Error in Position Y': pos[:,1] - df['y_'+str(projections-1)],
#     'Error in Position Z': pos[:,2] - df['z_'+str(projections-1)],
#     'Known Velocity U': vel[:,0],
#     'Known Velocity V': vel[:,1],
#     'Known Velocity W': vel[:,2],
#     'Estimated Velocity U': df['u'],
#     'Estimated Velocity V': df['v'],
#     'Estimated Velocity W': df['w'],
#     'Error in Velocity U': vel[:,0] - df['u'],
#     'Error in Velocity V': vel[:,1] - df['v'],
#     'Error in Velocity W': vel[:,2] - df['w'],
#     'Known Acceleration a_x': acc[:,0],
#     'Known Acceleration a_y': acc[:,1],
#     'Known Acceleration a_z': acc[:,2],
#     'Estimated Acceleration a_x': df['a_x'],
#     'Estimated Acceleration a_y': df['a_y'],
#     'Estimated Acceleration a_z': df['a_z'],
#     'Error in Acceleration X': acc[:,0] - df['a_x'],
#     'Error in Acceleration Y': acc[:,1] - df['a_y'],
#     'Error in Acceleration Z': acc[:,2] - df['a_z']
# })

# print(comparison[['Known Position X', 'Estimated Position X', 'Error in Position X']])
# print(comparison[['Known Position Y', 'Estimated Position Y', 'Error in Position Y']])
# print(comparison[['Known Position Z', 'Estimated Position Z', 'Error in Position Z']])
# print(comparison[['Known Velocity U', 'Estimated Velocity U', 'Error in Velocity U']])
# print(comparison[['Known Velocity V', 'Estimated Velocity V', 'Error in Velocity V']])
# print(comparison[['Known Velocity W', 'Estimated Velocity W', 'Error in Velocity W']])
# print(comparison[['Known Acceleration a_x', 'Estimated Acceleration a_x']])
# print(comparison[['Known Acceleration a_y', 'Estimated Acceleration a_y']])
# print(comparison[['Known Acceleration a_z', 'Estimated Acceleration a_z']])

# # Note that if results do not respect bounds, assume poor results

# print(df["y_" + str(projections-1)])


# print(b)


##### END ####


