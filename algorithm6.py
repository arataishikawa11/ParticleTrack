import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import lsq_linear, least_squares
from tabulate import tabulate
#from testcases import pos,vel,acc # Import known initial positions and velocities for cross-referencing
#from testcases2 import x_global, y_global, z_global, vel, acc # Import data from synthetic data generation
from testcases3 import x_global, y_global, z_global, vel, acc # Import data from synthetic data generation
from trackpy_test import coords_test # Import data from preprocessing
from initial_vals import * # Import initial values

coords = coords_test

### BLOCKS ###

# We note by pattern-matching that there are 3 blocks that occur every time
# Yellow
block1 = np.array([[T*np.cos(theta), -T*np.sin(theta), 0, 0.5*np.cos(theta)*T**2, -0.5*np.sin(theta)*T**2, 0],
                  [T*np.sin(theta), T*np.cos(theta), 0, 0.5*np.sin(theta)*T**2, 0.5*np.cos(theta)*T**2, 0],
                  [0, 0, T, 0, 0, 0.5*T**2]])

# Green
def block2(frame, p_id): # input is the projection number (int) and the particle id (int), output is the block for that projection

    frame_coords = coords[coords['frame']==frame] # all particle coords in frame
    x_pi = frame_coords.iloc[p_id,0] # x coord indexed by particle id in frame
    z_pi = frame_coords.iloc[p_id,1] # z coord indexed by particle id in frame

    block = np.array([[1, -x_pi/SDD, 0],
                       [0, -z_pi/SDD, 1]])

    return block

# Blue
block3 = np.array([[np.cos(theta), -np.sin(theta), 0, -1.0, 0, 0],
                   [np.sin(theta), np.cos(theta), 0, 0, -1.0, 0],
                   [0, 0, 1.0, 0, 0, -1.0]])

### END BLOCKS ###

### b VECTOR ASSEMBLY ###

# Define a function for extending our vector of constants (known values)
def extend(frame, p_id): # input is the projection number (int) and the particle id (int), output is a vector of constants

    frame_coords = coords[coords['frame']==frame] # all particle coords in frame
    x_pi = frame_coords.iloc[p_id,0] # x coord indexed by particle id in frame
    z_pi = frame_coords.iloc[p_id,1] # z coord indexed by particle id in frame


    return np.array([0,0,0, SOD/SDD*x_pi, SOD/SDD*z_pi])

### END b VECTOR ASSEMBLY ###


### DECOUPLED Z and X MATRIX and CONSTANTS ###

### DECOUPLED Z MATRIX BLOCKS ###

# Magnification Equation for z
def mag_z(frame, p_id):
    frame_coords = coords[coords['frame']==frame] # all particle coords in frame
    z_pi = frame_coords.iloc[p_id,1] # z coord indexed by particle id in frame

    return np.array([-z_pi/SDD, 1])

# Dynamics block for z
z_dynamics = np.array([T, 0.5*T**2]) # dynamics for z

# Define a function for extending our vector of constants (known values)
def extend_z(frame, p_id): # input is the projection number (int) and the particle id (int), output is a vector of constants

    frame_coords = coords[coords['frame']==frame] # all particle coords in frame
    z_pi = frame_coords.iloc[p_id,1] # z coord indexed by particle id in frame

    return np.array([0,(SOD/SDD)*z_pi])

### END DECOUPLED Z MATRIX and CONSTANTS ###

### DECOUPLED X MATRIX AND CONSTANTS ###

def mag_x(frame, p_id):
    frame_coords = coords[coords['frame']==frame] # all particle coords in frame
    x_pi = frame_coords.iloc[p_id,0] # x coord indexed by particle id in frame

    return np.array([1, -x_pi/SDD])


x_dynamics1 = np.array([[T*np.cos(theta), -T*np.sin(theta), 0.5*np.cos(theta)*T**2, -0.5*np.sin(theta)*T**2],
                      [T*np.sin(theta), T*np.cos(theta), 0.5*np.sin(theta)*T**2, 0.5*np.cos(theta)*T**2]]) # dynamics for x-focused matrix

x_dynamics2 = np.array([[np.cos(theta), -np.sin(theta), -1.0, 0],
                        [np.sin(theta), np.cos(theta), 0, -1.0]])


def extend_x(frame, p_id): # input is the projection number (int) and the particle id (int), output is a vector of constants

    frame_coords = coords[coords['frame']==frame] # all particle coords in frame
    x_pi = frame_coords.iloc[p_id,0] # x coord indexed by particle id in frame

    return np.array([0,0,(SOD/SDD)*x_pi])

### END DECOUPLED X MATRIX and CONSTANTS ###

# Non-linear least_squares with scaling weight
# def func(x, A, b):
#     residuals = 60*(A @ x- b)
#     target = np.linalg.norm(x[3] - y_global[p,0])  # Encourage z_0 to be close to known initial z position
#     return residuals + target

def func(x, A, b):
    residuals = A @ x - b   
    return residuals

# def func(x, A, b):
#     residuals = A @ x - b
#     yvals = x[2::2]
#     zvals = x[3::2]
#     zpvals = (SDD/(SOD+yvals))*zvals
#     particle_coords = coords[coords['particle'] == 0]
#     z_pi = particle_coords.iloc[:,2].to_numpy() # true projection coords
#     target = np.zeros_like(residuals)
#     target[4::2] = np.linalg.norm(zpvals - z_pi)
#     return residuals + target


# for y in y_vals:
#     for z in z_val:
#         x0[2] = y
#         x0[3] = z

#         x_solution = least_squares(func, x0, args=(M_z, b_z))
#         if best_result is None or x_solution.cost < best_result.cost:
#             best_result = x_solution
#             print("New best cost for particle %d: %f at y=%2.4f, z=%2.4f" % (p, best_result.cost, y, z))

# x0 = np.zeros(cols) # Initial guess
# best_result = None
# num_iters = 100
# for i in range(num_iters):
#     # Solve using least squares
#     res_lsq = least_squares(func, x0, args=(M_z, b_z))
#     if res_lsq.cost < (best_result.cost if best_result else float('inf')):
#         print("Found better result on iteration %d with cost %.20f" % (i, res_lsq.cost))
#         best_result = res_lsq
#     y0 = np.random.uniform(low=-3.0, high=3.0)
#     z0 = np.random.uniform(low=-2.0, high=2.0)
#     print(x0)
#     x0[2] = y0  # New random initial guess for next iteration
#     x0[3] = z0
# # Solve using least squares

### Algorithm ###
num_iters = 10

for n in range(num_iters):
    print (f"\nIteration #{n+1}\n")
    # Store results from Decoupled Z Calculation
    z_results = []

    # Now construct the corresponding decoupled Z matrix
    z_p1 = coords.iloc[:,1].to_numpy() # Grab all z coords in first frame (frame 0)
    for p in range(num_p):
        rows = 1
        cols = 4

        M_z = np.zeros((rows,cols)) # initialize/reset the matrix

        M_z[:,-2:] = mag_z(0, p) # Initial Block for the pth particle

        b_z = np.array([(SOD/SDD)*z_p1[p]]) # Initial vector for z calculations

        for i in range(projections-1):

            # add 2 rows and 2 rows each time
            new_rows = 2
            new_cols = 2
    
            rows += new_rows
            cols += new_cols
    
    
            # Enlarge matrix by new_rows down, new_cols right. fill these w/ 0
            M_z = np.pad(M_z, ((0,new_rows),(0,new_cols)), mode = 'constant', constant_values=0)

            # Insert blocks
            M_z[rows-new_rows:-1, :2] = z_dynamics
            M_z[rows-new_rows:-1, -3:] = np.array([1.0, 0, -1.0])
            M_z[-1:, -2:] = mag_z(i+1, p) # i+1 because we already built the first initial block 2
        
            # Extend our vector of constants
            b_z = np.concatenate((b_z, extend_z(i+1, p))) # i+1 because we already initilized b for 1 projection

    
        # y_vals = np.linspace(-3, 3, 20)
        # z_val = np.linspace(-3, 3, 20)

        print("b_z constants vector for particle " + str(p) + ":\n" + str(b_z))
        x0 = np.zeros(cols) # Initial guess
        if n != 0:
            x0[2] = result[p][5]
        z_result = least_squares(func, x0, args=(M_z, b_z), max_nfev=1000)

        z_results.append(z_result.x)

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
        'Known Position Y': y_global[:,0], #first frame y positions
        'Known Position Z': z_global[:,0], #first frame z positions
        'Estimated Position Y': df['y_0'],
        'Estimated Position Z': df['z_0'],
        'Error in Position Y': y_global[:,0] - df['y_0'],
        'Error in Position Z': z_global[:,0] - df['z_0'],
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
    print(comparison[['Known Acceleration a_z', 'Estimated Acceleration a_z', 'Error in Acceleration Z']])

    print("Y differences:" + str(y_global- z_results[0][2::2]))
    print("Z differences:" + str(z_global- z_results[0][3::2]))


    # Now construct the corresponding decoupled X matrix

    result = []
    x_p1 = coords.iloc[:,0].to_numpy() # Grab all x coords in first frame (frame 0)
    for p in range(num_p):
        rows = 1
        cols = 6

        M_x = np.zeros((rows,cols)) # initialize/reset the matrix

        M_x[:,-2:] = mag_x(0, p) # Initial Block for the pth particle

        b_x = np.array([(SOD/SDD)*x_p1[p]]) # Initial vector for x calculations

        for i in range(projections-1):
            # add 3 rows and 2 cols each time
            new_rows = 3
            new_cols = 2

            rows += new_rows
            cols += new_cols

            M_x = np.pad(M_x, ((0,new_rows),(0,new_cols)), mode = 'constant', constant_values=0)

            M_x[rows-new_rows:-1, :4] = x_dynamics1
            M_x[rows-new_rows:-1, -4:] = x_dynamics2
            M_x[-1:, -2:] = mag_x(i+1, p) # i+1 because we already built the first initial block 2
            b_x = np.concatenate((b_x, extend_x(i+1, p))) # i+1 because we already initilized b for 1 projection

        x_0_improved = np.zeros(cols) # Initial guess
        print("New y_0 value from Z matrix: " + str(z_results[p][2]))
        x_0_improved[5] = z_results[p][2] # Use estimated y position from z calculation as initial guess for x calculation
        print("x_0_improved for particle " + str(p) + ": " + str(x_0_improved))
        x_solution = least_squares(func, x_0_improved, args=(M_x, b_x), max_nfev=1000)
        result.append(x_solution.x)

    labels = ['u', 'v', 'a_x', 'a_y']
    for i in range(projections):
        labels.append('x_' + str(i))
        labels.append('y_' + str(i))

    df_x = pd.DataFrame(result, columns=labels)
    print("Final DataFrame of results: (each row corresponds to a particle)")
    print(df_x)

    comparison_x = pd.DataFrame({
        'Known Position X': x_global[:,0], #first frame x positions
        'Known Position Y': y_global[:,0], #first frame y positions
        'Estimated Position X': df_x['x_0'],
        'Estimated Position Y': df_x['y_0'],
        'Error in Position X': x_global[:,0] - df_x['x_0'],
        'Error in Position Y': y_global[:,0] - df_x['y_0'],
        'Known Velocity U': vel[:,0],
        'Known Velocity V': vel[:,1],
        'Estimated Velocity U': df_x['u'],
        'Estimated Velocity V': df_x['v'],
        'Error in Velocity U': vel[:,0] - df_x['u'],
        'Error in Velocity V': vel[:,1] - df_x['v'],
        'Known Acceleration a_x': acc[:,0],
        'Known Acceleration a_y': acc[:,1],
        'Estimated Acceleration a_x': df_x['a_x'],
        'Estimated Acceleration a_y': df_x['a_y'],
        'Error in Acceleration a_x': acc[:,0] - df_x['a_x'],
        'Error in Acceleration a_y': acc[:,1] - df_x['a_y']
    })

    print(comparison_x[['Known Position X', 'Estimated Position X', 'Error in Position X']])
    print(comparison_x[['Known Position Y', 'Estimated Position Y', 'Error in Position Y']])
    print(comparison_x[['Known Velocity U', 'Estimated Velocity U', 'Error in Velocity U']])
    print(comparison_x[['Known Velocity V', 'Estimated Velocity V', 'Error in Velocity V']])
    print(comparison_x[['Known Acceleration a_x', 'Estimated Acceleration a_x']])
    print(comparison_x[['Known Acceleration a_y', 'Estimated Acceleration a_y']])

    print("X differences:" + str(x_global- x_solution.x[4::2]))
    print("Y differences:" + str(y_global- x_solution.x[5::2]))


##### END ####


