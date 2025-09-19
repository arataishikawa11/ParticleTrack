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






# Define a function for extending our vector of constants (known values)
def extend(frame, p_id): # input is the projection number (int) and the particle id (int), output is a vector of constants

    frame_coords = coords[coords['frame']==frame] # all particle coords in frame
    x_pi = frame_coords.iloc[p_id,0] # x coord indexed by particle id in frame
    z_pi = frame_coords.iloc[p_id,1] # z coord indexed by particle id in frame


    return np.array([0,0,0, SOD/SDD*x_pi, SOD/SDD*z_pi])
print("vector of constants:\n" + str(extend(1, 0))) # First particle has id p_id = 0


### Algorithm ###

# We will store our results in a list. Each entry will be a numpy array of unknowns with index corresponding to the particle id. (result[0] corresponds to particle_id = 0)
result = []


x_p1 = coords.iloc[:,0].to_numpy() # Grab all x coords in first frame (frame 0)
z_p1 = coords.iloc[:,1].to_numpy() # Grab all z coords in first frame
for p in range(num_p):

    # starts with 2 rows and 9 cols every time
    rows = 2
    cols = 9

    M = np.zeros((rows,cols)) # reset the matrix

     # Initial Block for the pth particle
    M[:,0:3] = block2(0,p)
    
    # Initialize vector
    # vector of known constant values (2 for one projection)
    b = np.zeros(2)
    b[0], b[1] = (SOD/SDD)*x_p1[p], (SOD/SDD)*z_p1[p] 

    
    for i in range(projections-1):
        # add 5 rows and 3 cols each time
        new_rows = 5
        new_cols = 3
    
        rows += new_rows
        cols += new_cols
    
    
        # Enlarge matrix by new_rows down, new_cols right. fill these w/ 0
        M = np.pad(M, ((0,new_rows),(0,new_cols)), mode = 'constant', constant_values=0)
    
        # Insert blocks
        M[-2:, -9:-6] = block2(i+1, p) # i+1 because we already built the first initial block 2
        M[-5:-2, -12:-6] = block3
        
        # Extend our vector of constants
        b = np.concatenate((b,extend(i+1, p))) # i+1 because we already initilized b for 1 projection
    
    for i in range(projections-1):
        #M[2*i+2:2*i+5, -6:] = block1
        M[5*i+2:5*i+5, -6:] = block1


#    # Implemening one validation check
#    validation = np.zeros(cols)
#    validation[7] = 1.0
#    validation[1] = (projections-1)*T
#    validation[4] = 0.5*(projections-1)*T**2
#    validation[-2] = -1.0
#
#    #b = np.append(b, 0) # Append a 0 to b
#    #M = np.vstack((M, validation)) # Append the validation row to M
#    M=M*10e2
#    b=b*10e2
#    print(tabulate(M))
#    #print("final b vector: \n" + str(b))
#
#    # Solve
#    # linalg.lstsq returns vector, residuals, rank, s values
#    x, residuals, rank, svals = np.linalg.lstsq(M, b, rcond=None)
#    result.append(x)
#    # print("For particle_id: " + str(p))
#    # print(result[p])
#    
#
#
#    # print(np.round(result[p]))
#
#
#    # Calculate condition number
    cond_num = np.linalg.cond(M)
    print("Condition number (before): " + str(cond_num))
#
#
#    # Tikhonov regularization
#    # We will use the identity matrix as the regularization matrix
#
#    # Regularization parameter
#    lam = 1e-2
#
    n_params = M.shape[1] # number of parameters (columns in M)
#    I = np.eye(n_params) # identity matrix of size n_params x n_params
#
#    M_aug = np.vstack((M, lam * I)) # Augment M with regularization
#    b_aug = np.concatenate((b, np.zeros(n_params))) # Augment b with zeros
#
#    # Solve the augmented system
#    x_reg, residuals_reg, rank_reg, svals_reg = np.linalg.lstsq(M_aug, b_aug, rcond=None)
#
#    result[p] = x_reg  # Update result with regularized solution
#
    #print(tabulate(M_aug))
    #print("Regularized solution for particle_id: " + str(p))
    #print(result[p])
    #print(np.round(result[p]))
    #print("Regularized condition number: " + str(np.linalg.cond(M_aug)))

    #print("Shape of M, non-regularized" + str(np.shape(M)))
    #print("Shape of M, regularized" + str(np.shape(M_aug)))
    #print("Shape of b, non-regularized" + str(np.shape(b)))
    #print("Shape of b, regularized" + str(np.shape(b_aug)))

    # Implement bounds
    upper_bounds = 3 * np.ones(n_params)  # Upper bounds for each parameter
    #upper_bounds = np.inf * np.ones(n_params) # Upper bounds for each parameter
    upper_bounds[-6:] = np.inf  # No bounds for the first 6 parameters
    lower_bounds = -3 * np.ones(n_params) # Lower bounds for each parameter
    #lower_bounds = -np.inf * np.ones(n_params)  # Upper bounds for each parameter
    lower_bounds[-6:] = -np.inf  # No bounds for the first 6 parameters
    print(type(lower_bounds))
    print(type(upper_bounds))
    print("Lower bounds: " + str(lower_bounds))
    print("Upper bounds: " + str(upper_bounds))

    res = lsq_linear(M, b)  # Solve with bounds
    print("\n Bounded solution for particle_id " + str(p) + " with regularization:")
    print(res.x)
    result.append(res.x)

    # Calculate condition number
    cond_num = np.linalg.cond(M)
    print("Condition number (after): " + str(cond_num))


print(tabulate(M))
print(b)
labels = ['u', 'v', 'w', 'a_x', 'a_y', 'a_z']
list = []
for i in range(projections):
    list.append('x_' + str(i))
    list.append('y_' + str(i))
    list.append('z_' + str(i))
labels = list + labels

df = pd.DataFrame(result, columns=labels)
print("Final DataFrame of results: (each row corresponds to a particle)")
print(df)

# Cross reference results with known initial positions and velocities
# Note that we only have access to initial positions and velocities, not accelerations
print("\n Cross-referencing results with known initial positions and velocities:")
comparison = pd.DataFrame({
    'Known Position X': pos[:,0],
    'Known Position Y': pos[:,1],
    'Known Position Z': pos[:,2],
    'Estimated Position X': df['x_' + str(projections-1)],
    'Estimated Position Y': df['y_' + str(projections-1)],
    'Estimated Position Z': df['z_' + str(projections-1)],
    'Error in Position X': pos[:,0] - df['x_'+str(projections-1)],
    'Error in Position Y': pos[:,1] - df['y_'+str(projections-1)],
    'Error in Position Z': pos[:,2] - df['z_'+str(projections-1)],
    'Known Velocity U': vel[:,0],
    'Known Velocity V': vel[:,1],
    'Known Velocity W': vel[:,2],
    'Estimated Velocity U': df['u'],
    'Estimated Velocity V': df['v'],
    'Estimated Velocity W': df['w'],
    'Error in Velocity U': vel[:,0] - df['u'],
    'Error in Velocity V': vel[:,1] - df['v'],
    'Error in Velocity W': vel[:,2] - df['w'],
    'Known Acceleration a_x': acc[:,0],
    'Known Acceleration a_y': acc[:,1],
    'Known Acceleration a_z': acc[:,2],
    'Estimated Acceleration a_x': df['a_x'],
    'Estimated Acceleration a_y': df['a_y'],
    'Estimated Acceleration a_z': df['a_z'],
    'Error in Acceleration X': acc[:,0] - df['a_x'],
    'Error in Acceleration Y': acc[:,1] - df['a_y'],
    'Error in Acceleration Z': acc[:,2] - df['a_z']
})

print(comparison[['Known Position X', 'Estimated Position X', 'Error in Position X']])
print(comparison[['Known Position Y', 'Estimated Position Y', 'Error in Position Y']])
print(comparison[['Known Position Z', 'Estimated Position Z', 'Error in Position Z']])
print(comparison[['Known Velocity U', 'Estimated Velocity U', 'Error in Velocity U']])
print(comparison[['Known Velocity V', 'Estimated Velocity V', 'Error in Velocity V']])
print(comparison[['Known Velocity W', 'Estimated Velocity W', 'Error in Velocity W']])
print(comparison[['Known Acceleration a_x', 'Estimated Acceleration a_x']])
print(comparison[['Known Acceleration a_y', 'Estimated Acceleration a_y']])
print(comparison[['Known Acceleration a_z', 'Estimated Acceleration a_z']])

# Note that if results do not respect bounds, assume poor results

print(df["y_" + str(projections-1)])

print(flags)
##### END ####

