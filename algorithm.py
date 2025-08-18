import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import lsq_linear
from tabulate import tabulate
#from track import coords # Import data from preprocessing
from trackpy_test import coords_test # Import data from preprocessing
#from testcases import coords_test


# Print out the DataFrame
#print(coords)

# Stationary Case
# Caution with the naming of columns
#print(coords_test)
#print(coords_test2)
coords = coords_test

# Initialize Values
SDD = 500 #mm source to detector
SOD = 250 #mm source to object
T = 0.01 # time step across frames (0.1 sec per one time step)
theta = np.deg2rad(T*0.5) # (delta) radians per time step

# Number of frames
projections = 5
assert projections >= 1

# Number of particles
num_p = 1
assert num_p >= 1



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


# Algorithm

# We will store our results in a list. Each entry will be a numpy array of unknowns with index corresponding to the particle id. (result[0] corresponds to particle_id = 0)
result = []


x_p1 = coords.iloc[:,0].to_numpy() # Grab all x coords in first frame (frame 0)
z_p1 = coords.iloc[:,1].to_numpy() # Grab all z coords in first frame
for p in range(num_p):

    # starts with 2 rows and 9 cols every time
    rows = 2
    cols = 9

    M = np.zeros((rows,cols)) # reset the matrix

    M[-2:, -3:] = block2(0, p) # Initial Block for the pth particle
    
    # Initialize vector
    # vector of known constant values (2 for one projection)
    b = np.zeros(2)
    b[0], b[1] = (SOD/SDD)*x_p1[p], (SOD/SDD)*z_p1[p] 
    print("b initial: \n" + str(b))

    
    for i in range(projections-1):
        # add 5 rows and 3 cols each time
        new_rows = 5
        new_cols = 3
    
        rows += new_rows
        cols += new_cols
    
    
        # Enlarge matrix by new_rows down, new_cols right. fill these w/ 0
        M = np.pad(M, ((0,new_rows),(0,new_cols)), mode = 'constant', constant_values=0)
    
        # Insert blocks
        M[rows-new_rows:-2, :6] = block1
        M[-2:, -3:] = block2(i+1, p) # i+1 because we already built the first initial block 2
        M[rows-new_rows:-2, 6+i*3:] = block3
        
        # Extend our vector of constants
        b = np.concatenate((b,extend(i+1, p))) # i+1 because we already initilized b for 1 projection
    
    
    print(tabulate(M))
    print("final b vector: \n" + str(b))
    
    # Solve
    # linalg.lstsq returns vector, residuals, rank, s values
    x, residuals, rank, svals = np.linalg.lstsq(M, b, rcond=None)
    result.append(x)
    print("For particle_id: " + str(p))
    print(result[p])
    


    print(np.round(result[p]))


    # Calculate condition number
    cond_num = np.linalg.cond(M)
    print("Condition number: " + str(cond_num))

    # Tikhonov regularization
    # We will use the identity matrix as the regularization matrix

    # Regularization parameter
    lam = 1e-1

    n_params = M.shape[1] # number of parameters (columns in M)
    I = np.eye(n_params) # identity matrix of size n_params x n_params

    M_aug = np.vstack((M, lam * I)) # Augment M with regularization
    b_aug = np.concatenate((b, np.zeros(n_params))) # Augment b with zeros

    # Solve the augmented system
    x_reg, residuals_reg, rank_reg, svals_reg = np.linalg.lstsq(M_aug, b_aug, rcond=None)

    result[p] = x_reg  # Update result with regularized solution

    print(tabulate(M_aug))
    print("Regularized solution for particle_id: " + str(p))
    print(result[p])
    print(np.round(result[p]))
    print("Regularized condition number: " + str(np.linalg.cond(M_aug)))




    # Implement bounds
    upper_bounds = 3 * np.ones(n_params)  # Upper bounds for each parameter
    upper_bounds[:6] = np.inf  # No bounds for the first 6 parameters
    lower_bounds = -3 * np.ones(n_params)  # Lower bounds for each parameter
    lower_bounds[:6] = -np.inf  # No bounds for the first 6 parameters

    res = lsq_linear(M, b, bounds=(lower_bounds, upper_bounds))  # Solve with bounds
    print("Bounded solution for particle_id: " + str(p))
    print(res.x)
    print(np.round(res.x))
##### END ####