import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import lsq_linear, least_squares
from tabulate import tabulate
from testcases import pos,vel,acc,flags # Import known initial positions and velocities for cross-referencing
from trackpy_test import coords_test # Import data from preprocessing
from initial_vals import * # Import initial values

coords = coords_test


# Algorithm 3: Here, we will be eliminating the x_0 variable by solving for it beforehand

### BLOCKS ###


# We note by pattern-matching that there are 3 blocks that occur every time
# Yellow
block1 = np.array([[T*np.cos(theta), -T*np.sin(theta), 0, 0.5*np.cos(theta)*T**2, -0.5*np.sin(theta)*T**2, 0],
                  [T*np.sin(theta), T*np.cos(theta), 0, 0.5*np.sin(theta)*T**2, 0.5*np.cos(theta)*T**2, 0],
                  [0, 0, T, 0, 0, 0.5*T**2]])
print("block 1:\n" + tabulate(block1))


# We no longer have the manification equation for x since we used it.
# Green
def block2(frame, p_id): # input is the projection number (int) and the particle id (int), output is the block for that projection

    frame_coords = coords[coords['frame']==frame] # all particle coords in frame
    z_pi = frame_coords.iloc[p_id,1] # z coord indexed by particle id in frame

    block = np.array([0, -z_pi/SDD, 1])

    return block
print("block2: \n" + str(block2(0, 0)))


# We now have to define block 3 as a function because we need x_pi
# Blue
def block3(frame, p_id): # input is the projection number (int) and the particle id (int), output is the block for that projection

    frame_coords = coords[coords['frame']==frame] # all particle coords in frame
    x_pi = frame_coords.iloc[p_id,0] # x coord indexed by particle id in frame

    block = np.array([[0, (x_pi/SDD)*np.cos(theta) - np.sin(theta), 0, -1.0, 0, 0],
                   [0, (x_pi/SDD)*np.sin(theta) + np.cos(theta), 0, 0, -1.0, 0],
                   [0, 0, 1.0, 0, 0, -1.0]])

    return block
print("block 3:\n" + tabulate(block3(0, 0)))


#### BLOCKS FOR C MATRIX (ROTATION CONTRIBUTION ONLY) ###
#
## Yellow
#cblock1 = np.array([[T*np.cos(theta), -T*np.sin(theta), 0, 0.5*np.cos(theta)*T**2, -0.5*np.sin(theta)*T**2, 0],
#                  [T*np.sin(theta), T*np.cos(theta), 0, 0.5*np.sin(theta)*T**2, 0.5*np.cos(theta)*T**2, 0]])
#
#
## Blue
#cblock3 = np.array([[np.cos(theta), -np.sin(theta), 0, -1.0, 0, 0],
#                   [np.sin(theta), np.cos(theta), 0, 0, -1.0, 0]])
#

### END ###



# Define a function for extending our vector of constants (known values)
def extend(frame, p_id): # input is the projection number (int) and the particle id (int), output is a vector of constants

    frame_coords = coords[coords['frame']==frame] # all particle coords in frame
    x_pi = frame_coords.iloc[p_id,0] # x coord indexed by particle id in frame
    z_pi = frame_coords.iloc[p_id,1] # z coord indexed by particle id in frame


    return np.array([-(SOD/SDD)*x_pi*np.cos(theta), -(SOD/SDD)*x_pi*np.sin(theta), 0, SOD/SDD*z_pi])
print("vector of constants:\n" + str(extend(1, 0))) # First particle has id p_id = 0


### Algorithm ###

# We will store our results in a list. Each entry will be a numpy array of unknowns with index corresponding to the particle id. (result[0] corresponds to particle_id = 0)
result = []


x_p1 = coords.iloc[:,0].to_numpy() # Grab all x coords in first frame (frame 0)
z_p1 = coords.iloc[:,1].to_numpy() # Grab all z coords in first frame

for p in range(num_p):

    # starts with 1 row and 9 cols every time
    rows = 1
    cols = 9

    M = np.zeros((rows,cols)) # reset the matrix

    M[0:, -3:] = block2(0, p) # Initial Block (0th frame) for the pth particle


    #C = np.zeros((rows,cols)) # reset the rotation contribution matrix

    
    # Initialize vector
    # vector of known constant values (2 for one projection)
    b = np.zeros(1)
    b[0] = (SOD/SDD)*z_p1[p] 

    # Initialize d vector (for rotation contribution only)
    #d = np.zeros(2)

    
    for i in range(projections-1):
        # add 5 rows and 3 cols each time
        new_rows = 4
        new_cols = 3
    
        rows += new_rows
        cols += new_cols
    
    
        # Enlarge matrix by new_rows down, new_cols right. fill these w/ 0
        M = np.pad(M, ((0,new_rows),(0,new_cols)), mode = 'constant', constant_values=0)

        # Insert blocks
        M[rows-new_rows:-1, :6] = block1
        M[-1:, -3:] = block2(i+1, p) # i+1 because we already built the first initial block 2
        M[rows-new_rows:-1, 6+i*3:] = block3(i, p)
        
        # Extend our vector of constants
        b = np.concatenate((b,extend(i+1, p))) # i+1 because we already initilized b for 1 projection

        # Rotation contribution matrix
        #C = np.pad(C, ((0,new_rows),(0,new_cols)), mode = 'constant', constant_values=0)

        # Insert blocks (No green block contribution)
        #C[rows-new_rows:-3, :6] = cblock1
        #C[rows-new_rows:-3, 6+i*3:] = cblock3
        
        # Extend our vector of constants
        #d = np.concatenate((d, np.zeros(5))) # i+1 because we already initilized b for 1 projection
    
    


    # Implementing one validation check
    validation = np.zeros(cols)
    validation[7] = 1.0
    validation[1] = (projections-1)*T
    validation[4] = 0.5*(projections-1)*T**2
    validation[-2] = -1.0

    #b = np.append(b, 0) # Append a 0 to b
    #M = np.vstack((M, validation)) # Append the validation row to M
    #M=M*10e2
    #b=b*10e2
    #print("final b vector: \n" + str(b))

    # Solve
    # linalg.lstsq returns vector, residuals, rank, s values
    #x, residuals, rank, svals = np.linalg.lstsq(M, b, rcond=None)
    #result.append(x)
    # print("For particle_id: " + str(p))
    # print(result[p])
    
    # Implement Kalman Filter



    # print(np.round(result[p]))


    # Calculate condition number
    #cond_num = np.linalg.cond(M)
    #print("Condition number (before): " + str(cond_num))


    # Tikhonov regularization
    # We will use the identity matrix as the regularization matrix

    # Regularization parameter
    #lam = 1e-2

    #n_params = M.shape[1] # number of parameters (columns in M)
    #I = np.eye(n_params) # identity matrix of size n_params x n_params

    #M_aug = np.vstack((M, lam * I)) # Augment M with regularization
    #b_aug = np.concatenate((b, np.zeros(n_params))) # Augment b with zeros

    ## Solve the augmented system
    #x_reg, residuals_reg, rank_reg, svals_reg = np.linalg.lstsq(M_aug, b_aug, rcond=None)

    #result[p] = x_reg  # Update result with regularized solution

    #print(tabulate(M_aug))
    #print("Regularized solution for particle_id: " + str(p))
    #print(result[p])
    #print(np.round(result[p]))
    #print("Regularized condition number: " + str(np.linalg.cond(M_aug)))

    #print("Shape of M, non-regularized" + str(np.shape(M)))
    #print("Shape of M, regularized" + str(np.shape(M_aug)))
    #print("Shape of b, non-regularized" + str(np.shape(b)))
    #print("Shape of b, regularized" + str(np.shape(b_aug)))
    #print(tabulate(M))

    ## Implement bounds
    #upper_bounds = 3 * np.ones(n_params)  # Upper bounds for each parameter
    ##upper_bounds = np.inf * np.ones(n_params) # Upper bounds for each parameter
    #upper_bounds[:6] = np.inf  # No bounds for the first 6 parameters
    #lower_bounds = -3 * np.ones(n_params) # Lower bounds for each parameter
    ##lower_bounds = -np.inf * np.ones(n_params)  # Upper bounds for each parameter
    #lower_bounds[:6] = -np.inf  # No bounds for the first 6 parameters
    #print(type(lower_bounds))
    #print(type(upper_bounds))
    #print("Lower bounds: " + str(lower_bounds))
    #print("Upper bounds: " + str(upper_bounds))

    #res = lsq_linear(M, b, verbose = 2)  # Solve with bounds
    #print("\n Bounded solution for particle_id " + str(p) + " with regularization:")
    #result.append(res.x)
    ## Calculate condition number
    #cond_num = np.linalg.cond(M)
    #print("Condition number (after): " + str(cond_num))

    #print(tabulate(C))
    print(tabulate(M))

    # Non-linear least_squares
    def func(x, A, b, S=np.eye(rows)):
        residuals = np.linalg.norm((A @ x - b))
        return residuals

    x0 = np.zeros(cols) # Initial guess
    #y0 = np.zeros(cols) # Initial guess for rotation contribution
    
    # Create scaling matrix
   # diag = np.ones(rows)
   # for i in range(projections+1):
   #     diag[-1 -i*3]=100.0 #z
   #     diag[-2 -i*3]=1.0 #y
   #     diag[-3 -i*3]=1.0 #x
   # S = np.diag(diag)



    res_lsq = least_squares(func, x0, args=(M, b))
    print("Non-linear least squares solution:")
    #print(res_lsq.x)


    result.append(res_lsq.x)


    # SVD
   # U, s, Vt = np.linalg.svd(M, full_matrices=False)
   # print("s before inverse:" + str(s))
   # s = 1/s
   # Sigma = np.diag(s)
   # print(U)
   # print("Shape U: " + str(np.shape(U)))
   # print(Sigma)
   # print("Shape Sigma: " + str(np.shape(Sigma)))
   # print(s)
   # print("Shape s: " + str(np.shape(s)))
   # print(Vt)
   # print("Shape Vt: " + str(np.shape(Vt)))

   # A_plus = Vt.T @ Sigma @ U.T

   # x_svd = A_plus @ b
   # print("SVD solution:")
   # print(x_svd)
   # result.append(x_svd)



labels = ['u', 'v', 'w', 'a_x', 'a_y', 'a_z']
for i in range(projections):
    labels.append('x_' + str(i))
    labels.append('y_' + str(i))
    labels.append('z_' + str(i))

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


