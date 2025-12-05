import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import lsq_linear, least_squares
from tabulate import tabulate
#from testcases import pos,vel,acc,flags # Import known initial positions and velocities for cross-referencing
#from testcases2 import x_global, y_global, z_global, vel, acc # Import known global positions for cross-referencing
from testcases3 import x_global, y_global, z_global, vel, acc # Import known global positions for cross-referencing
from trackpy_test import tp, coords_test # Import data from preprocessing
from initial_vals import * # Import initial values

coords = coords_test
tp.quiet()
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


### END ###



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

    M[0:, -3:] = block2(0, p) # Initial Block for the pth particle

    
    # Initialize vector
    # vector of known constant values (2 for one projection)
    b = np.zeros(2)
    b[0], b[1] = (SOD/SDD)*x_p1[p], (SOD/SDD)*z_p1[p] 

    # Initialize d vector (for rotation contribution only)
    d = np.zeros(2)

    
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


    # Non-linear least_squares
    def func(x, A, b):
        residuals = (A @ x - b)
        xvals = x[6::3]
        yvals = x[7::3]
        zvals = x[8::3]
        xpvals = (SDD/(SOD+yvals))*xvals
        zpvals = (SDD/(SOD+yvals))*zvals
        particle_coords = coords[coords['particle'] == 0]
        x_pi = particle_coords.iloc[:,0].to_numpy() # true projection coords
        z_pi = particle_coords.iloc[:,2].to_numpy() # true projection coords
        target=np.zeros_like(residuals)
        target[6::3] = np.linalg.norm(xpvals - x_pi) #+ np.linalg.norm(zpvals - z_pi) # target is the sum of the distances between the projected coordinates and the actual coordinates
        target[8::3] = np.linalg.norm(zpvals - z_pi) #+ np.linalg.norm(zpvals - z_pi) # target is the sum of the distances between the projected coordinates and the actual coordinates
        
        return residuals + target

    # def func(x, A, b):
    #     residuals = (A @ x - b)
    #     return residuals

    y_vals = np.linspace(-3, 3, 50)
    z_val = np.linspace(-3, 3, 50)

    x0 = np.zeros(cols) # Initial guess
    best_result = None

    # for y in y_vals:
    #     for z in z_val:
    #         x0[7] = y
    #         x0[8] = z


    #         x_solution = least_squares(func, x0, args=(M, b))
    #         if best_result is None or x_solution.cost < best_result.cost:
    #             if best_result != None and np.abs(x_solution.cost - best_result.cost) < 0.1:
    #                 break
    #             best_result = x_solution
    #             print("New best cost for particle %d: %f at y=%2.4f, z=%2.4f" % (p, best_result.cost, y, z))

    # num_iters = 10
    # for i in range(num_iters):
    #     # Solve using least squares
    #     res_lsq = least_squares(func, x0, args=(M, b), max_nfev=2000)
    #     if res_lsq.cost < (best_result.cost if best_result else float('inf')):
    #         print("Found better result on iteration %d with cost %f" % (i, res_lsq.cost))
    #         best_result = res_lsq
    #     x0 = np.random.uniform(low=-3.0, high=3.0, size=cols)  # Random restart
    # # Solve using least squares

    
    

    best_result = least_squares(func, x0, args=(M, b))

    result.append(best_result.x)




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
    'Known Position X': x_global[:,0], # initial global x positions
    'Known Position Y': y_global[:,0], # initial global y positions
    'Known Position Z': z_global[:,0], # initial global z positions 
    'Estimated Position X': df['x_0'],
    'Estimated Position Y': df['y_0'],
    'Estimated Position Z': df['z_0'],
    'Error in Position X': x_global[:,0] - df['x_0'],
    'Error in Position Y': y_global[:,0] - df['y_0'],
    'Error in Position Z': z_global[:,0] - df['z_0'],
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


print("diff y" + str(y_global[:,0] - result[0][7::3]))
print("diff z" + str(z_global[:,0] - result[0][8::3]))

##### END ####


