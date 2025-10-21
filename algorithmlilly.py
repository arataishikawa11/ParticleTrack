import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import lsq_linear, least_squares
from tabulate import tabulate
from testcases import pos,vel,acc,flags # Import known initial positions and velocities for cross-referencing
from trackpy_test import coords_test # Import data from preprocessing
from initial_vals import * # Import initial values


coords = coords_test


# THIS ONE DOES IMPLEMENT THE NEW LEAST SQUARES WITH BOUNDS AND TIKHONOV REGULARIZATION
# AND LILLY ADDED THE -Y FLIP AND THE TRIANGULATION AS A WHOLE


### LILLYS ADDED BLOCKS :) ###
#  rotation about z
def Rz(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)
#  build from one measurement
def camera_ray_world(xp, zp, k, theta, SOD, SDD):
    Rk = Rz(k*theta)
    c_world = Rk @ np.array([0.0, +SOD, 0.0])              # source position in world
    d_cam   = np.array([xp, -SDD, zp], dtype=float)        # ray through detector pixel
    d_cam  /= np.linalg.norm(d_cam) + 1e-15
    d_world = Rk @ d_cam
    d_world /= np.linalg.norm(d_world) + 1e-15
    return c_world, d_world
# closest midpoint between two rays
def closest_point_between_rays(c1, d1, c2, d2):
    r = c1 - c2
    a = 1.0
    b = float(d1 @ d2)
    c = 1.0
    d = float(d1 @ r)
    e = float(d2 @ r)
    denom = a*c - b*b
    if abs(denom) < 1e-12:
        t = -d
        s = 0.0
    else:
        t = (b*e - c*d) / denom
        s = (a*e - b*d) / denom
    p1 = c1 + t*d1
    p2 = c2 + s*d2
    return 0.5*(p1 + p2)
# y-hints bytriangulation across frames
def compute_y_hints_for_particle(df_coords, p_id, projections, theta, SOD, SDD):
    if 'particle' in df_coords.columns:
        sub = df_coords[df_coords['particle'] == p_id].sort_values('frame')
    else:
        sub = (df_coords.sort_values('frame')
                          .groupby('frame', as_index=False)
                          .nth(p_id)
                          .sort_values('frame'))
    xp = sub['x'].to_numpy()
    zp = sub['z'].to_numpy()
    P  = len(xp)


    rays = [camera_ray_world(xp[k], zp[k], k, theta, SOD, SDD) for k in range(P)]
    mids = []
    for k in range(P-1):
        c1,d1 = rays[k]
        c2,d2 = rays[k+1]
        mids.append(closest_point_between_rays(c1,d1,c2,d2))


    y_hint = np.zeros(P)
    if P == 1:
        y_hint[0] = mids[0][1] if mids else 0.0
    elif P == 2:
        y_hint[:] = mids[0][1]
    else:
        y_hint[0] = mids[0][1]
        for k in range(1, P-1):
            y_hint[k] = 0.5*(mids[k-1][1] + mids[k][1])
        y_hint[P-1] = mids[-1][1]
    return y_hint


### BLOCKS ###


# We note by pattern-matching that there are 3 blocks that occur every time
# Yellow
block1 = np.array([[T*np.cos(theta), -T*np.sin(theta), 0, 0.5*np.cos(theta)*T**2, -0.5*np.sin(theta)*T**2, 0],
                  [T*np.sin(theta), T*np.cos(theta), 0, 0.5*np.sin(theta)*T**2, 0.5*np.cos(theta)*T**2, 0],
                  [0, 0, T, 0, 0, 0.5*T**2]])
# print("block 1:\n" + tabulate(block1))  # LILLY COMMENTED OUT
# Green
def block2(frame, p_id): # input is the projection number (int) and the particle id (int), output is the block for that projection


    frame_coords = coords[coords['frame']==frame] # all particle coords in frame
    x_pi = frame_coords.iloc[p_id,0] # x coord indexed by particle id in frame
    z_pi = frame_coords.iloc[p_id,1] # z coord indexed by particle id in frame


    block = np.array([[1, -x_pi/SDD, 0],
                       [0, -z_pi/SDD, 1]])


    return block
# print("block2: \n" + str(block2(0, 0)))  # LILLY COMMENTED OUT


# Blue
block3 = np.array([[np.cos(theta), -np.sin(theta), 0, -1.0, 0, 0],
                   [np.sin(theta), np.cos(theta), 0, 0, -1.0, 0],
                   [0, 0, 1.0, 0, 0, -1.0]])
# print("block 3:\n" + tabulate(block3)) # LILLY COMMENTED OUT




# Define a function for extending our vector of constants (known values)
def extend(frame, p_id): # input is the projection number (int) and the particle id (int), output is a vector of constants


    frame_coords = coords[coords['frame']==frame] # all particle coords in frame
    x_pi = frame_coords.iloc[p_id,0] # x coord indexed by particle id in frame
    z_pi = frame_coords.iloc[p_id,1] # z coord indexed by particle id in frame




    return np.array([0,0,0, SOD/SDD*x_pi, SOD/SDD*z_pi])
# print("vector of constants:\n" + str(extend(1, 0))) # First particle has id p_id = 0 # LILLY COMMENTED OUT




### Algorithm ###


# We will store our results in a list. Each entry will be a numpy array of unknowns with index corresponding to the particle id. (result[0] corresponds to particle_id = 0)
result = []


x_p1 = coords.iloc[:,0].to_numpy() # Grab all x coords in first frame (frame 0)
z_p1 = coords.iloc[:,1].to_numpy() # Grab all z coords in first frame


w_yhint = 10.0   # weight for y-hint prior # LILLYS ADDITION


for p in range(num_p):


    # starts with 2 rows and 9 cols every time
    rows = 2
    cols = 9


    M = np.zeros((rows,cols)) # reset the matrix


     # Initial Block for the pth particle
    M[:, -3:] = block2(0,p)
   
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
        M[rows-new_rows:-2, :6] = block1
        M[-2:, -3:] = block2(i+1, p) # i+1 because we already built the first initial block 2
        M[rows-new_rows:-2, 6+i*3:] = block3
       
        # Extend our vector of constants
        b = np.concatenate((b,extend(i+1, p))) # i+1 because we already initilized b for 1 projection
   

    # add y-hint prior rows for each particle
    y_hint = compute_y_hints_for_particle(coords, p, projections, theta, SOD, SDD)

    # Reshape y_hint vector into same shape as solution vector x
    y_triangulated = np.zeros(6)
    for y in y_hint:
        y_triangulated = np.concatenate((y_triangulated, [0, y, 0]))
    
    # Create A matrix
    diag = np.zeros(cols)
    for i in range(projections+1):
        diag[-3 -i*3]=0.0 #x
        diag[-2 -i*3]=1.0 #y
        diag[-1 -i*3]=0.0 #z
    A = np.diag(diag)
    A = np.pad(A, ((0, rows - np.shape(A)[0]),(0,0)), mode = 'constant', constant_values = 0) # add rows of zero to make A the same shape as M
    print(np.shape(A))
    print(tabulate(A))

    n_params = M.shape[1] # number of parameters (columns in M)

    w = 10.0 #weight

    # Solve for each particle p
    
    # Non-linear least_squares
    def func(x, M, A, y, b, w=0.0):
        residuals = (M @ x - b) + w * A @ (x - y)
        return residuals

    x0 = np.zeros(cols) # Initial guess
    print(np.shape(x0))

    res_lsq = least_squares(func, x0, args=(M, A, y_triangulated, b, w), 
                            method = 'lm', max_nfev=200)
    print("Non-linear least squares solution:")
    #print(res_lsq.x)


    #result.append(res_lsq.x)

    M_eff = M + w * A
    b_eff = b + w * A @ y_triangulated
    # Compute SVD of M_eff
    U, S, Vt = np.linalg.svd(M_eff, full_matrices=False)

    # Compute the pseudoinverse of M_eff using SVD
    S_inv = np.diag(1 / S)
    M_pinv = Vt.T @ S_inv @ U.T

    # Solve for x
    x_svd = M_pinv @ b_eff

    print("Solution using SVD decomposition:")
    #result.append(x_svd)

    lambda_reg = 1e-3
    S_reg = S / (S**2 + lambda_reg)
    x_svd_reg = (Vt.T * S_reg) @ (U.T @ b_eff)
    result.append(x_svd_reg)


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


# SORRY THIS IS JANK AND IM NOT SURE WHY ITS NEEDED YET - LILLY
for i in range(projections):
    df[f'y_{i}'] *= -1


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
print(comparison[['Known Position Y', 'Estimated Position Y', 'Error in Position Y']]) # LILLY COMMENTED OUT EVERYTHING :)
print(comparison[['Known Position Z', 'Estimated Position Z', 'Error in Position Z']])
print(comparison[['Known Velocity U', 'Estimated Velocity U', 'Error in Velocity U']])
print(comparison[['Known Velocity V', 'Estimated Velocity V', 'Error in Velocity V']])
print(comparison[['Known Velocity W', 'Estimated Velocity W', 'Error in Velocity W']])
print(comparison[['Known Acceleration a_x', 'Estimated Acceleration a_x']])
print(comparison[['Known Acceleration a_y', 'Estimated Acceleration a_y']])
print(comparison[['Known Acceleration a_z', 'Estimated Acceleration a_z']])


# Note that if results do not respect bounds, assume poor results


print(df["y_" + str(projections-1)])




# print(flags)
##### END ####




