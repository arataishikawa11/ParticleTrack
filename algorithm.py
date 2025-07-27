import numpy as np
import pandas as pd
import scipy as sp
from tabulate import tabulate
from track import coords # Import data from preprocessing


# Print out the DataFrame
print(coords)

# Consider Particle 1 first

# Initialize Values
SDD = 500 #mm source to detector
SOD = 250 #mm source to object
T = 0.1 # time step across frames (0.1 sec per one time step)
theta = np.deg2rad(T*0.5) # radians per time step

x_p1 = coords.iloc[:3,0] # Grab all x coords in first frame (frame 0)
z_p1 = coords.iloc[:3,2] # Grab all z coords in first frame

##### ONE PROJECTION BASE CASE (w/ ONE PARTICLE) #####

# Initialize vector 
b = np.zeros(2) # vector of known constant values (2 for one projection)
b[0], b[1] = SOD*x_p1[0], SOD*z_p1[0]
print(b)

# Initiliaze Matrix 
projections = 1 # Must have at least one projection
rows = 2
cols = 9
M = np.zeros((rows,cols))
# Set the values of the matrix for one projection
M[0,6], M[0,7], M[1,7], M[1,8] = SDD, -x_p1[0], -z_p1[0], SDD


print(tabulate(M))


# Solve
print(M.shape)
print(b.shape)
# linalg.lstsq returns vector, residuals, rank, s
result = np.linalg.lstsq(M, b.T, rcond=None)[0]
print("For one particle:")
print(result)

##### END #####



# Now lets incorporate more projections. Note that we add 5 rows and 3 columns each time
##### MULTIPLE PROJECTIONS #####
projections = 5
assert projections >= 1

num_p = 1 #add a forloop that loops over all particles later?

# We note by pattern-matching that there are 3 blocks that occur every time
# Yellow
block1 = np.array([[T*np.cos(theta), -T*np.sin(theta), 0, 0.5*np.cos(theta)*T**2, -0.5*np.sin(theta)*T**2, 0],
                  [T*np.cos(theta), T*np.sin(theta), 0, 0.5*np.cos(theta)*T**2, 0.5*np.sin(theta)*T**2, 0],
                  [0, 0, T, 0, 0, 0.5*T**2]])
print("block 1:\n" + tabulate(block1))

# Green
def block2(frame): # input is the projection number (int), output is the block for that projection
    x_pi = coords.iloc[frame-1:frame+2, 0].to_numpy()# Grab all x coords  Note: slicing rows frame-1: (frame+3)-1
    z_pi = coords.iloc[frame-1:frame+2, 2].to_numpy() # Grab all z coords
    block = np.array([[SDD, -x_pi[num_p-1], 0],
                       [0, -z_pi[num_p-1], SDD]])

    return block
print("block2: \n" + str(block2(2)))

# Blue
block3 = np.array([[np.cos(theta), -np.sin(theta), 0, -1, 0, 0],
                   [np.cos(theta), np.sin(theta), 0, 0, -1, 0],
                   [0, 0, 1, 0, 0, -1]])
print("block 3:\n" + tabulate(block1))


# Define a function for extending our vector of constants (known values)
def extend(frame):
    x_pi = coords.iloc[frame-1:frame+2, 0].to_numpy()# Grab all x coords  Note: slicing rows frame-1: (frame+3)-1
    z_pi = coords.iloc[frame-1:frame+2, 2].to_numpy() # Grab all z coords
    return np.array([0,0,0, SOD*x_pi[num_p-1], SOD*z_pi[num_p-1]])


print("vector of constants:\n" + str(extend(1)))

# redundant reminder
# starts with 2 rows and 9 cols
rows = 2
cols = 9

M[-2:, -3:] = block2(1) # Initial Block

# Initialize vector
# vector of known constant values (2 for one projection)
b = np.zeros(2)
b[0],b[1] = SOD*x_p1[num_p-1], SOD*z_p1[num_p-1] 
print(b)

for i in range(projections-1):
    # add 5 rows and 3 cols each time
    new_rows = 5
    new_cols = 3

    rows += new_rows
    cols += new_cols


    # Enlarge matrix by new_rows down, new_cols right. fill these w/ 0
    M = np.pad(M, ((0,new_rows),(0,new_cols)), mode = 'constant', constant_values=0)
    print("shape of M right now: " + str(M.shape))

    # Insert blocks
    M[rows-new_rows:-2, :6] = block1
    M[-2:, -3:] = block2(i+2) # i+2 because we already built the first initial block 2
    M[rows-new_rows:-2, 6+i*3:] = block3
    
    # Extend our vector of constants
    b = np.concatenate((b,extend(i+2))) # i+2 because we already initilized b for 1 projection


print(tabulate(M))
print(b)

# Solve
print(M.shape)
print(b.shape)
# linalg.lstsq returns vector, residuals, rank, s
result = np.linalg.lstsq(M, b.T, rcond=None)[0]
print("For one particle:")
print(result)


##### END #####