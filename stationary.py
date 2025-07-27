import numpy as np
import pandas as pd
from tabulate import tabulate

# Goal: Create a pandas dataframe that holds projection coordinates such that in the global frame, it satisfies stationary case


initial_pos = (1,1,1) #(x,y,z) in the global frame, this stays constant throughout

# 5 projections needed to solve unknowns
projections = 5

# Initialize Values
SDD = 500 #mm source to detector
SOD = 250 #mm source to object
T = 0.1 # time step across frames (0.1 sec per one time step)
theta = np.deg2rad(T*0.5) # radians per time step (this is actually delta theta)

# Initialize a numpy array that stores x coord in projection frame
x_p, y_p = np.zeros(5), np.zeros(5)
z_p = np.full(5,[initial_pos[2]]) # constant z across projections and object frames (rotation is about z axis)
angle = 0 # inital angle in radians

for i in range(projections):
    x_p[i] = (SDD*initial_pos[0])/(SOD + ((np.cos(angle)-1)/np.sin(angle))*initial_pos[0])
    z_p[i] = (SDD*initial_pos[0])/(SOD + ((np.cos(angle)-1)/np.sin(angle))*initial_pos[0])
    angle += theta
