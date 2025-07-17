import numpy as np
import pandas as pd
import scipy as sp

# Initialize input vector 
x = np.arange(stop = 9, dtype = int)
print(x)

# Initiliaze Matrix
M = np.identity(9)
print(M)

# Invert matrix
M_inv = np.linalg.inv(M)
print(M_inv)

# Solve
result = x.dot(M_inv)
print(result)