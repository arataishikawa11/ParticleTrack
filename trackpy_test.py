# import libraries
import matplotlib as mpl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trackpy as tp # trackpy is a particle tracking toolkit https://soft-matter.github.io/trackpy/v0.6.4/

from testcases import coords_test

print(coords_test)
coords = coords_test

# Link particle trajectories from coords dataframe with prediction enabled
# Beware of column names 
pred = tp.predict.NearestVelocityPredict()
traj_pred = pred.link_df(coords, search_range = 50, pos_columns = ['x','z'], memory = 5)


# # Trace trajectories (2D)
fig, ax = plt.subplots()

# We have two trajectories to choose from, one without prediction (traj) or with dynamic prediction (traj_pred)
tp.plot_traj(traj_pred, ax=ax, pos_columns=['x','z'], colorby='particle')
ax.scatter(coords['x'],coords['z'])

plt.xlim([np.max(coords['x'])-50,np.max(coords['x']+50)])
plt.ylim([np.max(coords['z'])-50,np.max(coords['z']+50)])
plt.show()

coords_test = traj_pred[["x","z","frame","particle"]]
print(coords_test)

