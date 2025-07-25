# import libraries
import matplotlib as mpl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pims # PIMS is a lazy-loading interface to sequential data with numpy-like slicing. https://soft-matter.github.io/pims/v0.7/
import trackpy as tp # trackpy is a particle tracking toolkit https://soft-matter.github.io/trackpy/v0.6.4/

mpl.rc('image', cmap='gray')


# Read data
@pims.pipeline
def gray(image):
    return image[:, :, 1] 

# Set frames. (Pick case)
frames = gray(pims.open('../ParticleTrack/test_cases/stationary/*.png'))

# Note down the size of the frames
#print(frames)
# In this case, 5 frames, shape (445, 800, 4)


# Show an example frame
# plt.imshow(frames[0])
# plt.show()

# Locating features (Gaussian Blob Detection)
# For first frame:
features = tp.locate(frames[0], diameter=11, invert = True) # set invert to true as because features in this image are dark, not bright
# Probably best to decide size experimentally and playing around with it. Better to overshoot

# features is a pandas DataFrame
#print(features)

# Circle detected particles for a certain frame
# tp.annotate(features, frames[0]) # annotate t1.png

# In practice, we could further filter based on a threshold (max or min) of 

# Turn off trackpy console log printing
tp.quiet() 

# Locating features for ALL frames/images
f = tp.batch(frames, 11, invert = True, processes=1, minmass = 50) # By trialand error, found minmass 5000 to be a good number for 'noisy' dataset. Otherwise use 50
#print(f)

# Add z coord column with fixed z values
f['z'] = 10 # arbitrary
#print(f)


# Link features into particle trajectories (no prediction)
traj = tp.link_df(f, search_range=50, memory = 5) # search range of 50 pixels, memory of 3 particles. Search range of 100 for memory test


# Add dynamic prediction
pred = tp.predict.NearestVelocityPredict()
traj_pred = pred.link_df(f, search_range = 50, memory = 5)


# # we could further filter out spurious trajectories if needed with tp.filter_stubs
# # at this point in stage, use experiment to filter out regions of good particles from spurious ones
# # in this case, we have a simple one particle example so not necessary

# Check the number of particles were caught in the trajectory (should be 1 for example_photos_1)
#print("The number of particles detected: " + str(traj_pred['particle'].nunique()))

# # Trace trajectories (2D)
fig, ax = plt.subplots()

# We have two trajectories to choose from, one without prediction (traj) or with dynamic prediction (traj_pred)
tp.plot_traj(traj_pred, ax=ax, colorby='particle', superimpose = frames[0])
#ax.scatter(f['x'],f['y'])
#plt.show()

# Note that current dataframe deals with 2D projections (not 3D lab frame)


# For 3D, use trackpy.plot_traj3d
# To use, we need to use the incorporated z-coords for particles. In this case, use synthetic data.
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
#tp.plot_traj3d(traj_pred, ax=ax, pos_columns = ['x','y','z'], minmass = 50)
#ax.scatter(f['x'], f['y'], f['z'])

#plt.show()

coords = traj_pred[["x","y","z","frame","particle"]]
#print(coords)