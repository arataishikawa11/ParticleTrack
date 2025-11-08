import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import trackpy as tp

#   FIXED CAMERA  —  ROTATING SAMPLE GEOMETRY
# WITH LINEAR LOSS FUNCTION
# with track.py

# KNOBS
SDD = 500.0
SOD = 150.0
T = 0.1
theta = np.deg2rad(2.5)
projections = 60
num_p = 6
W = projections

#   TRUE TRAJECTORY GENERATION  (rotating sample) fixed this :)

# np.random.seed(0)
pos0_TRUE = np.random.uniform(-1, 1, size=(num_p, 3))
vel0_TRUE = np.random.uniform(-1, 1, size=(num_p, 3))
acc0_TRUE = np.random.uniform(-1, 1, size=(num_p, 3))

# turning off acceleration for now to see if that helps
# acc0_TRUE = acc0_TRUE * 0
# use_acc = False
use_acc = True

pos_all_TRUE = np.zeros((num_p, W, 3))
x_PROJECTION = np.zeros((num_p, W))
z_PROJECTION = np.zeros((num_p, W))
flags = np.zeros((num_p, W))

for p in range(num_p):
    pos_TRUE = pos0_TRUE[p].copy()
    vel_TRUE = vel0_TRUE[p].copy()
    acc_TRUE = acc0_TRUE[p].copy()

    for k in range(W):
        pos_all_TRUE[p, k] = pos_TRUE

        # rotating sample (+phi), fixed camera
        phi = k * theta
        c, s = np.cos(phi), np.sin(phi)
        x_c =  c * pos_TRUE[0] - s * pos_TRUE[1]
        y_c =  s * pos_TRUE[0] + c * pos_TRUE[1]
        z_c =  pos_TRUE[2]

        denom = SOD + y_c
        if abs(denom) < 1e-8:
            denom = np.sign(denom) * 1e-8 if denom != 0 else 1e-8
        x_PROJECTION[p, k] = (SDD / denom) * x_c
        z_PROJECTION[p, k] = (SDD / denom) * z_c

        # update WORLD motion
        pos_TRUE = pos_TRUE + vel_TRUE * T + 0.5 * acc_TRUE * T**2
        vel_TRUE = vel_TRUE + acc_TRUE * T

        if np.any(np.abs(pos_TRUE) > 3.0):
            flags[p, k] = 1.0








# FORCING INTO PIXELS AND THEN ADDING NOISE TO PIXEL CENTERS


# depends on detector pixel size and magnification
PIXEL_PITCH_DET = 0.1#72       # mm per pixel
MAGNIFICATION = SDD / SOD
PIXEL_PITCH_OBJ = PIXEL_PITCH_DET / MAGNIFICATION

# saves true projections before grid and noise
x_true = x_PROJECTION.copy()
z_true = z_PROJECTION.copy()

# force into the pixels (to pixel centers)
x_proj_px = np.floor(x_true / PIXEL_PITCH_DET) + 0.5
z_proj_px = np.floor(z_true / PIXEL_PITCH_DET) + 0.5
x_quantized = x_proj_px * PIXEL_PITCH_DET
z_quantized = z_proj_px * PIXEL_PITCH_DET

# add the noise hops
sigma_px = 1.0    # standard deviation in pixels
max_hop = 2    # max integer pixel hop

def discrete_gaussian(shape, sigma, max_hop):
    # draw from normal, round to nearest int, and clip to max_hop
    vals = np.round(np.random.normal(0, sigma, size=shape))
    return np.clip(vals, -max_hop, max_hop).astype(int)

dx_px = discrete_gaussian(x_true.shape, sigma_px, max_hop)
dz_px = discrete_gaussian(z_true.shape, sigma_px, max_hop)

# apply integer pixel hops
x_noisy = (x_proj_px + dx_px) * PIXEL_PITCH_DET
z_noisy = (z_proj_px + dz_px) * PIXEL_PITCH_DET

# replacing projections for solver
x_PROJECTION = x_noisy
z_PROJECTION = z_noisy









# Plotting the new noise
# plotting diagnostic
subset = np.random.choice(num_p, size=min(40, num_p), replace=False)

plt.figure(figsize=(9, 9))
for p in subset:
    plt.plot(x_true[p, :], z_true[p, :],
             'bo', alpha=0.25, label='True' if p == subset[0] else "")
    plt.plot(x_quantized[p, :], z_quantized[p, :],
             'kx', alpha=0.6, label='Quantized center' if p == subset[0] else "")
    plt.plot(x_noisy[p, :], z_noisy[p, :],
             'r.', markersize=6, alpha=0.7, label='Noisy (pixel center)' if p == subset[0] else "")

plt.xlabel('x projection [mm]')
plt.ylabel('z projection [mm]')
plt.title('Projection Noise Diagnostic\nBlue=True, Black=Quantized, Red=Noisy (pixel centers)')
plt.axis('equal')

# pixel grid lines
xlim = plt.xlim(); ylim = plt.ylim()
x_edge_start = np.floor(xlim[0] / PIXEL_PITCH_DET) * PIXEL_PITCH_DET
x_edge_end   = np.ceil(xlim[1] / PIXEL_PITCH_DET) * PIXEL_PITCH_DET
y_edge_start = np.floor(ylim[0] / PIXEL_PITCH_DET) * PIXEL_PITCH_DET
y_edge_end   = np.ceil(ylim[1] / PIXEL_PITCH_DET) * PIXEL_PITCH_DET

for gx in np.arange(x_edge_start, x_edge_end + PIXEL_PITCH_DET, PIXEL_PITCH_DET):
    plt.axvline(gx, color='gray', linewidth=0.4, alpha=0.3)
for gy in np.arange(y_edge_start, y_edge_end + PIXEL_PITCH_DET, PIXEL_PITCH_DET):
    plt.axhline(gy, color='gray', linewidth=0.4, alpha=0.3)

plt.legend(loc='upper right')
plt.grid(False)
plt.show()











### INTEGRATE TRACKPY ###
print("the synthetic data (x): \n" + str(x_PROJECTION))
print("the shape: " + str(np.shape(x_PROJECTION)))
print("X_projection", x_PROJECTION)
print("Z_projection", z_PROJECTION)
# add noise
for k in range(projections):
    noise=np.random.normal(0,0.01,num_p)
    x_PROJECTION[:,k]=x_PROJECTION[:,k]+noise
    z_PROJECTION[:,k]=z_PROJECTION[:,k]+noise
print("X_noise", x_PROJECTION)
print("Z_noise", z_PROJECTION)

# Flatten the arrays
flattened_x = x_PROJECTION.flatten()
flattened_z = z_PROJECTION.flatten()

# Create frames column
frames = np.arange(projections)
frames = np.tile(frames,num_p)

# Shuffle
shuffled_indices = np.random.permutation(len(flattened_x))
flattened_x = flattened_x[shuffled_indices]
flattened_z = flattened_z[shuffled_indices]
frames = frames[shuffled_indices]


data_array = np.array((flattened_x, flattened_z, frames)).T
coords = pd.DataFrame(data_array, columns = ['x','z','frame'])
print(coords)


# Link particle trajectories from coords dataframe with prediction enabled
# Beware of column names 
pred = tp.predict.NearestVelocityPredict()
traj_pred = pred.link_df(coords, search_range = 10, pos_columns = ['x','z'], memory = 5)

# Final coordinate dataframe with particle IDs
coords = traj_pred[["x","z","frame","particle"]]
print(coords)

# Check that trackpy integration doesn't alter the script
print(coords.to_numpy())

# We should be able to reconstruct X_PROJECTION and Z_PROJECTION from coords
xarray = np.zeros((num_p, W)) # Where we store our x data, indexed by (particle, frame)
zarray = np.zeros((num_p, W)) # Same for z data

for p in range(num_p):
    particle_data = coords[coords['particle'] == p]
    xarray[p] = particle_data['x'].to_numpy()
    zarray[p] = particle_data['z'].to_numpy()

# Check that they hold the same values
# Updated the sanity check section that compares the trackpy output to the synthetic data 
# to ensure parrticle trajectories are identical regardless of order in the array
print("xarray = \n" + str(xarray))
print("x_PROJECTION = \n" + str(x_PROJECTION))
sanity_check=np.zeros((np.shape(x_PROJECTION)[0],1))
for r in range(num_p):
    for rr in range(num_p):
        temp=np.array_equal(x_PROJECTION[r,:],xarray[rr,:])
        if temp==True:
            sanity_check[r]=temp
            break

print("Sanity Check:", sanity_check.flatten())
### END ###




































#   NONLINEAR LEASTSQUARES SOLVER
def solve_particle_nonlinear(xp_vec, zp_vec, W, SDD, SOD, theta, T,
                             use_acc=True, init_guess=None, verbose=0):

    n_params = 9 if use_acc else 6
    if init_guess is None:
        init_guess = np.zeros(n_params)

    def residuals(u):
        if use_acc:
            p0, v, a = u[0:3], u[3:6], u[6:9]
        else:
            p0, v, a = u[0:3], u[3:6], np.zeros(3)

        res = np.zeros(2 * W)
        for k in range(W):
            phi = k * theta
            c, s = np.cos(phi), np.sin(phi)
            R = np.array([[ c, -s, 0],
                          [ s,  c, 0],
                          [ 0,  0, 1]])

            # predicted world position
            pos_k = p0 + k * T * v + 0.5 * (k * T)**2 * a

            # rotate sample by +phi 
            x_c, y_c, z_c = R @ pos_k

            denom = SOD + y_c
            if abs(denom) < 1e-8:
                denom = np.sign(denom) * 1e-8 if denom != 0 else 1e-8
            x_pred = (SDD / denom) * x_c
            z_pred = (SDD / denom) * z_c

            res[2*k]   = xp_vec[k] - x_pred
            res[2*k+1] = zp_vec[k] - z_pred
        return res



    # I show in concept6.py that the best loss function is 'linear' here
# I show in concept6.py that the best loss function is 'linear' here
    result = least_squares(
        residuals, init_guess,
        method='trf',
        loss='linear',
        max_nfev=2000,
        verbose=verbose
    )


    u = result.x
    if use_acc:
        p0, v, a = u[0:3], u[3:6], u[6:9]
    else:
        p0, v, a = u[0:3], u[3:6], np.zeros(3)

    pos_BEST = np.zeros((W, 3))
    for k in range(W):
        pos_BEST[k] = p0 + k*T*v + 0.5*(k*T)**2*a

    return pos_BEST, v, a, result


pos_all_BEST = np.zeros_like(pos_all_TRUE)
vel_all_BEST = np.zeros((num_p, 3))
acc_all_BEST = np.zeros((num_p, 3))

for p in range(num_p):
    pos_BEST, vel_BEST, acc_BEST, res = solve_particle_nonlinear(
        xp_vec=x_PROJECTION[p],
        zp_vec=z_PROJECTION[p],
        W=W, SDD=SDD, SOD=SOD,
        theta=theta, T=T,
        use_acc=use_acc, verbose=0
    )
    pos_all_BEST[p] = pos_BEST
    vel_all_BEST[p] = vel_BEST
    acc_all_BEST[p] = acc_BEST















#   POST-PROCESSING

rows = []
for p in range(num_p):
    for k in range(W):
        tx, ty, tz = pos_all_TRUE[p, k]
        ex, ey, ez = pos_all_BEST[p, k]
        rows.append({
            "particle": p, "frame": k,
            "x_TRUE": tx, "x_EST": ex, "err_x": tx - ex,
            "y_TRUE": ty, "y_EST": ey, "err_y": ty - ey,
            "z_TRUE": tz, "z_EST": ez, "err_z": tz - ez,
        })
tab = pd.DataFrame(rows)
pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 25)
print(tab.to_string(index=False))

vel_rows, acc_rows = [], []
for p in range(num_p):
    vx_t, vy_t, vz_t = vel0_TRUE[p]
    vx_e, vy_e, vz_e = vel_all_BEST[p]
    vel_rows.append({
        "particle": p,
        "vx_TRUE": vx_t, "vx_EST": vx_e, "err_vx": vx_t - vx_e,
        "vy_TRUE": vy_t, "vy_EST": vy_e, "err_vy": vy_t - vy_e,
        "vz_TRUE": vz_t, "vz_EST": vz_e, "err_vz": vz_t - vz_e
    })
    ax_t, ay_t, az_t = acc0_TRUE[p]
    ax_e, ay_e, az_e = acc_all_BEST[p]
    acc_rows.append({
        "particle": p,
        "ax_TRUE": ax_t, "ax_EST": ax_e, "err_ax": ax_t - ax_e,
        "ay_TRUE": ay_t, "ay_EST": ay_e, "err_ay": ay_t - ay_e,
        "az_TRUE": az_t, "az_EST": az_e, "err_az": az_t - az_e
    })

vel_tab = pd.DataFrame(vel_rows)
acc_tab = pd.DataFrame(acc_rows)

print("\nVELOCITY COMPARISON ======")
print(vel_tab.to_string(index=False))
print("\nACCELERATION COMPARISON ======")
print(acc_tab.to_string(index=False))

for p in range(num_p):
    err_pos = pos_all_TRUE[p] - pos_all_BEST[p]
    rmse_pos = np.sqrt(np.mean(err_pos**2, axis=0))
    err_vel = vel0_TRUE[p] - vel_all_BEST[p]
    rmse_vel = np.sqrt(np.mean(err_vel**2))
    err_acc = acc0_TRUE[p] - acc_all_BEST[p]
    rmse_acc = np.sqrt(np.mean(err_acc**2))

    print(f"\nParticle {p} RMSE:")
    print(f"  pos (x,y,z) = ({rmse_pos[0]:.6f}, {rmse_pos[1]:.6f}, {rmse_pos[2]:.6f})")
    print(f"  vel_rmse = {rmse_vel:.6f}")
    print(f"  acc_rmse = {rmse_acc:.6f}")
    print(f"  v_EST = {vel_all_BEST[p]}, a_EST = {acc_all_BEST[p]}")



#  (average RMSE over all particles)
rmse_pos_all = np.zeros((num_p, 3))
rmse_vel_all = np.zeros(num_p)
rmse_acc_all = np.zeros(num_p)

for p in range(num_p):
    err_pos = pos_all_TRUE[p] - pos_all_BEST[p]
    rmse_pos_all[p] = np.sqrt(np.mean(err_pos**2, axis=0))

    err_vel = vel0_TRUE[p] - vel_all_BEST[p]
    rmse_vel_all[p] = np.sqrt(np.mean(err_vel**2))

    err_acc = acc0_TRUE[p] - acc_all_BEST[p]
    rmse_acc_all[p] = np.sqrt(np.mean(err_acc**2))

avg_rmse_pos = np.mean(rmse_pos_all, axis=0)
avg_rmse_vel = np.mean(rmse_vel_all)
avg_rmse_acc = np.mean(rmse_acc_all)

print(f"AVERAGE RMSE over {num_p} particles:")
print(f"  pos_rmse_avg (x,y,z) = ({avg_rmse_pos[0]:.6f}, {avg_rmse_pos[1]:.6f}, {avg_rmse_pos[2]:.6f})")
print(f"  vel_rmse_avg = {avg_rmse_vel:.6f}")
print(f"  acc_rmse_avg = {avg_rmse_acc:.6f}")























# THREE DIMENSIONAL PLOTTING OF A SINGLE PARTICLE'S TRAJECTORY
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# WE'RE PICKING THE WOOOORRRSST ONE TO SEE THE DIFFERENCE ALL THE OTHER PARTICLES 
# SHOULD HAVE MUCH NICER CLEANER FITS BUT THIS IS THE WORST ONE
rmse_pos_all = np.zeros((num_p, 3))
for p in range(num_p):
    err = pos_all_TRUE[p] - pos_all_BEST[p]
    rmse_pos_all[p] = np.sqrt(np.mean(err**2, axis=0))
per_particle_rmse = np.linalg.norm(rmse_pos_all, axis=1)
p_show = int(np.argmax(per_particle_rmse))  # or set p_show = 0

P_true = pos_all_TRUE[p_show]   # (W,3)
P_est  = pos_all_BEST[p_show]   # (W,3)
err    = P_true - P_est

# print(f"\n[Plot] Particle {p_show}  RMSE_xyz = {rmse_pos_all[p_show]}  |RMSE| = {per_particle_rmse[p_show]:.6e}")
# print(f" max |error| per-axis: {np.max(np.abs(err), axis=0)}")

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f"TRUE vs EST (particle {p_show})")

# trajectories
ax.plot(P_true[:,0], P_true[:,1], P_true[:,2], color='gray',  lw=1.5, label='TRUE', alpha=0.9)
ax.plot(P_est[:,0],  P_est[:,1],  P_est[:,2],  color='red',   lw=1.8, label='EST',  alpha=0.9)

ax.scatter(P_true[:,0], P_true[:,1], P_true[:,2], s=12, color='gray', alpha=0.8)
ax.scatter(P_est[:,0],  P_est[:,1],  P_est[:,2],  s=12, color='red',  alpha=0.8)

# # tiny quivers to show error direction plotted just to make sure we werent 
# getting like the correct path but wrong direction
# scale = 1.0
# ax.quiver(P_est[:,0], P_est[:,1], P_est[:,2],
#           err[:,0],   err[:,1],   err[:,2],
#           length=1.0, normalize=False, color='black', alpha=0.4)

ax.set_xlabel("X [world]"); ax.set_ylabel("Y [world]"); ax.set_zlabel("Z [world]")
ax.legend(loc='upper left')
ax.view_init(elev=22, azim=45)
ax.grid(True)

# equal aspect ratio
lims = np.array([
    [P_true[:,0].min(), P_true[:,0].max()],
    [P_true[:,1].min(), P_true[:,1].max()],
    [P_true[:,2].min(), P_true[:,2].max()]
])
mins = lims[:,0]; maxs = lims[:,1]
cent = (mins + maxs)/2.0
rad  = np.max(maxs - mins)/2.0
ax.set_xlim(cent[0]-rad, cent[0]+rad)
ax.set_ylim(cent[1]-rad, cent[1]+rad)
ax.set_zlim(cent[2]-rad, cent[2]+rad)

plt.tight_layout()
plt.show()

t = np.arange(W)
fig2, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
labs = ['X', 'Y', 'Z']
for i in range(3):
    axes[i].plot(t, P_true[:,i], '-',  lw=1.8, label='TRUE', color='gray')
    axes[i].plot(t, P_est[:,i],  '--', lw=1.6, label='EST',  color='red')
    axes[i].plot(t, err[:,i],   ':',  lw=1.2, label='ERR',  color='black', alpha=0.6)
    axes[i].set_ylabel(labs[i])
    axes[i].grid(True)
axes[-1].set_xlabel('frame')
axes[0].legend(ncol=3, loc='best')
plt.tight_layout()
plt.show()








# BEST AVERAGE AND WORST
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Compute per-particle RMSE ---
rmse_pos_all = np.zeros((num_p, 3))
for p in range(num_p):
    err = pos_all_TRUE[p] - pos_all_BEST[p]
    rmse_pos_all[p] = np.sqrt(np.mean(err**2, axis=0))
per_particle_rmse = np.linalg.norm(rmse_pos_all, axis=1)

# --- Identify best, median, and worst ---
sorted_idx = np.argsort(per_particle_rmse)
p_best = sorted_idx[0]
p_med = sorted_idx[len(sorted_idx)//2]
p_worst = sorted_idx[-1]

representatives = [
    ("Best", p_best, "green"),
    ("Average", p_med, "orange"),
    ("Worst", p_worst, "red"),
]

# print("\nRepresentative particles:")
# for label, pid, color in representatives:
#     print(f"  {label:<8}  ID={pid:3d}   |RMSE|={per_particle_rmse[pid]:.6e}   RMSE_xyz={rmse_pos_all[pid]}")

# --- Create one figure with 3 subplots ---
fig = plt.figure(figsize=(18, 6))
for i, (label, pid, color) in enumerate(representatives, start=1):
    ax = fig.add_subplot(1, 3, i, projection='3d')

    P_true = pos_all_TRUE[pid]
    P_est  = pos_all_BEST[pid]

    ax.plot(P_true[:,0], P_true[:,1], P_true[:,2],
            color='gray', lw=1.8, label='TRUE', alpha=0.9)
    ax.plot(P_est[:,0], P_est[:,1], P_est[:,2],
            color=color, lw=1.8, label='EST', alpha=0.9)
    ax.scatter(P_true[:,0], P_true[:,1], P_true[:,2],
               s=10, color='gray', alpha=0.7)
    ax.scatter(P_est[:,0], P_est[:,1], P_est[:,2],
               s=10, color=color, alpha=0.7)

    # equal aspect ratio
    lims = np.array([
        [P_true[:,0].min(), P_true[:,0].max()],
        [P_true[:,1].min(), P_true[:,1].max()],
        [P_true[:,2].min(), P_true[:,2].max()]
    ])
    mins = lims[:,0]; maxs = lims[:,1]
    cent = (mins + maxs)/2.0
    rad  = np.max(maxs - mins)/2.0
    ax.set_xlim(cent[0]-rad, cent[0]+rad)
    ax.set_ylim(cent[1]-rad, cent[1]+rad)
    ax.set_zlim(cent[2]-rad, cent[2]+rad)

    # --- Text annotation below subplot ---
    avg_rmse = np.mean(rmse_pos_all[pid])
    ax.set_title(f"{label} RMSE\n(particle {pid})", fontsize=11, pad=10)
    ax.text2D(0.5, -0.12, f"Average position RMSE = {avg_rmse:.6e}",
              transform=ax.transAxes, ha='center', va='center', fontsize=9, color=color)

    ax.set_xlabel("X [world]")
    ax.set_ylabel("Y [world]")
    ax.set_zlabel("Z [world]")
    ax.view_init(elev=20, azim=40)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True)

fig.suptitle("TRUE vs EST Trajectories — Best / Average / Worst RMSE", fontsize=14)
plt.tight_layout()
plt.subplots_adjust(top=0.88, bottom=0.15)
plt.show()
