import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#   FIXED CAMERA  —  ROTATING SAMPLE GEOMETRY
# WITHOUT TRACK PY 
# WITH LINEAR LOSS FUNCTION

# KNOBS
SDD = 500.0
SOD = 150.0
T = 0.15
theta = np.deg2rad(4)
projections = 40
num_p = 100
NOISE_STD = 0.009 # this is BY FAR the most sensitive parameter other than projections/theta... even 0.001 makes a big difference
W = projections

#   TRUE TRAJECTORY GENERATION  (rotating sample) fixed this :)

# np.random.seed(0)
pos0_TRUE = np.random.uniform(-1, 1, size=(num_p, 3))
vel0_TRUE = np.random.uniform(-1, 1, size=(num_p, 3))
acc0_TRUE = np.random.uniform(-1, 1, size=(num_p, 3))

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

# optional: add Gaussian noise which actually turns out to be almost the same as poisson noise 
# depending on the mu variable for poisson which ill ask about
if NOISE_STD > 0:
    x_PROJECTION += np.random.normal(0, NOISE_STD, size=x_PROJECTION.shape)
    z_PROJECTION += np.random.normal(0, NOISE_STD, size=z_PROJECTION.shape)


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
    result = least_squares(
        residuals, init_guess,
        method='trf',
        loss='linear',
        f_scale=max(NOISE_STD, 1e-3),
        max_nfev=5000,
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
        use_acc=True, verbose=0
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

print(f"\n[Plot] Particle {p_show}  RMSE_xyz = {rmse_pos_all[p_show]}  |RMSE| = {per_particle_rmse[p_show]:.6e}")
print(f" max |error| per-axis: {np.max(np.abs(err), axis=0)}")

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
