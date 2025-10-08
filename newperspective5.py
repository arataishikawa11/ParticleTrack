import numpy as np
import pandas as pd
from scipy.optimize import least_squares

SDD = 500.0     # Source-to-Detector Distance [mm]
SOD = 250.0     # Source-to-Object Distance [mm]
T = 0.1         # Time step between projections [s]
theta = np.deg2rad(2.0)   # Detector rotation per frame [radians]
projections = 12          # Number of projection frames
num_p = 5                 # Number of particles

W = projections

pos0_TRUE = np.random.uniform(-1, 1, size=(num_p, 3))
vel0_TRUE = np.random.uniform(-1, 1, size=(num_p, 3))
acc0_TRUE = np.random.uniform(-1, 1, size=(num_p, 3))

# For baseline static test, zero out velocity/acceleration
#vel0_TRUE = np.zeros_like(vel0_TRUE)
#acc0_TRUE = np.zeros_like(acc0_TRUE)

# setting these up for storage
pos_all_TRUE = np.zeros((num_p, W, 3))
x_PROJECTION = np.zeros((num_p, W))
z_PROJECTION = np.zeros((num_p, W))
flags = np.zeros((num_p, W))

# PROJECT TRUE TRAJECTORY ONTO DETECTOR
for p in range(num_p):
    pos_TRUE = pos0_TRUE[p].copy()
    vel_TRUE = vel0_TRUE[p].copy()
    acc_TRUE = acc0_TRUE[p].copy()

    for k in range(W):
        pos_all_TRUE[p, k] = pos_TRUE

        # Detector rotation (world to camera coordinates)
        phi = k * theta
        c, s = np.cos(-phi), np.sin(-phi)
        x_c = c * pos_TRUE[0] + s * pos_TRUE[1]
        y_c = -s * pos_TRUE[0] + c * pos_TRUE[1]
        z_c = pos_TRUE[2]

        # Perspective projection ... can add noise here later
        denom = SOD + y_c
        if np.abs(denom) < 1e-8:
            denom = np.sign(denom) * 1e-8 if denom != 0 else 1e-8
        x_PROJECTION[p, k] = (SDD / denom) * x_c
        z_PROJECTION[p, k] = (SDD / denom) * z_c

        # Update motion in world frame
        pos_TRUE = pos_TRUE + vel_TRUE * T + 0.5 * acc_TRUE * T**2
        vel_TRUE = vel_TRUE + acc_TRUE * T

        if np.any(np.abs(pos_TRUE) > 3.0):
            flags[p, k] = 1.0

# NONLINEAR LEAST-SQUARES SOLVER

def solve_particle_nonlinear(xp_vec, zp_vec, W, SDD, SOD, theta, T,
                             use_acc=True, init_guess=None, verbose=0):

    n_params = 9 if use_acc else 6
    if init_guess is None:
        init_guess = np.zeros(n_params)

    def residuals(u):
        """Compute projection residuals for current guessed parameters."""
        if use_acc:
            p0_ITERATING = u[0:3]
            v_ITERATING  = u[3:6]
            a_ITERATING  = u[6:9]
        else:
            p0_ITERATING = u[0:3]
            v_ITERATING  = u[3:6]
            a_ITERATING  = np.zeros(3)

        res = np.zeros(2*W)
        for k in range(W):
            phi = k * theta
            c, s = np.cos(phi), np.sin(phi)
            R = np.array([[c, -s, 0],
                          [s,  c, 0],
                          [0,  0, 1]])

            # Predicted world-frame position at this frame
            pos_k_ITERATING = p0_ITERATING + k*T*v_ITERATING + 0.5*(k*T)**2*a_ITERATING

            # Rotate to camera frame
            x_c_ITERATING, y_c_ITERATING, z_c_ITERATING = R @ pos_k_ITERATING

            # Project onto detector
            denom = SOD + y_c_ITERATING
            if np.abs(denom) < 1e-8:
                denom = np.sign(denom) * 1e-8 if denom != 0 else 1e-8
            x_pred_ITERATING = (SDD / denom) * x_c_ITERATING
            z_pred_ITERATING = (SDD / denom) * z_c_ITERATING

            # Residuals in projection space (measured - predicted)
            res[2*k]   = xp_vec[k] - x_pred_ITERATING
            res[2*k+1] = zp_vec[k] - z_pred_ITERATING

        return res

    # Run least-squares optimization
    result = least_squares(residuals, init_guess, method='lm',
                           max_nfev=200, verbose=verbose)

    # final best-guess parameters
    u = result.x
    if use_acc:
        p0_BESTGUESS, v_BESTGUESS, a_BESTGUESS = u[0:3], u[3:6], u[6:9]
    else:
        p0_BESTGUESS, v_BESTGUESS, a_BESTGUESS = u[0:3], u[3:6], np.zeros(3)

    # Reconstruct trajectory from best-guess parameters
    pos_BESTGUESS = np.zeros((W,3))
    for k in range(W):
        pos_BESTGUESS[k] = p0_BESTGUESS + k*T*v_BESTGUESS + 0.5*(k*T)**2*a_BESTGUESS

    return pos_BESTGUESS, v_BESTGUESS, a_BESTGUESS, result


# 5) RUN SOLVER FOR EACH PARTICLE
pos_all_BESTGUESS = np.zeros_like(pos_all_TRUE)
vel_all_BESTGUESS = np.zeros((num_p, 3))
acc_all_BESTGUESS = np.zeros((num_p, 3))

for p in range(num_p):
    pos_BESTGUESS, vel_BESTGUESS, acc_BESTGUESS, res = solve_particle_nonlinear(
        xp_vec=x_PROJECTION[p], zp_vec=z_PROJECTION[p],
        W=W, SDD=SDD, SOD=SOD, theta=theta, T=T,
        use_acc=True, verbose=0
    )
    pos_all_BESTGUESS[p] = pos_BESTGUESS
    vel_all_BESTGUESS[p] = vel_BESTGUESS
    acc_all_BESTGUESS[p] = acc_BESTGUESS

# 6) POST-PROCESSING AND COMPARISON TABLES
rows = []
for p in range(num_p):
    for k in range(W):
        tx, ty, tz = pos_all_TRUE[p, k]
        ex, ey, ez = pos_all_BESTGUESS[p, k]
        rows.append({
            "particle": p,
            "frame": k,
            "x_TRUE": tx, "x_BESTGUESS": ex, "err_x": tx - ex,
            "y_TRUE": ty, "y_BESTGUESS": ey, "err_y": ty - ey,
            "z_TRUE": tz, "z_BESTGUESS": ez, "err_z": tz - ez,
        })

tab = pd.DataFrame(rows)
pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 25)
print(tab.to_string(index=False))

# 7) COMPARISON

vel_rows = []
acc_rows = []

for p in range(num_p):
    vx_t, vy_t, vz_t = vel0_TRUE[p]
    vx_e, vy_e, vz_e = vel_all_BESTGUESS[p]
    vel_rows.append({
        "particle": p,
        "vx_TRUE": vx_t, "vx_BESTGUESS": vx_e, "err_vx": vx_t - vx_e,
        "vy_TRUE": vy_t, "vy_BESTGUESS": vy_e, "err_vy": vy_t - vy_e,
        "vz_TRUE": vz_t, "vz_BESTGUESS": vz_e, "err_vz": vz_t - vz_e,
    })

    ax_t, ay_t, az_t = acc0_TRUE[p]
    ax_e, ay_e, az_e = acc_all_BESTGUESS[p]
    acc_rows.append({
        "particle": p,
        "ax_TRUE": ax_t, "ax_BESTGUESS": ax_e, "err_ax": ax_t - ax_e,
        "ay_TRUE": ay_t, "ay_BESTGUESS": ay_e, "err_ay": ay_t - ay_e,
        "az_TRUE": az_t, "az_BESTGUESS": az_e, "err_az": az_t - az_e,
    })

vel_tab = pd.DataFrame(vel_rows)
acc_tab = pd.DataFrame(acc_rows)

print("\nVELOCITY COMPARISON ======")
print(vel_tab.to_string(index=False))
print("\nACCELERATION COMPARISON ======")
print(acc_tab.to_string(index=False))

for p in range(num_p):
    err_pos = pos_all_TRUE[p] - pos_all_BESTGUESS[p]
    rmse_pos = np.sqrt(np.mean(err_pos**2, axis=0))
    err_vel = vel0_TRUE[p] - vel_all_BESTGUESS[p]
    rmse_vel = np.sqrt(np.mean(err_vel**2))
    err_acc = acc0_TRUE[p] - acc_all_BESTGUESS[p]
    rmse_acc = np.sqrt(np.mean(err_acc**2))

    print(f"\nParticle {p} RMSE:")
    print(f"  pos (x,y,z) = ({rmse_pos[0]:.6f}, {rmse_pos[1]:.6f}, {rmse_pos[2]:.6f})")
    print(f"  vel_rmse = {rmse_vel:.6f}")
    print(f"  acc_rmse = {rmse_acc:.6f}")
    print(f"  v_BESTGUESS = {vel_all_BESTGUESS[p]}, a_BESTGUESS = {acc_all_BESTGUESS[p]}")

