import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# maniac crazy good with acceleration
# checking my math

# KNOBS GEOMETRY
SDD = 500.0
SOD = 250.0
T = 0.1
theta = np.deg2rad(2.0)

projections = 12
num_p = 10
np.random.seed(num_p)

# TRAJECTORIES KNOBS COMMENT AND UNCOMMENT AS NEEDED
pos0 = np.random.uniform(-1, 1, size=(num_p, 3))
vel0 = np.random.uniform(-1, 1, size=(num_p, 3))
acc0 = np.random.uniform(-1, 1, size=(num_p, 3))
# set to zero to match baseline linear case
#acc0 = np.zeros_like(acc0)
W = projections

pos_all = np.zeros((num_p, W, 3))
x_p = np.zeros((num_p, W))
z_p = np.zeros((num_p, W))
flags = np.zeros((num_p, W))

for p in range(num_p):
    pos = pos0[p].copy()
    vel = vel0[p].copy()
    acc = acc0[p].copy()

    for k in range(W):
        pos_all[p, k] = pos

        phi = k * theta
        c, s = np.cos(-phi), np.sin(-phi)
        # world -> camera
        x_c = c * pos[0] + s * pos[1]
        y_c = -s * pos[0] + c * pos[1]
        z_c = pos[2]

        denom = SOD + y_c
        if np.abs(denom) < 1e-8:
            denom = np.sign(denom) * 1e-8 if denom != 0 else 1e-8
        x_p[p, k] = (SDD / denom) * x_c
        z_p[p, k] = (SDD / denom) * z_c

        # motion update 
        pos = pos + vel * T + 0.5 * acc * T**2
        vel = vel + acc * T

        if np.any(np.abs(pos) > 3.0):
            flags[p, k] = 1.0

# 3) NONLINEAR LEAST-SQUARES SOLVER

def solve_particle_nonlinear(xp_vec, zp_vec, W, SDD, SOD, theta, T,
                             use_acc=True, init_guess=None, verbose=0):


    if use_acc:
        n_params = 9
    else:
        n_params = 6

    if init_guess is None:
        init_guess = np.zeros(n_params)

    def residuals(u):
        if use_acc:
            p0 = u[0:3]; v = u[3:6]; a = u[6:9]
        else:
            p0 = u[0:3]; v = u[3:6]; a = np.zeros(3)

        res = np.zeros(2*W)
        for k in range(W):
            phi = k * theta
            c, s = np.cos(phi), np.sin(phi)
            R = np.array([[c, -s, 0],
                          [s,  c, 0],
                          [0,  0, 1]])
            pos_k = p0 + k*T*v + 0.5*(k*T)**2*a
            x_c, y_c, z_c = R @ pos_k
            denom = SOD + y_c
            if np.abs(denom) < 1e-8:
                denom = np.sign(denom)*1e-8 if denom!=0 else 1e-8
            x_pred = (SDD/denom) * x_c
            z_pred = (SDD/denom) * z_c
            res[2*k]   = xp_vec[k] - x_pred
            res[2*k+1] = zp_vec[k] - z_pred
        return res

    result = least_squares(residuals, init_guess, method='lm', max_nfev=200, verbose=verbose)

    u = result.x
    if use_acc:
        p0, v, a = u[0:3], u[3:6], u[6:9]
    else:
        p0, v, a = u[0:3], u[3:6], np.zeros(3)

    pos_est = np.zeros((W,3))
    for k in range(W):
        pos_est[k] = p0 + k*T*v + 0.5*(k*T)**2*a

    return pos_est, v, a, result


# SOLVEING HERE

all_pos_est = np.zeros_like(pos_all)
all_vel_est = np.zeros((num_p, 3))
all_acc_est = np.zeros((num_p, 3))

for p in range(num_p):
    pos_est, vel_est, acc_est, res = solve_particle_nonlinear(
        xp_vec=x_p[p], zp_vec=z_p[p],
        W=W, SDD=SDD, SOD=SOD, theta=theta, T=T,
        use_acc=True, verbose=0  # change to True to include acceleration
    )
    all_pos_est[p] = pos_est
    all_vel_est[p] = vel_est
    all_acc_est[p] = acc_est

# COMPARISON OUTPUTS
rows = []
for p in range(num_p):
    for k in range(W):
        tx, ty, tz = pos_all[p, k]
        ex, ey, ez = all_pos_est[p, k]
        rows.append({
            "particle": p,
            "frame": k,
            "true_x": tx, "est_x": ex, "err_x": tx - ex,
            "true_y": ty, "est_y": ey, "err_y": ty - ey,
            "true_z": tz, "est_z": ez, "err_z": tz - ez,
        })
tab = pd.DataFrame(rows)
pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 25)
print(tab.to_string(index=False))

vel_rows = []
acc_rows = []
for p in range(num_p):
    vx_t, vy_t, vz_t = vel0[p]
    vx_e, vy_e, vz_e = all_vel_est[p]
    vel_rows.append({
        "particle": p,
        "true_vx": vx_t, "est_vx": vx_e, "err_vx": vx_t - vx_e,
        "true_vy": vy_t, "est_vy": vy_e, "err_vy": vy_t - vy_e,
        "true_vz": vz_t, "est_vz": vz_e, "err_vz": vz_t - vz_e,
    })
    ax_t, ay_t, az_t = acc0[p]
    ax_e, ay_e, az_e = all_acc_est[p]
    acc_rows.append({
        "particle": p,
        "true_ax": ax_t, "est_ax": ax_e, "err_ax": ax_t - ax_e,
        "true_ay": ay_t, "est_ay": ay_e, "err_ay": ay_t - ay_e,
        "true_az": az_t, "est_az": az_e, "err_az": az_t - az_e,
    })

vel_tab = pd.DataFrame(vel_rows)
acc_tab = pd.DataFrame(acc_rows)

print("\n VELOCITY comparison ======")
print(vel_tab.to_string(index=False))
print("\n ACCELERATION comparison ======")
print(acc_tab.to_string(index=False))

for p in range(num_p):
    err_pos = pos_all[p] - all_pos_est[p]
    rmse_pos = np.sqrt(np.mean(err_pos**2, axis=0))
    err_vel = vel0[p] - all_vel_est[p]
    rmse_vel = np.sqrt(np.mean(err_vel**2))
    err_acc = acc0[p] - all_acc_est[p]
    rmse_acc = np.sqrt(np.mean(err_acc**2))

    print(f"\nparticle {p} RMSE:")
    print(f"  pos (x,y,z) = ({rmse_pos[0]:.6f}, {rmse_pos[1]:.6f}, {rmse_pos[2]:.6f})")
    print(f"  vel rmse = {rmse_vel:.6f}")
    print(f"  acc rmse = {rmse_acc:.6f}")
    print(f"  v_est = {all_vel_est[p]}, a_est = {all_acc_est[p]}")
