import numpy as np
import pandas as pd

# generate random 3d particles that move linearly (with optional acceleration) 
# ACCELERATION IN THE WORKS IT DOESNT WORK YET CAUSE 
# I ACTUALLY HAVENT ADDED IT TO THE MATRIX YET OR TO THE SYS OF EQS
# i tried it and then it messed up so i think i just need to try again cause it should NOT be that hard
# I am probably just making a silly mistake because its literally just adding another variable the exact same way as the
# other code implements it...
# WILL DO SO SOON

# this takes over initial_vals
SDD = 500.0
SOD = 250.0
T = 0.1
theta = np.deg2rad(6.0)

projections = 12
num_p = 7
np.random.seed(num_p)

# this was testcases
pos0 = np.random.uniform(-1, 1, size=(num_p, 3))
vel0 = np.random.uniform(-1, 1, size=(num_p, 3))
acc0 = np.random.uniform(-1, 1, size=(num_p, 3))
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
        pos_all[p, k, :] = pos

        phi = k * theta
        c, s = np.cos(-phi), np.sin(-phi)

        # world -> camera
        x_c = c * pos[0] + s * pos[1]
        y_c = -s * pos[0] + c * pos[1]
        z_c = pos[2]

        # perspective projection
        denom = SOD + y_c
        if np.abs(denom) < 1e-8:
            denom = np.sign(denom) * 1e-8 if denom != 0 else 1e-8
        x_p[p, k] = (SDD / denom) * x_c
        z_p[p, k] = (SDD / denom) * z_c

        # motion update
        pos = pos + vel * T + 0.5 * acc * T**2

        if np.any(np.abs(pos) > 3.0):
            flags[p, k] = 1.0

frames = np.arange(W)
frames_tiled = np.tile(frames, num_p)
particles_rep = np.repeat(np.arange(num_p), W)
data_array = np.array((x_p.flatten(), z_p.flatten(), frames_tiled, particles_rep)).T
coords_test = pd.DataFrame(data_array, columns=['x', 'z', 'frame', 'particle'])
coords_test['x'] = coords_test['x'].astype(float)
coords_test['z'] = coords_test['z'].astype(float)
coords_test['frame'] = coords_test['frame'].astype(int)
coords_test['particle'] = coords_test['particle'].astype(int)



def solve_particle_from_projections(xp_vec, zp_vec, W, SDD, SOD, theta, T, lambda_reg=1e-8):
    n_pos = 3 * W
    n_vel = 3
    n_unk = n_pos + n_vel
    rows_meas = 2 * W
    rows_kin = 3 * (W - 1) if W > 1 else 0
    rows_total = rows_meas + rows_kin

    M = np.zeros((rows_total, n_unk))
    b = np.zeros(rows_total)

    def ix(k): return 3 * k
    def iy(k): return 3 * k + 1
    def iz(k): return 3 * k + 2

    # measurement eqns
    row = 0
    for k in range(W):
        phi = k * theta
        xpi = xp_vec[k]
        zpi = zp_vec[k]

        B = np.array([
            [ np.cos(phi) - (xpi/SDD)*np.sin(phi), -np.sin(phi) - (xpi/SDD)*np.cos(phi), 0.0 ],
            [ -(zpi/SDD)*np.sin(phi),               -(zpi/SDD)*np.cos(phi),               1.0 ]
        ])
        rhs = np.array([ (SOD/SDD) * xpi, (SOD/SDD) * zpi ])

        M[row:row+2, ix(k):iz(k)+1] = B
        b[row:row+2] = rhs
        row += 2

    # kinematic eqns: p_{k+1} - p_k - T*v = 0
    for k in range(W-1):
        rk = rows_meas + 3*k
        M[rk:rk+3, ix(k+1):iz(k+1)+1] = np.eye(3)
        M[rk:rk+3, ix(k):iz(k)+1]     = -np.eye(3)
        M[rk:rk+3, n_pos:n_pos+3]     = -T*np.eye(3)

    AtA = M.T @ M
    Atb = M.T @ b
    sol = np.linalg.solve(AtA + lambda_reg * np.eye(n_unk), Atb)
    pos_est = sol[:n_pos].reshape(W, 3)
    vel_est = sol[n_pos:n_pos+3]
    return pos_est, vel_est

# solving all particles
all_pos_est = np.zeros_like(pos_all)
all_vel_est = np.zeros((num_p, 3))
for p in range(num_p):
    pos_est, vel_est = solve_particle_from_projections(
        xp_vec=x_p[p], zp_vec=z_p[p],
        W=W, SDD=SDD, SOD=SOD,
        theta=theta, T=T, lambda_reg=1e-8
    )
    all_pos_est[p] = pos_est
    all_vel_est[p] = vel_est

# reporting and tabling
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
    ax_e, ay_e, az_e = (0.0, 0.0, 0.0)
    acc_rows.append({
        "particle": p,
        "true_ax": ax_t, "est_ax": ax_e, "err_ax": ax_t - ax_e,
        "true_ay": ay_t, "est_ay": ay_e, "err_ay": ay_t - ay_e,
        "true_az": az_t, "est_az": az_e, "err_az": az_t - az_e,
    })

vel_tab = pd.DataFrame(vel_rows)
acc_tab = pd.DataFrame(acc_rows)

print("\n=== VELOCITY comparison ===")
print(vel_tab.to_string(index=False))
print("\n=== ACCELERATION comparison ===")
print(acc_tab.to_string(index=False))

for p in range(num_p):
    err_pos = pos_all[p] - all_pos_est[p]
    rmse_pos = np.sqrt(np.mean(err_pos**2, axis=0))
    err_vel = vel0[p] - all_vel_est[p]
    rmse_vel = np.sqrt(np.mean(err_vel**2))
    err_acc = acc0[p] - np.zeros(3)
    rmse_acc = np.sqrt(np.mean(err_acc**2))

    print(f"\nparticle {p} rmse:")
    print(f"  pos (x,y,z) = ({rmse_pos[0]:.6f}, {rmse_pos[1]:.6f}, {rmse_pos[2]:.6f})")
    print(f"  vel rmse = {rmse_vel:.6f}")
    print(f"  acc rmse = {rmse_acc:.6f}")
    print(f"  v_est = {all_vel_est[p]}")
