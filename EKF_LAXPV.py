"""EXTENDED KALMAN FILTER FOR LIMITED ANGLE XRAY PARTICLE VELOCIMETY"""
# By: Alaa Ali
# Date: Oct-2025
# Design an EKF that is robust to noise for limited angle particle tracking
# Upcoming improvements: 
# 1) improve linkage between frames and/or implement trackpy, statistical correlation
# 2) Modify the kalman filter model to incorporate more models (IMM) 

import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

"""""""""" SYNTHETIC DATA GENERATION """""""""""""""
# 1a. Synthetic data generation system dynamics simulation

def f(params,x):
    pos_TRUE=x[0:3] #initialize
    vel_TRUE=x[3:6]
    acc_TRUE=x[6:9]

    # update motion in world coordinates
    pos_TRUE = pos_TRUE + vel_TRUE * params["T"] + 0.5 * acc_TRUE * params["T"]**2
    vel_TRUE = vel_TRUE + acc_TRUE * params["T"]

    x_next=np.hstack((pos_TRUE,vel_TRUE,acc_TRUE))
    return x_next

# 1b. Rotation and world to image coordinates conversion for one particle, i.e. measurement

def h(params,x,k): #k is the projection number/increment  
    # rotating sample (+phi), fixed camera
    phi = k * params["theta"]
    c, s = np.cos(phi), np.sin(phi)
    x_c =  c * x[0] - s * x[1]
    y_c =  s * x[0] + c * x[1]
    z_c =  x[2]
            
    denom = params["SOD"] + y_c
    if abs(denom) < 1e-8:
        denom = np.sign(denom) * 1e-8 if denom != 0 else 1e-8
    x_PROJECTION = (params["SDD"] / denom) * x_c
    z_PROJECTION = (params["SDD"] / denom) * z_c
    
    z_measured = np.hstack((x_PROJECTION,z_PROJECTION))

    return z_measured

"""""""""""""""""END"""""""""""""""""
"""""""""" EKF FUNCTIONS """""""""""""""
#2. System Jacobian Model F 
# (constant acceleration assumed - less constrain on the data acquisition time since velocity and acceleration are computed every time step)
def F_jacobian(params):
    F=np.eye(9)
    F[0:3,3:6] = np.eye(3)*params["T"]
    F[0:3,6:9] = np.eye(3)*0.5*params["T"]**2
    F[3:6,6:9] = np.eye(3)*params["T"]
    return F

#3. Measurement Jacobian Model H (non-linear, hence, use extended kalman filter and partial derivatives)
def H_jacobian(params, x, k):
     phi = k * params["theta"]
     c, s = np.cos(phi), np.sin(phi)
     x_c = c * x[0] - s * x[1]
     y_c = s * x[0] + c * x[1]
     z_c = x[2]
     D = params["SOD"] + y_c
     SDD = params["SDD"]

     H = np.zeros((2, 9))
     H[0, 0] = SDD * (c / D - (x_c * s) / (D**2))
     H[0, 1] = SDD * (-s / D - (x_c * c) / (D**2))
     H[1, 0] = -SDD * (z_c * s) / (D**2)
     H[1, 1] = -SDD * (z_c * c) / (D**2)
     H[1, 2] = SDD / D
     return H

#4. EKF-Predict 
def ekf_predict(params,x,P,Q):
    F=F_jacobian(params)
    x_pred=f(params,x)
    P_pred=F @ P @ F.T + Q
    return x_pred,P_pred

#5. EKF-Update
def ekf_update(params,x_pred,P_pred,z,R,k):
    z_pred=h(params,x_pred,k)
    H=H_jacobian(params,x_pred,k)
    y = z-z_pred
    S=H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S) #Kalman gain
    x_update=x_pred + K @ y
    P_update=(np.eye(len(x_pred))- K @ H) @ P_pred

    return x_update, P_update
"""""""""""""""""END"""""""""""""""""
"""""""""" Particle Tracking FUNCTIONS """""""""""""""
#to be improved
def link_particles(params,zA,zB,max_distance=np.inf): #max distance in pixels
    #Compute the distance between all pairs 
    # And store the index of the closest particle at the current time step to each particle in the previous time step
    dist=distance.cdist(zA,zB,metric='euclidean')
    threshold=dist>max_distance
    dist[threshold] = np.inf
    nearest_ind=np.argmin(dist,axis=1)
    z_nearest = zB[nearest_ind,:]
    return z_nearest
"""""""""""""""""END"""""""""""""""""
# Main Script
# Define system parameters
params={
    "SDD": 500.0,    # Source-to-Detector Distance [mm]
    "SOD": 250.0,     # Source-to-Object Distance [mm]
    "T": 0.2,         # Time step between projections [s]
    "theta": np.deg2rad(0.1),   # Detector rotation per frame [radians]
    "num_projections": 160,        # Number of projection frames
    "num_p": 5,                 # Number of particles
    "motion": "constant"      #stationary, non-accelerating, constant
}

# Initialize state estimates and covariances for all particles at one point in time
x_estimate=np.zeros((params["num_p"],9))
if params["motion"]=="constant":
    P_matrices=np.array([np.eye(9)*5 for _ in range(params["num_p"])]) #3D array (num_p x 9 x 9), uncertainty about initial position
elif params["motion"]=="stationary":
    P_matrices=np.array([np.eye(9)*0.05 for _ in range(params["num_p"])])
else:
    P_matrices=np.array([np.eye(9)*0.5 for _ in range(params["num_p"])])

Q=np.eye(9)*1e-8 #Process noise - how certain are we about the equations
R=np.diag([0.1,0.1]) #Measurement noise variance

# Simulate true data and noisy measurements
np.random.seed(10)  # For reproducibility
x_true = np.zeros((params["num_projections"],params["num_p"],9))
z_measured = np.zeros((params["num_projections"],params["num_p"],2))
x_estimate_ALL_time=np.zeros((params["num_projections"],params["num_p"],9))

#Initial positions, velocities and accelerations
x_true[0,0:params["num_p"],0:3]=np.random.uniform(-2, 2, size=(params["num_p"],3))
if params["motion"]=="non-accelerating":
    x_true[0,0:params["num_p"],3:6]=np.random.uniform(-1, 1, size=(params["num_p"],3))
    x_true[0,0:params["num_p"],6:9]=np.zeros((params["num_p"],3))
elif params["motion"]=="stationary":
    x_true[0,0:params["num_p"],3:6]=np.zeros((params["num_p"],3))
    x_true[0,0:params["num_p"],6:9]=np.zeros((params["num_p"],3))
else:
    x_true[0,0:params["num_p"],3:6]=np.random.uniform(-1, 1, size=(params["num_p"],3))
    x_true[0,0:params["num_p"],6:9]=np.random.uniform(-0.5, 0.5, size=(params["num_p"],3))

for i in range(params["num_p"]):
    z_measured[0,i,:]= h(params,x_true[0,i,:],0)

for k in range(1,params["num_projections"]):
    for i in range(params["num_p"]):
        #true motion
        x_true[k,i] = f(params,x_true[k-1,i])
        z_true = h(params,x_true[k,i],k)
        noise = np.random.multivariate_normal([0,0],R)
        z_measured[k,i]=z_true + noise

#shuffle the frame data
z_shuffle_1=z_measured.copy()
for k in range(1,params["num_projections"]):
    shuffle_ind=np.random.permutation(np.shape(z_measured)[1])
    for i in range(params["num_p"]):     
        z_shuffle_1[k,i,:]=z_measured[k,shuffle_ind[i],:]

z_shuffle_2=z_shuffle_1.copy()

#link the closest particles and restore z_measured from z_shuffle
for k in range (1,params["num_projections"]):
    z_shuffle_2[k,:,:] = link_particles(params, z_shuffle_2[k-1,:,:],z_measured[k,:,:])
        
for k in range(1,params["num_projections"]):
    for i in range(params["num_p"]):
        #prediction
        x_pred, P_pred = ekf_predict(params,x_estimate[i],P_matrices[i,:,:],Q)
        #update
        x_update, P_update = ekf_update(params,x_pred,P_pred,z_measured[k,i],R,k)
        x_estimate[i]=x_update
        P_matrices[i,:,:]=P_update
        x_estimate_ALL_time[k,i] = x_update

#Backward Calculation (using last time step/projection measurements to recalculate the noisefree initial positions, velocities and accelerations):
x_estimate_corrected=np.zeros_like(x_estimate_ALL_time)
x_estimate_corrected[-1,:,:]=x_estimate_ALL_time[-1,:,:]
for k in reversed(range(params["num_projections"]-1)):
    for i in range(params["num_p"]):
        for j in reversed(range(9)):
            if j==8 or j==7 or j==6:
                x_estimate_corrected[k,i,j]=x_estimate_corrected[k+1,i,j]
            elif j==5:
                x_estimate_corrected[k,i,j]=x_estimate_corrected[k+1,i,j]-x_estimate_corrected[k,i,8]*params["T"]
            elif j==4:
                x_estimate_corrected[k,i,j]=x_estimate_corrected[k+1,i,j]-x_estimate_corrected[k,i,7]*params["T"]
            elif j==3:
                x_estimate_corrected[k,i,j]=x_estimate_corrected[k+1,i,j]-x_estimate_corrected[k,i,6]*params["T"]
            elif j==2:
                x_estimate_corrected[k,i,j]=x_estimate_corrected[k+1,i,j]-x_estimate_corrected[k,i,5]*params["T"]-0.5*x_estimate_corrected[k,i,8]*params["T"]**2
            elif j==1:
                x_estimate_corrected[k,i,j]=x_estimate_corrected[k+1,i,j]-x_estimate_corrected[k,i,4]*params["T"]-0.5*x_estimate_corrected[k,i,7]*params["T"]**2
            elif j==0:
                x_estimate_corrected[k,i,j]=x_estimate_corrected[k+1,i,j]-x_estimate_corrected[k,i,3]*params["T"]-0.5*x_estimate_corrected[k,i,6]*params["T"]**2

#Sanity Checks
print(np.array_equal(z_measured,z_shuffle_1))
print(np.array_equal(z_measured,z_shuffle_2))

#Error Calculations
pos_rows = []
vel_rows = []
acc_rows = []

#After correction
for p in range(params["num_p"]):
    for k in range(params["num_projections"]):
        sx, sy, sz = x_true[k,p,0:3]
        esx, esy, esz = x_estimate_corrected[k,p,0:3]

        vx, vy, vz = x_true[k,p,3:6]
        evx, evy, evz = x_estimate_corrected[k,p,3:6]

        ax, ay, az = x_true[k,p,6:9]
        eax, eay, eaz = x_estimate_corrected[k,p,6:9]
        
        if np.mod(k,10) == 0:
            pos_rows.append({
                "particle": p,
                "frame": k,
                "x_true": sx, "sx_estimate": esx, "err_sx": sx - esx,
                "y_true": sy, "sy_estimate": esy, "err_sy": sy - esy,
                "z_true": sz, "sz_estimate": esz, "err_sz": sz - esz,
            })

            vel_rows.append({
                "particle": p,
                "frame": k,
                "vx_true": vx, "vx_estimate": evx, "err_vx": vx - evx,
                "vy_true": vy, "vy_estimate": evy, "err_vy": vy - evy,
                "vz_true": vz, "vz_estimate": evz, "err_vz": vz - evz,
            })

            acc_rows.append({
                "particle": p,
                "frame": k,
                "ax_true": ax, "ax_estimate": eax, "err_x": ax - eax,
                "ay_true": ay, "ay_estimate": eay, "err_y": ay - eay,
                "az_true": az, "az_estimate": eaz, "err_z": az - eaz,
            })
        else:
            continue

pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 25)

pos_tab = pd.DataFrame(pos_rows)
vel_tab = pd.DataFrame(vel_rows)
acc_tab = pd.DataFrame(acc_rows)

print("\nPosition L1 Errors After Correction======")
print(pos_tab.to_string(index=False))
print("\nVelocity L1 Errors After Correction ======")
print(vel_tab.to_string(index=False))
print("\nAcceleration Errors After Correction ======")
print(acc_tab.to_string(index=False))

#RMSE errors per particle between true and estimate at the last projection value
#After correction
for p in range(params["num_p"]):
    err_pos = x_true[-1,p,0:3] - x_estimate_corrected[-1,p,0:3]
    rmse_pos = np.sqrt(np.mean(err_pos**2))
    err_vel =  x_true[-1,p,3:6] - x_estimate_corrected[-1,p,3:6]
    rmse_vel = np.sqrt(np.mean(err_vel**2))
    err_acc =  x_true[-1,p,6:9] - x_estimate_corrected[-1,p,6:9]
    rmse_acc = np.sqrt(np.mean(err_acc**2))

    print(f"\nParticle {p} RMSE:")
    print(f"  pos (x,y,z) = ({rmse_pos:.4e})")
    print(f"  vel_rmse = {rmse_vel:.4e}")
    print(f"  acc_rmse = {rmse_acc:.4e}")

print(np.array_equal(z_measured,z_shuffle_2))
print(np.shape(x_estimate_ALL_time))

# Plot for a single particle to check
plt.figure(1)
plt.plot(x_true[:,4,0], label='True p_x (particle 4)')
plt.plot(x_estimate_ALL_time[:,4,0], '--', label='Estimated p_x (particle 4)')

plt.plot(x_true[:,4,1], label='True p_y (particle 4)')
plt.plot(x_estimate_ALL_time[:,4,1], '--', label='Estimated p_y (particle 4)')

plt.plot(x_true[:,4,2], label='True p_z (particle 4)')
plt.plot(x_estimate_ALL_time[:,4,2], '--', label='Estimated p_z (particle 4)')

plt.xlabel('Time step')
plt.ylabel('Position')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(x_true[:,4,3], label='True v_x (particle 4)')
plt.plot(x_estimate_ALL_time[:,4,3], '--', label='Estimated v_x (particle 4)')

plt.plot(x_true[:,4,4], label='True v_y (particle 4)')
plt.plot(x_estimate_ALL_time[:,4,4], '--', label='Estimated v_y (particle 4)')

plt.plot(x_true[:,4,5], label='True v_z (particle 4)')
plt.plot(x_estimate_ALL_time[:,4,5], '--', label='Estimated v_z (particle 4)')

plt.xlabel('Time step')
plt.ylabel('Velocity')
plt.legend()
plt.show()

plt.figure(3)
plt.plot(x_true[:,4,6], label='True a_x (particle 4)')
plt.plot(x_estimate_ALL_time[:,4,6], '--', label='Estimated a_x (particle 4)')

plt.plot(x_true[:,4,7], label='True a_y (particle 4)')
plt.plot(x_estimate_ALL_time[:,4,7], '--', label='Estimated a_y (particle 4)')

plt.plot(x_true[:,4,8], label='True a_z (particle 4)')
plt.plot(x_estimate_ALL_time[:,4,8], '--', label='Estimated a_z (particle 4)')

plt.xlabel('Time step')
plt.ylabel('Acceleration')
plt.legend()
plt.show()

# Plot after correction values for a single particle to check
plt.figure(4)
plt.plot(x_true[:,4,0], label='True p_x (particle 4)')
plt.plot(x_estimate_corrected[:,4,0], '--', label='Estimated p_x (particle 4)')

plt.plot(x_true[:,4,1], label='True p_y (particle 4)')
plt.plot(x_estimate_corrected[:,4,1], '--', label='Estimated p_y (particle 4)')

plt.plot(x_true[:,4,2], label='True p_z (particle 4)')
plt.plot(x_estimate_corrected[:,4,2], '--', label='Estimated p_z (particle 4)')

plt.xlabel('Time step')
plt.ylabel('Position')
plt.legend()
plt.show()

plt.figure(5)
plt.plot(x_true[:,4,3], label='True v_x (particle 4)')
plt.plot(x_estimate_corrected[:,4,3], '--', label='Estimated v_x (particle 4)')

plt.plot(x_true[:,4,4], label='True v_y (particle 4)')
plt.plot(x_estimate_corrected[:,4,4], '--', label='Estimated v_y (particle 4)')

plt.plot(x_true[:,4,5], label='True v_z (particle 4)')
plt.plot(x_estimate_corrected[:,4,5], '--', label='Estimated v_z (particle 4)')

plt.xlabel('Time step')
plt.ylabel('Velocity')
plt.legend()
plt.show()

plt.figure(6)
plt.plot(x_true[:,4,6], label='True a_x (particle 4)')
plt.plot(x_estimate_corrected[:,4,6], '--', label='Estimated a_x (particle 4)')

plt.plot(x_true[:,4,7], label='True a_y (particle 4)')
plt.plot(x_estimate_corrected[:,4,7], '--', label='Estimated a_y (particle 4)')

plt.plot(x_true[:,4,8], label='True a_z (particle 4)')
plt.plot(x_estimate_corrected[:,4,8], '--', label='Estimated a_z (particle 4)')

plt.xlabel('Time step')
plt.ylabel('Acceleration')
plt.legend()
plt.show()