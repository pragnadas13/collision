#!/usr/bin/env python
# coding: utf-8

# # Redundancy Resolution with Multiple Objective Functions &ndash; Spatial Mechanism

# In[1]:
import sys
sys.path.insert(0, '/home/pragna/Documents/Documents/collision/obstacleavoidance_IK/')
sys.path.insert(0, '/home/pragna/Documents/Documents/collision/obstacleavoidance_IK/main/')
from mechanism_library.Serial.Kinova import kinematics as spatial_mechanism
from main import collision_gradient as collision
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.linalg import expm, logm
from scipy.optimize import approx_fprime
import seaborn as sns


# In[2]:


sns.set(style="whitegrid")
sns.set_palette('tab10')


# In[3]:


#get_ipython().run_line_magic('matplotlib', 'inline')


# ## Simulation Parameters

# In[4]:


# Simulation time step
h = 0.001

# Simulation time
t_initial = 0.0
t_final = 8.0
t_vec = np.arange(t_initial, t_final + h, h)

# Time scaling
A = np.array([[1.0, t_initial, t_initial**2, t_initial**3],
              [0.0, 1.0, 2.0 * t_initial, 3.0 * t_initial**2],
              [1.0, t_final, t_final**2, t_final**3],
              [0.0, 1.0, 2.0 * t_final, 3.0 * t_final**2]])
b = np.array([0.0, 0.0, 1.0, 0.0])

# coefficients of the cubic time scaling polynomial (s(t) = a[0] * t**3+ a[1] * t**2 + a[2] * t + a[3] * t)
a = (la.inv(A) @ b)[::-1]

# coefficients of the first derivative of the cubic time scaling polynomial
ap = np.polyder(a)

# coefficients of the first derivative of the cubic time scaling polynomial
app = np.polyder(ap)

# Initial and final configuration of the manipulator
q_initial = np.array([0.0, -np.pi / 3, 0.0, -np.pi / 3, 0.0, -np.pi / 3, 0.0])

p_initial, R_initial = spatial_mechanism.forward_kinematics(q_initial)
p_final, R_final = spatial_mechanism.forward_kinematics(np.array([np.pi / 6, np.pi / 3, np.pi / 6, np.pi / 3, np.pi / 6, np.pi / 3, 0.0]))

# In homogeneous coordinates
T_initial = np.eye(4)
T_final = np.eye(4)

T_initial[0:3, 3] = p_initial
T_final[0:3, 3] = p_final

T_initial[0:3, 0:3] = R_initial.as_matrix()
T_final[0:3, 0:3] = R_final.as_matrix()

# Joint limits of the manipulator
q_min = np.array([np.finfo(np.float64).min / 10,
                  -126 * np.pi / 180.0,
                  np.finfo(np.float64).min / 10,
                  -147 * np.pi / 180.0,
                  np.finfo(np.float64).min / 10,
                  -117 * np.pi / 180.0,
                  np.finfo(np.float64).min / 10])
q_max = np.array([np.finfo(np.float64).max / 10,
                  126 * np.pi / 180.0,
                  np.finfo(np.float64).max / 10,
                  147 * np.pi / 180.0,
                  np.finfo(np.float64).max / 10,
                  117 * np.pi / 180.0,
                  np.finfo(np.float64).max / 10])


# ## Functions

# Smooth straight trajectory

# In[5]:


def trajectory_generation(T_start, T_end, coeffs, t):
    s_values = np.polyval(coeffs, t)
    sp_values = np.polyval(np.polyder(coeffs), t)
    return np.array([T_start @ expm(logm(la.inv(T_start) @ T_end) * s) for s in s_values]),            np.array([T_start @ logm(la.inv(T_start) @ T_end) @ expm(logm(la.inv(T_start) @ T_end) * s) * sp for (s, sp) in zip(s_values, sp_values)])


# Manipulability metric and its gradient with respect to joint angles

# In[6]:


def manipulability(joint_angles, jacobian_function):
    return np.sqrt(la.det(jacobian_function(joint_angles) @ jacobian_function(joint_angles).transpose()))


# In[7]:


def manipulability_gradient(joint_angles, jacobian_function):
    return approx_fprime(joint_angles, manipulability, np.sqrt(np.finfo(float).eps), jacobian_function)


# Joint limit metric and its gradient with respect to joint angles

# In[8]:


def joint_limits(joint_angles, q_min, q_max):
    q_bar = (q_min + q_max) / 2
    
    return -1 / 6 * np.sum(((joint_angles - q_bar) / (q_max - q_min))**2)


# In[9]:


def joint_limits_gradient(joint_angles, q_min, q_max):
    return approx_fprime(joint_angles, joint_limits, np.sqrt(np.finfo(float).eps), q_min, q_max)


# Mechanism kinematics, Jacobian of the manipulator

# In[10]:


def jacobian_spatial(joint_angles):
#     return spatial_mechanism.jacobian(joint_angles)[0:3, :]
    return spatial_mechanism.jacobian(joint_angles)


# ### Trajectory generation

# In[11]:


# Desired trajectory
desired_trajectory_pos_tf, desired_trajectory_vel_tf = trajectory_generation(T_initial, T_final, a, t_vec)

desired_trajectory_pos = np.array([T[0:3, 3] for T in desired_trajectory_pos_tf])
desired_trajectory_rot = [T[0:3, 0:3] for T in desired_trajectory_pos_tf]

desired_trajectory_lin_vel = np.array([T[0:3, 3] for T in desired_trajectory_vel_tf])
desired_trajectory_ang_vel = np.array([[T[2, 1], T[0, 2], T[1, 0]] for T in desired_trajectory_vel_tf])

desired_trajectory_vel = np.concatenate((desired_trajectory_lin_vel, desired_trajectory_ang_vel), axis=1)


# Numerical calculation of the first derivative

# In[12]:


# T_dot = [(T_now - T_prev) / h for (T_now, T_prev) in zip(desired_trajectory_pos_tf[1:], desired_trajectory_pos_tf[:-1])]

# linear_velocity = np.zeros((len(t_vec), 3))
# angular_velocity = np.zeros((len(t_vec), 3))

# linear_velocity[1:] = np.array([Tp[0:3, 3] for Tp in T_dot])
# angular_velocity[1:] = np.array([[Tp[2, 1], Tp[0, 2], Tp[1, 0]] for Tp in T_dot])


# In[19]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*6.472135955, 4), dpi=96)

ax1.plot(t_vec, desired_trajectory_lin_vel[:, 0], label='x')
ax1.plot(t_vec, desired_trajectory_lin_vel[:, 1], label='y')
ax1.plot(t_vec, desired_trajectory_lin_vel[:, 2], label='z')
ax2.plot(t_vec, desired_trajectory_ang_vel[:, 0], label='x')
ax2.plot(t_vec, desired_trajectory_ang_vel[:, 1], label='y')
ax2.plot(t_vec, desired_trajectory_ang_vel[:, 2], label='z')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Position [m]')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Velocity [m]')
ax2.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
fig.tight_layout()
fig.savefig('Figure_[19].png')


# ### Singularity analysis

# ### Inverse kinematics with least-squares

# In[14]:


q_ls = np.zeros((len(t_vec), 7))
q_ls[0, :] = q_initial

for i, t in enumerate(t_vec[1:]):
    J = jacobian_spatial(q_ls[i, :])
    qp_ls = la.pinv(J) @ desired_trajectory_vel[i, :]
    q_ls[i+1, :] = q_ls[i, :] + qp_ls * h


# In[15]:


fig, ax = plt.subplots(1, 1, figsize=(6.472135955, 4), dpi=96)
ax.plot(t_vec, q_ls[:, 0], label=r'$q_1$')
ax.plot(t_vec, q_ls[:, 1], label=r'$q_2$')
ax.plot(t_vec, q_ls[:, 2], label=r'$q_3$')
ax.plot(t_vec, q_ls[:, 3], label=r'$q_4$')
ax.plot(t_vec, q_ls[:, 4], label=r'$q_5$')
ax.plot(t_vec, q_ls[:, 5], label=r'$q_6$')
ax.plot(t_vec, q_ls[:, 6], label=r'$q_7$')

# Plot joint limits
# Joint 2
# ax.fill_between(t_vec, -126 * np.pi / 180.0, 126 * np.pi / 180.0, facecolor='tab:orange', alpha=0.1)

# Joint 2
# ax.fill_between(t_vec, -147 * np.pi / 180.0, 147 * np.pi / 180.0, facecolor='tab:red', alpha=0.1)

# Joint 3
# ax.fill_between(t_vec, -117 * np.pi / 180.0, 117 * np.pi / 180.0, facecolor='tab:brown', alpha=0.1)

ax.set_xlabel('Time [s]')
ax.set_ylabel('Position [rad]')
ax.set_title('Inverse kinematics (least squares)')
ax.legend(loc='ceq_inputnter right', bbox_to_anchor=(1.2, 0.5))
fig.tight_layout()
fig.savefig('Figure_[15].png')

# In[16]:


# Manipulability on this trajectory with singularity avoidance
m_ls = np.array([manipulability(qi, jacobian_spatial) for qi in q_ls])

# Joint limit metric on this trajectory
j_ls = np.array([joint_limits(qi, q_min, q_max) for qi in q_ls])


# In[17]:


fig, ax = plt.subplots(1, 1, figsize=(6.472135955, 4), dpi=96)
ax.plot(t_vec, m_ls)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Manipulability')
fig.tight_layout()
fig.savefig('Figure_[17].png')

# In[18]:


fig, ax = plt.subplots(1, 1, figsize=(6.472135955, 4), dpi=96)
ax.plot(t_vec, j_ls)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Joint Limit')
fig.tight_layout()
fig.savefig('Figure_[18].png')

# ### Inverse kinematics with damped least-squares

# In[20]:


q_dls = np.zeros((len(t_vec), 7))
q_dls[0, :] = q_initial

for i, t in enumerate(t_vec[1:]):
    J = jacobian_spatial(q_dls[i, :])
    qp_dls = J.transpose() @ la.inv(J @ J.transpose() + 0.001**2 * np.eye(6)) @ desired_trajectory_vel[i, :]
    q_dls[i+1, :] = q_dls[i, :] + qp_dls * h


# In[21]:


fig, ax = plt.subplots(1, 1, figsize=(6.472135955, 4), dpi=96)
ax.plot(t_vec, q_dls[:, 0], label=r'$q_1$')
ax.plot(t_vec, q_dls[:, 1], label=r'$q_2$')
ax.plot(t_vec, q_dls[:, 2], label=r'$q_3$')
ax.plot(t_vec, q_dls[:, 3], label=r'$q_4$')
ax.plot(t_vec, q_dls[:, 4], label=r'$q_5$')
ax.plot(t_vec, q_dls[:, 5], label=r'$q_6$')
ax.plot(t_vec, q_dls[:, 6], label=r'$q_7$')

# Plot joint limits
# Joint 2
# ax.fill_between(t_vec, -126 * np.pi / 180.0, 126 * np.pi / 180.0, facecolor='tab:orange', alpha=0.1)

# Joint 2
# ax.fill_between(t_vec, -147 * np.pi / 180.0, 147 * np.pi / 180.0, facecolor='tab:red', alpha=0.1)

# Joint 3
# ax.fill_between(t_vec, -117 * np.pi / 180.0, 117 * np.pi / 180.0, facecolor='tab:brown', alpha=0.1)

ax.set_xlabel('Time [s]')
ax.set_ylabel('Position [rad]')
ax.set_title('Inverse kinematics (damped least squares)')
ax.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
fig.tight_layout()
fig.savefig('Figure_[21].png')

# In[22]:


# Manipulabilq_vecity on this trajectory with singularity avoidance
m_dls = np.array([manipulability(qi, jacobian_spatial) for qi in q_dls])

# Joint limit metric on this trajectory
j_dls = np.array([joint_limits(qi, q_min, q_max) for qi in q_dls])


# In[23]:


fig, ax = plt.subplots(1, 1, figsize=(6.472135955, 4), dpi=96)
ax.plot(t_vec, m_dls)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Manipulability')
fig.tight_layout()
fig.savefig('Figure_[23].png')

# In[24]:


fig, ax = plt.subplots(1, 1, figsize=(6.472135955, 4), dpi=96)
ax.plot(t_vec, j_dls)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Joint Limit')
fig.tight_layout()


# ### Inverse kinematics with singularity avoidance

# In[29]:


q_sa = np.zeros((len(t_vec), 7))
q_sa[0, :] = q_initial
k0 = 1

for i, t in enumerate(t_vec[1:]):
    J = jacobian_spatial(q_sa[i, :])
    w_der = manipulability_gradient(q_sa[i, :], jacobian_spatial)
    q0 = k0 * w_der
    qp_sa = la.pinv(J) @ desired_trajectory_vel[i, :] + (np.eye(7) - la.pinv(J) @ J) @ q0
    q_sa[i+1, :] = q_sa[i, :] + qp_sa * h


# In[30]:


fig, ax = plt.subplots(1, 1, figsize=(6.472135955, 4), dpi=96)
ax.plot(t_vec, q_sa[:, 0], label=r'$q_1$')
ax.plot(t_vec, q_sa[:, 1], label=r'$q_2$')
ax.plot(t_vec, q_sa[:, 2], label=r'$q_3$')
ax.plot(t_vec, q_sa[:, 3], label=r'$q_4$')
ax.plot(t_vec, q_sa[:, 4], label=r'$q_5$')
ax.plot(t_vec, q_sa[:, 5], label=r'$q_6$')
ax.plot(t_vec, q_sa[:, 6], label=r'$q_7$')

# Plot joint limits
# Joint 2
# ax.fill_between(t_vec, -126 * np.pi / 180.0, 126 * np.pi / 180.0, facecolor='tab:orange', alpha=0.1)

# Joint 2
# ax.fill_between(t_vec, -147 * np.pi / 180.0, 147 * np.pi / 180.0, facecolor='tab:red', alpha=0.1)

# Joint 3
# ax.fill_between(t_vec, -117 * np.pi / 180.0, 117 * np.pi / 180.0, facecolor='tab:brown', alpha=0.1)

ax.set_xlabel('Time [s]')
ax.set_ylabel('Position [rad]')
ax.set_title('Inverse kinematics (singularity avoidance)')
ax.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
fig.tight_layout()
fig.savefig('Figure_[30].png')

# In[31]:


# Manipulability on this trajectory with singularity avoidance
m_sa = np.array([manipulability(qi, jacobian_spatial) for qi in q_sa])

# Joint limit metric on this trajectory
j_sa = np.array([joint_limits(qi, q_min, q_max) for qi in q_sa])


# In[32]:


fig, ax = plt.subplots(1, 1, figsize=(6.472135955, 4), dpi=96)
ax.plot(t_vec, m_sa)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Manipulability')
fig.tight_layout()


# In[33]:


fig, ax = plt.subplots(1, 1, figsize=(6.472135955, 4), dpi=96)
ax.plot(t_vec, j_sa)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Joint Limit')
fig.tight_layout()


# ### Inverse kinematics with joint limits

# In[35]:


q_jl = np.zeros((len(t_vec), 7))
q_jl[0, :] = q_initial

k0 = 1

for i, t in enumerate(t_vec[1:]):
    J = jacobian_spatial(q_jl[i, :])
    w_der = joint_limits_gradient(q_jl[i, :], q_min, q_max)
    q0 = k0 * w_der
    qp_jl = la.pinv(J) @ desired_trajectory_vel[i, :] + (np.eye(7) - la.pinv(J) @ J) @ q0
    q_jl[i+1, :] = q_jl[i, :] + qp_jl * h


# In[36]:


fig, ax = plt.subplots(1, 1, figsize=(6.472135955, 4), dpi=96)
ax.plot(t_vec, q_jl[:, 0], label=r'$q_1$')
ax.plot(t_vec, q_jl[:, 1], label=r'$q_2$')
ax.plot(t_vec, q_jl[:, 2], label=r'$q_3$')
ax.plot(t_vec, q_jl[:, 3], label=r'$q_4$')
ax.plot(t_vec, q_jl[:, 4], label=r'$q_5$')
ax.plot(t_vec, q_jl[:, 5], label=r'$q_6$')
ax.plot(t_vec, q_jl[:, 6], label=r'$q_7$')

# Plot joint limits
# Joint 2
# ax.fill_between(t_vec, -126 * np.pi / 180.0, 126 * np.pi / 180.0, facecolor='tab:orange', alpha=0.1)

# Joint 2
# ax.fill_between(t_vec, -w_der147 * np.pi / 180.0, 147 * np.pi / 180.0, facecolor='tab:red', alpha=0.1)

# Joint 3
# ax.fill_between(t_vec, -117 * np.pi / 180.0, 117 * np.pi / 180.0, facecolor='tab:brown', alpha=0.1)

ax.set_xlabel('Time [s]')
ax.set_ylabel('Position [rad]')
ax.set_title('Inverse kinematics (singularity avoidance)')
ax.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
fig.tight_layout()


# In[37]:


# Manipulability on this trajectory with singularity avoidance
m_jl = np.array([manipulability(qi, jacobian_spatial) for qi in q_jl])

# Joint limit metric on this trajectory
j_jl = np.array([joint_limits(qi, q_min, q_max) for qi in q_jl])


# In[38]:


fig, ax = plt.subplots(1, 1, figsize=(6.472135955, 4), dpi=96)
ax.plot(t_vec, m_jl)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Manipulability')
fig.tight_layout()


# In[39]:


fig, ax = plt.subplots(1, 1, figsize=(6.472135955, 4), dpi=96)
ax.plot(t_vec, j_jl)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Joint Limit')
fig.tight_layout()


# ### Inverse kinematics with singularity avoidance and joint limits

# In[40]:


q_sa = np.zeros((len(t_vec), 7))
q_sa[0, :] = q_initial
k0 = 1

for i, t in enumerate(t_vec[1:]):
    J = jacobian_spatial(q_sa[i, :])
    # In the following line, remove "manipulability_gradient(q_sa[i, :], jacobian_spatial)"
    # and add the gradient of the neural network model
    w_der = collision(q_sa[i, :])
    w_der = w_der.numpy()
    print("W_der", w_der)
    print("W_der.shape", w_der.shape)
    q0 = k0 * w_der
    qp_sa = la.pinv(J) @ desired_trajectory_vel[i, :] + (np.eye(7) - la.pinv(J) @ J) @ q0
    q_sa[i+1, :] = q_sa[i, :] + qp_sa * h


# In[41]:


fig, ax = plt.subplots(1, 1, figsize=(6.472135955, 4), dpi=96)
ax.plot(t_vec, q_sa[:, 0], label=r'$q_1$')
ax.plot(t_vec, q_sa[:, 1], label=r'$q_2$')
ax.plot(t_vec, q_sa[:, 2], label=r'$q_3$')
ax.plot(t_vec, q_sa[:, 3], label=r'$q_4$')
ax.plot(t_vec, q_sa[:, 4], label=r'$q_5$')
ax.plot(t_vec, q_sa[:, 5], label=r'$q_6$')
ax.plot(t_vec, q_sa[:, 6], label=r'$q_7$')

# Plot joint limits
# Joint 2
# ax.fill_between(t_vec, -126 * np.pi / 180.0, 126 * np.pi / 180.0, facecolor='tab:orange', alpha=0.1)

# Joint 2
# ax.fill_between(t_vec, -147 * np.pi / 180.0, 147 * np.pi / 180.0, facecolor='tab:red', alpha=0.1)

# Joint 3
# ax.fill_between(t_vec, -117 * np.pi / 180.0, 117 * np.pi / 180.0, facecolor='tab:brown', alpha=0.1)

ax.set_xlabel('Time [s]')
ax.set_ylabel('Position [rad]')
ax.set_title('Inverse kinematics (multicriteria)')
ax.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
fig.tight_layout()
fig.savefig('Figure_collision.png')

# In[42]:


# Manipulability on this trajectory with singularity avoidance
# m_mc = np.array([manipulability(qi, jacobian_spatial) for qi in q_mc])

# Joint limit metric on this trajectory
#j_mc = np.array([joint_limits(qi, q_min, q_max) for qi in q_mc])


# In[43]:


# fig, ax = plt.subplots(1, 1, figsize=(6.472135955, 4), dpi=96)
# ax.plot(t_vec, m_mc)
# ax.set_xlabel('Time [s]')
# ax.set_ylabel('Manipulability')
# fig.tight_layout()
#
#
# # In[44]:
#
#
# fig, ax = plt.subplots(1, 1, figsize=(6.472135955, 4), dpi=96)
# ax.plot(t_vec, j_mc)
# ax.set_xlabel('Time [s]')
# ax.set_ylabel('Joint Limit')
# fig.tight_layout()
#
#
# # ### All solutions
#
# # In[45]:
#
#
# fig, ax = plt.subplots(1, 1, figsize=(6.472135955, 4), dpi=96)
# ax.plot(t_vec, m_ls, label='Least squares')
# ax.plot(t_vec, m_dls, label='Damped least squares')
# ax.plot(t_vec, m_sa, label='Singularity avoidance')
# ax.plot(t_vec, m_jl, label='Joint limits')
# ax.plot(t_vec, m_mc, label='Multicriteria')
# ax.set_xlabel('Time [s]')
# ax.set_ylabel('Manipulability')
# ax.legend()
# fig.tight_layout()
# fig.savefig('planar_manipulability.png')
#
#
# # In[46]:
#
#
# fig, ax = plt.subplots(1, 1, figsize=(6.472135955, 4), dpi=96)
# ax.plot(t_vec, j_ls, label='Least squares')
# ax.plot(t_vec, j_dls, label='Damped least squares')
# ax.plot(t_vec, j_sa, label='Singularity avoidance')
# ax.plot(t_vec, j_jl, label='Joint limits')
# ax.plot(t_vec, j_mc, label='Multicriteria')
# ax.set_xlabel('Time [s]')
# ax.set_ylabel('Joint Limit')
# ax.legend()
# fig.tight_layout()
# fig.savefig('planar_joint_limits.png')


# ### Different weight combinations

# In[47]:


k0 = 1

alpha_vec = np.arange(0, 1.1, 0.1)
q_pareto = []

for alpha in alpha_vec:
    q_mc = np.zeros((len(t_vec), 7))
    q_mc[0, :] = q_initial
    for i, t in enumerate(t_vec[1:]):
        J = jacobian_spatial(q_mc[i, :])
        w1_der = manipulability_gradient(q_mc[i, :], jacobian_spatial)
        w2_der = joint_limits_gradient(q_mc[i, :], q_min, q_max)
        q0 = k0 * (alpha * w1_der + (1 - alpha) * w2_der)
        qp_mc = la.pinv(J) @ desired_trajectory_vel[i, :] + (np.eye(7) - la.pinv(J) @ J) @ q0
        q_mc[i+1, :] = q_mc[i, :] + qp_mc * h
    q_pareto.append(q_mc) 


# In[48]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*6.472135955, 4), dpi=96)

for q, alpha in zip(q_pareto, alpha_vec):
    manipulability_cost = np.array([manipulability(qi, jacobian_spatial) for qi in q])
    joint_limit_cost = np.array([joint_limits(qi, q_min, q_max) for qi in q])
    
    color = np.random.rand(3,)
    
    if alpha == 0.0 or alpha == 1.0:
        lw = 3
    else:
        lw = 1
    
    ax1.plot(t_vec, manipulability_cost, label=(r'$\alpha=$' + '{0:.1f}'.format(alpha)), color=color, linewidth=lw)
    ax2.plot(t_vec, joint_limit_cost, label=(r'$\alpha=$' + '{0:.1f}'.format(alpha)), color=color, linewidth=lw)
    
ax1.legend(ncol=2, prop={'size': 7})
ax1.set_xlabel('Time [s]')
ax2.set_xlabel('Time [s]')
ax1.set_ylabel('Manipulability')
ax2.set_ylabel('Joint Limits')
fig.tight_layout()
fig.savefig('spatial_pareto.png')


# In[ ]:




