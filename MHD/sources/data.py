import tensorflow as tf
import numpy as np
from .parameters import *

# Lower bounds
lb = tf.constant([tmin, xmin, ymin], dtype=DTYPE)
# Upper bounds
ub = tf.constant([tmax, xmax, ymax], dtype=DTYPE)

def gen_train_data(N_r = 10_000, N_0 = 10_000, N_b = 10_000):

    # Set random seed for reproducible results
    tf.random.set_seed(0)

    # Draw uniformly sampled collocation points
    t_r = tf.random.uniform((N_r,1), lb[0], ub[0], dtype=DTYPE)
    x_r = tf.random.uniform((N_r,1), lb[1], ub[1], dtype=DTYPE)
    y_r = tf.random.uniform((N_r,1), lb[2], ub[2], dtype=DTYPE)
    X_r = tf.concat([t_r, x_r, y_r], axis=1)

    # Draw uniform sample points for initial data
    t_0 = tf.ones((N_0,1), dtype=DTYPE)*lb[0]
    x_0 = tf.random.uniform((N_0,1), lb[1], ub[1], dtype=DTYPE)
    y_0 = tf.random.uniform((N_0,1), lb[2], ub[2], dtype=DTYPE)
    X_0 = tf.concat([t_0, x_0, y_0], axis=1)

    # Evaluate intitial condition at (x_0,y_0)
    x_0, y_0 = x_0.numpy(), y_0.numpy()
    omega_0 = fun_omega_0(x_0, y_0)
    phi_0 = fun_phi_0(x_0, y_0)
    J_0 = fun_J_0(x_0, y_0)
    psi_0 = fun_psi_0(x_0, y_0)

    u_0 = tf.concat([omega_0, phi_0, J_0, psi_0], axis=1)

    # Boundary data
    N_b = N_b//2
    t_b1 = tf.random.uniform((N_b,1), lb[0], ub[0], dtype=DTYPE)
    x_b1 = lb[1] + (ub[1] - lb[1]) * tf.keras.backend.random_bernoulli((N_b,1), 0.5, dtype=DTYPE)
    y_b1 = tf.random.uniform((N_b,1), lb[2], ub[2], dtype=DTYPE)

    t_b2 = tf.random.uniform((N_b,1), lb[0], ub[0], dtype=DTYPE)
    x_b2 = tf.random.uniform((N_b,1), lb[1], ub[1], dtype=DTYPE)
    y_b2 = lb[2] + (ub[2] - lb[2]) * tf.keras.backend.random_bernoulli((N_b,1), 0.5, dtype=DTYPE)

    t_b = tf.concat([t_b1, t_b2], axis=0)
    x_b = tf.concat([x_b1, x_b2], axis=0)
    y_b = tf.concat([y_b1, y_b2], axis=0)

    X_b = tf.concat([t_b, x_b, y_b], axis=1)

    # Evaluate boundary condition at (t_b,x_b,y_b)
    x_b, y_b = x_b.numpy(), y_b.numpy()
    omega_b = fun_omega_0(x_b, y_b)
    phi_b = fun_phi_0(x_b, y_b)
    J_b = fun_J_0(x_b, y_b)
    psi_b = fun_psi_0(x_b, y_b)

    u_b = tf.concat([omega_b, phi_b, J_b, psi_b], axis=1)

    # Collect boundary and inital data in lists
    X_data = [X_0, X_b]
    u_data = [u_0, u_b]

    return X_r, X_data, u_data