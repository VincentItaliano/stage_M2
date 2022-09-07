import tensorflow as tf
import matplotlib.pyplot as plt
from .parameters import *

def gen_train_data(lb, ub, nu=False, N_r = 10_000, N_0 = 10_000, N_b = 10_000):

    # Set random seed for reproducible results
    tf.random.set_seed(0)

    # Draw uniformly sampled collocation points
    t_r = tf.random.uniform((N_r,1), lb[0], ub[0], dtype=DTYPE)
    x_r = tf.random.uniform((N_r,1), lb[1], ub[1], dtype=DTYPE)

    if nu:
        nu_r = tf.random.uniform((N_r,1), lb[2], ub[2], dtype=DTYPE)
        X_r = tf.concat([t_r, x_r, nu_r], axis=1)
    else:
        X_r = tf.concat([t_r, x_r], axis=1)

    # Draw uniform sample points for initial boundary data
    t_0 = tf.ones((N_0,1), dtype=DTYPE)*lb[0]
    x_0 = tf.random.uniform((N_0,1), lb[1], ub[1], dtype=DTYPE)

    if nu:
        nu_0 = tf.random.uniform((N_0,1), lb[2], ub[2], dtype=DTYPE)
        X_0 = tf.concat([t_0, x_0, nu_0], axis=1)
    else:
        X_0 = tf.concat([t_0, x_0], axis=1)

    # Evaluate intitial condition at (x_0)
    u_0 = fun_u_0(x_0)

    # Boundary data
    t_b = tf.random.uniform((N_b,1), lb[0], ub[0], dtype=DTYPE)
    x_b = lb[1] + (ub[1] - lb[1]) * tf.keras.backend.random_bernoulli((N_b,1), 0.5, dtype=DTYPE)

    if nu:
        nu_b = tf.random.uniform((N_b,1), lb[2], ub[2], dtype=DTYPE)
        X_b = tf.concat([t_b, x_b, nu_b], axis=1)
    else:
        X_b = tf.concat([t_b, x_b], axis=1)

    # Evaluate boundary condition at (t_b,x_b)
    u_b = fun_u_b(t_b, x_b)

    # Collect boundary and inital data in lists
    X_data = [X_0, X_b]
    u_data = [u_0, u_b]

    return X_r, X_data, u_data

def plot_train_data(X_r, X_data, u_data):

    t_r, x_r = X_r[:,0], X_r[:,1]
    t_0, x_0 = X_data[0][:,0], X_data[0][:,1]
    t_b, x_b = X_data[1][:,0], X_data[1][:,1]

    u_0, u_b = u_data[0][:,0], u_data[1][:,0]

    fig = plt.figure(figsize=(9,6))
    img = plt.scatter(t_r, x_r, c='r', marker='.', alpha=0.2)
    img2 = plt.scatter(t_0, x_0, c=u_0, marker='.', alpha=0.2)
    plt.scatter(t_b, x_b, c=u_b, marker='.', alpha=0.2)
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    cbar = plt.colorbar(img2)
    cbar.set_alpha(1)
    cbar.draw_all()

    plt.title('Positions of collocation points and boundary data');