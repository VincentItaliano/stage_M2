import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import meshio
from .parameters import *


class data_generator:

    def __init__(self, lb, ub):

        self.lb = lb
        self.ub = ub

    def gen_train_data(self, N_r = 10_000, N_0 = 10_000, N_b = 10_000):

        # Set random seed for reproducible results
        tf.random.set_seed(0)

        # Draw uniformly sampled collocation points
        t_r = tf.random.uniform((N_r,1), self.lb[0], self.ub[0], dtype=DTYPE)
        x_r = tf.random.uniform((N_r,1), self.lb[0], self.ub[0], dtype=DTYPE)
        y_r = tf.random.uniform((N_r,1), self.lb[0], self.ub[0], dtype=DTYPE)
        X_r = tf.concat([t_r, x_r, y_r], axis=1)

        # Draw uniform sample points for initial data
        t_0 = tf.ones((N_0,1), dtype=DTYPE)*self.lb[0]
        x_0 = tf.random.uniform((N_0,1), self.lb[1], self.ub[1], dtype=DTYPE)
        y_0 = tf.random.uniform((N_0,1), self.lb[2], self.ub[2], dtype=DTYPE)
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
        t_b1 = tf.random.uniform((N_b,1), self.lb[0], self.ub[0], dtype=DTYPE)
        x_b1 = self.lb[1] + (self.ub[1] - self.lb[1]) * tf.keras.backend.random_bernoulli((N_b,1), 0.5, dtype=DTYPE)
        y_b1 = tf.random.uniform((N_b,1), self.lb[2], self.ub[2], dtype=DTYPE)

        t_b2 = tf.random.uniform((N_b,1), self.lb[0], self.ub[0], dtype=DTYPE)
        x_b2 = tf.random.uniform((N_b,1), self.lb[1], self.ub[1], dtype=DTYPE)
        y_b2 = self.lb[2] + (self.ub[2] - self.lb[2]) * tf.keras.backend.random_bernoulli((N_b,1), 0.5, dtype=DTYPE)

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

    def add_data(self, all_times):

        X_data = []
        u_data = []

        for index, t in enumerate(all_times):

            # load data from files
            reader = meshio.read(DATA_PATH+'tiltC'+str(t)+'.vtk')
            x = reader.points
            J = reader.point_data['current']
            reader = meshio.read(DATA_PATH+'tiltO'+str(t)+'.vtk')
            omega = reader.point_data['vorticity']
            reader = meshio.read(DATA_PATH+'tiltphi'+str(t)+'.vtk')
            phi = reader.point_data['stream']
            reader = meshio.read(DATA_PATH+'tiltpsi'+str(t)+'.vtk')
            psi = reader.point_data['champ']

            my_t = (all_times[index] - all_times[0])/100
            X_data1 = tf.concat([my_t*tf.ones((len(x),1), dtype=DTYPE), x[:,0:1], x[:,1:2]], axis=1)
            u_data1 = np.vstack((omega, phi, J, psi)).T
            u_data1 = tf.cast(u_data1, DTYPE)

            if index==0:
                X_data, u_data = [X_data1], [u_data1]
            else:
                X_data.append(X_data1)
                u_data.append(u_data1)
                #X_data = tf.concat([X_data,X_data1],0)
                #u_data = tf.concat([u_data,u_data1],0)

        return X_data, u_data


def plot_exact_solution(t):

    # load exact solution
    reader = meshio.read(DATA_PATH+'tiltC'+str(t)+'.vtk')
    x = reader.points
    triangles =  reader.cells_dict['triangle']
    J = reader.point_data['current']
    reader = meshio.read(DATA_PATH+'tiltO'+str(t)+'.vtk')
    omega = reader.point_data['vorticity']
    reader = meshio.read(DATA_PATH+'tiltphi'+str(t)+'.vtk')
    phi = reader.point_data['stream']
    reader = meshio.read(DATA_PATH+'tiltpsi'+str(t)+'.vtk')
    psi = reader.point_data['champ']

    # plot
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
    axs = axs.flatten()

    # plot omega
    axs[0].set_xlabel('$x$')
    axs[0].set_ylabel('$y$')
    axs[0].set_aspect('equal')
    img0 = axs[0].tricontourf(x[:,0], x[:,1], triangles, omega, 800, cmap='jet');
    cbar0 = plt.colorbar(img0, ax=axs[0])
    axs[0].set_title('Omega')

    # plot phi
    axs[1].set_xlabel('$x$')
    axs[1].set_ylabel('$y$')
    axs[1].set_aspect('equal')
    img1 = axs[1].tricontourf(x[:,0], x[:,1], triangles, phi, 800, cmap='jet');
    cbar1 = plt.colorbar(img1, ax=axs[1])
    axs[1].set_title('Phi')

    # plot J
    axs[2].set_xlabel('$x$')
    axs[2].set_ylabel('$y$')
    axs[2].set_aspect('equal')
    img2 = axs[2].tricontourf(x[:,0], x[:,1], triangles, J, 800, cmap='jet');
    cbar2 = plt.colorbar(img2, ax=axs[2])
    axs[2].set_title('J')

    # plot psi
    axs[3].set_xlabel('$x$')
    axs[3].set_ylabel('$y$')
    axs[3].set_aspect('equal')
    img3 = axs[3].tricontourf(x[:,0], x[:,1], triangles, psi, 800, cmap='jet');
    cbar3 = plt.colorbar(img3, ax=axs[3])
    axs[3].set_title('Psi')

    #fig.tight_layout()
    plt.suptitle('Exact solution at t={}'.format(t/100));
