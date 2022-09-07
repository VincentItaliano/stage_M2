import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from IPython.display import display, clear_output

from .parameters import *

class PINNSolver():
    def __init__(self, model, batch_size):
        self.model = model
        
        # Initialize history of losses and global iteration counter
        self.hist = []
        self.epoch = 0
        self.batch_size = batch_size
        self.best_loss = 1e3
        self.count_since_best_loss = 0
    
    def get_r(self, X):
        
        t, x, y = X[:,0:1], X[:,1:2], X[:,2:3]
        
        with tf.GradientTape(persistent=True) as tape:
            # Watch variables representing t, x and y during this GradientTape
            tape.watch(t)
            tape.watch(x)
            tape.watch(y)
        
            # Compute current values u(t,x,y) and unpack variable
            u = self.model(tf.stack([t[:,0], x[:,0], y[:,0]], axis=1))
            omega, phi, J, psi = u[:,0:1], u[:,1:2], u[:,2:3], u[:,3:4]
            
            phi_x = tape.gradient(phi, x)
            phi_y = tape.gradient(phi, y)
            
            psi_x = tape.gradient(psi, x)
            psi_y = tape.gradient(psi, y)
            
            omega_x = tape.gradient(omega, x)
            omega_y = tape.gradient(omega, y)
            
        omega_t = tape.gradient(omega, t)
        psi_t = tape.gradient(psi, t)
        
        J_x = tape.gradient(J, x)
        J_y = tape.gradient(J, y)
        
        omega_xx = tape.gradient(omega_x, x)
        omega_yy = tape.gradient(omega_y, y)
        psi_xx = tape.gradient(psi_x, x)
        psi_yy = tape.gradient(psi_y, y)
        phi_xx = tape.gradient(phi_x, x)
        phi_yy = tape.gradient(phi_y, y)

        del tape
        
        r1 = omega_t + (omega_x*phi_y - omega_y*phi_x) - (J_x*psi_y - J_y*psi_x) - viscosity*(omega_xx + omega_yy)
        r2 = phi_xx + phi_yy + omega
        r3 = psi_t + (psi_x*phi_y - psi_y*phi_x) - viscosity*(psi_xx + psi_yy) - viscosity*fun_J_0(x,y)
        r4 = psi_xx + psi_yy + J
        
        return r1, r2, r3, r4
    
    def all_losses(self, X_r, X_data, u_data):
        
        # Compute phi_r
        r1, r2, r3, r4= self.get_r(X_r)
        
        phi_r1 = tf.reduce_mean(tf.square(r1))
        phi_r2 = tf.reduce_mean(tf.square(r2))
        phi_r3 = tf.reduce_mean(tf.square(r3))
        phi_r4 = tf.reduce_mean(tf.square(r4))
        
        # Compute phi_0 and phi_b
        u_pred = self.model(X_data[0])
        phi_0 = tf.reduce_mean(tf.square(u_data[0] - u_pred))
        
        u_pred = self.model(X_data[1])
        phi_b = tf.reduce_mean(tf.square(u_data[1] - u_pred))
        
        return phi_r1, phi_r2, phi_r3, phi_r4, phi_0, phi_b
    
    def loss_fn(self, X_r, X_data, u_data):
        
        phi_r1, phi_r2, phi_r3, phi_r4, phi_0, phi_b = self.all_losses(X_r, X_data, u_data)

        # Loss
        loss = 1e12*phi_r1 + 1e10*phi_r2 + 1e4*phi_r3 + 1e10*phi_r4 + 1e10*phi_0
        
        return loss
    
    def get_grad(self, batch, X_data, u_data):
        with tf.GradientTape(persistent=True) as tape:
            # This tape is for derivatives with
            # respect to trainable variables
            tape.watch(self.model.trainable_variables)
            loss = self.loss_fn(batch, X_data, u_data)
            
        g = tape.gradient(loss, self.model.trainable_variables)
        del tape
        
        return loss, g
    

    def solve_with_TFoptimizer(self, optimizer, X_r, X_data, u_data):
        """This method performs a gradient descent type optimization."""
        
        #@tf.function
        def train_step(batch):
            loss, grad_theta = self.get_grad(batch, X_data, u_data)
            
            # Perform gradient descent step
            optimizer.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            return loss
        
        num_epochs = 10_000
        for epoch in range(num_epochs):
                        
            for step in range(1):

                # tire les points 
                indices = np.random.choice(len(X_r), size=self.batch_size, replace=False)
                batch = tf.gather(X_r, indices)

                # descente de gradient
                loss = train_step(batch)

            self.phi_r1, self.phi_r2, self.phi_r3, self.phi_r4, self.phi_0, self.phi_b = self.all_losses(X_r[:1_000], X_data, u_data)
            self.current_loss = self.phi_r1 + self.phi_r2 + self.phi_r3 + self.phi_r4 + self.phi_0 + self.phi_b
            self.callback()
            
        self.model.set_weights(self.best_weights)
        self.model.save_weights(WEIGHT_PATH+'best_weight_MHD.h5')
        
    def callback(self, xr=None):
        self.count_since_best_loss+=1
        
        if self.epoch % 100 == 0:
            clear_output(wait=True)
            print('''Epoch {:06_}: 

                    phi_r1       {:6.4e}  
                    phi_r2       {:6.4e}
                    phi_r3       {:6.4e}
                    phi_r4       {:6.4e}
                    phi_0        {:6.4e}
                    phi_b        {:6.4e}
                    total_loss   {:6.4e}\n'''.format(self.epoch, self.phi_r1, self.phi_r2, self.phi_r3, self.phi_r4, self.phi_0, self.phi_b, self.current_loss), end="")
            
            if self.current_loss < self.best_loss:
                self.best_loss = self.current_loss
                self.count_since_best_loss = 0
                self.best_weights = self.model.get_weights()
                
                if self.epoch % 1_000 == 0:
                    self.model.save_weights(WEIGHT_PATH + 'best_weight_MHD.h5')
                
            if self.count_since_best_loss == 1_000:
                self.model.set_weights(self.best_weights)
                
        self.hist.append(self.current_loss)
        self.epoch+=1
        
    def plot_loss_history(self, ax=None):

        if not ax:
            fig = plt.figure(figsize=(7,5))
            ax = fig.add_subplot(111)

        ax.semilogy(range(len(self.hist)), self.hist, 'k-', label='Loss')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')

        return ax


class multiPINNSolver():
    def __init__(self, model_omega, model_phi, model_J, model_psi, batch_size):

        # Initialize models
        self.model_omega = model_omega
        self.model_phi = model_phi
        self.model_J = model_J
        self.model_psi = model_psi
        
        # Initialize losses history
        self.hist_omega = []
        self.hist_phi = []
        self.hist_J = []
        self.hist_psi = []

        # Initialize global iteration counter
        self.epoch = 0
        self.batch_size = batch_size
        
    def get_r(self, X):
        
        t, x, y = X[:,0:1], X[:,1:2], X[:,2:3]
        
        with tf.GradientTape(persistent=True) as tape:
            # Watch variables representing t, x and y during this GradientTape
            tape.watch(t)
            tape.watch(x)
            tape.watch(y)
        
            # Compute models predictions
            omega = self.model_omega(tf.stack([t[:,0], x[:,0], y[:,0]], axis=1))
            phi = self.model_phi(tf.stack([t[:,0], x[:,0], y[:,0]], axis=1))
            J = self.model_J(tf.stack([t[:,0], x[:,0], y[:,0]], axis=1))
            psi = self.model_psi(tf.stack([t[:,0], x[:,0], y[:,0]], axis=1))
            
            phi_x = tape.gradient(phi, x)
            phi_y = tape.gradient(phi, y)
            
            psi_x = tape.gradient(psi, x)
            psi_y = tape.gradient(psi, y)
            
            omega_x = tape.gradient(omega, x)
            omega_y = tape.gradient(omega, y)
            
        omega_t = tape.gradient(omega, t)
        #psi_t = tape.gradient(psi, t)
        J_t = tape.gradient(J, t)
        
        J_x = tape.gradient(J, x)
        J_y = tape.gradient(J, y)
        
        omega_xx = tape.gradient(omega_x, x)
        omega_yy = tape.gradient(omega_y, y)
        psi_xx = tape.gradient(psi_x, x)
        psi_yy = tape.gradient(psi_y, y)
        phi_xx = tape.gradient(phi_x, x)
        phi_yy = tape.gradient(phi_y, y)

        phi_xy = tape.gradient(phi_x, y)
        psi_xy = tape.gradient(psi_x, x)

        del tape

        g = 2*(phi_xy*(psi_xx-psi_yy)  - psi_xy*(phi_xx-phi_yy))

        r1 = omega_t + (omega_x*phi_y - omega_y*phi_x) - (J_x*psi_y - J_y*psi_x) - viscosity*(omega_xx + omega_yy)
        r2 = phi_xx + phi_yy + omega
        #r3 = psi_t + (psi_x*phi_y - psi_y*phi_x) - viscosity*(psi_xx + psi_yy) - viscosity*fun_J_0(x,y)
        r3 = J_t + (J_x*phi_y - J_y*phi_x)  - (omega_x*psi_y - omega_y*psi_x) - g - viscosity*(psi_xx + psi_yy) - viscosity*fun_J_0(x,y)
        r4 = psi_xx + psi_yy + J
        
        return r1, r2, r3, r4
    
    def all_losses(self, X_r, X_data, u_data):
        
        # Compute residuals
        r1, r2, r3, r4= self.get_r(X_r)
        
        phi_r1 = tf.reduce_mean(tf.square(r1))
        phi_r2 = tf.reduce_mean(tf.square(r2))
        phi_r3 = tf.reduce_mean(tf.square(r3))
        phi_r4 = tf.reduce_mean(tf.square(r4))
        
        # Compute phi_0
        omega_pred = self.model_omega(X_data)
        phi_omega0 = tf.reduce_mean(tf.square(u_data[:,0:1] - omega_pred))

        phi_pred = self.model_phi(X_data)
        phi_phi0 = tf.reduce_mean(tf.square(u_data[:,1:2] - phi_pred))

        J_pred = self.model_J(X_data)
        phi_J0 = tf.reduce_mean(tf.square(u_data[:,2:3] - J_pred))

        psi_pred = self.model_psi(X_data)
        phi_psi0 = tf.reduce_mean(tf.square(u_data[:,3:4] - psi_pred))

        loss_omega, loss_phi, loss_J, loss_psi = phi_r1 + 1e6*phi_omega0, phi_r2 + 1e6*phi_phi0, phi_r3 + 1e6*phi_J0, phi_r4 + 1e6*phi_psi0
        
        return loss_omega, loss_phi, loss_J, loss_psi
    
    def loss_fn(self, X_r, X_data, u_data):
        
        loss_omega, loss_phi, loss_J, loss_psi = self.all_losses(X_r, X_data, u_data)

        return loss_omega, loss_phi, loss_J, loss_psi
    
    def get_grad(self, batch, X_data, u_data):
        with tf.GradientTape(persistent=True) as tape:
            # This tape is for derivatives with
            # respect to trainable variables
            tape.watch(self.model_omega.trainable_variables)
            tape.watch(self.model_phi.trainable_variables)
            tape.watch(self.model_J.trainable_variables)
            tape.watch(self.model_psi.trainable_variables)

            loss_omega, loss_phi, loss_J, loss_psi = self.loss_fn(batch, X_data, u_data)
            
        g_omega = tape.gradient(loss_omega, self.model_omega.trainable_variables)
        g_phi = tape.gradient(loss_phi, self.model_phi.trainable_variables)
        g_J = tape.gradient(loss_J, self.model_J.trainable_variables)
        g_psi = tape.gradient(loss_psi, self.model_psi.trainable_variables)
        del tape
        
        return loss_omega, loss_phi, loss_J, loss_psi, g_omega, g_phi, g_J, g_psi
    

    def solve(self, optimizer_omega, optimizer_phi, optimizer_J, optimizer_psi, X_r, X_data, u_data):
        """This method performs a gradient descent type optimization."""
        
        #@tf.function
        def train_step(batch):
            loss_omega, loss_phi, loss_J, loss_psi, g_omega, g_phi, g_J, g_psi = self.get_grad(batch, X_data, u_data)
            
            # Perform gradient descent step
            optimizer_omega.apply_gradients(zip(g_omega, self.model_omega.trainable_variables))
            optimizer_phi.apply_gradients(zip(g_phi, self.model_phi.trainable_variables))
            optimizer_J.apply_gradients(zip(g_J, self.model_J.trainable_variables))
            optimizer_psi.apply_gradients(zip(g_psi, self.model_psi.trainable_variables))

            return loss_omega, loss_phi, loss_J, loss_psi
        
        num_epochs = 20_000
        for epoch in range(num_epochs):
                        
            for step in range(1):

                # tire les points 
                indices = np.random.choice(len(X_r), size=self.batch_size, replace=False)
                batch = tf.gather(X_r, indices)

                # descente de gradient
                loss_omega, loss_phi, loss_J, loss_psi = train_step(batch)

            self.loss_omega, self.loss_phi, self.loss_J, self.loss_psi = loss_omega, loss_phi, loss_J, loss_psi
            self.current_loss = self.loss_omega + self.loss_phi + self.loss_J + self.loss_psi
            self.callback()
        
    def callback(self, xr=None):
        
        if self.epoch % 100 == 0:
            clear_output(wait=True)
            print('''Epoch {:06_}: 

                    loss_omega   {:6.4e}  
                    loss_phi     {:6.4e}
                    loss_J       {:6.4e}
                    loss_psi     {:6.4e}
                    total_loss   {:6.4e}\n'''.format(self.epoch, self.loss_omega, self.loss_phi, self.loss_J, self.loss_psi, self.current_loss), end="")
                
        self.hist_omega.append(self.loss_omega)
        self.hist_phi.append(self.loss_phi)
        self.hist_J.append(self.loss_J)
        self.hist_psi.append(self.loss_psi)

        self.epoch+=1

    def plot_losses_history(self):

        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
        axs = axs.flatten()

        axs[0].semilogy(range(len(self.hist_omega)), self.hist_omega, 'k-', label='Loss omega')
        axs[0].set_xlabel('Iteration')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        axs[1].semilogy(range(len(self.hist_phi)), self.hist_phi, 'k-', label='Loss phi')
        axs[1].set_xlabel('Iteration')
        axs[1].set_ylabel('Loss')
        axs[1].legend()

        axs[2].semilogy(range(len(self.hist_J)), self.hist_J, 'k-', label='Loss J')
        axs[2].set_xlabel('Iteration')
        axs[2].set_ylabel('Loss')
        axs[2].legend()

        axs[3].semilogy(range(len(self.hist_psi)), self.hist_psi, 'k-', label='Loss psi')
        axs[3].set_xlabel('Iteration')
        axs[3].set_ylabel('Loss')
        axs[3].legend()

        fig.tight_layout()


class PINNSolver_2var():
    def __init__(self, model, batch_size):
        self.model = model
        
        # Initialize history of losses and global iteration counter
        self.hist = []
        self.epoch = 0
        self.batch_size = batch_size
        self.best_loss = 1.0
        self.count_since_best_loss = 0
    
    def get_r(self, X):
        
        t, x, y = X[:,0:1], X[:,1:2], X[:,2:3]
        
        with tf.GradientTape(persistent=True) as tape:
            # Watch variables representing t, x and y during this GradientTape
            tape.watch(t)
            tape.watch(x)
            tape.watch(y)
        
            # Compute current values u(t,x,y) and unpack variable
            u = self.model(tf.stack([t[:,0], x[:,0], y[:,0]], axis=1))
            phi, psi = u[:,0:1], u[:,1:2]
            
            phi_x = tape.gradient(phi, x)
            phi_y = tape.gradient(phi, y)
            
            psi_x = tape.gradient(psi, x)
            psi_y = tape.gradient(psi, y)

            psi_xx = tape.gradient(psi_x, x)
            psi_yy = tape.gradient(psi_y, y)
            phi_xx = tape.gradient(phi_x, x)
            phi_yy = tape.gradient(phi_y, y)

            laplacien_phi = phi_xx + phi_yy
            laplacien_psi = psi_xx + psi_yy

            phi_xxx = tape.gradient(phi_xx, x)
            phi_yyy = tape.gradient(phi_yy, y)

            phi_xxy = tape.gradient(phi_xx, y)
            phi_yyx = tape.gradient(phi_yy, x)
            
        psi_t = tape.gradient(psi, t)
        
        phi_xxt = tape.gradient(phi_xx, t)
        phi_yyt = tape.gradient(phi_yy, t)
        phi_xxxx = tape.gradient(phi_xxx, x)
        phi_yyyy = tape.gradient(phi_yyy, y)
        phi_xxyy = tape.gradient(phi_xxy, y)

        laplacien_phi_x = tape.gradient(laplacien_phi, x)
        laplacien_phi_y = tape.gradient(laplacien_phi, y)

        laplacien_psi_x = tape.gradient(laplacien_psi, x)
        laplacien_psi_y = tape.gradient(laplacien_psi, y)

        del tape
        
        phi_laplacienphi = (phi_x*laplacien_phi_y - phi_y*laplacien_phi_x)
        psi_laplacienpsi = (psi_x*laplacien_psi_y - psi_y*laplacien_psi_x)

        r1 = viscosity*(phi_xxxx + 2*phi_xxyy + phi_yyyy) -  phi_xxt - phi_yyt + phi_laplacienphi - psi_laplacienpsi
        r2 = psi_t + (psi_x*phi_y - psi_y*phi_x) - viscosity*(psi_xx + psi_yy) - viscosity*fun_J_0(x,y)
        
        return r1, r2
    
    def all_losses(self, X_r, X_data, u_data):
        
        # Compute phi_r
        r1, r2 = self.get_r(X_r)
        
        phi_r1 = tf.reduce_mean(tf.square(r1))
        phi_r2 = tf.reduce_mean(tf.square(r2))
        
        # Compute phi_0 and phi_b
        u_pred = self.model(X_data[0])
        phi_0 = tf.reduce_mean(tf.square(u_data[0] - u_pred))
         
        return phi_r1, phi_r2, phi_0
    
    def loss_fn(self, X_r, X_data, u_data):
        
        phi_r1, phi_r2, phi_0 = self.all_losses(X_r, X_data, u_data)

        # Loss
        loss = 1e5*phi_r1 + 1e4*phi_r2 + 1e5*phi_0 
        
        return loss
    
    def get_grad(self, batch, X_data, u_data):
        with tf.GradientTape(persistent=True) as tape:
            # This tape is for derivatives with
            # respect to trainable variables
            tape.watch(self.model.trainable_variables)
            loss = self.loss_fn(batch, X_data, u_data)
            
        g = tape.gradient(loss, self.model.trainable_variables)
        del tape
        
        return loss, g
    

    def solve_with_TFoptimizer(self, optimizer, X_r, X_data, u_data):
        """This method performs a gradient descent type optimization."""
        
        #@tf.function
        def train_step(batch):
            loss, grad_theta = self.get_grad(batch, X_data, u_data)
            
            # Perform gradient descent step
            optimizer.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            return loss
        
        num_epochs = 20_000
        for epoch in range(num_epochs):
                        
            for step in range(1):

                # tire les points 
                indices = np.random.choice(len(X_r), size=self.batch_size, replace=False)
                batch = tf.gather(X_r, indices)

                # descente de gradient
                loss = train_step(batch)

            self.phi_r1, self.phi_r2, self.phi_0 = self.all_losses(X_r[:100], X_data, u_data)
            self.current_loss = self.phi_r1 + self.phi_r2 + self.phi_0
            self.callback()
            
        self.model.load_weights(WEIGHT_PATH + 'best_weight_MHD_2var.h5')
            
    def callback(self, xr=None):
        self.count_since_best_loss+=1
        
        if self.epoch % 100 == 0:
            clear_output(wait=True)
            print('''Epoch {:06_}: 

                    phi_r1       {:6.4e}  
                    phi_r2       {:6.4e}
                    phi_0        {:6.4e}
                    total_loss   {:6.4e}\n'''.format(self.epoch, self.phi_r1, self.phi_r2, self.phi_0, self.current_loss), end="")
            
            if self.current_loss < self.best_loss:
                self.best_loss = self.current_loss
                self.count_since_best_loss = 0
                self.best_weights = self.model.get_weights()
                
                if self.epoch % 1_000 == 0:
                    self.model.save_weights(WEIGHT_PATH + 'best_weight_MHD_2var.h5')
                
            if self.count_since_best_loss == 1_000:
                self.model.set_weights(self.best_weights)
                
        self.hist.append(self.current_loss)
        self.epoch+=1
        
    def plot_loss_history(self, ax=None):

        if not ax:
            fig = plt.figure(figsize=(7,5))
            ax = fig.add_subplot(111)

        ax.semilogy(range(len(self.hist)), self.hist, 'k-', label='Loss')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')

        return ax

class Baseline():
    def __init__(self, model, batch_size):
        self.model = model
        
        # Initialize history of losses and global iteration counter
        self.hist = []
        self.epoch = 0
        self.batch_size = batch_size
        self.best_loss = 1e3
        self.count_since_best_loss = 0

    def loss_fn(self, X_r, X_data, u_data):
        
        u_pred = self.model(X_data)
        phi_0 = tf.reduce_mean(tf.square(u_data - u_pred))
        loss =  1e4*phi_0
        
        return loss
    
    def get_grad(self, batch, X_data, u_data):
        with tf.GradientTape(persistent=True) as tape:
            # This tape is for derivatives with
            # respect to trainable variables
            tape.watch(self.model.trainable_variables)
            loss = self.loss_fn(batch, X_data, u_data)
            
        g = tape.gradient(loss, self.model.trainable_variables)
        del tape
        
        return loss, g
    

    def solve_with_TFoptimizer(self, optimizer, X_r, X_data, u_data):
        """This method performs a gradient descent type optimization."""
        
        #@tf.function
        def train_step(batch):
            loss, grad_theta = self.get_grad(batch, X_data, u_data)
            
            # Perform gradient descent step
            optimizer.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            return loss
        
        num_epochs = 10_000
        for epoch in range(num_epochs):
                        
            for step in range(1):

                # tire les points 
                indices = np.random.choice(len(X_r), size=self.batch_size, replace=False)
                batch = tf.gather(X_r, indices)

                # descente de gradient
                loss = train_step(batch)

            self.phi_0 = self.loss_fn(X_r[:1_000], X_data, u_data)
            self.current_loss = self.phi_0
            self.callback()
            
        self.model.set_weights(self.best_weights)
        self.model.save_weights(WEIGHT_PATH+'best_weight_MHD.h5')
        
    def callback(self, xr=None):
        self.count_since_best_loss+=1
        
        if self.epoch % 100 == 0:
            clear_output(wait=True)
            print('''Epoch {:06_}: 

                    phi_0        {:6.4e}
                    total_loss   {:6.4e}\n'''.format(self.epoch, self.phi_0, self.current_loss), end="")
            
            if self.current_loss < self.best_loss:
                self.best_loss = self.current_loss
                self.count_since_best_loss = 0
                self.best_weights = self.model.get_weights()
                
                if self.epoch % 1_000 == 0:
                    self.model.save_weights(WEIGHT_PATH + 'best_weight_MHD.h5')
                
            if self.count_since_best_loss == 1_000:
                self.model.set_weights(self.best_weights)
                
        self.hist.append(self.current_loss)
        self.epoch+=1
        
    def plot_loss_history(self, ax=None):

        if not ax:
            fig = plt.figure(figsize=(7,5))
            ax = fig.add_subplot(111)

        ax.semilogy(range(len(self.hist)), self.hist, 'k-', label='Loss')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')

        return ax

class VinciSolver():
    def __init__(self, model, batch_size):
        self.model = model
        
        # Initialize history of losses and global iteration counter
        self.hist = []
        self.epoch = 0
        self.batch_size = batch_size
        self.best_loss = 1e3
        self.count_since_best_loss = 0
    
    def get_r(self, X):
        
        t, x, y = X[:,0:1], X[:,1:2], X[:,2:3]
        
        with tf.GradientTape(persistent=True) as tape:
            # Watch variables representing t, x and y during this GradientTape
            tape.watch(t)
            tape.watch(x)
            tape.watch(y)
        
            # Compute current values u(t,x,y) and unpack variable
            u = self.model(tf.stack([t[:,0], x[:,0], y[:,0]], axis=1))
            omega, phi, J, psi = u[:,0:1], u[:,1:2], u[:,2:3], u[:,3:4]
            
            phi_x = tape.gradient(phi, x)
            phi_y = tape.gradient(phi, y)
            
            psi_x = tape.gradient(psi, x)
            psi_y = tape.gradient(psi, y)
            
            omega_x = tape.gradient(omega, x)
            omega_y = tape.gradient(omega, y)
            
        omega_t = tape.gradient(omega, t)
        psi_t = tape.gradient(psi, t)
        
        J_x = tape.gradient(J, x)
        J_y = tape.gradient(J, y)
        
        omega_xx = tape.gradient(omega_x, x)
        omega_yy = tape.gradient(omega_y, y)
        psi_xx = tape.gradient(psi_x, x)
        psi_yy = tape.gradient(psi_y, y)
        phi_xx = tape.gradient(phi_x, x)
        phi_yy = tape.gradient(phi_y, y)

        del tape
        
        r1 = omega_t + (omega_x*phi_y - omega_y*phi_x) - (J_x*psi_y - J_y*psi_x) - viscosity*(omega_xx + omega_yy)
        r2 = phi_xx + phi_yy + omega
        r3 = psi_t + (psi_x*phi_y - psi_y*phi_x) - viscosity*(psi_xx + psi_yy) - viscosity*fun_J_0(x,y)
        r4 = psi_xx + psi_yy + J
        
        return r1, r2, r3, r4
    
    def all_losses(self, X_r, X_data, u_data):
        
        # Compute phi_r
        r1, r2, r3, r4= self.get_r(X_r)
        
        phi_r1 = tf.reduce_mean(tf.square(r1))
        phi_r2 = tf.reduce_mean(tf.square(r2))
        phi_r3 = tf.reduce_mean(tf.square(r3))
        phi_r4 = tf.reduce_mean(tf.square(r4))
        
        # Compute phi_0
        u_pred = self.model(X_data)
        phi_0_omega = tf.reduce_mean(tf.square(u_data[:,0:1] - u_pred[:,0:1]))
        phi_0_phi = tf.reduce_mean(tf.square(u_data[:,1:2] - u_pred[:,1:2]))
        phi_0_J = tf.reduce_mean(tf.square(u_data[:,2:3] - u_pred[:,2:3]))
        phi_0_psi = tf.reduce_mean(tf.square(u_data[:,3:4] - u_pred[:,3:4]))
        
       
        return phi_r1, phi_r2, phi_r3, phi_r4, phi_0_omega, phi_0_phi, phi_0_J, phi_0_psi

    def loss_fn(self, X_r, X_data, u_data):
        
        phi_r1, phi_r2, phi_r3, phi_r4, phi_0_omega, phi_0_phi, phi_0_J, phi_0_psi = self.all_losses(X_r, X_data, u_data)

        # Loss
        loss = 1e12*phi_r1 + 1e12*phi_r2 + 1e7*phi_r3 + 1e12*phi_r4 + 1e14*phi_0_omega + 1e14*phi_0_phi + 1e12*phi_0_J + 1e11*phi_0_psi
        
        return loss
    
    def get_grad(self, batch, X_data, u_data):
        with tf.GradientTape(persistent=True) as tape:
            # This tape is for derivatives with
            # respect to trainable variables
            tape.watch(self.model.trainable_variables)
            loss = self.loss_fn(batch, X_data, u_data)
            
        g = tape.gradient(loss, self.model.trainable_variables)
        del tape
        
        return loss, g
    

    def solve_with_TFoptimizer(self, optimizer, X_r, X_data, u_data):
        """This method performs a gradient descent type optimization."""
        
        #@tf.function
        def train_step(batch):
            loss, grad_theta = self.get_grad(batch, X_data, u_data)
            
            # Perform gradient descent step
            optimizer.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            return loss
        
        num_epochs = 10_000
        for epoch in range(num_epochs):
                        
            for step in range(1):

                # tire les points 
                indices = np.random.choice(len(X_r), size=self.batch_size, replace=False)
                batch = tf.gather(X_r, indices)

                # descente de gradient
                loss = train_step(batch)

            self.phi_r1, self.phi_r2, self.phi_r3, self.phi_r4, self.phi_0_omega, self.phi_0_phi, self.phi_0_J, self.phi_0_psi = self.all_losses(X_r[:1_000], X_data, u_data)
            self.current_loss = self.phi_r1 + self.phi_r2 + self.phi_r3 + self.phi_r4 + self.phi_0_omega + self.phi_0_phi + self.phi_0_J + self.phi_0_psi
            self.callback()
            
        self.model.set_weights(self.best_weights)
        self.model.save_weights(WEIGHT_PATH+'best_weight_MHD.h5')
        
    def callback(self, xr=None):
        self.count_since_best_loss+=1
        
        if self.epoch % 100 == 0:
            clear_output(wait=True)
            print('''Epoch {:06_}: 

                    phi_r1       {:6.4e}  
                    phi_r2       {:6.4e}
                    phi_r3       {:6.4e}
                    phi_r4       {:6.4e}
                    phi_0_omega  {:6.4e}
                    phi_0_phi    {:6.4e}
                    phi_0_J      {:6.4e}
                    phi_0_psi    {:6.4e}
                    total_loss   {:6.4e}\n'''.format(self.epoch, self.phi_r1, self.phi_r2, self.phi_r3, self.phi_r4, self.phi_0_omega, self.phi_0_phi, self.phi_0_J, self.phi_0_psi, self.current_loss), end="")
            
            if self.current_loss < self.best_loss:
                self.best_loss = self.current_loss
                self.count_since_best_loss = 0
                self.best_weights = self.model.get_weights()
                
                if self.epoch % 1_000 == 0:
                    self.model.save_weights(WEIGHT_PATH + 'best_weight_MHD.h5')
                
            if self.count_since_best_loss == 1_000:
                self.model.set_weights(self.best_weights)
                
        self.hist.append(self.current_loss)
        self.epoch+=1
        
    def plot_loss_history(self, ax=None):

        if not ax:
            fig = plt.figure(figsize=(7,5))
            ax = fig.add_subplot(111)

        ax.semilogy(range(len(self.hist)), self.hist, 'k-', label='Loss')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')

        return ax


class vincisolver2():
    def __init__(self, model_hydro, model_magneto, batch_size):

        # Initialize models
        self.model_hydro = model_hydro
        self.model_magneto = model_magneto
        
        # Initialize losses history
        self.hist_hydro = []
        self.hist_magneto = []

        self.hist_r1 = []
        self.hist_r2 = []
        self.hist_r3 = []
        self.hist_r4 = []
        self.hist_phi_0_omega = []
        self.hist_phi_0_phi = []
        self.hist_phi_0_J = []
        self.hist_phi_0_psi= []

        # Initialize global iteration counter
        self.epoch = 0
        self.batch_size = batch_size

        # Weights for the data fitting part 
        self.data_weights = tf.constant([1e2, 1e1, 1., 1., 1., 1., 1., 1., 1., 0.5, 1e-1, 1e-1, 1])
        
    def get_r(self, X):
        
        t, x, y = X[:,0:1], X[:,1:2], X[:,2:3]
        
        with tf.GradientTape(persistent=True) as tape:
            # Watch variables representing t, x and y during this GradientTape
            tape.watch(t)
            tape.watch(x)
            tape.watch(y)
        
            # Compute models predictions
            hydro = self.model_hydro(tf.stack([t[:,0], x[:,0], y[:,0]], axis=1))
            magneto = self.model_magneto(tf.stack([t[:,0], x[:,0], y[:,0]], axis=1))
            omega, phi = hydro[:,0:1], hydro[:,1:2]
            J, psi = magneto[:,0:1], magneto[:,1:2]
            
            # compute derivatives
            phi_x = tape.gradient(phi, x)
            phi_y = tape.gradient(phi, y)
            
            psi_x = tape.gradient(psi, x)
            psi_y = tape.gradient(psi, y)
            
            omega_x = tape.gradient(omega, x)
            omega_y = tape.gradient(omega, y)
            
        omega_t = tape.gradient(omega, t)
        psi_t = tape.gradient(psi, t)
        
        J_x = tape.gradient(J, x)
        J_y = tape.gradient(J, y)
        
        omega_xx = tape.gradient(omega_x, x)
        omega_yy = tape.gradient(omega_y, y)
        psi_xx = tape.gradient(psi_x, x)
        psi_yy = tape.gradient(psi_y, y)
        phi_xx = tape.gradient(phi_x, x)
        phi_yy = tape.gradient(phi_y, y)

        del tape
        
        r1 = omega_t + (omega_x*phi_y - omega_y*phi_x) - (J_x*psi_y - J_y*psi_x) - viscosity*(omega_xx + omega_yy)
        r2 = phi_xx + phi_yy + omega
        r3 = psi_t + (psi_x*phi_y - psi_y*phi_x) - viscosity*(psi_xx + psi_yy) - viscosity*fun_J_0(x,y)
        r4 = psi_xx + psi_yy + J
        
        return r1, r2, r3, r4
    
    def all_losses(self, X_r, X_data, u_data):
        
        # Compute residuals
        r1, r2, r3, r4 = self.get_r(X_r)
        
        phi_r1 = tf.reduce_mean(tf.square(r1))
        phi_r2 = tf.reduce_mean(tf.square(r2))
        phi_r3 = tf.reduce_mean(tf.square(r3))
        phi_r4 = tf.reduce_mean(tf.square(r4))
        
        # Compute phi_0
        phi_0_omega = 0
        phi_0_phi = 0
        phi_0_J = 0
        phi_0_psi = 0

        for i in range(len(X_data)):

            #indices = np.random.choice(len(X_data[i]), size=3_000, replace=False)
            #X = tf.gather(X_data[i], indices)
            #u = tf.gather(u_data[i], indices)

            X = X_data[i]
            u = u_data[i]

            hydro_pred = self.model_hydro(X)
            magneto_pred = self.model_magneto(X)

            phi_0_omega += self.data_weights[i] * tf.reduce_mean(tf.square(u[:,0:1] - hydro_pred[:,0:1]))
            phi_0_phi +=  self.data_weights[i] * tf.reduce_mean(tf.square(u[:,1:2] - hydro_pred[:,1:2]))
            phi_0_J +=  self.data_weights[i] * tf.reduce_mean(tf.square(u[:,2:3] - magneto_pred[:,0:1]))
            phi_0_psi +=  self.data_weights[i] * tf.reduce_mean(tf.square(u[:,3:4] - magneto_pred[:,1:2]))
        
        return phi_r1, phi_r2, phi_r3, phi_r4, phi_0_omega, phi_0_phi, phi_0_J, phi_0_psi
    
    def loss_fn(self, X_r, X_data, u_data):
        
        phi_r1, phi_r2, phi_r3, phi_r4, phi_0_omega, phi_0_phi, phi_0_J, phi_0_psi = self.all_losses(X_r, X_data, u_data)

        # Losses
        loss_hydro = 1e2*phi_r1 + 1e4*phi_r2 + 1e6*phi_0_omega + 1e8*phi_0_phi
        loss_magneto = 1e2*phi_r3 + 1e4*phi_r4  + 1e7*phi_0_J + 1e7*phi_0_psi
        
        return loss_hydro, loss_magneto
    
    def get_grad(self, batch, X_data, u_data):
        with tf.GradientTape(persistent=True) as tape:
            # This tape is for derivatives with
            # respect to trainable variables
            tape.watch(self.model_hydro.trainable_variables)
            tape.watch(self.model_magneto.trainable_variables)
            
            loss_hydro, loss_magneto = self.loss_fn(batch, X_data, u_data)
            
        g_hydro= tape.gradient(loss_hydro, self.model_hydro.trainable_variables)
        g_magneto = tape.gradient(loss_magneto, self.model_magneto.trainable_variables)
        del tape
        
        return loss_hydro, loss_magneto, g_hydro, g_magneto
    
    def solve(self, optimizer_hydro, optimizer_magneto, X_r, X_data, u_data):
        """This method performs a gradient descent type optimization."""
        
        #@tf.function
        def train_step(batch):
            loss_hydro, loss_magneto, g_hydro, g_magneto = self.get_grad(batch, X_data, u_data)
            
            # Perform gradient descent step
            optimizer_hydro.apply_gradients(zip(g_hydro, self.model_hydro.trainable_variables))
            optimizer_magneto.apply_gradients(zip(g_magneto, self.model_magneto.trainable_variables))

            return loss_hydro, loss_magneto
        
        num_epochs = 10_000
        for epoch in range(num_epochs):
                        
            for step in range(1):

                # tire les points 
                indices = np.random.choice(len(X_r), size=self.batch_size, replace=False)
                batch = tf.gather(X_r, indices)

                # descente de gradient
                loss_hydro, loss_magneto = train_step(batch)
            self.callback(X_r, X_data, u_data)
            
        #self.model.set_weights(self.best_weights)
        #self.model.save_weights(WEIGHT_PATH+'best_weight_MHD.h5')
        
    def callback(self, X_r, X_data, u_data):
        
        if self.epoch % 100 == 0:
            self.phi_r1, self.phi_r2, self.phi_r3, self.phi_r4, self.phi_0_omega, self.phi_0_phi, self.phi_0_J, self.phi_0_psi = self.all_losses(X_r[:1_000], X_data, u_data)
            self.current_loss = self.phi_r1 + self.phi_r2 + self.phi_r3 + self.phi_r4 + self.phi_0_omega + self.phi_0_phi + self.phi_0_J + self.phi_0_psi

            clear_output(wait=True)
            print('''Epoch {:06_}: 

                    phi_r1       {:6.4e}  
                    phi_r2       {:6.4e}
                    phi_r3       {:6.4e}
                    phi_r4       {:6.4e}
                    phi_0_omega  {:6.4e}
                    phi_0_phi    {:6.4e}
                    phi_0_J      {:6.4e}
                    phi_0_psi    {:6.4e}
                    total_loss   {:6.4e}\n'''.format(self.epoch, self.phi_r1, self.phi_r2, self.phi_r3, self.phi_r4, self.phi_0_omega, self.phi_0_phi, self.phi_0_J, self.phi_0_psi, self.current_loss), end="")
            
            self.hist_r1.append(self.phi_r1)
            self.hist_r2.append(self.phi_r2)
            self.hist_r3.append(self.phi_r3)
            self.hist_r4.append(self.phi_r4)
            self.hist_phi_0_omega.append(self.phi_0_omega)
            self.hist_phi_0_phi.append(self.phi_0_phi)
            self.hist_phi_0_J.append(self.phi_0_J)
            self.hist_phi_0_psi.append(self.phi_0_psi)
            
            if self.epoch == 15_000:
                self.data_weights = tf.constant([1., 1., 1., 1., 1, 1, 1, 0., 0., 0.])
                
            if self.epoch == 25_000:
                self.data_weights = tf.constant([1., 1., 1., 1., 1., 1, 1., 1, 1, 1])

        self.epoch+=1

    def plot_loss_history(self, ax=None):

        if not ax:
            fig = plt.figure(figsize=(10,6))
            ax = fig.add_subplot(111)

        ax.semilogy(range(len(self.hist_r1)),  self.hist_r1, '-', label='r_1')
        ax.semilogy(range(len(self.hist_r2)),  self.hist_r2, '-', label='r_2')
        ax.semilogy(range(len(self.hist_r3)),  self.hist_r3, '-', label='r_3')
        ax.semilogy(range(len(self.hist_r4)),  self.hist_r4, '-', label='r_4')
        ax.semilogy(range(len(self.hist_phi_0_omega)),  self.hist_phi_0_omega, '--', label='Omega_0')
        ax.semilogy(range(len(self.hist_phi_0_phi)), self.hist_phi_0_phi, '--', label='Phi_0')
        ax.semilogy(range(len(self.hist_phi_0_J)), self.hist_phi_0_J, '--', label='J_0')
        ax.semilogy(range(len(self.hist_phi_0_psi)), self.hist_phi_0_psi, '--', label='Psi_0')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.legend();

        return ax