import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

from .parameters import *

class PINNSolver():
    def __init__(self, model, batch_size, weight_filename = 'best_weight_burgers.h5'):

        # Initialize model
        self.model = model
        
        # Initialize history of losses and global iteration counter
        self.hist = []
        self.epoch = 0
        self.batch_size = batch_size
        self.best_loss = 1.0
        self.count_since_best_loss = 0

        # file to store the best weights
        self.weight_filename = weight_filename
    
    def get_r(self, X):
        """ Compute pde residual """
        
        t, x = X[:,0:1], X[:,1:2]
        
        with tf.GradientTape(persistent=True) as tape:
            # Watch variables representing t,x and y during this GradientTape
            tape.watch(t)
            tape.watch(x)
        
            # Compute current values u(t,x,y)
            u = self.model(tf.stack([t[:,0], x[:,0]], axis=1))
            
            u_x = tape.gradient(u, x)
            
        u_t = tape.gradient(u, t)
        u_xx = tape.gradient(u_x, x)

        del tape
        
        return u_t + u*u_x  - viscosity*u_xx
    
    def all_losses(self, X_r, X_data, u_data):
        """ returns each terms in the loss """
        
        # Compute phi_r
        r = self.get_r(X_r)
        phi_r = tf.reduce_mean(tf.square(r))
        
        # Compute phi_0
        u_pred = self.model(X_data[0])
        phi_0 = tf.reduce_mean(tf.square(u_data[0] - u_pred))
        
        # Compute phi_b
        u_pred = self.model(X_data[1])
        phi_b = tf.reduce_mean(tf.square(u_data[1] - u_pred))
        
        return phi_r, phi_0, phi_b
    
    def loss_fn(self, X_r, X_data, u_data):
        """ Loss function """
        
        phi_r, phi_0, phi_b = self.all_losses(X_r, X_data, u_data)

        # Loss
        loss = 1e4*phi_r + 1e5*phi_0 + phi_b
        
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
    

    def solve_with_TFoptimizer(self, optimizer, X_r, X_data, u_data, alpha):
        """This method performs a gradient descent type optimization."""
        
        # alpha is the ration of uniformly sampled points (alpha!=1 to use preference sampling)
        self.nb_uniform  = int(alpha * self.batch_size)
        self.nb_residual = int((1-alpha) * self.batch_size)
        
        @tf.function
        def train_step(batch):
            loss, grad_theta = self.get_grad(batch, X_data, u_data)
            
            # Perform gradient descent step
            optimizer.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            return loss
        
        num_epochs = 20_000
        for epoch in range(num_epochs):
                        
            # compute q_i
            if alpha == 1.:
                J_rho = self.get_r(X_r) 
                sum_J_rho = tf.reduce_sum(J_rho**2)
                q = J_rho**2 / sum_J_rho
                q = q.numpy().flatten()
            else:
                q = np.ones(self.batch_size)/self.batch_size
        
            for step in range(1):

                # sample points
                indices_uniform  = np.random.choice(len(X_r), size=self.nb_uniform, replace=False)
                indices_residual = np.random.choice(len(X_r), size=self.nb_residual, replace=False, p=q)
                indices = np.concatenate((indices_uniform, indices_residual), axis=None)
                batch = tf.gather(X_r, indices)

                # gradient descent
                loss = train_step(batch)

            self.phi_r, self.phi_0, self.phi_b = self.all_losses(X_r, X_data, u_data)
            self.current_loss = self.phi_r + self.phi_0 + self.phi_b
            self.callback()
            
        # retrieve best weights at the end of the training
        self.model.load_weights(WEIGHT_PATH + self.weight_filename)
        
    def callback(self, xr=None):
        self.count_since_best_loss+=1
        
        if self.epoch % 100 == 0:
            print('\rEpoch {:06_}: phi_r  {:6.4e}    phi_0  {:6.4e}    phi_b  {:6.4e}    total_loss  {:6.4e}'.format(self.epoch, self.phi_r, self.phi_0, self.phi_b, self.current_loss), end="")
            
            if self.current_loss < self.best_loss:
                self.best_loss = self.current_loss
                self.count_since_best_loss = 0
                self.best_weights = self.model.get_weights()
                
                if self.epoch % 1_000 == 0:
                    self.model.save_weights(WEIGHT_PATH + self.weight_filename)
                
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


class PINNSolver2():
    def __init__(self, model, batch_size, weight_filename = 'best_weight_burgers.h5'):

        # Initialize model
        self.model = model
        
        # Initialize history of losses and global iteration counter
        self.hist = []
        self.epoch = 0
        self.batch_size = batch_size
        self.best_loss = 1.0
        self.count_since_best_loss = 0

        # file to store the best weights
        self.weight_filename = weight_filename
    
    def get_r(self, X):
        """ Compute pde residual """
        
        t, x, nu = X[:,0:1], X[:,1:2], X[:,2:3]
        
        with tf.GradientTape(persistent=True) as tape:
            # Watch variables representing t,x and y during this GradientTape
            tape.watch(t)
            tape.watch(x)
        
            # Compute current values u(t,x,y)
            u = self.model(tf.stack([t[:,0], x[:,0], nu[:,0]], axis=1))
            
            u_x = tape.gradient(u, x)
            
        u_t = tape.gradient(u, t)
        u_xx = tape.gradient(u_x, x)

        del tape
        
        return u_t + u*u_x  - nu*u_xx
    
    def all_losses(self, X_r, X_data, u_data):
        """ returns each terms in the loss """
        
        # Compute phi_r
        r = self.get_r(X_r)
        phi_r = tf.reduce_mean(tf.square(r))
        
        # Compute phi_0
        u_pred = self.model(X_data[0])
        phi_0 = tf.reduce_mean(tf.square(u_data[0] - u_pred))
        
        # Compute phi_b
        u_pred = self.model(X_data[1])
        phi_b = tf.reduce_mean(tf.square(u_data[1] - u_pred))
        
        return phi_r, phi_0, phi_b
    
    def loss_fn(self, X_r, X_data, u_data):
        """ Loss function """
        
        phi_r, phi_0, phi_b = self.all_losses(X_r, X_data, u_data)

        # Loss
        loss = 1e4*phi_r + 1e5*phi_0 + phi_b
        
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
    

    def solve_with_TFoptimizer(self, optimizer, X_r, X_data, u_data, alpha):
        """This method performs a gradient descent type optimization."""
        
        self.nb_uniform  = int(alpha * self.batch_size)
        self.nb_residual = int((1-alpha) * self.batch_size)
        
        @tf.function
        def train_step(batch):
            loss, grad_theta = self.get_grad(batch, X_data, u_data)
            
            # Perform gradient descent step
            optimizer.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            return loss
        
        num_epochs = 20_000
        for epoch in range(num_epochs):
                        
            # compute q_i 
            if alpha == 1.:
                J_rho = self.get_r(X_r) 
                sum_J_rho = tf.reduce_sum(J_rho**2)
                q = J_rho**2 / sum_J_rho
                q = q.numpy().flatten()
            else:
                q = np.ones(self.batch_size)/self.batch_size
        
            for step in range(1):

                # sample points 
                indices_uniform  = np.random.choice(len(X_r), size=self.nb_uniform, replace=False)
                indices_residual = np.random.choice(len(X_r), size=self.nb_residual, replace=False, p=q)
                indices = np.concatenate((indices_uniform, indices_residual), axis=None)
                batch = tf.gather(X_r, indices)

                # gradient descent
                loss = train_step(batch)

            self.phi_r, self.phi_0, self.phi_b = self.all_losses(X_r, X_data, u_data)
            self.current_loss = self.phi_r + self.phi_0 + self.phi_b
            self.callback()
            
        self.model.load_weights(WEIGHT_PATH + self.weight_filename)
        
    def callback(self, xr=None):
        self.count_since_best_loss+=1
        
        if self.epoch % 100 == 0:
            print('\rEpoch {:06_}: phi_r  {:6.4e}    phi_0  {:6.4e}    phi_b  {:6.4e}    total_loss  {:6.4e}'.format(self.epoch, self.phi_r, self.phi_0, self.phi_b, self.current_loss), end="")
            
            if self.current_loss < self.best_loss:
                self.best_loss = self.current_loss
                self.count_since_best_loss = 0
                self.best_weights = self.model.get_weights()
                
                if self.epoch % 1_000 == 0:
                    self.model.save_weights(WEIGHT_PATH + self.weight_filename)
                
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