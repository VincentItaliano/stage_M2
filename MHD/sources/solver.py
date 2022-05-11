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
        loss = 1e3*phi_r1 + 1e1*phi_r2 + 1e3*phi_r3 + 1e2*phi_r4 + 1e3*phi_0 + 1e2*phi_b
        
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
        
        num_epochs = 30_000
        for epoch in range(num_epochs):
                        
            for step in range(1):

                # tire les points 
                indices = np.random.choice(len(X_r), size=self.batch_size, replace=False)
                batch = tf.gather(X_r, indices)

                # descente de gradient
                loss = train_step(batch)

            self.phi_r1, self.phi_r2, self.phi_r3, self.phi_r4, self.phi_0, self.phi_b = self.all_losses(X_r)
            self.current_loss = self.phi_r1 + self.phi_r2 + self.phi_r3 + self.phi_r4 + self.phi_0 + self.phi_b
            self.callback()
            
        self.model.load_weights(WEIGHT_PATH + 'best_weight_MHD.h5')
        
    def solve_with_ScipyOptimizer(self, X_r, X_data, u_data, method='L-BFGS-B'):
        
        def get_weight_tensor():
            """Function to return current variables of the model
            as 1d tensor as well as corresponding shapes as lists."""
            
            weight_list = []
            shape_list = []
            
            # Loop over all variables, i.e. weight matrices, bias vectors and unknown parameters
            for v in self.model.variables:
                shape_list.append(v.shape)
                weight_list.extend(v.numpy().flatten())
                
            weight_list = tf.convert_to_tensor(weight_list)
            return weight_list, shape_list

        x0, shape_list = get_weight_tensor()
        
        def set_weight_tensor(weight_list):
            """Function which sets list of weights
            to variables in the model."""
            idx = 0
            for v in self.model.variables:
                vs = v.shape
                
                # Weight matrices
                if len(vs) == 2:  
                    sw = vs[0]*vs[1]
                    new_val = tf.reshape(weight_list[idx:idx+sw],(vs[0],vs[1]))
                    idx += sw
                
                # Bias vectors
                elif len(vs) == 1:
                    new_val = weight_list[idx:idx+vs[0]]
                    idx += vs[0]
                    
                # Variables (in case of parameter identification setting)
                elif len(vs) == 0:
                    new_val = weight_list[idx]
                    idx += 1
                    
                # Assign variables (Casting necessary since scipy requires float64 type)
                v.assign(tf.cast(new_val, DTYPE))
        
        def get_loss_and_grad(w):
            """Function that provides current loss and gradient
            w.r.t the trainable variables as vector. This is mandatory
            for the LBFGS minimizer from scipy."""
            
            # Update weights in model
            set_weight_tensor(w)
            # Determine value of \phi and gradient w.r.t. \theta at w
            loss, grad = self.get_grad(X_r, X_data, u_data)
            
            # Store current loss for callback function            
            loss = loss.numpy().astype(np.float64)
            self.current_loss = loss            
            
            # Flatten gradient
            grad_flat = []
            for g in grad:
                grad_flat.extend(g.numpy().flatten())
            
            # Gradient list to array
            grad_flat = np.array(grad_flat,dtype=np.float64)
            
            # Return value and gradient of \phi as tuple
            return loss, grad_flat
        
        
        return scipy.optimize.minimize(fun=get_loss_and_grad,
                                       x0=x0,
                                       jac=True,
                                       method=method,
                                       callback=self.callback,
                                       options={'maxiter': 3000,
                                                'maxfun': 50000,
                                                'maxcor': 50,
                                                'maxls': 50,
                                                'ftol': 1.0*np.finfo(float).eps})

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