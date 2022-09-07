import tensorflow as tf
import numpy as np

tf.get_logger().setLevel('ERROR')
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

# Path to data and saved weights
WEIGHT_PATH = '../saved_models/'
DATA_PATH = '../data/'
FIG_PATH = '../figures/'

# Set data type
DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)

pi = tf.constant(np.pi, dtype=DTYPE)
viscosity = tf.constant(3e-3, dtype=DTYPE)

# Define initial condition
def fun_u_0(x):
    return -tf.sin(pi * x)

# Define boundary condition
def fun_u_b(t, x):
    n = x.shape[0]
    return tf.zeros((n,1), dtype=DTYPE)