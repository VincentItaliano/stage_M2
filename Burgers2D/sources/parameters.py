import tensorflow as tf

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

viscosity = tf.constant(1e-3, dtype=DTYPE)

# Define initial condition
def fun_u_0(x, y):
    return 1.5-2.0*x

# Define boundary condition
def fun_u_b(t, x, y):
    return 1.5-2.0*x