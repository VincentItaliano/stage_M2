import tensorflow as tf
import numpy as np
from scipy.special import j0, j1 #bessel functions
import requests

# Manage tensorflow memory
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

viscosity = tf.constant(3e-3, dtype=DTYPE)

# Set boundary
tmin = 0.
tmax = 6.01
xmin = -3.
xmax = 3.
ymin = -3. 
ymax = 3.

#Define initial condition
def fun_omega_0(x, y):
    n = x.shape[0]
    return np.zeros((n,1), dtype=np.float32)

def fun_phi_0(x, y):
    n = x.shape[0]
    return np.zeros((n,1), dtype=np.float32)

k = 3.831705970
alpha = -2./(k*j0(k))

def fun_J_0(x, y):
    r = np.sqrt(x**2 + y**2)
    J_0 = np.where(r <= 1., k**2*alpha*j1(k*r)*y/r, 0.)
    return J_0

def fun_psi_0(x,y):
    r = np.sqrt(x**2 + y**2)
    psi_0 = np.where(r <= 1., alpha*j1(k*r)*y/r, (1/r-r)*y/r)
    return psi_0

def alert(message):
    bot_token = '5169497350:AAHiAyn6D8q9QDoRzTC8HnQU8kYJ-M6xoDg'
    bot_chatID = '1061558745'
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text='
    send_text += 'Les calculs sont finito pipo : ' + message
    requests.get(send_text)