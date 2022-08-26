import tensorflow as tf
import numpy as np
import requests

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

# Set boundary
tmin = 0.
tmax = 1.
xmin = -1.
xmax = 1.

numin = 1e-4
numax = 1e-2

# Define initial condition
def fun_u_0(x):
    return -tf.sin(pi * x)

# Define boundary condition
def fun_u_b(t, x):
    n = x.shape[0]
    return tf.zeros((n,1), dtype=DTYPE)

def alert(message):
    bot_token = '5169497350:AAHiAyn6D8q9QDoRzTC8HnQU8kYJ-M6xoDg'
    bot_chatID = '1061558745'
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text='
    send_text += 'Les calculs sont finito pipo : ' + message
    requests.get(send_text)