import tensorflow as tf
import requests

# Path to data and saved weights
WEIGHT_PATH = '../saved_models/'
DATA_PATH = '../data/'
FIG_PATH = '../figures/'

# Set data type
DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)

viscosity = tf.constant(1e-3, dtype=DTYPE)

# Set boundary
tmin = 0.
tmax = 1.5
xmin = 0.
xmax = 1.
ymin = 0. 
ymax = 1.

# Define initial condition
def fun_u_0(x, y):
    return 1.5-2.0*x

# Define boundary condition
def fun_u_b(t, x, y):
    return 1.5-2.0*x

def alert(message):
    bot_token = '5169497350:AAHiAyn6D8q9QDoRzTC8HnQU8kYJ-M6xoDg'
    bot_chatID = '1061558745'
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text='
    send_text += 'Les calculs sont finito pipo : ' + message
    requests.get(send_text)