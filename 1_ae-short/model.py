import tensorflow as tf

from config import wave_len
from config import window_size
from config import data_dim
from config import latent_dim
from config import batch_size

def res_block(inputs, filters, kernel_size, strides=(1, 1), activation=tf.nn.relu, kernel_initializer=None):
    x = inputs

    x = tf.layers.conv2d(x, filters, kernel_size=kernel_size, strides=strides, padding="same", kernel_initializer=kernel_initializer)
    x = tf.layers.batch_normalization(x, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=kernel_initializer)
    x = activation(x)

    x = tf.layers.conv2d(x, filters, kernel_size=kernel_size, strides=strides, padding="same", kernel_initializer=kernel_initializer)
    x = tf.layers.batch_normalization(x, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=kernel_initializer)
    x = activation(x + inputs)

    return x

def leaky_relu(x, alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def encoder(inputs):
    h = 128#64
    initializer = tf.random_normal_initializer(0, 0.02)

    x = inputs

    # -------
    x = tf.layers.conv2d(x, h, kernel_size=(window_size, 1), strides=(1, 1), padding="same", kernel_initializer=initializer) # 2->h
    x = res_block(x, h, kernel_size=(window_size, 1), kernel_initializer=initializer)
    x = tf.layers.average_pooling2d(x, (2, 1), (2, 1), padding='same') # 160*1*h -> 80*1*h
    
    # -------
    x = res_block(x, h, kernel_size=(window_size, 1), kernel_initializer=initializer)
    x = tf.layers.average_pooling2d(x, (2, 1), (2, 1), padding='same') # 80*1*h -> 40*1*h

    # -------
    x = res_block(x, h, kernel_size=(window_size, 1), kernel_initializer=initializer)
    x = tf.layers.average_pooling2d(x, (2, 1), (2, 1), padding='same') # 40*1*h -> 20*1*h

    # -------
    x = res_block(x, h, kernel_size=(window_size, 1), kernel_initializer=initializer)

    # -------
    x = tf.reshape(x, shape=(-1, h * 20)) # 1280
    x = tf.layers.dense(x, latent_dim, kernel_initializer=initializer) # 64
    
    x = tf.nn.tanh(x)
    return x
    
#     epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], latent_dim])) 
#     z_mean = tf.layers.dense(x, latent_dim, kernel_initializer=initializer)
#     z_log_var = tf.layers.dense(x, latent_dim, kernel_initializer=initializer)
#     z = z_mean + tf.multiply(epsilon, tf.exp(0.5*z_log_var))

#     return z, z_mean, z_log_var


def decoder(inputs):
    h = 128#64
    initializer = tf.random_normal_initializer(0, 0.02)
    
    x = inputs # 160
    
    # -------
    x = tf.layers.dense(x, h * 20, kernel_initializer=initializer) # 64*20 = 1280
    x = tf.reshape(x, shape=(-1, 20, 1, h)) # 20*1*h

    # -------
    x = res_block(x, h, kernel_size=(window_size, 1), kernel_initializer=initializer)
    x = tf.image.resize_nearest_neighbor(x, (x.shape[1] * 2, x.shape[2])) # 40*1*h

    # -------
    x = res_block(x, h, kernel_size=(window_size, 1), kernel_initializer=initializer)
    x = tf.image.resize_nearest_neighbor(x, (x.shape[1] * 2, x.shape[2])) # 80*1*h

    # -------
    x = res_block(x, h, kernel_size=(window_size, 1), kernel_initializer=initializer)
    x = tf.image.resize_nearest_neighbor(x, (x.shape[1] * 2, x.shape[2])) # 160*1*h

    # -------
    x = res_block(x, h, kernel_size=(window_size, 1), kernel_initializer=initializer)
    x = tf.layers.conv2d(x, data_dim, kernel_size=1, padding="same", kernel_initializer=initializer)

    return x