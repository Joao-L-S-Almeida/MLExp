import tensorflow as tf

def mse(output_data_ph, output_data_pred):
    return tf.reduce_sum(tf.square(output_data_ph - output_data_pred))

def loss_switcher(case):

    losses = {'mse': mse}

    return losses.get(case)

