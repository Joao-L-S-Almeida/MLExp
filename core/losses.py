import tensorflow as tf

def mse(output_data_ph, output_data_pred, regularization_penalty=None, weights=None):

    weights_square_sum = sum([tf.reduce_mean(tf.square(weight)) for weight in weights])

    return (
            tf.reduce_mean(tf.square(output_data_ph - output_data_pred))
            + regularization_penalty*weights_square_sum
            )

def mse_normed(output_data_ph, output_data_pred, regularization_penalty=None, weights=None):

    weights_square_sum = sum([tf.reduce_mean(tf.square(weight)) for weight in weights])
    max_value = tf.reduce_max(tf.abs(output_data_ph))

    return (
            tf.reduce_mean(tf.square((output_data_ph - output_data_pred)/max_value))
            + regularization_penalty*weights_square_sum
            )

def loss_switcher(case):

    losses = {'mse': mse, 'mse_normed': mse_normed}

    return losses.get(case)

