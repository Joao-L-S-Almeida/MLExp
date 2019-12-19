import numpy as np
import tensorflow as tf
import os
from core.losses import loss_switcher

class DenseNetwork:

    def __init__(self, setup):

        self.layers_cells_list = setup['layers_cells_list']
        self.dropouts_rates_list = setup['dropouts_rates_list']
        self.learning_rate = setup['learning_rate']
        self.l2_reg = setup['l2_reg']
        self.activation_function = setup['activation_function']
        self.optimizer = setup['optimizer']
        self.loss_function = setup['loss_function']
        self.n_epochs = setup['n_epochs']
        self.outputpath = setup['outputpath']
        self.model_name = setup['model_name']

        self.model = None

        self.weights, self.biases = self.initialize_neural_net(self.layers_cells_list)

    def construct(self, input_dim, output_dim):

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.saver = tf.train.Saver()

        self.input_data_ph = tf.placeholder(tf.float32, shape=[None, input_dim])
        self.output_data_ph = tf.placeholder(tf.float32, shape=[None, output_dim])
        self.output_data_pred = self.network(self.input_data_ph, self.weights, self.biases)


        self.loss = loss_switcher(self.loss_function)(self.output_data_ph,
                                                      self.output_data_pred,
                                                      regularization_penalty=self.l2_reg,
                                                      weights=self.weights)

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                                beta1=0.9,
                                                                beta2=0.999,
                                                                epsilon=1e-08)

        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        init = tf.global_variables_initializer()

        self.sess.run(init)

    # Based on https://github.com/maziarraissi/PINNs/blob/master/main/continuous_time_identification%20(Navier-Stokes)/NavierStokes.py
    def initialize_neural_net(self, layers):

        weights = list()
        biases = list()
        num_layers = len(layers)

        for l in range(0, num_layers - 1):

            W = self.xavier_init(size=[layers[l], layers[l + 1]], index=l)
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32),
                            dtype=tf.float32,
                            name='biases_{}'.format(l))
            weights.append(W)
            biases.append(b)

        return weights, biases

    # Based on https://github.com/maziarraissi/PINNs/blob/master/main/continuous_time_identification%20(Navier-Stokes)/NavierStokes.py
    def xavier_init(self, size, index):

        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))

        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim],
                                               stddev=xavier_stddev),
                                               dtype=tf.float32,
                                               name='weights_{}'.format(index))

    def network(self, input_data, weights, biases):

        H = input_data

        for ll, layer in enumerate(self.layers_cells_list[:-2]):

            W = weights[ll]
            b = biases[ll]
            H = tf.nn.elu(tf.add(tf.matmul(H, W), b))

        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)

        return Y

    def callback(self, loss):

        print('Loss: %.3e' % loss)

    def fit(self, input_data, output_data):

        input_dim = input_data.shape[1]
        output_dim = output_data.shape[1]

        self.construct(input_dim, output_dim)

        var_map = {self.input_data_ph: input_data, self.output_data_ph: output_data}

        for it in range(self.n_epochs):

            self.sess.run(self.train_op_Adam, var_map)

            if it % 10 == 0:

                loss_value = self.sess.run(self.loss, var_map)
                print('It: %d, Loss: %.3e' % (it, loss_value))

        self.optimizer.minimize(self.sess,
                                feed_dict=var_map,
                                fetches=[self.loss],
                                loss_callback=self.callback)

        savepath = self.outputpath+self.outputpath+'/'

        if not os.path.isdir(savepath):
            os.mkdir(savepath)

        self.saver.save(self.sess, savepath + self.model_name)

    def load(self):

        pass



