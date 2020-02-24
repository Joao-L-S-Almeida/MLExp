import numpy as np
from keras.layers import Input, Dense, Dropout, Conv2D
from keras.models import Model
from keras.models import load_model
from keras import regularizers

class AutoEncoder:

    def __init__(self, layers_configuration, setup):

        self.layers_configuration = layers_configuration
        self.learning_rate = setup['learning_rate']
        self.l2_reg = setup['l2_reg']
        self.optimizer = setup['optimizer']
        self.loss_function = setup['loss_function']
        self.n_epochs = setup['n_epochs']
        self.model = None

    def construct(self, n_channels, n_rows, n_columns):

        input_tensor = Input(shape=(n_rows, n_columns, n_channels))

        # Encoder
        encoder_layers = self.layers_configuration.get('encoder')

        input_layer = input_tensor
        for layer_key, layer in encoder_layers.items():

            filters_layer = layer.get('filters')
            kernel_size_layer = layer.get('kernel_size')
            strides_layer = layer.get('strides')
            padding_layer = layer.get('padding')
            activation_layer = layer.get('activation')

            layer_op = Conv2D(filters_layer,
                                    kernel_size_layer,
                                    strides=strides_layer,
                                    padding=padding_layer,
                                    activation=activation_layer)

            layer_output = layer_op(input_layer)
            layer_input = layer_output

        output_tensor = layer_output
        print("Encoder constructed")

        model = Model(inputs=input_tensor, outputs=output_tensor)
        model.compile(optimizer=self.optimizer,
                      loss=self.loss_function)

        return model

    def fit(self, input_data, output_data, model=None):

        n_rows, n_columns, n_channels = input_data.shape[1:]

        if not model:
            model = self.construct(n_channels, n_rows, n_columns)

        model.fit(input_data, output_data,
                  batch_size=input_data.shape[0],
                  epochs=self.n_epochs)

        self.model = model

    def save(self, path):

        self.model.save(path)

    def load(self, path):

        self.model = load_model(path)
