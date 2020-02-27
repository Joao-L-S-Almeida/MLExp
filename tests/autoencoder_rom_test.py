import numpy as np
from argparse import ArgumentParser
from core.rom import AutoEncoder

if __name__ == "__main__":

    parser = ArgumentParser(description="Reading input arguments")
    parser.add_argument('--data_path', type=str)

    args = parser.parse_args()

    data_path = args.data_path

    # Loading the training and testing data
    # It considers that the input format is Numpy.
    # TODO Generalize it for other formats, such as HDF5

    data = np.load(data_path)

    print("Input data loaded.")

    batch_size = data.shape[0]

    training_data = data[:batch_size, :, :, :]
    testing_data = data[batch_size:, :, :, :]

    # The number of channels is equivalent to the number of variables
    n_channels = data.shape[3]
    n_rows = data.shape[1]
    n_columns = data.shape[2]

    layers_configuration = {
                            'encoder': {
                                        'conv1': {
                                                    'filters': 2*n_channels,
                                                    'kernel_size': (10, 20),
                                                    'strides': (5, 5),
                                                    'padding': "valid",
                                                    'activation': "relu"
                                                },
                                        'conv2': {
                                                    'filters': 4 * n_channels,
                                                    'kernel_size': (5, 10),
                                                    'strides': (3, 5),
                                                    'padding': "valid",
                                                    'activation': 'relu'
                                                },
                                        'conv3': {
                                            'filters': 8 * n_channels,
                                            'kernel_size': (3, 3),
                                            'strides': (3, 4),
                                            'padding': "valid",
                                            'activation': 'relu'
                                        }
                                },
                                'decoder': {
                                    'conv1': {
                                        'filters': 8 * n_channels,
                                        'kernel_size': (5, 5),
                                        'strides': (15, 16),
                                        'padding': "valid",
                                        'activation': "relu"
                                    },
                                    'conv2': {
                                        'filters': 4 * n_channels,
                                        'kernel_size': (7, 7),
                                        'strides': (7, 13),
                                        'padding': "same",
                                        'activation': 'relu'
                                    },
                                    'conv3': {
                                        'filters': n_channels,
                                        'kernel_size': (2, 2),
                                        'strides': (1, 1),
                                        'padding': "same",
                                        'activation': 'relu'
                                    }
                                }
                            }

    setup = {
                'learning_rate': 1e-5,
                'optimizer': 'adam',
                'loss_function': 'mse',
                'l2_reg': 1e-05,
                'n_epochs' : 1000
            }

    autoencoder_rom = AutoEncoder(layers_configuration, setup)

    autoencoder_rom.fit(training_data, training_data)