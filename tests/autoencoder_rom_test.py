import numpy as np
from argparse import ArgumentParser

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

    
