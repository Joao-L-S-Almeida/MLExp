import numpy as np 
import os
from argparse import ArgumentParser
import tables 
import glob
import pyvista

# Script for reading multiple VTK files and write a HDF5 file. 
def name_index(input_string):
    
    strings = input_string.split("_")
    index = strings[-1]
    index = int(index) 
   
    return index

def extract_data_from_name(vti_file):

    sub_names = vti_file.split("\\")
    file_name = sub_names[-1]
    file_indices = file_name.split('_')[-1]
    file_indices = file_indices[:-4]
    coordinates = file_indices.split('i')[1:]
    iteration = int(coordinates[0][1:])
    partition = int(coordinates[1][1:])

    return iteration, partition

parser = ArgumentParser(description='Argument parsers')
parser.add_argument('--path', type=str)
parser.add_argument('--case', type=str)

args = parser.parse_args()

path = args.path
case = args.case

data_directories = glob.glob(path+"\\"+case+"*.vti")

variables = ['physPressure', 'physVelocity', 'physTemperature']
n_variables = len(variables) + 1

number_of_snapshots = len(data_directories)
number_of_partitions = 7

# Reading the information about the number of points
hdf5_path = path + 'rayleighBernard.hdf5'

# Creating the HDF5 file
h5f = tables.open_file(hdf5_path, mode='w')
#ds = h5f.create_array(h5f.root, 'AllData', np.empty((number_of_snapshots, n_points, n_variables)))

solution_dict = {partition: list() for partition in range(number_of_partitions)}
points_dict = dict()

for ss, vti_file in enumerate(data_directories):

    data = pyvista.read(vti_file)
    points = data.points
    n_points = points.shape[0]
    data_dimensions = data.GetDimensions()[:-1]

    # Recovering important information from the file name
    iteration, partition = extract_data_from_name(vti_file)

    points_dict[partition] = points.reshape(data_dimensions+(points.shape[-1],))

    print("File {} read".format(vti_file))

    data = pyvista.read(vti_file)

    print("File {} read".format(vti_file))

    variables_list = list() 

    variables_dict = data.point_arrays

    spacing = data.GetSpacing()

    for name in variables:
        
        array = variables_dict[name]

        if len(array.shape) == 1:
            array = array[:, None]

        array = array.reshape(data_dimensions + (array.shape[-1],))

        variables_list.append(array)
        print("Variable {} read".format(name))

    variables_array = np.dstack(variables_list)
    solution_dict[partition].append(variables_array)


    #ds[ss, :, :] = variables_array

x_list = list()
y_list = list()

for partition in range(number_of_partitions):

    list_of_arrays = solution_dict[partition]
    solution_dict[partition] = np.stack(list_of_arrays, 0)

    points_array = points_dict[partition]

    number_of_snapshots = solution_dict[partition].shape[0]

    x_max = points_array[:, :, 0].max()
    x_min = points_array[:, :, 0].min()

    y_max = points_array[:, :, 1].max()
    y_min = points_array[:, :, 1].min()

    x_list.append(x_max)
    x_list.append(x_min)

    y_list.append(y_max)
    y_list.append(y_min)

    y_list.append(y_max)

x_max = np.array(x_list).max()
x_min = np.array(x_list).min()

y_max = np.array(y_list).max()
y_min = np.array(y_list).min()

x_dim = int((x_max - x_min)/spacing[0])+1
y_dim = int((y_max - y_min)/spacing[1])+1

# TODO Now we need to reconstruct the original domain
# This allocation method should be replaced by a more convenient one
# such as HDF5 allocation in disk
global_x_array = np.linspace(x_min, x_max, x_dim)
global_y_array = np.linspace(y_min, y_max, y_dim)
global_z_array = np.zeros(1)

global_points_array = np.meshgrid(global_x_array, global_y_array, global_z_array)
global_points_array = np.dstack(global_points_array)

global_indices_array = ((global_points_array - np.tile(np.array([x_min, y_min, 0]), (y_dim, x_dim, 1)))\
                        /np.array(spacing)).astype(int)

global_solution_array = np.zeros((number_of_snapshots, y_dim, x_dim, n_variables))


for partition in range(number_of_partitions):

    points_array = points_dict[partition]
    local_x_dim, local_y_dim = points_array.shape[:-1]
    indices_array = ((points_array - np.tile(np.array([x_min, y_min, 0]), (local_x_dim, local_y_dim, 1)))\
                        /np.array(spacing)).astype(int)
    solution_array = solution_dict[partition]

    print("Connecting the multiple partitions.")

# Spatially connecting the lattice points
h5f.close()  

print("Input arguments read")
