import numpy as np 
import os
from argparse import ArgumentParser 
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import tables 
import glob
import pyvista

# Script for reading multiple VTK files and write a HDF5 file. 
def name_index(input_string):
    
    strings = input_string.split("_")
    index = strings[-1]
    index = int(index) 
   
    return index

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
    sub_names = vti_file.split("\\")
    file_name = sub_names[-1]
    file_indices = file_name.split('_')[-1]
    file_indices = file_indices[:-4]
    coordinates = file_indices.split('i')[1:]
    iteration = int(coordinates[0][1:])
    partition = int(coordinates[1][1:])

    points_dict[partition] = {'points': points.reshape(data_dimensions+(points.shape[-1],))}

    print("File {} read".format(vti_file))

    data = pyvista.read(vti_file)

    print("File {} read".format(vti_file))

    variables_list = list() 

    variables_dict = data.point_arrays

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

# Spatially connecting the lattice points
h5f.close()  

print("Input arguments read")
