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

data_directories = glob.glob(path+'//'+case+"*.vti")
#data_directories = sorted(data_directories, key=name_index)

variables = ['physPressure', 'physVelocity', 'physTemperature']
n_variables = len(variables)+1

number_of_snapshots = len(data_directories)

# Reading the information about the number of cells 
directory = data_directories[1]

data = pyvista.read(directory)
points = data.points
n_points = points.shape[0]

hdf5_path = path + 'rayleighBernard.hdf5'

# Creating the HDF5 file
h5f = tables.open_file(hdf5_path, mode='w')
ds = h5f.create_array(h5f.root, 'AllData', np.empty((number_of_snapshots, n_points, n_variables)))

for ss, vti_file in enumerate(data_directories):

    print("File {} read".format(vti_file))

    data = pyvista.read(vti_file)

    print("File {} readed".format(vti_file))

    variables_list = list() 
    print(data.points.shape[0])
    variables_dict = data.point_arrays

    for name in variables:
        
        array = variables_dict[name]

        if len(array.shape) == 1: array = array[:,None]
        variables_list.append(array)
        print("Variable {} readed".format(name))

    variables_array = np.hstack(variables_list)
    #ds[ss, :, :] = variables_array

h5f.close()  

print("Input arguments read")
