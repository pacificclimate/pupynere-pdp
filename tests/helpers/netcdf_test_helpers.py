# This file contains utility functions to make writing tests easier.
# It contains no actual tests.

import tempfile
from pupynere import netcdf_variable, netcdf_file, nc_generator
import os
import numpy as np

# This function tests the multiple ways pupynere can create a netCDF files.
# It accepts variables, dimensions, and global attributes, along with a test
# function that makes one or more assert() checks against a netcdf_file object
# created from those arguments. Multiple files are generated to check 
# file creation, reading, and streaming:
#   1. a netCDF file created on the disk
#   2. the above file is closed, opened fresh, and rechecked
#   3. a netCDF file created in memory only
#   4. the memory-only netCDF *streamed* to a file on disk, opened and and retested
# 3 and 4 can be skipped by passing in True as the last argument - this is
# to avoid a known bug.
def multiformat_netcdf_testing(dimensions, variables, attributes, test_assertions, test_stream = True):
    
    # First, create a standard file on the hard drive
    print("Testing standard file creation on the hard drive") 
    filename = tempfile.mktemp(".nc")
    nc = netcdf_file(filename, 'w')
    for d in dimensions:
        nc.createDimension(d, dimension_lengths[d])
    for v in variables:
        var = variables[v]
        nc.createVariable(v, var.dtype, var.dimensions)
        nc.variables[v].data = var.data
    
    test_assertions(nc)
    nc.close()
    
    print(filename)

    
    # reopen the file and test it again
    print("Testing reading the file written to the hard drive")
    nc2 = netcdf_file(filename, 'r')
    test_assertions(nc2)
    nc2.close()
    
    print(filename)
    os.remove(filename)
    
    if test_stream:
        #now create a memory-only netCDF
        print("Creating a memory-resident netCDF")
        nc3 = netcdf_file(None)
        for d in dimensions:
            nc3.createDimension(d, dimension_lengths[d])
        for v in variables:
            var = variables[v]
            nc3.createVariable(v, var.dtype, var.dimensions)
            nc3.variables[v].data = var.data
    
        test_assertions(nc3)
    
        #stream the memory-only netCDf to a file, and open and test that
        print("Testing streaming the memory-resident netCDF to a file")
        def variable_data_generator():
            for v in variables:
                print("V = {}".format(v))
                yield variables[v].data
    
        streamer = nc_generator(nc3, variable_data_generator())
        filename2 = tempfile.mktemp(".nc")
        writer = open(filename2, 'w')
        for chunk in streamer:
            writer.write(chunk)
        writer.close()
    
        nc4 = netcdf_file(filename2)
        test_assertions(nc4)
        nc4.close()
    
        os.remove(filename2)
    else:
        print("Skipping memory only and stream tests")

#This generator returns as many strings of the form variableN as are requested
def variable_namer():
    n = 0
    while True:
        yield "variable{}".format(n)
        n = n + 1
    
#This generator returns tuples of the form ("attributeN", "This is attribute N on variable V")
def attribute_creator(var=None):
    n = 0
    while True:
        if var:
            yield("attribute{}".format(n), "This is attribute {} on variable {}".format(n, var))
        else:
            yield("attribute{}".format(n), "This is global attribute {}".format(n))
        n = n + 1
    
#test dimensions
dimension_lengths = {"one": 1, 
                     "two": 2, 
                     "three": 3, 
                     "twelve": 12, 
                     "thousand": 1000,
                     "unlimited": 0}

# This is equivalent to numpy.arrange, which isn't present in the older version of numoy
# used in this respository.
def arrange(n):
    arr = np.empty(n)
    for i in range(n):
        arr[i] = i
    return arr
    
