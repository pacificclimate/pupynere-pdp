import math
import unittest
import os
import tempfile
import numpy as np
from pupynere import netcdf_variable, netcdf_file, nc_generator
from numpy.testing import assert_almost_equal
import pytest
from itertools import chain

# this file tests individual functions on the netcdf_variable class

dimension_lengths = {"one": 1, 
                     "two": 2, 
                     "three": 3, 
                     "twelve": 12, 
                     "thousand": 1000,
                     "unlimited": 0}

attributes = {"test_attribute": "test value",
              "other_attribute": "other_value",
              "third_attribute": "third_value"}

# This function tests the multiple ways pupynere can create a netCDF files.
# It accepts variables, dimensions, and global attributes, along with a test
# function that should contain one or more assert() checks.

# First it creates a netCDF file on the disk with the received arguments,
# then passes the file to the function.

# Secondly it closes and reopens the file, and passes it to the test function 
# again. Then it deletes the file.

# Thirdly, it creates a netCDF in memory, no file, and passes it to the test
# function.

# Finally, it streams the netCDF, directs the stream into a file, and 
# opens and tests that.
def multiformat_netcdf_testing(dimensions, variables, attributes, test_assertions):
    
    # First, create a standard file on the hard drive
    print("testing file writes") 
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
    print("testing file reads")
    nc2 = netcdf_file(filename, 'r')
    test_assertions(nc2)
    nc2.close()
    
    print(filename)
    os.remove(filename)
    
    #now create a memory-only netCDF
    print("testing memory-only netcdf")
    nc3 = netcdf_file(None)
    for d in dimensions:
        nc3.createDimension(d, dimension_lengths[d])
    for v in variables:
        var = variables[v]
        nc3.createVariable(v, var.dtype, var.dimensions)
        nc3.variables[v].data = var.data
    
    test_assertions(nc3)
    
    #stream the memory-only netCDf to a file, and open and test that
    print("testing streaming memory-only netCDF to a file")
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
    

@pytest.mark.parametrize('dimensions',[
                         ('one',),
                         ('one', 'two'),
                         ('three', 'twelve'),
                         ('thousand', 'one'),
                         ('thousand', 'two'),
                         ('three', 'thousand'),
                         ('unlimited',),
                         ('unlimited', 'one'),
                         ('unlimited', 'twelve', 'two')])
def test_variable_shape(dimensions):
    shape = tuple(map(lambda d: dimension_lengths[d], dimensions))
    
    data = np.ones(shape)
    var = netcdf_variable(data, 'f8', shape, dimensions)

    
    def data_shape_asserts(nc):
        assert(nc.variables["testvar"].dimensions == dimensions)
        assert(np.prod(nc.variables["testvar"].shape) == np.prod(shape))
        
    
    multiformat_netcdf_testing(dimensions, {"testvar": var}, {}, data_shape_asserts)