import math
import os
import tempfile
import numpy as np
from pupynere import netcdf_variable, netcdf_file, nc_generator
from numpy.testing import assert_almost_equal
import pytest
from itertools import chain
import netcdf_test_helpers as helpers

# this file tests each individual functions on the netcdf_variable class

# a large gap in testing occurs around adding additional data to an unlimited 
# variable / dimensions.

@pytest.mark.parametrize('dimensions,num_atts', [
                        (('one',), 0),
                        (('one', 'two'), 1),
                        (('thousand', 'twelve'), 2),
                        (('unlimited',), 0),
                        (('unlimited', 'one'), 1),
                        (('unlimited', 'twelve', 'two'), 2)])
def test_netcdf_variable_init(dimensions, num_atts):
    
    shape = helpers.shape_from_dimensions(dimensions)
    data = np.ones(shape)
    
    vn = helpers.variable_namer()
    varname = vn.next()
    
    ac = helpers.attribute_creator()
    attributes = {}
    
    # add some attributes at variable creation
    for i in range(num_atts):
        attr = ac.next()
        print(attr)
        attributes[attr[0]] = attr[1]
    
    var = netcdf_variable(data, 'f8', shape, dimensions, attributes)
    
    def check_initial_values(nc):
        var = nc.variables[varname]
        assert(var.dimensions == dimensions)
        assert(np.prod(var.shape) == np.prod(shape))
        ac = helpers.attribute_creator()
        for i in range(num_atts):
            attr = ac.next()
            assert(getattr(var, attr[0]) == attr[1])   
    
    helpers.multiformat_netcdf_testing(dimensions, {varname: var}, attributes, check_initial_values)



def test_netcdf_variable_setattr():
    dimensions = ('one', 'two')
    shape = helpers.shape_from_dimensions(dimensions)
    data = np.ones(shape)
    
    vn = helpers.variable_namer()
    varname = vn.next()
    
    ac = helpers.attribute_creator()
    attributes = {}
    
    # add some attributes at variable creation
    for i in range(0, 2):
        attr = ac.next()
        attributes[attr[0]] = attr[1]
            
    var = netcdf_variable(data, 'f8', shape, dimensions, attributes)

        
    #add some attributes later
    for i in range(0, 2):
        attr = ac.next()
        var.__setattr__(attr[0], attr[1])
    
    ac = helpers.attribute_creator()
    for i in range(0, 4):
        attr = ac.next()
        assert(getattr(var, attr[0]) == attr[1])   
    #TODO: test with streaming, etc.


@pytest.mark.parametrize('dimensions,record',[
                         (('one',), False),
                         (('one', 'two'), False),
                         (('thousand', 'twelve'), False),
                         (('unlimited',), True),
                         (('unlimited', 'one'), True),
                         (('unlimited', 'twelve', 'two'), True)])    
def test_netcdf_variable_isrec(dimensions, record):
    var = helpers.default_variable_from_dimensions(dimensions)
    vn = helpers.variable_namer()
    varname = vn.next()
    
    def isrec_assert(nc):
        assert(nc.variables[varname].isrec == record)
        
    helpers.multiformat_netcdf_testing(dimensions, {varname: var}, {}, isrec_assert)

# this test doesn't work, but it's not clear why not
@pytest.mark.skip
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
def test_netcdf_variable_shape(dimensions):
    shape = helpers.shape_from_dimensions(dimensions)
    var = helpers.default_variable_from_dimensions(dimensions)
    vn = helpers.variable_namer()
    varname = vn.next()
    
    def data_shape_asserts(nc):
        assert(np.prod(nc.variables[varname].shape) == np.prod(shape))        
    
    helpers.multiformat_netcdf_testing(dimensions, {varname: var}, {}, data_shape_asserts)

    
def test_netcdf_variable_getvalue():
    assert(True)
    
def test_netcdf_variable_assignvalue():
    assert(True)

# there are multiple valid sets of types.
# newer set:  ('f4','f8','i2', 'i1','S1','i4')
# older set: ('f','d','h', 's','b','B','c','i','l')
# this function returns the older set, which we can compare to the new set
@pytest.mark.parametrize('type,code', [
    ('f4', 'f'),
    ('f8', 'd'),
    ('i2', 'h'),
    ('i1', 'b'),
    ('i4', 'i')])
def test_netcdf_variable_typecode(type, code):
    dimensions = ('three', 'twelve')
    
    shape = helpers.shape_from_dimensions(dimensions)
    
    data = np.ones(shape)
    var = netcdf_variable(data, type, shape, dimensions)

    vn = helpers.variable_namer()
    varname = vn.next()
    
    def typecode_asserts(nc):
        assert(nc.variables[varname].typecode() == code)        
    
    helpers.multiformat_netcdf_testing(dimensions, {varname: var}, {}, typecode_asserts)
    assert(True)

@pytest.mark.parametrize('type,size', [
    ('f4', 4),
    ('f8', 8),
    ('i2', 2),
    ('i1', 1),
    ('i4', 4),])
def test_netcdf_variable_itemsize(type, size):
    dimensions = ('three', 'twelve')
    shape = helpers.shape_from_dimensions(dimensions)
    
    data = np.ones(shape)
    var = netcdf_variable(data, type, shape, dimensions)

    vn = helpers.variable_namer()
    varname = vn.next()
    
    def typesize_asserts(nc):
        assert(nc.variables[varname].itemsize == size)        
    
    helpers.multiformat_netcdf_testing(dimensions, {varname: var}, {}, typesize_asserts)
    
@pytest.mark.parametrize('value', [0, 1, 30.7])
def test_netcdf_variable_getvalue(value):
    dimensions = ("one",)
    shape = (1,)
    data = np.full(shape, value)
    var = netcdf_variable(data, "f8", shape, dimensions)
    
    vn = helpers.variable_namer()
    varname = vn.next()
    
    def get_scalar_assert(nc):
        assert(nc.variables[varname].getValue() == value)
    
    helpers.multiformat_netcdf_testing(dimensions, {varname: var}, {}, get_scalar_assert)

@pytest.mark.xfail
@pytest.mark.parametrize('value', [0, -4, 30.7])
def test_netcdf_variable_assignvalue(value):
    var = default_variable_from_dimensions(("one",))    
    vn = helpers.variable_namer()
    varname = vn.next()
    
    def setitem_scalar_assert(nc):
        nc.variables[varname].assignValue(value)
        assert(nc.variables[varname].getValue() == value)
    
    #skip memory-only / streaming parts of test due to known bug
    helpers.multiformat_netcdf_testing(dimensions, {varname: var}, {}, setitem_scalar_assert, False)


@pytest.mark.parametrize('dimensions',[
                         ('one', 'two'),
                         ('three', 'twelve'),
                         ('thousand', 'one'),
                         ('thousand', 'two'),
                         ('three', 'thousand')])
def test_netcdf_variable_getitem(dimensions):
    shape = helpers.shape_from_dimensions(dimensions)
    data = helpers.arrange(helpers.data_from_dimensions(dimensions).size)
    var = netcdf_variable(data, 'f8', shape, dimensions)
    vn = helpers.variable_namer()
    varname = vn.next()
    
    def variable_value_asserts(nc):
        var = nc.variables[varname]
        for i in range(var.data.size):
            assert(var[i] == i)
    
    helpers.multiformat_netcdf_testing(dimensions, {varname: var}, {}, variable_value_asserts)    
    

@pytest.mark.parametrize('dimensions',[
                         ('one', 'two'),
                         ('three', 'twelve'),
                         ('thousand', 'one'),
                         ('thousand', 'two'),
                         ('three', 'thousand')])
def test_netcdf_variable_setitem(dimensions):
    shape = tuple(map(lambda d: helpers.dimension_lengths[d], dimensions))
        
    data = np.ones(shape)
    size = data.size
    data = helpers.arrange(size)
    var = netcdf_variable(data, 'f8', shape, dimensions)

    vn = helpers.variable_namer()
    varname = vn.next()
    
    var[0] = 10
    var[size-1] = 30
    if size > 2:
        var[size/2] = 4
    
    def variable_value_asserts(nc):
        var = nc.variables[varname]
        size = var.data.size
        assert(var[0] == 10)
        assert(var[size-1] == 30)
        if size > 2:
            assert(var[size /2] == 4)
    
    helpers.multiformat_netcdf_testing(dimensions, {varname: var}, {}, variable_value_asserts, True)    
