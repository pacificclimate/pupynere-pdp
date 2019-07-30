import math
import os
import tempfile
import numpy as np
from pupynere import netcdf_variable, netcdf_file, nc_generator
from numpy.testing import assert_almost_equal
import pytest
from itertools import chain, combinations
import netcdf_test_helpers as helpers

# this file tests each individual functions on the netcdf_file class

# Unfortunately, some of the functions that read and write files are
# difficult to test independently. They're implicitly tested in aggregate
# as part of the end-to-end testing done by the multiformat_netcdf_testing()
# function, which creates and reads multiple files, but it would be even better
# to have individual tests, too.

# netcdf_file instance used to call some class functions that don't 
# use any file attributes
empty_file = netcdf_file(None)

@pytest.mark.skip
def test_netcdf_file_init():
    assert(True)

@pytest.mark.parametrize('num_attrs', [0, 1, 2, 10, 100])    
def test_netcdf_file_setattr(num_attrs):
    
    def global_attr_asserts(nc):
        ac = helpers.attribute_creator()
        for i in range(num_attrs):
            attr = ac.next()
            nc.__setattr__(attr[0], attr[1])
            
        atts = nc._attributes
        print(atts)
        for i in range(num_attrs):
            attname = "attribute{}".format(i)
            attvalue = "This is global attribute {}".format(i)
            assert(atts[attname]==attvalue)

    helpers.multiformat_netcdf_testing([], [], {}, global_attr_asserts)
    
def test_netcdf_file_close():
    assert(True)

#TODO - test unlimited dimensions.
@pytest.mark.parametrize("dimensions,global_atts,variable_atts,expected", [
                         (('one',), 4, 3, 460),
                         (('one', 'two'), 5, 1, 648),
                         (('three', 'twelve'), 0, 0, 608),
                         (('thousand', 'one'), 5, 5, 17244),
                         (('thousand', 'two'), 1, 100, 39864),
                         (('three', 'thousand'), 100, 4, 38048)])
def test_netcdf_file_filesize(dimensions, global_atts, variable_atts, expected):
    variables = {}
    global_attributes = {}
    vn = helpers.variable_namer()
    ac = helpers.attribute_creator()
    
    nc = netcdf_file(None)
    for d in dimensions:
        nc.createDimension(d, helpers.dimension_lengths[d])
    for i in range(global_atts):
        attr = ac.next()
        nc.__setattr__(attr[0], attr[1])

    # create a variable for each combination of dimensions
    for r in range(len(dimensions)):
        for dim_set in list(combinations(dimensions, r+1)):
            varname = vn.next()
            var = helpers.default_variable_from_dimensions(dim_set)
            var_attrs = {}
            # assign variable_atts attributes to each variable
            for i in range(variable_atts):
                attr = ac.next()
                var_attrs[attr[0]] = attr[1]
            nc.createVariable(varname, "f8", dim_set, var_attrs)
                
    assert (nc.filesize == expected)

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
def test_netcdf_file_createdimension(dimensions):
    def dimension_asserts(nc):
        for d in dimensions:
            assert d in nc.dimensions
            #TODO - check length?
    helpers.multiformat_netcdf_testing(dimensions, {}, {}, dimension_asserts)

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
def test_netcdf_file_createvariable(dimensions):
    variables = helpers.combinatoric_variables(dimensions)
    
    def variable_check_asserts(nc):
        for v in variables:
            assert(v in nc.variables)
            assert(nc.variables[v].dimensions == variables[v].dimensions)
    helpers.multiformat_netcdf_testing(dimensions, variables, {}, variable_check_asserts)

@pytest.mark.skip    
def test_netcdf_file_flush():
    assert(True)

@pytest.mark.parametrize('dimensions',[
                         ('three', 'twelve'),
                         ('thousand', 'one'),
                         ('thousand', 'two'),
                         ('three', 'thousand'),
                         ('unlimited',),
                         ('unlimited', 'one'),
                         ('unlimited', 'twelve', 'two')])
def test_netcdf_file_recvars(dimensions):
    variables = helpers.combinatoric_variables(dimensions)
    recvars = []
    for v in variables:
        if "unlimited" in variables[v].dimensions:
            recvars.append(v)
    def check_recvars(nc):
        for rec in recvars:
            assert(rec in nc.recvars)
        for rec in nc.recvars:
            assert(rec in recvars)
    helpers.multiformat_netcdf_testing(dimensions, variables, {}, check_recvars)

@pytest.mark.parametrize('dimensions',[
                         ('three', 'twelve'),
                         ('thousand', 'one'),
                         ('thousand', 'two'),
                         ('three', 'thousand'),
                         ('unlimited',),
                         ('unlimited', 'one'),
                         ('unlimited', 'twelve', 'two')])    
def test_netcdf_file_nonrecvars(dimensions):
    variables = helpers.combinatoric_variables(dimensions)
    nonrecvars = []
    for v in variables:
        if not "unlimited" in variables[v].dimensions:
            nonrecvars.append(v)
    def check_nonrecvars(nc):
        for nonrec in nonrecvars:
            assert(nonrec in nc.non_recvars)
        for nonrec in nc.non_recvars:
            assert(nonrec in nonrecvars)
    helpers.multiformat_netcdf_testing(dimensions, variables, {}, check_nonrecvars)

# tested by multiformat_netcdf_testing
@pytest.mark.skip
def test_netcdf_file_generate():
    assert(True)

@pytest.mark.parametrize("dimensions,global_atts,variable_atts,expected_key", [
                         (('one',), 0, 1, "test1"),
                         (('one',), 1, 0, "test2"),
                         (('one',), 0, 0, "test3"),
                         (('one',), 1, 1, "test4"),
                         (('one', 'two'), 0, 1, "test5"),
                         (('one', 'two'), 1, 0, "test6"),
                         (('one', 'two'), 0, 0, "test7"),
                         (('one', 'two'), 1, 1, "test8")])
def test_netcdf_file_header(dimensions, global_atts, variable_atts, expected_key):
    expected_results = {"test1": "CDF\x01\x00\x00\x00\x00\x00\x00\x00\n"
                        "\x00\x00\x00\x01\x00\x00\x00\x03one0\x00\x00"
                        "\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                        "\x00\x00\x0b\x00\x00\x00\x01\x00\x00\x00\tvar"
                        "iable0000\x00\x00\x00\x01\x00\x00\x00\x00\x00"
                        "\x00\x00\x0c\x00\x00\x00\x01\x00\x00\x00\n"
                        "attribute000\x00\x00\x00\x02\x00\x00\x00\x1aTh"
                        "is is global attribute 000\x00\x00\x00\x06\x08"
                        "\x00\x00\x00\x00\x00\x00\x00",
                        
                        "test2": "CDF\x01\x00\x00\x00\x00\x00\x00\x00\n"
                        "\x00\x00\x00\x01\x00\x00\x00\x03one0\x00\x00"
                        "\x00\x01\x00\x00\x00\x0c\x00\x00\x00\x01\x00"
                        "\x00\x00\n"
                        "attribute000\x00\x00\x00\x02\x00\x00\x00\x1a"
                        "This is global attribute 000\x00\x00\x00\x0b"
                        "\x00\x00\x00\x01\x00\x00\x00\tvariable0000"
                        "\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00"
                        "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06\x08"
                        "\x00\x00\x00\x00\x00\x00\x00",
                        
                        "test3": "CDF\x01\x00\x00\x00\x00\x00\x00\x00\n"
                        "\x00\x00\x00\x01\x00\x00\x00\x03one0\x00\x00"
                        "\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                        "\x00\x00\x0b\x00\x00\x00\x01\x00\x00\x00\tvar"
                        "iable0000\x00\x00\x00\x01\x00\x00\x00\x00\x00"
                        "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06"
                        "\x08\x00\x00\x00\x00\x00\x00\x00",
                        
                        "test4": "CDF\x01\x00\x00\x00\x00\x00\x00\x00\n"
                        "\x00\x00\x00\x01\x00\x00\x00\x03one0\x00\x00"
                        "\x00\x01\x00\x00\x00\x0c\x00\x00\x00\x01\x00"
                        "\x00\x00\nattribute000\x00\x00\x00\x02\x00\x00"
                        "\x00\x1aThis is global attribute 000\x00\x00"
                        "\x00\x0b\x00\x00\x00\x01\x00\x00\x00\tvariabl"
                        "e0000\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00"
                        "\x00\x0c\x00\x00\x00\x01\x00\x00\x00\nattribute1"
                        "00\x00\x00\x00\x02\x00\x00\x00\x1aThis is glob"
                        "al attribute 100\x00\x00\x00\x06\x08\x00\x00"
                        "\x00\x00\x00\x00\x00",
                        
                        "test5": "CDF\x01\x00\x00\x00\x00\x00\x00\x00\n"
                        "\x00\x00\x00\x02\x00\x00\x00\x03one0\x00\x00"
                        "\x00\x01\x00\x00\x00\x03two0\x00\x00\x00\x02"
                        "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                        "\x0b\x00\x00\x00\x03\x00\x00\x00\tvariable00"
                        "00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00"
                        "\x0c\x00\x00\x00\x01\x00\x00\x00\n"
                        "attribute000\x00\x00\x00\x02\x00\x00\x00\x1aTh"
                        "is is global attribute 000\x00\x00\x00\x06\x08"
                        "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\tvaria"
                        "ble1000\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00"
                        "\x00\x0c\x00\x00\x00\x01\x00\x00\x00\n"
                        "attribute100\x00\x00\x00\x02\x00\x00\x00\x1aThis"
                        " is global attribute 100\x00\x00\x00\x06\x10\x00"
                        "\x00\x00\x00\x00\x00\x00\x00\x00\x00\tvariable20"
                        "00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01"
                        "\x00\x00\x00\x0c\x00\x00\x00\x01\x00\x00\x00\n"
                        "attribute200\x00\x00\x00\x02\x00\x00\x00\x1aThis is "
                        "global attribute 200\x00\x00\x00\x06\x10\x00\x00"
                        "\x00\x00\x00\x00\x00",
                        
                        "test6": "CDF\x01\x00\x00\x00\x00\x00\x00\x00\n"
                        "\x00\x00\x00\x02\x00\x00\x00\x03one0\x00\x00"
                        "\x00\x01\x00\x00\x00\x03two0\x00\x00\x00\x02"
                        "\x00\x00\x00\x0c\x00\x00\x00\x01\x00\x00\x00\n"
                        "attribute000\x00\x00\x00\x02\x00\x00\x00\x1aT"
                        "his is global attribute 000\x00\x00\x00\x0b"
                        "\x00\x00\x00\x03\x00\x00\x00\tvariable0000\x00"
                        "\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                        "\x00\x00\x00\x00\x00\x00\x06\x08\x00\x00\x00\x00"
                        "\x00\x00\x00\x00\x00\x00\tvariable1000\x00\x00"
                        "\x00\x01\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00"
                        "\x00\x00\x00\x00\x00\x06\x10\x00\x00\x00\x00\x00"
                        "\x00\x00\x00\x00\x00\tvariable2000\x00\x00\x00"
                        "\x02\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00"
                        "\x00\x00\x00\x00\x00\x00\x00\x00\x06\x10\x00\x00"
                        "\x00\x00\x00\x00\x00",
                        
                        "test7": "CDF\x01\x00\x00\x00\x00\x00\x00\x00\n"
                        "\x00\x00\x00\x02\x00\x00\x00\x03one0\x00\x00"
                        "\x00\x01\x00\x00\x00\x03two0\x00\x00\x00\x02"
                        "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                        "\x0b\x00\x00\x00\x03\x00\x00\x00\tvariable00"
                        "00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00"
                        "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06\x08"
                        "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\tvar"
                        "iable1000\x00\x00\x00\x01\x00\x00\x00\x01\x00"
                        "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06"
                        "\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                        "\tvariable2000\x00\x00\x00\x02\x00\x00\x00\x00"
                        "\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00"
                        "\x00\x00\x00\x00\x06\x10\x00\x00\x00\x00\x00"
                        "\x00\x00",

                        "test8": "CDF\x01\x00\x00\x00\x00\x00\x00\x00\n"
                        "\x00\x00\x00\x02\x00\x00\x00\x03one0\x00\x00"
                        "\x00\x01\x00\x00\x00\x03two0\x00\x00\x00\x02"
                        "\x00\x00\x00\x0c\x00\x00\x00\x01\x00\x00\x00\n"
                        "attribute000\x00\x00\x00\x02\x00\x00\x00\x1aTh"
                        "is is global attribute 000\x00\x00\x00\x0b\x00"
                        "\x00\x00\x03\x00\x00\x00\tvariable0000\x00\x00"
                        "\x00\x01\x00\x00\x00\x00\x00\x00\x00\x0c\x00"
                        "\x00\x00\x01\x00\x00\x00\n"
                        "attribute100\x00\x00\x00\x02\x00\x00\x00\x1aThi"
                        "s is global attribute 100\x00\x00\x00\x06\x08"
                        "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\tvaria"
                        "ble1000\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00"
                        "\x00\x0c\x00\x00\x00\x01\x00\x00\x00\n"
                        "attribute200\x00\x00\x00\x02\x00\x00\x00\x1aThis"
                        " is global attribute 200\x00\x00\x00\x06\x10\x00"
                        "\x00\x00\x00\x00\x00\x00\x00\x00\x00\tvariable20"
                        "00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00"
                        "\x01\x00\x00\x00\x0c\x00\x00\x00\x01\x00\x00"
                        "\x00\n"
                        "attribute300\x00\x00\x00\x02\x00\x00\x00\x1aThis "
                        "is global attribute 300\x00\x00\x00\x06\x10\x00"
                        "\x00\x00\x00\x00\x00\x00"
                        }
    variables = {}
    global_attributes = {}
    vn = helpers.variable_namer()
    ac = helpers.attribute_creator()
    
    nc = netcdf_file(None)
    for d in dimensions:
        nc.createDimension(d, helpers.dimension_lengths[d])
    for i in range(global_atts):
        attr = ac.next()
        nc.__setattr__(attr[0], attr[1])

    # create a variable for each combination of dimensions
    for r in range(len(dimensions)):
        for dim_set in list(combinations(dimensions, r+1)):
            varname = vn.next()
            var = helpers.default_variable_from_dimensions(dim_set)
            var_attrs = {}
            # assign variable_atts attributes to each variable
            for i in range(variable_atts):
                attr = ac.next()
                var_attrs[attr[0]] = attr[1]
            nc.createVariable(varname, "f8", dim_set, var_attrs)
                
    assert (nc._header() == expected_results[expected_key])
    
def test_netcdf_file_write():
    assert(True)
    
def test_netcdf_file_data():
    assert(True)

@pytest.mark.skip    
def test_netcdf_file_calc_begins(dimensions, global_atts, variable_atts, expected):
    assert(True)

@pytest.mark.xfail
@pytest.mark.parametrize('dimensions',[
                         ('three', 'twelve'),
                         ('thousand', 'one'),
                         ('thousand', 'two'),
                         ('three', 'thousand'),
                         ('unlimited',),
                         ('unlimited', 'one'),
                         ('unlimited', 'twelve', 'two')])
@pytest.mark.parametrize('records', [0, 2, 10])     
def test_netcdf_file_set_numrecs(dimensions, records):
    variables = helpers.combinatoric_variables(dimensions)
    recvars = False
    for v in variables:
        if "unlimited" in variables[v].dimensions and records > 0:
            recvars = True
            for i in range(records):
                variables[v].data[i] = 10
            print(variables[v][:])
    def check_rec_count(nc):
        if recvars:
            assert(nc.__dict__['_recs'] == records)
        else:
            assert(nc.__dict__['_recs'] == 0)
    helpers.multiformat_netcdf_testing(dimensions, variables, {}, check_rec_count)
    
def test_netcdf_file_numrecs():
    assert(True)
    
def test_netcdf_file_dim_array():
    assert(True)
    
def test_netcdf_file_att_array():
    assert(True)
    
def test_netcdf_file_att_array():
    assert(True)
    
def test_netcdf_file_init():
    assert(True)

def test_netcdf_file_var_array():
    assert(True)
    
def test_netcdf_file_var_metadata():
    assert(True)
    
def test_netcdf_file_values():
    assert(True)

# all the read* and unpack* functions just directly access the
# filestream attribute of the class and parse what they get.
# This makes them hard to test - we'd have to mock up a filestream
# somehow. I'm sure pytest can do it, but it's not a priority right
# now, as 1) these are mostly straightforward functions, and 
# 2) they're tested as part of the overall read/writes in 
# multiformat_netcdf_testing
@pytest.mark.skip    
def test_netcdf_file_read():
    assert(True)

@pytest.mark.skip    
def test_netcdf_file_read_numrecs():
    assert(True)

@pytest.mark.skip    
def test_netcdf_file_read_dimarray():
    assert(True)
    
@pytest.mark.skip
def test_netcdf_file_read_gattarray():
    assert(True)

@pytest.mark.skip    
def test_netcdf_file_read_attarray():
    assert(True)

@pytest.mark.skip    
def test_netcdf_file_read_vararray():
    assert(True)

@pytest.mark.skip    
def test_netcdf_file_read_var():
    assert(True)

@pytest.mark.skip        
def test_netcdf_file_read_values():
    assert(True)

@pytest.mark.parametrize("int,packed", [
    (0, '\x00\x00\x00\x00'),
    (1, '\x00\x00\x00\x01'),
    (-4, '\xff\xff\xff\xfc'),
    (-800, '\xff\xff\xfc\xe0'),
    (200000000, '\x0b\xeb\xc2\x00')])    
def test_netcdf_file_pack_begin(int, packed):
    assert (empty_file._pack_begin(int) == packed)


def test_netcdf_file_pack_begin():
    assert(True)

@pytest.mark.parametrize('int,packed', [
    (0, '\x00\x00\x00\x00'),
    (1, '\x00\x00\x00\x01'),
    (-4, '\xff\xff\xff\xfc'),
    (-800, '\xff\xff\xfc\xe0'),
    (200000000, '\x0b\xeb\xc2\x00')])
def test_netcdf_file_pack_int(int, packed):
    assert(empty_file._pack_int(int) == packed)

@pytest.mark.skip
def test_netcdf_file_unpack_int():
    assert(True)
    
@pytest.mark.parametrize('int,packed', [
    (0, '\x00\x00\x00\x00\x00\x00\x00\x00'),
    (1, '\x00\x00\x00\x00\x00\x00\x00\x01'),
    (-4, '\xff\xff\xff\xff\xff\xff\xff\xfc'),
    (-800, '\xff\xff\xff\xff\xff\xff\xfc\xe0'),
    (200000000, '\x00\x00\x00\x00\x0b\xeb\xc2\x00'),
    (5000000000, '\x00\x00\x00\x01*\x05\xf2\x00'),
    (-6000000000, '\xff\xff\xff\xfe\x9a_D\x00')])    
def test_netcdf_file_pack_int64(int, packed):
    assert(empty_file._pack_int64(int) == packed)

@pytest.mark.skip    
def test_netcdf_file_unpack_int64():
    assert(True)

@pytest.mark.parametrize('string,packed', [
    ("text", '\x00\x00\x00\x04text'),
    ("$#%\"`", '\x00\x00\x00\x05$#%"`000'),
    ("12896", '\x00\x00\x00\x0512896000'),
    ("\r\t\n", '\x00\x00\x00\x03\r\t\n0')])      
def test_netcdf_file_pack_string(string, packed):
    assert(empty_file._pack_string(string) == packed)
    assert(True)

@pytest.mark.skip        
def test_netcdf_file_unpack_string():
    assert(True)