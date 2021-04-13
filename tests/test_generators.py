# -*- coding: utf-8 -*-
import unittest
import tempfile
import os
import math
import numpy as np
from numpy import dtype

from pupynere import netcdf_file, byteorderer, nc_streamer, nc_generator, nc_writer

VAR_NAME='temp'
VAR_TYPE='>f4'
VAR_VAL=math.pi
FILE_NAME = tempfile.mktemp(".nc")
n1dim = 4
n2dim = 10
n3dim = 8
ranarr = 100.*np.zeros(shape=(n1dim,n2dim,n3dim))

class TestGeneratorNonrecvars(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        f = netcdf_file(self.file, 'w')
        temp = f.createVariable(VAR_NAME,VAR_TYPE)
        temp.assignValue(VAR_VAL)
        f.close()

    def tearDown(self):
        # Remove the temporary file
        os.remove(self.file)

    def runTest(self):
        f = netcdf_file(FILE_NAME, 'r')
        pipeline = nc_generator(f, _input())
        with tempfile.NamedTemporaryFile(suffix=".nc") as fn:
            for block in pipeline:
                fn.write(block)

            nc = netcdf_file(fn.name, 'r')
            assert nc.variables.has_key(VAR_NAME)
            assert nc.variables[VAR_NAME].data == np.float32(VAR_VAL)
            assert nc.variables[VAR_NAME].dtype == dtype(VAR_TYPE)

            nc.close()

        f.close()


class TestGeneratorRecvars(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        self.keys = ('n1','n2','n3')
        self.dims = [None, n2dim, n3dim]

        f = netcdf_file(self.file, 'w')
        for i in range(len(self.keys)):
            f.createDimension(self.keys[i], self.dims[i])
        foo = f.createVariable('data1', ranarr.dtype.str[1:], self.keys)

        # write some data to it.
        foo[:] = ranarr
        foo[n1dim:,:,:] = 2.*ranarr
        f.close()

    def tearDown(self):
        # Remove the temporary file
        os.remove(self.file)

    def runTest(self):
        f = netcdf_file(FILE_NAME, 'r')
        pipeline = nc_generator(f, _input())
        with tempfile.NamedTemporaryFile(suffix=".nc") as fn:
            for block in pipeline:
                fn.write(block)

            nc = netcdf_file(fn.name, 'r')
            assert nc.variables.has_key('data1')
            for i, n in enumerate(self.keys):
                assert nc.dimensions.has_key(n)
                assert nc.dimensions[n] == self.dims[i]

            nc.close()

        f.close()


def _input():
    yield np.arange(10000).reshape(100, 100)
