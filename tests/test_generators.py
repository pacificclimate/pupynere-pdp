# -*- coding: utf-8 -*-
import unittest
import tempfile
import math
import numpy as np
from numpy import dtype

from pupynere import netcdf_file, nc_generator

VAR_NAME = 'temp'
VAR_TYPE = 'f4'
VAR_VAL = math.pi

n1dim = 4
n2dim = 10
n3dim = 8
ranarr = 100.*np.zeros(shape=(n1dim,n2dim,n3dim))

class TestGeneratorNonrecvars(unittest.TestCase):

    def runTest(self):
        f = netcdf_file(None, 'w')
        temp = f.createVariable(VAR_NAME, dtype(VAR_TYPE))
        temp.assignValue(VAR_VAL)

        # Test filesize property of a virtual file with nonrecvars
        assert f.filesize == 88

        # Test generator
        pipeline = nc_generator(f, _input())
        with tempfile.NamedTemporaryFile(suffix=".nc") as fn:
            for block in pipeline:
                fn.write(block)

            fn.flush()
            nc = netcdf_file(fn.name, 'r')
            assert nc.variables.has_key(VAR_NAME)
            assert nc.variables[VAR_NAME].data == np.float32(VAR_VAL)
            assert nc.variables[VAR_NAME].dtype == dtype('>f4')

            nc.close()

        f.close()


class TestGeneratorRecvars(unittest.TestCase):

    def runTest(self):
        keys = ('n1','n2','n3')
        dims = [None, n2dim, n3dim]

        f = netcdf_file(None, 'w')
        for i in range(len(keys)):
            f.createDimension(keys[i], dims[i])
        foo = f.createVariable('data1', ranarr.dtype, keys)

        # write some data to it.
        foo[:] = ranarr
        foo[n1dim:,:,:] = 2.*ranarr

        # Test filesize
        with self.assertRaises(ValueError):
            f.filesize

        # Test generator
        pipeline = nc_generator(f, _input())
        with tempfile.NamedTemporaryFile(suffix=".nc") as fn:
            for block in pipeline:
                fn.write(block)

            fn.flush()
            nc = netcdf_file(fn.name, 'r')
            assert nc.variables.has_key('data1')
            for i, n in enumerate(keys):
                assert nc.dimensions.has_key(n)
                assert nc.dimensions[n] == dims[i]

            nc.close()

        f.close()


def _input():
    yield np.arange(10000).reshape(100, 100)
