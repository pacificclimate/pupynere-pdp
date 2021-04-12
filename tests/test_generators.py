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

class TestGenerators(unittest.TestCase):

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
        '''Testing a basic read case'''
        def _input():
            yield np.arange(10000).reshape(100, 100)

        f = netcdf_file(FILE_NAME, 'r')
        pipeline = nc_generator(f, _input())
        with tempfile.NamedTemporaryFile(suffix=".nc") as fn:
            for block in pipeline:
                fn.write(block)

            nc = netcdf_file(fn.name, 'r')
            assert nc.variables.has_key(VAR_NAME)
            assert nc.variables[VAR_NAME].data == np.float32(VAR_VAL)
            assert nc.variables[VAR_NAME].dtype == dtype(VAR_TYPE)

        f.close()
