# -*- coding: utf-8 -*-
import unittest
import tempfile
import os

import pupynere

FILE_NAME = tempfile.mktemp(".nc")

class TestBasicReadTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        f = pupynere.netcdf_file(self.file, 'w')
        f.history = 'Created for a test'
        f.location = u'北京'
        f.createDimension('time', 10)
        time = f.createVariable('time', 'i', ('time',))
        time[:] = range(10)
        time.units = u'µs since 2008-01-01'
        f.close()

    def tearDown(self):
        # Remove the temporary file
        os.remove(self.file)

    def runTest(self):
        '''Testing a basic read case'''
        f = pupynere.netcdf_file(FILE_NAME, 'r')
        self.assertEqual(f.history, 'Created for a test')
        self.assertEqual(f.location, u'北京')
        time = f.variables['time']
        self.assertEqual(time.units, u'µs since 2008-01-01')
        self.assertEqual(time.shape, (10,))
        self.assertEqual(time[-1], 9)
        f.close()
