import math
import unittest
import os
import tempfile
import numpy as NP
from pupynere import netcdf_file as netCDF4
from numpy.testing import assert_almost_equal

# test attribute creation.
FILE_NAME = tempfile.mktemp(".nc")
VAR_NAME="dummy_var"
DIM1_NAME="x"
DIM1_LEN=2
DIM2_NAME="y"
DIM2_LEN=3
DIM3_NAME="z"
DIM3_LEN=25
STRATT = 'string attribute'
EMPTYSTRATT = ''
INTATT = 1
FLOATATT = math.pi
SEQATT = NP.arange(10)

class AttributesTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        f = netCDF4(self.file,'w')

        f.stratt = STRATT
        f.emptystratt = EMPTYSTRATT
        f.intatt = INTATT
        f.floatatt = FLOATATT
        f.seqatt = SEQATT

        f.createDimension(DIM1_NAME, DIM1_LEN)
        f.createDimension(DIM2_NAME, DIM2_LEN)
        f.createDimension(DIM3_NAME, DIM3_LEN)

        v = f.createVariable(VAR_NAME, 'f8',(DIM1_NAME,DIM2_NAME,DIM3_NAME))
        v.stratt = STRATT
        v.emptystratt = EMPTYSTRATT
        v.intatt = INTATT
        v.floatatt = FLOATATT
        v.seqatt = SEQATT

        f.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """Testing that attributes are properly set"""
        f  = netCDF4(self.file, 'r')
        v = f.variables[VAR_NAME]

        # global attributes.
        self.assertEquals(f.intatt, INTATT)
        assert_almost_equal(f.floatatt, FLOATATT, 6)
        self.assertEquals(f.stratt, STRATT)
        self.assertEquals(f.emptystratt, EMPTYSTRATT)
        self.assertEquals(f.seqatt.tolist(), SEQATT.tolist())

        # variable attributes
        self.assertEquals(v.intatt, INTATT)
        self.assertEquals(v.stratt, STRATT)
        self.assertEquals(v.seqatt.tolist(), SEQATT.tolist())
        assert_almost_equal(v.floatatt, FLOATATT, 6)
        self.assertEquals(v.emptystratt, EMPTYSTRATT)

        f.close()

if __name__ == '__main__':
    unittest.main()
