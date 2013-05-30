import sys
import unittest
import os
import tempfile
import numpy as NP
from numpy.random.mtrand import uniform 
import pupynere

# test creating variables with unlimited dimensions,
# writing to and retrieving data from such variables.

# create an n1dim by n2dim by n3dim random array
n1dim = 4
n2dim = 10
n3dim = 8
ranarr = 100.*NP.zeros(shape=(n1dim,n2dim,n3dim))

class TestNonFirstUnlim(unittest.TestCase):
    def runTest(self):
        """Should not be able to create a non-first unlimited dimension"""
        f = pupynere.netcdf_file(None, 'w')
        f.createDimension('n1', n1dim)
        with self.assertRaises(ValueError):
            f.createDimension('n2', None)
        self.assertEquals(f.dimensions, {'n1': n1dim})

class TestMultipleUnlim(unittest.TestCase):
    def runTest(self):
        """Should not be able to create a variable where there are multiple unlimited dimensions"""
        f = pupynere.netcdf_file(None, 'w')
        f.createDimension('n1', None)
        f.createDimension('n2', n2dim)
        with self.assertRaises(ValueError):
            f.createVariable('data1', None, ('n2', 'n1'))
        f.close()

class FirstUnlimdimTestCase(unittest.TestCase):

    def setUp(self):
        self.file = tempfile.mktemp(".nc")
        f  = pupynere.netcdf_file(self.file, 'w')
        # foo has a single unlimited dimension
        f.createDimension('n1', None)
        f.createDimension('n2', n2dim)
        f.createDimension('n3', n3dim)
        foo = f.createVariable('data1', ranarr.dtype.str[1:], ('n1','n2','n3'))
        # write some data to it.
        foo[:] = ranarr 
        foo[n1dim:,:,:] = 2.*ranarr
        f.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """Testing the creation of an unlimited dimensions"""
        f  = pupynere.netcdf_file(self.file, 'r')
        foo = f.variables['data1']
        # check shape.
        self.assertEquals(foo.shape, (2*n1dim,n2dim,n3dim))
        # check data.
        self.assertTrue((foo[0:n1dim,:,:] == ranarr).all())
        self.assertTrue((foo[n1dim:3*n1dim,:,:] == 2.*ranarr).all())
        f.close()

class GreaterThanFourGB(unittest.TestCase):
    pass

if __name__ == '__main__':
    unittest.main()
