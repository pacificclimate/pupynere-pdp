import unittest
import os
import tempfile
import numpy as np
import pupynere
from itertools import chain

# test creating variables with unlimited dimensions,
# writing to and retrieving data from such variables.

# create an n1dim by n2dim by n3dim random array
n1dim = 4
n2dim = 3
n3dim = 3
#arr = np.arange(n1dim*n2dim*n3dim).reshape(n1dim, n2dim, n3dim)
arr = np.random.rand(n1dim, n2dim, n3dim).astype(np.float32)

FILE_NAME = tempfile.mktemp(".nc")
FILE_NAME = 'in.nc' # Change to tempfile once testing verified
OUT_FILE = tempfile.mktemp(".nc")
OUT_FILE = 'foo.nc' # Change to tempfile once testing verified

class NcGeneratorWithRecVarsTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        self.outfile = OUT_FILE
        nc = pupynere.netcdf_file(self.file, 'w')
        # add attributes, dimensions and variables to the netcdf_file object
        nc.createDimension('n1', None)
        nc.createDimension('n2', n2dim)
        nc.createDimension('n3', n3dim)
        v1 = nc.createVariable('data1', np.dtype('float32'), ('n1','n2','n3'))
        print arr.dtype.str[1:]
        v1[:] = arr
        self.nc = nc

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """Testing iterating over files with unlimited dimensions"""
        nc = self.nc
        print 'Recvars: {}'.format(nc.recvars.items())
        print 'Non Recvars: {}'.format(nc.non_recvars.items())
        print nc.variables['data1'][:]
        print nc.non_recvars.items()
        pipeline = pupynere.nc_generator(nc, chain(arr))
        n_out = open(self.outfile, 'w')
        for block in pipeline:
            n_out.write(block)

class NcGeneratorNoRecVarsTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        self.outfile = OUT_FILE
        nc = pupynere.netcdf_file(self.file, 'w')
        # add attributes, dimensions and variables to the netcdf_file object
        nc.createDimension('n1', n1dim)
        nc.createDimension('n2', n2dim)
        nc.createDimension('n3', n3dim)
        v1 = nc.createVariable('data1', arr.dtype.str[1:], ('n1','n2','n3'))
        v1[:] = arr
        self.nc = nc

    def runTest(self):
        """Testing iterating over files without unlimited dimensions"""
        nc = self.nc
        print 'Recvars: {}'.format(nc.recvars.items())
        print 'Non Recvars: {}'.format(nc.non_recvars.items())
        print nc.variables['data1'][:]
        print nc.non_recvars.items()
        pipeline = pupynere.nc_generator(nc, chain(arr))
        n_out = open(self.outfile, 'w')
        for block in pipeline:
            n_out.write(block)


if __name__ == '__main__':
    unittest.main()
