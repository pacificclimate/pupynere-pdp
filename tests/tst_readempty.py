import unittest
import os
import tempfile
import pupynere

# test attribute creation.
FILE_NAME = tempfile.mktemp(".nc")
ATTR0 = 'my first attribute'
ATTR1 = 'my second attribute'
ATTDICT = {'attr0': ATTR0,
           'attr1': ATTR1}

class EmptyIsReadableTestCase(unittest.TestCase):

    def setUp(self):
        self.file = FILE_NAME
        f = pupynere.netcdf_file(self.file,'w')
        f.attr0 = ATTR0
        f.attr1 = ATTR1
        f.close()

    def tearDown(self):
        # Remove the temporary files
        os.remove(self.file)

    def runTest(self):
        """Testing whether we can read a file with no variables"""
        f  = pupynere.netcdf_file(self.file, 'r')
        # check attributes in root group.
        # global attributes.
        # check __dict__ method for accessing all netCDF attributes.
        for key,val in ATTDICT.items():
            assert f.__dict__[key] == val
        # check accessing individual attributes.
        assert f.attr0 == ATTR0
        assert f.attr1 == ATTR1
        f.close()

if __name__ == '__main__':
    unittest.main()
