# Tests for the functions that aren't used by the PDP, but are part of this package.
# Some of them are used to stream file data (the PDP only uses headers), some of them
# are written to use coroutines instead of iterative control flows.

# need to test: nc_writer, nc_streamer,
import unittest
from pupynere import coroutine, byteorderer, check_byteorder, nc_writer, nc_streamer, netcdf_file
import numpy as np
import tempfile

class TestCoroutineDecorator(unittest.TestCase):
    
    def setUp(self):
        self.setup_side_effect = 0

    def plain_generator(self):
        self.setup_side_effect = 1
        for n in range(5,10):
            yield(n)

    @coroutine
    def co_generator(self):
        self.setup_side_effect = 2
        #first yield is lost?
        yield(None)
        for n in range(5,10):
            yield(n)

    def runTest(self):
        # verify that the non-coroutine generator doesn't finish "setup"
        # until data is requested
        test_func = self.plain_generator()
        self.assertEquals(self.setup_side_effect, 0)
        self.assertEquals(next(test_func), 5)
        self.assertEquals(self.setup_side_effect, 1)
        
        # verify that the coroutine version pre-loads setup
        # should this be the way this works? The first generated item is
        # lost.
        test_func = self.co_generator()
        self.assertEquals(self.setup_side_effect, 2)
        self.assertEquals(next(test_func), 5)


class TestByteOrder(unittest.TestCase):
    
    def setUp(self):
        if np.little_endian:
            self.little_endian_array = np.array([1, 2, 3, 4, 5])
            self.big_endian_array = self.little_endian_array.newbyteorder()
        else:
            self.big_endian_array = np.array([1, 2, 3, 4, 5])
            self.little_endian_array = self.little_endian_array.newbyteorder()
        self.reset_buffer()
        # make sure we actually have two different arrays
        self.assertEquals((self.little_endian_array == self.big_endian_array).all(), False)


    def downstream_generator(self):
        # a generator data can be sent to. Just writes the
        # data to a buffer for further testing
        while True:
            data = yield
            self.buffer = data
    
    def upstream_generator(self):
        # a generator that sends data. It sends whatever is in
        # the buffer
        while True:
            yield self.buffer
    
    def check_is_big_endian(self, num):
        # compares it to the exemplay big endian array,
        # not a general check
        return((num == self.big_endian_array).all())
    
    def check_buffer_is_big_endian(self):
        return self.check_is_big_endian(self.buffer)

    def reset_buffer(self):
        self.buffer = None
        
    def test_byteorderer(self):
        # whether we supply the generator with a big endian or little endian 
        # array, we should get a big endian array back.
        downstream = self.downstream_generator()
        next(downstream)
        process = byteorderer(downstream)
        process.send(self.big_endian_array)
        self.assertTrue(self.check_buffer_is_big_endian())
        self.reset_buffer()
        process.send(self.little_endian_array)
        self.assertTrue(self.check_buffer_is_big_endian())
        self.reset_buffer()
    
    def test_check_byteorder(self):
        # whether we supply the generator with a big endian or little
        # endian array, we should geta big endian array back
        upstream = self.upstream_generator()
        process = check_byteorder(upstream)
        
        self.buffer = self.big_endian_array
        output = next(process)
        self.assertTrue(self.check_is_big_endian(output))        
        self.buffer = self.little_endian_array
        output = next(process)
        self.assertTrue(self.check_is_big_endian(output))        
        self.reset_buffer()
        
class TestNCStreamer(unittest.TestCase):
    #tests the nc_streamer + nc_writer pipeline. this is not the one used
    # by the pdp - that one uses nc_generator
    
    def setUp(self):
        #create a netCDF file we'll use to test streaming and writing
        self.ncdata = {}
        self.ncattributes = {}
        self.ncdimensions = {}

        self.virtual_netcdf = netcdf_file(None, 'w')
        self.ncdimensions["lat"] = 5
        self.virtual_netcdf.createDimension("lat", self.ncdimensions["lat"])
        self.ncdimensions["lon"] = 3
        self.virtual_netcdf.createDimension("lon", self.ncdimensions["lon"])
        self.ncdimensions["time"] = 10
        self.virtual_netcdf.createDimension("time", self.ncdimensions["time"])
        
        self.virtual_netcdf.history = "created by TestNCStreamer"
        
        self.ncdata["lat"] = np.array([51, 52, 53, 54, 55])
        self.ncattributes["lat"] = {"units": "degrees_north"}
        latvar = self.virtual_netcdf.createVariable("lat", "f4", dimensions=["lat"], 
                                    attributes=self.ncattributes["lat"])
        latvar[:] = self.ncdata["lat"]
        
        self.ncdata["lon"] = np.array([-120, -121, -122])
        self.ncattributes["lon"] = {"units": "degrees_east"}
        lonvar = self.virtual_netcdf.createVariable("lon", "f4", dimensions=["lon"],
                                    attributes=self.ncattributes["lon"])
        lonvar[:] = self.ncdata["lon"]
        
        self.ncdata["time"] = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.ncattributes["time"] = {"units": "days since 1955-1-1", "calendar": "standard"}
        timevar = self.virtual_netcdf.createVariable("time", "f4", dimensions=["time"], 
                                    attributes=self.ncattributes["time"])
        timevar[:] = self.ncdata["time"]
        
        self.ncdata["precipitation"] = np.array(range(150)).reshape(5,3,10)
        self.ncattributes["precipitation"] = {"units": "mm/day"}
        precipvar = self.virtual_netcdf.createVariable("precipitation", "f4", 
                                dimensions=["lat", "lon", "time"],
                                attributes = self.ncattributes["precipitation"])
        precipvar[:] = self.ncdata["precipitation"]
        
    def verify_netCDF(self, nc_file):
        # verify that a written file matches the virtual test file
        nc = netcdf_file(nc_file)
        
        # make sure dimensions exist and have right length
        for dimension in ("lat", "lon", "time"):
            self.assertTrue(nc.dimensions.has_key(dimension))
            self.assertTrue(nc.dimensions[dimension] == self.ncdimensions[dimension])
        
        # make sure variables exist and have right data and attributes
        for variable in ("lat", "lon", "time", "precipitation"):
            self.assertTrue(nc.variables.has_key(variable))
            self.assertTrue((nc.variables[variable][:] == self.ncdata[variable]).all())
            for att in self.ncattributes[variable].keys():
                self.assertTrue(nc.variables[variable][att] == self.ncattributes[variable][att])
        
        # make sure global attribute is correct
        self.assertEquals(nc.global_attribute, "created by TestNCStreamer")
        
        nc.close()
    
    def test_nc_streamer(self):
        pipeline = byteorderer(nc_streamer(self.virtual_netcdf, nc_writer("/tmp/testfile.nc")))
        for var in self.virtual_netcdf.variables:
            data = self.virtual_netcdf.variables[var][:]
            pipeline.send(data)
        pipeline.close()
        self.verify_netCDF("/tmp/testfile.nc")
