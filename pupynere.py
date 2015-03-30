# -*- coding: utf-8 -*-
from __future__ import division
from warnings import warn
import logging

u"""
NetCDF reader/writer module.

This module is used to read and create NetCDF files. NetCDF files are
accessed through the `netcdf_file` object. Data written to and from NetCDF
files are contained in `netcdf_variable` objects. Attributes are given
as member variables of the `netcdf_file` and `netcdf_variable` objects.

Notes
-----
NetCDF files are a self-describing binary data format. The file contains
metadata that describes the dimensions and variables in the file. More
details about NetCDF files can be found `here
<http://www.unidata.ucar.edu/software/netcdf/docs/netcdf.html>`_. There
are three main sections to a NetCDF data structure:

1. Dimensions
2. Variables
3. Attributes

The dimensions section records the name and length of each dimension used
by the variables. The variables would then indicate which dimensions it
uses and any attributes such as data units, along with containing the data
values for the variable. It is good practice to include a
variable that is the same name as a dimension to provide the values for
that axes. Lastly, the attributes section would contain additional
information such as the name of the file creator or the instrument used to
collect the data.

When writing data to a NetCDF file, there is often the need to indicate the
'record dimension'. A record dimension is the unbounded dimension for a
variable. For example, a temperature variable may have dimensions of
latitude, longitude and time. If one wants to add more temperature data to
the NetCDF file as time progresses, then the temperature variable should
have the time dimension flagged as the record dimension.

This module implements the Scientific.IO.NetCDF API to read and create
NetCDF files. The same API is also used in the PyNIO and pynetcdf
modules, allowing these modules to be used interchangeably when working
with NetCDF files. The major advantage of this module over other
modules is that it doesn't require the code to be linked to the NetCDF
libraries.

In addition, the NetCDF file header contains the position of the data in
the file, so access can be done in an efficient manner without loading
unnecessary data into memory. It uses the ``mmap`` module to create
Numpy arrays mapped to the data on disk, for the same purpose.

Examples
--------

To create a NetCDF file:

    >>> f = netcdf_file('simple.nc', 'w')
    >>> f.history = 'Created for a test'
    >>> f.location = u'北京'
    >>> f.createDimension('time', 10)
    >>> time = f.createVariable('time', 'i', ('time',))
    >>> time[:] = range(10)
    >>> time.units = u'µs since 2008-01-01'
    >>> f.close()

Note the assignment of ``range(10)`` to ``time[:]``.  Exposing the slice
of the time variable allows for the data to be set in the object, rather
than letting ``range(10)`` overwrite the ``time`` variable.

To read the NetCDF file we just created:

    >>> f = netcdf_file('simple.nc', 'r')
    >>> print f.history
    Created for a test
    >>> print f.location
    北京
    >>> time = f.variables['time']
    >>> print time.units
    µs since 2008-01-01
    >>> print time.shape
    (10,)
    >>> print time[-1]
    9
    >>> f.close()

"""

__all__ = ['netcdf_file']


from operator import mul
from mmap import mmap, ACCESS_READ
from os import stat
try:
    from mmap import ALLOCATIONGRANULARITY
except ImportError:
    ALLOCATIONGRANULARITY = None

import numpy as np
from numpy.compat import asbytes, asstr
from numpy import fromstring, ndarray, dtype, empty, array, asarray
from numpy import little_endian as LITTLE_ENDIAN


ABSENT       = asbytes('\x00\x00\x00\x00\x00\x00\x00\x00')
ZERO         = asbytes('\x00\x00\x00\x00')
NC_BYTE      = asbytes('\x00\x00\x00\x01')
NC_CHAR      = asbytes('\x00\x00\x00\x02')
NC_SHORT     = asbytes('\x00\x00\x00\x03')
NC_INT       = asbytes('\x00\x00\x00\x04')
NC_FLOAT     = asbytes('\x00\x00\x00\x05')
NC_DOUBLE    = asbytes('\x00\x00\x00\x06')
NC_DIMENSION = asbytes('\x00\x00\x00\n')
NC_VARIABLE  = asbytes('\x00\x00\x00\x0b')
NC_ATTRIBUTE = asbytes('\x00\x00\x00\x0c')

logger = logging.getLogger(__name__)

# Map from Netcdf types and how they should be read. Netcdf is big endian.
def TYPEMAP(nctype):
    static = { NC_BYTE:   dtype(np.byte),
               #NC_CHAR:   dtype('c'),
               NC_SHORT:  dtype(np.int16).newbyteorder('>'),
               NC_INT:    dtype(np.int32).newbyteorder('>'),
               NC_FLOAT:  dtype(np.float32).newbyteorder('>'),
               NC_DOUBLE: dtype(np.float64).newbyteorder('>'),
            }
    if nctype in static:
        return static[nctype]
    elif nctype == NC_CHAR:
        return dtype('|S1')
    else:
        raise Exception("numpy has no corresponding type to NetCDF3 type %s" % nctype)

# Map between Numpy types and the corresponding Netcdf type.
def REVERSE(nptype):
    static = { dtype(np.byte):    NC_BYTE,
               #dtype('c'):        NC_CHAR,
               dtype('<U1'):      NC_INT,
               dtype(np.int16):   NC_SHORT,
               dtype(np.int32):   NC_INT,
               dtype(np.int64):   NC_INT,  # will be converted to int32
               dtype(np.float32): NC_FLOAT,
               dtype(np.float64): NC_DOUBLE,
            }
    if nptype in static:
        return static[nptype]
    elif nptype.char == 'S':
        return NC_CHAR
    else:
        raise Exception("NetCDF 3 does not support type %s" % nptype)


class netcdf_file(object):
    """
    A file object for NetCDF data.

    A `netcdf_file` object has two standard attributes: `dimensions` and
    `variables`. The values of both are dictionaries, mapping dimension
    names to their associated lengths and variable names to variables,
    respectively. Application programs should never modify these
    dictionaries.

    All other attributes correspond to global attributes defined in the
    NetCDF file. Global file attributes are created by assigning to an
    attribute of the `netcdf_file` object.

    Parameters
    ----------
    filename : string or file-like
        string -> filename
    mode : {'r', 'w'}, optional
        read-write mode, default is 'r'
    mmap : None or bool, optional
        Whether to mmap `filename` when reading.  Default is True
        when `filename` is a file name, False when `filename` is a
        file-like object
    version : {1, 2}, optional
        version of netcdf to read / write, where 1 means *Classic
        format* and 2 means *64-bit offset format*.  Default is 1.  See
        `here <http://www.unidata.ucar.edu/software/netcdf/docs/netcdf/Which-Format.html>`_
        for more info.
    maskandscale : True or False
        Whether data is automagically scaled and masked.

    """
    def __init__(self, filename, mode='r', mmap=None, version=1, maskandscale=False):
        """Initialize netcdf_file from fileobj (str or file-like)."""
        self._dims = []
        self._recs = 0
        self._recsize = 0

        if not filename: # Just a metadata object... no reading or writing
            self.fp = self.filename = self.mode = mode = None

        elif hasattr(filename, 'seek'):  # file-like
            self.fp = filename
            self.filename = 'None'
            if mmap is None:
                mmap = False
            elif mmap and not hasattr(filename, 'fileno'):
                raise ValueError('Cannot use file object for mmap')

        elif type(filename) == str:  # string?
            self.filename = filename
            self.fp = open(self.filename, '%sb' % mode)
            if mmap is None:
                mmap = True
        else:
            raise TypeError('filename argument to netcdf_file.__init__() must be of type string, be file-like (have a "seek" attribute) or be None. Instead received', filename)

        self.use_mmap = mmap
        self.version_byte = version
        self.maskandscale = maskandscale

        # FIXME: We need an append mode
        if not mode in ('r', 'w', None):
            raise ValueError("Mode must be either 'r', 'w', or None.")
        self.mode = mode

        self.dimensions = {}
        self.variables = NcOrderedDict()

        self._attributes = {}

        # FIXME: Really? Read the entire thing into memory? That seems like a bad choice
        if mode == 'r':
            self._read()

    def __setattr__(self, attr, value):
        # Store user defined attributes in a separate dict,
        # so we can save them to file later.
        try:
            self._attributes[attr] = value
        except AttributeError:
            pass
        self.__dict__[attr] = value

    def close(self):
        """Closes the NetCDF file."""
        if not self.fp:
            return
        if not self.fp.closed:
            try:
                self.flush()
            finally:
                self.fp.close()

    __del__ = close

    @property
    def filesize(self):
        """
        For files on disk, returns the size of the file (in bytes)
        For virtual files to be generated, returns the expected size of
        the resulting file after all data has been streamed through
        """
        if self.fp:
            return stat(self.fp.name).st_size

        if self.recvars:
            if not self._recs:
                raise ValueError("The number or records is not set so it is impossible to calculate the filesize")
            recvar0 = self.recvars.values()[0]
            if not hasattr(recvar0, '_begin'):
                self._calc_begins()
            return int(recvar0._begin + (self._recs * self._recsize))
        else:
            lastvar = self.non_recvars.values()[-1]
            if not hasattr(lastvar, '_begin'):
                self._calc_begins()
            return int(lastvar._begin + lastvar._vsize)


    def createDimension(self, name, length):
        """
        Adds a dimension to the Dimension section of the NetCDF data structure.

        Note that this function merely adds a new dimension that the variables can
        reference.  The values for the dimension, if desired, should be added as
        a variable using `createVariable`, referring to this dimension.

        Parameters
        ----------
        name : str
            Name of the dimension (Eg, 'lat' or 'time').
        length : int
            Length of the dimension.

        See Also
        --------
        createVariable

        """
        if self.dimensions and not length:
            raise ValueError('Unlimited dimension must be the first dimension of a netcdf file')
        assert type(length) == int or length == None
        self.dimensions[name] = length
        self._dims.append(name)

    def createVariable(self, name, type, dimensions=None, attributes=None):
        """
        Create an empty variable for the `netcdf_file` object, specifying its data
        type and the dimensions it uses.

        Parameters
        ----------
        name : str
            Name of the new variable.
        type : dtype or str
            Data type of the variable.
        dimensions : sequence of str
            List of the dimension names used by the variable, in the desired order.
        attributes : dict, optional
            Attribute values (any type) keyed by string names.  These attributes
            become attributes for the netcdf_variable object.

        Returns
        -------
        variable : netcdf_variable
            The newly created ``netcdf_variable`` object.
            This object has also been added to the `netcdf_file` object as well.

        See Also
        --------
        createDimension

        Notes
        -----
        Any dimensions to be used by the variable should already exist in the
        NetCDF data structure or should be created by `createDimension` prior to
        creating the NetCDF variable.

        """
        if not dimensions:
            dimensions = ()

        for dim in dimensions:
            if dim not in self.dimensions:
                raise ValueError('Cannot create variable with dimension "{0}". The netcdf file\'s dimensions are {1}.'.format(dim, self.dimensions))

        shape = tuple([self.dimensions[dim] for dim in dimensions])
        shape_ = tuple([dim or 0 for dim in shape])  # replace None with 0 for numpy

        if None in shape and shape.index(None) != 0:
            raise ValueError("Unlimited dimension must be the first dimensionn to variable %s. Instead got dimension number %d" % (name, shape.index(None)))

        if isinstance(type, basestring):
            type = dtype(type)

        data = empty(shape_, type)

        self.variables[name] = netcdf_variable(
                data, type, shape, dimensions,
                maskandscale=self.maskandscale,
                attributes=attributes)
        return self.variables[name]

    def flush(self):
        """
        Perform a sync-to-disk flush if the `netcdf_file` object is in write mode.

        See Also
        --------
        sync : Identical function

        """
        if hasattr(self, 'mode') and self.mode is 'w':
            self._write()
    sync = flush

    def recvars(self):
        return OrderedDict( filter(lambda (k, v): v.isrec, self.variables.items()) )
    recvars = property(recvars)

    def non_recvars(self):
        return OrderedDict( filter(lambda (k, v): not v.isrec, self.variables.items()) )
    non_recvars = property(non_recvars)

    def __generate__(self):
        self._calc_begins()
        yield self._header()
        for chunk in self._data():
            yield chunk

    def _header(self):
        return asbytes('CDF') + \
               array(self.version_byte, '>b').tostring() + \
               self._numrecs() + \
               self._dim_array() + \
               self._gatt_array() + \
               self._var_array()

    def _write(self):
        self.fp.writelines(self.__generate__())

    def _data(self):
        if self.variables:
            for var in self.variables.values():
                if (var.data.dtype.byteorder == '<' or
                    (var.data.dtype.byteorder == '=' and LITTLE_ENDIAN)):
                    var.data = var.data.byteswap()
                    
            for var in self.non_recvars.values():
                yield var.data.tostring()
                count = var.data.size * var.itemsize
                yield asbytes('0') * (var._vsize - count)

            # Record variables
            for i in range(self._recs):
                for var in self.recvars.values():
                    stride = var.data[i,:] if len(var.data.shape) > 1 else var.data[i]
                    yield stride.tostring()

    def _calc_begins(self):
        '''Each netcdf variable has a metadata item named 'begin' which is an offset to the location in
           the file where the data values begin. This method calculates the offset for each variable and
           sets it in the variable property _begin
        '''
        prev = False
        for name, var in self.variables.items():
            if not prev:
                var.__dict__['_begin'] = len(self._header())
            else:
                var.__dict__['_begin'] = prev._begin + prev._vsize
            prev = var

    def set_numrecs(self, numrecs):
        assert type(numrecs) == int
        self.__dict__['_recs'] = numrecs

    def _numrecs(self):
        if self._recs == 0:
            # Get highest record count from all record variables.
            self.__dict__['_recs'] = 0 if not self.recvars else max([ len(var.data) for var in self.recvars.values() ])
        return self._pack_int(self._recs)

    def _dim_array(self):
        if self.dimensions:
            buf = NC_DIMENSION + self._pack_int(len(self.dimensions))
            for name in self._dims:
                buf += self._pack_string(name)
                length = self.dimensions[name]
                buf += self._pack_int(length or 0)  # replace None with 0 for record dimension
            return buf
        else:
            return ABSENT

    def _gatt_array(self):
        return self._att_array(self._attributes)

    def _att_array(self, attributes):
        if attributes:
            buf = NC_ATTRIBUTE
            buf += self._pack_int(len(attributes))
            for name, values in attributes.items():
                buf += self._pack_string(name)
                buf += self._values(values)
            return buf
        else:
            return ABSENT

    def _var_array(self):
        if self.variables:
            buf = NC_VARIABLE
            buf += self._pack_int(len(self.variables))

            # Set the metadata for all variables.
            for name in self.variables.keys():
                buf += self._var_metadata(name)
            # Now that we have the metadata, we know the vsize of
            # each record variable, so we can calculate recsize.
            self.__dict__['_recsize'] = sum([
                    var._vsize for var in self.recvars.values()
                    ])
            return buf

        else:
            logger.debug("_var_array returning ABSENT")
            return ABSENT

    def _var_metadata(self, name):
        '''
        Returns a string representing a single 'var' from the BNF grammar here:
        http://www.unidata.ucar.edu/software/netcdf/docs/netcdf/File-Format-Specification.html
        '''
        var = self.variables[name]

        # name
        buf = self._pack_string(name)

        # nelems
        buf += self._pack_int(len(var.dimensions))

        # dimid ...
        for dimname in var.dimensions:
            dimid = self._dims.index(dimname)
            buf += self._pack_int(dimid)

        # vatt_array
        buf += self._att_array(var._attributes)

        # nc_type
        nc_type = REVERSE(var.dtype)
        buf += asbytes(nc_type)

        # vsize
        if not var.isrec:
            vsize = var.data.size * var.itemsize
            vsize += -vsize % 4
        else:  # record variable
            try:
                vsize = np.prod(var.shape[1:]) * var.itemsize
            except IndexError:
                vsize = 0
                warn("Could not determine vsize for variable", name, "so I'm defaulting to 0")
            if len(self.recvars.items()) > 1:
                vsize += -vsize % 4
        self.variables[name].__dict__['_vsize'] = vsize
        # But according to "Note on vsize:" from NetCDF spec, vsize is:
        # a) redundant, since it can be determined from other header info
        # b) insufficient, since it's only 32 bits and files can be > 4GB
        # therefore, clip it... the spec made me do it!
        buf += array(min(vsize, 2**32 - 4), 'int32').tostring()

        # begin
        # Pack a bogus begin, if it hasn't been calculated yet
        if hasattr(self.variables[name], '_begin'):
            buf += self._pack_begin(self.variables[name]._begin)
        else:
            buf += self._pack_begin(0)

        return buf

    # FIXME: This is a little messy... the try/except blocks don't really express the program logic in a good way
    def _values(self, values):
        # FIXME: what exactly do we do for arrays of characters? e.g. ['mary', 'had', 'a', 'little', 'lamb']
        # I think that they are illegal, actually
        if type(values) == list:
            raise ValueError("I don't know how to handle a python list type. 'values' parameter should be a numpy array.")
        try:
            nc_type = REVERSE(values.dtype)
        except:
            # FIXME: if REVERSE(values.dtype) raises an attribute error then so will this!
            types = [
                    (int, NC_INT),
                    (long, NC_INT),
                    (float, NC_FLOAT),
                    (basestring, NC_CHAR),
                    ]
            try:
                sample = values[0]
            except (IndexError, TypeError):
                sample = values
            if isinstance(sample, unicode):
                if not isinstance(values, unicode):
                    raise ValueError("NetCDF requires that text be encoded as UTF-8")
                values = values.encode('utf-8')
            for class_, nc_type in types:
                if isinstance(sample, class_): break
            # FIXME: raise exception here? If we get to this point... nc_type is undefined

        # Special case for character types
        # numpy.asarray will detect the length for character types and set dtype accordingly
        if nc_type == NC_CHAR:
            values = asarray(values)
            nelems = values.itemsize
        else:
            values = asarray(values, TYPEMAP(nc_type))
            nelems = values.size

        buf = asbytes(nc_type)
        buf += self._pack_int(nelems)

        if not values.shape and (values.dtype.byteorder == '<' or
                (values.dtype.byteorder == '=' and LITTLE_ENDIAN)):
            values = values.byteswap()
        buf += values.tostring()
        count = values.size * values.itemsize
        buf += asbytes('0') * (-count % 4) # pad
        return buf

    def _read(self):
        # Check magic bytes and version
        magic = self.fp.read(3)
        if not magic == asbytes('CDF'):
            raise TypeError("Error: %s is not a valid NetCDF 3 file" %
                            self.filename)
        self.__dict__['version_byte'] = fromstring(self.fp.read(1), '>b')[0]

        # Read file headers and set data.
        self._read_numrecs()
        self._read_dim_array()
        self._read_gatt_array()
        self._read_var_array()

    def _read_numrecs(self):
        self.__dict__['_recs'] = self._unpack_int()

    def _read_dim_array(self):
        header = self.fp.read(4)
        if not header in [ZERO, NC_DIMENSION]:
            raise ValueError("Unexpected header.")
        count = self._unpack_int()

        for dim in range(count):
            name = asstr(self._unpack_string())
            length = self._unpack_int() or None  # None for record dimension
            self.dimensions[name] = length
            self._dims.append(name)  # preserve order

    def _read_gatt_array(self):
        for k, v in self._read_att_array().items():
            self.__setattr__(k, v)

    def _read_att_array(self):
        header = self.fp.read(4)
        if not header in [ZERO, NC_ATTRIBUTE]:
            raise ValueError("Unexpected header.")
        if header == ZERO:
            more = self.fp.read(4)
            assert more == ZERO
            return {}
            
        count = self._unpack_int()

        attributes = {}
        for attr in range(count):
            name = asstr(self._unpack_string())
            attributes[name] = self._read_values()
        return attributes

    def _read_var_array(self):
        header = self.fp.read(4)
        if not header in [ZERO, NC_VARIABLE]:
            raise ValueError("Unexpected header.")

        if header == ZERO:
            more = self.fp.read(4)
            assert more == ZERO
            return            
        
        records = 0
        dtypes = {'names': [], 'formats': []}
        rec_vars = []
        count = self._unpack_int()
        for var in range(count):
            name, dimensions, shape, attributes, type, start, vsize = self._read_var()
            # http://www.unidata.ucar.edu/software/netcdf/docs/netcdf.html
            # Note that vsize is the product of the dimension lengths
            # (omitting the record dimension) and the number of bytes
            # per value (determined from the type), increased to the
            # next multiple of 4, for each variable. If a record
            # variable, this is the amount of space per record. The
            # netCDF "record size" is calculated as the sum of the
            # vsize's of all the record variables.
            #
            # The vsize field is actually redundant, because its value
            # may be computed from other information in the header. The
            # 32-bit vsize field is not large enough to contain the size
            # of variables that require more than 2^32 - 4 bytes, so
            # 2^32 - 1 is used in the vsize field for such variables.
            if shape and shape[0] is None:  # record variable
                rec_vars.append(name)
                # The netCDF "record size" is calculated as the sum of
                # the vsize's of all the record variables.
                self.__dict__['_recsize'] += vsize
                # Store the position where record arrays start.
                if records == 0:
                    records = start
                dtypes['names'].append(name)
                dtypes['formats'].append(str(shape[1:]) + '>' + type.char)

                # Handle padding with a virtual variable.
                if type.char in 'bch':
                    actual_size = reduce(mul, (1,) + shape[1:]) * type.itemsize
                    padding = -actual_size % 4
                    if padding:
                        dtypes['names'].append('_padding_%d' % var)
                        dtypes['formats'].append('(%d,)>b' % padding)

                # Data will be set later.
                data = None
            else:  # not a record variable
                # Calculate size to avoid problems with vsize (above)
                size = reduce(mul, shape, 1) * type.itemsize
                pos = self.fp.tell()
                if self.use_mmap:
                    if ALLOCATIONGRANULARITY:
                        pages = start // ALLOCATIONGRANULARITY
                        offset = pages * ALLOCATIONGRANULARITY
                        start = start % ALLOCATIONGRANULARITY
                        mm = mmap(self.fp.fileno(), start+size, access=ACCESS_READ, offset=offset)
                    else:
                        mm = mmap(self.fp.fileno(), start+size, access=ACCESS_READ)

                    data = ndarray.__new__(ndarray, shape, dtype=type,
                            buffer=mm, offset=start, order=0)
                else:
                    self.fp.seek(start)
                    data = fromstring(self.fp.read(size), type)
                    data.shape = shape
                self.fp.seek(pos)

            # Add variable.
            self.variables[name] = netcdf_variable(
                    data, type, shape, dimensions, attributes,
                    maskandscale=self.maskandscale)

        if rec_vars:
            dtypes['formats'] = [f.replace('()', '').replace(' ', '') for f in dtypes['formats']]
            # Remove padding when only one record variable.
            if len(rec_vars) == 1:
                dtypes['names'] = dtypes['names'][:1]
                dtypes['formats'] = dtypes['formats'][:1]

            # Build rec array.
            pos = self.fp.tell()
            if self.use_mmap:
                if ALLOCATIONGRANULARITY:
                    pages = records // ALLOCATIONGRANULARITY
                    offset = pages * ALLOCATIONGRANULARITY
                    records = records % ALLOCATIONGRANULARITY
                    mm = mmap(self.fp.fileno(), records+self._recs*self._recsize, access=ACCESS_READ, offset=offset)
                else:
                    mm = mmap(self.fp.fileno(), records+self._recs*self._recsize, access=ACCESS_READ)

                rec_array = ndarray.__new__(ndarray, (self._recs,), dtype=dtypes,
                        buffer=mm, offset=records, order=0)
            else:
                self.fp.seek(records)
                rec_array = fromstring(self.fp.read(self._recs*self._recsize), dtype=dtypes)
                rec_array.shape = (self._recs,)
            self.fp.seek(pos)

            for var in rec_vars:
                self.variables[var].__dict__['data'] = rec_array[var]

    def _read_var(self):
        name = asstr(self._unpack_string())
        dimensions = []
        shape = []
        dims = self._unpack_int()

        for i in range(dims):
            dimid = self._unpack_int()
            dimname = self._dims[dimid]
            dimensions.append(dimname)
            dim = self.dimensions[dimname]
            shape.append(dim)
        dimensions = tuple(dimensions)
        shape = tuple(shape)

        attributes = self._read_att_array()
        nc_type = self.fp.read(4)
        vsize = int(fromstring(self.fp.read(4), 'int32')[0])

        start = [self._unpack_int, self._unpack_int64][self.version_byte-1]()
        type = TYPEMAP(nc_type)

        return name, dimensions, shape, attributes, type, start, vsize

    def _read_values(self):
        nc_type = self.fp.read(4)
        n = self._unpack_int()

        type = TYPEMAP(nc_type)

        count = n*type.itemsize
        values = self.fp.read(int(count))
        self.fp.read(-count % 4)  # read padding

        if type.char not in ('S', 'a'):
            values = fromstring(values, type)
            if values.shape == (1,): values = values[0]
        else:
            ## text values are encoded via UTF-8, per NetCDF standard
            values = values.rstrip(asbytes('\x00')).decode('utf-8', 'replace')
        return values

    def _pack_begin(self, begin):
        if self.version_byte == 1:
            return self._pack_int(begin)
        elif self.version_byte == 2:
            return self._pack_int64(begin)

    def _pack_int(self, value):
        return array(value, '>i').tostring()
    _pack_int32 = _pack_int

    def _unpack_int(self):
        return int(fromstring(self.fp.read(4), '>i')[0])
    _unpack_int32 = _unpack_int

    def _pack_int64(self, value):
        return array(value, '>q').tostring()

    def _unpack_int64(self):
        return fromstring(self.fp.read(8), '>q')[0]

    def _pack_string(self, s):
        count = len(s)
        pad = asbytes('0') * (-count % 4)
        return self._pack_int(count) + asbytes(s) + pad

    def _unpack_string(self):
        count = self._unpack_int()
        s = self.fp.read(count).rstrip(asbytes('\x00'))
        self.fp.read(-count % 4)  # read padding
        return s


class netcdf_variable(object):
    """
    A data object for the `netcdf` module.

    `netcdf_variable` objects are constructed by calling the method
    `netcdf_file.createVariable` on the `netcdf_file` object. `netcdf_variable`
    objects behave much like array objects defined in numpy, except that their
    data resides in a file. Data is read by indexing and written by assigning
    to an indexed subset; the entire array can be accessed by the index ``[:]``
    or (for scalars) by using the methods `getValue` and `assignValue`.
    `netcdf_variable` objects also have attribute `shape` with the same meaning
    as for arrays, but the shape cannot be modified. There is another read-only
    attribute `dimensions`, whose value is the tuple of dimension names.

    All other attributes correspond to variable attributes defined in
    the NetCDF file. Variable attributes are created by assigning to an
    attribute of the `netcdf_variable` object.

    Parameters
    ----------
    data : array_like
        The data array that holds the values for the variable.
        Typically, this is initialized as empty, but with the proper shape.
    type: numpy dtype
        Desired data-type for the data array.
    shape : sequence of ints
        The shape of the array.  This should match the lengths of the
        variable's dimensions.
    dimensions : sequence of strings
        The names of the dimensions used by the variable.  Must be in the
        same order of the dimension lengths given by `shape`.
    attributes : dict, optional
        Attribute values (any type) keyed by string names.  These attributes
        become attributes for the netcdf_variable object.
    maskandscale: True or False
        Whether data is automagically scaled and masked.


    Attributes
    ----------
    dimensions : list of str
        List of names of dimensions used by the variable object.
    isrec, shape
        Properties

    See also
    --------
    isrec, shape

    """
    def __init__(self, data, type, shape, dimensions, attributes=None, maskandscale=False):
        self.data = data
        self.dtype = type
        self._shape = shape
        self.dimensions = dimensions
        self.maskandscale = maskandscale

        self._attributes = attributes or {}
        for k, v in self._attributes.items():
            self.__dict__[k] = v

    def __setattr__(self, attr, value):
        # Store user defined attributes in a separate dict,
        # so we can save them to file later.
        try:
            self._attributes[attr] = value
        except AttributeError:
            pass
        self.__dict__[attr] = value

    def isrec(self):
        """Returns whether the variable has a record dimension or not.

        A record dimension is a dimension along which additional data could be
        easily appended in the netcdf data structure without much rewriting of
        the data file. This attribute is a read-only property of the
        `netcdf_variable`.

        """
        return (hasattr(self.data, 'shape') and self.data.shape) and (not self._shape[0])
    isrec = property(isrec)

    def shape(self):
        """Returns the shape tuple of the data variable.

        This is a read-only attribute and can not be modified in the
        same manner of other numpy arrays.
        """
        return self.data.shape
    shape = property(shape)

    def getValue(self):
        """
        Retrieve a scalar value from a `netcdf_variable` of length one.

        Raises
        ------
        ValueError
            If the netcdf variable is an array of length greater than one,
            this exception will be raised.

        """
        return self.data.item()

    def assignValue(self, value):
        """
        Assign a scalar value to a `netcdf_variable` of length one.

        Parameters
        ----------
        value : scalar
            Scalar value (of compatible type) to assign to a length-one netcdf
            variable. This value will be written to file.

        Raises
        ------
        ValueError
            If the input is not a scalar, or if the destination is not a length-one
            netcdf variable.

        """
        if not self.data.flags.writeable:
            # Work-around for a bug in NumPy.  Calling itemset() on a read-only
            # memory-mapped array causes a seg. fault.
            # See NumPy ticket #1622, and SciPy ticket #1202.
            # This check for `writeable` can be removed when the oldest version
            # of numpy still supported by scipy contains the fix for #1622.
            raise RuntimeError("variable is not writeable")

        self.data.itemset(value)

    def typecode(self):
        """
        Return the typecode of the variable.

        Returns
        -------
        typecode : char
            The character typecode of the variable (eg, 'i' for int).

        """
        return self.dtype.char

    def itemsize(self):
        """
        Return the itemsize of the variable.

        Returns
        -------
        itemsize : int
            The element size of the variable (eg, 8 for float64).

        """
        return self.dtype.itemsize
    itemsize = property(itemsize)

    def __getitem__(self, index):
        # scalar
        if not self.shape:
            return self.data.item()
        if not self.maskandscale:
            return self.data[index]

        data = self.data[index].copy()
        missing_value = (
                self._attributes.get('missing_value') or
                self._attributes.get('_FillValue'))
        if missing_value is not None:
            data = np.ma.masked_values(data, missing_value)
        scale_factor = self._attributes.get('scale_factor')
        add_offset = self._attributes.get('add_offset')
        if add_offset is not None or scale_factor is not None:
            data = data.astype(np.float64)
        if scale_factor is not None:
            data = data * scale_factor
        if add_offset is not None:
            data += add_offset

        return data

    def __setitem__(self, index, data):
        if self.maskandscale:
            missing_value = (
                    self._attributes.get('missing_value') or
                    self._attributes.get('_FillValue') or
                    getattr(data, 'fill_value', 999999))
            self._attributes.setdefault('missing_value', missing_value)
            self._attributes.setdefault('_FillValue', missing_value)
            data = np.ma.asarray(data).filled(missing_value)
            data = (data - self._attributes.get('add_offset', 0.0)) / self._attributes.get('scale_factor', 1.0)

        # Expand data for record vars?
        if self.isrec:
            if isinstance(index, tuple):
                rec_index = index[0]
            else:
                rec_index = index
            if isinstance(rec_index, slice):
                recs = (rec_index.start or 0) + len(data)
            else:
                recs = rec_index + 1
            if recs > len(self.data):
                shape = (recs,) + self._shape[1:]
                self.data.resize(shape)
        self.data[index] = data

from collections import OrderedDict
class NcOrderedDict(OrderedDict):
    '''Store items in the order in which they will be written to the NetCDF file
       i.e. non-record variables first, followed by record variables'''
    def __setitem__(self, key, value):
        if key in self:
            OrderedDict.__setitem__(self, key, value)
        else:
            items = self.items() + [(key, value)] if len(self) > 0 else [(key, value)]
            reordered = sorted(items, key= lambda t: (t[1]._shape and not t[1].isrec, t[0]))
            reordered.reverse() # keep the original pupynere ordering for comparison
            for key in self.keys():
                del self[key]
            for key, val in reordered:
                OrderedDict.__setitem__(self, key, val)


def coroutine(func):
    '''A decorator that advances
    the execution to the first 'yield'
    in a generator so that this generator
    is "primed".
    '''
    def generator(*args, **kwargs):
        primed_func = func(*args, **kwargs)
        primed_func.next()
        return primed_func
    return generator

@coroutine
def byteorderer(target):
    ''' coroutine to be used in the pipeline with nc_streamer
        for example when instantiating the pipeline, for source-driven
        netcdf writing, do something like this:

Examples
--------

    >>> nc = netcdf_file(None)
    >>> # add attributes, dimensions and variables to the netcdf_file object
    >>> pipeline = byteorderer(nc_streamer(nc, nc_writer(filename)))
    >>> while True:
    >>>     pipeline.send(stuff)

    '''
    try:
        while True:
            data = (yield)
            if (data.dtype.byteorder == '<' or
                (data.dtype.byteorder == '=' and LITTLE_ENDIAN)):
                data = data.byteswap()
            target.send(data)
    except StopIteration:
        pass

def check_byteorder(input):
    ''' an iteration-based approach to byteorderer above
        input should be an iterable of numpy arrays with which to
        fill the netcdf file
    '''
    try:
        while True:
            data = input.next()
            if (data.dtype.byteorder == '<' or
                (data.dtype.byteorder == '=' and LITTLE_ENDIAN)):
                data = data.byteswap()
            yield data
    except StopIteration:
        pass

@coroutine
def nc_streamer(ncfile, target):
    ''' coroutine implementation for source-driven netcdf generation
        Parameters
        ----------
        ncfile : an instance of netcdf_file
        target : function which serves as a data "sink",
                 using yield expressions to consume data sent to it

        Examples
        --------
    >>> nc = netcdf_file(None)
    >>> # add attributes, dimensions and variables to the netcdf_file object
    >>> pipeline = byteorderer(nc_streamer(nc, nc_writer(filename)))
    >>> while True:
    >>>     pipeline.send(stuff)

    '''
    assert type(ncfile) == netcdf_file
    count = 0
    ncfile._calc_begins()
    target.send(ncfile._header())
    count += len(ncfile._header())
    try:
        if ncfile.variables:
            for name, var in ncfile.non_recvars.items():
                end = var._begin + var._vsize if var.dimensions else var._begin
                while count < end:
                    data = (yield)
                    bytes = data.tostring()

                    count += len(bytes)
                    # padding
                    if (end - count < data.itemsize):
                        bytes += asbytes('0') * (end - count)
                        count = end
                    target.send(bytes)

            # Record variables... keep taking data until it stops coming
            while True:
                vars = ncfile.recvars.values()
                for i in range(ncfile._recs):
                    for var in vars:
                        data = (yield)
                        bytes = data.tostring()
                        target.send(bytes)
                        padding = len(bytes) % 4
                        if padding:
                            yield asbytes('0') * padding
                        count += len(bytes) + padding

    except StopIteration:
        pass

def nc_generator(ncfile, input):
    ''' an iteration-based approach to nc_streamer above
        input should be an iterable of numpy arrays with which to
        fill the netcdf file

        Examples
        --------
    >>> from itertools import chain
    >>> import numpy
    >>> nc = netcdf_file(None)
    >>> # add attributes, dimensions and variables to the netcdf_file object
    >>> def input():
    >>>     yield numpy.random(100, 100)
    >>> def more_input():
    >>>     yield numpy.arange(10000).reshape(100, 100)
    >>> pipeline = nc_generator(nc, chain(input, more_input))
    >>> f = open('foo.nc', 'w')
    >>> for block in pipeline:
    >>>     f.write(block)

    '''
    input = check_byteorder(input)
    assert type(ncfile) == netcdf_file
    count = 0
    ncfile._calc_begins()
    yield ncfile._header()
    count += len(ncfile._header())

    try:
        if ncfile.variables and ncfile.non_recvars:
            for name, var in ncfile.non_recvars.items():
                end = var._begin + var._vsize if var.dimensions else var._begin
                while count < end:
                    data = input.next()

                    bytes = data.tostring()
                    count += len(bytes)
                    # padding
                    if (end - count < data.itemsize):
                        bytes += asbytes('0') * (end - count)
                        count = end
                    yield bytes

        # Record variables... keep taking data until it stops coming (i.e. a StopIteration is raised)
        if ncfile.variables and ncfile.recvars:
            while True:
                vars = ncfile.recvars.values()
                while True:
                    for var in vars:
                        data = input.next()

                        bytes = data.tostring()
                        yield bytes
                        padding = len(bytes) % 4
                        if padding:
                            yield asbytes('0') * padding
                        count += len(bytes) + padding

    except StopIteration:
        pass

@coroutine
def nc_writer(filename):
    with open(filename, 'w') as f:
        while True:
            bytes = (yield)
            f.write(bytes)

NetCDFFile = netcdf_file
NetCDFVariable = netcdf_variable
