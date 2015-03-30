from itertools import chain
from pupynere import netcdf_file, nc_generator
import numpy

from pdb import set_trace

n1dim = 4
n2dim = 2
n3dim = 2
arr = numpy.arange(n1dim*n2dim*n3dim).reshape(n1dim, n2dim, n3dim)
arr2 = numpy.random.rand(n1dim, n2dim, n3dim)
# arr2 = numpy.arange(n1dim*n2dim*n3dim, 2*n1dim*n2dim*n3dim).reshape(n1dim, n2dim, n3dim)
print arr.shape

nc = netcdf_file(None)
# add attributes, dimensions and variables to the netcdf_file object
nc.createDimension('n1', None)
nc.createDimension('n2', n2dim)
nc.createDimension('n3', n3dim)
v1 = nc.createVariable('data1', arr.dtype.str[1:], ('n1','n2','n3'))
#set_trace()

v1[:] = arr

print arr

# def i():
#     yield numpy.random.rand(n1dim, n2dim, n3dim)

# def i2():
#     yield numpy.arange(n1dim*n2dim*n3dim)

#print nc.recvars.items()
#print nc.non_recvars.items()

for i in chain(arr):
    print i

pipeline = nc_generator(nc, iter(arr))
# pipeline = nc_generator(nc, i())
with open('foo.nc', 'w') as f:
    for block in pipeline:
        f.write(block)

for i in chain(arr2):
    print i

pipeline = nc_generator(nc, iter(arr2))
# pipeline = nc_generator(nc, i())
with open('foo2.nc', 'w') as f:
    for block in pipeline:
        f.write(block)
