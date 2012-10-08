import enthought.mayavi.mlab as m
import numpy

r = numpy.load('results.npy')

m.figure()
m.pipeline.volume(m.pipeline.scalar_field(r[0, 0]))
m.show()


