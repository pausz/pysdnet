
import os
import string

from pycuda.compiler import SourceModule

here = os.path.dirname(os.path.abspath(__file__))

class srcmod(object):
    
    def __init__(self, name, fns, **kwds):
        
        with open(here + os.path.sep + name, 'r') as fd:
            self.src = string.Template(fd.read())

        self._module = SourceModule(self.src.substitute(**kwds))

        for f in fns:
            setattr(self, f, self._module.get_function(f))

