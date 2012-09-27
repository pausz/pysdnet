
"""
We could have even more fun with this, jbwc, monkey patching the import
system to allow, e.g.

    >>> import pysdnet.cuda.kernels as ks
    >>> from ks.parsweep import kernel as parsweep
    >>> parsweep(*args, **launch_conf)

"""

import os
import string

from pycuda.compiler import SourceModule

here = os.path.dirname(os.path.abspath(__file__))

class srcmod(object):
    
    def __init__(self, name, fns, _debug=False, **kwds):
        
        with open(here + os.path.sep + name, 'r') as fd:
            self.src = string.Template(fd.read())

        final_src = self.src.substitute(**kwds)

        if _debug:
            print "srcmod: final source for %s is \n%s" % (name, final_src)

        self._module = SourceModule(final_src)

        for f in fns:
            setattr(self, f, self._module.get_function(f))


