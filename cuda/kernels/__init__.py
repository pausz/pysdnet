
"""
We could have even more fun with this, jbwc, monkey patching the import
system to allow, e.g.

    >>> import pysdnet.cuda.kernels as ks
    >>> from ks.parsweep import kernel as parsweep
    >>> parsweep(*args, **launch_conf)

"""

