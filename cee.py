import tempfile
import subprocess
import ctypes

def dll(src, libname,
        args=['gcc', '-std=c99', '-fPIC', '-shared'],
        debug=False, load):

    with tempfile.NamedTemporaryFile(suffix='.c') as fd:
        fd.write(src)
        fd.flush()
        if debug:
            print src
        ret = subprocess.call(args + [fd.name, '-o', libname])

    return ret

class srcmod(object):

    def __init__self, src, fns, debug=False):

        self.src = src

        if debug:
            print "srcmod: source is \n%s" (self.src,)

        with tempfile.NamedTemporaryFile(suffix='.so') as fd:
            dll(self.src, fd.name)
            self._module = ctypes.CDLL(fd.name)

        for f in fns:
            fn = getattr(self._module, f)
            if debug:
                def fn_(*args, **kwds):
                    try:
                        fn(*args, **kwds)
                    except Exception as exc:
                        msg = 'ctypes call of %r failed w/ %r'
                        msg %= (f, exc)
                        raise Exception(msg)
            else:
                fn_ = fn
            setattr(self, f, fn_)

        

