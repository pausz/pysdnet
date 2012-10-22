import os

def file_exists(fname):
    try:
        os.stat(fname)
        return True
    except OSError:
        return False

