
import glob, numpy, collections, os

magic_path = os.path.dirname(os.path.abspath(__file__))

w8s = glob.glob('%s/*Capacity*.csv' % (magic_path,))
dxs = glob.glob('%s/*Distance*.csv' % (magic_path,))
mni = magic_path + '/mni.csv'

load_args = {'mni': dict(usecols=(2, 3, 4), skiprows=1, delimiter=','),
             'w8s': dict(skiprows=1, delimiter=','),
             'dxs': dict(delimiter=',', skiprows=1)}

def list_capacities():
    for w8_path in w8s:
        print w8_path

def list_distances():
    for dx_path in dxs:
        print dx_path

def ls():
    list_capacities()

def find_file(file_list, idx):
    if type(idx) in (str, unicode):
        for cap_file in file_list:
            if idx in cap_file:
                return cap_file
    else:
        return file_list[idx]

def load_capacity(idx, m1_to_0=True):
    mat = numpy.loadtxt(find_file(w8s, idx), **load_args['w8s'])
    if m1_to_0:
        mat[mat==-1.0] = 0.0
    return mat

def load_distance(idx, m1_to_0=True):
    mat = numpy.loadtxt(find_file(dxs, idx), **load_args['dxs'])
    if m1_to_0:
        mat[mat==-1.0] = 0.0
    return mat

def load_mni():
    return numpy.loadtxt(mni, **load_args['mni'])

def load_roi_labels():
    with open(mni, 'r') as fd:
        lines = fd.readlines()
    # first line gives column names
    return [line.split(',')[0] for line in lines[1:]]


Dataset = collections.namedtuple('Dataset', 'weights distances centers')

def load_dataset(idx):
    return Dataset(load_capacity(idx),
                   load_distance(idx),
                   load_mni())

def ndarray_json(nda):
    return {"data": list(nda.flat),
            "shape": nda.shape,
            "strides": map(lambda x: x/nda.itemsize, nda.strides)}

def dataset_json(data, filename=None):
    import json

    if type(data) is not Dataset:
        data = load_dataset(data)

    data = {"weights": ndarray_json(data.weights),
            "distances": ndarray_json(data.distances),
            "centers": ndarray_json(data.centers)}

    res = ''
    
    if filename:
        with open(filename, 'w') as fd:
            json.dump(data, fd)
    else:
        res = json.dumps(data)

    return res
