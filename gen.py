import cgen as c
import cgen.cuda as cc

class RPointer(c.Pointer):
    def get_decl_pair(self):
        sub_tp, sub_decl = self.subdecl.get_decl_pair()
        return sub_tp, ("* __restricted__ %s" % sub_decl)

class Wrapper(object):
    
    def __init__(self, horizon):
        self.horizon = horizon

    def generate(self, gpu=False):

        h = self.horizon

        fndecl = c.FunctionDeclaration(c.Value('int', 'wrap'), [c.Value('int', 'i')])
        if gpu:
            fndecl = cc.CudaDevice(fndecl)
        fndecl = c.DeclSpecifier(fndecl, 'inline')

        body = [c.If('i>=0', c.Block([c.Statement('return i %% %d' % h)]),
                    c.Block([c.If('i == - $horizon', 
                                c.Block([c.Statement('return 0')]),
                                c.Block([c.Statement('%d + (i %% %d)' % (h, h))]))]))]

        for line in c.FunctionBody(fndecl, c.Block(body)).generate():
            yield line
    
class Step(object):

    def __init__(self, n, nsv):
        self.n = n
        self.nsv = nsv

    def generate(self, gpu=False):
        dtype = 'float' if gpu else 'double'

        fnargs = [c.Value('int', 'step'),
                  RPointer(c.Value('int', 'idel'))]\
               + [RPointer(c.Value('float', arg))
                  for arg in ['hist', 'conn', 'X', 'gsc', 'exc']]

        fndecl = c.FunctionDeclaration(c.Value('void', 'step'), fnargs)

        if gpu:
            fndecl = cc.CudaGlobal(fndecl)

        body = []

        if gpu:
            body += [c.Initializer(c.Value('int', 'parij'),
                                   "blockDim.x*blockIdx.x + threadIdx.x"),
                     c.Initializer(c.Value('int', 'nthr'),
                                   "blockDim.x*gridDim.x")]

        body += [c.Value('int', 'hist_idx'), c.Value(dtype, 'input'),
                 c.Value('int', 'i'), c.Value('int', 'j')]

        hist_idx = '%d*nthr*%s + nthr*j + parij' if gpu else '%d*%s + j'

        # unroll inner loop 
        body.append(c.For('i=0', 'i<%d' % self.n, 'i++', c.Block([ 
            c.Assign('input', '0.0'),
            c.For('j=0', 'j<%d' % self.n, 'j++, idel++, conn++', c.Block([
                c.Assign('hist_idx', hist_idx % (self.n, 'wrap(step - 1 - *idel)')),
                c.Statement('input += (*conn)*hist[hist_idx]')
            ]))
        ])))

        model_args = ['X + %s*i' % (('nthr*%d' if gpu else '%d')%self.nsv, ),
                      'exc' + (' + parij' if gpu else ''),
                      'gsc%s*input/%d' % ('[parij]' if gpu else '', self.n)]

        if gpu:
            model_args += ['nthr', 'parij']

        body.append(c.Statement('model(%s)' % (', '.join(model_args), )))

        for line in c.FunctionBody(fndecl, c.Block(body)).generate():
            yield line


class Model(object):

    def __init__(self, eqns, pars):
        self.eqns = eqns
        self.pars = pars

    def generate(self, dt=0.1, gpu=False):
        dtype = 'float' if gpu else 'double'

        # arg names: X, pars, n_thr, par_ij, input

        X = c.Value(dtype, 'X')
        input = c.Value(dtype, 'input')

        args = []
        args.append(RPointer(X))
        args.append(RPointer(c.Value('void', 'pars')))
        args.append(input)

        if gpu:
            nthr = c.Value('int', 'nthr')
            parij = c.Value('int', 'parij')
            args.append(nthr)
            args.append(parij)

        Xrefs = [("X[ntr*%d + parij]" if gpu else "X[%d]") % i 
                    for i in range(len(self.eqns))]

        body = []

        for i, p in enumerate(self.pars):
            pval = c.Value(dtype, p)
            body.append(c.Initializer(c.Value(dtype, p), 
                                       "((%s*) pars)[%d]" % (dtype, i)))

        for i, var in enumerate(self.eqns):
            var, _ = var
            body.append(c.Initializer(c.Value(dtype, var), Xrefs[i]))

        for var, deriv in self.eqns:
            body.append(c.Initializer(c.Value(dtype, 'd'+var), deriv))
            
        for i, var in enumerate(self.eqns):
            var = var[0]
            body.append(c.Assign(Xrefs[i], '%s + %f*d%s' % (var, dt, var)))
                    

        fndecl = c.FunctionDeclaration(c.Value('void', 'fn'), args)
        if gpu:
            fndecl = cc.CudaDevice(fndecl)
        fndecl = c.DeclSpecifier(fndecl, 'inline')

        for line in c.FunctionBody(fndecl, c.Block(body)).generate():
            yield line


fhn = Model(
    eqns=[('x', '(x - x*x*x/3.0 + y)*3.0/5.0'),
          ('y', '(a - x)/3.0/5.0 + input')],
    pars = ['a']
)

pitch = Model(
    eqns=[('x', '(x - x*x*x/3.0)/5.0 + lambda')],
    pars=['lambda']
)


