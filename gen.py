import cgen as c
import cgen.cuda as cc

class RPointer(c.Pointer):
    def get_decl_pair(self):
        sub_tp, sub_decl = self.subdecl.get_decl_pair()
        return sub_tp, ("* __restricted__ %s" % sub_decl)
    

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

