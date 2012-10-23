"""
The gen module implements code generation for integrating a
system's equations in different execution contexts and strategies.

    - CPU
    - CUDA

can envision as well, where sensible, use of 

    - CPU w/ OpenMP
    - OpenCL

Below, functions exist emitting various code:

    - wrap - implements Python's % op in C
    - step - advance integration one step
    - model - differential equations for a particular node model

Going forward, 

    - Generalize coupling as f(x_post(t), x_pre(t-tau)), insert into
        step's inner loop over delayed step
    - Generalize integration scheme
    - Add noise
    - Add flexible monitor components
    - Refactor into better pieces

"""

from cgen import *
from cgen.cuda import *


class RPointer(Pointer):
    c99 = True
    def get_decl_pair(self):
        sub_tp, sub_decl = self.subdecl.get_decl_pair()
        restrict = " restrict " if self.c99 else " __restricted__ "
        return sub_tp, "*%s%s" % (restrict, sub_decl)

def wrap(horizon, gpu=False, inline=True):
    """
    Emits a function that behaves like Python's modulo operator.
    """

    h = horizon

    fndecl = FunctionDeclaration(Value('int', 'wrap'), [Value('int', 'i')])
    if gpu:
        fndecl = CudaDevice(fndecl)
    if inline:
        fndecl = DeclSpecifier(fndecl, 'inline')

    body = [If('i>=0', Block([Statement('return i %% %d' % h)]),
                Block([If('i == - %d' % h, 
                            Block([Statement('return 0')]),
                            Block([Statement('return %d + (i %% %d)' % (h, h))]))]))]

    return FunctionBody(fndecl, Block(body))

def model(eqns, pars, name='model', dt=0.1, gpu=False, inline=True):

    dtype = 'float' if gpu else 'double'

    args = [RPointer(Value(dtype, 'X')), RPointer(Value('void', 'pars')), Value(dtype, 'input')]\
         + ([Value('int', 'nthr'),Value('int', 'parij')] if gpu else [])

    fndecl = FunctionDeclaration(Value('void', name), args)
    if gpu:
        fndecl = CudaDevice(fndecl)
    if inline:
        fndecl = DeclSpecifier(fndecl, 'inline')

    Xrefs = [("X[ntr*%d + parij]" if gpu else "X[%d]") % i for i in range(len(eqns))]

    body = [Initializer(Value(dtype, p), "((%s*) pars)[%d]" % (dtype, i)) for i, p in enumerate(pars)]\
         + [Initializer(Value(dtype, eqn[0]), Xref) for Xref, eqn in zip(Xrefs, eqns)]\
         + [Initializer(Value(dtype, 'd'+var), deriv) for var, deriv in eqns]\
         + [Assign(Xref, '%s + %f*d%s' % (eqn[0], dt, var)) for Xref, eqn in zip(Xrefs, eqns)]

    return FunctionBody(fndecl, Block(body))


def step(n, nsv, cvar=0, gpu=False, nunroll=1, model='model'):
    """
    Emits a function that advances the system one step.
    """

    dtype = 'float' if gpu else 'double'

    fndecl = FunctionDeclaration(Value('void', 'step'),
               [Value('int', 'step'), RPointer(Value('int', 'idel'))]\
             + [RPointer(Value(dtype, arg)) for arg in ['hist', 'conn', 'X', 'gsc', 'exc']])

    if gpu:
        fndecl = CudaGlobal(fndecl)

    body = []

    if gpu:
        body += [Initializer(Value('int', 'parij'), "blockDim.x*blockIdx.x + threadIdx.x"),
                 Initializer(Value('int', 'nthr'),  "blockDim.x*gridDim.x")]

    body += [Value('int', 'hist_idx'), Value(dtype, 'input'),
             Value('int', 'i'), Value('int', 'j')]

    hist_idx = '%d*nthr*%s + nthr*j + parij' if gpu else '%d*%s + j'

    inner_loop_body = [Assign('hist_idx', hist_idx % (n, 'wrap(step - 1 - *idel)')),
                       Statement('input += (*conn)*hist[hist_idx]')]\
                    + [Statement(v+'++') for v in ['j', 'idel', 'conn']]

    update_loop = [
        Assign('hist[' + ('nthr*%d*wrap(step) + nthr*i + parij'%n if gpu else '%d*wrap(step) + i'%n) + ']', 
               'X[' + ('%d*nthr*i + nthr*%d + parij' if gpu else '%d*i + %d')%(nsv, cvar) + ']'),
        Statement('i++')
    ]

    model_args = ['X + %s*i' % (('nthr*%d' if gpu else '%d')%nsv, ),
                  'exc' + (' + parij' if gpu else ''),
                  '%s*input/%d' % ('gsc[parij]' if gpu else '(*gsc)', n)]\
               + (['nthr', 'parij'] if gpu else [])

    body += [For('i=0', 'i<%d' % n, 'i++', Block([ 
        Assign('input', '0.0'),
        For('j=0', 'j<%d' % (n - n%nunroll,), '', Block(inner_loop_body*nunroll))
    ] + inner_loop_body*(n%nunroll) + [Statement('%s(%s)' % (model, ', '.join(model_args), ))]
    )),

    # I thought this needed to be in a separate CUDA kernel to synch correctly, but I'm an idiot
    For('i=0', 'i<%d' % (n - n%nunroll), '', Block(update_loop*nunroll))
    ] + update_loop * (n%nunroll)

    return FunctionBody(fndecl, Block(body))


def module(model, step, wrap, gpu=False):
    return Module([wrap, model, step])

# put this into descr.py or somewhere else
fhn = dict(
    name='fhn',
    eqns=[('x', '(x - x*x*x/3.0 + y)*3.0/5.0'),
          ('y', '(a - x)/3.0/5.0 + input')],
    pars = ['a']
)

pitch = dict(
    name='pitch',
    eqns=[('x', '(x - x*x*x/3.0)/5.0 + lambda + input')],
    pars=['lambda']
)

    

