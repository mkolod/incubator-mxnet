import mxnet as mx
from test_tensorrt_lenet5 import *

def detect_cycle_from(sym, visited, stack):
  visited.add(sym.handle.value)
  stack.add(sym.handle.value)
  for s in s.get_children():
    if s.handle.value not in visited:
      if detect_cycle_from(sym, visited, stack):
        return True
    elif s.handle.value in stack:
      return True
  stack.remove(sym.handle.value)
  return False

def detect_cycle(sym):
  visited = set()
  stack = set()
  all_nodes = sym.get_internals()
  for s in all_nodes:
    if s.handle.value in visited:
      if detect_cycle_from(s, visited, stack):
        return True
  return False

def test_simple_cycle():
  inp = mx.sym.Variable('input', shape=[1,10])
  A = mx.sym.FullyConnected(data=inp, num_hidden=10, no_bias=True, name='A')
  B = mx.sym.FullyConnected(data=A, num_hidden=10, no_bias=True, name='B')
  D = mx.sym.sin(data=A, name='D')
  C = mx.sym.elemwise_add(lhs=B, rhs=D, name='C')
  arg_params = {'A_weight': mx.nd.zeros([10,10]), 'B_weight': mx.nd.zeros([10,10])}
  set_use_tensorrt(True)
  executor = C.simple_bind(ctx=mx.gpu(0), data=(1,10), softmax_label=(1,),
                           shared_buffer=arg_params, grad_req='null', force_rebind=True)


if __name__ == '__main__':
  test_simple_cycle()
