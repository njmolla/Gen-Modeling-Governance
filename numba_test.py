from numba import jit
import numpy as np
import time

x = np.arange(1000).reshape(10, 10, 10)

"""
This does
   lhs[conditions] = rhs[conditions]
lhs and rhs are any numpy arrays with the same shape
conditions is a boolean numpy array with the same shape

For example, to do
   x[x > 0] = y[x > 0]
use
    assign_when(x, y, x > 0)
"""
@jit(nopython=True)
def assign_when(lhs, rhs, conditions):
  for nd_index in np.ndindex(lhs.shape):
    if conditions[nd_index]:
      lhs[nd_index] = rhs[nd_index]


"""
This does
   lhs[conditions] = rhs
lhs is a numpy array, rhs is a single value
conditions is a boolean numpy array with the same shape as lhs

For example, to do
   x[x > 0] = c
use
    assign_when(x, c, x > 0)
"""
@jit(nopython=True)
def assign_scalar_when(lhs, rhs, conditions):
  for nd_index in np.ndindex(lhs.shape):
    if conditions[nd_index]:
      lhs[nd_index] = rhs


# Non-jit version
def go_fast(a): # Function is compiled and runs in machine code
  b = 2
  product = a*b
#  product[product>5000] = 1000
  assign_scalar_when(product, 1000, product > 5000)
#    trace = 0.0
#    for i in range(a.shape[0]):
#        trace += np.tanh(a[i, i])
  return product

# Non-jit version
start = time.time()
go_fast(x)
end = time.time()
print("Elapsed (without using jit) = %s" % (end - start))


@jit(nopython=True)
def go_fast(a): # Function is compiled and runs in machine code
  b = 2
  product = a*b
#  product[product>5000] = 1000
  assign_scalar_when(product, 1000, product > 5000)
  np.concatenate((np.ones((2,2)),np.zeros((2,2))))
  np.sum(product)
  #np.diag(product)
#    trace = 0.0
  for i in range(a.shape[0]):
    a[i,:,i] = 3
  return product

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
go_fast(x)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
go_fast(x)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))
