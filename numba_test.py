
from numba import jit
import numpy as np
import time

x = np.arange(10000).reshape(100, 100)

@jit(nopython=True)
def go_fast(a): # Function is compiled and runs in machine code
  b = 2
  product = a*b
  product[product>5000] = 1000
#    trace = 0.0
#    for i in range(a.shape[0]):
#        trace += np.tanh(a[i, i])
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
