from GM_code import run_multiple
import numpy as np
from mpi4py import MPI
import sys
import pickle

comm = MPI.COMM_WORLD

num_processors = 160

print(comm.rank)

if comm.size != num_processors:
  print('ERROR running on %d processors' % comm.size)
  sys.exit()

size_ranges = np.arange(5,15,1)
connectance_ranges = np.linspace(0.01,0.6,32)
num_samples = 200

cells_per_row = len(connectance_ranges)
cells_per_processor = len(size_ranges)*cells_per_row/num_processors
processors_per_row = cells_per_row/cells_per_processor

processor_num = comm.rank
size = size_ranges[int(processor_num//processors_per_row)] # number of the row
start_index = int(cells_per_processor*(processor_num%processors_per_row))
connectances = connectance_ranges[start_index:int(start_index+cells_per_processor)]

np.random.seed(comm.rank+666)

num_stable_webs = np.zeros(int(cells_per_processor))
num_converged = np.zeros(int(cells_per_processor))

for i, connectance in enumerate(connectances):
  num_stable_webs[i], num_converged[i] = run_multiple(size,connectance,num_samples)

with open('PSW_%s'%(comm.rank), 'wb') as f:
  pickle.dump(num_stable_webs, f)

with open('convergence_%s'%(comm.rank), 'wb') as f2:
  pickle.dump(num_converged, f2)
