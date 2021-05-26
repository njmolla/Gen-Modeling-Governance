from GM_code import run_system
from sample_setup import sample_composition
import numpy as np
from mpi4py import MPI
import sys
import pickle
import pandas as pd

comm = MPI.COMM_WORLD

num_processors = 160

print(comm.rank)

if comm.size != num_processors:
  print('ERROR running on %d processors' % comm.size)
  sys.exit()

def run_cm_samples(size,C,num_samples,filename):
  '''
  Run num_samples samples and return total connectance and stability data
  '''
  # record information for the correlation experiment
  data = pd.DataFrame(columns = ['Total connectance','stability','converged'])
  stability_list = []
  connectance_list = []
  convergence_list = []

  for i in range(num_samples):
    # --------------------------------------------------------------------------
    # Set up system composition if not specified
    # --------------------------------------------------------------------------
    np.random.seed(i)
    N,N1,N2,N3,K,M,T = sample_composition(size)
    print((N1,N2,N3,K,M))
    try:
      result = run_system(N1,N2,N3,K,M,T,C,sample_exp=False)
    except Exception as e:
      print(e)
      continue
    stability = result[0] # stability is the first return value
    converged = result[2]
    connectance = result[5]
    stability_list.append(stability)
    connectance_list.append(connectance)
    convergence_list.append(converged)

    print('adding data')
    data['stability'] = stability_list
    data['Total_connectance'] = connectance_list
    with open(filename, 'wb') as f:
      pickle.dump(data, f)


size_ranges = np.arange(5,20,1)
connectance_ranges = np.linspace(0.1,0.8,32)
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
final_connectances = np.zeros(int(cells_per_processor))

for i, connectance in enumerate(connectances):
  run_cm_samples(size,connectance,num_samples,'data_%s'%(comm.rank))

