from GM_code import run_system
from sample_setup import sample_composition
import numpy as np
from mpi4py import MPI
import sys
import pickle
import pandas as pd

comm = MPI.COMM_WORLD

num_processors = 96

print(comm.rank)

if comm.size != num_processors:
  print('ERROR running on %d processors' % comm.size)
  sys.exit()

def run_cm_samples(size,C,num_samples,stabilities,total_connectances,convergences):
  '''
  Run num_samples samples and return total connectance and stability data
  '''
  for i in range(num_samples):
    N,N1,N2,N3,K,M,T = sample_composition(size)
    print((N1,N2,N3,K,M,T,C))
    try:
      result = run_system(N1,N2,N3,K,M,T,C,sample_exp=False)
    except Exception as e:
      print(e)
      continue
    stability = result[0] # stability is the first return value
    converged = result[2]
    connectance = result[5]
    stabilities.append(stability)
    total_connectances.append(connectance)
    convergences.append(converged)


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

# dataframe and lists to store results from each processor
data = pd.DataFrame(columns = ['Total_connectance','size','stability','converged'])
data['size'] = np.ones(num_samples*len(connectances))*size

stabilities = []
total_connectances = []
convergences = []

for i, connectance in enumerate(connectances):
  run_cm_samples(size,connectance,num_samples,stabilities,total_connectances,convergences)


print('adding data')
data['stability'] = stabilities
data['Total_connectance'] = total_connectances
data['converged'] = convergences

with open('data_%s'%(processor_num), 'wb') as f:
  pickle.dump(data, f)
