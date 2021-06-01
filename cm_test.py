from GM_code import run_system
from sample_setup import sample_composition
import numpy as np
import pickle
import pandas as pd


num_processors = 96

def run_cm_samples(size,C,num_samples,filename):
  '''
  Run num_samples samples and return total connectance and stability data
  '''
  data = pd.DataFrame(columns = ['Total connectance','size','stability','converged'])
  data = pd.DataFrame(columns = ['Total_connectance','size','stability','converged'])
  data['size'] = np.ones(num_samples)*size
  data['Total_connectance'] = np.zeros(num_samples)
  data['stability'] = np.zeros(num_samples)
  data['converged'] = np.zeros(num_samples)
  stability_list = []
  connectance_list = []
  convergence_list = []
  data['size'] = np.ones(num_samples)*size
  for i in range(num_samples):

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
    print(stability_list)
  print('adding data')
  data['stability'] = stability_list
  data['Total_connectance'] = connectance_list
  with open(filename, 'wb') as f:
    pickle.dump(data, f)

processor_num = 0
size_ranges = np.arange(5,20,1)
connectance_ranges = np.linspace(0.1,0.8,32)
num_samples = 1

cells_per_row = len(connectance_ranges)
cells_per_processor = len(size_ranges)*cells_per_row/num_processors
processors_per_row = cells_per_row/cells_per_processor

size = size_ranges[int(processor_num//processors_per_row)] # number of the row
start_index = int(cells_per_processor*(processor_num%processors_per_row))
connectances = connectance_ranges[start_index:int(start_index+cells_per_processor)]

np.random.seed(processor_num+666)

num_stable_webs = np.zeros(int(cells_per_processor))
num_converged = np.zeros(int(cells_per_processor))
final_connectances = np.zeros(int(cells_per_processor))

for i, connectance in enumerate(connectances):
  run_cm_samples(size,connectance,num_samples,'data_%s'%(processor_num))

