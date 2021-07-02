from GM_code import run_system
from sample_setup import sample_composition
import numpy as np
from mpi4py import MPI
import sys
import pickle
import pandas as pd

comm = MPI.COMM_WORLD

def run_cm_samples(size,C,num_samples,stabilities,total_connectances,convergences, composition = np.array([None,None,None,None,None,None]),sample = True):
  '''
  Run num_samples samples and return total connectance and stability data
  '''
  for i in range(num_samples):
    if sample == True:
      N,N1,N2,N3,K,M,T = sample_composition(size, partial_composition = composition)
    else:
      N,N1,N2,N3,K,M = tuple(composition)
      T = sum((N,M,K)) + 1
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

#--------------------------------------------------------------------------------------
# Initial colormap experiment (sizes and connectances)
# -------------------------------------------------------------------------------------
num_processors = comm.size
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

np.random.seed(comm.rank+766)

# dataframe and lists to store results from each processor
data = pd.DataFrame(columns = ['Total_connectance','size','stability','converged'])
data['size'] = np.ones(num_samples*len(connectances))*size

stabilities = []
total_connectances = []
partial_conn = []
convergences = []

for i, connectance in enumerate(connectances):
  run_cm_samples(size,connectance,num_samples,stabilities,total_connectances,convergences)
  partial_conn.append(list([connectance])*num_samples)


data['stability'] = stabilities
data['Total_connectance'] = total_connectances
data['converged'] = convergences
data['connectance'] = partial_conn

with open('data_%s'%(processor_num), 'wb') as f:
 pickle.dump(data, f)

#--------------------------------------------------------------------------------------
# Exploring proportions
# -------------------------------------------------------------------------------------
# Total_size = 10
# sizes = np.arange(2,9,1)
# connectances = np.linspace(0.1,0.8,10)
# num_samples = 2

# # dataframe and lists to store results from each processor
# data = pd.DataFrame(columns = ['Total_connectance','size','stability','converged'])

# stabilities = []
# total_connectances = []
# convergences = []
# size_list = []

# for size in sizes:
#   for connectance in connectances:
#       size_list = size_list + [size]*num_samples
#       run_cm_samples(Total_size,connectance,num_samples,stabilities,total_connectances,convergences, composition = np.array([size,None,None,None,None,None]))

# data['stability'] = stabilities
# data['Total_connectance'] = total_connectances
# data['converged'] = convergences
# data['size'] = size_list

# with open('data', 'wb') as f:
#   pickle.dump(data, f)

#--------------------------------------------------------------------------------------
# Simplex experiment (overall composition)
# -------------------------------------------------------------------------------------

#np.random.seed(comm.rank+866)
#
#total_size = 10
#N_min = 2
#M_min = 1
#C = 0.4
#num_samples = 300
#points = []
#
#for i in range(total_size - N_min - M_min + 1):
#  for j in range(total_size - N_min - M_min - i + 1):
#    N = i + N_min
#    M = j + M_min
#    K = total_size - N - M
#    points.append((N,M,K))
#
#
#if comm.size != len(points):
#  print('ERROR running on %d processors' % comm.size)
#  print(len(points))
#  sys.exit()
#
#point = points[comm.rank]
#N,M,K = point
#
#data = pd.DataFrame(columns = ['N','M','K','stability'])
#data['N'] = np.ones(num_samples)*N
#data['M'] = np.ones(num_samples)*M
#data['K'] = np.ones(num_samples)*K
#
#stabilities = []
#total_connectances = []
#convergences = []
#
#composition = np.array([N,None,None,None,K,M])
#
#run_cm_samples(total_size,C,num_samples,stabilities,total_connectances,convergences, composition = composition)
#
#data['stability'] = stabilities
#with open('data_%s'%(comm.rank), 'wb') as f:
#  pickle.dump(data, f)
#
##--------------------------------------------------------------------------------------
## Simplex experiment (resource user composition)
## -------------------------------------------------------------------------------------
# np.random.seed(comm.rank+876)

# total_size = 10
# N = 8
# M = 2
# C = 0.4
# K = 0
# num_samples = 3

# points = []

# for N1 in range(N + 1):
#   for N3 in range(N - N1 + 1):
#     N2 = N - N1 - N3
#     points.append((N1,N2,N3))

# if comm.size != len(points):
#   print('ERROR running on %d processors' % comm.size)
#   print(len(points))
#   sys.exit()

# point = points[comm.rank]
# print(point)
# print(comm.rank)
# N1,N2,N3 = point

# data = pd.DataFrame(columns = ['N1','N2','N3','stability'])
# data['N1'] = np.ones(num_samples) * N1
# data['N2'] = np.ones(num_samples) * N2
# data['N3'] = np.ones(num_samples) * N3

# stabilities = []
# total_connectances = []
# convergences = []

# composition = np.array([N,N1,N2,N3,K,M])

# run_cm_samples(total_size, C, num_samples, stabilities, total_connectances,
#                convergences, composition = composition, sample = False)

# data['stability'] = stabilities
# with open('data_%s'%(comm.rank), 'wb') as f:
#   pickle.dump(data, f)
