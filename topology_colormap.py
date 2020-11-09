from GM_code import run
import numpy as np
import csv
from mpi4py import MPI

comm = MPI.COMM_WORLD

num_processors = 10

size_ranges = np.arange(5,15,1)
connectance_ranges = np.linspace(0.2,0.6,10)
PSW = np.zeros((len(size_ranges),10))

size = size_ranges[comm.rank]
for j,C1 in enumerate(connectance_ranges):
  rand = np.random.rand(size)
  N = np.sum(rand < 0.5)
  #K = np.sum(rand > 0.4) - np.sum(rand > 0.6)
  K = 0
  M = np.sum(rand > 0.5)
  rand2 = np.random.rand(N)
  N1 = np.sum(N < 0.33)
  N2 = np.sum(N > 0.33) - np.sum(N > 0.66)
  N3 = np.sum(N > 0.66)
  N = N1 + N2 + N3 + K # total number of resource users
  T = N + M + 1 # total number of state variables
  C2 = 0.2 # gov org-gov org connectance
  PSW[comm.rank,j] = run_multiple(N1,N2,N3,K,M,T,C1,C2,1)

  with open('PSW.csv', 'w+') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerows(PSW)

