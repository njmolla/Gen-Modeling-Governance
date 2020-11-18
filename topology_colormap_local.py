from GM_code import run_multiple
import numpy as np
import csv


size_ranges = np.arange(5,15,1)
connectance_ranges = np.linspace(0.2,0.6,10)
PSW = np.zeros((len(size_ranges),10))

np.random.seed(0)
for i, size in enumerate(size_ranges):
  for j,C1 in enumerate(connectance_ranges):
    # Need at least 2 resource users and one gov org
    N = 2
    M = 1
    rand = np.random.rand(size-3)
    N += np.sum(rand < 0.6)
    K = np.sum(rand < 0.8) - np.sum(rand < 0.6)
    M += np.sum(rand > 0.8)
    rand2 = np.random.rand(N-1)
    # choose at random whether guaranteed extractor is just extractor or extractor + accessor
    rand3 = np.random.rand(1)
    if rand3 < 0.5:
      N1 = 1 + np.sum(rand2 < 0.33)
      N2 = np.sum(rand2 > 0.33) - np.sum(rand2 > 0.66)
    else:
      N1 = np.sum(rand2 < 0.33)
      N2 = 1 + np.sum(rand2 > 0.33) - np.sum(rand2 > 0.66)
    N3 = np.sum(rand2 > 0.66)
    N = N1 + N2 + N3 + K # total number of resource users
    T = N + M + 1 # total number of state variables

    C2 = 0.2 # gov org-gov org connectance
    print((N1,N2,N3,K,M))
    PSW[i,j] = run_multiple(N1,N2,N3,K,M,T,C1,C2,1)

    with open('PSW.csv', 'w+') as f:
      csvwriter = csv.writer(f)
      csvwriter.writerows(PSW)