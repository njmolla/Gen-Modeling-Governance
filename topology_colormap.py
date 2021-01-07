from GM_code import run_multiple
import numpy as np
import csv
from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD

num_processors = 100

if comm.size != num_processors:
  print('ERROR running on %d processors' % comm.size)
  sys.exit()

size_ranges = np.arange(5,15,1)
connectance_ranges = np.linspace(0.2,0.6,10)
num_samples = 300

size = size_ranges[comm.rank//10]
C1 = connectance_ranges[comm.rank%10]
np.random.seed(2)
C2 = 0.2

num_stable_webs, num_stable_webs_filtered, num_converged = run_multiple(size,C1,C2,num_samples)

print(num_stable_webs)
print(num_stable_webs_filtered)

with open('PSW_%s.csv'%(comm.rank), 'w+') as f:
  csvwriter = csv.writer(f)
  csvwriter.writerow((num_stable_webs, num_stable_webs_filtered, num_converged))

