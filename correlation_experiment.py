import numpy as np
from GM_code import run_multiple
from mpi4py import MPI

## Run in serial ##
#size = 5
#C = 0.5
#num_samples = 10
#filename = 'corr_data'
#num_stable_webs, num_converged, data = run_multiple(size,C,num_samples,filename,record_data=True)
#

## Run in parallel ##
comm = MPI.COMM_WORLD

np.random.seed(comm.rank+876)

total_model_runs = 5000
model_runs_per_proc = int(total_model_runs / comm.size)

size = 5
C = 0.4
filename = 'corr_data_'+str(comm.rank)

num_stable_webs, num_converged, data = run_multiple(size,C,model_runs_per_proc,filename,record_data=True)


