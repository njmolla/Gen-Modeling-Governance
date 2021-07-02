import numpy as np
from GM_code import run_system
from mpi4py import MPI
import pandas as pd
import pickle


## Run in parallel ##
comm = MPI.COMM_WORLD

np.random.seed(comm.rank+876)

total_model_runs = 9600
model_runs_per_proc = int(total_model_runs / comm.size)

data = pd.DataFrame(columns = ['phi', 'psis', 'alphas', 'betas', 'beta_hats','beta_tildes','sigmas','etas','lambdas','eta_bars','mus','ds_dr',
                               'de_dr','de_dg','dg_dF','dg_dy','dp_dy','db_de','da_dr','dq_da','da_dp','dp_dH','dc_dw_p','dc_dw_n','dl_dx',
                               'di_dK_p','di_dK_n','di_dy_p','di_dy_n','F','H','W','K_p','stability'])
stability_list = []

#specify composition
N1 = 1
N2 = 1
N3 = 0
K = 1
M = 2
T = N1+N2+N3+K+M+1
C = 0.4

filename = 'corr_data_'+str(comm.rank)

for i in range(model_runs_per_proc):
  try:
    result = run_system(N1,N2,N3,K,M,T,C,sample_exp = True)
  except Exception as e:
    print(e)
    continue
  stability = result[0] # stability is the first return value
  stability_list.append(stability)


  param_series = pd.Series(result[6:], index = data.columns[:-1])
  data = data.append(param_series, ignore_index=True)

data['stability'] = stability_list
with open(filename, 'wb') as f:
  pickle.dump(data, f)




