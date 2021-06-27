import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import pandas as pd


def _key_func(file):
  """Returns the number portion of filename."""
  return int(file.stem[len('data_') : ]) #int(file[len(folder + 'PSW_'):len(file)])

#import single file
# with open('prelim_cmaps\data_M','rb') as f:
#   data = pickle.load(f)

# import folder of files
data_name = 'colormap_expanded//'
folder = Path.cwd().joinpath(data_name)
file_names = sorted(folder.glob('data_*'), key=_key_func)
connectances = np.linspace(0.1,0.8,32)
cells_per_row = len(connectances)
num_processors = 96
cells_per_processor = 15*cells_per_row/num_processors
processors_per_row = cells_per_row/cells_per_processor
frames = []
num_samples = 200

for file in file_names:
  with open(file, 'rb') as f:
    df = pickle.load(f)
    processor_num = _key_func(file)
    start_index = int(cells_per_processor*(processor_num%processors_per_row))
    piece_connectances = np.expand_dims(connectances[start_index:int(start_index+cells_per_processor)],axis=1)
    piece_connectances = np.broadcast_to(piece_connectances,(len(piece_connectances),num_samples)).flatten()
    df['connectance'] = piece_connectances
  frames.append(df)

data = pd.concat(frames,ignore_index = True)


# data_name = 'simplex_data//RUs//'
# folder = Path.cwd().joinpath(data_name)
# file_names = sorted(folder.glob('data_*'), key=_key_func)
# print(len(file_names))

# connectances = np.linspace(0.1,0.8,32)
# frames = []

# for file in file_names:
#   with open(file, 'rb') as f:
#     df = pickle.load(f)
#   frames.append(df)

# data = pd.concat(frames,ignore_index = True)


#--------------------------------------------------------------------------------------
# overall colormap/contour map
# -------------------------------------------------------------------------------------

# bin data based on the total connectance
#ranges,bins = pd.qcut(data['Total_connectance'],15,retbins=True)
#
# num_stable = data.groupby([ranges,'size'])['stability'].agg(['sum']).values
# num_total = data.groupby([ranges,'size'])['stability'].agg(['count']).values
# PSW = np.squeeze(np.where(num_total==0,0,num_stable/num_total))
#connectances = np.squeeze(data.groupby([ranges,'size'])['Total_connectance'].agg(['mean']).values)
#sizes = np.squeeze(data.groupby([ranges,'size'])['size'].agg(['mean']).values)

#from scipy.interpolate import griddata
#zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')

# group by total connectance and size, unstack to get 2d "z" for colormap
num_stable = data.groupby(['connectance','size'])['stability'].agg(['sum']).unstack().values
num_total = 300
PSW = np.transpose(np.where(num_total==0,0,num_stable/num_total))

#--------------------------------------------------------------------------------------
# Simplex experiment
# -------------------------------------------------------------------------------------
# num_stable = data.groupby(['N'])['stability'].agg(['sum']).values
# num_total = data.groupby(['N'])['stability'].agg(['count']).values
# PSW = np.squeeze(np.where(num_total==0,0,num_stable/num_total))

# num_stable = data.groupby(['N1','N3'])['stability'].agg(['sum']).unstack().values
# num_total = data.groupby(['N1','N3'])['stability'].agg(['count']).unstack().values
# PSW = np.squeeze(np.where(num_total==0,0,num_stable/num_total))

# num_stable = data.groupby(['N','M'])['stability'].agg(['sum']).values
# num_total = data.groupby(['N','M'])['stability'].agg(['count']).values
# N = (np.squeeze(data.groupby(['N','M'])['N'].agg(['mean']).values)-2)/7
# print(N)
# M = (np.squeeze(data.groupby(['N','M'])['M'].agg(['mean']).values)-1)/7
# print(M)
# K = 1 - N - M
# K[K<0] = 0
# print(K)
# PSW = np.squeeze(np.where(num_total==0,0,num_stable/num_total))

#--------------------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------------------

## filled contour map with automatic interpolation
#plt.figure()
#plt.tricontourf(sizes,connectances,PSW,levels=np.linspace(PSW.min(), PSW.max(), 20))
#plt.colorbar()
#plt.xlabel('size', size = 'large')
#plt.ylabel('connectance', size = 'large')


# colormap

plt.figure()
plt.imshow(PSW,origin = 'lower')
# plt.xlabel('N1', size = 'large')
# plt.ylabel('N3', size = 'large')
plt.colorbar()
# plt.savefig('cm_N1_N3.svg')
plt.xlabel('Connectance', size = 'large')
plt.ylabel('Total System Size', size = 'large')

# ternary contour plot
# import plotly.figure_factory as ff
# fig = ff.create_ternary_contour(np.array([N, K, M]), PSW,
#                                 pole_labels=['N', 'K', 'M'],
#                                 interp_mode='cartesian',
#                                 ncontours=10,
#                                 colorscale='Viridis',
#                                 showscale=True)
# fig.update_traces(opacity=0, selector=dict(type='contour'))
# fig.write_image("ternary_N_M_K.svg")