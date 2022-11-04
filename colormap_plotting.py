import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import pandas as pd


'''Script for constructing colormaps from the experiments ran by topology_colormaps.py'''

def _key_func(file):
  '''Returns the number portion of filename.'''
  return int(file.stem[len('data_') : ]) #int(file[len(folder + 'PSW_'):len(file)])

# import single file (this is only for the preliminary colormaps)
with open('data\prelim_cmaps\data_M','rb') as f:
  data = pickle.load(f)

def process_size_vs_connectance():
  '''Processes and plots results of the size vs connectance experiments'''
  
  # import folder of files
  data_name = 'data//colormap_data_updated//'
  folder = Path.cwd().joinpath(data_name)
  # file_names = sorted(folder.glob('round_1//data_*'), key=_key_func)
  # file_names = file_names +(sorted(folder.glob('round_2//data_*'), key=_key_func))
  connectances = np.linspace(0.1,0.8,32)
  cells_per_row = len(connectances)
  num_processors = 96
  cells_per_processor = 15*cells_per_row/num_processors
  processors_per_row = cells_per_row/cells_per_processor
  frames = []
  num_samples = 400

  for file in file_names:
    with open(file, 'rb') as f:
      df = pickle.load(f)
      processor_num = _key_func(file)
      start_index = int(cells_per_processor*(processor_num%processors_per_row))
    frames.append(df)

  data = pd.concat(frames,ignore_index = True)

  # Colormap with de/dg connectance
  # group by connectance and size, unstack to get 2d "z" for colormap
  num_stable = data.groupby(['connectance','size'])['stability'].agg(['sum']).unstack().values
  num_total = num_samples
  PSW = np.transpose(np.where(num_total==0,0,num_stable/num_samples))
  plt.figure()
  plt.imshow(PSW,origin = 'lower',extent =
                   [0.1,0.8,5,20],
                   aspect = 0.7/15)
  cbar = plt.colorbar()
  cbar.set_label('Proportion of Stable Systems', size = 'large')
  plt.xlabel('Connectance', size = 'large')
  plt.ylabel('Total System Size', size = 'large')
  plt.savefig('cm_overall.svg')

  # Colormap with total connectance
  # bin data based on the total connectance
  ranges,bins = pd.qcut(data['Total_connectance'],15,retbins=True)

  # group by total connectance and size
  num_stable = data.groupby([ranges,'size'])['stability'].agg(['sum']).values
  num_total = data.groupby([ranges,'size'])['stability'].agg(['count']).values
  PSW = np.squeeze(np.where(num_total==0,0,num_stable/num_total))
  connectances = np.squeeze(data.groupby([ranges,'size'])['Total_connectance'].agg(['mean']).values)
  sizes = np.squeeze(data.groupby([ranges,'size'])['size'].agg(['mean']).values)

  # Grid data and plot on colormap
  conn_gridded = np.linspace(min(connectances),max(connectances),50)
  size_gridded = np.arange(5, 20+1, 1)
  from scipy.interpolate import griddata
  PSW_gridded = griddata((connectances, sizes), PSW, (conn_gridded[None, :], size_gridded[:, None]), method='linear')

  plt.figure()
  plt.imshow(PSW_gridded,origin = 'lower',extent =
                   [5,20,0.1,0.8],
                   aspect = 16/30)
  plt.colorbar()
  plt.savefig('cm_overall_tot_conn.svg')
  plt.xlabel('Total Connectance', size = 'large')
  plt.ylabel('Total System Size', size = 'large')

  # Filled contour map with automatic interpolation
  plt.figure()
  plt.tricontourf(sizes,connectances,PSW,levels=np.linspace(PSW.min(), PSW.max(), 20))
  plt.colorbar()
  plt.xlabel('size', size = 'large')
  plt.ylabel('connectance', size = 'large')

def process_ternary_N_vs_K_vs_M():
  # Simplex experiment - overall
  data_name = 'data//simplex_data//overall//'
  folder = Path.cwd().joinpath(data_name)
  file_names = sorted(folder.glob('data_*'), key=_key_func)
  file_names = file_names +(sorted(folder.glob('run2//data_*'), key=_key_func))
  frames = []

  for file in file_names:
    with open(file, 'rb') as f:
      df = pickle.load(f)
    frames.append(df)
    
  data = pd.concat(frames,ignore_index = True)

  #Group by number of actors (N) and decision centers (M)
  num_stable = data.groupby(['N','M'])['stability'].agg(['sum']).values
  num_total = data.groupby(['N','M'])['stability'].agg(['count']).values
  N = (np.squeeze(data.groupby(['N','M'])['N'].agg(['mean']).values)-2)/7
  M = (np.squeeze(data.groupby(['N','M'])['M'].agg(['mean']).values)-1)/7
  K = 1 - N - M
  K[K<0] = 0
  PSW = np.squeeze(np.where(num_total==0,0,num_stable/num_total))

  ## ternary contour plot
  import plotly.figure_factory as ff
  fig = ff.create_ternary_contour(np.array([M, N, K]), PSW,
                                  pole_labels=['Resource Users', 'Decision Centers', 'Non-RU Actors'],
                                  interp_mode='cartesian',
                                  ncontours=10,
                                  colorscale='Viridis',
                                  showscale=True)
  fig.update_traces(opacity=0, selector=dict(type='contour'))
  fig.write_image("ternary_N_M_K.svg")

  # colormap
  num_stable = data.groupby(['N','M'])['stability'].agg(['sum']).unstack().values
  num_total = data.groupby(['N','M'])['stability'].agg(['count']).unstack().values
  N = (np.squeeze(data.groupby(['N','M'])['N'].agg(['mean']).values)-2)/7
  print(N)
  M = (np.squeeze(data.groupby(['N','M'])['M'].agg(['mean']).values)-1)/7
  print(M)
  K = 1 - N - M
  K[K<0] = 0
  PSW = np.squeeze(np.where(num_total==0,0,num_stable/num_total))

  plt.figure()
  plt.imshow(PSW,origin = 'lower')
  plt.xlabel('N', size = 'large')
  plt.ylabel('M', size = 'large')
  plt.colorbar()

def process_ternary_N1_vs_N2_vs_N3():
  # Simplex experiment - Types of RUs

  data_name = 'data//simplex_data//RUs//'
  folder = Path.cwd().joinpath(data_name)
  file_names = sorted(folder.glob('data_*'), key=_key_func)

  frames = []

  for file in file_names:
    with open(file, 'rb') as f:
      df = pickle.load(f)
    frames.append(df)

  # Group by number of extractors (N1) and accessors (N3)
  num_stable = data.groupby(['N1','N3'])['stability'].agg(['sum']).unstack().values
  num_total = data.groupby(['N1','N3'])['stability'].agg(['count']).unstack().values
  PSW = np.squeeze(np.where(num_total==0,0,num_stable/num_total))
  N1 = (np.squeeze(data.groupby(['N1','N3'])['N1'].agg(['mean']).values))/8
  N2 = (np.squeeze(data.groupby(['N1','N3'])['N2'].agg(['mean']).values))/8
  N3 = 1 - N1 - N2
  N3[N3<0] = 0


  # colormap
  plt.figure()
  plt.imshow(PSW,origin = 'lower')
  plt.xlabel('N1', size = 'large')
  plt.ylabel('N3', size = 'large')
  plt.colorbar()
  plt.savefig('cm_overall.svg')
  plt.xlabel('Connectance', size = 'large')
  plt.ylabel('Total System Size', size = 'large')



  #ternary contour plot
  import plotly.figure_factory as ff
  fig = ff.create_ternary_contour(np.array([N1, N2, N3]), PSW,
                                  pole_labels=['Extractors', 'Extractors & Accessors', 'Accessors'],
                                  interp_mode='cartesian',
                                  ncontours=10,
                                  colorscale='Viridis',
                                  showscale=True)
  fig.update_traces(opacity=0, selector=dict(type='contour'))
  fig.write_image("ternary_Ns.svg")


process_size_vs_connectance()
# process_composition_vs_connectance()
# process_ternary_N_vs_K_vs_M()
# process_ternary_N1_vs_N2_vs_N3()