import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
from pathlib import Path


def _key_func(file):
  """Returns the number portion of filename."""
  return int(file.stem[len('PSW_') : ]) #int(file[len(folder + 'PSW_'):len(file)])

def restitch_data(folder, num_points, cells_per_file):
  """

  """
  PSW = np.zeros((num_points, 10))
  PSW_filtered = np.zeros((num_points, 10))
  files = glob.glob(folder + 'PSW_*')
  for file in files:
    n = int(file[len(folder + 'PSW_'):len(file)]) # piece number
    print(n)
    y_index = n//num_points # size
    x_index =  (cells_per_file*n)%num_points # connectance
    print((x_index,y_index))
    with open(file, 'rb') as f:
      data = pickle.load(f)
    print(data)
    print(PSW[x_index:x_index+cells_per_file, y_index])
    PSW[x_index:x_index+cells_per_file, y_index] = data/200
#    if data[2] == 0:
#      PSW_filtered[x_indices, y_index] = 0
#    else:
#      PSW_filtered[x_index, y_index] = data[1]/data[2]
  return PSW, PSW_filtered

folder = 'colormap_data_updated\\'
num_points = 32
folder = Path.cwd().joinpath('colormap_data_updated')
file_names = sorted(folder.glob('PSW_*'), key=_key_func)

#files = glob.glob(folder + 'PSW_*')
#files_sorted = sorted(files, key=_key_func)
PSW = []

for file in file_names:
  with open(file, 'rb') as f:
    data = pickle.load(f)
  PSW.append(data)
PSW = np.array(PSW).reshape(10,32)/200

#PSW, PSW_filtered = restitch_data(folder, num_points, 2)
#interpolate missing points
#missing = np.array([6, 76, 90, 92])
#missing_x = missing%num_points
#missing_y = missing//num_points
#PSW[missing_x[0],missing_y[0]] = 0.10
#PSW[missing_x[1],missing_y[1]] = 0.0175
#PSW[missing_x[2],missing_y[2]] = 0.01
#PSW[missing_x[3],missing_y[3]] = 0.007

plt.figure()
plt.imshow(PSW,interpolation='kaiser',origin = 'lower', extent =
                   [0.1,0.6,5,15],
                   aspect = 0.6/15)
plt.colorbar()
plt.xlabel('Decision Center–Resource User Connectance', size = 'large')
plt.ylabel('Total System Size', size = 'large')