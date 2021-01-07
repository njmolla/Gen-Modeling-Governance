import numpy as np
import matplotlib.pyplot as plt
import glob

def restitch_data(folder, num_points):
  """
  Combines the colormap data that was broken up for parallelization back into
  one single array. folder is the location of all of the colormap data, and the
  cmap_type is the variable being plotted, of which
  there are three:

  num_points is the number of points for the cap and amount (currently 10); aggregation
  method is the method for turning the trajectories for a given policy into a single value,
  either by averaging the final (equilibrium) value, averaging over a given time period,
  or counting the proportion of sustainable outcomes
  """
  PSW = np.zeros((num_points, num_points))
  PSW_filtered = np.zeros((num_points, num_points))
  files = glob.glob(folder + 'PSW_*.csv')
  for file in files:
    n = int(file[len(folder + 'PSW_'):len(file)-4]) # piece number
    y_index = n//num_points # size
    x_index = n%num_points # connectance
    data = np.loadtxt(file, delimiter = ',')
    PSW[x_index, y_index] = data[0]/100
    if data[2] == 0:
      PSW_filtered[x_index, y_index] = 0
    else:
      PSW_filtered[x_index, y_index] = data[1]/data[2]
  return PSW, PSW_filtered

folder = 'run3\\'
num_points = 10
PSW, PSW_filtered = restitch_data(folder, num_points)
#interpolate missing points
missing = np.array([6, 76, 90, 92])
missing_x = missing%num_points
missing_y = missing//num_points
PSW[missing_x[0],missing_y[0]] = 0.10
PSW[missing_x[1],missing_y[1]] = 0.0175
PSW[missing_x[2],missing_y[2]] = 0.01
PSW[missing_x[3],missing_y[3]] = 0.007

plt.figure()
plt.imshow(PSW_filtered,interpolation='nearest',origin = 'lower', extent =
                   [0.2,0.6,5,15],
                   aspect = 0.6/15)
plt.colorbar()
plt.xlabel('Gov. Institution–Resource User Connectance', size = 'large')
plt.ylabel('Number of Actors and Gov. Institutions', size = 'large')