import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import pandas as pd


def _key_func(file):
  """Returns the number portion of filename."""
  return int(file.stem[len('data_') : ]) #int(file[len(folder + 'PSW_'):len(file)])


folder = 'colormap_expanded\\'
num_points = 32
folder = Path.cwd().joinpath('colormap_expanded')
file_names = sorted(folder.glob('data_*'), key=_key_func)


frames = []
data = pd.DataFrame(columns = ['Total_connectance','size','stability','converged'])

for file in file_names:
  with open(file, 'rb') as f:
    df = pickle.load(f)
  frames.append(df)

data.append(frames)
#PSW = np.array(PSW).reshape(10,32)/200

# bin data based on the total connectance
ranges = pd.cut(data['Total_connectance'],bins=30)
# group by total connectance and size, unstack to get 2d "z" for colormap
num_stable = data.groupby([ranges,'size'])['stability'].agg(['sum']).unstack().values
num_total = data.groupby([ranges,'size'])['stability'].agg(['count']).unstack().values
PSW = np.where(num_total==0,0,num_stable/num_total)

## x and y are bounds, so z should be the value *inside* those bounds.
## Therefore, remove the last value from the z array.
#z = z[:-1, :-1]
#levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())
#
#
## pick the desired colormap, sensible levels, and define a normalization
## instance which takes data values and translates those into levels.
#cmap = plt.get_cmap('PiYG')
#norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
#

#
## contours are *point* based plots, so convert our bound into point
## centers
#cf = ax1.contourf(x[:-1, :-1] + dx/2.,
#                  y[:-1, :-1] + dy/2., z, levels=levels,
#                  cmap=cmap)
#fig.colorbar(cf, ax=ax1)
#ax1.set_title('contourf with levels')
#
## adjust spacing between subplots so `ax1` title and `ax0` tick labels
## don't overlap
#fig.tight_layout()
#
#plt.show()
#
#plt.figure()
#plt.imshow(PSW,origin = 'lower', extent =
#                   [0.1,0.6,5,15],
#                   aspect = 0.6/15)
#plt.colorbar()
#plt.xlabel('Decision Center–Resource User Connectance', size = 'large')
#plt.ylabel('Total System Size', size = 'large')