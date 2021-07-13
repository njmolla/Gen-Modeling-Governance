import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import glob


def compute_correlation(param, stability):
  PSW = sum(stability==True)/len(stability) # proportion stable webs
  num = sum(param[stability==True])-PSW*sum(param)
  denom = len(stability)*np.std(param)*np.std(stability)
  return np.where(denom < 1e-10, 0, num/denom)
#
folder = 'Correlation_15//run3_revisedEqn//'
files = glob.glob(folder + 'corr_data' + '_*')
frames = []
for file in files:
  with open(file, 'rb') as f:
    df = pickle.load(f)
    frames.append(df)

data = pd.concat(frames, ignore_index = True)
# dataframes to save the aggregated values of the parameters
mean_values = pd.DataFrame(columns = data.columns)
var_values = pd.DataFrame(columns = data.columns)

correlation_df = pd.DataFrame (index = data.columns[:-1], columns = ['Mean Correlation', 'St Dev Correlation', 'Mean CI', 'St Dev CI'])

for column in data.columns[:-1]:
  param = np.stack(data[column].values)
  axes = np.arange(len(np.shape(param)))[1:] # shape is # of points x dimensions of the parameter itself
  param_averaged = np.mean(param, axis = tuple(axes))
  mean_values[column] = param_averaged
  # 1d case
  if len(np.shape(np.squeeze(param)))==2:
    param_var = np.std(np.squeeze(param), axis=1)
    var_values[column] = param_var
  # 2d case
  elif len(np.shape(np.squeeze(param)))==3:
    param_var = np.std(np.mean(np.squeeze(param),axis=2),axis=1)
    var_values[column] = param_var
  # 3d case (dg_dF, dp_dH, F, H)
  elif len(np.shape(np.squeeze(param)))==4:
    param_var = np.std(np.mean(np.squeeze(param),axis=(1,2)),axis=1)
    var_values[column] = param_var
  else:
    param_var = np.zeros(len(param))
    var_values[column] = param_var
  stability = data['stability'].values
  
  if column == ('de_dg' or 'dg_dy' or 'dp_dy' or 'da_dp'): # these are MxN
    param_var = np.std(np.mean(np.squeeze(param),axis=1),axis=1)
  # compute correlation of parameter with stability
  correlation_df['Mean Correlation'][column] = compute_correlation(param_averaged, stability)
  correlation_df['St Dev Correlation'][column] = compute_correlation(param_var, stability)

# bootstrap 95% confidence intervals
  param = mean_values[column].values
  num_points = len(param)
  num_samples = 100
  sample_indices = np.random.randint(0,num_points,(num_points*num_samples)) # get indices for 100 samples of 1e6 each
  sample_params = param[sample_indices]
  sample_stability = stability[sample_indices]
  sample_params = np.reshape(sample_params, (num_samples,num_points))
  sample_stability = np.reshape(sample_stability, (num_samples,num_points))
  sample_corrs = np.zeros(num_samples)
  for i in range(num_samples):
   sample_corrs[i] = compute_correlation(sample_params[i],sample_stability[i])
  sorted_corrs = np.sort(sample_corrs)
  correlation_df['Mean CI'][column] = np.array([sorted_corrs[int(num_samples*0.025)],sorted_corrs[int(num_samples*0.975)]]) # take 5th and 95th percentile of

# bootstrap 95% confidence intervals
  param = var_values[column].values
  num_points = len(param)
  num_samples = 100
  sample_indices = np.random.randint(0,num_points,(num_points*num_samples)) # get indices for 100 samples of 1e6 each
  sample_params = param[sample_indices]
  sample_stability = stability[sample_indices]
  sample_params = np.reshape(sample_params, (num_samples,num_points))
  sample_stability = np.reshape(sample_stability, (num_samples,num_points))
  sample_corrs = np.zeros(num_samples)
  for i in range(num_samples):
   sample_corrs[i] = compute_correlation(sample_params[i],sample_stability[i])
  sorted_corrs = np.sort(sample_corrs)
  correlation_df['St Dev CI'][column] = np.array([sorted_corrs[int(num_samples*0.025)],sorted_corrs[int(num_samples*0.975)]]) # take 5th and 95th percentile of


mean_corr = correlation_df['Mean Correlation'].dropna()
mean_CI = correlation_df['Mean CI'].dropna().values
mean_CI = np.stack(mean_CI)
mean_corr_sorted = np.concatenate([np.sort(mean_corr.values)[:5],np.sort(mean_corr.values)[-5:]])
mean_label_indices = np.concatenate([np.argsort(mean_corr.values)[:5],np.argsort(mean_corr.values)[-5:]])
mean_CI = mean_CI[mean_label_indices]
mean_yerr = np.c_[mean_corr_sorted-mean_CI[:,0],mean_CI[:,1]-mean_corr_sorted ].T
plt.figure()
plt.bar(np.arange(len(mean_corr_sorted)), mean_corr_sorted, yerr = mean_yerr, align='center', alpha=0.5)
labels = [r'$\phi$', r'$\psi$', r'$\alpha$', r'$\beta$', r'$\hat{\beta}$',r'$\tilde{\beta}$',r'$\overline{\beta}$',r'$\sigma$',r'$\eta$',r'$\lambda$',r'$\bar{\eta}$',r'$\mu$',r'$\dfrac{\partial s}{\partial r}$',
          r'$\dfrac{\partial e}{\partial r}$',r'$\dfrac{\partial e}{\partial g}$',r'$\dfrac{\partial g}{\partial F}$',r'$\dfrac{\partial g}{\partial y}$',
          r'$\dfrac{\partial p}{\partial y}$',r'$\dfrac{\partial b}{\partial e}$',r'$\dfrac{\partial a}{\partial r}$',r'$\dfrac{\partial q}{\partial a}$',
          r'$\dfrac{\partial a}{\partial p}$',r'$\dfrac{\partial p}{\partial H}$',r'$\dfrac{\partial c}{\partial W_p}$',r'$\dfrac{\partial c}{\partial w_n}$',
          r'$\dfrac{\partial l}{\partial x}$',r'$\dfrac{\partial u}{\partial x}$',r'$\dfrac{\partial i}{\partial K_p}$',r'$\dfrac{\partial i}{\partial K_n}$',r'$\dfrac{\partial i}{\partial y_p}$',
          r'$\dfrac{\partial i}{\partial y_n}$','F','H','W','K']

plt.xticks(np.arange(len(mean_label_indices)), np.array(labels)[mean_label_indices])
plt.title('Correlation of parameters with stability')
plt.savefig('Correlation_15.svg')
plt.show()

std_corr = correlation_df['St Dev Correlation'].dropna()
std_CI = correlation_df['St Dev CI'].dropna().values
std_CI = np.stack(std_CI)
std_corr_sorted = np.concatenate([np.sort(std_corr.values)[:5],np.sort(std_corr.values)[-5:]])
std_label_indices = np.concatenate([np.argsort(std_corr.values)[:5],np.argsort(std_corr.values)[-5:]])
std_CI = std_CI[std_label_indices]
std_yerr = np.c_[std_corr_sorted-std_CI[:,0],std_CI[:,1]-std_corr_sorted ].T
plt.figure()
plt.bar(np.arange(len(std_corr_sorted)), std_corr_sorted, yerr = std_yerr, align='center', alpha=0.5)

plt.xticks(np.arange(len(std_label_indices)), np.array(labels)[std_label_indices])
plt.title('Correlation of standard deviation in parameters with stability')
plt.savefig('Correlation_15_std.svg')
plt.show()