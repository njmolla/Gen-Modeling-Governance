import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


def compute_correlation(param, stability):
  PSW = sum(stability==True)/len(stability) # proportion stable webs
  num = sum(param[stability==True])-PSW*sum(param)
  denom = len(stability)*np.std(param)*np.std(stability)
  return num/denom

#
#with open("stability_data_v3", 'rb') as f:
#  df = pickle.load(f)
#
#data = df[['alpha_r', 'alpha_u', 'alpha_w', 'beta_r', 'dh_dr', 'ds_dr', 'de_dr', 'de_dw', 'db_dr', 'db_df', 'dl_dr',
#        'db_dw', 'dl_dw', 'dg_dl', 'dd_du', 'eigvals']]
#
#eigvals = np.stack(data['eigvals'].values)
#stability_1 = np.all(eigvals[:,0:3].real<0,axis = 1) # one-sided stability (non-zero GM param side)
#stability_2 = np.all(eigvals[:,3:].real<0,axis = 1) # one-sided stability (zero side)
#stability_final = np.all(eigvals.real<0,axis = 1) # overall stability
#
#correlation_df = pd.DataFrame (index = data.columns, columns = ['Correlation 1','Correlation 2','Overall Correlation', 'CI'])
#
#for column in data.columns:
#  param = data[column].values
#  # compute correlation of parameter with stability
#  correlation_df['Correlation 1'][column] = compute_correlation(param, stability_1)
#  correlation_df['Correlation 2'][column] = compute_correlation(param, stability_2)
#  correlation_df['Overall Correlation'][column] = compute_correlation(param, stability_final)
#  # bootstrap 95% confidence intervals
#  num_points = 10 #len(param)
#  num_samples = 10
#  sample_indices = np.random.randint(0,num_points,(num_points*num_samples)) # get indices for 100 samples of 1e6 each
#  sample_params = param[sample_indices]
#  sample_stability = stability_final[sample_indices]
#  sample_params = np.reshape(sample_params, (num_samples,num_points))
#  sample_stability = np.reshape(sample_stability, (num_samples,num_points))
#  sample_corrs = np.zeros(num_samples)
#  for i in range(num_samples):
#    result = compute_correlation(sample_params[i],sample_stability[i])
#    print(result)
#    sample_corrs[i] = result
#  sorted_corrs = np.sort(sample_corrs)
#  correlation_df['CI'][column] = [sorted_corrs[int(num_samples*0.025)],sorted_corrs[int(num_samples*0.975)]] # take 5th and 95th percentile of


#with open("correlation_data", 'wb') as f:
#  pickle.dump(correlation_df, f)

with open("correlation_data", 'rb') as f:
  correlation_df = pickle.load(f)
correlation = correlation_df['Correlation']
plt.figure()
plt.bar(np.arange(len(correlation.values())), correlation.values(), align='center', alpha=0.5)
labels = [r'$\alpha_{r}$',r'$\alpha_{u}$',r'$\alpha_{w}$',r'$\beta_{r}$', r'$\dfrac{\partial h}{\partial r}$',
r'$\dfrac{\partial s}{\partial r}$',r'$\dfrac{\partial e}{\partial r}$',r'$\dfrac{\partial e}{\partial w}$',
r'$\dfrac{\partial b}{\partial r}$',r'$\dfrac{\partial b}{\partial f}$',r'$\dfrac{\partial l}{\partial r}$',
r'$\dfrac{\partial b}{\partial w}$',r'$\dfrac{\partial l}{\partial w}$',
r'$\dfrac{\partial g}{\partial l}$',r'$\dfrac{\partial d}{\partial l}$',r'$\dfrac{\partial d}{\partial u}$']
plt.xticks(np.arange(len(correlation.values())), labels)
plt.title('Correlation of parameters with stability')

plt.show()