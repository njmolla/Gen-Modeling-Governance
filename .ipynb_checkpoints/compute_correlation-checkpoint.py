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
folder = 'Correlation_15\Run2_fixed_comp'
files = glob.glob(folder + '\corr_data' + '_*')
frames = []
for file in files:
  with open(file, 'rb') as f:
    df = pickle.load(f)
    frames.append(df)

data = pd.concat(frames, ignore_index = True)


correlation_df = pd.DataFrame (index = data.columns[:-1], columns = ['Correlation', 'CI'])

for column in data.columns[:-1]:
  param = np.stack(data[column].values)
  axes = np.arange(len(np.shape(param)))[1:]
  param_averaged = np.mean(param, axis = tuple(axes)) #np.mean(np.var(data[column][i],axis=0))
  stability = data['stability'].values
  # compute correlation of parameter with stability
  correlation_df['Correlation'][column] = compute_correlation(param_averaged, stability)

#  # bootstrap 95% confidence intervals
#  num_points = len(param)
#  num_samples = 100
#  sample_indices = np.random.randint(0,num_points,(num_points*num_samples)) # get indices for 100 samples of 1e6 each
#  sample_params = param[sample_indices]
#  sample_stability = stability_final[sample_indices]
#  sample_params = np.reshape(sample_params, (num_samples,num_points))
#  sample_stability = np.reshape(sample_stability, (num_samples,num_points))
#  sample_corrs = np.zeros(num_samples)
#  for i in range(num_samples):
#    sample_corrs[i] = compute_correlation(sample_params[i],sample_stability[i])
#  sorted_corrs = np.sort(sample_corrs)
#  correlation_df['CI'][column] = [sorted_corrs[int(num_samples*0.025)],sorted_corrs[int(num_samples*0.975)]] # take 5th and 95th percentile of


#with open("correlation_data_var_15", 'wb') as f:
#  pickle.dump(correlation_df, f)

#with open("correlation_data", 'rb') as f:
#  correlation_df = pickle.load(f)


corr = correlation_df['Correlation'].dropna()
corr_sorted = np.concatenate([np.sort(corr.values)[:5],np.sort(corr.values)[-5:]])
plt.figure()
plt.bar(np.arange(len(corr_sorted)), corr_sorted, align='center', alpha=0.5)
labels = [r'$\phi$', r'$\psi$', r'$\alpha$', r'$\beta$', r'$\hat{\beta}$',r'$\tilde{\beta}$',r'$\sigma$',r'$\eta$',r'$\lambda$',r'$\bar{\eta}$',r'$\mu$',r'$\dfrac{\partial s}{\partial r}$',
          r'$\dfrac{\partial e}{\partial r}$',r'$\dfrac{\partial e}{\partial g}$',r'$\dfrac{\partial g}{\partial F}$',r'$\dfrac{\partial g}{\partial y}$',
          r'$\dfrac{\partial p}{\partial y}$',r'$\dfrac{\partial b}{\partial e}$',r'$\dfrac{\partial a}{\partial r}$',r'$\dfrac{\partial q}{\partial a}$',
          r'$\dfrac{\partial a}{\partial p}$',r'$\dfrac{\partial p}{\partial H}$',r'$\dfrac{\partial c}{\partial W_p}$',r'$\dfrac{\partial c}{\partial w_n}$',
          r'$\dfrac{\partial l}{\partial x}$',r'$\dfrac{\partial i}{\partial K_p}$',r'$\dfrac{\partial i}{\partial K_n}$',r'$\dfrac{\partial i}{\partial y_p}$',
          r'$\dfrac{\partial i}{\partial y_n}$','F','H','W','K']
label_indices = np.concatenate([np.argsort(corr.values)[:5],np.argsort(corr.values)[-5:]])
plt.xticks(np.arange(len(label_indices)), np.array(labels)[label_indices])
plt.title('Correlation of parameters with stability')

plt.show()