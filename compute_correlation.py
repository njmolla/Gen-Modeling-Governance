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
folder = 'Correlation_15//raw_data//'
files = glob.glob(folder + 'corr_data' + '_*')
frames = []
for file in files:
  with open(file, 'rb') as f:
    df = pickle.load(f)
    frames.append(df)

data = pd.concat(frames, ignore_index = True)
#add columns to dataframe for different versions of strategy parameters

F = np.stack(data['F'].values)
F_p = np.zeros(np.shape(F))
F_p[F>0] = F[F>0]
data['F_p'] = F_p.tolist()
F_n = np.zeros(np.shape(F))
F_n[F<0] = np.abs(F[F<0])
data['F_n'] = F_n.tolist()
data['F_abs'] = np.abs(F).tolist()

H = np.stack(data['H'].values)
H_p = np.zeros(np.shape(H))
H_p[H>0] = H[H>0]
data['H_p'] = H_p.tolist()
H_n = np.zeros(np.shape(H))
H_n[H<0] = np.abs(H[H<0])
data['H_n'] = H_n.tolist()
data['H_abs'] = np.abs(H).tolist()

W = np.stack(data['W'].values)
W_p = np.zeros(np.shape(W))
W_p[W>0] = W[W>0]
data['W_p'] = W_p.tolist()
W_n = np.zeros(np.shape(W))
W_n[W<0] = np.abs(W[W<0])
data['W_n'] = W_n.tolist()
data['W_abs'] = np.abs(W).tolist()

K = np.stack(data['K_p'].values)
K_p = np.zeros(np.shape(K))
K_p[K>0] = K[K>0]
data['K_plus'] = K_p.tolist()
K_n = np.zeros(np.shape(K))
K_n[K<0] = np.abs(K[K<0])
data['K_n'] = K_n.tolist()
data['K_abs'] = np.abs(K).tolist()

# dataframes to save the aggregated values of the parameters
mean_values = pd.DataFrame(columns = data.columns)
var_values = pd.DataFrame(columns = data.columns)

correlation_df = pd.DataFrame(index = list(data.columns[:35]) + list(data.columns[36:]),
                              columns = ['Mean Correlation', 'St Dev Correlation', 'Mean CI', 'St Dev CI'])

for column in list(data.columns[:35]) + list(data.columns[36:]):
  param = np.stack(data[column].values)
  axes = np.arange(len(np.shape(param)))[1:] # shape is # of points x dimensions of the parameter itself
  param_averaged = np.mean(param, axis = tuple(axes))
  mean_values[column] = param_averaged
  # 1d case
  if len(np.shape(np.squeeze(param))) == 2: # usually 1xN
    param_var = np.std(np.squeeze(param), axis=1)
    var_values[column] = param_var
  # 2d case
  elif len(np.shape(np.squeeze(param))) == 3: # Either NxM or NxN
    #param_var = np.std(np.mean(np.squeeze(param),axis=2),axis=1)
    param_var = np.std(np.mean(np.squeeze(param),axis=1),axis=1)
    var_values[column] = param_var
  # 3d case (dg_dF, dp_dH, F, H)
  elif len(np.shape(np.squeeze(param))) == 4: # NxMxN
    #param_var = np.std(np.mean(np.squeeze(param),axis=(1,2)),axis=1)
    param_var = np.std(np.mean(np.squeeze(param),axis=(2,3)),axis=1)
    var_values[column] = param_var
  else:
  # scalar
    param_var = np.zeros(len(param))
    var_values[column] = param_var
  stability = data['stability'].values
  
  if column == ('de_dg' or 'dg_dy' or 'dp_dy' or 'da_dp'): # these are MxN
    param_var = np.std(np.mean(np.squeeze(param),axis=1),axis=1)
  # compute correlation of parameter with stability
  correlation_df['Mean Correlation'][column] = compute_correlation(param_averaged, stability)
  correlation_df['St Dev Correlation'][column] = compute_correlation(param_var, stability)

  # bootstrap 95% confidence intervals for correlation of avg
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
   sample_corrs[i] = compute_correlation(sample_params[i], sample_stability[i])
  sorted_corrs = np.sort(sample_corrs)
  correlation_df['Mean CI'][column] = np.array([sorted_corrs[int(num_samples*0.025)], sorted_corrs[int(num_samples*0.975)]]) # take 5th and 95th percentile of

  #bootstrap 95% confidence intervals for correlation of standard deviation
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

with open('corr_data_15_2', 'wb') as f:
  pickle.dump(correlation_df, f)

mean_corr = correlation_df['Mean Correlation'].dropna().values
mean_CI = correlation_df['Mean CI'].dropna().values
mean_CI = np.stack(mean_CI)
mean_yerr = np.c_[mean_corr-mean_CI[:,0],mean_CI[:,1]-mean_corr ].T
# get indices of parameters that are significant (conf int don't include 0 and corr is greater than 0.01)
significant_ind = (mean_CI[:,0]*mean_CI[:,1] > 0) & (abs(np.where(mean_corr<0,np.max(mean_CI,axis=1),np.min(mean_CI,axis=1))) > 5e-3)
significant_ind_sorted = np.argsort(mean_corr[significant_ind])
mean_corr_sorted = mean_corr[significant_ind][significant_ind_sorted]
mean_CI = mean_CI[significant_ind][significant_ind_sorted]
mean_yerr_sorted = mean_yerr[:,significant_ind][:,significant_ind_sorted]

plt.figure()
plt.bar(np.arange(len(mean_corr_sorted)), mean_corr_sorted, yerr = mean_yerr_sorted, align='center', alpha=0.5)
labels = [r'$\phi$', r'$\psi$', r'$\alpha$', r'$\beta$', r'$\hat{\beta}$',r'$\tilde{\beta}$',r'$\overline{\beta}$',r'$\sigma$',r'$\eta$',r'$\lambda$',r'$\bar{\eta}$',r'$\mu$',r'$\dfrac{\partial s}{\partial r}$',
          r'$\dfrac{\partial e}{\partial r}$',r'$\dfrac{\partial e}{\partial g}$',r'$\dfrac{\partial g}{\partial F}$',r'$\dfrac{\partial g}{\partial y}$',
          r'$\dfrac{\partial p}{\partial y}$',r'$\dfrac{\partial b}{\partial e}$',r'$\dfrac{\partial a}{\partial r}$',r'$\dfrac{\partial q}{\partial a}$',
          r'$\dfrac{\partial a}{\partial p}$',r'$\dfrac{\partial p}{\partial H}$',r'$\dfrac{\partial c}{\partial W_p}$',r'$\dfrac{\partial c}{\partial w_n}$',
          r'$\dfrac{\partial l}{\partial x}$',r'$\dfrac{\partial u}{\partial x}$',r'$\dfrac{\partial i}{\partial K_p}$',r'$\dfrac{\partial i}{\partial K_n}$',r'$\dfrac{\partial i}{\partial y_p}$',
          r'$\dfrac{\partial i}{\partial y_n}$','|F|','|H|','|W|','|K|']

mean_label_indices = np.array(labels)[significant_ind][significant_ind_sorted]
plt.xticks(np.arange(len(mean_label_indices)), mean_label_indices)
plt.title('Correlation of parameters with stability')
plt.savefig('Correlation_15.svg')
plt.show()

std_corr = correlation_df['St Dev Correlation'].dropna()
std_CI = correlation_df['St Dev CI'].dropna().values
std_CI = np.stack(std_CI)
std_yerr = np.c_[std_corr-std_CI[:,0],std_CI[:,1]-std_corr].T
significant_ind = (std_CI[:,0]*std_CI[:,1] > 0) & (abs(np.where(std_corr<0,np.max(std_CI,axis=1),np.min(std_CI,axis=1))) > 5e-3)
significant_ind_sorted = np.argsort(std_corr[significant_ind])
#std_label_indices = np.concatenate([np.argsort(std_CI[:,0])[:5],np.argsort(std_CI[:,0])[-5:]])

std_corr_sorted = std_corr.values[significant_ind][significant_ind_sorted]
std_CI = std_CI[significant_ind][significant_ind_sorted]
std_yerr = std_yerr[:,significant_ind][:,significant_ind_sorted]
plt.figure()
plt.bar(np.arange(len(std_corr_sorted)), std_corr_sorted, yerr = std_yerr, align='center', alpha=0.5)

std_label_indices = np.array(labels)[significant_ind][significant_ind_sorted]
plt.xticks(np.arange(len(std_label_indices)), std_label_indices)
plt.title('Correlation of standard deviation in parameters with stability')
plt.savefig('Correlation_15_std.svg')
plt.show()