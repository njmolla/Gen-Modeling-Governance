import numpy as np
import networkx as nx
from compute_J import determine_stability
from strategy_optimization import nash_equilibrium


def sample_scale_params(N1,N2,N3,K,M,T,C):

  N = N1 + N2 + N3 + K

  # ------------------------------------------------------------------------
  # Initialize scale parameters
  # ------------------------------------------------------------------------
  phi = np.random.rand(1) # $
  psis = np.zeros([1,N]) # $
  psis[0,0:N1+N2] = np.random.dirichlet(np.ones(N1+N2),1) # need to sum to 1
  alphas = np.random.rand(1,N) # $
  betas = np.zeros([1,N]) # $
  beta_hats = np.zeros([1,N]) # $
  beta_tildes = np.zeros([1,N]) # $

  # beta parameters for extractors
  if N==1:
    #special case if there is only one actor (no collaboration possible)
    if N1 == 1:
      betas[0] = 1
    elif N2 == 1:
      betas[0] = np.random.rand(1)
      beta_hats = 1 - betas
  else:
    betas[0,0:N1] = np.random.rand(N1)
    beta_tildes[0,0:N1] = 1 - betas[0,0:N1]
    # beta parameters for resource users with both uses
    beta_params = np.random.dirichlet(np.ones(3),N2).transpose()
    betas[0,N1:N2+N1] = beta_params[0]
    beta_tildes[0,N1:N2+N1] = beta_params[1]
    beta_hats[0,N1:N2+N1] = beta_params[2]
    # beta parameters for resource users with non-extractive use
    beta_tildes[0,N2+N1:N-K] =  np.random.rand(N3)
    beta_hats[0,N1+N2:N1+N2+N3] = 1 - beta_tildes[0,N1+N2:N1+N2+N3]
    # beta parameters for bridging organizations
    beta_tildes[0,N-K:N] = np.ones(K) # one for all bridging orgs

  sigmas = np.zeros([N,N]) # sigma_k,n is kxn $
  sigmas = np.random.dirichlet(np.ones(N),N)

  etas = np.random.rand(1,N) # $
  eta_bars = (1-etas)[0] # TODO: fix for 1 actor/no undermining

  lambdas = np.zeros([N,N])  # lambda_k,n is kxn $
  lambdas = np.random.dirichlet(np.ones(N),N)

  mus = np.random.rand(1,M) # $
  return phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus

def sample_exp_params(N1,N2,N3,K,M,T,C):
  # ------------------------------------------------------------------------
  # Initialize exponent parameters
  # ------------------------------------------------------------------------
  N = N1 + N2 + N3 + K
  ds_dr = np.random.uniform(-1,1,(1))  # 0-1 $
  de_dr = np.random.uniform(1,2,(1,N)) # 0-2
  de_dg = np.zeros((1,M,N))  # $
  links = np.random.rand(N1+N2) < C
  # resample until at least one gov-extraction interaction
  while np.count_nonzero(links) == 0:
    links = np.random.rand(N1+N2) < C
    #print('resampling links')
  de_dg[:,:,0:N1+N2][:,:,links] = np.random.uniform(-1,1,(1,M,sum(links)))
  dg_dF = np.random.uniform(0,2,(N,M,N))  # dg_m,n/(dF_i,m,n * x_i) is ixmxn $
  dg_dy = np.random.rand(M,N)*2 # $
  dp_dy = np.random.rand(M,N)*2 # $
  db_de = np.random.uniform(-1,1,(1,N))
  da_dr = np.random.rand(1,N)*2 # $
  dq_da = np.random.uniform(-1,1,(1,N)) # $
  da_dp = np.random.uniform(-1,1,((1,M,N)))
  links = np.random.rand(N2+N3) < C
  da_dp[:,:,N1:N-K][:,:,links] = np.random.uniform(-1,1,(1,M,sum(links)))
  dp_dH = np.random.uniform(0,2,(N,M,N)) # dp_m,n/(dH_i,m,n * x_i) is ixmxn $
  dc_dw_p = np.random.uniform(0,2,(N,N)) #dc_dw_p_i,n is ixn $
  indices = np.arange(0,N)
  dc_dw_p[indices,indices] = 0
  dc_dw_n = np.random.uniform(0,2,(N,N)) #dc_dw_n_i,n is ixn $
  dc_dw_n[indices,indices] = 0
  dl_dx = np.random.uniform(0.5,1,(N)) # more likely to converge if this is >=0.8
  di_dK_p = np.random.uniform(0,2,(N,M))
  di_dK_n = np.random.uniform(0,2,(N,M))
  di_dy_p = np.random.rand(1,M)  #
  di_dy_n = np.random.uniform(0,2,(1,M))  #

  return ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n

def fix_exp_params(N1,N2,N3,K,M,T,C):
  # ------------------------------------------------------------------------
  # Initialize exponent parameters
  # ------------------------------------------------------------------------
  N = N1 + N2 + N3 + K
  ds_dr = np.array([-0.5]) #np.random.uniform(-1,1,(1))
  de_dr = np.ones((1,N))*1.5 #np.random.uniform(1,2,(1,N))
  de_dg = np.zeros((1,M,N))  #
  links = np.random.rand(N1+N2) < C
  # resample until at least one gov-extraction interaction
  while np.count_nonzero(links) == 0:
    links = np.random.rand(N1+N2) < C
    #print('resampling links')
  de_dg[:,:,0:N1+N2][:,:,links] = np.random.uniform(-1,1,(1,M,sum(links)))
  dg_dF = np.ones((N,M,N)) #np.random.uniform(0,2,(N,M,N))  # dg_m,n/(dF_i,m,n * x_i) is ixmxn $
  dg_dy = np.ones((M,N)) #np.random.rand(M,N)*2 #
  dp_dy = np.ones((M,N)) #np.random.rand(M,N)*2 #
  db_de = np.ones((1,N))*0.5 #np.random.uniform(-1,1,(1,N))
  da_dr = np.ones((1,N)) #np.random.rand(1,N)*2 # $
  dq_da = np.ones((1,N))*0.5 #np.random.uniform(-1,1,(1,N)) # $
  da_dp = np.zeros((1,M,N))
  links = np.random.rand(N2+N3) < C
  da_dp[:,:,N1:N-K][:,:,links] = np.random.uniform(-1,1,(1,M,sum(links)))
  dp_dH = np.ones((N,M,N)) #np.random.uniform(0,2,(N,M,N)) # dp_m,n/(dH_i,m,n * x_i) is ixmxn $
  dc_dw_p = np.ones((N,N)) #np.random.uniform(0,2,(N,N)) #dc_dw_p_i,n is ixn $
  indices = np.arange(0,N)
  dc_dw_p[indices,indices] = 0
  dc_dw_n = np.ones((N,N)) #np.random.uniform(0,2,(N,N)) #dc_dw_n_i,n is ixn $
  dc_dw_n[indices,indices] = 0
  dl_dx = np.ones((N)) #np.random.uniform(0.5,1,(N)) # more likely to converge if this is >=0.8
  di_dK_p = np.ones((N,M)) #np.random.uniform(0,2,(N,M))
  di_dK_n = np.ones((N,M)) #np.random.uniform(0,2,(N,M))
  di_dy_p = np.ones((1,M))*0.5 #np.random.rand(1,M)  #
  di_dy_n = np.ones((1,M)) #np.random.uniform(0,2,(1,M))  #

  return ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n

def run_system(N1,N2,N3,K,M,T,C,sample_exp):
  N = N1 + N2 + N3 + K
  is_connected = False
  while is_connected == False:
    phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus = sample_scale_params(N1,N2,N3,K,M,T,C)
    if sample_exp == True:
      ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n = sample_exp_params(N1,N2,N3,K,M,T,C)
    else:
      ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n = fix_exp_params(N1,N2,N3,K,M,T,C)
    # Dummy effort allocation parameters (only for checking that following condition is satisfied)
    F = np.ones((N,M,N))  # F_i,m,n is ixmxn positive effort for influencing resource extraction governance $
    H = np.ones((N,M,N))  # effort for influencing resource access governance $
    W = np.ones((N,N))  # effort for collaboration. W_i,n is ixn $
    K_p = np.ones((N,M))  # effort for more influence for gov orgs $

    # calculate Jacobian
    J = determine_stability(N,K,M,T,
      phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n,
      F,H,W,K_p)
    # filter out systems that are not weakly connected even with all strategy params turned on
    adjacency_matrix = np.zeros([T,T])
    adjacency_matrix[J != 0] = 1
    graph = nx.from_numpy_array(adjacency_matrix,create_using=nx.DiGraph)
    is_connected = nx.is_weakly_connected(graph)
    if is_connected == False:
      print('resample all parameters')
      continue

    # Filter out systems that are trivially unstable (unstable in any individual state variable)
    # resample resource parameters if drdot_dr is positive
    while J[0,0]>0:
      ds_dr = np.random.uniform(-1,1,(1))  # 0-1 $
      de_dr = np.random.uniform(1,2,(1,N)) # 0-2
      phi = np.random.rand(1) # $
      psis = np.zeros([1,N]) # $
      psis[0,0:N1+N2] = np.random.dirichlet(np.ones(N1+N2),1) # need to sum to 1
      J = determine_stability(N,K,M,T,
      phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n,
      F,H,W,K_p)
      #print('resampling resource params')
    # resample all actor parameters if any dxdot_dx are positive
  #      while np.any(np.diagonal(J)[1:-M] > 0):
  #        db_de = np.random.uniform(-1,1,(1,N))
  #        dq_da = np.random.uniform(-1,1,(1,N))
  #        dl_dx = np.random.uniform(0.5,1,(N))
  #        de_dg[:,:,0:N1+N2][:,:,links] = np.random.uniform(-1,1,(1,M,sum(links)))
  #        dg_dF = np.random.uniform(0,2,(N,M,N))
  #        J, eigvals, stability = determine_stability(N,K,M,T,
  #                                                     phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,
  #                                                     F,H,W,K_p,D_jm)
    # resample all governance parameters if any dydot_dy are positive
    while np.any(np.diagonal(J)[-M:] > 0):
      di_dy_p = np.random.rand(1,M)
      di_dy_n = np.random.uniform(0,2,(1,M))
      J = determine_stability(N,K,M,T,
      phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n,
      F,H,W,K_p)
      #print('resampling gov params')


    # ------------------------------------------------------------------------
    # Strategy optimization
    # ------------------------------------------------------------------------

    # find nash equilibrium strategies
    if N<10:
      max_steps = 1000*(N)
    else:
      max_steps = 10000
    F,H,W,K_p,sigmas, lambdas, converged, strategy_history, grad = nash_equilibrium(max_steps,J,N,K,M,T,
    phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,
    dc_dw_n,dl_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n)


    # ------------------------------------------------------------------------
    # Compute Jacobian and see if system is weakly connected
    # ------------------------------------------------------------------------

    # check stability and use Jacobian to check whether system is weakly connected
    J = determine_stability(N,K,M,T,
	  phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n,
	  F,H,W,K_p)

    adjacency_matrix = np.zeros([T,T])
    adjacency_matrix[J != 0] = 1
    graph = nx.from_numpy_array(adjacency_matrix,create_using=nx.DiGraph)
    is_connected = nx.is_weakly_connected(graph)
    if is_connected == False:
      print('not weakly connected')
  # --------------------------------------------------------------------------
  # Compute the eigenvalues of the Jacobian and check stability
  # --------------------------------------------------------------------------
  eigvals = np.linalg.eigvals(J)
  if np.all(eigvals.real < 10e-5):  # stable if real part of eigenvalues is negative
    stability = True
  else:
    stability = False  # unstable if real part is positive, inconclusive if 0

  # compute actual total connectance
  total_connectance = (np.count_nonzero(de_dg) + np.count_nonzero(da_dp)
    + np.count_nonzero(F) + np.count_nonzero(H) + np.count_nonzero(W) + np.count_nonzero(K_p)) \
    /(np.size(de_dg) + np.size(da_dp) + np.size(F) + np.size(H) + np.size(W) + np.size(K_p))

  return (stability, J, converged, strategy_history, grad, total_connectance,
      phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n,
      F,H,W,K_p)

##########################################################################################

#def run_multiple(size,C,num_samples,filename=None,record_data=False, sample_exp=True, composition = None):
#  '''
#  Run num_samples samples and return the proportion of webs that are stable.
#  '''
#  num_stable_webs = 0
#  num_converged = 0
#  # record information for the correlation experiment
#  if record_data:
#    # set up data frame
#    data = pd.DataFrame(columns = ['phi', 'psis', 'alphas', 'betas', 'beta_hats','beta_tildes','sigmas','etas','lambdas','eta_bars','mus','ds_dr',
#                                   'de_dr','de_dg','dg_dF','dg_dy','dp_dy','db_de','da_dr','dq_da','da_dp','dp_dH','dc_dw_p','dc_dw_n','dl_dx',
#                                   'di_dK_p','di_dK_n','di_dy_p','di_dy_n','stability'])
#  stability_list = []
##  num_dR_matched = 0
##  num_matched = 0
##  num_condition_met = 0
#  for i in range(num_samples):
#    if composition == None:
#    # --------------------------------------------------------------------------
#    # Set up system composition if not specified
#    # --------------------------------------------------------------------------
#      np.random.seed(i)
#      # Need at least 2 resource users and one gov org
#      N = 2
#      M = 1
#      rand = np.random.rand(size-3)
#      # want 60% resource users
#      N += np.sum(rand < 0.6)
#      # 20% bridging orgs
#      K = np.sum(rand < 0.8) - np.sum(rand < 0.6)
#      # 20% gov orgs
#      M += np.sum(rand > 0.8)
#      rand2 = np.random.rand(N-1)
#      # choose at random whether guaranteed extractor is just extractor or extractor + accessor
#      N1orN2choose = np.random.rand(1)
#      if N1orN2choose < 0.5:
#        # guaranteed extractor only and 1/3 chance of additional RUs being extractor only
#        N1 = 1 + np.sum(rand2 < 0.33)
#        # 1/3 chance of additional RUs being extractors + accessors
#        N2 = np.sum(rand2 > 0.33) - np.sum(rand2 > 0.66)
#      else:
#        # 1/3 chance of additional RUs being extractor only
#        N1 = np.sum(rand2 < 0.33)
#        # guaranteeed extractor + accessor, and 1/3 chance of additional RUs being both
#        N2 = 1 + np.sum(rand2 > 0.33) - np.sum(rand2 > 0.66)
#      # 1/3 chance of being accessor only
#      N3 = np.sum(rand2 > 0.66)
#    else:
#      N1, N2, N3, K, M = composition
#    N = N1 + N2 + N3 + K # total number of actors
#    T = N + M + 1 # total number of state variables
#    print((N1,N2,N3,K,M))
#    try:
#      result = run_system(N1,N2,N3,K,M,T,C,sample_exp)
#    except Exception as e:
#      print(e)
#      continue
#    stability = result[0] # stability is the first return value
#    converged = result[2]
#    connectance = result[5]
#    stability_list.append(stability)
#
#    if stability:
#      num_stable_webs += 1
##      if converged:
##        num_stable_webs_filtered += 1
#    if converged:
#      num_converged += 1
#
#    if record_data:
#      print('adding data')
#      param_series = pd.Series(result[5:-4], index = data.columns[:-1])
#      data = data.append(param_series, ignore_index=True)
#
#  if record_data:
#    data['stability'] = stability_list
#    with open(filename, 'wb') as f:
#      pickle.dump(data, f)
#
#  return num_stable_webs, num_converged, connectance  # proportion of stable webs


def run_once(N1,N2,N3,K,M,T,C,sample_exp=True):
  '''
  Do a single run and return more detailed output.
  '''
  np.random.seed(667)

  (stability, J, converged, strategy_history, grad, total_connectance,
      phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n,
      F,H,W,K_p) = run_system(N1,N2,N3,K,M,T,C, sample_exp)


  return (N1,N2,N3,K,M,T,C,stability, J, converged, strategy_history, grad, total_connectance,
      phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n,
      F,H,W,K_p)


def main():
  # Size of system
  N1 = 2 # number of resource users that benefit from extraction only
  N2 = 1 # number of users with both extractive and non-extractive use
  N3 = 0  # number of users with only non-extractive use
  K = 0 # number of bridging orgs
  M = 2  # number of gov orgs
  T = N1 + N2 + N3 + K + M + 1  # total number of state variables

  # Connectance of system (for different interactions)
  C = 0.1  # Connectance between governance organizations and resource users.
            # (proportion of resource extraction/access interactions influenced by governance)

  return run_once(N1,N2,N3,K,M,T,C)


if __name__ == "__main__":
  (N1,N2,N3,K,M,T,C,stability, J, converged, strategy_history, grad, total_connectance,
      phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n,
      F,H,W,K_p) = main()