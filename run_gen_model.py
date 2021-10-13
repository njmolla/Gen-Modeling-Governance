import numpy as np
import networkx as nx
from compute_J import compute_Jacobian
from strategy_optimization import nash_equilibrium


def sample_scale_params(N1,N2,N3,K,M,T,C):
  '''
  Takes in system meta-parameters and samples the scale parameters
  
  Inputs:
    N1, N2, and N3: number of (exclusively) extractors, combined extractors and 
      accessors, and (exclusively) accessors, respectively
    K: number of non-resource user actors
    M: number of decision centers
    T: total number of state variables (this is N + K + M + 1)
    C: density of decision center interventions
  Outputs:
    All of the scale parameters 
  '''
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
  beta_bars = np.zeros([1,N])

  # beta parameters for extractors
  if N==1:
    #special case if there is only one actor (no collaboration or undermining possible)
    etas = np.zeros((N,N))
    eta_bars = np.ones(N)
    if N1 == 1:
      betas[0] = 1
    elif N2 == 1:
      betas[0] = np.random.rand(1)
      beta_hats = 1 - betas
  else:
  # extractors only have gain from extraction and collaboration
    betas[0,0:N1] = np.random.rand(N1)
    beta_tildes[0,0:N1] = 1 - betas[0,0:N1]
    # resource users with both uses have gain from extraction, collaboration, and access
    beta_params = np.random.dirichlet(np.ones(3),N2).transpose()
    betas[0,N1:N2+N1] = beta_params[0]
    beta_tildes[0,N1:N2+N1] = beta_params[1]
    beta_hats[0,N1:N2+N1] = beta_params[2]
    # resource users with non-extractive use haave gain from access and collaboration
    beta_tildes[0,N2+N1:N-K] =  np.random.rand(N3)
    beta_hats[0,N1+N2:N1+N2+N3] = 1 - beta_tildes[0,N1+N2:N1+N2+N3]
    # non-resource user actors have natural gain and gain from collaboration
    beta_bars[0,N-K:N] = np.random.rand(K)
    beta_tildes[0,N-K:N] = 1 - beta_bars[0,N-K:N]
    
    # loss scale parameters
    etas = np.random.rand(1,N) # $
    eta_bars = (1-etas)[0] # 

  sigmas = np.zeros([N,N]) # sigma_k,n is kxn $
  sigmas = np.transpose(np.random.dirichlet(np.ones(N),N))

  lambdas = np.zeros([N,N])  # lambda_k,n is kxn $
  lambdas = np.transpose(np.random.dirichlet(np.ones(N),N))

  mus = np.random.rand(1,M) # $
  return phi,psis,alphas,betas,beta_hats,beta_tildes,beta_bars,sigmas,etas,lambdas,eta_bars,mus

def sample_exp_params(N1,N2,N3,K,M,T,C):
  '''
  Takes in system meta-parameters and samples the exponent parameters
  (used for the correlation experiments)
  
  Inputs:
    N1, N2, and N3: number of (exclusively) extractors, combined extractors and 
      accessors, and (exclusively) accessors, respectively
    K: number of non-resource user actors
    M: number of decision centers
    T: total number of state variables (this is N + K + M + 1)
    C: density of decision center interventions
  Outputs:
    All of the exponent parameters 
  '''
  # ------------------------------------------------------------------------
  # Initialize exponent parameters
  # ------------------------------------------------------------------------
  N = N1 + N2 + N3 + K
  de_dr = np.random.uniform(1,2,(1,N)) # 1-2
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
  du_dx = np.random.uniform(0,1,(N))
  dl_dx = np.random.uniform(0.5,1,(N)) 
  di_dK_p = np.random.uniform(0,2,(N,M))
  di_dK_n = np.random.uniform(0,2,(N,M))
  # Ensure di_dy_p <  di_dy_n
  di_dy_n = np.zeros((1,M))  #
  di_dy_p = np.zeros((1,M)) #
  di_dy = np.random.rand(2,M)
  di_dy_n[0] = np.amax(di_dy,axis=0)  #
  di_dy_p[0] = np.amin(di_dy,axis=0) #

  return de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,du_dx,dl_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n

def set_fixed_exp_params(N1,N2,N3,K,M,T,C):
  '''
  Takes in system meta-parameters and sets the exponent parameters to fixed values
  (used for the topology experiments)
  
  Inputs:
    N1, N2, and N3: number of (exclusively) extractors, combined extractors and 
      accessors, and (exclusively) accessors, respectively
    K: number of non-resource user actors
    M: number of decision centers
    T: total number of state variables (this is N + K + M + 1)
    C: density of decision center interventions
  Outputs:
    All of the exponent parameters 
  '''
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
  du_dx = np.ones((N)) * 0.5
  dl_dx = np.ones((N)) 
  di_dK_p = np.ones((N,M)) #np.random.uniform(0,2,(N,M))
  di_dK_n = np.ones((N,M)) #np.random.uniform(0,2,(N,M))
  di_dy_p = np.ones((1,M))*0.5 #np.random.rand(1,M)  #
  di_dy_n = np.ones((1,M)) #np.random.uniform(0,2,(1,M))  #

  return ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,du_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n

##########################################################################################

def run_system(N1,N2,N3,K,M,T,C,sample_exp=True):
  '''
  Takes in system meta-parameters and produces a system parameterization and computes
  stability of that system
  
  Inputs:
    N1, N2, and N3: number of (exclusively) extractors, combined extractors and 
      accessors, and (exclusively) accessors, respectively
    K: number of non-resource user actors
    M: number of decision centers
    T: total number of state variables (this is N + K + M + 1)
    C: density of decision center interventions
  Outputs:
    stability: a boolean, True indicates stable, False indicates unstable
    J: the Jacobian
    converged: a boolean, True indicates the optimization reached convergence, False
      indicates the optimization ran into the maximum number of iterations
    strategy_history: a list of all the strategies of all of the actors throughout the 
      optimization, used to debug the optimization
    grad: the gradient of the objective function for the chosen strategy. used to debug
      the optimization
    total_connectance: the proportion of all possible interactions that exist in the 
      final system
    The remaining outputs are all of the sampled or computed scale, exponent, and strategy parameters.
  '''
  N = N1 + N2 + N3 + K
  is_connected = False
  while is_connected == False:
    phi,psis,alphas,betas,beta_hats,beta_tildes,beta_bars,sigmas,etas,lambdas,eta_bars,mus = sample_scale_params(N1,N2,N3,K,M,T,C)
    if sample_exp == True:
      de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,du_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n = sample_exp_params(N1,N2,N3,K,M,T,C)
      # ensure ds_dr < sum of (psis * de_dr)
      upper_bound = np.sum(psis*de_dr)
      ds_dr = np.random.uniform(-1,min(1,upper_bound),(1))
    else:
      ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,du_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n = set_fixed_exp_params(N1,N2,N3,K,M,T,C)
    
    # ensure dxdot_dx is negative for non-RUs
    while np.any(beta_bars*du_dx - eta_bars*dl_dx > 0):
      du_dx[N-K:] = np.random.uniform(0,1,(K,))
      dl_dx[N-K:] = np.random.uniform(0.5,1,(K,))
      # resample scale params and ensure they still add to 1
      beta_bars[0,N-K:N] = np.random.rand(K)
      beta_tildes[0,N-K:N] = 1 - beta_bars[0,N-K:N]
      etas[0,N-K:N] = np.random.rand(1,K) # $
      eta_bars[N-K:N]  = (1-etas[0,N-K:N])

      
    # Dummy effort allocation parameters (only for checking that following condition is satisfied)
    F = np.ones((N,M,N))  # F_i,m,n is ixmxn positive effort for influencing resource extraction governance $
    H = np.ones((N,M,N))  # effort for influencing resource access governance $
    W = np.ones((N,N))  # effort for collaboration. W_i,n is ixn $
    K_p = np.ones((N,M))  # effort for more influence for gov orgs $

    # calculate Jacobian
    J = compute_Jacobian(N,K,M,T,
      phi,psis,alphas,betas,beta_hats,beta_tildes,beta_bars,sigmas,etas,lambdas,eta_bars,mus,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,du_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n,
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
      J = compute_Jacobian(N,K,M,T,
      phi,psis,alphas,betas,beta_hats,beta_tildes,beta_bars,sigmas,etas,lambdas,eta_bars,mus,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,du_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n,
      F,H,W,K_p)
      print('resampling resource params')

    # resample all governance parameters if any dydot_dy are positive - this shouldn't happen anymore
    while np.any(np.diagonal(J)[-M:] > 0):
      di_dy_p = np.random.rand(1,M)
      di_dy_n = np.random.uniform(0,2,(1,M))
      J = compute_Jacobian(N,K,M,T,
      phi,psis,alphas,betas,beta_hats,beta_tildes,beta_bars,sigmas,etas,lambdas,eta_bars,mus,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,du_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n,
      F,H,W,K_p)
      print('resampling gov params')

    # ------------------------------------------------------------------------
    # Strategy optimization
    # ------------------------------------------------------------------------

    # find nash equilibrium strategies
    if N<10:
      max_steps = 1000*(N)
    else:
      max_steps = 10000
    F,H,W,K_p,sigmas, lambdas, converged, strategy_history, grad = nash_equilibrium(max_steps,J,N,K,M,T,
    phi,psis,alphas,betas,beta_hats,beta_tildes,beta_bars,sigmas,etas,lambdas,eta_bars,mus,
    ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,du_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n)
    
    # ------------------------------------------------------------------------
    # Compute Jacobian and see if system is weakly connected
    # ------------------------------------------------------------------------

    J = compute_Jacobian(N,K,M,T,
      phi,psis,alphas,betas,beta_hats,beta_tildes,beta_bars,sigmas,etas,lambdas,eta_bars,mus,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,du_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n,
      F,H,W,K_p)

    # check for positive dxdot_dx 
    if np.any(np.diagonal(J)[1:-M] > 0):
      print('Warning: positive dxdot_dx')
        
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
  if np.all(eigvals.real < 0):  # stable if real part of eigenvalues is negative
    stability = True
  else:
    stability = False  # unstable if real part is positive, inconclusive if 0

  # Compute actual total connectance
  
  # Zero out strategy parameters that are close to 0 for purposes of computing connectance
  F[np.abs(F)<1e-5] = 0
  H[np.abs(H)<1e-5] = 0
  W[np.abs(W)<1e-5] = 0
  K_p[np.abs(K_p)<1e-5] = 0
  total_connectance = (np.count_nonzero(de_dg) + np.count_nonzero(da_dp)
    + np.count_nonzero(F) + np.count_nonzero(H) + np.count_nonzero(W) + np.count_nonzero(K_p)) \
    /(np.size(de_dg) + np.size(da_dp) + np.size(F) + np.size(H) + np.size(W) + np.size(K_p))
    
  num_zero_betas_RUs = sum(beta_tildes[0, :N-K]==0)
  print('zeros-RUs '*num_zero_betas_RUs)
  num_zero_betas_nonRUs = sum(beta_tildes[0, -K:]==0)
  print('zeros-nonRUs '*num_zero_betas_nonRUs)
  
  return (stability, J, converged, strategy_history, grad, total_connectance,
      phi,psis,alphas,betas,beta_hats,beta_tildes,beta_bars,sigmas,etas,lambdas,eta_bars,mus,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,du_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n,
      F,H,W,K_p)


def main():
  '''
  Set meta-parameters and do a single run
  '''
  # Size of system
  N1 = 2 # number of resource users that benefit from extraction only
  N2 = 1 # number of users with both extractive and non-extractive use
  N3 = 0  # number of users with only non-extractive use
  K = 0 # number of bridging orgs
  M = 2  # number of gov orgs
  T = N1 + N2 + N3 + K + M + 1  # total number of state variables

  # Connectance of system
  C = 0.1  # Connectance between governance organizations and resource users.
            # (proportion of resource extraction/access interactions influenced by governance)
            
  np.random.seed(667)

  return run_system(N1,N2,N3,K,M,T,C)


if __name__ == "__main__":
  (N1,N2,N3,K,M,T,C,stability, J, converged, strategy_history, grad, total_connectance,
      phi,psis,alphas,betas,beta_hats,beta_tildes,beta_bars,sigmas,etas,lambdas,eta_bars,mus,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,du_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n,
      F,H,W,K_p) = main()