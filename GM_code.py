import numpy as np
import networkx as nx
from compute_J import determine_stability
from strategy_optimization import nash_equilibrium

def sample(N1,N2,N3,K,M,T, C1,C2):
  N = N1 + N2 + N3 + K
  is_connected = False
  while is_connected == False:
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
    eta_bars = np.squeeze(1-etas) # TODO: fix for 1 actor/no undermining

    lambdas = np.zeros([N,N])  # lambda_k,n is kxn $
    lambdas = np.random.dirichlet(np.ones(N),N)

    mus = np.random.rand(1,M) # $

    rhos = np.random.rand(1,M) # $ should be 1 for M=1 based on rho_bar
    rho_bars = np.reshape(1 - rhos,(1,M,1)) # want to be 1,m,1$

    omegas = np.zeros([1,M,M]) # omegas_m,j is 1xjxm, want to sum to 1 along m

    if M != 1:
        links = (np.random.rand(1,M,M) < C2) # gov org to  gov org connectance
        links_per_col = np.squeeze(np.sum(links, axis=1))
        for i in range(M):
          # skip if no links in that col
          if links_per_col[i] == 0:
            continue
          else:
            omegas[0,links[0,:,i],i] = np.squeeze(np.random.dirichlet(np.ones(links_per_col[i]),1))

    thetas = np.random.rand(1,M)# $
    theta_bars = (1 - thetas).reshape(1,1,M) # want to be 1,1,m $

    theta_bars_j = np.sum(np.multiply(rho_bars,omegas),axis=2)

    epsilons = np.zeros([1,M,M]) # epsilon_m,j is 1xjxm $
    epsilons = np.multiply(omegas,np.squeeze(rho_bars,axis=2)/
                           np.where(theta_bars_j == 0,np.nan,theta_bars_j))
    epsilons[np.isnan(epsilons)] = 0


    # ------------------------------------------------------------------------
    # Initialize exponent parameters
    # ------------------------------------------------------------------------
    ds_dr = np.random.rand(1)*2  # 0-2 $
    de_dr = np.random.rand(1,N)  # $
    de_dg = np.zeros((1,M,N))  # $
    links = np.random.rand(N1+N2) < C1
    # resample until at least one gov-extraction interaction
    while np.count_nonzero(links) == 0:
      links = np.random.rand(N1+N2) < C1
    de_dg[:,:,0:N1+N2][:,:,links] = np.random.uniform(-1,1,(1,M,sum(links)))
    dg_dF = np.random.uniform(0,2,(N,M,N))  # dg_m,n/(dF_i,m,n * x_i) is ixmxn $###############
                                            # should be positive!
    dg_dy = np.random.rand(M,N)*2 # $
    dp_dy = np.random.rand(M,N)*2 # $
    db_de = np.random.rand(1,N)*2 # $
    da_dr = np.random.rand(1,N)*2 # $
    dq_da = np.random.uniform(-2,2,(1,N)) # $
    da_dp = np.random.uniform(-1,1,((1,M,N)))
    links = np.random.rand(N2+N3) < C1
    da_dp[:,:,N1:N-K][:,:,links] = np.random.uniform(-1,1,(1,M,sum(links)))
    dp_dH = np.random.uniform(0,2,(N,M,N)) # dp_m,n/(dH_i,m,n * x_i) is ixmxn $
    dc_dw_p = np.random.uniform(0,2,(N,N)) #dc_dw_p_i,n is ixn $
    indices = np.arange(0,N)
    dc_dw_p[indices,indices] = 0
    dc_dw_n = np.random.uniform(0,2,(N,N)) #dc_dw_n_i,n is ixn $
    dc_dw_n[indices,indices] = 0
    dl_dx = np.random.rand(N)
    di_dK_p = np.random.uniform(0,2,(N,M))
    di_dK_n = np.random.uniform(0,2,(N,M))
    di_dy_p = np.random.rand(1,M)  # $
    di_dy_n = np.random.rand(1,M)  # $
    dt_dD_jm = np.random.uniform(0,2,(N,M,M))  # dt_j->m/d(D_i,j->m * x_i) is ixmxj  $
    dtjm_dym = np.random.rand(M,M)  # 1xmxj
    dtmj_dym = np.random.uniform(-1,0,(1,M,M))  # 1xjxm


    # ------------------------------------------------------------------------
    # Effort allocation parameters, initial guesses
    # ------------------------------------------------------------------------
    F = np.random.rand(N,M,N)  # F_i,m,n is ixmxn positive effort for influencing resource extraction governance $
    H = np.random.rand(N,M,N)  # effort for influencing resource access governance $
    W = np.random.rand(N,N)  # effort for collaboration. W_i,n is ixn $
    K_p = np.random.rand(N,M)  # effort for more influence for gov orgs $
    D_jm = np.random.rand(N,M,M)  # D_i,j,m is ixmxj effort for transferring power from each gov org to each other $


    # ------------------------------------------------------------------------
    # Strategy optimization
    # ------------------------------------------------------------------------

    # calculate Jacobian
    J, eigvals, stability = determine_stability(N,K,M,T,
        phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,
        F,H,W,K_p,D_jm)
    adjacency_matrix = np.zeros([T,T])
    adjacency_matrix[J != 0] = 1
    graph = nx.from_numpy_array(adjacency_matrix,create_using=nx.DiGraph)
    is_connected = nx.is_weakly_connected(graph)
    if is_connected == False:
      continue

#    # find nash equilibrium strategies
#    F_p,F_n,H_p,H_n,W_p,W_n,K_p,K_n,D_jm, sigmas, lambdas, converged = nash_equilibrium(2000, J, N,K,M,T,
#        phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,
#        F_p,F_n,H_p,H_n,W_p,W_n,K_p,K_n,D_jm)
    converged = True

    # ------------------------------------------------------------------------
    # See if system is stable and if it is weakly connected
    # ------------------------------------------------------------------------

    # check stability and use Jacobian to check whether system is weakly connected
    J, eigvals, stability = determine_stability(N,K,M,T,
        phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,
        F,H,W,K_p,D_jm)

    adjacency_matrix = np.zeros([T,T])
    adjacency_matrix[J != 0] = 1
    graph = nx.from_numpy_array(adjacency_matrix,create_using=nx.DiGraph)
    is_connected = nx.is_weakly_connected(graph)
    if is_connected == False:
      print('not weakly connected')

  return (stability, J, converged,
      phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,
      F,H,W,K_p,D_jm)


def run_multiple(size,C1,C2,num_samples):
  '''
  Run num_samples samples and return the proportion of webs that are stable.
  '''

  num_stable_webs = 0
  num_stable_webs_filtered = 0
  num_converged = 0

  for _ in range(num_samples):
      # Need at least 2 resource users and one gov org
    N = 2
    M = 1
    rand = np.random.rand(size-3)
    N += np.sum(rand < 0.6)
    K = np.sum(rand < 0.8) - np.sum(rand < 0.6)
    M += np.sum(rand > 0.8)
    rand2 = np.random.rand(N-1)
    # choose at random whether guaranteed extractor is just extractor or extractor + accessor
    rand3 = np.random.rand(1)
    if rand3 < 0.5:
      N1 = 1 + np.sum(rand2 < 0.33)
      N2 = np.sum(rand2 > 0.33) - np.sum(rand2 > 0.66)
    else:
      N1 = np.sum(rand2 < 0.33)
      N2 = 1 + np.sum(rand2 > 0.33) - np.sum(rand2 > 0.66)
    N3 = np.sum(rand2 > 0.66)
    N = N1 + N2 + N3 + K # total number of resource users
    T = N + M + 1 # total number of state variables
    print((N1,N2,N3,K,M))
    try:
      result = sample(N1,N2,N3,K,M,T,C1,C2)
    except:
      continue
    stability = result[0] # stability is the first return value
    converged = result[2]
    if stability:
      num_stable_webs += 1
      if converged:
        num_stable_webs_filtered += 1
    if converged:
      num_converged += 1

  return num_stable_webs, num_stable_webs_filtered, num_converged  # proportion of stable webs


def run_once(N1,N2,N3,K,M,T, C1,C2):
  '''
  Do a single run and return more detailed output.
  '''
  np.random.seed(5)
  (stability, J, converged,
      phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,
      F,H,W,K_p,D_jm) = sample(N1,N2,N3,K,M,T,C1,C2)

#  total_connectance = (np.count_nonzero(de_dg) + np.count_nonzero(da_dp)
#      + np.count_nonzero(F_p) + np.count_nonzero(F_n) + np.count_nonzero(H_p) + np.count_nonzero(H_p)
#      + np.count_nonzero(W_p) + np.count_nonzero(W_n) + np.count_nonzero(K_p) + np.count_nonzero(K_n)
#      + np.count_nonzero(omegas) + np.count_nonzero(epsilons) + np.count_nonzero(D_jm)) \
#      /(np.size(de_dg) + np.size(da_dp) + np.size(F_p) + np.size(F_n) + np.size(H_p) + np.size(H_n)
#      + np.size(W_p) + np.size(W_n) + np.size(K_p) + np.size(K_n) + np.size(omegas)
#      + np.size(epsilons) + np.size(D_jm))

  return (stability, J,
          phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,
          F,H,W,K_p,D_jm)


def main():
  # Size of system
  N1 = 1 # number of resource users that benefit from extraction only
  N2 = 0 # number of users with both extractive and non-extractive use
  N3 = 0  # number of users with only non-extractive use
  K = 0 # number of bridging orgs
  M = 1  # number of gov orgs
  T = N1 + N2 + N3 + K + M + 1  # total number of state variables

  # Connectance of system (for different interactions)
  C1 = 0.2  # Connectance between governance organizations and resource users.
            # (proportion of resource extraction/access interactions influenced by governance)
  C2 = 0.2  # Connectance between governance organizations and other governance organizations.

  return run_once(N1,N2,N3,K,M,T, C1,C2)


if __name__ == "__main__":
  (stability, J,
      phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,
      F,H,W,K_p,D_jm) = main()



def test_calibration():
  (stability, total_connectance, J,
      phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,
      F,H,W,K_p,D_jm) = main()

