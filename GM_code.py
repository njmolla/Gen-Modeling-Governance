import numpy as np
import networkx as nx
from strategy_optimization import nash_equilibrium

def correct_scale_params(scale_params,alloc_params,i):
  '''
  Corrects scale parameters (either sigmas or lambdas) to be consisent with optimization
  results. Takes in scale parameters (2d) and strategy parameters for a particular actor (1d), and
  sets scale parameters to 0 if the corresponding strategy parameters are 0, then ensures
  that the scale parameters still add to 1.
  '''
  scale_params[:,i][alloc_params==0] = 0
  for i in range(sum(alloc_params==0)):
    scale_params[alloc_params==0][i][scale_params[alloc_params==0][i]!=0] = np.squeeze(np.random.dirichlet(np.ones(len(scale_params[alloc_params==0][i][scale_params[alloc_params==0][i]!=0])),1))
  return scale_params


def determine_stability(N,K,M,T,phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,
theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,
dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,F,H,w_p,w_n,K_p,K_n,D_jm):
  # compute Jacobian (vectorized)
  J = np.zeros([T,T])
  # dr•/dr
  J[0,0] = phi*(ds_dr - np.sum(np.squeeze(psis*de_dr)))
                                            # 1xn
  # dr•/dx (1x(N))
  # For the NxMxN stuff: i = axis 0, m = axis 1, n = axis 2
  J[0,1:N+1] = -phi * np.sum(
        np.multiply(psis,np.sum(np.multiply(de_dg, dg_dF * F),axis = 1)),
                                          # 1xmxn   ixmxn
       axis = 1)

  # dr•/dy
  J[0,N+1:] = -phi * np.sum(
        np.multiply(psis, np.squeeze(de_dg) * dg_dy),
                   # 1xn             1xmxn      mxn
       axis = 1)
  # dx•/dr
  J[1:N+1,0] = np.squeeze(alphas * (betas*db_de*de_dr + beta_hats*dq_da*da_dr))
                                           # 1xn
  # dx•/dx for n != i
  J[1:N+1,1:N+1] = np.transpose(np.multiply(alphas,
        np.multiply(betas*db_de,     np.sum(np.multiply(de_dg,dg_dF*F),axis = 1))
                     #  1xn                                 ixmxn
        + np.multiply(beta_hats*dq_da, np.sum(np.multiply(da_dp,dp_dH*H),axis = 1))
        + np.multiply(beta_tildes,sigmas*dc_dw_p*w_p)
                       # 1xn            ixn
        - np.multiply(etas,lambdas*dc_dw_n*w_n)
       ))
  # dx•/dx for n = i (overwrite the diagonal)
  indices = np.arange(1,N+1)  # Access the diagonal of the actor part.
  J[indices,indices] = np.squeeze(alphas) * (
        np.squeeze(betas*db_de)*np.sum(np.squeeze(de_dg)*np.diagonal(dg_dF,axis1=0, axis2=2)*np.diagonal(F,axis1=0, axis2=2),axis=0)
        #                                                          mxn                                 mxn
        + np.squeeze(beta_hats*dq_da)*np.sum(np.squeeze(da_dp)*np.diagonal(dp_dH,axis1=0, axis2=2)*np.diagonal(H,axis1=0, axis2=2),axis=0)
        - eta_bars*dl_dx
      )
  # dx•/dy
  J[1:N+1,N+1:] = np.transpose(alphas * (
        np.multiply(betas*db_de,np.squeeze(de_dg)*dg_dy))
                    # 1n   1n               1mn     mn
        + np.multiply(beta_hats*dq_da,np.squeeze(da_dp)*dp_dy)
      )

  # dy•/dr = 0
  # dy•/dx, result is mxi
  J[N+1:,1:N+1] = np.transpose(np.multiply(mus,
      np.multiply(rhos,di_dK_p*K_p)
      - np.multiply(thetas,di_dK_n*K_n)
      + np.multiply(np.squeeze(rho_bars,axis=2),np.sum(np.multiply(omegas,dt_dD_jm*D_jm),axis=2))
                                                                         # ixmxj
      - np.multiply(np.squeeze(theta_bars,axis=1),np.sum(np.multiply(epsilons,dt_dD_jm*D_jm), axis=1))))
                                                                               # ixjxm

  # dy•/dy for m != j, result is mxj
  J[N+1:,N+1:] = np.multiply(np.transpose(mus),(np.squeeze(
                                         # 1m
        np.multiply(rho_bars,omegas*dtmj_dym)
                    # 1m1        1mj
        - np.transpose(np.multiply(theta_bars,epsilons*dtjm_dym),(0,2,1))
                                    # 11m            1jm
      )))

  # dy•/dy for m = j
  indices = np.arange(N+1,T)  # Access the diagonal of the governing agency part.
  J[indices,indices] = np.squeeze(mus*rhos*di_dy_p - thetas*di_dy_n
      + np.squeeze(rho_bars, axis = 2)*np.sum(omegas*dtjm_dym, axis = 2)
      - np.squeeze(theta_bars, axis = 1)*np.sum(epsilons*dtmj_dym, axis=1))
  # compute the eigenvalues of the Jacobian
  eigvals = np.linalg.eigvals(J)
  if all(eigvals.real) < 10e-5: # stable if real part of eigenvalues is negative
    stability = True
  else:
    stability = False # unstable if real part is positive, inconclusive if 0
  return J,eigvals,stability


def sample(N1,N2,N3,K,M,T,C1,C2):  ## Sunshine: Maybe comment on each parameter that has multiple N's or M's
                                   ## to disambiguate what each index is.
  N = N1 + N2 + N3 + K
  is_connected = False
  while is_connected == False:
    # initialize arrays for parameters
    # scale parameters
    phi = np.random.rand(1) # $
    psis = np.zeros([1,N]) # $
    psis[0:N1+N2] = np.random.dirichlet(np.ones(N1+N2),1) # need to sum to 1
    alphas = np.random.rand(1,N) # $
    betas = np.zeros([1,N]) # $
    beta_hats = np.zeros([1,N]) # $
    beta_tildes = np.zeros([1,N]) # $

    # beta parameters for extractors
    betas[0,0:N1] = np.random.rand(N1)
    beta_tildes[0,0:N1] = 1 - betas[0,0:N1]
    # beta parameters for resource users with both uses
    beta_params = np.random.dirichlet(np.ones(3),N2-N1+1).transpose()
    betas[0,N1:N2+N1] = beta_params[0]
    beta_tildes[0,N1:N2+N1] = beta_params[1]
    beta_hats[0,N1:N2+N1] = beta_params[2]
    # beta parameters for resource users with non-extractive use
    beta_tildes[0,N2+N1:N] =  np.random.rand(N3)
    beta_hats[0,N1+N2:N1+N2+N3] = 1 - beta_tildes[0,N1+N2:N1+N2+N3]
    # beta parameters for bridging organizations
    beta_tildes[0,N-K:N] = np.ones(K) # one for all bridging orgs

    sigmas = np.zeros([N,N]) # sigma_k,n is kxn $
    sigmas = np.random.dirichlet(np.ones(N),N)

    etas = np.random.rand(1,N) # $
    eta_bars = np.squeeze(1-etas)

    lambdas = np.zeros([N,N])  # lambda_k,n is kxn $
    lambdas = np.random.dirichlet(np.ones(N),N)

    mus = np.random.rand(1,M) # $

    rhos = np.random.rand(1,M) # $
    rho_bars = np.reshape(1 - rhos,(1,M,1)) # want to be 1,m,1$

    omegas = np.zeros([1,M,M]) # omegas_m,j is 1xjxm $

    if M != 1:
        links = np.random.rand(1,M,M) < C2 # gov org to  gov org connectance
        omegas = construct_links(omegas,links)


    thetas = np.random.rand(1,M)# $
    theta_bars = (1 - thetas).reshape(1,1,M) # want to be 1,1,m $

    theta_bars_j = np.sum(np.multiply(rho_bars,omegas),axis=2)

    epsilons = np.zeros([1,M,M]) # epsilon_m,j is 1xjxm $
    epsilons = np.multiply(omegas,np.squeeze(rho_bars,axis=2)/
                           np.where(theta_bars_j==0,np.nan,theta_bars_j))
    epsilons[np.isnan(epsilons)] = 0

    # exponent parameters
    ds_dr = np.random.rand(1)*2 #0-2 $
    de_dr = np.random.rand(1,N) # $
    de_dg = np.zeros((1,M,N)) # $
    links = np.random.rand(N1+N2)<C1
    #resample until at least one gov-extraction interaction
    while np.count_nonzero(links) == 0:
      links = np.random.rand(N1+N2)<C1
    de_dg[:,:,0:N1+N2][:,:,links] = np.random.uniform(-1,1,(1,M,sum(links)))
    dg_dF = np.random.uniform(-2,2,(N,M,N)) #dg_m,n/(dF_i,m,n * x_i) is ixmxn $
    dg_dy = np.random.rand(M,N)*2 # $
    dp_dy = np.random.rand(M,N)*2 # $
    db_de = np.random.rand(1,N)*2 # $
    da_dr = np.random.rand(1,N)*2 # $
    dq_da = np.random.uniform(-2,2,(1,N)) # $
    da_dp = np.zeros((1,M,N)) # $
    links = np.random.rand(N2+N3)<C1
    da_dp[:,:,N1:N-K][:,:,links] = np.random.uniform(-1,1,(1,M,sum(links)))
    dp_dH = np.random.uniform(0,2,(N,M,N)) # dp_m,n/(dH_i,m,n * x_i) is ixmxn $
    dc_dw_p = np.random.uniform(0,2,(N,N)) #dc_dw_p_i,n is ixn $
    dc_dw_n = np.random.uniform(0,2,(N,N)) #dc_dw_n_i,n is ixn $
    dl_dx = np.random.rand(N)
    di_dK_p = np.random.uniform(0,2,(N,M)) # $
    di_dK_n = np.random.uniform(0,2,(N,M)) # $
    dt_dD_jm = np.random.uniform(0,2,(N,M,M)) #dt_j->m/d(D_i,j->m * x_i) is ixmxj  $
    di_dy_p = np.random.rand(1,M) # $
    di_dy_n = np.random.rand(1,M) # $
    dtjm_dym = np.random.rand(M,M) # 1xmxj
    dtmj_dym = np.random.uniform(-1,0,(1,M,M))# 1xjxm
    # Effort allocation parameters
    # initialize guesses for parameters
    F = np.random.rand(N,M,N) # F_i,m,n is ixmxn effort for influencing resource extraction governance $
    H = np.random.rand(N,M,N) # effort for influencing resource access governance $
    w_p = np.random.rand(N,N) # effort for collaboration. W_i,n is ixn $
    w_n = np.random.rand(N,N) # effort for undermining. W_i,n is ixn $
    K_p = np.random.rand(N,M) # effort for more influence for gov orgs $
    K_n = np.random.rand(N,M) # effort for less influence for gov orgs $
    D_jm = np.random.rand(N,M,M) # D_i,j,m is ixmxj effort for transferring power from each gov org to each other $
    # strategy optimization
    # calculate Jacobian
    J,eigvals,stability = determine_stability(N,K,M,T, phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,
        theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,
        dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,F,H,w_p,w_n,K_p,K_n,D_jm)
    # find nash equilibrium strategies
    nash_equilibrium(100,J,N,K,M,T,phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,
                     theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,
                     dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,F,H,w_p,w_n,K_p,K_n,D_jm)

    is_connected = True
    # compute Jacobian to check whether system is weakly connected
    J,eigvals,stability = determine_stability(N,K,M,T, phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,
        theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,
        dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,F,H,w_p,w_n,K_p,K_n,D_jm)


    adjacency_matrix = np.zeros([T,T])
    adjacency_matrix[J!=0] = 1
    graph = nx.from_numpy_array(adjacency_matrix,create_using=nx.DiGraph)
    is_connected = nx.is_weakly_connected(graph)

  return phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas, \
      theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n, \
      dt_dD_jm, di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,F,H,w_p,w_n,K_p,K_n,D_jm


# Size of system
N1 = 1 # number of resource users that benefit from extraction only
N2 = 0 # number of users with both extractive and non-extractive use
N3 = 2 # number of users with only non-extractive use
K = 0 # number of bridging orgs
N = N1 + N2 + N3 + K # total number of resource users
M = 1 # number of gov orgs
T = N + M + 1 # total number of state variables
# Connectance of system (for different interactions)
C1 = 0.5 # proportion of resource extraction/access interactions influenced by governance
C2 = 0.2 # gov org-gov org connectance

def main():
  num_stable_webs = 0
  num_samples = 1
  for _ in range(num_samples):
    phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas, \
        theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n, \
        dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,F,H,w_p,w_n,K_p,K_n,D_jm = sample(N1,N2,N3,K,M,T,C1,C2)

    total_connectance = (np.count_nonzero(de_dg) + np.count_nonzero(da_dp)
        + np.count_nonzero(F) + np.count_nonzero(H) + np.count_nonzero(w_p)
        + np.count_nonzero(w_n) + np.count_nonzero(K_p) + np.count_nonzero(K_n)
        + np.count_nonzero(omegas) + np.count_nonzero(epsilons) + np.count_nonzero(D_jm)) \
        /(np.size(de_dg) + np.size(da_dp) + np.size(F) + np.size(H)
        + np.size(w_p) + np.size(w_n) + np.size(K_p) + np.size(K_n) + np.size(omegas)
        + np.size(epsilons) + np.size(D_jm))

    if stability:
      num_stable_webs += 1
  print(num_stable_webs/num_samples)
  return phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas, \
      theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n, \
      dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,F,H,w_p,w_n,K_p,K_n,D_jm,J

if __name__ == "__main__":
  phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas, \
      theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n, \
      dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,F,H,w_p,w_n,K_p,K_n,D_jm,J = main()
