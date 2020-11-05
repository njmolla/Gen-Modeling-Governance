from GM_code import run
import numpy as np

size_ranges = np.arange(5,15,1)
connectance_ranges = np.linspace(0.2,0.6,10)
PSW = np.zeros((len(size_ranges),10))

for i,size in enumerate(size_ranges):
  for j,C1 in enumerate(connectance_ranges):
    rand = np.random.rand(size)
    N = np.sum(rand<0.5)
    #K = np.sum(rand>0.4) - np.sum(rand>0.6)
    K = 0
    M = np.sum(rand>0.5)
    rand2 = np.random.rand(N)
    N1 = np.sum(N<0.33)
    N2 = np.sum(N>0.33) - np.sum(N>0.66)
    N3 = np.sum(N>0.66)
    N = N1 + N2 + N3 + K # total number of resource users
    T = N + M + 1 # total number of state variables
    C2 = 0.2 # gov org-gov org connectance
    PSW[i,j], total_connectance, phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas, \
    theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n, \
    dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,F_p,F_n,H_p,H_n,w_p,w_n,K_p,K_n,D_jm,J = run(N1,N2,N3,K,M,T,C1,C2,1)



