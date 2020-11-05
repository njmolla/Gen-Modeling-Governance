import numpy as np


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
dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,F_p,F_n,H_p,H_n,w_p,w_n,K_p,K_n,D_jm):
  # compute Jacobian (vectorized)
  J = np.zeros([T,T])
  # dr•/dr
  J[0,0] = phi*(ds_dr - np.sum(np.squeeze(psis*de_dr)))
                                            # 1xn
  # dr•/dx (1x(N))
  # For the NxMxN stuff: i = axis 0, m = axis 1, n = axis 2
  J[0,1:N+1] = -phi * np.sum(
        np.multiply(psis,np.sum(np.multiply(de_dg, dg_dF * (F_p - F_n)), axis = 1)),
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
        np.multiply(betas*db_de,     np.sum(np.multiply(de_dg, dg_dF*(F_p - F_n)), axis = 1))
                     #  1xn                                 ixmxn
        + np.multiply(beta_hats*dq_da, np.sum(np.multiply(da_dp, dp_dH*(H_p - H_n)), axis = 1))
        + np.multiply(beta_tildes,sigmas*dc_dw_p*w_p)
                       # 1xn            ixn
        - np.multiply(etas,lambdas*dc_dw_n*w_n)
       ))
  # dx•/dx for n = i (overwrite the diagonal)
  indices = np.arange(1,N+1)  # Access the diagonal of the actor part.
  J[indices,indices] = np.squeeze(alphas) * (
        np.squeeze(betas*db_de)*np.sum(np.squeeze(de_dg)*np.diagonal(dg_dF,axis1=0, axis2=2)*np.diagonal((F_p - F_n), axis1=0, axis2=2),axis=0)
        #                                                          mxn                                 mxn
        + np.squeeze(beta_hats*dq_da)*np.sum(np.squeeze(da_dp)*np.diagonal(dp_dH,axis1=0, axis2=2)*np.diagonal((H_p - H_n),axis1=0, axis2=2),axis=0)
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
