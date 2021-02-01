import numpy as np
from numba import jit


#@jit(nopython=True)
def determine_stability(N,K,M,T,
	  phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,
	  F,H,W,K_p,D_jm):

  # --------------------------------------------------------------------------
  # Compute Jacobian (vectorized)
  # --------------------------------------------------------------------------
  J = np.zeros((T,T))
  # dr•/dr
  J[0,0] = (phi*(ds_dr - np.sum(psis*de_dr)))[0]
                               # 1xn
  # dr•/dx (1x(N))
  # For the NxMxN stuff: i = axis 0, m = axis 1, n = axis 2
  J[0,1:N+1] = -phi * np.sum(
        np.multiply(psis,np.sum(np.multiply(de_dg, dg_dF * F), axis = 1)),
                                          # 1xmxn   ixmxn
       axis = 1)

  # dr•/dy
  J[0,N+1:] = -phi * np.sum(
        np.multiply(psis, de_dg[0] * dg_dy),
                   # 1xn             1xmxn     mxn
       axis = 1)

  # dx•/dr
  J[1:N+1,0] = (alphas * (betas*db_de*de_dr + beta_hats*dq_da*da_dr))[0]
                                           # 1xn
  # dx•/dx for n != i
  W_p = np.zeros((N,N))
  W_p[W>0.001] = W[W>0.001]
  W_n = np.zeros((N,N))
  W_n[W<-0.001] = W[W<-0.001]
  J[1:N+1,1:N+1] = np.transpose(np.multiply(alphas,
        np.multiply(betas*db_de,       np.sum(np.multiply(de_dg, dg_dF*F), axis = 1))
                     #  1xn                                ixmxn
        + np.multiply(beta_hats*dq_da, np.sum(np.multiply(da_dp, dp_dH*H), axis = 1))
        + np.multiply(beta_tildes,sigmas*dc_dw_p*W_p)
                       # 1xn            ixn
        - np.multiply(etas,lambdas*dc_dw_n*W_n)
      ))

  # dx•/dx for n = i (overwrite the diagonal)
  indices = np.arange(1,N+1)  # Access the diagonal of the actor part.
  J[indices,indices] = alphas[0] * (
        (betas*db_de)[0]*np.sum(de_dg[0]*np.diagonal(dg_dF,axis1=0, axis2=2)*np.diagonal(F, axis1=0, axis2=2),axis=0)
        #                                                          mxn                                 mxn
        + (beta_hats*dq_da)[0]*np.sum(da_dp[0]*np.diagonal(dp_dH,axis1=0, axis2=2)*np.diagonal(H,axis1=0, axis2=2),axis=0)
        - eta_bars*dl_dx
      )
  # dx•/dy
  J[1:N+1,N+1:] = np.transpose(alphas * (
        np.multiply(betas*db_de, de_dg[0]*dg_dy)
                    # 1n   1n                1mn     mn
        + np.multiply(beta_hats*dq_da, da_dp[0]*dp_dy)
      ))

  # dy•/dr = 0
  # dy•/dx, result is mxi
  K_plus = np.zeros((N,M))
  K_plus[K_p>0] = K_p[K_p>0]
  K_n = np.zeros((N,M))
  K_n[K_p<0] = abs(K_p[K_p<0])
  J[N+1:,1:N+1] = np.transpose(np.multiply(mus,
        np.multiply(rhos,di_dK_p*K_plus)
        - np.multiply(thetas,di_dK_n*K_n)
        + np.multiply(rho_bars[:,:,0],np.sum(np.multiply(omegas,dt_dD_jm*D_jm),axis=2))
                                                                         # ixmxj
        - np.multiply(theta_bars[:,0,:],np.sum(np.multiply(epsilons,dt_dD_jm*D_jm), axis=1))
                                                                               # ixjxm
      ))

  # dy•/dy for m != j, result is mxj
  J[N+1:,N+1:] = np.multiply(np.transpose(mus),((
                                         # 1m
        np.multiply(rho_bars,omegas*dtmj_dym)
                    # 1m1        1mj
        - np.transpose(np.multiply(theta_bars,epsilons*dtjm_dym),(0,2,1))
                                    # 11m            1jm
      )[0]))

  # dy•/dy for m = j
  indices = np.arange(N+1,T)  # Access the diagonal of the governing agency part.
  J[indices,indices] = mus[0]*(rhos*di_dy_p - thetas*di_dy_n
        + rho_bars[:,:,0]*np.sum(omegas*dtjm_dym, axis = 2)
        - theta_bars[:,0,:]*np.sum(epsilons*dtmj_dym, axis=1)
      )[0]


  # --------------------------------------------------------------------------
  # Compute the eigenvalues of the Jacobian
  # --------------------------------------------------------------------------
  eigvals = np.linalg.eigvals(J)
  if np.all(eigvals.real < 10e-5):  # stable if real part of eigenvalues is negative
    stability = True
  else:
    stability = False  # unstable if real part is positive, inconclusive if 0

  return J, eigvals, stability
