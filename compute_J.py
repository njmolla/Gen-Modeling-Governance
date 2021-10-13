import numpy as np
from numba import jit


@jit(nopython=True)
def compute_Jacobian(N,K,M,T,
      phi,psis,alphas,betas,beta_hats,beta_tildes,beta_bars,sigmas,etas,lambdas,eta_bars,mus,
      ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,du_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n,
      F,H,W,K_p):

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
        np.multiply(psis,np.sum(np.multiply(de_dg, dg_dF * F), axis = 1)),axis = 1)
                                          # 1xmxn   ixmxn

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
  assign_when(W_p, W, W>=0)
  W_n = np.zeros((N,N))
  assign_when(W_n, W, W<0)
  J[1:N+1,1:N+1] = np.transpose(np.multiply(alphas,
        np.multiply(betas*db_de,       np.sum(np.multiply(de_dg, dg_dF*F), axis = 1))
                     #  1xn                                ixmxn
        + np.multiply(beta_hats*dq_da, np.sum(np.multiply(da_dp, dp_dH*H), axis = 1))
        + np.multiply(beta_tildes,sigmas*dc_dw_p*W_p)
                       # 1xn            ixn
        - np.multiply(etas,lambdas*dc_dw_n*W_n)
      ))

 # dx•/dx for n = i (overwrite the diagonal)
  for i in range(N):

    J[i+1,i+1] = alphas[0,i] * (
                     betas[0,i]     * db_de[0,i] * np.sum(de_dg[0,:,i] * dg_dF[i,:,i] * F[i,:,i])
                   + beta_hats[0,i] * dq_da[0,i] * np.sum(da_dp[0,:,i] * dp_dH[i,:,i] * H[i,:,i])
                   + beta_bars[0,i] * du_dx[i]
                   - eta_bars[i] * dl_dx[i]
                 )

#  indices = np.arange(1,N+1)  # Access the diagonal of the actor part.
#  J[indices,indices] = alphas[0] * (
#        (betas*db_de)[0]*np.sum(de_dg[0]*dg_dF[np.arange(N),:,np.arange(N)].transpose()*F[np.arange(N),:,np.arange(N)].transpose(),axis=0)
#        #                                                          mxn                                 mxn
#        + (beta_hats*dq_da)[0]*np.sum(da_dp[0]*dp_dH[np.arange(N),:,np.arange(N)].transpose()*H[np.arange(N),:,np.arange(N)].transpose(),axis=0)
#        - eta_bars*dl_dx
#      )

  # dx•/dy
  J[1:N+1,N+1:] = np.transpose(alphas * (
        np.multiply(betas*db_de, de_dg[0]*dg_dy)
                    # 1n   1n                1mn     mn
        + np.multiply(beta_hats*dq_da, da_dp[0]*dp_dy)
      ))

  # dy•/dr = 0
  # dy•/dx, result is mxi
  K_plus = np.zeros((N,M))
  assign_when(K_plus, K_p, K_p>=0)
  K_n = np.zeros((N,M))
  assign_when(K_n, np.abs(K_p), K_p<0)
  J[N+1:,1:N+1] = np.transpose(np.multiply(mus,di_dK_p*K_plus - di_dK_n*K_n))  # ixjxm

#
  # dy•/dy = 0 for m != j


  # dy•/dy for m = j
  for i in range(M):
    J[-M+i:,-M+i:] = mus[0,i]*(di_dy_p[0,i] - di_dy_n[0,i])

#  indices = np.arange(N+1,T)  # Access the diagonal of the governing agency part.
#  J[indices,indices] = mus[0]*(di_dy_p - di_dy_n)[0]

  return J


@jit(nopython=True)
def assign_when(lhs, rhs, conditions):
  """
  This does
     lhs[conditions] = rhs[conditions]
  (this is done because Numba doesn't like logical indexing)
  lhs and rhs are any numpy arrays with the same shape
  conditions is a boolean numpy array with the same shape
  For example, to do
     x[x > 0] = y[x > 0]
  use
      assign_when(x, y, x > 0)
  """
  for nd_index in np.ndindex(lhs.shape):
    if conditions[nd_index]:
      lhs[nd_index] = rhs[nd_index]