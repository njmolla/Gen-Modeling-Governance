import numpy as np
from compute_J import determine_stability
from numba import jit

#@jit(nopython=True)
def objective_grad(strategy, n, l, J, N,K,M,T,
    phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,
    F,H,W,K_p,D_jm):
  '''
  inputs:
    strategy for a single actor (flattened)
    n is the actor whose objective we want to optimize
    l is the actor whose strategy it is
    J is the Jacobian (is it calculated or passed in?????)  TODO: remove this parameter
    N,K,M,T meta parameters
    scale parameters
    exponent parameters
    strategy parameters
  modified variables:
    strategy parameters F,H,W,K_p,D_jm will be modified to match input strategy for l
  return the gradient of the objective function at that point for that actor
  '''
  # Unpack strategy parameters.
  F[l] = strategy[0:M*N].reshape([M,N])
  H[l] = strategy[M*N:2*M*N].reshape([M,N])
  W[l] = strategy[2*M*N:2*M*N+N].reshape([N])
  K_p[l] = strategy[2*M*N+N:2*M*N+N+M].reshape([M])
  D_jm[l] = strategy[2*M*N+N+M:].reshape([M,M])

  # Compute Jacobian
  J, eigvals, stability = determine_stability(N,K,M,T,
      phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,
      F,H,W,K_p,D_jm)

  # Compute inverse Jacobian
  J_inv = np.linalg.inv(J)

  #TO DO: This section doesn't need to be calculated every time (except for dxdot_dws)
  # Compute how the rhs of system changes with respect to each strategy parameter
  drdot_dF = -phi*np.multiply(np.reshape(psis,(1,1,N)),np.multiply(de_dg,dg_dF))
  dxdot_dF = np.zeros([N,N,M,N])
  dxdot_dF[np.arange(0,N),:,:,np.arange(0,N)] = np.transpose(np.multiply(
             #n k m (gets rid of last index)
        np.reshape(alphas*betas*db_de, (1,1,N)),
                    # 1n    1n    1n
        np.multiply(de_dg,dg_dF)
                  # 1mn    kmn
      ), (2,0,1))  # transpose kmn -> nkm
  dydot_dF = np.zeros([M,N,M,N])

  drdot_dH = np.zeros([N,M,N])
  dxdot_dH = np.zeros([N,N,M,N])
  dxdot_dH[np.arange(0,N),:,:,np.arange(0,N)] = np.transpose(np.multiply(np.reshape(alphas*beta_hats*dq_da,(1,1,N)),
                                                             np.multiply(da_dp,dp_dH)), (2,0,1))
  dydot_dH = np.zeros([M,N,M,N])

  drdot_dW_p = np.zeros([N,N])
  dxdot_dW_p = np.zeros([N,N,N])
  # result is nxkxi
  dxdot_dW_p[np.arange(0,N),:,np.arange(0,N)] = np.transpose(np.multiply(alphas*beta_tildes,np.multiply(sigmas,dc_dw_p)))
  dydot_dW_p = np.zeros([M,N,N])

  drdot_dW_n = np.zeros([N,N])
  dxdot_dW_n = np.zeros([N,N,N])
  dxdot_dW_n[np.arange(0,N),:,np.arange(0,N)] = np.transpose(np.multiply(-alphas*etas,np.multiply(lambdas,dc_dw_n)))
  dydot_dW_n = np.zeros([M,N,N])

  drdot_dK_p = np.zeros([N,M])
  dxdot_dK_p = np.zeros([N,N,M])
  dydot_dK_p = np.zeros([M,N,M])
  # result is mxn
  dydot_dK_p[np.arange(0,M),:,np.arange(0,M)] = np.transpose(np.multiply(mus*rhos,di_dK_p))

  drdot_dK_n = np.zeros([N,M])
  dxdot_dK_n = np.zeros([N,N,M])
  dydot_dK_n = np.zeros([M,N,M])
  dydot_dK_n[np.arange(0,M),:,np.arange(0,M)] = np.transpose(np.multiply(-mus*thetas,di_dK_n))

  drdot_dDjm = np.zeros([N,M,M])
  dxdot_dDjm = np.zeros([N,N,M,M])
  dydot_dDjm = np.zeros([M,N,M,M]) # mxnxjxl
  # case for the gov org being transferred to (m=l), result is mxnxj
  dydot_dDjm[np.diag_indices(M,1),:,:,np.diag_indices(M,1)] = np.transpose(np.multiply(mus*rho_bars,np.multiply(omegas,dt_dD_jm)),(1,0,2)) # eq ___, mxnxj
  # case for the gov org transferring (m=j), result is mxnxl
  dydot_dDjm[np.diag_indices(M,1),:,np.diag_indices(M,1),:] = np.multiply(np.reshape(np.transpose(-mus*np.squeeze(theta_bars)),(M,1,1)),np.transpose(np.multiply(epsilons,dt_dD_jm),(2,0,1))) # eq __, mxnxl

  ## Compute how the steady state of the system changes with respect to each strategy parameter
  dR_dF, dX_dF, dY_dF = multiply_by_inverse_jacobian(drdot_dF, dxdot_dF, dydot_dF, J_inv, T,N,M)
  dR_dH, dX_dH, dY_dH = multiply_by_inverse_jacobian(drdot_dH, dxdot_dH, dydot_dH, J_inv, T,N,M)
  dR_dW_p, dX_dW_p, dY_dW_p = multiply_by_inverse_jacobian(drdot_dW_p, dxdot_dW_p, dydot_dW_p, J_inv, T,N,M)
  dR_dW_n, dX_dW_n, dY_dW_n = multiply_by_inverse_jacobian(drdot_dW_n, dxdot_dW_n, dydot_dW_n, J_inv, T,N,M)
  dR_dK_p, dX_dK_p, dY_dK_p = multiply_by_inverse_jacobian(drdot_dK_p, dxdot_dK_p, dydot_dK_p, J_inv, T,N,M)
  dR_dK_n, dX_dK_n, dY_dK_n = multiply_by_inverse_jacobian(drdot_dK_n, dxdot_dK_n, dydot_dK_n, J_inv, T,N,M)
  dR_dDjm, dX_dDjm, dY_dDjm = multiply_by_inverse_jacobian(drdot_dDjm, dxdot_dDjm, dydot_dDjm, J_inv, T,N,M)


  # calculate gradients of objective function for one actor
  # for extraction
  # n's objective, l's strategy (same for resource users) n,l used to be i,j
  if betas[0,n] > 0.000001:  # Check if we are optimizing n's extraction
                             #### TODO: replace 0.000001 with named constant, put elsewhere too
    # jxi
    grad_e_F = de_dr[0,n] * dR_dF[l] + np.sum(np.multiply(np.reshape(de_dg[0,:,n]*dg_dy[:,n], (M,1,1)), dY_dF[:,l])
                         # scalar                               jxi         # m                             mji
            + np.sum(
                np.multiply(  # Both factors need to be kmji
                    np.reshape(np.multiply(de_dg[:,:,n],dg_dF[:,:,n]*F[:,:,n]), (N,M,1,1)),
                                               # 1m            km          km
                    dX_dF[:,l:l+1,:,:]
                       # k1ji
                )
            ,axis=0)  # Sum over k
        ,axis=0)  # Sum over m

    grad_e_F[:,n] += np.multiply(de_dg[0,:,n], dg_dF[n,:,n])

    grad_e_H = de_dr[0,n] * dR_dH[l] + np.sum(np.multiply(np.reshape(de_dg[0,:,n]*dg_dy[:,n], (M,1,1)), dY_dH[:,l])
                         # scalar                               jxi         # m                             mji
            + np.sum(
                np.multiply(  # Both factors need to be kmji
                    np.reshape(np.multiply(de_dg[:,:,n],dg_dF[:,:,n]*F[:,:,n]), (N,M,1,1)),
                                               # 1m            km          km
                    dX_dH[:,l:l+1,:,:]
                     # k1ji
                )
            ,axis=0)  # Sum over k
        ,axis=0)

    grad_e_W = np.zeros((N))
    grad_e_W_p = de_dr[0,n] * dR_dW_p[l] + np.sum(np.multiply(np.reshape(de_dg[0,:,n]*dg_dy[:,n], (M,1)), dY_dW_p[:,l])
              + np.sum(
                  np.multiply(  # Both factors need to be kmi
                      np.reshape(
                          np.multiply(
                              de_dg[:,:,n],
                              dg_dF[:,:,n] * F[:,:,n]
                          ), (N,M,1)
                      ),
                      dX_dW_p[:,l:l+1,:]  # k1i
                  )
              ,axis=0)  # Sum over k
          ,axis=0)

    grad_e_W_n =  de_dr[0,n] * dR_dW_n[l] + np.sum(np.multiply(np.reshape(de_dg[0,:,n]*dg_dy[:,n], (M,1)), dY_dW_n[:,l])
            + np.sum(
                np.multiply(  # Both factors need to be kmji
                    np.reshape(np.multiply(de_dg[:,:,n],dg_dF[:,:,n]*F[:,:,n]), (N,M,1)),
                    dX_dW_n[:,l:l+1,:]
                )
            ,axis=0)  # Sum over k
        ,axis=0)
    grad_e_W[W[l]>=0] = grad_e_W_p[W[l]>=0]
    grad_e_W[W[l]<0] = grad_e_W_n[W[l]<0]

    grad_e_K = np.zeros((M))
    grad_e_K_p = de_dr[0,n] * dR_dK_p[l] + np.sum(np.multiply(np.reshape(de_dg[0,:,n]*dg_dy[:,n], (M,1)), dY_dK_p[:,l])
            + np.sum(
                np.multiply(  # Both factors need to be kmji
                    np.reshape(np.multiply(de_dg[:,:,n],dg_dF[:,:,n]*F[:,:,n]), (N,M,1)),
                    dX_dK_p[:,l:l+1,:]
                )
            ,axis=0)  # Sum over k
        ,axis=0)


    grad_e_K_n = de_dr[0,n] * dR_dK_n[l] + np.sum(np.multiply(np.reshape(de_dg[0,:,n]*dg_dy[:,n], (M,1)), dY_dK_n[:,l])
            + np.sum(
                np.multiply(  # Both factors need to be kmji
                    np.reshape(np.multiply(de_dg[:,:,n],dg_dF[:,:,n]*F[:,:,n]), (N,M,1)),
                    dX_dK_n[:,l:l+1,:]
                )
            ,axis=0)  # Sum over k
        ,axis=0)

    grad_e_K[K_p[l]>=0] = grad_e_K_p[K_p[l]>=0]
    grad_e_K[K_p[l]<0] = grad_e_K_n[K_p[l]<0]

    grad_e_Djm = de_dr[0,n] * dR_dDjm[l] + np.sum(np.multiply(np.reshape(de_dg[0,:,n]*dg_dy[:,n], (M,1,1)), dY_dDjm[:,l])
            + np.sum(
                np.multiply(  # Both factors need to be kmji
                    np.reshape(np.multiply(de_dg[:,:,n],dg_dF[:,:,n]*F[:,:,n]), (N,M,1,1)),
                    dX_dDjm[:,l:l+1,:,:]
                )
            ,axis=0)  # Sum over k
        ,axis=0)


  if beta_hats[0,n] > 0:  # Check if we are optimizing n's access

    grad_a_F = da_dr[0,n] * dR_dF[l] + np.sum(np.multiply(np.reshape(da_dp[0,:,n]*dp_dy[:,n], (M,1,1)),dY_dF[:,l])
          + np.sum(
              np.multiply(
                np.reshape(np.multiply(da_dp[:,:,n],dp_dH[:,:,n]*H[:,:,n]),(N,M,1,1)),
                dX_dF[:,l:l+1,:,:]
              ),axis=0)
          ,axis=0)

    grad_a_H = da_dr[0,n] * dR_dH[l] + np.sum(np.multiply(np.reshape(da_dp[0,:,n]*dp_dy[:,n], (M,1,1)),dY_dH[:,l])
          + np.sum(
              np.multiply(
                np.reshape(np.multiply(da_dp[:,:,n],dp_dH[:,:,n]*H[:,:,n]),(N,M,1,1)),
                dX_dH[:,l:l+1,:,:]
              ),axis=0)
          ,axis=0)

    grad_a_H[:,n] += np.multiply(da_dp[0,:,n],dp_dH[n,:,n])

    grad_a_W = np.zeros(N)
    grad_a_W_p = da_dr[0,n]*dR_dW_p[l] + np.sum(np.multiply(np.reshape(da_dp[0,:,n]*dp_dy[:,n], (M,1)),dY_dW_p[:,l])
          + np.sum(
              np.multiply(
                np.reshape(np.multiply(da_dp[:,:,n],dp_dH[:,:,n]*H[:,:,n]),(N,M,1)),
                dX_dW_p[:,l:l+1,:]
              ),axis=0)
          ,axis=0)

    grad_a_W_n = da_dr[0,n] * dR_dW_n[l] + np.sum(np.multiply(np.reshape(da_dp[0,:,n]*dp_dy[:,n], (M,1)),dY_dW_n[:,l])
          + np.sum(
              np.multiply(
                np.reshape(np.multiply(da_dp[:,:,n],dp_dH[:,:,n]*H[:,:,n]),(N,M,1)),
                dX_dW_n[:,l:l+1,:]
              ),axis=0)
          ,axis=0)

    grad_a_W[W[l]>=0] = grad_a_W_p[W[l]>=0]
    grad_a_W[W[l]<0] = grad_a_W_n[W[l]<0]

    grad_a_K = np.zeros(M)
    grad_a_K_p = da_dr[0,n] * dR_dK_p[l] + np.sum(np.multiply(np.reshape(da_dp[0,:,n]*dp_dy[:,n], (M,1)),dY_dK_p[:,l])
          + np.sum(
              np.multiply(
                np.reshape(np.multiply(da_dp[:,:,n],dp_dH[:,:,n]*H[:,:,n]),(N,M,1)),
                dX_dK_p[:,l:l+1,:]
              ),axis=0)
          ,axis=0)

    grad_a_K_n =  da_dr[0,n] * dR_dK_n[l] + np.sum(np.multiply(np.reshape(da_dp[0,:,n]*dp_dy[:,n], (M,1)),dY_dK_n[:,l])
          + np.sum(
              np.multiply(
                np.reshape(np.multiply(da_dp[:,:,n],dp_dH[:,:,n]*H[:,:,n]),(N,M,1)),
                dX_dK_n[:,l:l+1,:]
              ),axis=0)
          ,axis=0)

    grad_a_K[K_p[l]>=0] = grad_a_K_p[K_p[l]>=0]
    grad_a_K[K_p[l]<0] = grad_a_K_n[K_p[l]<0]


    grad_a_Djm = da_dr[0,n] * dR_dDjm[l] + np.sum(np.multiply(np.reshape(da_dp[0,:,n]*dp_dy[:,n], (M,1,1)),dY_dDjm[:,l])
          + np.sum(
              np.multiply(
                np.reshape(np.multiply(da_dp[:,:,n],dp_dH[:,:,n]*H[:,:,n]),(N,M,1,1)),
                dX_dDjm[:,l:l+1,:,:]
              ),axis=0)
          ,axis=0)


#  print(np.concatenate((grad_e_F.flatten(),
#                           grad_e_H.flatten(),
#                           grad_e_W.flatten(),
#                           grad_e_K.flatten(),
#                           grad_e_Djm.flatten())))

  if betas[0,n] > 0 and beta_hats[0,n] > 0:  # Check if n is extractor and accessor
    # objective function gradient for RUs that extract and access the resource
    return np.concatenate(((grad_a_F + grad_e_F).flatten(),
                           (grad_a_H + grad_e_H).flatten(),
                           (grad_a_W + grad_e_W).flatten(),
                           (grad_a_K + grad_e_K).flatten(),
                           (grad_a_Djm + grad_e_Djm).flatten()))
  elif betas[0,n] > 0:
    # objective function gradient for extractors
    return np.concatenate((grad_e_F.flatten(),
                           grad_e_H.flatten(),
                           grad_e_W.flatten(),
                           grad_e_K.flatten(),
                           grad_e_Djm.flatten()))
  elif beta_hats[0,n] > 0:
    # objective function gradient for accessors
    return np.concatenate((grad_a_F.flatten(),
                           grad_a_H.flatten(),
                           grad_a_W.flatten(),
                           grad_a_K.flatten(),
                           grad_a_Djm.flatten()))



def multiply_by_inverse_jacobian(drdot_dp, dxdot_dp, dydot_dp, J_inv, T, N, M):
  # shape is the shape of strategy parameter p. For example, D_jm is (N,M,M).
  shape = drdot_dp.shape
  # dSdot_dp == how steady state changes wrt p, packed into one variable
  dSdot_dp = np.concatenate(
                 (np.broadcast_to(drdot_dp, (1, *shape)),
                 dxdot_dp,
                 dydot_dp),
             axis=0)

  dSdot_dp = dSdot_dp.reshape(T, np.prod(shape))  # this should already be true

  # do the actual computation
  dSS_dp = -J_inv @ dSdot_dp

  # unpack
  dSS_dp = dSS_dp.reshape((T, *shape))
  dR_dp = dSS_dp[0]
  dX_dp = dSS_dp[1:N+1]
  dY_dp = dSS_dp[N+1:N+1+M]

  return dR_dp, dX_dp, dY_dp
"""
  dSdot_dW_n = np.concatenate((np.broadcast_to(drdot_dW_n,(1,N,N)),dxdot_dW_n,dydot_dW_n), axis=0)
  dSdot_dW_n = dSdot_dW_n.reshape(T,(N)**2)
  dSS_dW_n = -J_inv @ dSdot_dW_n
  dSS_dW_n = dSS_dW_n.reshape(T,N,N)
  dR_dW_n = dSS_dW_n.reshape(T,N,N)[0]
  dX_dW_n = dSS_dW_n.reshape(T,N,N)[1:N+1]
  dY_dW_n = dSS_dW_n.reshape(T,N,N)[N+1:N+1+M]
"""