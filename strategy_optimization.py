import numpy as np
from scipy import optimize
from compute_J import determine_stability
from compute_J import correct_scale_params


def objective_grad(strategy, n, l, J, N,K,M,T,
    phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,
    F_p,F_n,H_p,H_n,W_p,W_n,K_p,K_n,D_jm):
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
    strategy parameters F,H,W_p,W_n,K_p,K_n,D_jm will be modified to match input strategy for l
  return the gradient of the objective function at that point for that actor
  '''
  # Unpack strategy parameters.
  F_p[l] = strategy[0:M*N].reshape([M,N])
  F_n[l] = strategy[M*N:2*M*N].reshape([M,N])
  H_p[l] = strategy[2*M*N:3*M*N].reshape([M,N])
  H_n[l] = strategy[3*M*N:4*M*N].reshape([M,N])
  W_p[l] = strategy[4*M*N:4*M*N+N].reshape([N])
  W_n[l] = strategy[4*M*N+N:4*M*N+2*N].reshape([N])
  K_p[l] = strategy[4*M*N+2*N:4*M*N+2*N+M].reshape([M])
  K_n[l] = strategy[4*M*N+2*N+M:4*M*N+2*N+2*M].reshape([M])
  D_jm[l] = strategy[4*M*N+2*N+2*M:4*M*N+2*N+2*M+M**2].reshape([M,M])

  # Compute Jacobian
  J, eigvals, stability = determine_stability(N,K,M,T,
      phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,
      F_p,F_n,H_p,H_n,W_p,W_n,K_p,K_n,D_jm)

  # Compute inverse Jacobian
  J_inv = np.linalg.inv(J)

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

  drdot_dW_n = np.zeros([N,N])  # Compute how the rhs of system changes with respect to each strategy parameter
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
#  print(np.shape(np.transpose(-mus*np.squeeze(theta_bars))))
#  print(np.shape(np.transpose(np.multiply(epsilons,dt_dD_jm),(2,0,1))))
#  print(np.shape(dydot_dDjm[np.diag_indices(M,1),:,np.diag_indices(M,1),:]))
  dydot_dDjm[np.diag_indices(M,1),:,np.diag_indices(M,1),:] = np.multiply(np.reshape(np.transpose(-mus*np.squeeze(theta_bars)),(M,1,1)),np.transpose(np.multiply(epsilons,dt_dD_jm),(2,0,1))) # eq __, mxnxl

  ## Compute how the steady state of the system changes with respect to each strategy parameter
  # dSdot_dF == how steady state changes wrt F, packed into one variable
  dSdot_dF = np.concatenate((np.broadcast_to(drdot_dF, (1,N,M,N)), dxdot_dF, dydot_dF), axis=0)
  dSdot_dF = dSdot_dF.reshape(T, (N)**2*M)
  # do the actual computation
  dSS_dF = -J_inv @ dSdot_dF
  # unpack
  dR_dF = dSS_dF.reshape(T,N,M,N)[0]
  dX_dF = dSS_dF.reshape(T,N,M,N)[1:N+1]
  dY_dF = dSS_dF.reshape(T,N,M,N)[N+1:N+1+M]

  dSdot_dH = np.concatenate((np.broadcast_to(drdot_dH,(1,N,M,N)),dxdot_dH,dydot_dH), axis=0)
  dSdot_dH = dSdot_dH.reshape(T,(N)**2*M)
  dSS_dH = -J_inv @ dSdot_dH
  dR_dH = dSS_dH.reshape(T,N,M,N)[0]
  dX_dH = dSS_dH.reshape(T,N,M,N)[1:N+1]
  dY_dH = dSS_dH.reshape(T,N,M,N)[N+1:N+1+M]

  dSdot_dW_p = np.concatenate((np.broadcast_to(drdot_dW_p,(1,N,N)),dxdot_dW_p,dydot_dW_p), axis=0)
  dSdot_dW_p = dSdot_dW_p.reshape(T,(N)**2)
  dSS_dW_p = -J_inv @ dSdot_dW_p
  dSS_dW_p = dSS_dW_p.reshape(T,N,N)
  dR_dW_p = dSS_dW_p.reshape(T,N,N)[0]
  dX_dW_p = dSS_dW_p.reshape(T,N,N)[1:N+1]
  dY_dW_p = dSS_dW_p.reshape(T,N,N)[N+1:N+1+M]

  dSdot_dW_n = np.concatenate((np.broadcast_to(drdot_dW_n,(1,N,N)),dxdot_dW_n,dydot_dW_n), axis=0)
  dSdot_dW_n = dSdot_dW_n.reshape(T,(N)**2)
  dSS_dW_n = -J_inv @ dSdot_dW_n
  dSS_dW_n = dSS_dW_n.reshape(T,N,N)
  dR_dW_n = dSS_dW_n.reshape(T,N,N)[0]
  dX_dW_n = dSS_dW_n.reshape(T,N,N)[1:N+1]
  dY_dW_n = dSS_dW_n.reshape(T,N,N)[N+1:N+1+M]

  dSdot_dK_p = np.concatenate((np.broadcast_to(drdot_dK_p,(1,N,M)),dxdot_dK_p,dydot_dK_p), axis=0)
  dSdot_dK_p = dSdot_dK_p.reshape(T,(N)*M)
  dSS_dK_p = -J_inv @ dSdot_dK_p
  dSS_dK_p = dSS_dK_p.reshape(T,N,M)
  dR_dK_p = dSS_dK_p.reshape(T,N,M)[0]
  dX_dK_p = dSS_dK_p.reshape(T,N,M)[1:N+1]
  dY_dK_p = dSS_dK_p.reshape(T,N,M)[N+1:N+1+M]

  dSdot_dK_n = np.concatenate((np.broadcast_to(drdot_dK_n,(1,N,M)),dxdot_dK_n,dydot_dK_n), axis=0)
  dSdot_dK_n = dSdot_dK_n.reshape(T,(N)*M)
  dSS_dK_n = -J_inv @ dSdot_dK_n
  dSS_dK_n = dSS_dK_n.reshape(T,N,M)
  dR_dK_n = dSS_dK_n.reshape(T,N,M)[0]
  dX_dK_n = dSS_dK_n.reshape(T,N,M)[1:N+1]
  dY_dK_n = dSS_dK_n.reshape(T,N,M)[N+1:N+1+M]

  dSdot_dDjm = np.concatenate((np.broadcast_to(drdot_dDjm,(1,N,M,M)),dxdot_dDjm,dydot_dDjm), axis=0)
  dSdot_dDjm = dSdot_dDjm.reshape(T,(N)*M**2)
  dSS_dDjm = -np.linalg.inv(J) @ dSdot_dDjm
  dR_dDjm = dSS_dDjm.reshape(T,N,M,M)[0]
  dX_dDjm = dSS_dDjm.reshape(T,N,M,M)[1:N+1]
  dY_dDjm = dSS_dDjm.reshape(T,N,M,M)[N+1:N+1+M]
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
  # dSdot_dF == how steady state changes wrt F, packed into one variable
  dSdot_dF = np.concatenate((np.broadcast_to(drdot_dF, (1,N,M,N)), dxdot_dF, dydot_dF), axis=0)
  dSdot_dF = dSdot_dF.reshape(T, (N)**2*M)
  # do the actual computation
  dSS_dF = -J_inv @ dSdot_dF
  # unpack
  dR_dF = dSS_dF.reshape(T,N,M,N)[0]
  dX_dF = dSS_dF.reshape(T,N,M,N)[1:N+1]
  dY_dF = dSS_dF.reshape(T,N,M,N)[N+1:N+1+M]

  dSdot_dH = np.concatenate((np.broadcast_to(drdot_dH,(1,N,M,N)),dxdot_dH,dydot_dH), axis=0)
  dSdot_dH = dSdot_dH.reshape(T,(N)**2*M)
  dSS_dH = -J_inv @ dSdot_dH
  dR_dH = dSS_dH.reshape(T,N,M,N)[0]
  dX_dH = dSS_dH.reshape(T,N,M,N)[1:N+1]
  dY_dH = dSS_dH.reshape(T,N,M,N)[N+1:N+1+M]

  dSdot_dW_p = np.concatenate((np.broadcast_to(drdot_dW_p,(1,N,N)),dxdot_dW_p,dydot_dW_p), axis=0)
  dSdot_dW_p = dSdot_dW_p.reshape(T,(N)**2)
  dSS_dW_p = -J_inv @ dSdot_dW_p
  dSS_dW_p = dSS_dW_p.reshape(T,N,N)
  dR_dW_p = dSS_dW_p.reshape(T,N,N)[0]
  dX_dW_p = dSS_dW_p.reshape(T,N,N)[1:N+1]
  dY_dW_p = dSS_dW_p.reshape(T,N,N)[N+1:N+1+M]

  dSdot_dW_n = np.concatenate((np.broadcast_to(drdot_dW_n,(1,N,N)),dxdot_dW_n,dydot_dW_n), axis=0)
  dSdot_dW_n = dSdot_dW_n.reshape(T,(N)**2)
  dSS_dW_n = -J_inv @ dSdot_dW_n
  dSS_dW_n = dSS_dW_n.reshape(T,N,N)
  dR_dW_n = dSS_dW_n.reshape(T,N,N)[0]
  dX_dW_n = dSS_dW_n.reshape(T,N,N)[1:N+1]
  dY_dW_n = dSS_dW_n.reshape(T,N,N)[N+1:N+1+M]

  dSdot_dK_p = np.concatenate((np.broadcast_to(drdot_dK_p,(1,N,M)),dxdot_dK_p,dydot_dK_p), axis=0)
  dSdot_dK_p = dSdot_dK_p.reshape(T,(N)*M)
  dSS_dK_p = -J_inv @ dSdot_dK_p
  dSS_dK_p = dSS_dK_p.reshape(T,N,M)
  dR_dK_p = dSS_dK_p.reshape(T,N,M)[0]
  dX_dK_p = dSS_dK_p.reshape(T,N,M)[1:N+1]
  dY_dK_p = dSS_dK_p.reshape(T,N,M)[N+1:N+1+M]

  dSdot_dK_n = np.concatenate((np.broadcast_to(drdot_dK_n,(1,N,M)),dxdot_dK_n,dydot_dK_n), axis=0)
  dSdot_dK_n = dSdot_dK_n.reshape(T,(N)*M)
  dSS_dK_n = -J_inv @ dSdot_dK_n
  dSS_dK_n = dSS_dK_n.reshape(T,N,M)
  dR_dK_n = dSS_dK_n.reshape(T,N,M)[0]
  dX_dK_n = dSS_dK_n.reshape(T,N,M)[1:N+1]
  dY_dK_n = dSS_dK_n.reshape(T,N,M)[N+1:N+1+M]

  dSdot_dDjm = np.concatenate((np.broadcast_to(drdot_dDjm,(1,N,M,M)),dxdot_dDjm,dydot_dDjm), axis=0)
  dSdot_dDjm = dSdot_dDjm.reshape(T,(N)*M**2)
  dSS_dDjm = -J_inv @ dSdot_dDjm
  dR_dDjm = dSS_dDjm.reshape(T,N,M,M)[0]
  dX_dDjm = dSS_dDjm.reshape(T,N,M,M)[1:N+1]
  dY_dDjm = dSS_dDjm.reshape(T,N,M,M)[N+1:N+1+M]

  # calculate gradients of objective function for one actor
  # for extraction
  # n's objective, l's strategy (same for resource users) n,l used to be i,j
  if betas[0,n] > 0.000001:  # Check if we are optimizing n's extraction
                             #### TODO: replace 0.000001 with named constant, put elsewhere too
    # jxi
    grad_e_F_p = de_dr[0,n] * dR_dF[l] + np.sum(np.multiply(np.reshape(de_dg[0,:,n]*dg_dy[:,n], (M,1,1)), dY_dF[:,l])
                         # scalar                               jxi         # m                             mji
            + np.sum(
                np.multiply(  # Both factors need to be kmji
                    np.reshape(np.multiply(de_dg[:,:,n],dg_dF[:,:,n]*(F_p[:,:,n]-F_n[:,:,n])), (N,M,1,1)),
                                               # 1m            km          km
                    dX_dF[:,l:l+1,:,:]
                       # k1ji
                )
            ,axis=0)  # Sum over k
        ,axis=0)  # Sum over m

    grad_e_F_p[:,n] += np.multiply(de_dg[0,:,n], dg_dF[n,:,n])

    grad_e_F_n = -grad_e_F_p

    grad_e_H_p = de_dr[0,n] * dR_dH[l] + np.sum(np.multiply(np.reshape(de_dg[0,:,n]*dg_dy[:,n], (M,1,1)), dY_dH[:,l])
                         # scalar                               jxi         # m                             mji
            + np.sum(
                np.multiply(  # Both factors need to be kmji
                    np.reshape(np.multiply(de_dg[:,:,n],dg_dF[:,:,n]*(F_p[:,:,n]-F_n[:,:,n])), (N,M,1,1)),
                                               # 1m            km          km
                    dX_dH[:,l:l+1,:,:]
                     # k1ji
                )
            ,axis=0)  # Sum over k
        ,axis=0)

    grad_e_H_n = - grad_e_H_p
    print(np.shape(dX_dW_p[:,l:l+1,:]))
    grad_e_W_p = de_dr[0,n] * dR_dW_p[l] + np.sum(np.multiply(np.reshape(de_dg[0,:,n]*dg_dy[:,n], (M,1,1)), dY_dW_p[:,l])
            + np.sum(
                np.multiply(  # Both factors need to be kmi
                    np.reshape(
                        np.multiply(
                            de_dg[:,:,n],
                            dg_dF[:,:,n] * (F_p[:,:,n]-F_n[:,:,n])
                        ), (N,M,1)
                    ),
                    dX_dW_p[:,l:l+1,:]  # k1i
                )
            ,axis=0)  # Sum over k
        ,axis=0)

    grad_e_W_n =  de_dr[0,n] * dR_dW_n[l] + np.sum(np.multiply(np.reshape(de_dg[0,:,n]*dg_dy[:,n], (M,1,1)), dY_dW_n[:,l])
            + np.sum(
                np.multiply(  # Both factors need to be kmji
                    np.reshape(np.multiply(de_dg[:,:,n],dg_dF[:,:,n]*(F_p[:,:,n]-F_n[:,:,n])), (N,M,1,1)),
                    dX_dW_n[:,l:l+1,:]
                )
            ,axis=0)  # Sum over k
        ,axis=0)

    grad_e_K_p =  de_dr[0,n] * dR_dK_p[l] + np.sum(np.multiply(np.reshape(de_dg[0,:,n]*dg_dy[:,n], (M,1,1)), dY_dK_p[:,l])
            + np.sum(
                np.multiply(  # Both factors need to be kmji
                    np.reshape(np.multiply(de_dg[:,:,n],dg_dF[:,:,n]*(F_p[:,:,n]-F_n[:,:,n])), (N,M,1,1)),
                    dX_dK_p[:,l:l+1,:]
                )
            ,axis=0)  # Sum over k
        ,axis=0)

    grad_e_K_n = de_dr[0,n] * dR_dK_n[l] + np.sum(np.multiply(np.reshape(de_dg[0,:,n]*dg_dy[:,n], (M,1,1)), dY_dK_n[:,l])
            + np.sum(
                np.multiply(  # Both factors need to be kmji
                    np.reshape(np.multiply(de_dg[:,:,n],dg_dF[:,:,n]*(F_p[:,:,n]-F_n[:,:,n])), (N,M,1,1)),
                    dX_dK_n[:,l:l+1,:]
                )
            ,axis=0)  # Sum over k
        ,axis=0)

    grad_e_Djm = de_dr[0,n] * dR_dDjm[l] + np.sum(np.multiply(np.reshape(de_dg[0,:,n]*dg_dy[:,n], (M,1,1)), dY_dDjm[:,l])
            + np.sum(
                np.multiply(  # Both factors need to be kmji
                    np.reshape(np.multiply(de_dg[:,:,n],dg_dF[:,:,n]*(F_p[:,:,n]-F_n[:,:,n])), (N,M,1,1)),
                    dX_dDjm[:,l:l+1,:,:]
                )
            ,axis=0)  # Sum over k
        ,axis=0)


  if beta_hats[0,n] > 0:  # Check if we are optimizing n's access

    grad_a_F_p = da_dr[0,n] * dR_dF[l] + np.sum(np.multiply(np.reshape(da_dp[0,:,n]*dp_dy[:,n], (M,1,1)),dY_dF[:,l])
          + np.sum(
              np.multiply(
                np.reshape(np.multiply(da_dp[:,:,n],dp_dH[:,:,n]*(H_p[:,:,n]-H_n[:,:,n])),(N,M,1,1)),
                dX_dF[:,l:l+1,:,:]
              ),axis=0)
          ,axis=1)

    grad_a_F_n = -grad_a_F_p

    grad_a_H_p = da_dr[0,n] * dR_dH[l] + np.sum(np.multiply(np.reshape(da_dp[0,:,n]*dp_dy[:,n], (M,1,1)),dY_dH[:,l])
          + np.sum(
              np.multiply(
                np.reshape(np.multiply(da_dp[:,:,n],dp_dH[:,:,n]*(H_p[:,:,n]-H_n[:,:,n])),(N,M,1,1)),
                dX_dH[:,l:l+1,:,:]
              ),axis=0)
          ,axis=1)

    grad_a_H_n = -grad_a_H_p

    grad_a_H_p[:,n] += np.multiply(da_dp[0,:,n],dp_dH[n,:,n])

    grad_a_W_p =  da_dr[0,n]*dR_dW_p[l] + np.sum(np.multiply(np.reshape(da_dp[0,:,n]*dp_dy[:,n], (M,1,1)),dY_dW_p[:,l])
          + np.sum(
              np.multiply(
                np.reshape(np.multiply(da_dp[:,:,n],dp_dH[:,:,n]*(H_p[:,:,n]-H_n[:,:,n])),(N,M,1,1)),
                dX_dW_p[:,l:l+1,:]
              ),axis=0)
          ,axis=0)

    grad_a_W_n = da_dr[0,n] * dR_dW_n[l] + np.sum(np.multiply(np.reshape(da_dp[0,:,n]*dp_dy[:,n], (M,1,1)),dY_dW_n[:,l])
          + np.sum(
              np.multiply(
                np.reshape(np.multiply(da_dp[:,:,n],dp_dH[:,:,n]*(H_p[:,:,n]-H_n[:,:,n])),(N,M,1,1)),
                dX_dW_n[:,l:l+1,:]
              ),axis=0)
          ,axis=0)

    grad_a_K_p = da_dr[0,n] * dR_dK_p[l] + np.sum(np.multiply(np.reshape(da_dp[0,:,n]*dp_dy[:,n], (M,1,1)),dY_dK_p[:,l])
          + np.sum(
              np.multiply(
                np.reshape(np.multiply(da_dp[:,:,n],dp_dH[:,:,n]*(H_p[:,:,n]-H_n[:,:,n])),(N,M,1,1)),
                dX_dK_p[:,l:l+1,:]
              ),axis=0)
          ,axis=0)

    grad_a_K_n =  da_dr[0,n] * dR_dK_n[l] + np.sum(np.multiply(np.reshape(da_dp[0,:,n]*dp_dy[:,n], (M,1,1)),dY_dK_n[:,l])
          + np.sum(
              np.multiply(
                np.reshape(np.multiply(da_dp[:,:,n],dp_dH[:,:,n]*(H_p[:,:,n]-H_n[:,:,n])),(N,M,1,1)),
                dX_dK_n[:,l:l+1,:]
              ),axis=0)
          ,axis=0)

    grad_a_Djm = da_dr[0,n] * dR_dDjm[l] + np.sum(np.multiply(np.reshape(da_dp[0,:,n]*dp_dy[:,n], (M,1,1)),dY_dDjm[:,l])
          + np.sum(
              np.multiply(
                np.reshape(np.multiply(da_dp[:,:,n],dp_dH[:,:,n]*(H_p[:,:,n]-H_n[:,:,n])),(N,M,1,1)),
                dX_dDjm[:,l:l+1,:,:]
              ),axis=0)
          ,axis=1)


  if betas[0,n] > 0 and beta_hats[0,n] > 0:  # Check if n is extractor and accessor
    # objective function gradient for RUs that extract and access the resource

    return np.concatenate(((grad_a_F_p + grad_e_F_p).flatten(),
                            (grad_a_F_n + grad_e_F_n).flatten(),
                            grad_a_H_p.flatten() + grad_e_H_p.flatten(),
                            grad_a_H_n.flatten() + grad_e_H_n.flatten(),
                            (grad_a_W_p + grad_e_W_p).flatten(), (grad_a_W_n + grad_e_W_n).flatten(),
                            (grad_a_K_p + grad_e_K_p).flatten(), (grad_a_K_n + grad_e_K_n).flatten(),
                            grad_a_Djm.flatten() + grad_e_Djm.flatten()))
  elif betas[0,n] > 0:
    # objective function gradient for extractors
    return np.concatenate((grad_e_F_p.flatten(), grad_e_F_n.flatten(),
                           grad_e_H_p.flatten(), grad_e_H_n.flatten(),
                           grad_e_W_p[0], grad_e_W_n[0],
                           grad_e_K_p[0], grad_e_K_n[0],
                           grad_e_Djm.flatten()))
  elif beta_hats[0,n] > 0:
    # objective function gradient for accessors
    return np.concatenate((grad_a_F_p.flatten(), grad_a_F_n.flatten(),
                           grad_a_H_p.flatten(), grad_a_H_n.flatten(),
                           grad_a_W_p[0], grad_a_W_n[0],
                           grad_a_K_p[0], grad_a_K_n[0],
                           grad_a_Djm.flatten()))


# If strategy does not have all efforts >= 0, project onto space of legal strategies
def boundary_projection(mu, strategy):
  return np.sum(np.maximum(strategy - mu, 0)) - 1


def grad_descent_constrained(initial_point, max_steps, n, l, J, N,K,M,T,
    phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,
    F_p,F_n,H_p,H_n,W_p,W_n,K_p,K_n,D_jm):
  '''
  inputs:
    initial_point is the initial strategy
    max_steps (usually low, don't go all the way to optimal)
    n is the actor whose objective we want to optimize
    l is the actor whose strategy it is
    J is the Jacobian
    N,K,M,T meta parameters
    scale parameters
    exponent parameters
    strategy parameters????
  return the new and improved strategy
  '''
  grad = objective_grad(initial_point, n, l, J, N,K,M,T,
                        phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,
                        F_p,F_n,H_p,H_n,W_p,W_n,K_p,K_n,D_jm)
  # Project gradient onto the plane sum(efforts) == 1
  grad = grad - np.sum(grad)/len(grad)
  grad_mag = np.linalg.norm(grad)  # to check for convergence

  x = initial_point  # strategy
  alpha = 0.05
  num_steps = 0
  while grad_mag > 1e-5 and num_steps < max_steps:
    # Follow the projected gradient for a fixed step size alpha
    x = x + alpha*grad

    # If strategy does not have all efforts >= 0, project onto space of legal strategies
    if np.any(x < 0):
      try:
        mu = optimize.brentq(boundary_projection, 0, 500, args=(x))
      except:
        print('bisection bounds did not work')
        print(x)
      x = np.maximum(x - mu, 0)

    # Compute new gradient and update strategy parameters to match x
    grad = objective_grad(x, n, l, J, N,K,M,T,
                          phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,
                          F_p,F_n,H_p,H_n,W_p,W_n,K_p,K_n,D_jm)

    # Project gradient onto the plane sum(efforts) == 1
    grad = grad - np.sum(grad)/len(grad)
    grad_mag = np.linalg.norm(grad)  # to check for convergence

    num_steps += 1
  return x


def nash_equilibrium(max_iters, J, N,K,M,T,
    phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,
    F_p,F_n,H_p,H_n,W_p,W_n,K_p,K_n,D_jm):
  '''
  inputs:
    max_iters
    J is the Jacobian
    N,K,M,T meta parameters
    scale parameters
    exponent parameters
    strategy parameters (initial value given by sample function)
  returns
    optimized strategy parameters
    updated sigmas and lamdas
  '''
  # Initialize strategy
  strategy = np.zeros((N, 4*M*N + 2*N + 2*M+M**2))
  for i in range(N):
    strategy[i] = np.concatenate((F_p[i].flatten(),F_n[i].flatten(),H_p[i].flatten(),H_n[i].flatten(),
                                  W_p[i].flatten(),W_n[i].flatten(),K_p[i].flatten(),K_n[i].flatten(),D_jm[i].flatten()))
    strategy[i] /= np.sum(strategy[i])
  # sample to get bridging org objectives
  objectives = np.random.randint(0,N-K,size = K)
  tolerance = 0.01
  strategy_difference = [1]  # arbitrary initial value, List of differences in euclidean distance between strategies in consecutive iterations
  iterations = 0
  strategy_prev = []  # a list of the strategies at each iteration
  strategy_prev.append(strategy.copy())

  while strategy_difference[-1] > tolerance and iterations < max_iters:
    # Loop through each actor i
    for i in range(N):
      if i <= N-K-1:
        objective = i
      else:
        objective = objectives[i-(N-K)]


      new_strategy = grad_descent_constrained(strategy[i], 3, objective, i, J, N,K,M,T,
          phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,
          F_p,F_n,H_p,H_n,W_p,W_n,K_p,K_n,D_jm)
      # Check if there are new zeros in the strategy parameters to see if we need to update scale parameters
      # (e.g. for portion of gain through collaboration) to make sure they are consistent with our new
      # strategy parameters.
      if np.count_nonzero(new_strategy[4*M*N:4*M*N + N]) < np.count_nonzero(strategy[i][2*M*N:2*M*N+N]):
        sigmas = correct_scale_params(sigmas,W_p[i],i)
      if np.count_nonzero(new_strategy[4*M*N+N:4*M*N + 2*N]) < np.count_nonzero(strategy[i][2*M*N+N:2*M*N+2*N]):
        lambdas = correct_scale_params(lambdas,W_n[i],i)

      # update strategy for this actor
      strategy[i] = new_strategy

    # update strategies for all actors
    strategy_prev.append(strategy.copy())
    # compute difference in strategies
    strategy_difference.append(np.linalg.norm((strategy_prev[-2] - strategy_prev[-1])))
    iterations += 1
    if iterations == max_iters - 1:
      print('max number of iterations')

  return F_p,F_n,H_p,H_n,W_p,W_n,K_p,K_n,D_jm, sigmas,lambdas


