import numpy as np
import scipy as sp  ## Sunshine: Is this abbreviation common (like np is)? You don't use sp much in the code.
from compute_J import determine_stability
from compute_J import correct_scale_params


def objective_grad(strategy,n,l,J,N,K,M,T,phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,
theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,
dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,F,H,w_p,w_n,K_p,K_n,D_jm):
  '''
  input strategy for a single actor (flattened) plus all of the other parameters in the model, get the
  gradient of the objective function at that point for that actor
  '''
  # Unpack strategy parameters.
  F[n] = strategy[0:M*N].reshape([M,N])
  H[n] = strategy[M*N:2*M*N].reshape([M,N])
  w_p[n] = strategy[2*M*N:2*M*N+N].reshape([N])
  w_n[n] = strategy[2*M*N+N:2*M*N+2*N].reshape([N])
  K_p[n] = strategy[2*M*N+2*N:2*M*N+2*N+M].reshape([M])
  K_n[n] = strategy[2*M*N+2*N+M:2*M*N+2*N+2*M].reshape([M])
  D_jm[n] = strategy[2*M*N+2*N+2*M:2*M*N+2*N+2*M+M**2].reshape([M,M])

  # Compute Jacobian
  J,eigvals,stability = determine_stability(N,K,M,T, phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,
      theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,
      dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,F,H,w_p,w_n,K_p,K_n,D_jm)


  J_inv = np.linalg.inv(J)
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
  dxdot_dW_n[np.arange(0,N),:,np.arange(0,N)] = np.transpose(np.multiply(alphas*etas,np.multiply(lambdas,dc_dw_n)))
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
  dydot_dDjm[np.diag_indices(M,1),:,np.diag_indices(M,1),:] = np.multiply(np.transpose(-mus*np.squeeze(theta_bars)),np.transpose(np.multiply(epsilons,dt_dD_jm),(2,0,1))) # eq __, mxnxl


  dSdot_dF = np.concatenate((np.broadcast_to(drdot_dF,(1,N,M,N)),dxdot_dF,dydot_dF), axis=0)
  dSdot_dF = dSdot_dF.reshape(T,(N)**2*M)
  dSS_dF = -J_inv @ dSdot_dF
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

  # calculate gradients of objective function for one actor
  # for extraction
  # n's objective, l's strategy (same for resource users) n,l used to be i,j
  if betas[0,n] > 0:
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

    grad_e_F[:,n] += np.multiply(de_dg[0,:,n],dg_dF[n,:,n])


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
    print(grad_e_H)
    grad_e_W_p = de_dr[0,n] * dR_dW_p[l] + np.sum(np.multiply(np.reshape(de_dg[0,:,n]*dg_dy[:,n], (M,1,1)), dY_dW_p[:,l])
            + np.sum(
                np.multiply(  # Both factors need to be kmji
                    np.reshape(np.multiply(de_dg[:,:,n],dg_dF[:,:,n]*F[:,:,n]), (N,M,1,1)),
                    dX_dW_p[:,l:l+1,:]
                )
            ,axis=0)  # Sum over k
        ,axis=0)

    grad_e_W_n =  de_dr[0,n] * dR_dW_n[l] + np.sum(np.multiply(np.reshape(de_dg[0,:,n]*dg_dy[:,n], (M,1,1)), dY_dW_n[:,l])
            + np.sum(
                np.multiply(  # Both factors need to be kmji
                    np.reshape(np.multiply(de_dg[:,:,n],dg_dF[:,:,n]*F[:,:,n]), (N,M,1,1)),
                    dX_dW_n[:,l:l+1,:]
                )
            ,axis=0)  # Sum over k
        ,axis=0)

    grad_e_K_p =  de_dr[0,n] * dR_dK_p[l] + np.sum(np.multiply(np.reshape(de_dg[0,:,n]*dg_dy[:,n], (M,1,1)), dY_dK_p[:,l])
            + np.sum(
                np.multiply(  # Both factors need to be kmji
                    np.reshape(np.multiply(de_dg[:,:,n],dg_dF[:,:,n]*F[:,:,n]), (N,M,1,1)),
                    dX_dK_p[:,l:l+1,:]
                )
            ,axis=0)  # Sum over k
        ,axis=0)

    grad_e_K_n = de_dr[0,n] * dR_dK_n[l] + np.sum(np.multiply(np.reshape(de_dg[0,:,n]*dg_dy[:,n], (M,1,1)), dY_dK_n[:,l])
            + np.sum(
                np.multiply(  # Both factors need to be kmji
                    np.reshape(np.multiply(de_dg[:,:,n],dg_dF[:,:,n]*F[:,:,n]), (N,M,1,1)),
                    dX_dK_n[:,l:l+1,:]
                )
            ,axis=0)  # Sum over k
        ,axis=0)

    grad_e_Djm = de_dr[0,n] * dR_dDjm[l] + np.sum(np.multiply(np.reshape(de_dg[0,:,n]*dg_dy[:,n], (M,1,1)), dY_dDjm[:,l])
            + np.sum(
                np.multiply(  # Both factors need to be kmji
                    np.reshape(np.multiply(de_dg[:,:,n],dg_dF[:,:,n]*F[:,:,n]), (N,M,1,1)),
                    dX_dDjm[:,l:l+1,:,:]
                )
            ,axis=0)  # Sum over k
        ,axis=0)

    # for access
  if beta_hats[0,n] > 0:

    grad_a_F = da_dr[0,n] * dR_dF[l] + np.sum(np.multiply(np.reshape(da_dp[0,:,n]*dp_dy[:,n], (M,1,1)),dY_dF[:,l])\
          + np.sum(
              np.multiply(
                np.reshape(np.multiply(da_dp[:,:,n],dp_dH[:,:,n]*H[:,:,n]),(N,M,1,1)),
                dX_dF[:,l:l+1,:,:]
              ),axis=0)
          ,axis=1)

    grad_a_H = da_dr[0,n] * dR_dH[l] + np.sum(np.multiply(np.reshape(da_dp[0,:,n]*dp_dy[:,n], (M,1,1)),dY_dH[:,l])\
          + np.sum(
              np.multiply(
                np.reshape(np.multiply(da_dp[:,:,n],dp_dH[:,:,n]*H[:,:,n]),(N,M,1,1)),
                dX_dH[:,l:l+1,:,:]
              ),axis=0)
          ,axis=1)

    grad_a_H[:,n] += np.multiply(da_dp[0,:,n],dp_dH[n,:,n])

    grad_a_W_p =  da_dr[0,n]*dR_dW_p[l] + np.sum(np.multiply(np.reshape(da_dp[0,:,n]*dp_dy[:,n], (M,1,1)),dY_dW_p[:,l])\
          + np.sum(
              np.multiply(
                np.reshape(np.multiply(da_dp[:,:,n],dp_dH[:,:,n]*H[:,:,n]),(N,M,1,1)),
                dX_dW_p[:,l:l+1,:]
              ),axis=0)
          ,axis=1)

    grad_a_W_n = da_dr[0,n] * dR_dW_n[l] + np.sum(np.multiply(np.reshape(da_dp[0,:,n]*dp_dy[:,n], (M,1,1)),dY_dW_n[:,l])\
          + np.sum(
              np.multiply(
                np.reshape(np.multiply(da_dp[:,:,n],dp_dH[:,:,n]*H[:,:,n]),(N,M,1,1)),
                dX_dW_n[:,l:l+1,:]
              ),axis=0)
          ,axis=1)

    grad_a_K_p = da_dr[0,n] * dR_dK_p[l] + np.sum(np.multiply(np.reshape(da_dp[0,:,n]*dp_dy[:,n], (M,1,1)),dY_dK_p[:,l])\
          + np.sum(
              np.multiply(
                np.reshape(np.multiply(da_dp[:,:,n],dp_dH[:,:,n]*H[:,:,n]),(N,M,1,1)),
                dX_dK_p[:,l:l+1,:]
              ),axis=0)
          ,axis=1)

    grad_a_K_n =  da_dr[0,n] * dR_dK_n[l] + np.sum(np.multiply(np.reshape(da_dp[0,:,n]*dp_dy[:,n], (M,1,1)),dY_dK_n[:,l])\
          + np.sum(
              np.multiply(
                np.reshape(np.multiply(da_dp[:,:,n],dp_dH[:,:,n]*H[:,:,n]),(N,M,1,1)),
                dX_dK_n[:,l:l+1,:]
              ),axis=0)
          ,axis=1)

    grad_a_Djm = da_dr[0,n] * dR_dDjm[l] + np.sum(np.multiply(np.reshape(da_dp[0,:,n]*dp_dy[:,n], (M,1,1)),dY_dDjm[:,l])\
          + np.sum(
              np.multiply(
                np.reshape(np.multiply(da_dp[:,:,n],dp_dH[:,:,n]*H[:,:,n]),(N,M,1,1)),
                dX_dDjm[:,l:l+1,:,:]
              ),axis=0)
          ,axis=1)


  if betas[0,n] > 0 and beta_hats[0,n] > 0:
    # objective function gradient for RUs that extract and access the resource
    return np.concatenate(((grad_a_F+grad_e_F).flatten(),grad_a_H.flatten()+grad_e_H.flatten(),
                           grad_a_W_p + grad_e_W_p [0], grad_a_W_n + grad_e_W_n[0],
                           grad_a_K_p + grad_e_K_p[0],grad_a_K_n + grad_e_K_n[0],
                           grad_a_Djm.flatten() + grad_e_Djm.flatten()))
  elif betas[0,n]>0:
    # objective function gradient for extractors
    return np.concatenate((grad_e_F.flatten(),grad_e_H.flatten(),
                           grad_e_W_p[0],grad_e_W_n[0],
                           grad_e_K_p[0],grad_e_K_n[0],
                           grad_e_Djm.flatten()))
  else:
    # objective function gradient for accessors
    return np.concatenate((grad_a_F.flatten(),grad_a_H.flatten(),
                           grad_a_W_p[0],grad_a_W_n[0],
                           grad_a_K_p[0],grad_a_K_n[0],
                           grad_a_Djm.flatten()))

def boundary_projection(mu, strategy):
  return np.sum(np.maximum(strategy - mu, 0)) - 1

def grad_descent_constrained(initial_point,max_steps,n,l,J,N,K,M,T,phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,
theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,
dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,F,H,w_p,w_n,K_p,K_n,D_jm):

  grad = objective_grad(initial_point,n,l,J,N,K,M,T,phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,
                        theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,
                        dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,F,H,w_p,w_n,K_p,K_n,D_jm)

  grad = grad - np.sum(grad)/len(grad)
  grad_mag = np.linalg.norm(grad)
  x = initial_point
  alpha = 0.01
  num_steps = 0
  while grad_mag > 1e-7 and num_steps < max_steps:
    x = x + alpha*grad
    if np.any(x<0):
      try:
        mu = sp.optimize.brentq(boundary_projection, 0, 100, args=(x)) #-np.sum(x[x<0]) didn't work
      except:
        print('bisection bounds did not work')
        print(x)
      x = np.maximum(x - mu, 0)

    grad = objective_grad(x,n,l,J,N,K,M,T,phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,
                          theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,
                          dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,F,H,w_p,w_n,K_p,K_n,D_jm)
    grad = grad - np.sum(grad)/len(grad)
    grad_mag = np.linalg.norm(grad)
    num_steps += 1
  return x

def nash_equilibrium(max_iters,J,N,K,M,T,phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,
theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,
dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,F,H,w_p,w_n,K_p,K_n,D_jm):

  strategy = np.zeros((N,2*M*N+2*N+2*M+M**2))
  for i in range(N):
    strategy[i] = np.concatenate((F[i].flatten(),H[i].flatten(),w_p[i].flatten(),w_n[i].flatten(),
                                 K_p[i].flatten(),K_n[i].flatten(),D_jm[i].flatten()))
    strategy[i] /= np.sum(strategy[i])

  tolerance = 0.005
  strategy_difference = [1]
  iterations = 0
  strategy_prev = [] # a list of the strategies at each iteration
  strategy_prev.append(strategy.copy())

  while strategy_difference[-1] > tolerance and iterations < max_iters:
    for i in range(N):

      new_strategy = grad_descent_constrained(strategy[i],3,i,i,J,N,K,M,T,phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,
                               theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,
                               dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,F,H,w_p,w_n,K_p,K_n,D_jm)

      if np.count_nonzero(new_strategy[2*M*N:2*M*N+N]) < np.count_nonzero(strategy[i][2*M*N:2*M*N+N]):
        sigmas = correct_scale_params(sigmas,w_p[i],i)
      if np.count_nonzero(new_strategy[2*M*N+N:2*M*N+2*N]) < np.count_nonzero(strategy[i][2*M*N+N:2*M*N+2*N]):
        lambdas = correct_scale_params(lambdas,w_n[i],i)

      # update strategy for this actor
      strategy[i] = new_strategy
    # update strategies for all actors
    strategy_prev.append(strategy.copy())
    # compute difference in strategies
    strategy_difference.append(np.linalg.norm((strategy_prev[-2] - strategy_prev[-1])))
    iterations += 1

    return F,H,w_p,w_n,K_p,K_n,D_jm,sigmas,lambdas






