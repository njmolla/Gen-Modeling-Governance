import numpy as np
from scipy import optimize
from objective_gradient import objective_grad
from objective_gradient import steady_state_gradient
# import matplotlib.pyplot as plt

def correct_scale_params(sigmas, lambdas, alloc_params, i, N, K, betas, beta_hats, beta_tildes, beta_bars, du_dx, etas, eta_bars):
  '''
  Corrects scale parameters (either sigmas or lambdas) to be consisent with optimization
  results. Takes in scale parameters (2d) and strategy parameters for a particular actor i (1d),
  and sets scale parameters to 0 if the corresponding strategy parameters are 0, ensuring
  that the scale parameters still add to 1.
  '''
  new_zeros_sigmas = (alloc_params <= 0)
  sigmas[i, new_zeros_sigmas] = 0
  all_zeros_sigmas = (np.sum(sigmas, axis=0) == 0)

  # for all columns that now have new zeros (but are not all zeros), renormalize to make sure the rows still sum to 1
  sigmas[:, ~all_zeros_sigmas & new_zeros_sigmas] /= np.broadcast_to(
          np.expand_dims(
              np.sum(sigmas[:, ~all_zeros_sigmas & new_zeros_sigmas], axis = 0),
              axis=0
          ),
          np.shape(sigmas[:, ~all_zeros_sigmas & new_zeros_sigmas])
      )
   
  # Create views of the part for resource users
  user_beta_tildes = beta_tildes[0, :N-K]
  user_beta_hats = beta_hats[0, :N-K]
  user_betas = betas[0, :N-K]
  user_zeros = all_zeros_sigmas[:N-K]

  user_beta_tildes[user_zeros] = 0
  user_beta_sums = user_beta_hats[user_zeros] + user_betas[user_zeros]
  user_beta_hats[user_zeros] /= user_beta_sums
  user_betas[user_zeros] /= user_beta_sums

  # Create views of the part for non-resource users
  nonuser_beta_tildes = beta_tildes[0, N-K:]
  nonuser_beta_bars = beta_bars[0, N-K:]
  nonuser_zeros = all_zeros_sigmas[N-K:]
  nonuser_du_dx = du_dx[N-K:]

  nonuser_beta_tildes[nonuser_zeros] = 0
  nonuser_du_dx[nonuser_zeros] *= nonuser_beta_bars[nonuser_zeros]
  nonuser_beta_bars[nonuser_zeros] = 1
     
  new_zeros_lambdas = (alloc_params >= 0)
  lambdas[i, new_zeros_sigmas] = 0
  all_zeros_lambdas = (np.sum(lambdas, axis=0) == 0)
  # for all columns that now have new zeros (but are not all zeros), renormalize to make sure the rows still sum to 1

  lambdas[:, ~all_zeros_lambdas & new_zeros_lambdas] /= np.broadcast_to(
          np.expand_dims(
              np.sum(lambdas[:, ~all_zeros_lambdas & new_zeros_lambdas], axis = 0),
              axis=0
          ),
          np.shape(lambdas[:, ~all_zeros_lambdas & new_zeros_lambdas])
      )

  etas[0, all_zeros_lambdas] = 0
  eta_bars[all_zeros_lambdas] = 1

  

# If strategy does not have all efforts >= 0, project onto space of legal strategies
def boundary_projection(mu, strategy, plane):
  return np.sum(np.maximum(strategy*plane - mu, 0)) - 1


def grad_descent_constrained(initial_point, alpha, v, n, l, J, N,K,M,T,
    phi,psis,alphas,betas,beta_hats,beta_tildes,beta_bars,sigmas,etas,lambdas,eta_bars,mus,
    ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,du_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n,
    F,H,W,K_p,drdot_dF,dxdot_dF, dydot_dF, drdot_dH, dxdot_dH, dydot_dH, drdot_dW_p, dxdot_dW_p, dydot_dW_p, drdot_dW_n, dxdot_dW_n, dydot_dW_n,drdot_dK_p,
    dxdot_dK_p, dydot_dK_p, drdot_dK_n, dxdot_dK_n, dydot_dK_n):
    
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

  x = initial_point  # strategy
  # calculate how steady state changes with respect to strategy parameters
  dR_dF, dX_dF, dY_dF, dR_dH, dX_dH, dY_dH, dR_dW_p, dX_dW_p, dY_dW_p, dR_dW_n, dX_dW_n, dY_dW_n, dR_dK_p, dX_dK_p, dY_dK_p, dR_dK_n, dX_dK_n, dY_dK_n\
  = steady_state_gradient(x, n, l, J, N,K,M,T,          
    phi,psis,alphas,betas,beta_hats,beta_tildes,beta_bars,sigmas,etas,lambdas,eta_bars,mus,
    ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,du_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n,
    F,H,W,K_p, drdot_dF, dxdot_dF, dydot_dF, drdot_dH, dxdot_dH, dydot_dH, drdot_dW_p, dxdot_dW_p, dydot_dW_p, drdot_dW_n, 
    dxdot_dW_n, dydot_dW_n,drdot_dK_p, dxdot_dK_p, dydot_dK_p, drdot_dK_n, dxdot_dK_n, dydot_dK_n)
          
          
  # calculate how objective changes wrt strategy parameters (the gradient)
  grad = objective_grad(x, n, l, J, N,K,M,T,
    phi,psis,alphas,betas,beta_hats,beta_tildes,beta_bars,sigmas,etas,lambdas,eta_bars,mus,
    ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,du_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n,
    F,H,W,K_p,dR_dF, dX_dF, dY_dF, dR_dH, dX_dH, dY_dH, dR_dW_p, dX_dW_p, dY_dW_p, dR_dW_n, dX_dW_n, dY_dW_n, dR_dK_p, dX_dK_p, dY_dK_p, dR_dK_n, dX_dK_n, dY_dK_n)

  d = len(x)
  v[n] = 0.9*v[n] + alpha*grad
  # Follow the projected gradient for a fixed step size alpha
  x = x + v[n]
  #print(x)
  plane = np.sign(x)
  plane[abs(plane)<0.00001] = 1
  if sum(abs(x)) > 1:
    #project point onto plane
    x = x + plane*(1-sum(plane*x))/d
    x[abs(x)<0.00001] = 0
    if np.sum(x*plane) != 0:
      x /= np.sum(x*plane) # Normalize to be sure (get some errors without this)

    # If strategy does not have all efforts >= 0, project onto space of legal strategies
    if np.any(x*plane < -0.0001):
      try:
        ub = np.sum(abs(x)) #np.sum(abs(x[x*plane>0]))
        mu = optimize.brentq(boundary_projection, 0, ub, args=(x, plane))
      except:
        print('bisection bounds did not work')
        raise Exception('bisection bounds did not work')
      x = plane * np.maximum(x*plane - mu, 0)

  return x, v # normally return only x

def ODE_gradient(N,K,M,T,
    phi,psis,alphas,betas,beta_hats,beta_tildes,beta_bars,sigmas,etas,lambdas,eta_bars,mus,
    ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,du_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n):
  # Compute how the rhs of system changes with respect to each strategy parameter
  drdot_dF = -phi*np.multiply(psis.reshape(1,1,N),np.multiply(de_dg,dg_dF))
  dxdot_dF = np.zeros([N,N,M,N])
#  for i in range(N):
#    dxdot_dF[i,:,:,i] = alphas[0,i]*betas[0,i]*db_de[0,i]*de_dg[0,:,i]*dg_dF[:,:,i]
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
  dxdot_dW_p = np.zeros([N,N,N]) # dxdot_n/dW_k,i is nxkxi
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
  dydot_dK_p[np.arange(0,M),:,np.arange(0,M)] = np.transpose(np.multiply(mus,di_dK_p))

  drdot_dK_n = np.zeros([N,M])
  dxdot_dK_n = np.zeros([N,N,M])
  dydot_dK_n = np.zeros([M,N,M])
  dydot_dK_n[np.arange(0,M),:,np.arange(0,M)] = np.transpose(np.multiply(-mus,di_dK_n))
  return (drdot_dF, dxdot_dF, dydot_dF, drdot_dH, dxdot_dH, dydot_dH, drdot_dW_p, dxdot_dW_p, dydot_dW_p, drdot_dW_n, dxdot_dW_n, dydot_dW_n,drdot_dK_p,
          dxdot_dK_p, dydot_dK_p, drdot_dK_n, dxdot_dK_n, dydot_dK_n)

def nash_equilibrium(max_iters,J,N,K,M,T,
    phi,psis,alphas,betas,beta_hats,beta_tildes,beta_bars,sigmas,etas,lambdas,eta_bars,mus,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,du_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n):
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
  F = np.zeros((N,M,N))  # F_i,m,n is ixmxn positive effort for influencing resource extraction governance $
  H = np.zeros((N,M,N))  # effort for influencing resource access governance $
  W = np.zeros((N,N))  # effort for collaboration. W_i,n is ixn $
  K_p = np.zeros((N,M))  # effort for more influence for gov orgs $
  # step size
  alpha = 0.0001

  # Initialize strategy
  strategy = np.zeros((N, 2*M*N + N + M))
  v = np.zeros((N, 2*M*N + N + M)) # velocity for gradient descent with momentum
  # sample to get bridging org objectives
  objectives = np.random.randint(0,N-K,size = K)
  tolerance = alpha #
  max_diff = 1  # arbitrary initial value, List of differences in euclidean distance between strategies in consecutive iterations
  iterations = 0
  strategy_diffs = []
  strategy_history = []  # a list of the strategies at each iteration
  strategy_sum = []
  strategy_history.append(strategy.copy())
  converged = True
  grad = np.zeros(np.shape(strategy))
  grad_history = []
  sum_below_1 = True
  has_zero_betas = False
  #
  (drdot_dF, dxdot_dF, dydot_dF, drdot_dH, dxdot_dH, dydot_dH, drdot_dW_p, dxdot_dW_p, dydot_dW_p, drdot_dW_n, dxdot_dW_n, dydot_dW_n,drdot_dK_p,
  dxdot_dK_p, dydot_dK_p, drdot_dK_n, dxdot_dK_n, dydot_dK_n) = ODE_gradient(N,K,M,T,
                                                                                     phi,psis,alphas,betas,beta_hats,beta_tildes,beta_bars,sigmas,etas,lambdas,eta_bars,mus,
                                                                                     ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,du_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n)
  while (max_diff > tolerance or sum_below_1) and iterations < max_iters:
    # Loop through each actor i
    for i in range(N):
      if i <= N-K-1:
        objective = i
      else:
        objective = objectives[i-(N-K)]

      new_strategy, v = grad_descent_constrained(strategy[i], alpha, v, objective, i, J, N,K,M,T,
          phi,psis,alphas,betas,beta_hats,beta_tildes,beta_bars,sigmas,etas,lambdas,eta_bars,mus,
          ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,du_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n,
          F,H,W,K_p,drdot_dF, dxdot_dF, dydot_dF, drdot_dH, dxdot_dH, dydot_dH, drdot_dW_p, dxdot_dW_p, dydot_dW_p, drdot_dW_n, dxdot_dW_n, dydot_dW_n,drdot_dK_p,
          dxdot_dK_p, dydot_dK_p, drdot_dK_n, dxdot_dK_n, dydot_dK_n)


      # Check if there are new zeros or changes in the sign of the strategy parameters to see if we need to update scale parameters
      # (e.g. for portion of gain through collaboration) to make sure they are consistent with our new
      # strategy parameters.
      
      # if there are more zeros in the new strategy than the previous, or changes in the sign, and the scale parameters aren't already all zeros
      if np.count_nonzero(new_strategy[2*M*N:2*M*N+N]) < np.count_nonzero(strategy[i][2*M*N:2*M*N+N]) or np.any(strategy[i][2*M*N:2*M*N+N] * new_strategy[2*M*N:2*M*N+N] < 0) and (np.any(sigmas > 0) or np.any(lambdas > 0)) :
        correct_scale_params(sigmas, lambdas, new_strategy[2*M*N:2*M*N+N], i, N, K, betas, beta_hats, beta_tildes, beta_bars, du_dx, etas, eta_bars)
        (drdot_dF, dxdot_dF, dydot_dF, drdot_dH, dxdot_dH, dydot_dH, drdot_dW_p, dxdot_dW_p, dydot_dW_p, drdot_dW_n, dxdot_dW_n, dydot_dW_n,drdot_dK_p,
         dxdot_dK_p, dydot_dK_p, drdot_dK_n, dxdot_dK_n, dydot_dK_n) = ODE_gradient(N,K,M,T,
                                                                                     phi,psis,alphas,betas,beta_hats,beta_tildes,beta_bars,sigmas,etas,lambdas,eta_bars,mus,
                                                                                     ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,du_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n)
        #print('updating scale params')
      # update strategy and gradient for this actor
      strategy[i] = new_strategy
      grad[i] = v[i]

    # update strategies for all actors
    strategy_history.append(strategy.copy())
    grad_history.append(grad.copy())
    strategy_sum.append(min(np.sum(abs(strategy), axis = 1)))
    if np.all(abs(np.sum(abs(strategy), axis = 1) - 1) < 0.01):
      sum_below_1 = False
    if iterations >= 30:
      # compute difference in strategies
      strategy_history_10 = np.array(strategy_history[-30:]).reshape((30,N*(2*M*N + N + M)))
      strategy_diff = np.linalg.norm(strategy_history_10[:29,:]-strategy_history_10[-1,:], axis = 1)
      strategy_diffs.append(strategy_diff[-1])
      max_diff = max(strategy_diff)

    iterations += 1
    if iterations == max_iters - 1:
      converged = False
  if has_zero_betas == True:
    print('sum of betas = 0!!')
  # plt.figure()
  # strategy_diffs = [0]*31 + strategy_diffs
  # plt.plot(np.array(strategy_diffs)/max(strategy_diffs))
  # strategy_history = np.array(strategy_history).reshape(len(strategy_history),N*(2*M*N + N + M))
  # grad_history = np.array(grad_history).reshape(len(grad_history),N*(2*M*N + N + M))
  # dist_from_conv = np.linalg.norm(strategy_history[-1] - strategy_history, axis = 1)
  # plt.plot(dist_from_conv/max(dist_from_conv),'.')
  # plt.plot(strategy_sum)
  return F,H,W,K_p, sigmas,lambdas, converged, strategy_history, grad_history