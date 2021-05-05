import numpy as np
from scipy import optimize
from objective_gradient import objective_grad
import matplotlib.pyplot as plt

def correct_scale_params(scale_params, alloc_params, i):
  '''
  Corrects scale parameters (either sigmas or lambdas) to be consisent with optimization
  results. Takes in scale parameters (2d) and strategy parameters for a particular actor i (1d),
  and sets scale parameters to 0 if the corresponding strategy parameters are 0, ensuring
  that the scale parameters still add to 1.
  '''
  scale_params[:,i][alloc_params==0] = 0
  for i in range(sum(alloc_params==0)):
    scale_params[alloc_params==0][i][scale_params[alloc_params==0][i] != 0] \
        = np.squeeze(np.random.dirichlet(np.ones(len(scale_params[alloc_params==0][i][scale_params[alloc_params==0][i]!=0])),1))
  return scale_params



# If strategy does not have all efforts >= 0, project onto space of legal strategies
def boundary_projection(mu, strategy, plane):
  return np.sum(np.maximum(strategy*plane - mu, 0)) - 1


def grad_descent_constrained(initial_point, alpha, n, l, J, N,K,M,T,
    phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n,
    F,H,W,K_p,dR_match):
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

  grad = objective_grad(x, n, l, J, N,K,M,T,
    phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n,
    F,H,W,K_p,dR_match)

  d = len(x)
    # Follow the projected gradient for a fixed step size alpha
  x = x + alpha*grad
  plane = np.sign(x) # added
  plane[abs(plane)<0.00001] = 1 # added
  if sum(abs(x)) > 1:
    #project point onto plane
    x = x + plane*(1-sum(plane*x))/d # added
    x[abs(x)<0.001] = 0
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

  return x, grad # normally return only x


def nash_equilibrium(max_iters,J,N,K,M,T,
    phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,
    dc_dw_n,dl_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n, match, dR_match, Jac_condition):
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
  alpha = 0.01 # was 0.0001

  # Initialize strategy
#  strategy = np.random.uniform(-1,0,size = (N, 2*M*N + N + M + M**2)) #FOR DEBUGGING
#  strategy /= np.sum(abs(np.squeeze(strategy)),axis=0)
  strategy = np.zeros((N, 2*M*N + N + M))

  # sample to get bridging org objectives
  objectives = np.random.randint(0,N-K,size = K)
  tolerance = alpha/10 #
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
  #
  while (max_diff > tolerance or sum_below_1) and iterations < max_iters:
    # Loop through each actor i
    for i in range(N):
      if i <= N-K-1:
        objective = i
      else:
        objective = objectives[i-(N-K)]

      new_strategy, raw_grad = grad_descent_constrained(strategy[i], alpha, objective, i, J, N,K,M,T,
          phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n,
          F,H,W,K_p,dR_match)

#      if np.sign(F)[0] != np.sign(K_p):
#        match[0] += 1
#
#      if np.any(np.diagonal(J)[1:-M] > 0):
#        Jac_condition[0] += 1

      # Check if there are new zeros in the strategy parameters to see if we need to update scale parameters
      # (e.g. for portion of gain through collaboration) to make sure they are consistent with our new
      # strategy parameters.
      if np.count_nonzero(new_strategy[2*M*N:2*M*N+N]) < np.count_nonzero(strategy[i][2*M*N:2*M*N+N]) and (np.any(sigmas>0) or np.any(lambdas > 0)) :
        sigmas = correct_scale_params(sigmas,W[i],i)
        lambdas = correct_scale_params(lambdas,W[i],i)

      # update strategy and gradient for this actor
      strategy[i] = new_strategy
      grad[i] = raw_grad

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
  plt.figure()
  strategy_diffs = [0]*31 + strategy_diffs
  plt.plot(np.array(strategy_diffs)/max(strategy_diffs))
  strategy_history = np.array(strategy_history).reshape(len(strategy_history),N*(2*M*N + N + M))
  grad_history = np.array(grad_history).reshape(len(grad_history),N*(2*M*N + N + M))
  dist_from_conv = np.linalg.norm(strategy_history[-1] - strategy_history, axis = 1)
  plt.plot(dist_from_conv/max(dist_from_conv),'.')
  plt.plot(strategy_sum)
  return F,H,W,K_p, sigmas,lambdas, converged, strategy_history, grad_history


