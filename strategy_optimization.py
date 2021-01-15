import numpy as np
from scipy import optimize
from objective_gradient import objective_grad
import csv
from numba import jit


#@jit(nopython=True)

def correct_scale_params(scale_params, alloc_params, i):
  '''
  Corrects scale parameters (either sigmas or lambdas) to be consisent with optimization
  results. Takes in scale parameters (2d) and strategy parameters for a particular actor i (1d),
  and sets scale parameters to 0 if the corresponding strategy parameters are 0, then ensures
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


def grad_descent_constrained(initial_point, max_steps, n, l, J, N,K,M,T,
    phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,
    F,H,W,K_p,D_jm):
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
  raw_grad = [] # for debugging
  projected_grad = [] # for debugging
  strategies = [] # for debugging
  grad = objective_grad(initial_point, n, l, J, N,K,M,T,
                        phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,
                        F,H,W,K_p,D_jm)

  raw_grad.append(grad) # for debugging
  x = initial_point  # strategy
  strategies.append(x) # for debugging

  # figure out which plane to project gradient onto
  plane = np.sign(x)
  plane[abs(x)<0.0001] = np.sign(grad[abs(x)<0.0001])
  plane[-(M**2):] = 1

  # Project gradient onto the plane sum(efforts) == 1
  grad = grad - np.dot(grad, plane)*plane/sum(abs(plane))
  projected_grad.append(grad) # for debugging
  grad_mag = np.linalg.norm(grad)  # to check for convergence

  alpha = 0.05
  num_steps = 0
  while grad_mag > 1e-5 and num_steps < max_steps:
    # Follow the projected gradient for a fixed step size alpha
    x = x + alpha*grad
#    print('banana:', end = ' ')
#    print(np.sum(x*plane))
    x /= np.sum(x*plane) # Normalize to be sure (get some errors without this)

    # If strategy does not have all efforts >= 0, project onto space of legal strategies
    if np.any(x*plane < 0):
      try:
        ub = np.sum(abs(x)) #np.sum(abs(x[x*plane>0]))
#        print()
#        print(np.sum(np.maximum(x*plane - 0, 0)) - 1)
#        print(x)
#        print(plane)
#        print(np.sum(x*plane))
#        print(np.sum(np.maximum(x*plane - ub, 0)) - 1)
        mu = optimize.brentq(boundary_projection, 0, ub, args=(x, plane))
      except:
        print('bisection bounds did not work')
        raise Exception('bisection bounds did not work')
      x = plane * np.maximum(x*plane - mu, 0)
    strategies.append(x)


    """
    print(raw_grad[-1])
    print(projected_grad[-1])
    print() # for debugging
    print('raw')
    print(raw_grad[-1]) # for debugging
    print('projected_grad')
    print(projected_grad[-1]) # for debugging
    print('point')
    print(x) # for debugging
    print('plane')
    print(plane) # for debugging
    #"""

    # Compute new gradient and update strategy parameters to match x
    grad = objective_grad(x, n, l, J, N,K,M,T,
                          phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,
                          F,H,W,K_p,D_jm)
    raw_grad.append(grad) # for debugging

    # figure out which plane to project gradient onto
    plane = np.sign(x)
    plane[abs(x)<0.001] = np.sign(grad[abs(x)<0.001])
    plane[-(M**2):] = 1 # for parameters that can only be positive, set to positive

    # Project gradient onto the plane abs(params)=1
    grad = grad - np.dot(grad, plane)*plane/sum(abs(plane))
    projected_grad.append(grad) # for debugging

    grad_mag = np.linalg.norm(grad)  # to check for convergence

    num_steps += 1
    if grad_mag < 1e-5:
      print('gradient descent convergence reached')
  return x, raw_grad, projected_grad, strategies # normally return only x


def nash_equilibrium(max_iters,J,N,K,M,T,
    phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,
    F,H,W,K_p,D_jm):
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
  strategy = np.zeros((N, 2*M*N + N + M + M**2))
  for i in range(N):
    strategy[i] = np.concatenate((F[i].flatten(),H[i].flatten(),
                                  W[i].flatten(),K_p[i].flatten(),D_jm[i].flatten()))
    strategy[i] /= np.sum(strategy[i])
  # sample to get bridging org objectives
  objectives = np.random.randint(0,N-K,size = K)
  tolerance = 0.01 #
  strategy_difference = [1]  # arbitrary initial value, List of differences in euclidean distance between strategies in consecutive iterations
  iterations = 0
  strategy_prev = []  # a list of the strategies at each iteration
  strategy_prev.append(strategy.copy())
  diff_with_eq = []
  converged = True
  while strategy_difference[-1] > tolerance and iterations < max_iters:
    # Loop through each actor i
    for i in range(N):
      if i <= N-K-1:
        objective = i
      else:
        objective = objectives[i-(N-K)]


      new_strategy = grad_descent_constrained(strategy[i], 1, objective, i, J, N,K,M,T,
          phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,
          F,H,W,K_p,D_jm)
      # Check if there are new zeros in the strategy parameters to see if we need to update scale parameters
      # (e.g. for portion of gain through collaboration) to make sure they are consistent with our new
      # strategy parameters.
      if np.count_nonzero(new_strategy[2*M*N:2*M*N+N]) < np.count_nonzero(strategy[i][2*M*N:2*M*N+N]):
        sigmas = correct_scale_params(sigmas,W[i],i)
        lambdas = correct_scale_params(lambdas,W[i],i)

      # update strategy for this actor
      strategy[i] = new_strategy

    # update strategies for all actors
    strategy_prev.append(strategy.copy())
    # compute difference in strategies
    strategy_difference.append(np.linalg.norm((strategy_prev[-2] - strategy_prev[-1])))
    iterations += 1
    if iterations == max_iters - 1:
      converged = False
  for i in range(len(strategy_prev)):
      diff_with_eq.append(np.linalg.norm(strategy_prev[i] - strategy_prev[-1]))
  with open('strategies_1actor.csv', 'w+') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerows(np.array(strategy_prev))
#    csvwriter.writerow(strategy_difference)
#    csvwriter.writerow(diff_with_eq)
  return F,H,W,K_p,D_jm, sigmas,lambdas, converged


