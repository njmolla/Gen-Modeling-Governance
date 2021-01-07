import numpy as np
from scipy import optimize


def objective_grad(x):
  v = [4,4,-2]
  return -2*(x-v)

def boundary_projection(mu, strategy):
  return np.sum(np.maximum(strategy - mu, 0)) - 1

def grad_descent(initial_point, max_steps):


  grad = objective_grad(initial_point)
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
        ub = np.sum(x[x>0])
        mu = optimize.brentq(boundary_projection, 0, ub, args=(x))
      except:
        print('bisection bounds did not work')
        raise Exception('bisection bounds did not work')
      x = np.maximum(x - mu, 0)

    # Compute new gradient and update strategy parameters to match x
    grad = objective_grad(x)

    # Project gradient onto the plane sum(efforts) == 1
    grad = grad - np.sum(grad)/len(grad)
#    print(x)
#    print(grad)
    grad_mag = np.linalg.norm(grad)  # to check for convergence

    num_steps += 1
    if grad_mag < 1e-5:
      print('gradient descent convergence reached')
  return x

x = np.random.rand(3)
x = x/sum(x)
sol = grad_descent(x,1500)