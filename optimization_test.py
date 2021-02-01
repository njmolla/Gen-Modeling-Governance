import numpy as np
from scipy import optimize

def gradient(strategy):
  a = -1/3
  b = -1/3
  c = -1/3
  return np.array([a-strategy[0],b-strategy[1],c-strategy[2]])

def boundary_projection(mu, strategy, plane):
  return np.sum(np.maximum(strategy*plane - mu, 0)) - 1

def grad_descent_constrained(f, initial_point, max_steps):

  raw_grad = [] # for debugging
  projected_grad = [] # for debugging
  strategies = [] # for debugging
  x = initial_point  # strategy
  strategies.append(x) # for debugging

  alpha = 0.05

  grad = f(x)

  raw_grad.append(grad) # for debugging

  # figure out which plane to project gradient onto
  plane = np.sign(x)
  plane[abs(x)<0.0001] = np.sign(grad[abs(x)<0.0001])
  #plane[-(M**2):] = 1

  # Project gradient onto the plane sum(efforts) == 1
  grad = grad - np.dot(grad, plane)*plane/sum(abs(plane))
  projected_grad.append(grad) # for debugging

  num_steps = 0

  while num_steps < max_steps:
    # Follow the projected gradient for a fixed step size alpha
    x = x + alpha*grad
#    print('banana:', end = ' ')
#    print(np.sum(x*plane))
    x[abs(x)<0.001] = 0
    x /= np.sum(x*plane) # Normalize to be sure (get some errors without this)

    # If strategy does not have all efforts >= 0, project onto space of legal strategies
    if np.any(x*plane < -0.01):
      try:
        ub = np.sum(abs(x)) #np.sum(abs(x[x*plane>0]))
#        print()
#        print('banana')
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

    # Compute new gradient and update strategy parameters to match x
    grad = f(x)
    raw_grad.append(grad) # for debugging

    # figure out which plane to project gradient onto
    plane = np.sign(x)
    plane[abs(x)<0.001] = np.sign(grad[abs(x)<0.001])
    # plane[-(M**2):] = 1 # for parameters that can only be positive, set to positive

    # Project gradient onto the plane abs(params)=1
    grad = grad - np.dot(grad, plane)*plane/sum(abs(plane))
    projected_grad.append(grad) # for debugging

    grad_mag = np.linalg.norm(grad)  # to check for convergence

    num_steps += 1
#    if grad_mag < 1e-5:
#      print('gradient descent convergence reached')
  return x, raw_grad, projected_grad, strategies # normally return only x

initial_point = np.random.rand(3)
initial_point = -initial_point/np.sum(initial_point) # normalize
print(initial_point)
sol, raw_grad, projected_grad, strategies = grad_descent_constrained(gradient, initial_point, 500)