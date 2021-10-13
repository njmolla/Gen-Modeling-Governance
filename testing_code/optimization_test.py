import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

a = -1
b = 1
c = 1

def gradient(strategy):
  return np.array([a-strategy[0],b-strategy[1],c-strategy[2]])

def boundary_projection(mu, strategy, plane):
  return np.sum(np.maximum(strategy*plane - mu, 0)) - 1

def grad_descent_constrained(f, initial_point, max_steps):

  raw_grad = [] # for debugging
  strategies = [] # for debugging
  x = initial_point  # strategy
  strategies.append(x) # for debugging

  alpha = 0.05

  grad = f(x)

  raw_grad.append(grad) # for debugging

  num_steps = 0
  d = len(x)
  while num_steps < max_steps:
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
      if np.any(x*plane < -0.01):
        try:
          ub = np.sum(abs(x)) #np.sum(abs(x[x*plane>0]))
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
    plane[abs(plane)<0.0001] = 1 # added

    num_steps += 1

  return x, raw_grad, strategies # normally return only x

initial_point = np.zeros(3)
#initial_point = initial_point/np.sum(initial_point) # normalize
sol, raw_grad, strategies = grad_descent_constrained(gradient, initial_point, 500)
print(sol)

#xx = np.arange(-1,1.05,0.05)
#X1,X2 = np.meshgrid(xx, xx)
#Z = -0.5*(X1 - a)**2 - 0.5*(X2 - b)**2
#plt.figure()
#plt.contour(X1,X2,Z,20,cmap=plt.cm.Blues_r)
#
## constraint(s)
#plt.plot(xx,(1-np.abs(xx)), color='r')
#plt.plot(xx,(-1+np.abs(xx)), color='r')
#
## decision variables
#xt = strategies
#xt = np.array(xt)
#plt.plot(xt[:,0], xt[:,1],'.-', color='k', linewidth=1)
#plt.scatter(0.4, 0.7)
#
#plt.xlabel('X1')
#plt.ylabel('X2')
#plt.show()