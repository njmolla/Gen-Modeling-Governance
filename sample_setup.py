import numpy as np

def sample_composition(size,p_N = 0.6,p_K = 0.2):
  '''
  inputs:
    size: total system size (excluding resource)
    p_N, p_M: probabilities of state variables being N and M, respectively
  outputs:
    system composition (number of each type of actor)
  '''
 # Need at least 2 resource users and one gov org
  N = 2
  M = 1
  rand = np.random.rand(size-3)
  # p_N probability of resource users
  N += np.sum(rand < p_N)
  # 20% bridging orgs
  K = np.sum(rand < p_N+p_K) - np.sum(rand < p_N)
  # 20% gov orgs
  M += np.sum(rand > p_N+p_K)
  rand2 = np.random.rand(N-1)
  # choose at random whether guaranteed extractor is just extractor or extractor + accessor
  N1orN2choose = np.random.rand(1)
  if N1orN2choose < 0.5:
    # guaranteed extractor only and 1/3 chance of additional RUs being extractor only
    N1 = 1 + np.sum(rand2 < 0.33)
    # 1/3 chance of additional RUs being extractors + accessors
    N2 = np.sum(rand2 > 0.33) - np.sum(rand2 > 0.66)
  else:
    # 1/3 chance of additional RUs being extractor only
    N1 = np.sum(rand2 < 0.33)
    # guaranteeed extractor + accessor, and 1/3 chance of additional RUs being both
    N2 = 1 + np.sum(rand2 > 0.33) - np.sum(rand2 > 0.66)
  # 1/3 chance of being accessor only
  N3 = np.sum(rand2 > 0.66)
  N = N+K # N includes K for parameter sampling purposes
  T = N + M + 1 # total number of state variables
  return N,N1,N2,N3,K,M,T