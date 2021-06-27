import numpy as np

def sample_composition(size,p_N = 0.6,p_K = 0.2, partial_composition=np.array([None,None,None,None,None,None])):
  '''
  inputs:
    size: total system size (excluding resource)
    p_N, p_K: probabilities of state variables being N1-N3 and K, respectively. Remaining probability
    is probability of M
    partial_composition: tuple with number of N, N1, N2, N3, K, and M. Assumes only one of N, N1, N2, or N3
      will be defined and the rest is sampled.

  outputs:
    system composition (number of each type of actor)
  '''
  if np.sum(partial_composition[partial_composition!=None] > size):
    print('composition not compatible with desired size')
  # if the types or number of RUs is not specified, sample until they add to N (which is either sampled or specified)
  if np.all(partial_composition[1:3] == None):
    # N is specified
    if partial_composition[0] != None:
      N = partial_composition[0]
      p_N = 0 # probability of additional RUs
      p_K = 0.5
      rand_actors = np.random.rand(size-(N+1))
    # N sampled
    else:
      print('N sampled')
      # Need at least 2 resource users
      N = 2
      if partial_composition[5] == None:
        rand_actors = np.random.rand(size-(N+1))
      else:
        rand_actors = np.random.rand(size-(N + partial_composition[5]))
      # p_N probability of resource users
      N += np.sum(rand_actors < p_N)

    # sample types of RUs so that they add up to N
    rand_users = np.random.rand(N-1) # for choosing number of each type of user
    # choose at random whether guaranteed extractor is just extractor or extractor + accessor
    N1orN2choose = np.random.rand(1)
    if N1orN2choose < 0.5:
      # guaranteed extractor only and 1/3 chance of additional RUs being extractor only
      N1 = 1 + np.sum(rand_users < 0.33)
      # 1/3 chance of additional RUs being extractors + accessors
      N2 = np.sum(rand_users > 0.33) - np.sum(rand_users > 0.66)
    else:
      # 1/3 chance of additional RUs being extractor only
      N1 = np.sum(rand_users < 0.33)
      # guaranteeed extractor + accessor, and 1/3 chance of additional RUs being both
      N2 = 1 + np.sum(rand_users > 0.33) - np.sum(rand_users > 0.66)
      # 1/3 chance of being accessor only
    N3 = np.sum(rand_users > 0.66)
  else:
    rand_actors = np.random.rand(size-(sum(partial_composition[partial_composition!=None])+1))

  if partial_composition[4] == None:
    # 20% bridging orgs
    K = np.sum(rand_actors < p_N+p_K) - np.sum(rand_actors < p_N)
  else:
    K = partial_composition[4]

  if partial_composition[5] == None:
    # Need at least one decision center
    M = 1
    # 20% decision centers
    M += np.sum(rand_actors > p_N+p_K)
  else:
    M = partial_composition[5]

  # if one of the three types of RUs is specified, evenly split the remaining actors between the other two
  if partial_composition[1] != None:
    # remainder is half N2, half N3
    N1 = partial_composition[1]
    rand2_3 = np.random.rand(size-N1-M-K)
    N2 = np.sum(rand2_3 < 0.5)
    N3 = np.sum(rand2_3 > 0.5)
  elif partial_composition[2] != None:
    # remainder is half N1, half N3
    N2 = partial_composition[2]
    rand1_3 = np.random.rand(size-M-N2-K)
    N1 = np.sum(rand1_3 < 0.5)
    N3 = np.sum(rand1_3 > 0.5)
  elif partial_composition[3] != None:
    # remainder is half N1, half N2
    N3 = partial_composition[3]
    rand1_2 = np.random.rand(size-M-N3-K)
    N1 = np.sum(rand1_2 < 0.5)
    N2 = np.sum(rand1_2 > 0.5)

  N = N1 + N2 + N3 + K # N includes K for parameter sampling purposes
  T = N + M + 1 # total number of state variables
  return N,N1,N2,N3,K,M,T