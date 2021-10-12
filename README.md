# A Dynamical Systems Approach to the Stability of Complex Governance

A modeling approach that allows for representing a variety of governance systems by sampling different topological and model parameters, and computing the stability of these systems.

**Requirements:** [NumPy](https://numpy.org/install/), [Matplotlib](https://matplotlib.org/stable/users/installing.html), [Pandas](https://pandas.pydata.org/), [Networkx](https://networkx.org/documentation/stable/install.html), [SciPy](https://www.scipy.org/install.html), [
Plotly](https://plotly.com/python/) (optional)

**Running a single model realization:** Define meta-parameters number of resource users, non-resource user actors, and decision centers and connectance. A single model run returns the specific parameters sampled in that realization, the stability, the Jacobian, convergence information for the strategy optimization and the optimized strategy parameters. This can also be achieved by running GM_main.py.
```
  # Example model meta-parameters
  N1 = 1 # number of resource users that benefit from extraction only
  N2 = 1 # number of users with both extractive and non-extractive use
  N3 = 1  # number of users with only non-extractive use
  K = 1 # number of bridging orgs
  M = 2  # number of gov orgs
  T = N1 + N2 + N3 + K + M + 1  # total number of state variables
  
  C = 0.1  # Connectance between governance organizations and resource users (proportion of resource extraction/access interactions influenced by governance)
  
  # Run model realization
  (stability, J, converged, strategy_history, grad, total_connectance, phi, psis, alphas, betas, beta_hats, beta_tildes, 
  beta_bars, sigmas, etas, lambdas, eta_bars, mus, ds_dr, de_dr, de_dg, dg_dF, dg_dy, dp_dy, db_de, da_dr, dq_da, da_dp, 
  dp_dH, dc_dw_p, dc_dw_n, dl_dx, du_dx, di_dK_p, di_dK_n, di_dy_p, di_dy_n, F, H, W, K_p) = run_system(N1, N2, N3, K, M, T, C, sample_exp)
```


sample  
&nbsp;&nbsp;samples stuff  
&nbsp;&nbsp;strategy optimization  
&nbsp;&nbsp;&nbsp;&nbsp;call determine_stability to get Jacobian  
&nbsp;&nbsp;&nbsp;&nbsp;call nash_equilibrium  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;loop until you get to equilibrium  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;loop over each actor  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;call grad_descent_constrained  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;loop until max_steps or convergence  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;call objective_grad  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;call determine_stability to get Jacobian  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;do giant calculation for everything else  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;do slick stuff  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;call correct_scale_params  
&nbsp;&nbsp;check if it's stable  
&nbsp;&nbsp;&nbsp;&nbsp;call determine_stability to check stability and get Jacobian  
&nbsp;&nbsp;&nbsp;&nbsp;if it is not weakly connected (checked using Jacobian), start sample process from the beginning  


----------------------------------------------------------------------------------------------------

**GM_code.py**

*sample*  
  Sets all parameters and then checks if it's stable  
  input: N1,N2,N3,K,M,T, C1,C2  
  output: stability, J, params, strategy_params  
  calls: determine_stability, nash_equilibrium  
  called by: run_multiple, run_once  

*run_multiple*  
  input: N1,N2,N3,K,M,T, C1,C2, num_samples  
  output: proportion of stable webs  
  calls: sample  
  called by: topology_colormap.py  

*run_once*  
  input: N1,N2,N3,K,M,T, C1,C2, num_samples  
  output: stability, total_connectance, J, params, strategy_params  
  calls: sample  
  called by: main  

*main*


----------------------------------------------------------------------------------------------------

**compute_J.py**

*correct_scale_params*  
  Corrects scale parameters to be consisent with optimization results.  
  input: scale_params, alloc_params (TODO: change to strategy_params), i (actor)  
  output: scale_params  
  calls: --  
  called by: nash_equilibrium  

*determine_stability*  
  input: N,K,M,T, params, strategy_params  
  output: J, eigvals, stability  
  calls: --  
  called by: objective_grad, sample  


----------------------------------------------------------------------------------------------------

**strategy_optimization.py**

*objective_grad*  
  .
  input: strategy (for single actor), n (actor whose objective we want to optimize), l (actor whose strategy it is),
         J, N,K,M,T, params, strategy_params  
  output: return the gradient of the objective function at that point for that actor  
  calls: determine_stability  
  called by: grad_descent_constrained  


*boundary_projection*  
  called by: grad_descent_constrained  


*grad_descent_constrained*  
  ???.
  input: initial_point, max_steps, n, l, J, N,K,M,T, params, strategy_params  
  output: new and improved strategy   
  calls: objective_grad, boundary_projection  
  called by: grad_descent_constrained  


*nash_equilibrium*  
  .
  input: max_iters, J, N,K,M,T, params, strategy_params (initial value given by sample function)  
  output: optimized strategy parameters, updated sigmas and lamdas  
  --> look into whether passing in initial strategy_params leads to lots of extra parameter passing  
  calls: grad_descent_constrained, correct_scale_params  
  called by: sample  


----------------------------------------------------------------------------------------------------

**topology_colormap.py**



