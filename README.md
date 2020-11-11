# Gen-Modeling-Governance


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



