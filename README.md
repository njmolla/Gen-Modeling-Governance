# Gen-Modeling-Governance


sample  
  samples stuff  
  strategy optimization  
    call determine_stability to get Jacobian  
    call nash_equilibrium  
      loop until you get to equilibrium  
        loop over each actor  
          call grad_descent_constrained  
            loop until max_steps or convergence  
              call objective_grad  
                call determine_stability to get Jacobian  
                do giant calculation for everything else  
              do slick stuff  
          call correct_scale_params  
  check if it's stable  
    call determine_stability to check stability and get Jacobian  
    if it is not weakly connected (checked using Jacobian), start sample process from the beginning  


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



