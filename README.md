# A Dynamical Systems Approach to the Stability of Complex Governance

A modeling approach that allows for representing a variety of governance systems by sampling different topological and model parameters, and computing the stability of these systems.

## Usage

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

**Diagram of process for a single model realization (as executed by function run_system)**

![code_flowchart](https://user-images.githubusercontent.com/44376656/137199655-fe7e6be0-d745-4419-a1f5-a86405874f79.png)

Reproducing the figures in the paper involves running the functions in ```colormap_experiments.py``` and ```correlation_experiment.py```, which call the function ```sample_composition``` in ```sample_setup.py``` to choose the meta-parameters for each run (only for the colormaps) and then call ```run_system``` repeatedly. This step is computationally intensive, and is recommended to run in parallel. The data is processed and plotted by the functions in ```colormap_plotting.py``` and ```correlation_plotting.py```, respectively. 



