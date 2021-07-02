from sympy.matrices import Matrix

phi, ds_dr, psi, de_dr, de_dg, dg_dF, dg_dy, alpha, beta, db_de, F, eta_bar, dl_dx, mu, rho, \
di_dK_p, K_p, theta, di_dK_n, K_n, di_dy_p, di_dy_n, theta = symbol('phi', 'ds_dr',
                                                                    'psi', 'de_dr', 'de_dg', 'dg_dF', 'dg_dy', 'alpha', 'beta', 'db_de', 'F', 'eta_bar', 'dl_dx', 'mu', 'rho', \
'di_dK_p', 'K_p', theta, di_dK_n, K_n, di_dy_p, di_dy_n, theta)

Jac = Matrix([[phi*(ds_dr-psi*de_dr),-phi*psi*de_dg*dg_dF*F,-phi*psi*de_dg*dg_dy],
             [alpha*(beta*db_de*de_dr),alpha*(beta*db_de*de_dg*dg_dF*F - eta_bar*dl_dx),alpha*(beta*db_de*de_dg*dg_dy)],
             [0,mu*(rho*di_dK_p*K_p-theta*di_dK_n*K_n),mu*(rho*di_dy_p-theta*di_dy_n)]])
Jac.inverse()