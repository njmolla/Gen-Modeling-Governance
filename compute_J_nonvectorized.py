import numpy as np

def compute_J_nonvectorized(N,K,T,phi,psis,alphas,betas,beta_hats,beta_tildes,sigmas,etas,lambdas,eta_bars,mus,rhos,rho_bars,thetas,
theta_bars,omegas,epsilons,ds_dr,de_dr,de_dg,dg_dF,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dw_p,dc_dw_n,dl_dx,di_dK_p,di_dK_n,
dt_dD_jm,di_dy_p,di_dy_n,dtjm_dym,dtmj_dym,dtjm_dyj,dtmj_dyj,F,H,w_p,w_n,K_p,K_n,D_jm):

  #compute Jacobian (non-vectorized)
  Jac = np.zeros([T,T])
  for i in range(T):
    for j in range(T):
      if i == 0:
        if j == 0:
          # drdot/dr
          Jac[i,j] = phi*(ds_dr-np.sum(psis*de_dr))
        elif j < N+1:
          # drdot/dx (1xN+K)
          n = j-1
          Jac[i,j] = -phi * np.sum(psis[n]*np.sum(de_dg[n]*dg_dF[n]*F[n]))
        else:
          # drdot/dy
          m = j - N - 1
          Jac[i,j] = -phi * np.sum(psis[m]*de_dg[m]*dg_dy[m])
      elif i < N+1:
        if j == 0:
          # dxdot/dr
          n = i-1
          Jac[i,j] = np.squeeze(alphas[n]*(betas[n]*db_de[n]*de_dr[n] + beta_hats[n]*dq_da[n]*da_dr[n]))
        elif j < N+1:
          # dxdot/dx
          n1 = i-1
          n2 = j-1
          if i == j:
            # diagonal
            Jac[i,j] = np.squeeze(alphas[n1]*(betas[n1]*db_de[n1]*np.sum(de_dg[n1,:,0]*dg_dF[n1,:,n2]*F[n1,:,n2])
                      + beta_hats[n1]*dq_da[n1]*np.sum(da_dp[n1,:,0]*dp_dH[n1,:,n2]*H[n1,:,n2])
                      - eta_bars[n1]*dl_dx[n1]))
          else:
            # off-diagonal
            Jac[i,j] = np.squeeze(alphas[n1]*(betas[n1]*db_de[n1]*np.sum(de_dg[n1,:,0]*dg_dF[n1,:,n2]*F[n1,:,n2])
                      + beta_hats[n1]*dq_da[n1]*np.sum(da_dp[n1,:,0]*dp_dH[n1,:,n2]*H[n1,:,n2])
                      + beta_tildes[n1]*sigmas[n1,n2]*dc_dw_p[n1,n2]*w_p[n1,n2] - etas[n1]*lambdas[n1,n2]*dc_dw_n[n1,n2]*w_n[n1,n2]))
        else:  # j >= N+1
          n = i-1
          m = j-N-1
          # dxdot/dy
          Jac[i,j] = np.squeeze(alphas[n]*(betas[n]*db_de[n]*de_dg[n,m]*dg_dy[n,m] + beta_hats[n]*dq_da[n]*da_dp[n,m]*dp_dy[n,m]))
      #dydot
      else:  # i >= N + 1
        if j==0:
          Jac[i,j]=0
        # dydot/dx
        elif j >= 1 and j < N+1:
          n = j - 1
          m = i - N - 1
          Jac[i,j] = np.squeeze(mus[0,m,0]*(rhos[0,m,0]*di_dK_p[n,m,0]*K_p[n,m]
             - thetas[0,m,0]*di_dK_n[n,m,0]*K_n[n,m]
             + rho_bars[0,m,0]*np.sum(omegas[0,:,m]*dt_dD_jm[n,:,m]*D_jm[n,:,m])
             - theta_bars[0,m,0]*np.sum(epsilons[0,m,:]*dt_dD_jm[n,m,:]*D_jm[n,m,:])))
        # dydot/dy
        else:
          m1 = i-N-1
          m2 = j-N-1
          if i == j:  # So m1 == m2
            Jac[i,j] = mus[0,m1,0]*(rhos[0,m1,0]*di_dy_p[0,m1,0] - thetas[0,m1,0]*di_dy_n[0,m1,0]
            + rho_bars[0,m1,0]*np.sum(np.multiply(omegas[0,m1,:],dtjm_dym[0,m1,:]))
            - theta_bars[0,m1,0]*np.sum(np.multiply(epsilons[0,m1,:],dtmj_dym[0,m1,:])))
          else:  # i != j, m1 != m2
            Jac[i,j] = mus[0,m1,0]*(rho_bars[0,m1,0]*omegas[:,m1,m2]*dtjm_dyj[:,m1,m2]-theta_bars[0,m1,0]*epsilons[0,m1,0]*dtmj_dyj[:,m1,m2])
    return Jac