# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## A life-cycle consumption model with a discrete choice
#
# - author: Tao Wang
# - date: Jan 2024

# - This notebook extends the basic life cycle class to solve an additional discrete choice problem using upper-envolope EGM 
#
# - We consider a simple example of a discrete choice: invest in financial literacy or not. 
#   - besides consumption, the agent chooses to invest in their own financial literacy, by incurring a utility cost in the current perior for the benefit of a higher return in saving next period. 

import numpy as np
from interpolation import interp, mlinterp
from numba import njit, float64, int64, boolean
from numba.experimental import jitclass
import matplotlib.pyplot as plt
from time import time
from Utility import cal_ss_2markov,mean_preserving_spread
from copy import copy 

from resources_jit import MeanOneLogNormal as lognorm
from SolveLifeCycle import EGM_br

# + code_folding=[0]
## plot configuration 

plt.style.use('seaborn')
plt.rcParams["font.family"] = "Times New Roman" #'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['axes.labelweight'] = 'bold'

## Set the 
plt.rc('font', size=25)
# Set the axes title font size
plt.rc('axes', titlesize=20)
# Set the axes labels font size
plt.rc('axes', labelsize=20)
# Set the font size for x tick labels
plt.rc('xtick', labelsize=20)
# Set the font size for y tick labels
plt.rc('ytick', labelsize=20)
# Set the legend font size
plt.rc('legend', fontsize=20)
# Set the font size of the figure title
plt.rc('figure', titlesize=20)


# -

# ## The Model Class and Solver

# + code_folding=[0]
lc_data = [
    ## model paras
    ('ρ', float64),              # utility parameter CRRA
    ('β', float64),              # discount factor
    ('R',float64),               # Real interest rate factor 
    ('W',float64),               # Wage rate
    ('P', float64[:, :]),        # transition probs for z_t, a persistent state  x
    ('z_val', float64[:]),       # values of z, grid values for the continuous (macro) persistent state    x
    ('sigma_psi', float64),      # permanent shock volatility              x
    ('sigma_eps', float64),      # transitory shock volatility
    ('x',float64),               # MA(1) coefficient, or essentially the autocorrelation coef of non-permanent income
    ('b_y', float64),            # loading of macro state to income        x 
    ('borrowing_cstr',boolean),  ## artificial borrowing constraint if True, natural borrowing constraint if False
    ('U',float64),               # the i.i.d. probability of being unemployed    * 
    ('sigma_psi_2mkv',float64[:]), # markov permanent risks, only 2 for now
    ('sigma_eps_2mkv',float64[:]), # markov transitory risk, only 2 for now
    ('init_b', float64),              ## Initial endowment (possibly from accidental bequests) 
    ('sigma_p_init',float64),         ## standard deviation of initial income
    ('T',int64),                 # years of work                          *   
    ('L',int64),                 # years of life                          * 
    ('G',float64[:]),            # growth rate of permanent income    *
    ('LivPrb',float64[:]),         # the probability of being alive next period 
    ('unemp_insurance',float64),   ## Unemployment insurance replacement ratio 
    ('pension',float64),           ## pension payment to permanent income ratio
    ('ue_markov', boolean),        ## True if 2-state emp/uemp markov 
    ('state_dependent_risk', boolean),     ## True if 2-state risks
    ('adjust_prob',float64),        ## the exogenous probability of being able to adjust consumption plan 
    ('λ', float64),                   ## Income tax rate
    ('λ_SS',float64),                 ## Social security tax 
    ('transfer', float64),            ## Transfer/current permanent income ratio
    ('bequest_ratio',float64),         ## zero: bequest thrown to the ocean; one: fully given to newborns
    ('theta',float64),           ## extrapolation parameter
    ## computational paras
    ('a_grid', float64[:]),      # Exogenous grid over savings
    ('eps_grid', float64[:]),    # Exogenous grid over transitory income shocks (for ma only)
    ('psi_shk_draws', float64[:]), ## draws of permanent income shock 
    ('eps_shk_draws', float64[:]), # draws of MA/transitory income shocks 
    ('shock_draw_size',int64),    ## nb of points drawn for shocks 
    ('psi_shk_mkv_draws',float64[:,:]),  ## 2-state markov on permanent risks 
    ('eps_shk_mkv_draws',float64[:,:]), ## 2-state markov on transitory risks
    ('init_p_draws', float64[:]),     ## Draws of initial permanent income
    ## regarding subjective beliefs
    ('P_sub',float64[:, :]),            # trans probs of belief states
    ('U2U_2mkv',float64[:]),            # state-dependent U2U prob
    ('E2E_2mkv',float64[:]),            # state-dependent E2E prob
    ('sigma_psi_true', float64),      # true permanent shock volatility              
    ('sigma_eps_true', float64),      # ture transitory shock volatility
    ('subjective',boolean),  ## belief is not necessarily equal to true 
    ('state_dependent_belief',boolean),  ## belief is state-dependent
    ('psi_shk_true_draws',float64[:]), ## draws of true permanent income shock 
    ('eps_shk_true_draws',float64[:]), ## draws of true transitory income shock 
    
    ## bequest motive 
    ('q',float64), ## q = 0 if no bequest motive 
    ('ρ_b',float64), ## elasticity of bequest 
    
    ### new value func
    ('value_func',boolean), ## boolean to decide if calculating value funcs 
]


# + code_folding=[1, 151, 155, 167, 173, 186, 189, 239]
@jitclass(lc_data)
class LifeCycleDC:
    """
    A class that stores primitives for the life cycle consumption problem.
    """

    def __init__(self,
                 ρ = 1.0,     ## relative risk aversion  
                 β = 0.99,    ## discount factor
                 P = np.array([[0.9,0.1],
                              [0.2,0.8]]),   ## transitory probability of markov state z
                 z_val = np.array([0.0,
                                   1.0]), ## markov state from low to high  
                 sigma_psi = 0.10,     ## size of permanent income shocks
                 sigma_eps = 0.10,   ## size of transitory income risks
                 x = 0.0,            ## MA(1) coefficient of non-permanent income shocks
                 borrowing_cstr = True,  ## artificial zero borrowing constraint 
                 U = 0.0,   ## unemployment risk probability (0-1)
                 LivPrb = 0.995*np.ones(60),       ## living probability 
                 b_y = 0.0,          ## loading of markov state on income  
                 R = 1.02,           ## interest factor 
                 W = 1.0,            ## Wage rate
                 T = 40,             ## work age, from 25 to 65 (including 65)
                 L = 60,             ## life length 85
                 G = np.ones(60),    ## growth factor list of permanent income 
                 shock_draw_size = 7,
                 grid_max = 5.0,
                 grid_size = 50,
                 ## subjective state dependent 
                 subjective = False,
                 state_dependent_belief = False,
                 P_sub = np.array([[0.5,0.5],
                              [0.5,0.5]]), 
                 sigma_psi_2mkv = np.array([0.05,0.2]),  ## permanent risks in 2 markov states
                 sigma_eps_2mkv = np.array([0.08,0.12]),  ## transitory risks in 2 markov states
                 U2U_2mkv = np.array([0.05,0.1]),         ## U2U in low and high risk mkv state
                 E2E_2mkv = np.array([0.95,0.9]),         ## E2E in low and high risk mkv state
                 theta = 2,               ## asymmetric extrapolation parameter
                 unemp_insurance = 0.0,   #  unemp_insurance = 0.0,   
                 pension = 1.0,           
                 ue_markov = False,    
                 state_dependent_risk = False,
                 adjust_prob = 1.0,
                 sigma_p_init = 0.01,
                 init_b = 0.0,
                 λ = 0.0,
                 λ_SS = 0.0,
                 transfer = 0.0,
                 bequest_ratio = 0.0,
                 sigma_psi_true = 0.10,     ## true size of permanent income shocks
                 sigma_eps_true = 0.10,     ## ture size of transitory income risks  
                 q = 0.0,     # no bequest by default
                 ρ_b = 1.0,
                 ####
                 value_func = False
                ): 
        self.ρ, self.β = ρ, β
        #####################
        self.q, self.ρ_b = q,ρ_b 
        ######################
        self.R = R 
        self.W = W
        self.P, self.z_val = P, z_val
        self.U = U
        #assert n_z == n_mkv, "the number of markov states for income and for risks should be equal"
        self.T,self.L = T,L
        self.G = G
        self.subjective = subjective 

        ###################################################
        ## fork depending on subjective or objective model ##
        #####################################################

        self.sigma_psi = sigma_psi
        self.sigma_eps = sigma_eps
            
        if self.subjective==False:
            self.sigma_psi_true = self.sigma_psi
            self.sigma_eps_true = self.sigma_eps
        else:
            print('reminder: needs to give true risk parameters: sigma_psi_true & sigma_eps_true!')
            self.sigma_psi_true = sigma_psi_true
            self.sigma_eps_true = sigma_eps_true
            
        self.x = x
        self.sigma_p_init = sigma_p_init
        self.init_b = init_b
        self.borrowing_cstr = borrowing_cstr
        self.b_y = b_y
        self.λ = λ
        self.λ_SS= λ_SS
        self.transfer = transfer 
        self.bequest_ratio = bequest_ratio 
        self.sigma_psi_2mkv = sigma_psi_2mkv
        self.sigma_eps_2mkv = sigma_eps_2mkv
        self.LivPrb = LivPrb 
        self.unemp_insurance = unemp_insurance
        self.pension = pension 
        self.ue_markov = ue_markov
        self.adjust_prob = adjust_prob
        self.state_dependent_risk = state_dependent_risk
        
        ## belief 
        self.P_sub = P_sub
        self.state_dependent_belief = state_dependent_belief

            
        ## shocks 
        
        self.shock_draw_size = shock_draw_size
        self.prepare_shocks()
        
        ## saving a grid
        a_grid_regular = np.exp(np.linspace(np.log(1e-6), np.log(grid_max), grid_size-1))
        self.a_grid = np.append(a_grid_regular,np.max(a_grid_regular)*100)
        
        ## ma(1) shock grid 
        if sigma_eps!=0.0 and x!=0.0:
            lb_sigma_ϵ = -sigma_eps**2/2-2*sigma_eps
            ub_sigma_ϵ = -sigma_eps**2/2+2*sigma_eps
            self.eps_grid = np.linspace(lb_sigma_ϵ,ub_sigma_ϵ,grid_size)
        else:
            self.eps_grid = np.array([0.0,0.001])  ## make two points for the c function to be saved correctly  

    
        ## extrapolation coefficient, i.e. higher theta, higher asymmetric response
        self.theta = theta
        
        # Test stability (not needed if it is life-cycle)
        ## this is for infinite horizon problem 
        #assert β * R < 1, "Stability condition failed."  

        
        ############
        self.value_func = value_func
        ############################
        
    ## utility function 
    def u(self,c):
        if self.ρ!=1:
            return c**(1-self.ρ)/(1-self.ρ)
        elif self.ρ==1:
            return np.log(c)
        
    def u_inv(self,u):
        if self.ρ!=1:
            return (u*(1-self.ρ))**(1/(1-self.ρ))
        elif self.ρ==1:
            return np.exp(u)
    
    # marginal utility
    def u_prime(self, c):
        return c**(-self.ρ)

    # inverse of marginal utility
    def u_prime_inv(self, c):
        return c**(-1/self.ρ)
    
    
    #########################
    ## transfer func for v func
    #####
    
    def vTransf(self,v):
        # inverse transform v function after linear interporlation\n",
        return self.u_inv(v)
    
    def vUntransf(self,m):
        # inverse transform v function after linear interporlation\n",
        return self.u(m)
    
    ## a function for the transitory/persistent income component
    ### the fork depending on if discrete-markov bool is on/off
    def Y(self, z, u_shk):
        #from the transitory/ma shock and ue realization  to the income factor
        if self.ue_markov ==False:
            ## z state continuously loading to income
            ## u_shk here represents the cumulated MA shock, for instance, for ma(1), u_shk = phi eps_(t-1) + eps_t
            ## income 
            return np.exp(u_shk + (z * self.b_y))
        elif self.ue_markov ==True:
            ## ump if z ==0 and emp if z==1
            assert len(self.P)==2,"unemployment/employment markov has to be 2 state markov"
            return (z==0)*(self.unemp_insurance) + (z==1)*np.exp(u_shk)
        
    # a function from the log permanent shock to the income factor
    def Γ(self,psi_shk):
        return np.exp(psi_shk)
    
    def prepare_shocks(self):
        subjective = self.subjective
        shock_draw_size = self.shock_draw_size
        sigma_psi = self.sigma_psi
        sigma_eps = self.sigma_eps
        sigma_psi_true = self.sigma_psi_true
        sigma_eps_true = self.sigma_eps_true
        sigma_p_init = self.sigma_p_init
        sigma_psi_2mkv = self.sigma_psi_2mkv
        sigma_eps_2mkv = self.sigma_eps_2mkv
        
        ##################################################################
         ## discretized distributions 
        ##################################################################
                
        psi_shk_dist = lognorm(sigma_psi,100000,shock_draw_size)
        self.psi_shk_draws = np.log(psi_shk_dist.X)  ## the discretized is lognormal variable itself, we work with the log of it
        eps_shk_dist = lognorm(sigma_eps,100000,shock_draw_size)
        self.eps_shk_draws = np.log(eps_shk_dist.X)
        
        ## the draws used for simulation in household block 
        if subjective==False:
            self.psi_shk_true_draws =  self.psi_shk_draws
            self.eps_shk_true_draws =  self.eps_shk_draws
        else:
            psi_shk_true_dist = lognorm(sigma_psi_true,100000,shock_draw_size)
            eps_shk_true_dist = lognorm(sigma_eps_true,100000,shock_draw_size)
            self.psi_shk_true_draws =  np.log(psi_shk_true_dist.X)
            self.eps_shk_true_draws =  np.log(eps_shk_true_dist.X)
            
        
        init_p_dist = lognorm(sigma_p_init,100000,shock_draw_size)
        self.init_p_draws = np.log(init_p_dist.X)
        
        ## draw shocks for various markov state of volatility 
        #sigma_psi_2mkv_r = sigma_psi_2mkv.reshape(n_mkv,-1)
        #sigma_eps_2mkv_r = sigma_eps_2mkv.reshape(n_mkv,-1)
        
        sigma_psi_2mkv_l = lognorm(sigma_psi_2mkv[0],100000,shock_draw_size)
        sigma_psi_2mkv_h = lognorm(sigma_psi_2mkv[1],100000,shock_draw_size)
        
        self.psi_shk_mkv_draws = np.stack((np.log(sigma_psi_2mkv_l.X),
                                         np.log(sigma_psi_2mkv_h.X)))
        
        sigma_eps_2mkv_l = lognorm(sigma_eps_2mkv[0],100000,shock_draw_size)
        sigma_eps_2mkv_h = lognorm(sigma_eps_2mkv[1],100000,shock_draw_size)
        
        self.eps_shk_mkv_draws = np.stack((np.log(sigma_eps_2mkv_l.X),
                                         np.log(sigma_eps_2mkv_h.X)))
        
    def terminal_solution(self):
        
        ## depending on if bequest 
        
        k = len(self.a_grid)
        k2 =len(self.eps_grid)
        n = len(self.P)
        σ_init = np.empty((k,k2,n))
        m_init = np.empty((k,k2,n))
        ################################
        v_init = np.empty((k,k2,n))
        d_init = np.ones((k,k2,n))
        ###############################
        if self.q ==0.0:
            for z in range(n):
                for j in range(k2):
                    m_init[:,j,z] = self.a_grid
                    σ_init[:,j,z] = m_init[:,j,z]
                    if self.value_func:
                        v_init[:,j,z] = self.u(σ_init[:,j,z])
        else:
            for z in range(n):
                for j in range(k2):
                    ## q*a**(-rho_b) = c**(-rho) 
                    ## c = q*a**(rho_b/rho)
                    ## m = a + c
                    #m_init[:,j,z] = self.a_grid 
                    σ_init[:,j,z] = (self.q*self.a_grid**(-self.ρ_b))**(-1/self.ρ)
                    m_init[:,j,z] = self.a_grid + σ_init[:,j,z]
                    if self.value_func:
                        v_init[:,j,z] = self.u(σ_init[:,j,z])+self.q*self.u(m_init[:,j,z]-σ_init[:,j,z])
        return m_init,σ_init,σ_init, d_init,v_init,v_init


# -

# ## Still under development

# + code_folding=[]
## This extends EGM_vfunc to incorporate upper envolope EGM for a hypothetic discrete choice 
#@njit
def EGM_DC(mϵ_in, 
           σ_1_in, ## the consumption policy of discrete choice 1
           σ_2_in, ## the consumption policy of discrete choice 2
           d_in, 
           v_1_in, ## the value function of discrete choice 1
           v_2_in, ## the value function of discrete choice 2
           age_id,
           lc):
    """
    The backward iterating operator for the life-cycle consumption problem,
    using the endogenous grid method.

        * lc is an instance of life cycle model
        * σ_in is a n1 x n2 x n3 dimension consumption policy 
          * n1 = dim(s), n2 = dim(eps), n3 = dim(z)
        * mϵ_in is the same sized grid points of the three state variable 
        * mϵ_in[:,j,z] is the vector of wealth grids corresponding to j-th grid of eps and z-th grid of z 
        * σ_in[i,j,z] is consumption at aϵ_in[i,j,z]
    """    
    # Simplify names
    u_func = lc.u
    u_prime, u_prime_inv = lc.u_prime, lc.u_prime_inv
    vTransf,vUntransf = lc.vTransf,lc.vUntransf
    ρ, β = lc.ρ, lc.β
    R = lc.R

    ## markov state
    P = lc.P
    W = lc.W
    z_val = lc.z_val
    a_grid,eps_grid = lc.a_grid,lc.eps_grid
    psi_shk_draws, eps_shk_draws= lc.psi_shk_draws, lc.eps_shk_draws
    psi_shk_mkv_draws, eps_shk_mkv_draws = lc.psi_shk_mkv_draws, lc.eps_shk_mkv_draws
    borrowing_cstr = lc.borrowing_cstr 
    ue_prob = lc.U  ## uemp prob
    unemp_insurance = lc.unemp_insurance
    adjust_prob = lc.adjust_prob  ## exogenous adjustment probability
    #adjust_cost = lc.adjust_cost  ## adjustment cost of consumption 
    Y = lc.Y
    ####################
    Γ = lc.Γ
    ####################################
    G = lc.G[age_id+1]  ## get the age specific growth rate, G[T] is the sudden drop in retirement from working age
    LivProb = lc.LivPrb[age_id+1]  ## live probabilities
    ####################################
    
    #################################################
    ## get the expected marginal utility scalor
    E_psi_rho = np.mean((G/Γ(psi_shk_draws))**(-ρ))
     #################################################
        
    x = lc.x
    λ = lc.λ
    λ_SS = lc.λ_SS
    transfer = lc.transfer
    pension = lc.pension
    ###################
    
    n = len(P)
    
    # Create consumption functions by linear interpolation
    ########################################################
    σ_1 = lambda m,ϵ,z: mlinterp((mϵ_in[:,0,z],eps_grid),σ_1_in[:,:,z], (m,ϵ)) 
    σ_2 = lambda m,ϵ,z: mlinterp((mϵ_in[:,0,z],eps_grid),σ_2_in[:,:,z], (m,ϵ)) 

    #########################################################
    ## Create value functions by two-step interpolation 
    #########################################################
    ## (interpolate over transfored value function and inverse the transformation)
    
    if lc.value_func:
        v_1_func_in = lambda m,ϵ,z: mlinterp((mϵ_in[:,0,z],
                                             eps_grid),
                                            vTransf(v_1_in[:,:,z]), 
                                            (m,ϵ)) 
        v_1_func_in  = lambda m,ϵ,z: vUntransf(v_1_func_in(m,ϵ,z))
        
        v_2_tfunc_in = lambda m,ϵ,z: mlinterp((mϵ_in[:,0,z],
                                             eps_grid),
                                            vTransf(v_2_in[:,:,z]), 
                                            (m,ϵ)) 
        v_2_func_in  = lambda m,ϵ,z: vUntransf(v_2_tfunc_in(m,ϵ,z))
    
    # Allocate memory
    σ_1_out = np.empty_like(σ_1_in)  ## grid_size_s X grid_size_ϵ X grid_size_z
    σ_2_out = np.empty_like(σ_2_in)  ## grid_size_s X grid_size_ϵ X grid_size_z

    d_out = np.empty_like(d_in)   
    
    v_1_out = np.empty_like(v_1_in)
    v_2_out = np.empty_like(v_2_in)
    
    ############################################################################
    ## repeat the following chunk for each discrete choice: adjust versus non-adjust
    ### the policy next period will be useful to compute marginal utilities 
    ##########################################################################
    
    ## for the choice=1 people, c is determined by foc as before
    # Obtain c_i at each a_i, z, store in σ_out[i, z], computing
    # the expectation term by computed by averaging over discretized equally probable points of the distributions
    for i, a in enumerate(a_grid):
        for j, eps in enumerate(eps_grid):
            for z in range(n):
                # Compute expectation
                Ez = 0.0
                Ev = 0.0
                for z_hat in range(n):
                    z_val_hat = z_val[z_hat]
                    ########################################
                    if lc.state_dependent_risk == False:
                        pass
                    else:
                        psi_shk_draws = psi_shk_mkv_draws[z_hat,:]
                        eps_shk_draws = eps_shk_mkv_draws[z_hat,:]
                    ########################################
                    for eps_shk in eps_shk_draws:
                        for psi_shk in psi_shk_draws:
                            ## for employed next period 
                            Γ_hat = Γ(psi_shk) 
                            u_shk = x*eps+eps_shk
                            age = age_id + 1
                            if age <=lc.T-1:   #till say 39, because consumption policy for t=40 changes   
                                ## work 
                                Y_hat = (1-λ)*(1-λ_SS)*Y(z_val_hat,u_shk) ## conditional on being employed
                                c_hat_1 = σ_1(R/(G*Γ_hat) * a + Y_hat+transfer,eps_shk,z_hat)
                                c_hat_2 = σ_2(R/(G*Γ_hat) * a + Y_hat+transfer,eps_shk,z_hat)
                                d_hat = d_in(R/(G*Γ_hat) * a + Y_hat+transfer,eps_shk,z_hat)==1
                                c_hat = d_hat*c_hat_1+(1-d_hat)*c_hat_2
                                    
                                utility = (G*Γ_hat)**(1-ρ)*u_prime(c_hat)
                                v_hat_1 = v_1_func_in(R/(G*Γ_hat) * a + Y_hat+transfer,eps_shk,z_hat)
                                v_hat_2 = v_2_func_in(R/(G*Γ_hat) * a + Y_hat+transfer,eps_shk,z_hat)
                                value = d_hat*v_hat_1+(1-d_hat)*v_hat_2

                                ## for unemployed next period
                                Y_hat_u = (1-λ)*unemp_insurance
                                c_hat_1_u = σ_1(R/(G*Γ_hat) * a + Y_hat_u+transfer,eps_shk,z_hat)
                                c_hat_2_u = σ_2(R/(G*Γ_hat) * a + Y_hat_u+transfer,eps_shk,z_hat)
                                d_hat_u = d_in(R/(G*Γ_hat) * a + Y_hat_u+transfer,eps_shk,z_hat)==1
                                c_hat_u = d_hat_u*c_hat_1_u+(1-d_hat_u)*c_hat_2_u

                                utility_u = (G*Γ_hat)**(1-ρ)*u_prime(c_hat_u)
                            
                                Ez += LivProb*((1-ue_prob)*utility * P[z, z_hat]+
                                           ue_prob*utility_u* P[z, z_hat]
                                          )
                                
                                v_hat_u_1 = v_1_func_in(R/(G*Γ_hat) * a + Y_hat_u+transfer,eps_shk,z_hat)
                                v_hat_u_2 = v_2_func_in(R/(G*Γ_hat) * a + Y_hat_u+transfer,eps_shk,z_hat)
                                value_u = d_hat_u*v_hat_u_1+(1-d_hat_u)*v_hat_u_2
                                                                
                                Ev += LivProb*((1-ue_prob)*value * P[z, z_hat]+
                                           ue_prob*value_u* P[z, z_hat]
                                          )
                            else:
                                ## retirement
                                Y_R = pension
                                ## no income shocks affecting individuals
                                Γ_hat = 1.0 
                                eps_shk = 0.0
                                c_hat_1_R = σ_1(R/(G*Γ_hat) * a + Y_R+transfer,eps_shk,z_hat)
                                c_hat_2_R = σ_2(R/(G*Γ_hat) * a + Y_R+transfer,eps_shk,z_hat)
                                m_cloest = np.digitize(R/(G*Γ_hat) * a + Y_R+transfer,mϵ_in[:,0,z_hat])
                                
                                d_hat_R = d_in[m_cloest,0,z_hat]==1
                                c_hat_R = d_hat_R*c_hat_1_R+(1-d_hat_R)*c_hat_2_R
                                                                    
                                utility = (G*Γ_hat)**(1-ρ)*u_prime(c_hat_R)
                                Ez += LivProb*utility * P[z, z_hat]
                                
                                v_hat_R_1 = v_1_func_in(R/(G*Γ_hat) * a + Y_R+transfer,eps_shk,z_hat)
                                v_hat_R_2 = v_2_func_in(R/(G*Γ_hat) * a + Y_R+transfer,eps_shk,z_hat)
                                value_R = d_hat_R*v_hat_R_1+(1-d_hat_R)*v_hat_R_2
                                
                                Ev += LivProb*value_R* P[z, z_hat]
                            
                Ez = Ez / (len(psi_shk_draws)*len(eps_shk_draws))
                Ev = Ev / (len(psi_shk_draws)*len(eps_shk_draws))

                ## the last step depends on if adjustment is fully flexible
                ## the two cases can collapse into one formula, but I keep them separately.
                if adjust_prob ==1.0:
                    σ_1_out[i, j, z] =  u_prime_inv(β * R* Ez)
                elif adjust_prob <1.0:
                    σ_1_out[i, j, z] =  u_prime_inv(β * R*adjust_prob/(1-β*R*(1-adjust_prob))* Ez)
                    ## need to check if LivProb should be in the denominator
                    
                ## value func 
                v_1_out[i,j,z] = u_func(σ_1_out[i, j, z])+β*LivProb*Ev
                
  
    # Calculate endogenous asset grid
    mϵ_1_out = np.empty_like(σ_1_out)

    for j,ϵ in enumerate(eps_grid):
        for z in range(n):
            mϵ_1_out[:,j,z] = a_grid + σ_1_out[:,j,z]

    # Fixing a consumption-asset pair at for the constraint region
    for j,ϵ in enumerate(eps_grid):
        for z in range(n):
            if borrowing_cstr==True:  ## either hard constraint is zero or positive probability of losing job
                σ_1_out[0,j,z] = 0.0
                mϵ_1_out[0,j,z] = 0.0
            else:
                if age <=lc.T-1:
                    σ_1_out[0,j,z] = 0.0
                    ## the lowest transitory draw at state z  
                    if lc.state_dependent_belief:
                        self_min_a = - np.exp(np.min(eps_shk_mkv_draws[z,:]))*G/R 
                    else:
                        self_min_a = - np.exp(np.min(eps_shk_draws))*G/R

                    self_min_a = min(self_min_a,-unemp_insurance/R)
                    mϵ_1_out[0,j,z] = self_min_a
                else:
                    σ_1_out[0,j,z] = 0.0
                    self_min_a = - pension*G/R
                    mϵ_1_out[0,j,z] = self_min_a
                    
                    
     ## for the choice=2 people
    
    R_high = R*1.1 ## 1.1 is the factor of benefit of financial literacy investment
    for i, a in enumerate(a_grid):
        for j, eps in enumerate(eps_grid):
            for z in range(n):
                # Compute expectation
                Ez = 0.0
                Ev = 0.0
                for z_hat in range(n):
                    z_val_hat = z_val[z_hat]
                    ########################################
                    if lc.state_dependent_risk == False:
                        pass
                    else:
                        psi_shk_draws = psi_shk_mkv_draws[z_hat,:]
                        eps_shk_draws = eps_shk_mkv_draws[z_hat,:]
                    ########################################
                    for eps_shk in eps_shk_draws:
                        for psi_shk in psi_shk_draws:
                            ## for employed next period 
                            Γ_hat = Γ(psi_shk) 
                            u_shk = x*eps+eps_shk
                            age = age_id + 1
                            if age <=lc.T-1:   #till say 39, because consumption policy for t=40 changes   
                                ## work 
                                Y_hat = (1-λ)*(1-λ_SS)*Y(z_val_hat,u_shk) ## conditional on being employed
                                c_hat_1 = σ_1(R_high/(G*Γ_hat) * a + Y_hat+transfer,eps_shk,z_hat)
                                c_hat_2 = σ_2(R_high/(G*Γ_hat) * a + Y_hat+transfer,eps_shk,z_hat)
                                #######################################################
                                #closest_m = np.digitize(R_high/(G*Γ_hat) * a + Y_R+transfer,mϵ_in[:,0,z_hat])
                                d_hat = d_in(R_high/(G*Γ_hat) * a + Y_R+transfer,eps_shk,z_hat)
                                c_hat = d_hat*c_hat_1+(1-d_hat)*c_hat_2
                                    
                                utility = (G*Γ_hat)**(1-ρ)*u_prime(c_hat)
                                v_hat_1 = v_1_func_in(R_high/(G*Γ_hat) * a + Y_hat+transfer,eps_shk,z_hat)
                                v_hat_2 = v_2_func_in(R_high/(G*Γ_hat) * a + Y_hat+transfer,eps_shk,z_hat)
                                value = d_hat*v_hat_1+(1-d_hat)*v_hat_2

                                ## for unemployed next period
                                Y_hat_u = (1-λ)*unemp_insurance
                                c_hat_1_u = σ_1(R_high/(G*Γ_hat) * a + Y_hat_u+transfer,eps_shk,z_hat)
                                c_hat_2_u = σ_2(R_high/(G*Γ_hat) * a + Y_hat_u+transfer,eps_shk,z_hat)
                                d_hat_u = d_in(R_high/(G*Γ_hat) * a + Y_hat_u+transfer,eps_shk,z_hat)==1
                                c_hat_u = d_hat_u*c_hat_1_u+(1-d_hat_u)*c_hat_2_u

                                utility_u = (G*Γ_hat)**(1-ρ)*u_prime(c_hat_u)
                            
                                Ez += LivProb*((1-ue_prob)*utility * P[z, z_hat]+
                                           ue_prob*utility_u* P[z, z_hat]
                                          )
                                
                                v_hat_u_1 = v_1_func_in(R_high/(G*Γ_hat) * a + Y_hat_u+transfer,eps_shk,z_hat)
                                v_hat_u_2 = v_2_func_in(R_high/(G*Γ_hat) * a + Y_hat_u+transfer,eps_shk,z_hat)
                                value_u = d_hat_u*v_hat_u_1+(1-d_hat_u)*v_hat_u_2
                                                                
                                Ev += LivProb*((1-ue_prob)*value * P[z, z_hat]+
                                           ue_prob*value_u* P[z, z_hat]
                                          )
                            else:
                                ## retirement
                                Y_R = pension
                                ## no income shocks affecting individuals
                                Γ_hat = 1.0 
                                eps_shk = 0.0
                                c_hat_1_R = σ_1(R_high/(G*Γ_hat) * a + Y_R+transfer,eps_shk,z_hat)
                                c_hat_2_R = σ_2(R_high/(G*Γ_hat) * a + Y_R+transfer,eps_shk,z_hat)
                                d_hat_R = d_in(R_high/(G*Γ_hat) * a + Y_R+transfer,eps_shk,z_hat)==1
                                c_hat_R = d_hat_R*c_hat_1_u+(1-d_hat_R)*c_hat_2_u
                                                                    
                                utility = (G*Γ_hat)**(1-ρ)*u_prime(c_hat_R)
                                Ez += LivProb*utility * P[z, z_hat]
                                v_hat_R_1 = v_1_func_in(R_high/(G*Γ_hat) * a + Y_R+transfer,eps_shk,z_hat)
                                v_hat_R_2 = v_2_func_in(R_high/(G*Γ_hat) * a + Y_R+transfer,eps_shk,z_hat)
                                value_R = d_hat_R*v_hat_R_1+(1-d_hat_R)*v_hat_R_2
                                
                                Ev += LivProb*value_R* P[z, z_hat]
                            
                Ez = Ez / (len(psi_shk_draws)*len(eps_shk_draws))
                Ev = Ev / (len(psi_shk_draws)*len(eps_shk_draws))

                ## the last step depends on if adjustment is fully flexible
                ## the two cases can collapse into one formula, but I keep them separately.
                if adjust_prob ==1.0:
                    σ_2_out[i, j, z] =  u_prime_inv(β * R_high* Ez)
                elif adjust_prob <1.0:
                    σ_2_out[i, j, z] =  u_prime_inv(β * R_high*adjust_prob/(1-β*R_high*(1-adjust_prob))* Ez)
                    ## need to check if LivProb should be in the denominator
                    
                ## value func 
                v_2_out[i,j,z] = u_func(σ_2_out[i, j, z])-0.5+β*LivProb*Ev
                ## 0.5 is the one-time utility cost
                
    # Calculate endogenous asset grid
    mϵ_2_out = np.empty_like(σ_2_out)

    for j,ϵ in enumerate(eps_grid):
        for z in range(n):
            mϵ_2_out[:,j,z] = a_grid + σ_2_out[:,j,z]

    # Fixing a consumption-asset pair at for the constraint region
    for j,ϵ in enumerate(eps_grid):
        for z in range(n):
            if borrowing_cstr==True:  ## either hard constraint is zero or positive probability of losing job
                σ_2_out[0,j,z] = 0.0
                mϵ_2_out[0,j,z] = 0.0
            else:
                if age <=lc.T-1:
                    σ_2_out[0,j,z] = 0.0
                    ## the lowest transitory draw at state z  
                    if lc.state_dependent_belief:
                        self_min_a = - np.exp(np.min(eps_shk_mkv_draws[z,:]))*G/R 
                    else:
                        self_min_a = - np.exp(np.min(eps_shk_draws))*G/R

                    self_min_a = min(self_min_a,-unemp_insurance/R)
                    mϵ_2_out[0,j,z] = self_min_a
                else:
                    σ_2_out[0,j,z] = 0.0
                    self_min_a = - pension*G/R
                    mϵ_2_out[0,j,z] = self_min_a
   
    ### need some function to recast v2 to v1's grid 
    
    if lc.value_func:
        v_2_tfunc_out = lambda m,ϵ,z: mlinterp((mϵ_2_out[:,0,z],
                                             eps_grid),
                                            vTransf(v_2_out[:,:,z]), 
                                            (m,ϵ)) 
        v_2_func_out  = lambda m,ϵ,z: vUntransf(v_2_tfunc_out(m,ϵ,z))
        
    for i,m in enumerate(mϵ_1_out[:,0,0]):
        for j,ϵ in enumerate(eps_grid):
            for z in range(n):
                 v_2_out[i,j,z] = v_2_func_out(m,0,z)
    
    ## using choice 1's grid for both choices 
    mϵ_out = mϵ_1_out
    
    ## decide discrete choice by comparing v1 and v2 over the same grids 
    
    d_out = v_1_out>v_2_out      

    ##############################################################   
    ### then compare the v_out_0, v_out_1 ... to get the d_out
    ### calc_log_sum_choice_probs is used here 
    ### calc_prim_kink gives the primary kink
    ##############################################################
    ## then need to calculate v_out 
    ## take care of the kink grids 
    ############################################################

    return mϵ_out, σ_1_out,σ_2_out, d_out, v_1_out,v_2_out


# -

# ## Need to modify 

# + code_folding=[1]
## for life-cycle/finite horizon problem 
def solve_model_backward_iter_DC(model,        # Class with model information
                              mϵ_vec,        # Initial condition for assets and MA shocks
                              σ_1_vec,        # Initial condition for consumption
                              σ_2_vec,        # Initial condition for consumption
                              d_vec,
                              v_1_vec,
                              v_2_vec,
                              br = False):

    ## memories for life-cycle solutions 
    n_grids1 = σ_1_vec.shape[0]
    n_grids2 = σ_1_vec.shape[1]
    n_z = len(model.P)                       
    mϵs_new =  np.empty((model.L,n_grids1,n_grids2,n_z),dtype = np.float64)
    
    σs_1_new =σs_2_new =  np.empty((model.L,n_grids1,n_grids2,n_z),dtype = np.float64)
    vs_1_new = vs_2_new = np.empty((model.L,n_grids1,n_grids2,n_z),dtype = np.float64)
    ds_new = np.empty((model.L,n_grids1,n_grids2,n_z),dtype =bool)
    
    mϵs_new[0,:,:,:] = mϵ_vec
    σs_1_new[0,:,:,:] = σ_1_vec
    σs_2_new[0,:,:,:] = σ_2_vec
    ds_new[0,:,:,:] = d_vec
    vs_1_new[0,:,:,:] = v_1_vec
    vs_2_new[0,:,:,:] = v_2_vec
        
    for year2L in range(1,model.L): ## nb of years till L from 0 to Model.L-2
        age = model.L-year2L
        age_id = age-1
        #print("at work age of "+str(age))
        mϵ_vec_next, σ_1_vec_next,σ_2_vec_next,d_vec_next,v_1_vec_next,v_2_vec_next = mϵs_new[year2L-1,:,:,:],σs_1_new[year2L-1,:,:,:],σs_2_new[year2L-1,:,:,:],ds_new[year2L-1,:,:,:],vs_1_new[year2L-1,:,:,:],vs_2_new[year2L-1,:,:,:]
        if br==False:
            mϵ_new, σ_1_new,σ_2_new,d_new,v_1_new,v_2_new = EGM_DC(mϵ_vec_next, σ_1_vec_next, σ_2_vec_next,d_vec_next, v_1_vec_next, v_2_vec_next, age_id, model)
        elif br==True:
            #print('subjective model with stochastic risk')
            mϵ_new, σ_new = EGM_br(mϵ_vec_next, σ_vec_next, age_id, model)
        mϵs_new[year2L,:,:,:] = mϵ_new
        σs_1_new[year2L,:,:,:] = σ_1_new
        σs_2_new[year2L,:,:,:] = σ_2_new
        ds_new[year2L,:,:,:] = d_new
        vs_1_new[year2L,:,:,:] = v_1_new
        vs_2_new[year2L,:,:,:] = v_2_new
    return mϵs_new, σs_1_new,σs_2_new,ds_new,vs_1_new,vs_2_new


# + code_folding=[1]
## for infinite horizon problem 
def solve_model_iter_DC(model,        # Class with model information
                     me_vec,        # Initial condition for assets and MA shocks
                     σ_1_vec,       
                     σ_2_vec,       
                     d_vec,
                     v_1_vec,
                     v_2_vec,
                      tol=1e-6,
                      max_iter=2000,
                      verbose=True,
                      print_skip=50,
                      br = False):

    # Set up loop
    i = 0
    error = tol + 1

    ## memories for life-cycle solutions
    while i < max_iter and error > tol:
        if br==False:
                me_new, σ_1_new,σ_2_new,d_new,v_1_new,v_2_new = EGM_DC(me_vec, σ_1_vec,σ_2_vec,d_vec,v_1_vec,v_2_vec,0,model)
        elif br==True:
            #print('subjective model with stochastic risk')
            mϵ_new, σ_new = EGM_br(me_vec, σ_vec, 0, model)

        error = np.max(np.abs(σ_vec - σ_new))
        i += 1
        if verbose and i % print_skip == 0:
            print(f"Error at iteration {i} is {error}.")
        me_vec, σ_1_vec,σ_2_vec,d_vec,v_1_vec,v_2_vec = np.copy(me_new), np.copy(σ_1_new),np.copy(σ_2_new), np.copy(d_new), np.copy(v_1_new),np.copy(v_2_new)

    if i == max_iter:
        print("Failed to converge!")

    if verbose and i < max_iter:
        print(f"\nConverged in {i} iterations.")

    return me_vec, σ_1_vec,σ_2_vec,d_vec,v_1_vec,v_2_vec


# -

# ## Initialize the model

# + code_folding=[0]
if __name__ == "__main__":


    ## parameters 
    ###################

    U = 0.2 ## transitory ue risk
    U0 = 0.0 ## transitory ue risk

    ## life cycle 

    T = 40
    L = 60
    G = np.ones(L)
    YPath = np.cumprod(G)

# + code_folding=[0]
## a deterministic income profile 
if __name__ == "__main__":

    plt.title('Deterministic Life-cycle Income Profile \n')
    plt.plot(YPath,'ro')
    plt.xlabel('Age')
    plt.ylabel(r'$\hat Y$')
# -

# ## Life-Cycle Problem 

# + code_folding=[0]
if __name__ == "__main__":
    lc_paras = {'sigma_psi':0.15, # permanent 
                'sigma_eps': 0.1, #transitory
                'U':U,
                'ρ':2,
                'R':1.01,
                'T':T,
                'L':L,
                'G':G,
                'β':0.97,
                'x':0.0,  # no MA shock 
                'borrowing_cstr':True,
                'b_y':0.0, #no persistent state
                'unemp_insurance':0.15,
                #'q':1.0,
               #'ρ_b':2.0,
                'value_func': True
               }

# + code_folding=[]
if __name__ == "__main__":
    lc_dc = LifeCycleDC(**lc_paras)

# + code_folding=[2]
# Initial the end-of-period consumption policy of σ = consume all assets

if __name__ == "__main__":

    ## initial consumption functions 
    
    m_init,σ_1_init,σ_2_init,d_init,v_1_init,v_2_init = lc_dc.terminal_solution()
    plt.title('Value function in the last period')
    plt.plot(m_init[int(m_init.shape[0]/2):-1,1,1],
             v_1_init[int(m_init.shape[0]/2):-1,1,1],
            'o-')
    plt.xlabel('m')
    plt.ylabel('v(m)')

# + code_folding=[]
if __name__ == "__main__":

    t_start = time()

    ### this line is very important!!!!
    #### need to regenerate shock draws for new sigmas
    lc_dc.prepare_shocks()

    ## terminal solution
    m_init,σ_1_init,σ_2_init,d_init,v_1_init,v_2_init = lc_dc.terminal_solution()
    
    m_init,σ_1_init,σ_2_init,d_init,v_1_init,v_2_init = EGM_DC(m_init,σ_1_init,σ_2_init,d_init,v_1_init,v_2_init, lc_dc.L-3, lc_dc)

        
    ## solve backward
    #ms_star_basic, σs_1_star_basic,σs_2_star_basic, ds_start_basic, vs_1_star_basic,vs_2_star_basic = solve_model_backward_iter_DC(lc_dc,
    #                                                                                                                               m_init,
    #                                                                                                                               σ_1_init,
    #                                                                                                                               σ_2_init,
    #                                                                                                                               d_init,
    #                                                                                                                               v_1_init,
    #                                                                                                                               v_2_init)
                             
    t_finish = time()

    print("Time taken, in seconds: "+ str(t_finish - t_start))

# + code_folding=[0]
if __name__ == "__main__":


    ## plot c func at different age /asset grid
    years_left = [0,1,30,40]
    start_m_at = 30
    
    n_sub = len(years_left)

    eps_fix = 0 ## the first eps grid 

    fig,axes = plt.subplots(1,n_sub,figsize=(6*n_sub,6))

    for x,year in enumerate(years_left):
        age = lc_pt.L-year
        i = lc_pt.L-age
        for k,sigma_psi in enumerate(sigma_psi_ls):
            m_plt,v_plt = ms_stars[k][i,start_m_at:-1,eps_fix,0],vs_stars[k][i,start_m_at:-1,eps_fix,0]
            axes[x].plot(m_plt,
                         v_plt,
                         label = r'$\sigma_\psi=$'+str(sigma_psi),
                         lw=3,
                        )
        axes[x].legend()
        axes[x].set_xlim(0.0,np.max(m_plt))
        axes[x].set_xlabel('asset')
        axes[0].set_ylabel('v')
        axes[x].set_title(r'$age={}$'.format(age))
# -

# ## Infinite horizon problem

# + code_folding=[]
if __name__ == "__main__":
    
    inf_liv_paras = copy(lc_paras)
    
    ## initialize a model instance

    inf_liv = LifeCycleDC(**inf_liv_paras)

    ## initial guess of consumption functions 

    m_init,σ_1_init,σ_2_init,d_init,v_1_init,v_2_init= inf_liv.terminal_solution()
   
    plt.title('Value function in the last period')
    plt.plot(m_init[int(m_init.shape[0]/2):-1,1,1],
             v_1_init[int(m_init.shape[0]/2):-1,1,1],
            'o-',label='choice=1')
    
    plt.plot(m_init[int(m_init.shape[0]/2):-1,1,1],
             v_2_init[int(m_init.shape[0]/2):-1,1,1],
            '*',label='choice=2')
    
    plt.xlabel('m')
    plt.ylabel('v(m)')
    plt.legend()
# -

if __name__ == "__main__":

    t_start = time()

    ### this line is very important!!!!
    #### need to regenerate shock draws for new sigmas
    lc.prepare_shocks()

    ## terminal solution
    m_init,σ_init,v_init = lc.terminal_solution()

    ## solve backward
    ms_star_basic, σs_star_basic,vs_star_basic = solve_model_backward_iter_vfunc(lc,
                                                                         m_init,
                                                                         σ_init,
                                                                         v_init)

    t_finish = time()

    print("Time taken, in seconds: "+ str(t_finish - t_start))

if __name__ == "__main__":

    t_start = time()

    ## set different ma parameters 
    m_inf_star, σ_1_inf_star,σ_2_inf_star,d_inf_star,v_1_inf_star,v_2_inf_star = solve_model_iter_DC(inf_liv,
                                                                                              m_init,
                                                                                              σ_1_init,
                                                                                              σ_2_init,
                                                                                              d_init,
                                                                                              v_1_init,
                                                                                              v_2_init)

    t_finish = time()

    print("Time taken, in seconds: "+ str(t_finish - t_start))   


    ## plot c func 

    plt.plot(m_inf_star[0:-1,0,0],
             σ_1_inf_star[0:-1,0,0],
             lw=3
            )
    plt.xlabel('asset')
    plt.ylabel('c')
    plt.title('Infinite horizon solution')

# ## Infinite horizon with adjustment inertia
#
#

# + code_folding=[0]
if __name__ == "__main__":


    ## Initialize a model instance
    imp_adjust_paras = copy(lc_paras)
    imp_adjust_paras['adjust_prob'] = 0.1
    
    imp_adjust = LifeCycle(**imp_adjust_paras)

    ## initial consumption functions 
    m_init,σ_init = imp_adjust.terminal_solution()

    t_start = time()

    x_ls = [0.0]
    ms_imp_stars =[]
    σs_imp_stars = []
    for i,x in enumerate(x_ls):

        ## set different ma parameters 
        inf_liv.x = x
        m_imp_star, σ_imp_star = solve_model_iter(imp_adjust,
                                                  m_init,
                                                  σ_init)
        ms_imp_stars.append(m_imp_star)
        σs_imp_stars.append(σ_imp_star)


    t_finish = time()

    print("Time taken, in seconds: "+ str(t_finish - t_start))   


    ## plot c func at different age /asset grid

    eps_ls = [0]

    ms_imp_star = ms_imp_stars[0]
    σs_imp_star = σs_imp_stars[0]

    for y,eps in enumerate(eps_ls):
        plt.plot(ms_imp_star[0:-1,eps,1],
                 σs_imp_star[0:-1,eps,1],
                 '-',
                 label = 'imperfect adjustment',
                 lw=3
                )
        plt.plot(ms_inf_star[0:-1,eps,1],
                 σs_inf_star[0:-1,eps,1],
                 '--',
                 label = 'perfect adjustment',
                 lw=3
                )
        plt.legend()
        plt.xlabel('asset')
        plt.ylabel('c')
        plt.title('Infinite horizon solution')
