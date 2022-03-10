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

# ## A life-cycle consumption  model under objective/subjective risk perceptions
#
# - author: Tao Wang
# - date: Feb 2022
# - this is a companion notebook to the paper "Perceived income risks"

# - This notebook builds on a standard life-cycle consumption model with uninsured income risks and extends it to allow subjective beliefs about income risks
#   - Preference/income process
#
#       - CRRA utility 
#       - During work: labor income risk: permanent + MA(1)/persistent/transitory/2-state Markov between UE and EMP or between low and high risk + i.i.d. unemployment shock
#        -  a deterministic growth rate of permanent income over the life cycle 
#       - During retirement: receives a constant pension proportional to permanent income (no permanent/transitory income risks)
#       - A positive probability of death before terminal age 
#   

import numpy as np
import pandas as pd
from quantecon.optimize import brent_max, brentq
from interpolation import interp, mlinterp
from scipy import interpolate
import numba as nb
from numba import jit,njit, float64, int64, boolean
from numba.experimental import jitclass
import matplotlib as mp
import matplotlib.pyplot as plt
# %matplotlib inline
from quantecon import MarkovChain
import quantecon as qe 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from time import time
from scipy import sparse as sp
import scipy.sparse.linalg
from scipy import linalg as lg 
from numba.typed import List
from Utility import cal_ss_2markov

from resources_jit import MeanOneLogNormal as lognorm

# + code_folding=[]
## figure plotting configurations

mp.rc('xtick', labelsize=11) 
mp.rc('ytick', labelsize=11) 

mp.rc('legend',fontsize=11)
plt.rc('font',size=11) 
# -

# ## The Model Class and Solver

# + code_folding=[]
lc_data = [
    ## model paras
    ('ρ', float64),              # utility parameter CRRA
    ('β', float64),              # discount factor
    ('R',float64),               # Real interest rate factor 
    ('W',float64),               # Wage rate
    ('P', float64[:, :]),        # transition probs for z_t, a persistent (macro) state  x
    ('z_val', float64[:]),       # values of z, grid values for the continous (macro) persistent state    x
    ('sigma_psi', float64),      # permanent shock volatility              x
    ('sigma_eps', float64),      # transitory shock volatility
    ('x',float64),               # MA(1) coefficient, or essentially the autocorrelation coef of non-permanent income
    ('b_y', float64),            # loading of macro state to income        x 
    ('borrowing_cstr',boolean),  ## artifitial borrowing constraint if True, natural borrowing constraint if False
    ('U',float64),               # the i.i.d. probability of being unemployed    * 
    ('sigma_psi_2mkv',float64[:]), # markov permanent risks, only 2 for now
    ('sigma_eps_2mkv',float64[:]), # markov transtory risk, only 2 for now
    ('init_b', float64),              ## Initial endowment (possibly from accidental bequests) 
    ('sigma_p_init',float64),         ## standard devaition of initial income
    ('T',int64),                 # years of work                          *   
    ('L',int64),                 # years of life                          * 
    ('G',float64[:]),            # growth rate of permanent income    *
    ('LivPrb',float64),         # the probability of being alive next period 
    ('unemp_insurance',float64),   ## Unemployment insurance replacement ratio 
    ('pension',float64),           ## pension payment to permanent income ratio
    ('ue_markov', boolean),        ## True if 2-state emp/uemp markov 
    ('adjust_prob',float64),        ## the exogenous probability of being able to adjust consumption plan 
    ('λ', float64),                   ## Income tax rate
    ('λ_SS',float64),                 ## Social security tax 
    ('transfer', float64),            ## Transfer/current permanent income ratio
    ('bequest_ratio',float64),         ## zero: bequest thrown to the ocea; one: fully given to newborns
    ('theta',float64),           ## extrapolation paramete
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
    ('sigma_psi_true', float64),      # true permanent shock volatility              
    ('sigma_eps_true', float64),      # ture transitory shock volatility
    ('subjective',boolean),  ## belief is not necessarily equal to true 
    ('psi_shk_true_draws',float64[:]), ## draws of true permanent income shock 
    ('eps_shk_true_draws',float64[:]) ## draws of true transitory income shock 
]


# + code_folding=[]
@jitclass(lc_data)
class LifeCycle:
    """
    A class that stores primitives for the life cycle consumption problem.
    """

    def __init__(self,
                 ρ = 1.0,     ## relative risk aversion  
                 β = 0.99,  ## discount factor
                 P = np.array([[0.9,0.1],
                              [0.2,0.8]]),   ## transitory probability of markov state z
                 z_val = np.array([0.0,
                                   1.0]), ## markov state from low to high  
                 sigma_psi = 0.10,     ## size of permanent income shocks
                 sigma_eps = 0.10,   ## size of transitory income risks
                 x = 0.0,            ## MA(1) coefficient of non-permanent inocme shocks
                 borrowing_cstr = True,  ## artificial zero borrowing constraint 
                 U = 0.0,   ## unemployment risk probability (0-1)
                 LivPrb = 0.995,       ## living probability 
                 b_y = 0.0,          ## loading of markov state on income  
                 sigma_psi_2mkv = np.array([0.05,0.2]),  ## permanent risks in 2 markov states
                 sigma_eps_2mkv = np.array([0.08,0.12]),  ## transitory risks in 2 markov states
                 R = 1.02,           ## interest factor 
                 W = 1.0,            ## Wage rate
                 T = 40,             ## work age, from 25 to 65 (including 65)
                 L = 60,             ## life length 85
                 G = np.ones(60),    ## growth factor list of permanent income 
                 shock_draw_size = 7,
                 grid_max = 5.0,
                 grid_size = 50,
                 theta = 2,               ## assymetric extrapolative parameter
                 unemp_insurance = 0.15,   #  unemp_insurance = 0.0,   
                 pension = 1.0,           
                 ue_markov = False,    
                 adjust_prob = 1.0,
                 sigma_p_init = 0.01,
                 init_b = 0.0,
                 λ = 0.10,
                 λ_SS = 0.1,
                 transfer = 0.0,
                 bequest_ratio = 0.0,
                 sigma_psi_true = 0.10,     ## true size of permanent income shocks
                 sigma_eps_true = 0.10,     ## ture size of transitory income risks  
                 subjective = False): 
        self.ρ, self.β = ρ, β
        self.R = R 
        self.W = W
        n_z = len(z_val)
        self.P, self.z_val = P, z_val
        n_mkv = len(sigma_psi_2mkv)
        assert n_z == n_mkv, "the number of markov states for income and for risks should be equal"
        self.T,self.L = T,L
        self.G = G
        self.subjective =subjective 
        
        ###################################################
        ## fork depending on subjective or objective model ##
        #####################################################
        self.sigma_psi = sigma_psi
        self.sigma_eps = sigma_eps
            
        if self.subjective==False:
            self.sigma_psi_true = self.sigma_psi
            self.sigma_eps_true = self.sigma_eps
        else:
            print('remidner: needs to give true risk parameters: sigma_psi_true & sigma_eps_true!')
            self.sigma_psi_true = self.sigma_psi_true
            self.sigma_eps_true = self.sigma_eps_true
            
        self.x = x
        self.sigma_p_init = sigma_p_init
        self.init_b = init_b
        self.borrowing_cstr = borrowing_cstr
        self.b_y = b_y
        self.λ = λ
        self.λ_SS= λ
        self.transfer = transfer 
        self.bequest_ratio = bequest_ratio 
        self.sigma_psi_2mkv = sigma_psi_2mkv
        self.sigma_eps_2mkv = sigma_eps_2mkv
        self.LivPrb = LivPrb 
        self.unemp_insurance = unemp_insurance
        self.pension = pension 
        self.ue_markov = ue_markov
        self.adjust_prob = adjust_prob
        
        
        ## shocks 
        
        self.shock_draw_size = shock_draw_size
        
        ##################################################################
         ## discretized distributions 
        ##################################################################
        
        ## these codes use equiprobable discretized distributions at the cost of not being jittable 
        
        psi_shk_dist = lognorm(sigma_psi,100000,shock_draw_size)
        self.psi_shk_draws = np.log(psi_shk_dist.X)  ## discretized is lognormal variable itself, we work with the log of it
        eps_shk_dist = lognorm(sigma_eps,100000,shock_draw_size)
        self.eps_shk_draws = np.log(eps_shk_dist.X)
        
        ## the draws used for simulation in household block 
        if self.subjective==False:
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
        sigma_psi_2mkv_r = sigma_psi_2mkv.reshape(n_mkv,-1)
        sigma_eps_2mkv_r = sigma_eps_2mkv.reshape(n_mkv,-1)
        
        sigma_psi_2mkv_l = lognorm(sigma_psi_2mkv[0],100000,shock_draw_size)
        sigma_psi_2mkv_h = lognorm(sigma_psi_2mkv[1],100000,shock_draw_size)
        
        self.psi_shk_mkv_draws = np.stack((np.log(sigma_psi_2mkv_l.X),
                                         np.log(sigma_psi_2mkv_h.X)))
        
        sigma_eps_2mkv_l = lognorm(sigma_eps_2mkv[0],100000,shock_draw_size)
        sigma_eps_2mkv_h = lognorm(sigma_eps_2mkv[1],100000,shock_draw_size)
        
        self.eps_shk_mkv_draws = np.stack((np.log(sigma_eps_2mkv_l.X),
                                         np.log(sigma_eps_2mkv_h.X)))
        
        
        ## saving a grid
        self.a_grid = np.exp(np.linspace(np.log(1e-6), np.log(grid_max), grid_size))
        
        ## ma(1) shock grid 
        if sigma_eps!=0.0:
            lb_sigma_ϵ = -sigma_eps**2/2-2*sigma_eps
            ub_sigma_ϵ = -sigma_eps**2/2+2*sigma_eps
            self.eps_grid = np.linspace(lb_sigma_ϵ,ub_sigma_ϵ,grid_size)
        else:
            self.eps_grid = np.array([0.0,0.001])  ## make two points for the c function to be saved correctly  

    
        ## extrapolaton coefficient, i.e. higher theta, higher asymmetric response
        self.theta = theta
        
        # Test stability (not needed if it is life-cycle)
        ## this is for infinite horizon problem 
        #assert β * R < 1, "Stability condition failed."

    ## utility function 
    def u(self,c):
        if self.ρ!=1:
            return c**(1-self.ρ)/(1-ρ)
        elif self.ρ==1:
            return np.log(c)
    
    # marginal utility
    def u_prime(self, c):
        return c**(-self.ρ)

    # inverse of marginal utility
    def u_prime_inv(self, c):
        return c**(-1/self.ρ)
    
    ## value function 
    def V(self,m):
        return None

    ## a function for the transitory/persistent income component
    ### the fork depending on if discrete-markov bool is on/off
    def Y(self, z, u_shk):
        #from the transitory/ma shock and ue realization  to the income factor
        if self.ue_markov ==False:
            ## z state continuously loading to inome
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


# + code_folding=[]
## This function takes the consumption values at different 
## grids of state variables variables from period t+1, and 
## the model class, then generates the consumption values at t.
## It depends on the age t since the income is different before 
## after the retirement. 

@njit
def EGM(mϵ_in,
        σ_in,
        age_id, ## the period id for which the c policy is computed, the first period age_id=0, last period age_id=L-1, retirement after age_id=T-1
        lc):
    """
    The Coleman--Reffett operator for the life-cycle consumption problem,
    using the endogenous grid method.

        * lc is an instance of life cycle model
        * σ_in is a n1 x n2 x n3 dimension consumption policy 
          * n1 = dim(s), n2 = dim(eps), n3 = dim(z)
        * mϵ_in is the same sized grid points of the three state variable 
        * mϵ_in[:,j,z] is the vector of wealth grids corresponding to j-th grid of eps and z-th grid of z 
        * σ_in[i,j,z] is consumption at aϵ_in[i,j,z]
    """    
    # Simplify names
    u_prime, u_prime_inv = lc.u_prime, lc.u_prime_inv
    R, ρ, P, β = lc.R, lc.ρ, lc.P, lc.β
    W = lc.W
    z_val = lc.z_val
    a_grid,eps_grid = lc.a_grid,lc.eps_grid
    psi_shk_draws, eps_shk_draws= lc.psi_shk_draws, lc.eps_shk_draws
    borrowing_cstr = lc.borrowing_cstr 
    ue_prob = lc.U  ## uemp prob
    LivProb = lc.LivPrb  ## live probabilituy 
    unemp_insurance = lc.unemp_insurance
    ue_markov = lc.ue_markov            ## bool for 2-state markov transition probability 
    adjust_prob = lc.adjust_prob  ## exogenous adjustment probability 
     
    Y = lc.Y
    ####################
    ρ = lc.ρ
    Γ = lc.Γ
    ####################################
    G = lc.G[age_id+1]  ## get the age specific growth rate, G[T] is the sudden drop in retirement from working age
    ####################################

    x = lc.x
    λ = lc.λ
    λ_SS = lc.λ_SS
    transfer = lc.transfer
    
    ###################
    
    n = len(P)

    # Create consumption functions by linear interpolation
    ########################################################
    σ = lambda m,ϵ,z: mlinterp((mϵ_in[:,0,z],eps_grid),σ_in[:,:,z], (m,ϵ)) 
    ########## need to replace with multiinterp 

    # Allocate memory
    σ_out = np.empty_like(σ_in)  ## grid_size_a X grid_size_ϵ X grid_size_z

    # Obtain c_i at each a_i, z, store in σ_out[i, z], computing
    # the expectation term by computed by averaging over discretized equally probable points of the distributions
    for i, a in enumerate(a_grid):
        for j, eps in enumerate(eps_grid):
            for z in range(n):
                # Compute expectation
                Ez = 0.0
                for z_hat in range(n):
                    z_val_hat = z_val[z_hat]
                    for eps_shk in eps_shk_draws:
                        for psi_shk in psi_shk_draws:
                            ## for employed next period 
                            Γ_hat = Γ(psi_shk) 
                            u_shk = x*eps+eps_shk
                            age = age_id + 1
                            if age <=lc.T-1:   #till say 39, because consumption policy for t=40 changes   
                                ## work 
                                Y_hat = (1-λ)*(1-λ_SS)*Y(z_val_hat,u_shk) ## conditional on being employed 
                                c_hat = σ(R/(G*Γ_hat) * a + Y_hat+transfer,eps_shk,z_hat)
                                utility = (G*Γ_hat)**(1-ρ)*u_prime(c_hat)

                                ## for unemployed next period
                                Y_hat_u = (1-λ)*unemp_insurance
                                c_hat_u = σ(R/(G*Γ_hat) * a + Y_hat_u+transfer,eps_shk,z_hat)
                                utility_u = (G*Γ_hat)**(1-ρ)*u_prime(c_hat_u)
                            
                                Ez += LivProb*((1-ue_prob)*utility * P[z, z_hat]+
                                           ue_prob*utility_u* P[z, z_hat]
                                          )
                            else:
                                ## retirement
                                Y_R = lc.pension
                                ## no income shcoks affecting individuals 
                                Γ_hat = 1.0 
                                eps_shk = 0.0
                                c_hat = σ(R/(G*Γ_hat) * a + (Y_R+transfer),eps_shk,z_hat)
                                utility = (G*Γ_hat)**(1-ρ)*u_prime(c_hat)
                                Ez += LivProb*utility * P[z, z_hat]
                            
                Ez = Ez / (len(psi_shk_draws)*len(eps_shk_draws))
                ## the last step depends on if adjustment is fully flexible
                if adjust_prob ==1.0:
                    σ_out[i, j, z] =  u_prime_inv(β * R* Ez)
                elif adjust_prob <1.0:
                    σ_out[i, j, z] =  adjust_prob/(1-LivProb*β*R*(1-adjust_prob))*u_prime_inv(β * R* Ez)

    # Calculate endogenous asset grid
    mϵ_out = np.empty_like(σ_out)

    for j,ϵ in enumerate(eps_grid):
        for z in range(n):
            mϵ_out[:,j,z] = a_grid + σ_out[:,j,z]

    # Fixing a consumption-asset pair at for the constraint region
    for j,ϵ in enumerate(eps_grid):
        for z in range(n):
            if borrowing_cstr==True:  ## either hard constraint is zero or positive probability of losing job
                σ_out[0,j,z] = 0.0
                mϵ_out[0,j,z] = 0.0
            #elif borrowing_cstr==False and ue_markov==True:
            #    σ_out[0,j,z] = 0.0
            #    aϵ_out[0,j,z] = min(0.0,-unemp_insurance/R)
            else:
                σ_out[0,j,z] = 0.0
                self_min_a = - np.exp(np.min(eps_shk_draws))*G/R
                self_min_a = min(self_min_a,-unemp_insurance/R)
                aϵ_out[0,j,z] = self_min_a

    return mϵ_out, σ_out


# + code_folding=[]
## the operator under markov stochastic risks 
## now the permanent and transitory risks are 
## different between markov states. 

@njit
def EGM_sv(mϵ_in, 
           σ_in, 
           age_id,
           lc):
    """
    The Coleman--Reffett operator for the life-cycle consumption problem,
    using the endogenous grid method.

        * lc is an instance of life cycle model
        * σ_in is a n1 x n2 x n3 dimension consumption policy 
          * n1 = dim(s), n2 = dim(eps), n3 = dim(z)
        * mϵ_in is the same sized grid points of the three state variable 
        * mϵ_in[:,j,z] is the vector of asset grids corresponding to j-th grid of eps and z-th grid of z 
        * σ_in[i,j,z] is consumption at aϵ_in[i,j,z]
    """

    # Simplify names
    u_prime, u_prime_inv = lc.u_prime, lc.u_prime_inv
    R, ρ, P, β = lc.R, lc.ρ, lc.P, lc.β
    z_val = lc.z_val
    a_grid,eps_grid = lc.a_grid,lc.eps_grid
    psi_shk_draws, eps_shk_draws= lc.psi_shk_draws, lc.eps_shk_draws
    borrowing_cstr = lc.borrowing_cstr
    ue_prob = lc.U  ## uemp prob
    unemp_insurance = lc.unemp_insurance
    LivProb = lc.LivPrb  ## live probabilituy
    ue_markov = lc.ue_markov
    psi_shk_mkv_draws, eps_shk_mkv_draws = lc.psi_shk_mkv_draws, lc.eps_shk_mkv_draws 
    adjust_prob = lc.adjust_prob  ## exogenous adjustment probability 
    Y = lc.Y
    ####################
    ρ = lc.ρ
    Γ = lc.Γ
    ####################################
    G = lc.G[age_id+1]   ## get the age specific 
    ####################################    
    x = lc.x
    λ = lc.λ
    transfer = lc.transfer
    pension = lc.pension
    
    ###################
    T = lc.T
    L = lc.L
    
    ###################
    n = len(P)

    # Create consumption function by linear interpolation
    ########################################################
    σ = lambda m,ϵ,z: mlinterp((mϵ_in[:,0,z],eps_grid),σ_in[:,:,z], (m,ϵ)) 
    ########## need to replace with multiinterp 

    # Allocate memory
    σ_out = np.empty_like(σ_in)  ## grid_size_s X grid_size_ϵ X grid_size_z

    # Obtain c_i at each s_i, z, store in σ_out[i, z], computing
    # the expectation term by averaging over different equally probable distrete points of shocks
    for i, a in enumerate(a_grid):
        for j, eps in enumerate(eps_grid):
            for z in range(n):
                # Compute expectation
                Ez = 0.0
                for z_hat in range(n):  
                    z_val_hat = z_val[z_hat]
                    psi_shk_draws = psi_shk_mkv_draws[z_hat,:]
                    eps_shk_draws = eps_shk_mkv_draws[z_hat,:]
                    for eps_shk in eps_shk_draws:
                        for psi_shk in psi_shk_draws:
                            Γ_hat = Γ(psi_shk) 
                            u_shk = x*eps+eps_shk
                            age = age_id + 1
                            if age<=lc.T-1:
                                # work  
                                Y_hat = (1-λ)*Y(z_val_hat,u_shk) ## conditional on being employed 
                                c_hat = σ(R/(G*Γ_hat) * a + Y_hat+transfer,eps_shk,z_hat)
                                utility = (G*Γ_hat)**(1-ρ)*u_prime(c_hat)

                                ## for unemployed next period
                                Y_hat_u = (1-λ)*unemp_insurance
                                c_hat_u = σ(R/(G*Γ_hat) * a + Y_hat_u+transfer,eps_shk,z_hat)
                                utility_u = (G*Γ_hat)**(1-ρ)*u_prime(c_hat_u)
                                Ez += LivProb*((1-ue_prob)*utility * P[z, z_hat]+
                                               ue_prob*utility_u* P[z, z_hat]
                                              )
                            else:
                                ## retirement
                                Y_R = lc.pension
                                ## no income shcoks affecting individuals 
                                Γ_hat = 1.0 
                                eps_shk = 0.0
                                c_hat = σ(R/(G*Γ_hat) * a + (Y_R+transfer),eps_shk,z_hat)
                                utility = (G*Γ_hat)**(1-ρ)*u_prime(c_hat)
                                Ez += LivProb*utility * P[z, z_hat]
                Ez = Ez / (len(psi_shk_draws)*len(eps_shk_draws))
                ## the last step depends on if adjustment is fully flexible
                if adjust_prob ==1.0:
                    σ_out[i, j, z] =  u_prime_inv(β * R* Ez)
                elif adjust_prob <=1.0:
                    σ_out[i, j, z] =  adjust_prob/(1-β*R*(1-adjust_prob))*u_prime_inv(β * R* Ez)

    # Calculate endogenous asset grid
    mϵ_out = np.empty_like(σ_out)
                
    for j,ϵ in enumerate(eps_grid):
        for z in range(n):
            mϵ_out[:,j,z] = a_grid + σ_out[:,j,z]

    # Fixing a consumption-asset pair at (0, 0) improves interpolation
    for j,ϵ in enumerate(eps_grid):
        for z in range(n):
            if borrowing_cstr==True:  ## either hard constraint is zero or positive probability of losing job
                σ_out[0,j,z] = 0.0
                mϵ_out[0,j,z] = 0.0
            #elif borrowing_cstr==True or ue_markov==True:
            #    print('case2')
            #    σ_out[0,j,z] = 0.0
            #    aϵ_out[0,j,z] = min(0.0,-unemp_insurance/R)
            else:
                if age <=T-1:
                    σ_out[0,j,z] = 0.0
                   
                    self_min_a = - np.exp(np.min(eps_shk_mkv_draws))*G/R 
                    ## the lowest among 2 markv states
                    self_min_a = min(self_min_a,-unemp_insurance/R)
                    mϵ_out[0,j,z] = self_min_a
                else:
                    σ_out[0,j,z] = 0.0
                    self_min_a = - pension*G/R
                    mϵ_out[0,j,z] = self_min_a


    return mϵ_out, σ_out


# + code_folding=[3]
## this function describes assymetric extrapolative rule from realized income shock to the perceived risk 

@njit
def extrapolate(theta,
                x,
                eps_shk):
    """
    extrapolation function from realized eps_shk from unbiased risk x to the subjective risk x_sub
    x_sub = x when eps_shk = 0  
    theta governs the degree of extrapolation 
    """
    if x ==0.0:
        alpha=0.0
    else:
        alpha=np.log((1-x)/x) ## get the alpha for unbiased x
    x_sub = 1/(1+np.exp(alpha-theta*eps_shk))
    return x_sub


# + code_folding=[4]
## subjective agent
### transitory shock affects risk perceptions

@njit
def EGM_br(mϵ_in, 
         σ_in, 
         age_id,
         lc):
    """
    UNDER BOUNDED RATIONALITY assumption
    The Coleman--Reffett operator for the life-cycle consumption problem. 
    using the endogenous grid method.

        * lc is an instance of life cycle model
        * σ_in is a n1 x n2 x n3 dimension consumption policy 
          * n1 = dim(s), n2 = dim(eps), n3 = dim(z)
        * mϵ_in is the same sized grid points of the three state variable 
        * mϵ_in[:,j,z] is the vector of asset grids corresponding to j-th grid of eps and z-th grid of z 
        * σ_in[i,j,z] is consumption at aϵ_in[i,j,z]
    """

    # Simplify names
    u_prime, u_prime_inv = lc.u_prime, lc.u_prime_inv
    R, ρ, P, β = lc.R, lc.ρ, lc.P, lc.β
    z_val = lc.z_val
    a_grid,eps_grid = lc.a_grid,lc.eps_grid
    psi_shk_draws, eps_shk_draws= lc.psi_shk_draws, lc.eps_shk_draws
    borrowing_cstr = lc.borrowing_cstr
    ue_prob = lc.U  ## uemp prob
    unemp_insurance = lc.unemp_insurance
    LivProb = lc.LivPrb  ## live probabilituy
    ue_markov = lc.ue_markov
    adjust_prob = lc.adjust_prob  ## exogenous adjustment probability 
    Y = lc.Y
    ####################
    ρ = lc.ρ
    Γ = lc.Γ
    ####################################
    G = lc.G[age_id+1]   ## get the age specific 
    ####################################  
    x = lc.x
    λ = lc.λ
    transfer = lc.transfer
    pension = lc.pension
    
    ###################
    T = lc.T
    L = lc.L
    
    ###################
    theta = lc.theta 
    sigma_eps = lc.sigma_eps
    eps_mean = -sigma_eps**2/2
    ###################
    
    n = len(P)

    # Create consumption function by linear interpolation
    ########################################################
    σ = lambda a,ϵ,z: mlinterp((mϵ_in[:,0,z],eps_grid),σ_in[:,:,z], (a,ϵ)) 
    ########## need to replace with multiinterp 

    # Allocate memory
    σ_out = np.empty_like(σ_in)  ## grid_size_s X grid_size_ϵ X grid_size_z

    # Obtain c_i at each s_i, z, store in σ_out[i, z], computing
    # the expectation term by Monte Carlo
    for i, a in enumerate(a_grid):
        for j, eps in enumerate(eps_grid):
            ##############################################################
            #x_sj = extrapolate(theta,
            #                   lc.x,
            #                   eps-eps_mean) ## sj: subjective 
            sigma_eps_sj = 0.05*np.sqrt((eps-eps_mean)**2)+0.95*lc.sigma_eps
            
            eps_shk_dist_sj= lognorm(sigma_eps_sj,100000,len(eps_shk_draws))
            eps_shk_draws_sj = np.log(eps_shk_dist_sj.X)
            #np.random.seed(166789)
            #eps_shk_draws_sj = sigma_eps_sj*np.random.randn(lc.shock_draw_size)-sigma_eps_sj**2/2
            #############################################################
            for z in range(n):
                # Compute expectation
                Ez = 0.0
                for z_hat in range(n):
                    z_val_hat = z_val[z_hat]
                    ################################
                    for eps_shk in eps_shk_draws_sj:
                        ############################
                        for psi_shk in psi_shk_draws:
                            Γ_hat = Γ(psi_shk) 
                            ###############
                            u_shk = x*eps+eps_shk
                            ####################
                            age = age_id+1
                            if age <=lc.T-1:
                                # work  
                                Y_hat = (1-λ)*Y(z_val_hat,u_shk) ## conditional on being employed 
                                c_hat = σ(R/(G*Γ_hat) * a + Y_hat+transfer,eps_shk,z_hat)
                                utility = (G*Γ_hat)**(1-ρ)*u_prime(c_hat)

                                ## for unemployed next period
                                Y_hat_u = (1-λ)*unemp_insurance
                                c_hat_u = σ(R/(G*Γ_hat) * a + Y_hat_u+transfer ,eps_shk,z_hat)
                                utility_u = (G*Γ_hat)**(1-ρ)*u_prime(c_hat_u)
                                Ez += LivProb*((1-ue_prob)*utility * P[z, z_hat]+
                                               ue_prob*utility_u* P[z, z_hat]
                                              )
                            else:
                                
                                ## retirement
                                Y_R = lc.pension
                                ## no income shcoks affecting individuals 
                                Γ_hat = 1.0 
                                eps_shk = 0.0
                                c_hat = σ(R/(G*Γ_hat) * a + (Y_R+transfer),eps_shk,z_hat)
                                utility = (G*Γ_hat)**(1-ρ)*u_prime(c_hat)
                                Ez += LivProb*utility * P[z, z_hat]
                Ez = Ez / (len(psi_shk_draws)*len(eps_shk_draws_sj))
                ## the last step depends on if adjustment is fully flexible
                if adjust_prob ==1.0:
                    σ_out[i, j, z] =  u_prime_inv(β * R* Ez)
                elif adjust_prob <=1.0:
                    σ_out[i, j, z] =  adjust_prob/(1-β*R*(1-adjust_prob))*u_prime_inv(β * R* Ez)

    # Calculate endogenous asset grid
    mϵ_out = np.empty_like(σ_out)
            
    for j,ϵ in enumerate(eps_grid):
        for z in range(n):
            mϵ_out[:,j,z] = a_grid + σ_out[:,j,z]

    # Fixing a consumption-asset pair at (0, 0) improves interpolation
    for j,ϵ in enumerate(eps_grid):
        for z in range(n):
            if borrowing_cstr==True:  ## either hard constraint is zero or positive probability of losing job
                σ_out[0,j,z] = 0.0
                mϵ_out[0,j,z] = 0.0
            #elif borrowing_cstr==True or ue_markov==True:
            #    σ_out[0,j,z] = 0.0
            #    mϵ_out[0,j,z] = min(0.0,-unemp_insurance/R)
            else:
                if age <=T-1:
                    σ_out[0,j,z] = 0.0
                    self_min_a = - np.exp(np.min(eps_shk_draws_sj))*G/R
                    self_min_a = min(self_min_a,-unemp_insurance/R)
                    mϵ_out[0,j,z] = self_min_a
                else:
                    σ_out[0,j,z] = 0.0
                    self_min_a = - pension*G/R
                    mϵ_out[0,j,z] = self_min_a

    return mϵ_out, σ_out


# + code_folding=[1]
## for life-cycle/finite horizon problem 
def solve_model_backward_iter(model,        # Class with model information
                              mϵ_vec,        # Initial condition for assets and MA shocks
                              σ_vec,        # Initial condition for consumption
                              br = False,
                             sv = False):

    ## memories for life-cycle solutions 
    n_grids1 = σ_vec.shape[0]
    n_grids2 = σ_vec.shape[1]
    n_z = len(model.P)                       
    mϵs_new =  np.empty((model.L,n_grids1,n_grids2,n_z),dtype = np.float64)
    σs_new =  np.empty((model.L,n_grids1,n_grids2,n_z),dtype = np.float64)
    
    mϵs_new[0,:,:,:] = mϵ_vec
    σs_new[0,:,:,:] = σ_vec
    
    for year2L in range(1,model.L): ## nb of years till L from 0 to Model.L-2
        age = model.L-year2L
        age_id = age-1
        print("at work age of "+str(age))
        mϵ_vec_next, σ_vec_next = mϵs_new[year2L-1,:,:,:],σs_new[year2L-1,:,:,:]
        if br==False:
            if sv ==False:
                #print('objective model without stochastic risk')
                mϵ_new, σ_new =EGM(mϵ_vec_next, σ_vec_next, age_id, model)
            else:
                #print('objective model with stochastic risk')
                mϵ_new, σ_new = EGM_sv(mϵ_vec_next, σ_vec_next, age_id, model)
        elif br==True:
            #print('subjective model with stochastic risk')
            mϵ_new, σ_new = EGM_br(mϵ_vec_next, σ_vec_next, age_id, model)
        mϵs_new[year2L,:,:,:] = mϵ_new
        σs_new[year2L,:,:,:] = σ_new

    return mϵs_new, σs_new


# + code_folding=[1]
## for infinite horizon problem 
def solve_model_iter(model,        # Class with model information
                     me_vec,        # Initial condition for assets and MA shocks
                     σ_vec,        # Initial condition for consumption
                      tol=1e-6,
                      max_iter=2000,
                      verbose=True,
                      print_skip=50,
                      br = False,
                      sv = False):

    # Set up loop
    i = 0
    error = tol + 1

    ## memories for life-cycle solutions 
    n_grids1 = σ_vec.shape[0]
    n_grids2 =σ_vec.shape[1]
    n_z = len(model.P)                       
    
    while i < max_iter and error > tol:
        me_new, σ_new = EGM(me_vec, σ_vec, 0,model)
        error = np.max(np.abs(σ_vec - σ_new))
        i += 1
        if verbose and i % print_skip == 0:
            print(f"Error at iteration {i} is {error}.")
        me_vec, σ_vec = np.copy(me_new), np.copy(σ_new)

    if i == max_iter:
        print("Failed to converge!")

    if verbose and i < max_iter:
        print(f"\nConverged in {i} iterations.")

    return me_vec, σ_vec


# -

# ## Initialize the model

# + code_folding=[0]
if __name__ == "__main__":


    ## parameters 
    ###################

    U = 0.2 ## transitory ue risk
    U0 = 0.0 ## transitory ue risk
    unemp_insurance = 0.15
    sigma_psi = 0.1 # permanent 
    sigma_eps = 0.0 # transitory 


    #λ = 0.0  ## tax rate
    #λ_SS = 0.0 ## social tax rate
    #transfer = 0.0  ## transfer 
    #pension = 1.0 ## pension


    ## life cycle 

    T = 40
    L = 60
    TGPos = int(L/3) ## one third of life sees income growth 
    GPos = 1.01*np.ones(TGPos)
    GNeg= 0.99*np.ones(L-TGPos)
    #G = np.concatenate([GPos,GNeg])
    G = np.ones(L)
    YPath = np.cumprod(G)


    ## other parameters 

    ρ = 2
    R = 1.01
    β = 0.97
    x = 0.0
    theta = 0.0 ## extrapolation parameter 

    ## no persistent state
    b_y = 0.0

    ## wether to have zero borrowing constraint 
    borrowing_cstr = True
# -

## a deterministic income profile 
if __name__ == "__main__":

    plt.title('Deterministic Life-cycle Income Profile \n')
    plt.plot(YPath,'ro')
    plt.xlabel('Age')
    plt.ylabel(r'$\hat Y$')

# ## Life-Cycle Problem 

# ### Consumption  the last period 

# + code_folding=[3]
"""
## this is the retirement consumption policy 

def policyPF(β,
             ρ,
             R,
             T,
             L):
    c_growth = β**(1/ρ)*R**(1/ρ-1)
    return (1-c_growth)/(1-c_growth**(L-T))
    
"""

# + code_folding=[]
if __name__ == "__main__":
    lc = LifeCycle(sigma_psi = sigma_psi,
                   sigma_eps = sigma_eps,
                   U=U,
                   ρ=ρ,
                   R=R,
                   T=T,
                   L=L,
                   G=G,
                   β=β,
                   x=x,
                   borrowing_cstr = borrowing_cstr,
                   b_y=b_y,
                   unemp_insurance = unemp_insurance,
                   )


# + code_folding=[]
# Initial the end-of-period consumption policy of σ = consume all assets

if __name__ == "__main__":

    ## initial consumption functions 

    k = len(lc.a_grid)
    k2 =len(lc.eps_grid)

    n = len(lc.P)
    σ_init = np.empty((k,k2,n))
    m_init = np.empty((k,k2,n))

    for z in range(n):
        for j in range(k2):
            m_init[:,j,z] = lc.a_grid
            σ_init[:,j,z] = m_init[:,j,z]
# -

if __name__ == "__main__":

    plt.title('Consumption in the last period')
    plt.plot(m_init[:,0,1],
             σ_init[:,0,1])

# ### Without MA idiosyncratic income shock 

# + code_folding=[]
if __name__ == "__main__":


    ## solve the model for a range of ma(1) coefficients
    ### x!=0, adds the transitory shock an additional state variable 

    t_start = time()

    sigma_psi_ls = [0.03,0.3]
    ms_stars =[]
    σs_stars = []
    for i,sigma_psi in enumerate(sigma_psi_ls):
        lc.sigma_psi = sigma_psi
        as_star, σs_star = solve_model_backward_iter(lc,
                                                     m_init,
                                                     σ_init)
        ms_stars.append(as_star)
        σs_stars.append(σs_star)



    t_finish = time()

    print("Time taken, in seconds: "+ str(t_finish - t_start))

# + code_folding=[0]
if __name__ == "__main__":


    ## plot c func at different age /asset grid
    years_left = [0,1,2,3]

    n_sub = len(years_left)

    eps_fix = 0 ## the first eps grid 

    ms_star = ms_stars[0]
    σs_star = σs_stars[0]

    fig,axes = plt.subplots(1,n_sub,figsize=(4*n_sub,4))

    for x,year in enumerate(years_left):
        age = lc.L-year
        i = lc.L-age
        for k,sigma_psi in enumerate(sigma_psi_ls):
            m_plt,c_plt = ms_stars[k][i,:,eps_fix,0],σs_stars[k][i,:,eps_fix,0]
            axes[x].plot(m_plt,
                         c_plt,
                         label = r'$\sigma_\psi=$'+str(sigma_psi),
                         lw=3,
                        )
        axes[x].legend()
        axes[x].set_xlim(0.0,np.max(m_plt))
        axes[x].set_xlabel('asset')
        axes[0].set_ylabel('c')
        axes[x].set_title(r'$age={}$'.format(age))

# + code_folding=[]
## interpolate consumption function on continuous s/eps grid 

#ms_list = []

#for i in range(lc.L):
#    this_σ= policyfuncMA(lc,
#                         ms_star[i,:,:,0],
#                         σs_star[i,:,:,0])
#    ms_list.append(this_σ)


# + code_folding=[]
"""
## plot contour for policy function 

m_grid = np.linspace(0.00001,5,20)
eps_grid = lc.eps_grid
mm,epss = np.meshgrid(m_grid,
                      eps_grid)

σ_this = σs_list[3]
c_stars = σ_this(m_grid,
                 eps_grid)

cp = plt.contourf(mm,epss,
                  c_stars)
plt.title(r'$c$')
plt.xlabel('wealth')
plt.ylabel('ma income shock')
"""
# -

## the size of consumption function is  T x nb_a x nb_eps x nb_z 
if __name__ == "__main__":
    print(σs_star.shape)

# + code_folding=[]
"""
## plot 3d consumption function 
#age,asset,inc_shk =σs_star[:,:,:,0]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mm, epss, c_stars, zdir='z', c= 'red')
ax.set_xlabel('wealth')
ax.set_ylabel('inc shock')
ax.set_title('consumption at a certain age')
"""


# + code_folding=[]
if __name__ == "__main__":

    ## plot 3d functions over life cycle 

    ages = np.array(range(ms_star.shape[0]))
    asset = ms_star[0,:,0,0]
    xx, yy = np.meshgrid(ages, asset)
    c_stars = σs_star[:,:,0,0].T

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, 
                         projection='3d')
    dem3d = ax.plot_surface(xx,
                            yy,
                            c_stars,
                            rstride=1, 
                            cstride=1,
                            cmap='viridis', 
                            edgecolor='none'
                           )
    ax.set_title('consumption over the life cycle')
    ax.set_xlabel('years left')
    ax.set_ylabel('wealth')
    ax.view_init(30, 30)
# -

# ### Different ma persistence
#
# - could be either individual unemployment state or macroeconomic state
#

# + code_folding=[]
if __name__ == "__main__":
    at_age = 4
    at_asset_id = 20

    for i,x in enumerate(x_ls):
        this_σs_star = σs_stars[i]
        plt.plot(lc.eps_grid,
                 this_σs_star[lc.L-at_age,
                              at_asset_id,:,0],
                 '-.',
                 label = r'$x={}$'.format(x),
                 lw=3)
    plt.legend(loc=0)
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'$c(m,\epsilon,age)$')
    plt.title(r'work age$={}$'.format(at_age))
# -

#
#

# ### With a Markov/persistent state: good versus bad 

# + code_folding=[4]
if __name__ == "__main__":
    ## initialize another 
    lc_ar = LifeCycle(sigma_psi=sigma_psi,
                     sigma_eps = sigma_eps,
                     U=U0,
                     ρ=ρ,
                     R=R,
                     T=T,
                     L=L,
                     G=G,
                     β=β,
                     x=0.0,  ## shut down ma(1)
                     borrowing_cstr = borrowing_cstr,
                     b_y=0.5)


# + code_folding=[]
if __name__ == "__main__":


    ## solve the model for different persistence 
    t_start = time()


    ar_ls = [0.99]
    ms_stars_ar=[]
    σs_stars_ar = []

    for i, ar in enumerate(ar_ls):

        ## tauchenize an ar1
        #σ = 0.18
        #constant = 0.00

        #mc = qe.markov.approximation.tauchen(ar, σ, b=constant, m=3, n=7)
        #z_ss_av = constant/(1-ar)
        #z_ss_sd = σ*np.sqrt(1/(1-ar**2))

        ## feed the model with a markov matrix of macro state 
        #lc_ar.z_val, lc_ar.P = mc.state_values, mc.P
        P = np.array([(0.8, 0.2),
                  (0.05, 0.95)])
        lc_ar.P = P

        ## initial guess
        k = len(lc_ar.a_grid)
        k2 =len(lc_ar.eps_grid)
        n = len(lc_ar.P)

        σ_init_ar = np.empty((k,k2,n))
        m_init_ar = np.empty((k,k2,n))

        for z in range(n):
            for j in range(k2):
                σ_init_ar[:,j,z] = 2*lc_ar.a_grid
                m_init_ar[:,j,z] = 2*lc_ar.a_grid

        ## solve the model 
        ms_star_ar, σs_star_ar = solve_model_backward_iter(lc_ar,
                                                         m_init_ar,
                                                         σ_init_ar)
        ms_stars_ar.append(ms_star_ar)
        σs_stars_ar.append(σs_star_ar)


    t_finish = time()

    print("Time taken, in seconds: "+ str(t_finish - t_start))

# + code_folding=[]
if __name__ == "__main__":


    ## compare two markov states good versus bad 

    years_left = [1,2,20,25]

    n_sub = len(years_left)


    eps_ls = [0]

    fig,axes = plt.subplots(1,n_sub,figsize=(4*n_sub,4))

    for x,year in enumerate(years_left):
        age = lc_ar.L-year
        i = lc_ar.L-age
        for eps in eps_ls:
            m_plt_l,c_plt_l = ms_stars_ar[0][i,:,eps,0],σs_stars_ar[0][i,:,eps,0]
            m_plt_h,c_plt_h  = ms_stars_ar[0][i,:,eps,1],σs_stars_ar[0][i,:,eps,1]
            axes[x].plot(m_plt_l,
                         c_plt_l,
                         '--',
                         label ='bad',
                         lw=3)
            axes[x].plot(m_plt_h,
                         c_plt_h,
                         '-.',
                         label ='good',
                         lw=3)
        axes[x].legend()
        axes[x].set_xlim((0.0,np.max(m_plt_h)))
        axes[x].set_xlabel('m')
        axes[0].set_ylabel('c')
        axes[x].set_title(r'c at $age={}$'.format(age))
# -

# ### State-dependent risks 

# + code_folding=[]
if __name__ == "__main__":

    ## transition matrix between low and high risk state

    P = np.array([(0.5, 0.5),
                  (0.5, 0.5)])   # markov transition matricies 

    ss_P = cal_ss_2markov(P)
    prob_l = P[0,0]
    prob_h = P[0,1]

    ## keep average risks the same 
    sigma_psi_l = 0.01*sigma_psi
    sigma_psi_h = np.sqrt((sigma_psi**2 - prob_l*sigma_psi_l**2)/prob_h)

    sigma_eps_l = 0.1*sigma_eps
    sigma_eps_h = np.sqrt((sigma_eps**2 - prob_l*sigma_eps_l**2)/prob_h)
    sigma_psi_2mkv = np.array([sigma_psi_l,sigma_psi_h]) 
    sigma_eps_2mkv = np.array([sigma_eps_l,sigma_eps_h]) 


    b_y = 0.0  ## set the macro state loading to be zero, i.e. only risks differ across two states
# -


if __name__ == "__main__":
    ## compute steady state 
    av_sigma_psi = np.sqrt(np.dot(P[0,:],sigma_psi_2mkv**2))
    av_sigma_eps = np.sqrt(np.dot(P[0,:],sigma_eps_2mkv**2))
    print('steady state is '+str(ss_P))
    print('transitory probability is '+str(P[0,:]))
    print('average permanent risk is '+str(av_sigma_psi)+' compared to objective model '+str(sigma_psi))
    print('average transitory risk is '+str(av_sigma_eps)+' compared to objective model '+str(sigma_eps))

if __name__ == "__main__":

    print('permanent risk state is '+str(sigma_psi_2mkv))
    print('transitory risk state is '+str(sigma_eps_2mkv))

# + code_folding=[]
if __name__ == "__main__":

    ## another model instance 

    lc_sv = LifeCycle(sigma_psi = sigma_psi,
                   sigma_eps = sigma_eps,
                   U=U,
                   ρ=ρ,
                   R=R,
                   T=T,
                   L=L,
                   G=G,
                   β=β,
                   x=x,
                   sigma_psi_2mkv = sigma_psi_2mkv,
                   sigma_eps_2mkv = sigma_eps_2mkv,
                   borrowing_cstr = borrowing_cstr,
                   b_y=b_y)


# + code_folding=[]
if __name__ == "__main__":

    ## solve the model for different transition matricies 

    t_start = time()

    P_ls = [P]
    ms_stars_sv=[]
    σs_stars_sv = []

    for i, P in enumerate(P_ls):

        ## feed the model with a markov matrix of macro state 
        lc_sv.P = P

        ## initial guess
        k = len(lc_sv.a_grid)
        k2 =len(lc_sv.eps_grid)
        n = len(lc_sv.P)

        σ_init_sv = np.empty((k,k2,n))
        m_init_sv = np.empty((k,k2,n))

        for z in range(n):
            for j in range(k2):
                σ_init_sv[:,j,z] = 2*lc_sv.a_grid
                m_init_sv[:,j,z] = 2*lc_sv.a_grid

        ## solve the model 
        ms_star_sv, σs_star_sv = solve_model_backward_iter(lc_sv,
                                                           m_init_sv,
                                                           σ_init_sv,
                                                           sv=True)
        ms_stars_sv.append(ms_star_sv)
        σs_stars_sv.append(σs_star_sv)

    t_finish = time()

    print("Time taken, in seconds: "+ str(t_finish - t_start))

# + code_folding=[]
if __name__ == "__main__":
    ## compare two markov states low versus high risk 

    years_left = [1,5,17,28]

    n_sub = len(years_left)

    eps_id = 0

    fig,axes = plt.subplots(1,n_sub,figsize=(4*n_sub,4))

    for x,year in enumerate(years_left):
        age = lc.L-year
        i = lc.L-age
        m_plt_l,c_plt_l = ms_stars_sv[0][i,:,eps_id,0],σs_stars_sv[0][i,:,eps_id,0]
        m_plt_h,c_plt_h = ms_stars_sv[0][i,:,eps_id,1],σs_stars_sv[0][i,:,eps_id,1]
        
        axes[x].plot(m_plt_l, ## 0 indicates the low risk state 
                     c_plt_l,
                     '--',
                     label ='low risk',
                     lw=3)
        
        axes[x].plot(m_plt_h, ## 1 indicates the high risk state 
                     c_plt_h,
                     '-.',
                     label ='high risk',
                     lw=3)
        axes[x].legend()
        axes[x].set_xlim((0.0,np.max(m_plt_h)))

        axes[x].set_xlabel('m')
        axes[0].set_ylabel('c')
        axes[x].set_title(r'c under SV at $age={}$'.format(age))
# -

# ### Comparison: objective and subjective risk perceptions

# + code_folding=[]
if __name__ == "__main__":


    ## compare subjective and objective models 
    years_left = [1,5,17,25]


    n_sub = len(years_left)

    eps_ls = [0]

    fig,axes = plt.subplots(1,n_sub,figsize=(6*n_sub,6))

    for x,year in enumerate(years_left):
        age = lc.L-year
        i = lc.L-age
        for eps in eps_ls:
            ## baseline: no ma shock 
            m_plt,c_plt = ms_star[i,:,eps,0],σs_star[i,:,eps,0]
            axes[x].plot(m_plt,
                         c_plt,
                         label = 'objective',
                         lw=3)
            ## persistent 
            #axes[x].plot(ms_stars_ar[0][i,:,eps,0],
            #             σs_stars_ar[0][i,:,eps,0],
            #             '--',
            #             label ='bad',
            #             lw=3)
            #axes[x].plot(ms_stars_ar[0][i,:,eps,1],
            #             σs_stars_ar[0][i,:,eps,1],
            #             '-.',
            #             label ='good',
            #             lw=3)
             ## stochastic volatility 
            m_plt_l,c_plt_l = ms_stars_sv[0][i,:,eps,0],σs_stars_sv[0][i,:,eps,0]
            m_plt_h,c_plt_h = ms_stars_sv[0][i,:,eps,1],σs_stars_sv[0][i,:,eps,1]
            
            axes[x].plot(m_plt_l, ## 0 indicates the low risk state 
                         c_plt_l,
                         '--',
                         label ='subjective: low risk',
                         lw=3)
            axes[x].plot(m_plt_h, ## 1 indicates the high risk state 
                         c_plt_h,
                         '-.',
                         label ='subjective: high risk',
                         lw=3)
            ## countercyclical 
            #axes[x].plot(ms_stars_cr[0][i,:,eps,0], ## 0 indicates the low risk state 
            #         σs_stars_cr[0][i,:,eps,0],
            #         '--',
            #         label ='sv: unemployed + high risk',
            #         lw=3)
            #axes[x].plot(ms_stars_cr[0][i,:,eps,1], ## 1 indicates the high risk state 
            #             σs_stars_cr[0][i,:,eps,1],
            #             '-.',
            #             label ='sv:employed + low risk',
            #             lw=3)
            ## subjective 
            #axes[x].plot(ms_br[i,:,eps,0],
            #             σs_br[i,:,eps,0],
            #             '*-',
            #             label = 'subjective:'+str(round(lc.eps_grid[eps],2)),
            #             lw=3)
            #axes[x].plot(ms_star[i,:,eps,0],
            #             σs_star[i,:,eps,0],
            #             '--',
            #             label ='objective:'+str(round(lc.eps_grid[eps],2)),
            #             lw=3)

        axes[0].legend()
        axes[x].set_xlim((0.0,np.max(m_plt)))
        axes[x].set_xlabel('m')
        axes[0].set_ylabel('c')
        axes[x].set_title(r'working age$={}$'.format(age))

    #plt.suptitle('Various Straight Lines',fontsize=20)

    fig.savefig('../Graphs/model/comparison1.png')
# -

# ### With a Markov/persistent unemployment state

# + code_folding=[]
if __name__ == "__main__":


    ## transition matrix between emp and uemp

    ## transition probability during normal times of the economy 

    P_uemkv = np.array([(0.2, 0.8),
                        (0.2, 0.8)])   # markov transition matricies 

    #P_uemkv = np.array([(0.4, 0.6),
    #                    (0.05, 0.95)])   # markov transition matricies 


# + code_folding=[0]
if __name__ == "__main__":


    ## initialize another 
    lc_uemkv = LifeCycle(sigma_psi=sigma_psi,
                         sigma_eps = sigma_eps,
                         U=U0,
                         ρ=ρ,
                         R=R,
                         T=T,
                         L=L,
                         G=G,
                         β=β,
                         x=0.0,  ## shut down ma(1)
                         borrowing_cstr = borrowing_cstr,
                         b_y = 0.0, ## markov state loading does not matter any more 
                         unemp_insurance = 0.3,
                         ue_markov = True)

# + code_folding=[0]
if __name__ == "__main__":

    ## solve the model for different transition matricies of UE markov
    t_start = time()

    P_ls = [P_uemkv]
    ms_stars_uemkv=[]
    σs_stars_uemkv = []

    for i, P in enumerate(P_ls):

        ## feed the model with a markov matrix of macro state 
        lc_uemkv.P = P

        ## initial guess
        k = len(lc_uemkv.a_grid)
        k2 =len(lc_uemkv.eps_grid)
        n = len(lc_uemkv.P)

        σ_init_uemkv = np.empty((k,k2,n))
        m_init_uemkv = np.empty((k,k2,n))

        for z in range(n):
            for j in range(k2):
                m_init_uemkv[:,j,z] = 2*lc_uemkv.a_grid
                σ_init_uemkv[:,j,z] = m_init_uemkv[:,j,z]


        ## solve the model 
        ms_star_uemkv, σs_star_uemkv = solve_model_backward_iter(lc_uemkv,
                                                                 m_init_uemkv,
                                                                 σ_init_uemkv,
                                                                 sv = False)
        ms_stars_uemkv.append(ms_star_uemkv)
        σs_stars_uemkv.append(σs_star_uemkv)

    t_finish = time()

    print("Time taken, in seconds: "+ str(t_finish - t_start))

# + code_folding=[]
if __name__ == "__main__":


    ## compare two markov states of emp and uemp 

    years_left = [1,2,20,25]

    n_sub = len(years_left)

    eps_id = 0

    fig,axes = plt.subplots(1,n_sub,figsize=(4*n_sub,4))

    for x,year in enumerate(years_left):
        age = lc_uemkv.L-year
        i = lc_uemkv.L-age
        m_plt_u, c_plt_u = ms_stars_uemkv[0][i,:,eps_id,0],σs_stars_uemkv[0][i,:,eps_id,0]
        m_plt_e, c_plt_e = ms_stars_uemkv[0][i,:,eps_id,1],σs_stars_uemkv[0][i,:,eps_id,1]

        axes[x].plot(m_plt_u, ## 0 indicates the low risk state 
                     c_plt_u,
                     '--',
                     label ='unemployed',
                     lw=3)
        axes[x].plot(m_plt_e, ## 1 indicates the high risk state 
                     c_plt_e,
                     '-.',
                     label ='employed',
                     lw=3)
        axes[x].legend()
        axes[x].set_xlim((0.0,np.max(m_plt_e)))
        axes[x].set_xlabel('asset')
        axes[0].set_ylabel('c')
        axes[x].set_title(r'c under UE Markov at $age={}$'.format(age))
# -

# ### "Countercylical" risks
#
# - unemployed perceive higher risks
#

# + code_folding=[]
if __name__ == "__main__":


    ss_P_cr= cal_ss_2markov(P_uemkv)
    prob_h_cr = P_uemkv[0,0]
    prob_l_cr = P_uemkv[0,1]

    ## keep average risks the same 
    sigma_psi_l_cr = 0.1*sigma_psi
    sigma_psi_h_cr = np.sqrt((sigma_psi**2 - prob_l_cr*sigma_psi_l_cr**2)/prob_h_cr)

    sigma_eps_l_cr = 0.1*sigma_eps
    sigma_eps_h_cr = np.sqrt((sigma_eps**2 - prob_l_cr*sigma_eps_l_cr**2)/prob_h_cr)

    ### notice here I put high risk at the first!!!
    sigma_psi_2mkv_cr= np.array([sigma_psi_h_cr,sigma_psi_l_cr**2])
    sigma_eps_2mkv_cr = np.array([sigma_eps_h_cr,sigma_eps_l_cr**2])

    ## again, zero loading from z
    b_y = 0.0

# +

if __name__ == "__main__":
    ## compute steady state 
    av_sigma_psi_cr = np.sqrt(np.dot(P_uemkv[0,:],sigma_psi_2mkv_cr**2))
    av_sigma_eps_cr = np.sqrt(np.dot(P_uemkv[0,:],sigma_eps_2mkv_cr**2))
    print('steady state is '+str(ss_P_cr))
    print('transitory probability is '+str(P_uemkv[0,:]))

    print('average permanent risk is '+str(av_sigma_psi_cr)+' compared to objective model '+str(sigma_psi))
    print('average transitory risk is '+str(av_sigma_eps_cr)+' compared to objective model '+str(sigma_eps))

# + code_folding=[]
if __name__ == "__main__":


    ## model instance 
    lc_cr= LifeCycle(sigma_psi = sigma_psi,
                     sigma_eps = sigma_eps,
                     U=U0,
                     ρ=ρ,
                     P=P, 
                     R=R,
                     T=T,
                     L=L,
                     G=G,
                     β=β,
                     sigma_psi_2mkv = sigma_psi_2mkv_cr,   # different 
                     sigma_eps_2mkv = sigma_eps_2mkv_cr,  # different 
                     shock_draw_size = 30,
                     borrowing_cstr = borrowing_cstr,
                     x = x,  ## shut down ma(1)
                     b_y = b_y,
                     ue_markov = True)

# + code_folding=[]
if __name__ == "__main__":


    ## solve the model for different transition matricies 

    t_start = time()

    P_ls = [P_uemkv]
    ms_stars_cr=[]
    σs_stars_cr = []

    for i, P in enumerate(P_ls):

        ## feed the model with a markov matrix of macro state 
        lc_cr.P = P

        ## initial guess
        k = len(lc_cr.a_grid)
        k2 =len(lc_cr.eps_grid)
        n = len(lc_cr.P)

        σ_init_cr = np.empty((k,k2,n))
        m_init_cr = np.empty((k,k2,n))

        for z in range(n):
            for j in range(k2):
                σ_init_cr[:,j,z] = 2*lc_cr.a_grid
                m_init_cr[:,j,z] = 2*lc_cr.a_grid

        ## solve the model 
        ms_star_cr, σs_star_cr = solve_model_backward_iter(lc_cr,
                                                           m_init_cr,
                                                           σ_init_cr,
                                                           sv= True)
        ms_stars_cr.append(ms_star_cr)
        σs_stars_cr.append(σs_star_cr)

    t_finish = time()

    print("Time taken, in seconds: "+ str(t_finish - t_start))

# + code_folding=[0, 13]
if __name__ == "__main__":


    ## compare two markov states low versus high risk 

    years_left = [1,5,20,25]

    n_sub = len(years_left)

    eps_id = 0

    fig,axes = plt.subplots(1,n_sub,figsize=(4*n_sub,4))

    for x,year in enumerate(years_left):
        age = lc.L-year
        i = lc.L-age
        m_plt_l,c_plt_l = ms_stars_cr[0][i,:,eps_id,0],σs_stars_cr[0][i,:,eps_id,0]
        m_plt_h,c_plt_h = ms_stars_cr[0][i,:,eps_id,1],σs_stars_cr[0][i,:,eps_id,1]
        
        axes[x].plot(m_plt_l, ## 0 indicates the low risk state 
                     c_plt_l,
                     '--',
                     label ='une+ high risk',
                     lw=3)
        axes[x].plot(m_plt_h, ## 1 indicates the high risk state 
                     c_plt_h,
                     '-.',
                     label ='emp+ low risk',
                     lw=3)
        axes[x].legend()
        axes[x].set_xlabel('m')
        axes[0].set_ylabel('c')
        axes[x].set_title(r'c under countercyclical risks at $age={}$'.format(age))
# -

# ### Objective and subject state-dependent profile

# + code_folding=[0, 13]
if __name__ == "__main__":

    ## compare subjective and objective models 

    years_left = [1,5,10]


    n_sub = len(years_left)

    eps_ls = [0]

    fig,axes = plt.subplots(1,n_sub,figsize=(6*n_sub,6))

    for x,year in enumerate(years_left):
        age = lc.L-year
        i = lc.L-age
        for eps in eps_ls:
            ## baseline: no ma shock 
            #axes[x].plot(ms_star[i,:,eps,0],
            #             σs_star[i,:,eps,0],
            #             label = 'objective',
            #             lw=3)
            ## persistent 
            m_plt_u, c_plt_u = ms_stars_uemkv[0][i,:,eps,0],σs_stars_uemkv[0][i,:,eps,0]
            m_plt_e, c_plt_e = ms_stars_uemkv[0][i,:,eps,1],σs_stars_uemkv[0][i,:,eps,1]
            axes[x].plot(m_plt_u,
                         c_plt_u,
                         '--',
                         label ='unemployed',
                         lw=3)
            axes[x].plot(m_plt_e,
                         c_plt_e,
                         '-.',
                         label ='employed',
                         lw=3)
             ## stochastic volatility 
            #axes[x].plot(ms_stars_sv[0][i,:,eps,0], ## 0 indicates the low risk state 
            #             σs_stars_sv[0][i,:,eps,0],
            #             '--',
            #             label ='sv:low risk',
            #             lw=3)
            #axes[x].plot(ms_stars_sv[0][i,:,eps,1], ## 1 indicates the high risk state 
            #             σs_stars_sv[0][i,:,eps,1],
            #             '-.',
            #             label ='sv:high risk',
            #             lw=3)
            ## countercyclical 
            m_plt_l,c_plt_l = ms_stars_cr[0][i,:,eps,0],σs_stars_cr[0][i,:,eps,0]
            m_plt_h,c_plt_h = ms_stars_cr[0][i,:,eps,1],σs_stars_cr[0][i,:,eps,1]
            axes[x].plot(m_plt_l, ## 0 indicates the low risk state 
                     c_plt_l,
                     '--',
                     label ='ue + high risk',
                     lw=3)
            axes[x].plot(m_plt_h, ## 1 indicates the high risk state 
                         c_plt_h,
                         '-.',
                         label ='emp + low risk',
                         lw=3)
            # subjective 
            #axes[x].plot(as_br[i,:,eps,0],
            #             σs_br[i,:,eps,0],
            #             '*-',
            #             label = 'subjective:'+str(round(lc.eps_grid[eps],2)),
            #             lw=3)
            #axes[x].plot(as_star[i,:,eps,0],
            #             σs_star[i,:,eps,0],
            #             '--',
            #             label ='objective:'+str(round(lc.eps_grid[eps],2)),
            #             lw=3)

        axes[0].legend()
        axes[x].set_xlim((0.0,np.max(m_plt_e)))
        axes[x].set_xlabel('m')
        axes[0].set_ylabel('c')
        axes[x].set_title(r'working age$={}$'.format(age))

    #plt.suptitle('Various Straight Lines',fontsize=20)

    fig.savefig('../Graphs/model/comparison2.png')


# -

# ### Subjective perceptions 

# + code_folding=[0]
if __name__ == "__main__":


    ## solve for subjective agent 
    ## agents extrapolate recent tarnsitory volatility to perceptions 

    t_start = time()


    ms_br, σs_br = solve_model_backward_iter(lc,
                                             m_init,
                                             σ_init,
                                             br = True) ## bounded rationality is true 



    t_finish = time()

    print("Time taken, in seconds: "+ str(t_finish - t_start))

# + code_folding=[]
if __name__ == "__main__":


    ## compare subjective and objective model 
    years_left = [1,3,5,6]
    n_sub = len(years_left)

    eps_ls = [0]

    fig,axes = plt.subplots(1,n_sub,figsize=(4*n_sub,4))

    for x,year in enumerate(years_left):
        age = lc.L-year
        i = lc.L-age
        for eps in eps_ls:
            axes[x].plot(ms_br[i,:,eps,0],
                         σs_br[i,:,eps,0],
                         '*-',
                         label = 'subjective:'+str(round(lc.eps_grid[eps],2)),
                         lw=3)
            axes[x].plot(ms_star[i,:,eps,0],
                         σs_star[i,:,eps,0],
                         '--',
                         label ='objective:'+str(round(lc.eps_grid[eps],2)),
                         lw=3)
        axes[x].legend()
        axes[x].set_xlabel('asset')
        axes[0].set_ylabel('c')
        axes[x].set_title(r'subjective c at $age={}$'.format(age))

# + code_folding=[]
if __name__ == "__main__":


    x_sj = extrapolate(5, 
                       lc.x,
                       lc.eps_grid) ## sj: subjective 

    plt.plot(lc.eps_grid,x_sj)

# + code_folding=[]
if __name__ == "__main__":


    at_age = 10
    at_asset_id = 15

    plt.plot(lc.eps_grid,
             σs_br[lc.T-at_age,at_asset_id,:,0],
                 'v-',
                 label = 'subjective',
                 lw=3)
    plt.plot(lc.eps_grid,
             σs_star[lc.T-at_age,at_asset_id,:,0],
             '--',
             label='objective',
             lw=3)
    plt.legend(loc=0)
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'$c(a,\epsilon,age)$')
    plt.title(r'subjectiv c at work age$={}$'.format(at_age))
# -
# ## Infinite horizon problem

# + code_folding=[]
if __name__ == "__main__":


    ## intialize a model instance

    inf_liv = LifeCycle(sigma_psi=sigma_psi,
                       sigma_eps = sigma_eps,
                       U=U,
                       ρ=ρ,
                       R=R,
                       T=T,
                       L=L,
                       β=β,
                       x=x,
                       theta=theta,
                       borrowing_cstr = borrowing_cstr,
                       b_y=b_y)


    ## initial consumption functions 

    k = len(inf_liv.a_grid)
    k2 =len(inf_liv.eps_grid)

    n = len(inf_liv.P)
    σ_init = np.empty((k,k2,n))
    m_init = np.empty((k,k2,n))

    for z in range(n):
        for j in range(k2):
            m_init[:,j,z] = inf_liv.a_grid
            σ_init[:,j,z] = 0.1*m_init[:,j,z]

    t_start = time()


    x_ls = [0.0]
    ms_inf_stars =[]
    σs_inf_stars = []
    for i,x in enumerate(x_ls):

        ## set different ma parameters 
        inf_liv.x = x
        m_inf_star, σ_inf_star = solve_model_iter(inf_liv,
                                                  m_init,
                                                  σ_init)
        ms_inf_stars.append(m_inf_star)
        σs_inf_stars.append(σ_inf_star)


    t_finish = time()

    print("Time taken, in seconds: "+ str(t_finish - t_start))   



    ## plot c func 

    eps_ls = [0,1]

    ms_inf_star = ms_inf_stars[0]
    σs_inf_star = σs_inf_stars[0]


    for eps in eps_ls:
        plt.plot(ms_inf_star[:,eps,0],
                 σs_inf_star[:,eps,0],
                 label = r'$\epsilon=$'+str(round(inf_liv.eps_grid[eps],2)),
                 lw=3
                )
        plt.legend()
        plt.xlabel('asset')
        plt.ylabel('c')
        plt.title('Inifite horizon solution')

# -

# ## Infinite horizon with adjustment inertia
#
#

# + code_folding=[5]
if __name__ == "__main__":


    ## intialize a model instance

    imp_adjust = LifeCycle(sigma_psi=sigma_psi,
                       sigma_eps = sigma_eps,
                       U=U,
                       ρ=ρ,
                       R=R,
                       T=T,
                       L=L,
                       β=β,
                       x=x,
                       theta=theta,
                       borrowing_cstr = borrowing_cstr,
                       b_y=b_y,
                       unemp_insurance = unemp_insurance,
                       adjust_prob = 0.6)

    ## initial consumption functions 

    k = len(imp_adjust.a_grid)
    k2 =len(imp_adjust.eps_grid)

    n = len(imp_adjust.P)
    σ_init = np.empty((k,k2,n))
    m_init = np.empty((k,k2,n))

    for z in range(n):
        for j in range(k2):
            m_init[:,j,z] = imp_adjust.a_grid
            σ_init[:,j,z] = 0.1*m_init[:,j,z]  ## c !=m because of infinite horizon

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
        plt.plot(ms_imp_star[:,eps,1],
                 σs_imp_star[:,eps,1],
                 '-',
                 label = 'imperfect adjustment',
                 lw=3
                )
        plt.plot(ms_inf_star[:,eps,1],
                 σs_inf_star[:,eps,1],
                 '--',
                 label = 'perfect adjustment',
                 lw=3
                )
        plt.legend()
        plt.xlabel('asset')
        plt.ylabel('c')
        plt.title('Inifite horizon solution')
# -
# # 

