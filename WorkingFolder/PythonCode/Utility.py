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

import numba as nb
import numpy as np
from interpolation import interp, mlinterp
from numba import jit,njit, float64, int64, boolean


# + code_folding=[0]
def policyfunc(lc,
               a_star,
               σ_star,
               discrete = True):
    """
     * ifp is an instance of IFP
        * a_star is the endogenous grid solution
        * σ_star is optimal consumption on the grid    
    """
    if discrete==True:
        # Create consumption function by linear interpolation
        σ =  lambda a, z_idx: interp(a_star[:, z_idx], σ_star[:, z_idx], a) 
    else:
        # get z_grid 
        z_val = lc.z_val 

        # Create consumption function by linear interpolation
        a = a_star[:,0]                                ## aseet grid 
        σ =  interpolate.interp2d(a, z_val, σ_star.T) 
    
    return σ


# + code_folding=[0]
def policyfuncMA(lc,
                 aϵ_star,
                 σ_star):
    """
     * ifp is an instance of IFP
        * aϵ_star is the endogenous grid solution
        * σ_star is optimal consumption on the grid    
    """
   
   # get s_grid and eps grid 
    s_grid = lc.s_grid
    eps_grid = lc.eps_grid 

    # Create consumption function by linear interpolation 
    σ =  interpolate.interp2d(eps_grid, s_grid, σ_star.T) 
    
    return σ


# + code_folding=[1]
@njit
def policyfuncMAjit(lc,
                 aϵ_star,
                 σ_star):
    """
     * ifp is an instance of IFP
        * aϵ_star is the endogenous grid solution
        * σ_star is optimal consumption on the grid    
    """
   
   # get z_grid 
    s_grid = lc.s_grid
    eps_grid = lc.eps_grid 
    
    # Create consumption function by linear interpolation
    σ =  lambda eps,a: mlinterp((eps_grid,s_grid),σ_star.T,(eps,a))
    
    return σ


# -

# ==============================================================================
# ============== Functions for generating state space grids  ===================
# ============= Copied from HARK  ==============================================
# ==============================================================================
def make_grid_exp_mult(ming, maxg, ng, timestonest=20):
    """
    Make a multi-exponentially spaced grid.

    Parameters
    ----------
    ming : float
        Minimum value of the grid
    maxg : float
        Maximum value of the grid
    ng : int
        The number of grid points
    timestonest : int
        the number of times to nest the exponentiation

    Returns
    -------
    points : np.array
        A multi-exponentially spaced grid

    Original Matab code can be found in Chris Carroll's
    [Solution Methods for Microeconomic Dynamic Optimization Problems]
    (http://www.econ2.jhu.edu/people/ccarroll/solvingmicrodsops/) toolkit.
    Latest update: 01 May 2015
    """
    if timestonest > 0:
        Lming = ming
        Lmaxg = maxg
        for j in range(timestonest):
            Lming = np.log(Lming + 1)
            Lmaxg = np.log(Lmaxg + 1)
        Lgrid = np.linspace(Lming, Lmaxg, ng)
        grid = Lgrid
        for j in range(timestonest):
            grid = np.exp(grid) - 1
    else:
        Lming = np.log(ming)
        Lmaxg = np.log(maxg)
        Lstep = (Lmaxg - Lming) / (ng - 1)
        Lgrid = np.arange(Lming, Lmaxg + 0.000001, Lstep)
        grid = np.exp(Lgrid)
    return grid


# ## Tools used for wealth distributions 

# + code_folding=[1]
## lorenz curve
def lorenz_curve(grid_distribution,
                 pdfs,
                 nb_share_grid = 50):
    """
    parameters
    ======
    grid_distribution: grid on which distribution is defined
    pdfs: the fractions/pdfs of each grid ranges 
    
    return
    ======
    lc_vals: the fraction of people corresponding whose total wealth reaches the corresponding share, x axis in lorenz curve
    share_grids: different grid points of the share of total wealth, y axis in lorenz curve
    """
    total = np.dot(grid_distribution,pdfs)
    share_grids = np.linspace(0.0,0.99,nb_share_grid)
    share_cum = np.multiply(grid_distribution,pdfs).cumsum()/total
    lc_vals = []
    for i,share in enumerate(share_grids):
        where = min([x for x in range(len(share_cum)) if share_cum[x]>=share])
        this_lc_val = pdfs[0:where].sum()
        lc_vals.append(this_lc_val)
    return np.array(lc_vals),share_grids



# -

# ## Tools for Markov regime switching 
#

# +
## some functions used for markov-related calculations 
## a simple function that computes steady state of 2-state markov
@njit
def cal_ss_2markov(P):
    ## an analytical solution for 2-state markov for double checking 
    ## when P's row sums up to 1
    #P.T = [[q,1-p],[1-q,p]]
    q = P.T[0,0]
    p = P.T[1,1]
    h = (1-p)/(2-p-q)
    return np.array([h,1-h])

def mkv2_M2Q(q,p):
    """
    input
    ======
    q and p are staying probs at monthly frequency 
    
    output
    ======
    qq and pp are quarterly counterparts 
    """
    
    ## different possibilities of staying in low state 
    qq0 = q**3   #LLLL
    qq1 = q*(1-q)*(1-p)    ## LLHL
    qq2 = (1-q)*(1-p)*q    ## LHLL
    qq3 = (1-q)*q*(1-q)    ## LHHL
    qq = qq0+qq1+qq2+qq3
    
    ## different possibilities of staying in high state
    
    pp0 = p**3             #HHHH
    pp1 = p*(1-p)*(1-q)    ## HHLH
    pp2 = (1-p)*(1-q)*p    ## HLHH
    pp3 = (1-q)*p*(1-p)    ## HLLH 
    pp = qq0+qq1+qq2+qq3
    
    return qq, pp

def mkv2_Q2Y(q,p):
    """
    input
    ======
    q and p are staying probs at quarterly frequency 
    
    output
    ======
    qq and pp are yearly counterparts 
    """
    
    ## 8 different possibilities of staying in low state 
    qq0 = q**4                               #L LLL L
    qq1 = q**2*(1-q)*(1-p)                   #L LLH L
    qq2 = q*(1-q)*(1-p)*q                    #L LHL L
    qq3 = q*(1-q)*p*(1-p)                    #L LHH L
    qq4 = (1-q)*(1-p)*q**2                   #L HLL L
    qq5 = (1-q)*(1-p)*(1-q)*(1-p)            #L HLH L
    qq6 = (1-q)*p*(1-p)*q                    #L HHL L
    qq7 = (1-q)*p**2*(1-p)                   #L HHH L
    qq = qq0+qq1+qq2+qq3+qq4+qq5+qq6+qq7
    
    ## 8 different possibilities of staying in high state
    
    pp0 = p**4                               #H HHH H
    pp1 = p**2*(1-p)*(1-q)                   #H HHL H
    pp2 = p*(1-p)*(1-q)*p                    #H HLH H
    pp3 = p*(1-p)*q*(1-q)                    #H HLL H
    pp4 = (1-p)*(1-q)*p**2                   #H LHH H
    pp5 = (1-p)*(1-q)*(1-p)*(1-q)            #H LHL H
    pp6 = (1-p)*q*(1-q)*p                    #H LLH H
    pp7 = (1-p)*q**2*(1-q)                   #H LLL H
    pp = pp0+pp1+pp2+pp3+pp4+pp5+pp6+pp7
    
    return qq, pp

def mkv2_Y2M(q,
             p):
    """
    input
    =====
    transition probs at the annual frequency 
    output
    =====
    monthly transition probs computed via continuous time Poisson rate 
    """
    
    ## to be completed 
    poisson_qM = -np.log(1-q)/12   ## = -np.log(1-qq)
    qq = 1-np.exp(-poisson_qM)
    
    poisson_pM = -np.log(1-p)/12   ## = -np.log(1-qq)
    pp = 1-np.exp(-poisson_pM)
    return qq,pp

def mkv2_Y2Q(q,
             p):
    """
    input
    =====
    transition probs at the annual frequency 
    output
    =====
    quarterly transition probs computed via continuous time Poisson rate 
    """
    
    ## to be completed 
    poisson_qM = -np.log(1-q)/3   ## = -np.log(1-qq)
    qq = 1-np.exp(-poisson_qM)
    
    poisson_pM = -np.log(1-p)/3   ## = -np.log(1-qq)
    pp = 1-np.exp(-poisson_pM)
    return qq,pp



# -
# ## Tools for the economy and market 


# + code_folding=[31]
class CDProduction:
    ## An economy class that saves market and production parameters 
    
    def __init__(self,
             Z = 1.00,     
             K = 1.00, 
             L = 1.00,
             α = 0.33, 
             δ = 0.025,  
             target_KY = 3.0,
             target_W = 1.0):  
        self.Z = Z
        self.K = K
        self.L = L
        self.α = α
        self.δ = δ
        self.target_KY = target_KY
        self.target_W = target_W
        
    def KY(self):
        return (self.K/self.Y())
    
    def Y(self):
        return self.Z*self.K**self.α*self.L**(1-self.α)
    
    def YL(self):
        return self.Z*(1-self.α)*(self.K/self.L)**self.α
    
    def YK(self):
        return self.Z*self.α*(self.L/self.K)**(1-self.α)
    
    def R(self):
        return 1+self.YK()-self.δ
    
    def normlize_Z(self,
                  N_ss = 1.0):
        from scipy.optimize import fsolve
        
        target_KY = self.target_KY
        target_W = self.target_W

        print('target KY',target_KY)
        print('target W',target_W)
        print('steady state emp pop',N_ss)

        def distance(ZK):
            self.N = N_ss
            self.Z,self.K = ZK
            distance1 = self.KY()- target_KY
            distance2 = self.YL()- target_W 
            return [distance1,distance2]

        Z_root,K_root = fsolve(distance,
                               [0.7,0.5])

        print('Normalized Z',Z_root)
        print('Normalized K',K_root)

        self.Z,self.K = Z_root,K_root

        W_fake = self.YL()
        KY_fake = self.KY()
        R_fake = self.R()

        print('W',W_fake)
        print('KY',KY_fake)
        print('R',R_fake)


# + code_folding=[2, 22, 43]
## get the stationary age distribution 
@njit
def stationary_age_dist(L,
                        n,
                       LivPrb):
    """
    stationary age distribution of the economy given 
    T: nb of periods of life 
    n: Population growth rate 
    ProbLiv: survival probability 
    """
    cum = 0.0
    for i in range(L):
        cum = cum + LivPrb**i/(1+n)
    sigma1 = 1/cum
    
    dist = np.empty(L)
    
    for i in range(L):
        dist[i] = sigma1*LivPrb**i/(1+n)
    return dist 

def unemp_insurance2tax(μ,
                        ue_fraction):
    """
    input
    =====
    μ: replcament ratio
    ue_fraction: fraction of the working population that is unemployed 
    output
    ======
    tax rate: labor income tax rate that balances government budget paying for uemp insurance 
    
    under balanced government budget, what is the tax rate on income corresponds to the ue benefit μ
    (1-ue_fraction)x tax + ue_fraction x mu x tax = ue_fraction x mu 
    --> tax = (ue_fraction x mu)/(1-ue_fraction)+ue_fraction x mu
    """
    num = (ue_fraction*μ)
    dem = (1-ue_fraction)+(ue_fraction*μ)
    return num/dem

## needs to test this function 

def SS2tax(SS, ## social security /pension replacement ratio 
           T,  ## retirement years
           age_dist,  ## age distribution in the economy 
           G,         ## permanent growth fractor lists over cycle
           emp_fraction):  ## fraction of employment in work age population 
    pic_share = np.cumprod(G)  ## generational permanent income share 
    pic_age_share = np.multiply(pic_share,
                               age_dist)  ## generational permanent income share weighted by population weights
    
    dependence_ratio = np.sum(pic_age_share[T:])/np.sum(pic_age_share[:T-1]*emp_fraction)
    ## old dependence ratio 
    
    λ_SS = SS*dependence_ratio 
    ## social security tax rate on labor income of employed 
    
    return λ_SS
