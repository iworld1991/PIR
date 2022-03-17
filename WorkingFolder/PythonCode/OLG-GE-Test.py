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

# ## Aggregate dynamics, stationary distribution and GE of a life-cycle economy
#
# - author: Tao Wang
# - date: November 2021
# - this is a companion notebook to the paper "Perceived income risks"

import numpy as np
import pandas as pd
from interpolation import interp, mlinterp
import numba as nb
from numba import jit, njit, float64, int64, boolean
from numba.typed import List
from numba.experimental import jitclass
import matplotlib as mp
import matplotlib.pyplot as plt
# %matplotlib inline
from quantecon import MarkovChain
import quantecon as qe 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from time import time
from Utility import make_grid_exp_mult
from scipy import sparse as sp
import scipy.sparse.linalg
import scipy.optimize as op
from scipy import linalg as lg 
from matplotlib import cm
import joypy
from copy import copy,deepcopy
from Utility import cal_ss_2markov,lorenz_curve


# + code_folding=[]
## figure plotting configurations

mp.rc('xtick', labelsize=14) 
mp.rc('ytick', labelsize=14) 

fontsize = 14
legendsize = 12

plt.style.use('seaborn')

# -

# ### The Life-cycle Model Class and Its Solver

# + code_folding=[]
from SolveLifeCycle import LifeCycle, EGM, solve_model_backward_iter,compare_2solutions
# -

# ### Initialize the model

from PrepareParameters import life_cycle_paras_q as lc_paras_Q
from PrepareParameters import life_cycle_paras_y as lc_paras_Y
## make a copy of the imported parameters 
lc_paras_y = copy(lc_paras_Y)
lc_paras_q = copy(lc_paras_Q)

# +
## make some modifications 
#lc_paras_y['P'] =np.array([[0.0,1.0],[0.0,1.0]])
#lc_paras_y['unemp_insurance'] = 0.0 
#lc_paras_y['init_b'] = 0.0 
#lc_paras_y['G'] = np.ones_like(lc_paras_y['G'])
# -

print(lc_paras_y)


# + code_folding=[]
## a deterministic income profile 

## income profile 
YPath = np.cumprod(lc_paras_y['G'])

plt.title('Deterministic Life-cycle Income Profile \n')
plt.plot(YPath,'ko-')
plt.xlabel('Age')
plt.ylabel(r'$\hat Y$')


# + code_folding=[]
#this is a fake life cycle income function 

def fake_life_cycle(L):
    LPath = np.arange(L+1)
    Y_fake = -0.01*(LPath-int(L/3))**2+0.03*LPath+20
    G = Y_fake[1:]/Y_fake[:-1]
    return G


# + code_folding=[]
## parameters for testing 

U = 0.0 ## transitory ue risk 
LivPrb = 0.99
unemp_insurance = 0.15
sigma_psi = np.sqrt(0.01) # permanent 
sigma_eps = np.sqrt(0.04) # transitory 
sigma_p_init = np.sqrt(0.03)
init_b = 0.0
λ = 0.0942 
λ_SS = 0.0
transfer = 0.0
pension = 0.5

T = 40
L = 60
TGPos = int(L/2)
GPos = 1.01*np.ones(TGPos)
GNeg= 0.99*np.ones(L-TGPos)
#G = np.concatenate([GPos,GNeg])
#YPath = np.cumprod(G)
G = fake_life_cycle(L)
YPath = np.cumprod(G)


## other parameters 
ρ = 1
R = 1.01
W = 1.0
β = 0.96
x = 0.0

## no persistent state
b_y = 0.0

## set the bool to be true to turn on unemployment/employment markov (persistent unemployment risks)
ue_markov = True
###################################

## natural borrowing constraint if False
borrowing_cstr = True

## extrapolation parameter

theta = 0.0

## bequest ratio 
bequest_ratio = 0.0
# -


# ### Solve the model with a Markov state: unemployment and employment 

# + code_folding=[126]
## initialize a class of life-cycle model with either calibrated or test parameters 

#################################
calibrated_model = True
model_frequency = 'yearly'
#################################

if calibrated_model == True:

    if model_frequency=='yearly':
        ## yearly parameters 

        lc_paras = lc_paras_y

    elif model_frequency=='quarterly':
        ## yearly parameters 
        lc_paras = lc_paras_q


    ## initialize the model with calibrated parameters 
    lc_mkv = LifeCycle(
        ## primitives
                   ρ = lc_paras['ρ'],     ## relative risk aversion  
                   β = lc_paras['β'],     ## discount factor
                   borrowing_cstr = borrowing_cstr,
                   adjust_prob = 1.0,
        
        ## prices 
                   R = lc_paras['R'],           ## interest factor
                   W = lc_paras['W'],           ## Wage rate
        
        ## life cycle 
                   T = lc_paras['T'],
                   L = lc_paras['L'],
                   G = lc_paras['G'],
                   LivPrb = lc_paras['LivPrb'],       ## living probability 
        
        ## income risks 
                   x = 0.0,
                   b_y= 0.0,
                   sigma_psi = lc_paras['σ_ψ'],
                   sigma_eps = lc_paras['σ_θ'],
                   subjective = True,
                   ue_markov = True,
                   P = lc_paras['P'],
                   U = lc_paras['U'],
                   z_val = lc_paras['z_val'], ## markov state from low to high 
                   sigma_psi_2mkv = lc_paras['σ_ψ_2mkv'],  ## permanent risks in 2 markov states
                   sigma_eps_2mkv = lc_paras['σ_θ_2mkv'],  ## transitory risks in 2 markov states
                   sigma_psi_true = lc_paras['σ_ψ'], ## true permanent
                   sigma_eps_true = lc_paras['σ_θ'], ## true transitory
        
        ## initial conditions 
                    sigma_p_init = lc_paras['σ_ψ_init'],
                    init_b = lc_paras['init_b'],

        ## policy 
                   unemp_insurance = lc_paras['unemp_insurance'],
                   pension = lc_paras['pension'], ## pension
                   λ = lc_paras['λ'],  ## tax rate
                   λ_SS = lc_paras['λ_SS'], ## social tax rate
                   transfer = lc_paras['transfer'],  ## transfer 
                   bequest_ratio = lc_paras['bequest_ratio'],
         ## solutions 
                   shock_draw_size = 10.0,
                   grid_max = 10
                   )
    print(lc_mkv.psi_shk_draws)
    print(lc_mkv.eps_shk_draws)
    print(lc_mkv.psi_shk_true_draws)
    print(lc_mkv.eps_shk_true_draws)

    ## for the subjective model, only change the belief 

    lc_mkv_sub = LifeCycle(
        ## primitives
                   ρ = lc_paras['ρ'],     ## relative risk aversion  
                   β = lc_paras['β'],     ## discount factor
                   borrowing_cstr = borrowing_cstr,
                   adjust_prob = 1.0,
        ## prices 
                   R = lc_paras['R'],           ## interest factor
                   W = lc_paras['W'],            ## Wage rate
        ## life cycle 
                   T = lc_paras['T'],
                   L = lc_paras['L'],
                   G = lc_paras['G'],
                   LivPrb = lc_paras['LivPrb'],       ## living probability 
        
        ## income risks 
                   x = 0.0,
                   b_y= 0.0,
                   sigma_psi = lc_paras['σ_ψ_sub'],
                   sigma_eps = lc_paras['σ_θ_sub'],
                   subjective =True,
                   ue_markov = True,
                   P = lc_paras['P'],
                   U = lc_paras['U'],
                   z_val = lc_paras['z_val'], ## markov state from low to high
                   sigma_psi_2mkv = lc_paras['σ_ψ_2mkv'],  ## permanent risks in 2 markov states
                   sigma_eps_2mkv = lc_paras['σ_θ_2mkv'],  ## transitory risks in 2 markov states
                   sigma_psi_true = lc_paras['σ_ψ'], ## true permanent
                   sigma_eps_true = lc_paras['σ_θ'], ## true transitory
        
        ## initial conditions 
                    sigma_p_init = lc_paras['σ_ψ_init'],
                    init_b = lc_paras['init_b'],

        ## policy 
                   unemp_insurance = lc_paras['unemp_insurance'],
                   pension = lc_paras['pension'], ## pension
                   λ = lc_paras['λ'],  ## tax rate
                   λ_SS = lc_paras['λ_SS'], ## social tax rate
                   transfer = lc_paras['transfer'],  ## transfer 
                   bequest_ratio = lc_paras['bequest_ratio'],
        ## solutions 
                  shock_draw_size =  10.0,
                  grid_max = 10
                   )
    
    print(lc_mkv_sub.psi_shk_draws)
    print(lc_mkv_sub.eps_shk_draws)
    print(lc_mkv_sub.psi_shk_true_draws)
    print(lc_mkv_sub.eps_shk_true_draws)


else:
    ## only for testing 
    lc_mkv = LifeCycle(sigma_psi = sigma_psi,
                       sigma_eps = sigma_eps,
                       U=U,
                       LivPrb = LivPrb,
                       ρ=ρ,
                       R=R,
                       W=W,
                       G=G,
                       T=T,
                       L=L,
                       β=β,
                       x=x,  ## shut down ma(1)
                       theta=theta,
                       borrowing_cstr = borrowing_cstr,
                       b_y = b_y, ## set the macro state loading to be zero, it does not matter for ue_markov
                       unemp_insurance = unemp_insurance, 
                       pension = pension,
                       ue_markov = ue_markov,
                       sigma_p_init =sigma_p_init,
                       init_b = init_b,
                       λ = λ,
                       transfer = transfer,
                       bequest_ratio = bequest_ratio
                      )
    
    lc_mkv_sub = LifeCycle(sigma_psi = 0.1*sigma_psi, ##  0.1 is arbitrary but just to make the PR lower
                       sigma_eps = 0.1*sigma_eps,
                       subjective =True,
                       sigma_psi_true = sigma_psi,
                       sigma_eps_true = sigma_eps,
                       U=U,
                       LivPrb = LivPrb,
                       ρ=ρ,
                       R=R,
                       W=W,
                       G=G,
                       T=T,
                       L=L,
                       β=β,
                       x=x,  ## shut down ma(1)
                       theta=theta,
                       borrowing_cstr = borrowing_cstr,
                       b_y = b_y, ## set the macro state loading to be zero, it does not matter for ue_markov
                       unemp_insurance = unemp_insurance, 
                       pension = pension,
                       ue_markov = ue_markov,
                       sigma_p_init =sigma_p_init,
                       init_b = init_b,
                       λ = λ,
                       transfer = transfer,
                       bequest_ratio = bequest_ratio
                      )


# + code_folding=[0]
## solve various models

models = [lc_mkv,lc_mkv_sub]
model_names=['objective','subjective']

ms_stars = []
σs_stars = []

t_start = time()
for i, model in enumerate(models):
    ## terminal solution
    m_init,σ_init = model.terminal_solution()

    ## solve backward
    ms_star, σs_star = solve_model_backward_iter(model,
                                                 m_init,
                                                 σ_init)
    ms_stars.append(ms_star)
    σs_stars.append(σs_star)

t_finish = time()

print("Time taken, in seconds: "+ str(t_finish - t_start))



# +
## be careful with the order 
## get the solution for the objective model 
ms_star_mkv, σs_star_mkv = ms_stars[0],σs_stars[0]

## get the solution for the subjective model 
ms_star_mkv_sub,σs_star_mkv_sub = ms_stars[1],σs_stars[1]

# +
## compare different models 

ojb_minus_sub = compare_2solutions(ms_stars,
                                   σs_stars)

plt.hist(ojb_minus_sub.flatten(),
         bins=50)
plt.title('Consumption in subjective model minus objective model')
print('should be NEGATIVE!!!!!')

# + code_folding=[0]
## compare solutions 

m_grid = np.linspace(0.0,10.0,200)
## plot c func at different age /asset grid
years_left = [0,21,40,59]

n_sub = len(years_left)

eps_fix = 0 ## the first eps grid 

fig,axes = plt.subplots(1,n_sub,figsize=(4*n_sub,4))

for x,year in enumerate(years_left):
    age = lc_mkv.L-year
    i = lc_mkv.L-age
    for k,model_name in enumerate(model_names):
        m_plt_u,c_plt_u = ms_stars[k][i,:,eps_fix,0],σs_stars[k][i,:,eps_fix,0]
        m_plt_e,c_plt_e = ms_stars[k][i,:,eps_fix,1],σs_stars[k][i,:,eps_fix,1]
        c_func_u = lambda m: interp(m_plt_u,c_plt_u,m)
        c_func_e = lambda m: interp(m_plt_e,c_plt_e,m)
        #axes[x].plot(m_grid,
        #             c_func_u(m_grid),
        #             label = model_name+',unemployed',
        #             lw=3
        #            )
        axes[x].plot(m_grid,
                     c_func_e(m_grid),
                     label = model_name+',employed',
                     lw=3
                    )
    axes[x].legend()
    #axes[x].set_xlim(0.0,np.max(m_plt_u))
    axes[x].set_xlabel('asset')
    axes[0].set_ylabel('c')
    axes[x].set_title(r'$age={}$'.format(age))

# + code_folding=[0]
## 3d graph 

ms_star_mkv_u_2d,cs_star_mkv_u_2d = ms_star_mkv[:,:,eps_fix,0],σs_star_mkv[:,:,eps_fix,0]
ms_star_mkv_e_2d,cs_star_mkv_e_2d  = ms_star_mkv[:,:,eps_fix,1],σs_star_mkv[:,:,eps_fix,1]

ms_star_mkv_u_sub_2d,cs_star_mkv_u_sub_2d = ms_star_mkv_sub[:,:,eps_fix,0],σs_star_mkv_sub[:,:,eps_fix,0]
ms_star_mkv_e_sub_2d,cs_star_mkv_e_sub_2d = ms_star_mkv_sub[:,:,eps_fix,1],σs_star_mkv_sub[:,:,eps_fix,1]

## get age grid 
ages = np.array(range(ms_star_mkv_u_2d.shape[0]))

## objective model 
m_mkv_u = ms_star_mkv_u_2d[0,:]
xx_mkv_u, yy_mkv_u = np.meshgrid(ages, m_mkv_u)
c_mkv_u = np.flip(cs_star_mkv_u_2d,axis=0).T

m_mkv_e = ms_star_mkv_e_2d[0,:]
xx_mkv_e, yy_mkv_e = np.meshgrid(ages, m_mkv_e)
c_mkv_e = np.flip(cs_star_mkv_e_2d,axis=0).T

## subjective model 
m_mkv_u_sub = ms_star_mkv_u_sub_2d[0,:]
xx_mkv_u_sub, yy_mkv_u_sub = np.meshgrid(ages, m_mkv_u_sub)
c_mkv_u_sub = np.flip(cs_star_mkv_u_sub_2d,axis=0).T

m_mkv_e_sub = ms_star_mkv_e_sub_2d[0,:]
xx_mkv_e_sub, yy_mkv_e_sub = np.meshgrid(ages, m_mkv_e_sub)
c_mkv_e_sub = np.flip(cs_star_mkv_e_sub_2d,axis=0).T


## plot 
fig = plt.figure(figsize=(15,7))
ax = fig.add_subplot(121, 
                     projection='3d')
dem3d = ax.plot_surface(xx_mkv_u,
                        yy_mkv_u,
                        c_mkv_u,
                        cmap="winter"
                       )
dem3d_sub = ax.plot_surface(xx_mkv_u_sub,
                        yy_mkv_u_sub,
                        c_mkv_u_sub,
                        cmap="winter"
                       )
ax.set_xlim(60,0)
ax.set_title('Consumption over the life cycle')
ax.set_xlabel('age')
#ax.grid(False)
ax.set_ylabel('wealth')
ax.view_init(10, 20)

ax2 = fig.add_subplot(122, 
                     projection='3d')
dem3d2 = ax2.plot_surface(xx_mkv_e,
                        yy_mkv_e,
                        c_mkv_e,
                        cmap="summer"
                       )
dem3d2_sub = ax2.plot_surface(xx_mkv_e_sub,
                        yy_mkv_e_sub,
                        c_mkv_e_sub,
                        cmap="summer"
                       )
ax2.set_xlim(60,0)
ax2.set_xlabel('age')
#ax.grid(False)
ax2.set_ylabel('wealth')
ax2.view_init(10, 20)
# -

# ## Aggregate steady state distributions

from Utility import stationary_age_dist
## a function that computes social security tax rate balances gov budget for pension
from Utility import unemp_insurance2tax  
## a function that computes tax rate balances gov budget for ue insurance
from Utility import SS2tax
from Utility import CDProduction  
from PrepareParameters import production_paras_y as production_paras


# + code_folding=[6, 101, 142, 467, 482, 489, 518, 525, 554, 568, 586]
#################################
## general functions used 
# for computing transition matrix
##################################

@njit
def jump_to_grid(model, 
                 m_vals, 
                 perm_vals, 
                 probs, 
                 dist_mGrid, 
                 dist_pGrid):

    '''
    Distributes values onto a predefined grid, maintaining the means. m_vals and perm_vals are realizations of market resources and permanent income while 
    dist_mGrid and dist_pGrid are the predefined grids of market resources and permanent income, respectively. That is, m_vals and perm_vals do not necesarily lie on their 
    respective grids. Returns probabilities of each gridpoint on the combined grid of market resources and permanent income.


    Parameters
    ----------
    m_vals: np.array
            Market resource values 

    perm_vals: np.array
            Permanent income values 

    probs: np.array
            Shock probabilities associated with combinations of m_vals and perm_vals. 
            Can be thought of as the probability mass function  of (m_vals, perm_vals).

    dist_mGrid : np.array
            Grid over normalized market resources

    dist_pGrid : np.array
            Grid over permanent income 

    Returns
    -------
    probGrid.flatten(): np.array
             Probabilities of each gridpoint on the combined grid of market resources and permanent income
    '''

    probGrid = np.zeros((len(dist_mGrid),len(dist_pGrid)))
    mIndex = np.digitize(m_vals,dist_mGrid) - 1 # Array indicating in which bin each values of m_vals lies in relative to dist_mGrid. Bins lie between between point of Dist_mGrid. 
    #For instance, if mval lies between dist_mGrid[4] and dist_mGrid[5] it is in bin 4 (would be 5 if 1 was not subtracted in the previous line). 
    mIndex[m_vals <= dist_mGrid[0]] = -1 # if the value is less than the smallest value on dist_mGrid assign it an index of -1
    mIndex[m_vals >= dist_mGrid[-1]] = len(dist_mGrid)-1 # if value if greater than largest value on dist_mGrid assign it an index of the length of the grid minus 1

    #the following three lines hold the same intuition as above
    pIndex = np.digitize(perm_vals,dist_pGrid) - 1
    pIndex[perm_vals <= dist_pGrid[0]] = -1
    pIndex[perm_vals >= dist_pGrid[-1]] = len(dist_pGrid)-1

    for i in range(len(m_vals)):
        if mIndex[i]==-1: # if mval is below smallest gridpoint, then assign it a weight of 1.0 for lower weight. 
            mlowerIndex = 0
            mupperIndex = 0
            mlowerWeight = 1.0
            mupperWeight = 0.0
        elif mIndex[i]==len(dist_mGrid)-1: # if mval is greater than maximum gridpoint, then assign the following weights
            mlowerIndex = -1
            mupperIndex = -1
            mlowerWeight = 1.0
            mupperWeight = 0.0
        else: # Standard case where mval does not lie past any extremes
        #identify which two points on the grid the mval is inbetween
            mlowerIndex = mIndex[i] 
            mupperIndex = mIndex[i]+1
        #Assign weight to the indices that bound the m_vals point. Intuitively, an mval perfectly between two points on the mgrid will assign a weight of .5 to the gridpoint above and below
            mlowerWeight = (dist_mGrid[mupperIndex]-m_vals[i])/(dist_mGrid[mupperIndex]-dist_mGrid[mlowerIndex]) #Metric to determine weight of gridpoint/index below. Intuitively, mvals that are close to gridpoint/index above are assigned a smaller mlowerweight.
            mupperWeight = 1.0 - mlowerWeight # weight of gridpoint/ index above

        #Same logic as above except the weights here concern the permanent income grid
        if pIndex[i]==-1: 
            plowerIndex = 0
            pupperIndex = 0
            plowerWeight = 1.0
            pupperWeight = 0.0
        elif pIndex[i]==len(dist_pGrid)-1:
            plowerIndex = -1
            pupperIndex = -1
            plowerWeight = 1.0
            pupperWeight = 0.0
        else:
            plowerIndex = pIndex[i]
            pupperIndex = pIndex[i]+1
            plowerWeight = (dist_pGrid[pupperIndex]-perm_vals[i])/(dist_pGrid[pupperIndex]-dist_pGrid[plowerIndex])
            pupperWeight = 1.0 - plowerWeight

        # Compute probabilities of each gridpoint on the combined market resources and permanent income grid by looping through each point on the combined market resources and permanent income grid, 
        # assigning probabilities to each gridpoint based off the probabilities of the surrounding mvals and pvals and their respective weights placed on the gridpoint.
        # Note* probs[i] is the probability of mval AND pval occurring
        probGrid[mlowerIndex][plowerIndex] = probGrid[mlowerIndex][plowerIndex] + probs[i]*mlowerWeight*plowerWeight # probability of gridpoint below mval and pval 
        probGrid[mlowerIndex][pupperIndex] = probGrid[mlowerIndex][pupperIndex] + probs[i]*mlowerWeight*pupperWeight # probability of gridpoint below mval and above pval
        probGrid[mupperIndex][plowerIndex] = probGrid[mupperIndex][plowerIndex] + probs[i]*mupperWeight*plowerWeight # probability of gridpoint above mval and below pval
        probGrid[mupperIndex][pupperIndex] = probGrid[mupperIndex][pupperIndex] + probs[i]*mupperWeight*pupperWeight # probability of gridpoint above mval and above pval

    return probGrid.flatten()

@njit
def jump_to_grid_fast(model, 
                      vals, 
                      probs,
                      Grid ):
    '''
    Distributes values onto a predefined grid, maintaining the means.
    ''' 

    probGrid = np.zeros(len(Grid))
    mIndex = np.digitize(vals,Grid) - 1
    # return the indices of the bins to which each value in input array belongs.
    mIndex[vals <= Grid[0]] = -1
    mIndex[vals >= Grid[-1]] = len(Grid)-1


    for i in range(len(vals)):
        if mIndex[i]==-1:
            mlowerIndex = 0
            mupperIndex = 0
            mlowerWeight = 1.0
            mupperWeight = 0.0
        elif mIndex[i]==len(Grid)-1:
            mlowerIndex = -1
            mupperIndex = -1
            mlowerWeight = 1.0
            mupperWeight = 0.0
        else:
            mlowerIndex = mIndex[i]
            mupperIndex = mIndex[i]+1
            mlowerWeight = (Grid[mupperIndex]-vals[i])/(Grid[mupperIndex] - Grid[mlowerIndex])
            mupperWeight = 1.0 - mlowerWeight

        probGrid[mlowerIndex] = probGrid[mlowerIndex] + probs[i]*mlowerWeight
        probGrid[mupperIndex] = probGrid[mupperIndex] + probs[i]*mupperWeight

    return probGrid.flatten()


## compute the list of transition matrix from age t to t+1 for all age 

@njit
def calc_transition_matrix(model, 
                           ms_star, ## new,  life cycle age x asset x tran shock x z state grid 
                           σs_star, ## new, life cycle consumptiona t age  x asset x tran shock x z state grid 
                           dist_mGrid_list, ## new, list, grid of m for distribution 
                           dist_pGrid_list,  ## new, list, grid of p for distribution 
                           finite_horizon = True, ## new 
                           fast = False   ## new 
                          ):
        '''
        Calculates how the distribution of agents across market resources 
        transitions from one period to the next. 
        If finite horizon problem, then calculates
        a list of transition matrices, consumption and asset policy grids for each period of the problem. 
        
        
        Parameters
        ----------
            # ms_star: array, sized of T x n_a x n_eps x n_z, wealth grid 
            # σs_star: array, sized of T x n_a x n_eps x n_z, consumption values at the grid
            # dist_mGrid_list, list, sized of 1, list of m grid sized of n_m
            # dist_pGrid_list, list, sized of T, list of permanent income grid for each age, sized of n_p
            # fast, bool, fast or slow method 

        Returns
        -------
            # tran_matrix_list, numba typed list, embedded list, sized of n_z,
            ## each of which is sized of T, each of which is sized of n_m x n_p 
        
        ''' 
        fix_epsGrid = 0.0
        
        ## nb of states 
        state_num = len(model.P)
        
        ## unemployment insurance 
        unemp_insurance = model.unemp_insurance
        
        ## tax rate
        λ = model.λ
        
        ## permanent income growth factor
        G = model.G
        
        ## grid holders
        cPol_Grid_e_list = [] # List of consumption policy grids for each period in T_cycle
        cPol_Grid_u_list = [] # List of consumption policy grids for each period in T_cycle

        aPol_Grid_e_list = [] # List of asset policy grids for each period in T_cycle
        aPol_Grid_u_list = [] # List of asset policy grids for each period in T_cycle
        
        
        tran_matrix_e_list = [] # List of transition matrices
        tran_matrix_u_list = [] # List of transition matrices
        

        #Obtain shocks and shock probabilities from income distribution in this period
        size_shk_probs  = len(model.eps_shk_true_draws)*len(model.psi_shk_true_draws)
        shk_prbs = np.ones(size_shk_probs)*1/size_shk_probs
        tran_shks = np.exp(np.repeat(model.eps_shk_true_draws,
                              len(model.psi_shk_true_draws)))
        perm_shks = np.exp(np.repeat(model.psi_shk_true_draws,
                              len(model.eps_shk_true_draws)))

        ## This is for the fast method 
        shk_prbs_ntrl =  np.multiply(shk_prbs,perm_shks)
        ## not used yet 
                        
        
                        
        for k in range(model.L): ## loop over agents at different ages, k
            
            age_id = k
            age = age_id + 1
            year_left = model.L-age
            
            markov_array2 = model.P

            this_dist_pGrid = dist_pGrid_list[0] #If here then use prespecified permanent income grid
            ## m-grid does not depend on period             
            this_dist_mGrid = dist_mGrid_list[0]
        
            
            ## compute different c at different a and eps
            
            n_mgrid = len(this_dist_mGrid)
            
            Cnow_u= np.empty(n_mgrid,dtype = np.float64)
            Cnow_e = np.empty(n_mgrid,dtype = np.float64)

            fix_epsGrid = 1.0 ## can be anything because c is not a function of eps
            
            for m_id,m in enumerate(this_dist_mGrid):
                this_Cnow_u = mlinterp((ms_star[year_left,:,0,0],   
                                        model.eps_grid),
                                       σs_star[year_left,:,:,0],
                                       (m,fix_epsGrid))
                Cnow_u[m_id] = this_Cnow_u
                
                #Cnow_u_list.append(this_Cnow_u)
                this_Cnow_e = mlinterp((ms_star[year_left,:,0,1],
                                        model.eps_grid),
                                       σs_star[year_left,:,:,1],
                                       (m,fix_epsGrid))
                Cnow_e[m_id] = this_Cnow_e
                #Cnow_e_list.append(this_Cnow_e)
                
            
            ## more generally, depending on the nb of markov states 
        

            cPol_Grid_u_list.append(Cnow_u)  # List of consumption policy grids for each age
            cPol_Grid_e_list.append(Cnow_e)  # List of consumption policy grids for each age

            aNext_u = this_dist_mGrid - Cnow_u # Asset policy grid in each age
            aNext_e = this_dist_mGrid - Cnow_e # Asset policy grid in each age
            
            aPol_Grid_u_list.append(aNext_u) # Add to list
            aPol_Grid_e_list.append(aNext_e) # Add to list

            bNext_u = model.R*aNext_u
            bNext_e = model.R*aNext_e

            
            if fast==True:  
                print('warning: the fast method is not fully developed yet!!!')
            
                # Generate Transition Matrix for u2u
                TranMatrix_uu = np.zeros((len(this_dist_mGrid),
                                          len(this_dist_mGrid))) 
                for i in range(len(this_dist_mGrid)):
                    if k <=model.T-1:
                        ## work age 
                        perm_shks_G = perm_shks*G[k+1]
                        mNext_ij = bNext_u[i]/perm_shks_G +model.transfer+(1-λ)*unemp_insurance 
                    else:
                        ## retirement 
                        perm_shks_none = np.ones_like(perm_shks)*G[k+1]
                        mNext_ij = bNext_u[i]/perm_shks_none +model.transfer+model.pension
                    # Compute next period's market resources given todays bank balances bnext[i]
                    TranMatrix_uu[:,i] = jump_to_grid_fast(model,
                                                           mNext_ij,
                                                           shk_prbs_ntrl,
                                                           this_dist_mGrid) 

                # Generate Transition Matrix for u2e
                TranMatrix_ue = np.zeros((len(this_dist_mGrid),
                                          len(this_dist_mGrid))) 
                for i in range(len(this_dist_mGrid)):
                    if k <=model.T-1:
                        ## work age 
                        perm_shks_G = perm_shks*G[k+1]
                        mNext_ij = bNext_u[i]/perm_shks_G +model.transfer+ (1-λ)*tran_shks  
                    else:
                        ## retirement 
                        perm_shks_none = np.ones_like(perm_shks)*G[k+1]
                        mNext_ij = bNext_u[i]/perm_shks_none +model.transfer+model.pension
                    # Compute next period's market resources given todays bank balances bnext[i]
                    TranMatrix_ue[:,i] = jump_to_grid_fast(model,
                                                            mNext_ij,
                                                            shk_prbs_ntrl,
                                                            this_dist_mGrid) 
                

                # Generate Transition Matrix for e2e 
                TranMatrix_ee = np.zeros((len(this_dist_mGrid),
                                          len(this_dist_mGrid))) 
                for i in range(len(this_dist_mGrid)):
                    if k <=model.T-1:
                        ## work age 
                        perm_shks_G = perm_shks*G[k+1]
                        mNext_ij = bNext_e[i]/perm_shks_G +model.transfer+ (1-λ)*tran_shks # Compute next period's market resources given todays bank balances bnext[i]
                    else:
                        ## retirement 
                        perm_shks_none = np.ones_like(perm_shks)*G[k+1]
                        mNext_ij = bNext_e[i]/perm_shks_none +model.transfer+model.pension
                    TranMatrix_ee[:,i] = jump_to_grid_fast(model,
                                                          mNext_ij,
                                                          shk_prbs_ntrl,
                                                          this_dist_mGrid)

                # Generate Transition Matrix for e2u 
                TranMatrix_eu = np.zeros((len(this_dist_mGrid),
                                          len(this_dist_mGrid))) 
                for i in range(len(this_dist_mGrid)):
                    if k <=model.T-1:
                        ## work age 
                        perm_shks_G = perm_shks*G[k+1]
                        mNext_ij = bNext_e[i]/perm_shks_G +model.transfer+ (1-λ)*unemp_insurance # Compute next period's market resources given todays bank balances bnext[i]
                    else:
                        ## retirement
                        perm_shks_none = np.ones_like(perm_shks)*G[k+1]
                        mNext_ij = bNext_e[i]/perm_shks_none +model.transfer+ model.pension
                    TranMatrix_eu[:,i] = jump_to_grid_fast(model,
                                                           mNext_ij, 
                                                           shk_prbs_ntrl,
                                                            this_dist_mGrid) 


            else:  ## slow method  (2-state Markov implemented)


                # Generate Transition Matrix for u2u 
                TranMatrix_uu = np.zeros((len(this_dist_mGrid)*len(this_dist_pGrid),
                                       len(this_dist_mGrid)*len(this_dist_pGrid))) 
                
                for i in range(len(this_dist_mGrid)):
                    for j in range(len(this_dist_pGrid)):
                        pNext_ij = this_dist_pGrid[j]*perm_shks*G[k+1] # Computes next period's permanent income level by applying permanent income shock
                        if k <=model.T-1:
                            perm_shks_G = perm_shks* G[k+1]
                            ## work age 
                            mNext_ij = bNext_u[i]/perm_shks_G +model.transfer+ (1-λ)*unemp_insurance # Compute next period's market resources given todays bank balances bnext[i]
                        else:
                            perm_shks_none = np.ones_like(perm_shks)*G[k+1]
                            ## retirement
                            mNext_ij = bNext_u[i]/perm_shks_none +model.transfer+ model.pension

                        TranMatrix_uu[:,i*len(this_dist_pGrid)+j] = jump_to_grid(model,
                                                                                 mNext_ij,
                                                                                 pNext_ij,
                                                                                 shk_prbs,
                                                                                 this_dist_mGrid, 
                                                                                this_dist_pGrid) 
                #TranMatrix = TranMatrix #columns represent the current state while rows represent the next state
                #the 4th row , 6th column entry represents the probability of transitioning from the 6th element of the combined perm and m grid (grid of market resources multiplied by grid of perm income) to the 4th element of the combined perm and m grid
                #tran_matrix_list.append(TranMatrix_uu)   
                
    
                # Generate Transition Matrix for u2e 
                TranMatrix_ue = np.zeros((len(this_dist_mGrid)*len(this_dist_pGrid),
                                       len(this_dist_mGrid)*len(this_dist_pGrid))) 
                
                for i in range(len(this_dist_mGrid)):
                    for j in range(len(this_dist_pGrid)):
                        pNext_ij = this_dist_pGrid[j]*perm_shks*G[k+1] # Computes next period's permanent income level by applying permanent income shock
                        if k <=model.T-1:
                            ## work age 
                            perm_shks_G = perm_shks* G[k+1]
                            mNext_ij = bNext_u[i]/perm_shks_G +model.transfer+ (1-λ)*tran_shks # Compute next period's market resources given todays bank balances bnext[i]
                        else:
                            ## retirement
                            perm_shks_none = np.ones_like(perm_shks)*G[k+1]
                            mNext_ij = bNext_u[i]/perm_shks_none +model.transfer+ model.pension
                        TranMatrix_ue[:,i*len(this_dist_pGrid)+j] = jump_to_grid(model,
                                                                               mNext_ij, 
                                                                               pNext_ij, 
                                                                               shk_prbs, 
                                                                               this_dist_mGrid, 
                                                                               this_dist_pGrid) 
                        
                # Generate Transition Matrix for e2u 
                TranMatrix_eu = np.zeros((len(this_dist_mGrid)*len(this_dist_pGrid),
                                       len(this_dist_mGrid)*len(this_dist_pGrid))) 
                
                for i in range(len(this_dist_mGrid)):
                    for j in range(len(this_dist_pGrid)):
                        pNext_ij = this_dist_pGrid[j]*perm_shks*G[k+1] # Computes next period's permanent income level by applying permanent income shock

                        if k <=model.T-1:
                            ## work age 
                            perm_shks_G = perm_shks* G[k+1]
                            mNext_ij = bNext_e[i]/perm_shks_G +model.transfer+ (1-λ)*unemp_insurance # Compute next period's market resources given todays bank balances bnext[i]
                        else:
                            ## retirement
                            perm_shks_none = np.ones_like(perm_shks)*G[k+1]
                            mNext_ij = bNext_e[i]/perm_shks_none +model.transfer+ model.pension
                        TranMatrix_eu[:,i*len(this_dist_pGrid)+j] = jump_to_grid(model,
                                                                                   mNext_ij, 
                                                                                   pNext_ij, 
                                                                                   shk_prbs, 
                                                                                   this_dist_mGrid, 
                                                                                   this_dist_pGrid) 
                        
                # Generate Transition Matrix for e2e 
                TranMatrix_ee = np.zeros((len(this_dist_mGrid)*len(this_dist_pGrid),
                                       len(this_dist_mGrid)*len(this_dist_pGrid))) 
                
                for i in range(len(this_dist_mGrid)):
                    for j in range(len(this_dist_pGrid)):
                        pNext_ij = this_dist_pGrid[j]*perm_shks*G[k+1] # Computes next period's permanent income level by applying permanent income shock
                        if k <=model.T-1:
                            perm_shks_G = perm_shks*G[k+1]
                            mNext_ij = bNext_e[i]/perm_shks_G +model.transfer+(1-λ)*tran_shks # Compute next period's market resources given todays bank balances bnext[i]
                        else:
                            ## retirement
                            perm_shks_none = np.ones_like(perm_shks)*G[k+1]
                            mNext_ij = bNext_e[i]/perm_shks_none +model.transfer+model.pension
                        TranMatrix_ee[:,i*len(this_dist_pGrid)+j] = jump_to_grid(model,
                                                                               mNext_ij, 
                                                                               pNext_ij, 
                                                                               shk_prbs, 
                                                                               this_dist_mGrid, 
                                                                               this_dist_pGrid) 

                        
        ###################################################
        ## back from the fork between slow and fast method 
        ##################################################
        ## transition matrix for each markov state 
            tran_matrix_u = markov_array2[0,1] * TranMatrix_ue  + markov_array2[0,0]* TranMatrix_uu #This is the transition for someone who's state today is unemployed
            tran_matrix_e = markov_array2[1,1]*TranMatrix_ee  +  markov_array2[1,0] * TranMatrix_eu # This is the transition for someone who's state is employed today

                
            ## merge to the life cycle list 
            tran_matrix_u_list.append( tran_matrix_u ) #This is the transition for someone who's state today is unemployed
            tran_matrix_e_list.append( tran_matrix_e )
            
                
        tran_matrix_list = List([tran_matrix_u_list,
                                 tran_matrix_e_list])
        
        ## return aggregate transition matrix and 
        ###.   the age/state dependent transition matricies necessary for computing aggregate consumption 
        
        
        ## consumption policy and saving grid on each m, p, z and k grid 
        cPol_Grid_list = List([cPol_Grid_u_list,
                               cPol_Grid_e_list])  # List of consumption policy grids for each period in T_cycle
        aPol_Grid_list = List([aPol_Grid_u_list,
                               aPol_Grid_e_list]) ## list of consumption 
        
        
        return tran_matrix_list, cPol_Grid_list,aPol_Grid_list

@njit
def aggregate_transition_matrix(model,
                                tran_matrix_lists,  ## size model.T 
                                dstn_0,    ## size n.z
                                age_dist): ## size model.T 
    ## aggregate different ages in the population
    n1,n2 = tran_matrix_lists[0][0].shape
    trans_matrix_agg = np.zeros((n1,n2),
                                dtype=np.float64)
    for z in range(len(dstn_0)):
        for k in range(model.L):
            trans_matrix_agg = trans_matrix_agg+dstn_0[z]*age_dist[k]*tran_matrix_lists[z][k] 
    return trans_matrix_agg


@njit
def initial_distribution_u(model,
                         dist_mGrid, ## new, array, grid of m for distribution 
                         dist_pGrid,  ## new, array, grid of p for distribution 
                        ):
    ## get the distribution of p after shocks 
    init_p_plus_shk_draws = np.sort(
        np.array(
            [np.exp(init_p) * np.exp(psi_shk) 
             for init_p in model.init_p_draws 
             for psi_shk in model.psi_shk_draws
            ]
        )
    )
    init_p_plus_shk_probs = np.ones(len(init_p_plus_shk_draws))/len(init_p_plus_shk_draws)
    shk_prbs = np.repeat(
        init_p_plus_shk_probs,
        len(model.eps_shk_draws)
    )*1/len(model.eps_shk_draws)
    
    λ = model.λ
    init_b = model.init_b
    ue_insurance = np.repeat(np.ones_like(model.eps_shk_draws),
                          len(init_p_plus_shk_probs))*model.unemp_insurance  
    init_p_draws = np.exp(np.repeat(init_p_plus_shk_draws,
                          len(model.eps_shk_draws)))
    
    ## this function starts with a state-specific initial distribution over m and p as a vector sized of (n_m x n_p) 
    NewBornDist = jump_to_grid(model,
                               np.ones_like(init_p_draws)*((1-λ)*ue_insurance+init_b/init_p_draws+model.transfer), ## initial unemployment insurance and accidental bequest transfer
                               init_p_draws,
                               shk_prbs,
                               dist_mGrid,
                               dist_pGrid)
    return NewBornDist

@njit
def initial_distribution_e(model,
                         dist_mGrid, ## new, array, grid of m for distribution 
                         dist_pGrid,  ## new, array, grid of p for distribution 
                        ):
    ## get the distribution of p after shocks 
    init_p_plus_shk_draws = np.sort(
        np.array(
            [np.exp(init_p) * np.exp(psi_shk) 
             for init_p in model.init_p_draws 
             for psi_shk in model.psi_shk_draws
            ]
        )
    )
    init_p_plus_shk_probs = np.ones(len(init_p_plus_shk_draws))/len(init_p_plus_shk_draws)
    shk_prbs = np.repeat(
        init_p_plus_shk_probs,
        len(model.eps_shk_draws)
    )*1/len(model.eps_shk_draws)
    
    λ = model.λ
    λ_SS = model.λ_SS
    init_b = model.init_b
    
    tran_shks = np.exp(np.repeat(model.eps_shk_draws,
                          len(init_p_plus_shk_probs)))
    init_p_draws = np.exp(np.repeat(init_p_plus_shk_draws,
                          len(model.eps_shk_draws)))
    ## this function starts with a state-specific initial distribution over m and p as a vector sized of (n_m x n_p) 
    NewBornDist = jump_to_grid(model,
                              (1-λ)*(1-λ_SS)*tran_shks+init_b/init_p_draws+model.transfer, ## initial transitory risks and accidental bequest transfer
                               init_p_draws,
                               shk_prbs,
                               dist_mGrid,
                               dist_pGrid)
    return NewBornDist

def AggregateDist(dist_lists,  ## size of nb markov state, each of which is sized model.L, each of which is sized n_m x n_p
              mp_pdfs_lists,  ## list of pdfs of over m and p grids given markov state and age
              mkv_dist, 
              age_dist):      ## distribution over ages
    X = 0.0
    for z in range(len(mkv_dist)):
        for k in range(len(age_dist)):
            x_flat = dist_lists[z][k].flatten()
            pdf_flat = mp_pdfs_lists[z][k].flatten()
            X+= np.dot(x_flat,pdf_flat)*age_dist[k]*mkv_dist[z]
    return X

## get the single vector of distribution 

def faltten_dist(grid_lists,      ## nb.z x T x nb x nm x np 
                 mp_pdfs_lists,   ## nb.z x T x nb x nm x np 
                 dstn,            ## size of nb.z 
                 age_dist):       ## size of T 
    mp_pdfs_lists_new = []
    for z in range(len(dstn)):
        for k in range(len(age_dist)):
            this_pdfs_lists = mp_pdfs_lists[z][k]*dstn[z]*age_dist[k]
            mp_pdfs_lists_new.append(this_pdfs_lists)
            
    grid_sort_id = np.array(grid_lists).flatten().argsort()
    grid_array = np.array(grid_lists).flatten()[grid_sort_id]
    mp_pdfs_array = np.array(mp_pdfs_lists_new).flatten()[grid_sort_id]
    
    return grid_array, mp_pdfs_array


"""
def calc_ergodic_dist(transition_matrix = None):

    '''
    Calculates the ergodic distribution for the transition_matrix, 
    here it is the distribution (before being reshaped) over normalized market resources and
    permanent income as the eigenvector associated with the eigenvalue 1.
    

    Parameters
    ----------
    transition_matrix: array 
                transition matrix whose ergordic distribution is to be solved

    Returns
    -------
    ergodic_distr: a vector array 
    The distribution is stored as a vector 
    ## reshaping it to (n_m, n_p) gives a reshaped array with the ij'th element representing
    the probability of being at the i'th point on the mGrid and the j'th
    point on the pGrid.
    '''

    #if transition_matrix == None:
    #    #transition_matrix = [self.tran_matrix]
    #    print('needs transition_matrix')
    
    eigen, ergodic_distr = sp.linalg.eigs(transition_matrix , k=1 , which='LM')  # Solve for ergodic distribution
    ergodic_distr = ergodic_distr.real/np.sum(ergodic_distr.real)   

    #vec_erg_dstn = ergodic_distr #distribution as a vector
    #erg_dstn = ergodic_distr.reshape((len(m_dist_grid_list[0]),
    #                                  len(p_dist_grid_list[0]))) # distribution reshaped into len(mgrid) by len(pgrid) array
    return ergodic_distr 
    
"""


# + code_folding=[0, 5, 17, 113, 280, 301, 341, 386]
class HH_OLG_Markov:
    """
    A class that deals with distributions of the household (HH) block
    """

    def __init__(self,
                 model = None):  

        self.model = model
        
        self.age_dist = stationary_age_dist(model.L,
                                            n = 0.0,
                                            LivPrb = model.LivPrb)
        
        self.ss_dstn = cal_ss_2markov(model.P)
        
    ## create distribution grid points 
    def define_distribution_grid(self,
                                 dist_mGrid = None, 
                                 dist_pGrid = None, 
                                 m_density = 0, 
                                 num_pointsM = 48,  
                                 num_pointsP = 50, 
                                 max_p_fac = 20.0):

            '''
            Defines the grid on which the distribution is defined. Stores the grid of market resources and permanent income as attributes of self.
            Grid for normalized market resources and permanent income may be prespecified 
            as dist_mGrid and dist_pGrid, respectively. If not then default grid is computed based off given parameters.

            Parameters
            ----------
            dist_mGrid : np.array
                    Prespecified grid for distribution over normalized market resources

            dist_pGrid : np.array
                    Prespecified grid for distribution over permanent income. 

            m_density: float
                    Density of normalized market resources grid. Default value is mdensity = 0.
                    Only affects grid of market resources if dist_mGrid=None.

            num_pointsM: float
                    Number of gridpoints for market resources grid.

            num_pointsP: float
                     Number of gridpoints for permanent income. 
                     This grid will be exponentiated by the function make_grid_exp_mult.

            max_p_fac : float
                    Factor that scales the maximum value of permanent income grid. 
                    Larger values increases the maximum value of permanent income grid.

            Returns
            -------
            List(dist_mGrid): numba typed list, sized of 1, each of which is sized n_m
            List(dist_pGrid): numba typed list, sized of T, each of which is sized n_p
            '''  
            
            ## model
            
            model = self.model 
            
            ## m distribution grid 
            if dist_mGrid == None:
                aXtra_Grid = make_grid_exp_mult(ming = model.a_grid[0], 
                                                maxg = model.a_grid[-1], 
                                                ng = num_pointsM, 
                                                timestonest = 3) #Generate Market resources grid given density and number of points

                for i in range(m_density):
                    axtra_shifted = np.delete(aXtra_Grid,-1) 
                    axtra_shifted = np.insert(axtra_shifted, 0,1.00000000e-04)
                    dist_betw_pts = aXtra_Grid - axtra_shifted
                    dist_betw_pts_half = dist_betw_pts/2
                    new_A_grid = axtra_shifted + dist_betw_pts_half
                    aXtra_Grid = np.concatenate((aXtra_Grid,new_A_grid))
                    aXtra_Grid = np.sort(aXtra_Grid)

                dist_mGrid =  [aXtra_Grid]

            else:
                dist_mGrid = [dist_mGrid] #If grid of market resources prespecified then use as mgrid

            ## permanent distribution grid 
            if dist_pGrid == None:
                dist_pGrid = [] #list of grids of permanent income    

                for i in range(model.L):
                    #Dist_pGrid is taken to cover most of the ergodic distribution
                    if model.sigma_psi!=0.0:
                        std_p = model.sigma_psi
                    else:
                        std_p = 1e-2
                    max_p = max_p_fac*std_p*(1/(1-model.LivPrb))**0.5 # Consider probability of staying alive this period
                    right_sided_grid = make_grid_exp_mult(1.0+1e-3, np.exp(max_p), num_pointsP, 2)
                    left_sided_gird = np.append(1.0/np.fliplr([right_sided_grid])[0],np.ones(1))
                    left_sided_gird = 1.0/np.fliplr([right_sided_grid])[0]
                    this_dist_pGrid = np.append(left_sided_gird,
                                                right_sided_grid) # Compute permanent income grid this period. Grid of permanent income may differ dependent on PermShkStd
                    dist_pGrid.append(this_dist_pGrid)

            else:
                dist_pGrid = [dist_pGrid] #If grid of permanent income prespecified then use as pgrid
                
            self.m_dist_grid_list = List(dist_mGrid)
            self.p_dist_grid_list = List(dist_pGrid)

            #return self.dist_mGrid, self.dist_pGrid
        
            
    ## get the distributions of each age by iterating forward over life cycle 

    def ComputeSSDist(self,
              ms_star = None,
              σs_star = None,
              m_dist_grid_list = None,
              p_dist_grid_list = None):
        
        model = self.model
        m_dist_grid_list = self.m_dist_grid_list
        p_dist_grid_list = self.p_dist_grid_list
        ss_dstn = self.ss_dstn
        age_dist = self.age_dist
        
        
        time_start = time()

        ## get the embedded list sized n_z x T x n_m x n_p

        tran_matrix_lists,c_PolGrid_list,a_PolGrid_list = calc_transition_matrix(model,
                                                                                 ms_star, ## 
                                                                                 σs_star,
                                                                                 m_dist_grid_list,
                                                                                 p_dist_grid_list,
                                                                                 fast = False)
        
        ## save the output into the model 
        self.tran_matrix_lists = tran_matrix_lists
        
        ## the initial distribution in the first period of life 
        initial_dist_u = initial_distribution_u(model,
                                              m_dist_grid_list[0],
                                              p_dist_grid_list[0])

        initial_dist_e = initial_distribution_e(model,
                                                m_dist_grid_list[0],
                                                p_dist_grid_list[0])
        
        self.initial_dist_u=initial_dist_u
        self.initial_dist_e=initial_dist_e


        ## iterate forward 

        n_m = len(m_dist_grid_list[0])


        dist_u_lists = []
        dist_e_lists = []
        dist_u_lists.append(initial_dist_u)
        dist_e_lists.append(initial_dist_e)


        mp_pdfs_lists_u_2d = []
        mp_pdfs_lists_e_2d = []
        mp_pdfs_lists_u_2d.append(initial_dist_u.reshape(n_m,-1))
        mp_pdfs_lists_e_2d.append(initial_dist_e.reshape(n_m,-1))

        mp_pdfs_lists_u = []
        mp_pdfs_lists_e = []
        mp_pdfs_lists_u.append(initial_dist_u.reshape(n_m,-1).sum(axis=1))
        mp_pdfs_lists_e.append(initial_dist_e.reshape(n_m,-1).sum(axis=1))


        ## policy grid lists 
        cp_u_PolGrid_list = []
        cp_e_PolGrid_list = []
        ap_u_PolGrid_list = []
        ap_e_PolGrid_list = []


        ## m/p distribution in the first period in life (newborns)
        this_dist_u = initial_dist_u
        this_dist_e = initial_dist_e


        ## iterate forward for all periods in life 
        for k in range(model.L-1): ## no transition matrix in the last period !
            ## uemp 
            this_dist_u = np.matmul(tran_matrix_lists[0][k],
                                    this_dist_u)
            dist_u_lists.append(this_dist_u)

            this_dist_u_2d = this_dist_u.reshape(n_m,-1)
            mp_pdfs_lists_u_2d.append(this_dist_u_2d)

            this_dist_u_2d_marginal = this_dist_u_2d.sum(axis=1)
            mp_pdfs_lists_u.append(this_dist_u_2d_marginal)

            ##emp
            this_dist_e = np.matmul(tran_matrix_lists[1][k],
                                     this_dist_e)
            dist_e_lists.append(this_dist_e)
            this_dist_e_2d = this_dist_e.reshape(n_m,-1)
            mp_pdfs_lists_e_2d.append(this_dist_e_2d)

            this_dist_e_2d_marginal = this_dist_e_2d.sum(axis=1)
            mp_pdfs_lists_e.append(this_dist_e_2d_marginal)

        for k in range(model.L):

            ## c and a for u 
            cp_PolGrid = np.multiply.outer(c_PolGrid_list[0][k],
                                           p_dist_grid_list[k])
            cp_u_PolGrid_list.append(cp_PolGrid)

            ap_PolGrid = np.multiply.outer(a_PolGrid_list[0][k],
                                           p_dist_grid_list[k])
            ap_u_PolGrid_list.append(ap_PolGrid)

            ## c and a for e 
            cp_PolGrid = np.multiply.outer(c_PolGrid_list[1][k],
                                           p_dist_grid_list[k])
            cp_e_PolGrid_list.append(cp_PolGrid)


            ap_PolGrid = np.multiply.outer(a_PolGrid_list[1][k],
                                           p_dist_grid_list[k])
            ap_e_PolGrid_list.append(ap_PolGrid)

        ## stack the distribution lists 
        dist_lists = [dist_u_lists,
                     dist_e_lists]

        ##  joint pdfs over m and p
        mp_pdfs_2d_lists = [mp_pdfs_lists_u_2d,
                           mp_pdfs_lists_e_2d]

        ## marginalized pdfs over m 
        mp_pdfs_lists = [mp_pdfs_lists_u,
                         mp_pdfs_lists_e]  ## size of n_z x model.T


        ## c policy grid 
        cp_PolGrid_list = [cp_u_PolGrid_list,
                          cp_e_PolGrid_list]

        # a policy grid 

        ap_PolGrid_list = [ap_u_PolGrid_list,
                          ap_e_PolGrid_list]


        time_end = time()
        print('time taken:'+str(time_end-time_start))
        
        self.dist_list = dist_lists
        self.mp_pdfs_2d_lists = mp_pdfs_2d_lists
        self.mp_pdfs_lists = mp_pdfs_lists
        self.ap_PolGrid_list = ap_PolGrid_list
        self.cp_PolGrid_list = cp_PolGrid_list
        
        
        ## also store flatten list of level of a and c
        self.ap_grid_dist, self.ap_pdfs_dist = faltten_dist(ap_PolGrid_list,
                                                            mp_pdfs_2d_lists,
                                                            ss_dstn,
                                                            age_dist)
            
        self.cp_grid_dist, self.cp_pdfs_dist = faltten_dist(cp_PolGrid_list,
                                                            mp_pdfs_2d_lists,
                                                            ss_dstn,
                                                            age_dist)

        #return tran_matrix_lists,dist_lists,mp_pdfs_2d_lists,mp_pdfs_lists,cp_PolGrid_list,ap_PolGrid_list
        

    ### Aggregate C or A

    def Aggregate(self):
        ## compute aggregate C 
        cp_PolGrid_list = self.cp_PolGrid_list
        ap_PolGrid_list = self.ap_PolGrid_list
        mp_pdfs_2d_lists = self.mp_pdfs_2d_lists
        ss_dstn = self.ss_dstn
        age_dist = self.age_dist


        self.C = AggregateDist(cp_PolGrid_list,
                              mp_pdfs_2d_lists,
                              ss_dstn,
                              age_dist)

        self.A = AggregateDist(ap_PolGrid_list,
                              mp_pdfs_2d_lists,
                              ss_dstn,
                              age_dist)

    ### Aggregate within age 
    
    def AggregatebyAge(self):
        
        model = self.model 
        
        cp_PolGrid_list = self.cp_PolGrid_list
        ap_PolGrid_list = self.ap_PolGrid_list
        mp_pdfs_2d_lists = self.mp_pdfs_2d_lists
        ss_dstn = self.ss_dstn
        age_dist = self.age_dist
        
        ### Aggregate distributions within age

        C_life = []
        A_life = []


        for t in range(model.L):
            age_dist_sparse = np.zeros(model.L)
            age_dist_sparse[t] = 1.0 ## a fake age distribution that gives the age t the total weight

            ## age-specific wealth 
            C_this_age = AggregateDist(cp_PolGrid_list,
                                   mp_pdfs_2d_lists,
                                   ss_dstn,
                                   age_dist_sparse)

            C_life.append(C_this_age)

            A_this_age = AggregateDist(ap_PolGrid_list,
                                  mp_pdfs_2d_lists,
                                  ss_dstn,
                                  age_dist_sparse)
            A_life.append(A_this_age)
            
        self.A_life = A_life
        self.C_life = C_life 
        
        
    ### Wealth distribution over life cycle 

    def get_lifecycle_dist(self):

        model = self.model 
        ap_PolGrid_list = self.ap_PolGrid_list
        cp_PolGrid_list = self.cp_PolGrid_list
        mp_pdfs_2d_lists = self.mp_pdfs_2d_lists
        ss_dstn = self.ss_dstn

        ## Flatten distribution by age
        ap_grid_dist_life = []
        ap_pdfs_dist_life = []
        cp_grid_dist_life = []
        cp_pdfs_dist_life = []


        for t in range(model.L):

            age_dist_sparse = np.zeros(model.L)
            age_dist_sparse[t] = 1.0

            ap_grid_dist_this_age, ap_pdfs_dist_this_age = faltten_dist(ap_PolGrid_list,
                                                                        mp_pdfs_2d_lists,
                                                                        ss_dstn,
                                                                        age_dist_sparse)

            ap_grid_dist_life.append(ap_grid_dist_this_age)
            ap_pdfs_dist_life.append(ap_pdfs_dist_this_age)

            cp_grid_dist_this_age, cp_pdfs_dist_this_age = faltten_dist(cp_PolGrid_list,
                                                                        mp_pdfs_2d_lists,
                                                                        ss_dstn,
                                                                        age_dist_sparse)

            cp_grid_dist_life.append(cp_grid_dist_this_age)
            cp_pdfs_dist_life.append(cp_pdfs_dist_this_age)


        self.ap_grid_dist_life = ap_grid_dist_life
        self.ap_pdfs_dist_life = ap_pdfs_dist_life

        self.cp_grid_dist_life = cp_grid_dist_life
        self.cp_pdfs_dist_life = cp_pdfs_dist_life
            
            
    ### Get lorenz weights  
    def Lorenz(self,
              variable='a'):
        """
        returns the lorenz weights and value 
        """
        ap_grid_dist = self.ap_grid_dist
        ap_pdfs_dist = self.ap_pdfs_dist
        cp_grid_dist = self.cp_grid_dist
        cp_pdfs_dist = self.cp_pdfs_dist
    
        
        if variable =='a':
            
            
        ## flatten the distribution of a and its corresponding pdfs 


             ## compute things needed for lorenz curve plot of asset accumulation 
            
            share_agents_ap, share_ap = lorenz_curve(ap_grid_dist,
                                                 ap_pdfs_dist,
                                                 nb_share_grid = 100)
            
            return share_agents_ap,share_ap
        
        elif variable =='c':
            
            
            ## compute things needed for lorenz curve plot of asset accumulation 

            share_agents_cp, share_cp = lorenz_curve(cp_grid_dist,
                                                 cp_pdfs_dist,
                                                 nb_share_grid = 100)
            
            return share_agents_cp,share_cp


# + code_folding=[0]
## testing of the household class 

HH = HH_OLG_Markov(model=lc_mkv)

## Markov transition matrix 

print("markov state transition matrix: \n",lc_mkv.P)
print('steady state of markov state:\n',HH.ss_dstn)


## computing transition matrix 
n_m = 30
n_p = 40

HH.define_distribution_grid(num_pointsM = n_m, 
                            num_pointsP = n_p)
HH.ComputeSSDist(ms_star = ms_star_mkv,
                  σs_star = σs_star_mkv)


# + code_folding=[]
## plot the initial distribution in the first period of life 

plt.title('Initial distributions over m and p given u')
plt.spy(HH.initial_dist_u.reshape(n_m,-1),
       markersize = 2)
plt.xlabel('p')
plt.ylabel('m')

# + code_folding=[]
## plot the initial distribution in the first period of life 

plt.title('Initial distributions over m and p given e')
plt.spy(HH.initial_dist_e.reshape(n_m,-1),
       markersize = 2)
plt.xlabel('p')
plt.ylabel('m')
# -

HH.Aggregate()
print('aggregate consumption under stationary distribution:', str(HH.C))
print('aggregate savings under stationary distribution:', str(HH.A))

# ### Stationary wealth/consumption distribution

share_agents_cp,share_cp = HH.Lorenz(variable='c')
share_agents_ap,share_ap = HH.Lorenz(variable='a')

# + code_folding=[0]
## get the wealth distribution from SCF (net worth)

SCF2016 = pd.read_stata('rscfp2016.dta')
SCF2016 = SCF2016.drop_duplicates(subset=['yy1'])

SCF_wealth, SCF_weights = np.array(SCF2016['networth']), np.array(SCF2016['wgt'])

## get the lorenz curve weights from SCF 
SCF_wealth_sort_id = SCF_wealth.argsort()
SCF_wealth_sort = SCF_wealth[SCF_wealth_sort_id]
SCF_weights_sort = SCF_weights[SCF_wealth_sort_id]
SCF_weights_sort_norm = SCF_weights_sort/SCF_weights_sort.sum()

SCF_share_agents_ap, SCF_share_ap = lorenz_curve(SCF_wealth_sort,
                                                 SCF_weights_sort_norm,
                                                 nb_share_grid = 200)



# + code_folding=[]
## Lorenz curve of steady state wealth distribution

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(share_agents_cp,share_cp, 'r--',label='Lorenz curve of consumption')
ax.plot(share_agents_cp,share_agents_cp, 'k-',label='equality curve')
ax.legend()
plt.xlim([0,1])
plt.ylim([0,1])
plt.savefig('../Graphs/model/lorenz_c_test.png')

## Lorenz curve of steady state wealth distribution

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(share_agents_ap,share_ap, 'r--',label='Lorenz curve of wealth: model')
ax.plot(SCF_share_agents_ap,SCF_share_ap, 'b-.',label='Lorenz curve of wealth: SCF')
ax.plot(share_agents_ap,share_agents_ap, 'k-',label='equality curve')
ax.legend()
plt.xlim([0,1])
plt.ylim([0,1])
plt.savefig('../Graphs/model/lorenz_a_test.png')

# + code_folding=[]
## Wealth distribution 

ap_grid_dist = HH.ap_grid_dist
ap_pdfs_dist = HH.ap_pdfs_dist
cp_grid_dist = HH.cp_grid_dist
cp_pdfs_dist = HH.cp_pdfs_dist


fig, ax = plt.subplots(figsize=(6,4))
ax.set_title('Wealth distribution')
ax.plot(np.log(ap_grid_dist+0.0000000001), 
         ap_pdfs_dist)
ax.set_xlim(-30,10)

ax.set_xlabel(r'$a$')
ax.set_ylabel(r'$prob(a)$')
fig.savefig('../Graphs/model/distribution_a_test.png')

fig, ax = plt.subplots(figsize=(6,4))
ax.set_title('Consumption distribution')
ax.plot(np.log(cp_grid_dist), 
         cp_pdfs_dist)
ax.set_xlabel(r'$c$')
ax.set_ylabel(r'$prob(a)$')
ax.set_xlim(-15,10)
fig.savefig('../Graphs/model/distribution_c_test.png')
# -

# ### Life-cycle profile and wealth distribution

# +
import pandas as pd
SCF_profile = pd.read_pickle('data/SCF_age_profile.pkl')

SCF_profile['mv_wealth'] = SCF_profile['av_wealth'].rolling(3).mean()

# + code_folding=[]
HH.AggregatebyAge()

A_life = HH.A_life
C_life = HH.C_life


# + code_folding=[11]
## plot life cycle profile

age_lc = SCF_profile.index

fig, ax = plt.subplots(figsize=(10,5))
plt.title('Life cycle profile of wealth')
ax.plot(age_lc[1:],
       np.log(A_life),
       'r-o',
       label='model')

ax.vlines(lc_mkv.T+25,
          np.min(np.log(A_life)),
          np.max(np.log(A_life)),
          color='k',
          label='retirement'
         )
#ax.set_ylim([-2.0,2.0])

ax2 = ax.twinx()
ax2.set_ylim([10.5,15])
ax2.bar(age_lc[1:],
        np.log(SCF_profile['mv_wealth'][1:]),
       label='SCF (RHS)')

#ax2.plot(age,
#        C_life,
#        'b--',
#        label='consumption (RHS)')

ax.set_xlabel('Age')
ax.set_ylabel('Log wealth in model')
ax2.set_ylabel('Log wealth SCF')
ax.legend(loc=1)
ax2.legend(loc=2)
fig.savefig('../Graphs/model/life_cycle_a_test.png')

# + code_folding=[0]
## get the within-age distribution 

HH.get_lifecycle_dist()

ap_grid_dist_life,ap_pdfs_dist_life = HH.ap_grid_dist_life,HH.ap_pdfs_dist_life
cp_grid_dist_life,cp_pdfs_dist_life = HH.cp_grid_dist_life,HH.cp_pdfs_dist_life

# + code_folding=[0]
## create the dataframe to plot distributions over the life cycle 
ap_pdfs_life = pd.DataFrame(ap_pdfs_dist_life).T
cp_pdfs_life = pd.DataFrame(cp_pdfs_dist_life).T

#ap_pdfs_life.index = np.log(ap_grid_dist_life[0]+1e-4)

ap_range = list(ap_pdfs_life.index)
cp_range = list(cp_pdfs_life.index)

# + code_folding=[1, 2, 11, 21]
joy = False
if joy == True:
    fig, axes = joypy.joyplot(ap_pdfs_life, 
                              kind="values", 
                              x_range=ap_range,
                              figsize=(6,10),
                              title="Wealth distribution over life cycle",
                             colormap=cm.winter)
    fig.savefig('../Graphs/model/life_cycle_distribution_a_test.png')
    #axes[-1].set_xticks(a_values);

    fig, axes = joypy.joyplot(cp_pdfs_life, 
                              kind="values", 
                              x_range=cp_range,
                              figsize=(6,10),
                              title="Consumption distribution over life cycle",
                             colormap=cm.winter)
    fig.savefig('../Graphs/model/life_cycle_distribution_c_test.png')

    #axes[-1].set_xticks(a_values);
    
else:
    pass


# -

# ### General Equilibrium 

# + code_folding=[0, 5, 24, 159]
class Market_OLG_mkv:
    """
    A class of the market
    """

    def __init__(self,
                 households=None,
                 production=None):  

        self.households = households   ## HH block 
        self.model = households.model  ## life-cycle model 
        
        ### normalize A based on the model parameters first
        ss_dstn = households.ss_dstn
        age_dist = households.age_dist 
        T =  self.model.T
        L_ss = np.sum(age_dist[:T-1])*ss_dstn[1] ## employment fraction for working age population
        self.households.emp_ss = L_ss
        production.normlize_Z(N_ss = L_ss)
        self.production = production   ## Produciton function 


    ## stationary asset demand for a given capital stock/factor price

    def StE_K_d(self,
                K_s,   ## given asset supply 
                dstn):  ## distribution between emp and unemp 
        """
        Given a proposed capital stock/asset supply ,
        this function generates the stationary asset demands 

        """
        model = self.model 
        households = self.households
        production = self.production 
        emp_now = households.emp_ss
        age_dist = households.age_dist
        T = self.model.T ## retirement age 
        ################################
        ## Step 0. Parameterize the model 
        ################################
        ## get the L based on current employed fraction
        uemp_now,emp_now = dstn[0]*np.sum(age_dist[:T-1]),dstn[1]*np.sum(age_dist[:T-1])
        print('Labor force',str(emp_now))


        ## obtaine factor prices from FOC of firms 
        
        production.K = K_s
        production.L = emp_now
        
        #print(nb.typeof(one_economy.K))
        print('Capital stock',str(K_s))
        W,R = production.YL(),production.R()
        print('Wage rate',str(W))
        print('Real interest rate',str(R))
        
        ##################################
        model.W, model.R = W,R
        ##################################

        ## stable age distribution 
        age_dist = households.age_dist
        #stationary_age_dist(model.L,#n = 0.0,#LivPrb =model.LivPrb)

        ## obtain tax rate from the government budget balance 

        model.λ = unemp_insurance2tax(model.unemp_insurance,
                                     uemp_now)
        print('Tax rate',str(model.λ))

        ## obtain social security rate balancing the SS replacement ratio 

        model.λ_SS = SS2tax(model.pension, ## social security /pension replacement ratio 
                            model.T,  ## retirement years
                            age_dist,  ## age distribution in the economy 
                            model.G,         ## permanent growth fractor lists over cycle
                            emp_now)

        print('Social security tax rate',str(model.λ_SS))

        ################################
        ## Step 1. Solve the model 
        ################################

        ## terminal period solution
        m_init,σ_init = model.terminal_solution()

        ## solve the model 
        ms_star, σs_star = solve_model_backward_iter(model,
                                                     m_init,
                                                     σ_init)

        ################################
        ## Step 2. StE distribution
        ################################


        ## accidental transfers 
        #model.init_b = init_b

        ## Get the StE K_d
        ## get the transition matrix and policy grid 
        
        n_m = 40
        n_p = 50
        households.define_distribution_grid(num_pointsM = n_m, 
                                            num_pointsP = n_p)
        households.ComputeSSDist(ms_star = ms_star,
                                 σs_star = σs_star)

        households.Aggregate()
        #m_dist_grid_list,p_dist_grid_list = define_distribution_grid(model,
        #                                                         num_pointsM = n_m,
        #                                                         num_pointsP = n_p)

        #tran_matrix_lists, dist_lists,mp_pdfs_2d_lists,mp_pdfs_lists,cp_PolGrid_list,ap_PolGrid_list = SSDist(model,
        #                                                                                                      as_star,
        #                                                                                                      σs_star,
        #                                                                                                      m_dist_grid_list,
        #                                                                                                   p_dist_grid_list)    


        #A = Aggregate(ap_PolGrid_list,
        #              mp_pdfs_2d_lists,
        #              ss_dstn,
        #              age_dist)

        K_d = households.A*model.W  ## no population growth otherwise diluted by (1+n)

        ## realized accidental transfers from age 2 to L

        #ap_PolGrid_list_old = [ap_PolGrid_list[0][1:],ap_PolGrid_list[1][1:]]
        #mp_pdfs_2d_lists_old = [mp_pdfs_2d_lists[0][1:],mp_pdfs_2d_lists[1][1:]]
        #age_dist_old =  age_dist[1:]
        #A_old = Aggregate(ap_PolGrid_list_old,
        #                  mp_pdfs_2d_lists_old,
        #                  ss_dstn,
        #                 age_dist_old)*model.W 

        #init_b_out = model.bequest_ratio*(1-model.LivPrb)*A_old*(1-age_dist[0])*model.R/age_dist[0]

        print('Induced capital stock',str(K_d))
        #print('Induced  bequest',str(init_b_out))

        return K_d
    
    def get_equilibrium_k(self):
        ss_dstn = self.households.ss_dstn
        
        ## function to solve the equilibrium 
        eq_func = lambda K: self.StE_K_d(K_s = K,
                                         dstn = ss_dstn)
        ## solve the fixed point 
        K_eq = op.fixed_point(eq_func,
                              x0 = 7.3)
        
        self.K_eq = K_eq
    
    def get_equilibrium_dist(self):
        
        households = self.households 
        model = self.model 
        
        ### get equilibrium values 
        K_eq = self.K_eq 
        L_ss = households.emp_ss
        
        ## compute factor prices in StE
        production.K = K_eq
        production.L = L_ss
        
        print('SS Capital stock',str(K_eq))
        W_eq,R_eq = production.YL(),production.R()
        print('SS Wage Rate',str(W_eq))
        print('SS Real interest rate',str(R_eq))

        ## get the distribution under SS
        model.W,model.R = W_eq,R_eq

        ## solve the model again 

        ## terminal period solution
        m_init,σ_init = model.terminal_solution()

        ms_star, σs_star = solve_model_backward_iter(model,
                                                     m_init,
                                                     σ_init)

        households.define_distribution_grid(num_pointsM = 40, 
                                            num_pointsP = 50)
        households.ComputeSSDist(ms_star = ms_star,
                                 σs_star = σs_star)

        ## operation for the StE, such as aggregation
        households.Aggregate()
        households.AggregatebyAge()
        
        
        self.households = households


# + code_folding=[0, 2]
## initialize a market and solve the equilibrium 

production = CDProduction(α = production_paras['α'],
                          δ = production_paras['δ'],
                          target_KY = production_paras['K2Y ratio'],
                         target_W = production_paras['W']) 

market_OLG_mkv = Market_OLG_mkv(households = HH,
                                production = production)

market_OLG_mkv.get_equilibrium_k()

# -

market_OLG_mkv.get_equilibrium_dist()

# + code_folding=[0]
## plot life cycle profile

age_lc = SCF_profile.index

fig, ax = plt.subplots(figsize=(10,5))
plt.title('Life cycle profile of wealth')
ax.plot(age_lc[:-2],
        np.log(market_OLG_mkv.households.A_life)[:-1],
       'r-o',
       label='model')


ax2 = ax.twinx()
ax2.set_ylim([10.5,15])
ax2.vlines(lc_mkv.T+25,
          10.5,
          15,
          color='k',
          label='retirement')

ax2.bar(age_lc,
        np.log(SCF_profile['mv_wealth']),
       #'k--',
       label='SCF (RHS)')

#ax2.plot(age,
#        C_life,
#        'b--',
#        label='consumption (RHS)')

ax.set_xlabel('Age')
ax.set_ylabel('Log wealth')
ax2.set_ylabel('Log wealth SCF')
ax.legend(loc=1)
ax2.legend(loc=2)
fig.savefig('../Graphs/model/life_cycle_a_eq.png')

# + code_folding=[]
## compute things needed for lorenz curve plot of asset accumulation 

share_agents_ap, share_ap = market_OLG_mkv.households.Lorenz(variable='a')

## Lorenz curve of steady state wealth distribution

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(share_agents_ap,share_cp, 'r--',label='Lorenz curve of level of wealth')
ax.plot(SCF_share_agents_ap,SCF_share_ap, 'b-.',label='Lorenz curve from SCF')
ax.plot(share_agents_ap,share_agents_ap, 'k-',label='equality curve')
ax.legend()
plt.xlim([0,1])
plt.ylim([0,1])
fig.savefig('../Graphs/model/lorenz_curve_a_eq.png')



# + code_folding=[0]
## Wealth distribution 

fig, ax = plt.subplots(figsize=(6,4))
ax.set_title('Wealth distribution')
ax.plot(np.log(market_OLG_mkv.households.ap_grid_dist+0.0000000001), 
         market_OLG_mkv.households.ap_pdfs_dist)
ax.set_xlim((-15,15))
ax.set_xlabel(r'$a$')
ax.set_ylabel(r'$prob(a)$')
fig.savefig('../Graphs/model/distribution_a_eq.png')

fig, ax = plt.subplots(figsize=(6,4))
ax.set_title('Consumption distribution')
ax.plot(np.log(market_OLG_mkv.households.cp_grid_dist), 
         market_OLG_mkv.households.cp_pdfs_dist)
ax.set_xlabel(r'$c$')
ax.set_xlim((-20,20))
ax.set_ylabel(r'$prob(a)$')
fig.savefig('../Graphs/model/distribution_c_eq.png')
# -
# ## Compare between the objective and subjective model 

# + code_folding=[]
## create a new subjective household block 

HH_sub = HH_OLG_Markov(model=lc_mkv_sub)

HH_sub.define_distribution_grid(num_pointsM = n_m, 
                            num_pointsP = n_p)
HH_sub.ComputeSSDist(ms_star = ms_star_mkv_sub,
                      σs_star = σs_star_mkv_sub)


HH_sub.Aggregate()
print('aggregate consumption under stationary distribution:', str(HH_sub.C))
print('aggregate savings under stationary distribution:', str(HH_sub.A))

share_agents_cp_sub,share_cp_sub = HH_sub.Lorenz(variable='c')
share_agents_ap_sub,share_ap_sub = HH_sub.Lorenz(variable='a')

## Lorenz curve of steady state wealth distribution

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(share_agents_cp_sub,share_cp_sub, 'r--',label='Lorenz curve of consumption')
ax.plot(share_agents_cp_sub,share_agents_cp_sub, 'k-',label='equality curve')
ax.legend()
plt.xlim([0,1])
plt.ylim([0,1])
plt.savefig('../Graphs/model/lorenz_c_sub_test.png')

## Lorenz curve of steady state wealth distribution

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(share_agents_ap,share_ap, 'r--',label='Objective model')
ax.plot(share_agents_ap_sub,share_ap_sub, 'g-',label='Subjective model')
ax.plot(SCF_share_agents_ap,SCF_share_ap, 'b-.',label='SCF')
ax.plot(share_agents_ap,share_agents_ap, 'k-',label='equality curve')
ax.legend()
plt.xlim([0,1])
plt.ylim([0,1])
plt.savefig('../Graphs/model/lorenz_a_sub_test.png')

## life cycle 

HH_sub.AggregatebyAge()

A_life_sub = HH_sub.A_life
C_life_sub = HH_sub.C_life


age_lc = SCF_profile.index

fig, ax = plt.subplots(figsize=(10,5))
plt.title('Life cycle profile of wealth')
ax.plot(age_lc[1:],
       np.log(A_life),
       'r-o',
       label='objective model')

ax.plot(age_lc[1:],
       np.log(A_life_sub),
       'g-o',
       label='subjective model')

ax.vlines(lc_mkv.T+25,
          np.min(np.log(A_life)),
          np.max(np.log(A_life)),
          color='k',
          label='retirement'
         )
#ax.set_ylim([-5.0,2.0])

ax2 = ax.twinx()
ax2.set_ylim([10.5,15])
ax2.bar(age_lc[1:],
        np.log(SCF_profile['mv_wealth'][1:]),
       label='SCF (RHS)')

#ax2.plot(age,
#        C_life,
#        'b--',
#        label='consumption (RHS)')

ax.set_xlabel('Age')
ax.set_ylabel('Log wealth in model')
ax2.set_ylabel('Log wealth SCF')
ax.legend(loc=1)
ax2.legend(loc=2)
fig.savefig('../Graphs/model/life_cycle_a_sub_test.png')


## general equilibrium 


market_OLG_mkv_sub = Market_OLG_mkv(households = HH_sub,
                                    production = production)

market_OLG_mkv_sub.get_equilibrium_k()
market_OLG_mkv_sub.get_equilibrium_dist()



## plot life cycle profile

age_lc = SCF_profile.index

fig, ax = plt.subplots(figsize=(10,5))
plt.title('Life cycle profile of wealth')
ax.plot(age_lc[:-2],
        np.log(market_OLG_mkv.households.A_life)[:-1],
       'r-o',
       label='objective model')

ax.plot(age_lc[:-2],
        np.log(market_OLG_mkv_sub.households.A_life)[:-1],
       'g-o',
       label='subjective model')


ax2 = ax.twinx()
ax2.set_ylim([10.5,15])
ax2.vlines(lc_mkv.T+25,
          10.5,
          15,
          color='k',
          label='retirement')

ax2.bar(age_lc,
        np.log(SCF_profile['mv_wealth']),
       #'k--',
       label='SCF (RHS)')

#ax2.plot(age,
#        C_life,
#        'b--',
#        label='consumption (RHS)')

ax.set_xlabel('Age')
ax.set_ylabel('Log wealth')
ax2.set_ylabel('Log wealth SCF')
ax.legend(loc=1)
ax2.legend(loc=2)
fig.savefig('../Graphs/model/life_cycle_a_sub_eq.png')

#######################################
## lorenz curve 
###################################

## compute things needed for lorenz curve plot of asset accumulation 

share_agents_ap_sub, share_ap_sub = market_OLG_mkv_sub.households.Lorenz(variable='a')

## Lorenz curve of steady state wealth distribution

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(share_agents_ap,share_cp, 'r--',label='Objective model')
ax.plot(share_agents_ap_sub,share_ap_sub, 'g--',label='Subjective model')

ax.plot(SCF_share_agents_ap,SCF_share_ap, 'b-.',label='SCF')
ax.plot(share_agents_ap,share_agents_ap, 'k-',label='equality curve')
ax.legend()
plt.xlim([0,1])
plt.ylim([0,1])
fig.savefig('../Graphs/model/lorenz_curve_a_sub_eq.png')


# -


