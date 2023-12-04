# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Aggregate StE of a life-cycle economy (with heterogeneous types of agents)
#
# - author: Tao Wang
# - this is a companion notebook to the paper "Perceived income risks"
# - Note: set types accordingly, depending on the heterogeneous types.

from psutil import Process 
import numpy as np
import pandas as pd
from interpolation import interp, mlinterp
from numba import njit
from numba.typed import List
import matplotlib as mp
import matplotlib.pyplot as plt
# %matplotlib inline
from time import time
from Utility import make_grid_exp_mult
import scipy.optimize as op
from matplotlib import cm
import joypy
from copy import deepcopy 
from copy import copy
from Utility import cal_ss_2markov,lorenz_curve, gini, h2m_ratio
from Utility import mean_preserving_spread
from Utility import jump_to_grid,jump_to_grid_fast,gen_tran_matrix,gen_tran_matrix_fast
import pickle
from scipy import sparse 

# + code_folding=[]
## figure plotting configurations

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
plt.rc('lines', linewidth=4)
# -

# ### The Life-cycle Model Class and the Solver

# + code_folding=[]
from SolveLifeCycle import LifeCycle, solve_model_backward_iter,compare_2solutions


# -

# ### Initialize the model

from PrepareParameters import life_cycle_paras_q as lc_paras_Q
from PrepareParameters import life_cycle_paras_y as lc_paras_Y
## make a copy of the imported parameters 
lc_paras_y = copy(lc_paras_Y)
lc_paras_q = copy(lc_paras_Q)

##############
## low_beta for liquid 
lc_paras_y['β'] = 0.96
lc_paras_y['β_h'] = 0.98
###############

print(lc_paras_y)


# + code_folding=[]
## a deterministic income profile 

## income profile 
YPath = np.cumprod(lc_paras_y['G'])

plt.title('Deterministic Life-cycle Income Profile \n')
plt.plot(YPath,'ko-')
plt.xlabel('Age')
plt.ylabel(r'$\hat Y$')
# -

# ### Solve the model with a Markov state: unemployment and employment 

# + code_folding=[7]
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

    lc_mkv_paras = { 
        ## primitives
                   'ρ':lc_paras['ρ'],     ## relative risk aversion  
                   'β':lc_paras['β'],     ## discount factor
                   'borrowing_cstr':True,
                   'adjust_prob':1.0,
        
        ## prices 
                   'R':lc_paras['R'],           ## interest factor
                   'W':lc_paras['W'],           ## Wage rate
        
        ## life cycle 
                   'T':lc_paras['T'],
                   'L':lc_paras['L'],
                   'G':lc_paras['G'],
                   'LivPrb':lc_paras['LivPrb'],       ## living probability 
        
        ## income risks 
                   'x':0.0,
                   'b_y':0.0,
                   'sigma_psi':lc_paras['σ_ψ'],
                   'sigma_eps':lc_paras['σ_θ'],
                   'ue_markov':True,
                   'P':lc_paras['P'],
                   'U':lc_paras['U'],
                   'z_val':lc_paras['z_val'], ## markov state from low to high 
                   'sigma_psi_2mkv':lc_paras['σ_ψ_2mkv'],  ## permanent risks in 2 markov states
                   'sigma_eps_2mkv':lc_paras['σ_θ_2mkv'],  ## transitory risks in 2 markov states
                   'sigma_psi_true':lc_paras['σ_ψ'], ## true permanent
                   'sigma_eps_true':lc_paras['σ_θ'], ## true transitory
        
        ## initial conditions 
                    'sigma_p_init':lc_paras['σ_ψ_init'],
                    'init_b':lc_paras['init_b'],

        ## policy 
                   'unemp_insurance':lc_paras['unemp_insurance'],
                   'pension':lc_paras['pension'], ## pension
                   'λ':lc_paras['λ'],  ## tax rate
                   'λ_SS':lc_paras['λ_SS'], ## social tax rate
                   'transfer':lc_paras['transfer'],  ## transfer 
                   'bequest_ratio':lc_paras['bequest_ratio'],
         ## solutions 
                   'shock_draw_size':10.0,
                   'grid_max':30,
         ## bequest
        ##############################
                    'q':1.0,
                    'ρ_b':lc_paras['ρ']}
    ####################################
    
    ## initialize the model with calibrated parameters 
    
    lc_mkv = LifeCycle(**lc_mkv_paras)    


# + code_folding=[2, 29, 72, 120, 171]
## help functions that make a list of consumer types different by parameters. 

def make_1dtypes(by,
                 vals,
                 subjective=False):
    """
    input
    ======
    model: lifecycle class instance
    by: a string, primitive variable's name on which types are made 
    vals: an array, the values taken by different types 
    output
    =======
    types: a list of different types of life-cycle consumers
    
    """
    
    nb_types = len(vals)
    types = []
    
    for i in range(nb_types):
        paras_this_type = copy(lc_mkv_paras)
        paras_this_type[by] = vals[i]
        if subjective:
            paras_this_type['subjective'] = True
        this_type = LifeCycle(**paras_this_type)
        types.append(this_type)
    return types 

def make_2dtypes(by_list,
                 vals_list,
                 perfect_correlated = False,
                subjective = False):
    
    """
    input
    ======
    model: lifecycle class instance
    by_list: a list of strings, primitive variable's name on which types are made 
    vals: a list of arrays, the values taken by different types 
    output
    =======
    types: a list of different types of life-cycle consumers
    
    """
    
    n_type1 = len(vals_list[0])
    n_type2 = len(vals_list[1])
    
    types = []
    
    if perfect_correlated:
         for x in range(n_type1):
                paras_this_type = copy(lc_mkv_paras)
                paras_this_type[by_list[0]] = vals_list[0][x]
                paras_this_type[by_list[1]] = vals_list[1][x]
                if subjective:
                    paras_this_type['subjective'] = True
                this_type = LifeCycle(**paras_this_type)
                types.append(this_type)     
    else:
        for x in range(n_type1):
            for y in range(n_type2):
                paras_this_type = copy(lc_mkv_paras)
                paras_this_type[by_list[0]] = vals_list[0][x]
                paras_this_type[by_list[1]] = vals_list[1][y]
                if subjective:
                    paras_this_type['subjective'] = True
                this_type = LifeCycle(**paras_this_type)
                types.append(this_type)
    return types 

def make_3dtypes(by_list,
                 vals_list,
                 perfect_correlated = False,
                subjective = False):
    
    """
    input
    ======
    model: lifecycle class instance
    by_list: a list of strings, primitive variable's name on which types are made 
    vals: a list of arrays, the values taken by different types 
    output
    =======
    types: a list of different types of life-cycle consumers
    
    """
    
    n_type1 = len(vals_list[0])
    n_type2 = len(vals_list[1])
    n_type3 = len(vals_list[2])


    types = []
    
    if perfect_correlated:
         for x in range(n_type1):
                paras_this_type = copy(lc_mkv_paras)
                paras_this_type[by_list[0]] = vals_list[0][x]
                paras_this_type[by_list[1]] = vals_list[1][x]
                paras_this_type[by_list[2]] = vals_list[2][x]
                if subjective:
                    paras_this_type['subjective'] = True
                this_type = LifeCycle(**paras_this_type)
                types.append(this_type)     
    else:
        for x in range(n_type1):
            for y in range(n_type2):
                for z in range(n_type3):
                    paras_this_type = copy(lc_mkv_paras)
                    paras_this_type[by_list[0]] = vals_list[0][x]
                    paras_this_type[by_list[1]] = vals_list[1][y]
                    paras_this_type[by_list[2]] = vals_list[2][z]
                    if subjective:
                        paras_this_type['subjective'] = True
                    this_type = LifeCycle(**paras_this_type)
                    types.append(this_type)
    return types 

def make_4dtypes(by_list,
                 vals_list,
                 perfect_correlated = False,
                subjective = False):
    
    """
    input
    ======
    model: lifecycle class instance
    by_list: a list of strings, primitive variable's name on which types are made 
    vals: a list of arrays, the values taken by different types 
    output
    =======
    types: a list of different types of life-cycle consumers
    
    """
    
    n_type1 = len(vals_list[0])
    n_type2 = len(vals_list[1])
    n_type3 = len(vals_list[2])
    n_type4 = len(vals_list[3])

    types = []
    
    if perfect_correlated:
         for x in range(n_type1):
                paras_this_type = copy(lc_mkv_paras)
                paras_this_type[by_list[0]] = vals_list[0][x]
                paras_this_type[by_list[1]] = vals_list[1][x]
                paras_this_type[by_list[2]] = vals_list[2][x]
                paras_this_type[by_list[3]] = vals_list[3][x]
                if subjective:
                    paras_this_type['subjective'] = True
                this_type = LifeCycle(**paras_this_type)
                types.append(this_type)     
    else:
        for x in range(n_type1):
            for y in range(n_type2):
                for z in range(n_type3):
                    for h in range(n_type4):
                        paras_this_type = copy(lc_mkv_paras)
                        paras_this_type[by_list[0]] = vals_list[0][x]
                        paras_this_type[by_list[1]] = vals_list[1][y]
                        paras_this_type[by_list[2]] = vals_list[2][z]
                        paras_this_type[by_list[3]] = vals_list[3][h]
                        if subjective:
                            paras_this_type['subjective'] = True
                        this_type = LifeCycle(**paras_this_type)
                        types.append(this_type)
    return types 

def make_5dtypes(by_list,
                 vals_list,
                 perfect_correlated = False,
                subjective = False):
    
    """
    input
    ======
    model: lifecycle class instance
    by_list: a list of strings, primitive variable's name on which types are made 
    vals: a list of arrays, the values taken by different types 
    output
    =======
    types: a list of different types of life-cycle consumers
    
    """
    
    n_type1 = len(vals_list[0])
    n_type2 = len(vals_list[1])
    n_type3 = len(vals_list[2])
    n_type4 = len(vals_list[3])
    n_type5 = len(vals_list[4])



    types = []
    
    if perfect_correlated:
         for x in range(n_type1):
                paras_this_type = copy(lc_mkv_paras)
                paras_this_type[by_list[0]] = vals_list[0][x]
                paras_this_type[by_list[1]] = vals_list[1][x]
                paras_this_type[by_list[2]] = vals_list[2][x]
                paras_this_type[by_list[3]] = vals_list[3][x]
                paras_this_type[by_list[4]] = vals_list[4][x]
                if subjective:
                    paras_this_type['subjective'] = True
                this_type = LifeCycle(**paras_this_type)
                types.append(this_type)     
    else:
        for x in range(n_type1):
            for y in range(n_type2):
                for z in range(n_type3):
                    for h in range(n_type4):
                        for s in range(n_type5):
                            paras_this_type = copy(lc_mkv_paras)
                            paras_this_type[by_list[0]] = vals_list[0][x]
                            paras_this_type[by_list[1]] = vals_list[1][y]
                            paras_this_type[by_list[2]] = vals_list[2][z]
                            paras_this_type[by_list[3]] = vals_list[3][h]
                            paras_this_type[by_list[4]] = vals_list[4][s]
                            if subjective:
                                paras_this_type['subjective'] = True
                            this_type = LifeCycle(**paras_this_type)
                            types.append(this_type)
    return types 

# + code_folding=[]
## load parameters estimated 

PR_est = pickle.load(open('./parameters/PR_est.pkl','rb'))

## PR
mu_PR_est_SCE = PR_est['mu_pr']
sigma_PR_est_SCE = PR_est['sigma_pr']

mu_exo_est_SCE = PR_est['mu_exp']
sigma_exp_est_SCE = PR_est['sigma_exp']
loc_exp_est_SCE = PR_est['loc_exp']

sigma_xi_SCE = PR_est['sigma_xi']
std_exp_SCE = PR_est['std_exp']

mu_trans_U2U_SCE = PR_est['mu_trans_U2U']
sigma_trans_U2U_SCE = PR_est['sigma_trans_U2U']
loc_trans_U2U_SCE = PR_est['loc_trans_U2U']

mu_trans_E2E_SCE = PR_est['mu_trans_E2E']
sigma_trans_E2E_SCE = PR_est['sigma_trans_E2E']
loc_trans_E2E_SCE = PR_est['loc_trans_E2E']


# + code_folding=[]
## make grids of sigma_psi and sigma_eps

from resources_jit import LogNormal as lognorm

## use the equalprobable distribution function 

nb_grid = 2

PRs_grid_dist = lognorm(mu_PR_est_SCE,
                        sigma_PR_est_SCE,
                        100000,
                        nb_grid)

PRs_grid = PRs_grid_dist.X
        
p2t_ratio = (lc_paras['σ_ψ']/lc_paras['σ_θ'])**2 

sigma_eps_grid = np.sqrt(PRs_grid/(p2t_ratio+1))
sigma_psi_grid = np.sqrt(p2t_ratio*PRs_grid/(p2t_ratio+1))

sigma_xi_psi = np.sqrt(sigma_xi_SCE**2*p2t_ratio/(1+p2t_ratio))
sigma_xi_eps = np.sqrt(sigma_xi_SCE**2/(1+p2t_ratio))

print('Equally probable sigma_psi grid',str(sigma_psi_grid))
print('Equally probable sigma_eps grid',str(sigma_eps_grid))

print('Heterogeneity in expected growth rate is', str(std_exp_SCE))
print('Unobserved permanent heterogeneity is',str(sigma_xi_psi))
print('Unobserved transitory heterogeneity',str(sigma_xi_eps))


exp_grid_dist = lognorm(mu_exo_est_SCE,
                        sigma_exp_est_SCE,
                        100000,
                        nb_grid)
                        
exp_grid = exp_grid_dist.X+loc_exp_est_SCE    

print('Equally probable exp grid is', str(exp_grid))

# + code_folding=[]
## make grids for U2U and E2E 

prob_func = lambda y: np.exp(y)/(1+np.exp(y))


U2U_grid_dist = lognorm(mu_trans_U2U_SCE,
                        sigma_trans_U2U_SCE,
                        100000,
                        nb_grid)

U2U_grid = prob_func(U2U_grid_dist.X + loc_trans_U2U_SCE)


E2E_grid_dist = lognorm(mu_trans_E2E_SCE,
                        sigma_trans_E2E_SCE,
                        100000,
                        nb_grid)

E2E_grid = prob_func(E2E_grid_dist.X + loc_trans_E2E_SCE)


print('Equally probable U2U grid is', str(U2U_grid))

print('Equally probable E2E grid is', str(E2E_grid))


# + code_folding=[]
## make deterministic profiles 

std_exp_SCE = 0.04
G_low_work = lc_paras['G'][:lc_paras['T']]-std_exp_SCE
G_low = np.concatenate([G_low_work,np.ones(lc_paras['L']-lc_paras['T'])])
G_low[lc_paras['T']] = 1/np.cumprod(G_low_work)[-1]

G_high_work = lc_paras['G'][:lc_paras['T']]+std_exp_SCE
G_high = np.concatenate([G_high_work,np.ones(lc_paras['L']-lc_paras['T'])])
G_high[lc_paras['T']] = 1/np.cumprod(G_high_work)[-1]

G_types = [G_low,
           lc_paras['G'],
           G_high]   

plt.title('Heterogeneous wage growth rates')
plt.plot(np.cumprod(G_types[0]),
         'k--',
         label='low growth')
plt.plot(np.cumprod(G_types[1]),
         'r-',
         label='mean growth')
plt.plot(np.cumprod(G_types[2]),
         'b-.',
         label='high growth')
plt.legend(loc=0)
plt.xlabel('Work age')
plt.ylabel('Wage relative to 25yr old')
plt.savefig('../Graphs/sce/hetero_growth_rates_life_cycle.pdf')


## tempoary 


G_types = [G_low,
           #lc_paras['G'],
           G_high]   

# + code_folding=[]
from Utility import average_heterogeneity,stationary_age_dist

# + code_folding=[0, 29, 36, 46, 56, 66, 76, 87, 98, 109, 117]
## create a list of consumer types with different risk parameters and any other primitive parameters 

sigma_psi_types = np.array(sigma_psi_grid)
sigma_eps_types = np.array(sigma_eps_grid)
U2U_types = U2U_grid
E2E_types = E2E_grid


P_types = [np.array([[U2U_types[i],1-U2U_types[i]],
                     [1-E2E_types[i],E2E_types[i]]]) 
           for i in range(len(U2U_types))
          ]


hetero_G_types = make_1dtypes('G',
                             G_types)
## LPR configuration
hetero_G_LPR_types = []

for this_type in hetero_G_types:
    this_type.sigma_psi = lc_paras['σ_ψ_sub']
    this_type.sigma_eps = lc_paras['σ_θ_sub']
    this_type.prepare_shocks()
    hetero_G_LPR_types.append(this_type)
    

## HPR configuration
hetero_p_t_risk_G_types = make_3dtypes(by_list = ['sigma_psi',
                                                'sigma_eps',
                                                 'G'],
                                    vals_list = [sigma_psi_types,
                                                 sigma_eps_types,
                                                G_types],
                                     perfect_correlated = False
                                    )

hetero_p_t_ue_risk_sub_types = make_3dtypes(by_list = ['sigma_psi',
                                                   'sigma_eps',
                                                  'P'],
                                    vals_list = [sigma_psi_types,
                                                 sigma_eps_types,
                                                P_types],
                                     perfect_correlated = False,
                                    subjective = True
                                    )
## HPRUR configuration
hetero_p_t_ue_risk_G_types = make_4dtypes(by_list = ['sigma_psi',
                                                        'sigma_eps',
                                                        'P',
                                                        'G'],
                                    vals_list = [sigma_psi_types,
                                                 sigma_eps_types,
                                                P_types,
                                                G_types],
                                     perfect_correlated = False
                                    )

## SHPRUR configuration
hetero_p_t_ue_risk_G_sub_types = make_4dtypes(by_list = ['sigma_psi',
                                                        'sigma_eps',
                                                        'P',
                                                        'G'],
                                              vals_list = [sigma_psi_types,
                                                           sigma_eps_types,
                                                            P_types,
                                                            G_types],
                                              perfect_correlated = False,
                                              subjective = True
                                    )

# -

# ### Solve consumption policies

# +
## SET TYPES HERE

##########################################
types = hetero_p_t_ue_risk_G_sub_types
##########################################

## hetero_p_t_ue_risk_types HPRUR
## hetero_p_t_ue_risk_G_types HPRURG 


# + code_folding=[12]
## solve various models


specs = ['ob']*len(types)

type_names= [str(x) for x in range(1,len(types)+1)]

ms_stars = []
σs_stars = []

t_start = time()
for i, model in enumerate(types):
    ## terminal solution
    m_init,σ_init = model.terminal_solution()

    ## solve backward
    if specs[i]!='cr':
        ms_star, σs_star = solve_model_backward_iter(model,
                                                     m_init,
                                                     σ_init)
    else:
        ms_star, σs_star = solve_model_backward_iter(model,
                                                     m_init,
                                                     σ_init,
                                                     sv = True)
    ms_stars.append(ms_star)
    σs_stars.append(σs_star)

t_finish = time()

print("Time taken, in seconds: "+ str(t_finish - t_start))



# + code_folding=[0, 12]
## compare solutions 

m_grid = np.linspace(0.0,10.0,200)
## plot c func at different age /asset grid
years_left = [0,21,40,59]

n_sub = len(years_left)

eps_fix = 0 ## the first eps grid 

fig,axes = plt.subplots(1,n_sub,figsize=(6*n_sub,6))

for x,year in enumerate(years_left):
    age = lc_mkv.L-year
    i = lc_mkv.L-age
    for k,type_name in enumerate(type_names):
        m_plt_u,c_plt_u = ms_stars[k][i,:,eps_fix,0],σs_stars[k][i,:,eps_fix,0]
        m_plt_e,c_plt_e = ms_stars[k][i,:,eps_fix,1],σs_stars[k][i,:,eps_fix,1]
        c_func_u = lambda m: interp(m_plt_u,c_plt_u,m)
        c_func_e = lambda m: interp(m_plt_e,c_plt_e,m)
        axes[x].plot(m_grid,
                     c_func_u(m_grid),
                     label = type_name+',unemployed',
                     lw=3
                    )
        #axes[x].plot(m_grid,
        #             c_func_e(m_grid),
        #             label = type_name+',employed',
        #             lw=3
        #            )
    axes[x].legend()
    #axes[x].set_xlim(0.0,np.max(m_plt_u))
    axes[x].set_xlabel('asset')
    axes[0].set_ylabel('c')
    axes[x].set_title(r'$age={}$'.format(age))
# -

# ## Aggregate steady state distributions

# + code_folding=[0]
## a function that computes social security tax rate balances gov budget for pension
from Utility import unemp_insurance2tax  
## a function that computes tax rate balances gov budget for ue insurance
from Utility import SS2tax
from Utility import CDProduction  
from PrepareParameters import production_paras_y as production_paras


# + code_folding=[0, 8, 237, 272, 308, 322]
#################################
## general functions used 
# for computing transition matrix
##################################

## compute the list of transition matrix from age t to t+1 for all age 

@njit
def calc_transition_matrix(model, 
                           ms_star, ## new,  life cycle age x asset x tran shock x z state grid 
                           σs_star, ## new, life cycle consumption t age  x asset x tran shock x z state grid
                           dist_mGrid_list, ## new, list, grid of m for distribution 
                           dist_pGrid_list,  ## new, list, grid of p for distribution 
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

        ## unemployment insurance 
        unemp_insurance = model.unemp_insurance
        
        ## tax rate
        λ = model.λ
        λ_SS = model.λ_SS
        
        ## permanent income growth factor
        G = model.G
        
        ## grid holders
 
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
                Cnow_u[m_id] = mlinterp((ms_star[year_left,:,0,0],   
                                        model.eps_grid),
                                       σs_star[year_left,:,:,0],
                                       (m,fix_epsGrid))
                
                Cnow_e[m_id] = mlinterp((ms_star[year_left,:,0,1],
                                        model.eps_grid),
                                       σs_star[year_left,:,:,1],
                                       (m,fix_epsGrid))
                
            
            ## more generally, depending on the nb of markov states 
        
            aNext_u = this_dist_mGrid - Cnow_u # Asset policy grid in each age
            aNext_e = this_dist_mGrid - Cnow_e # Asset policy grid in each age
            
            aPol_Grid_u_list.append(aNext_u) # Add to list
            aPol_Grid_e_list.append(aNext_e) # Add to list

            bNext_u = model.R*aNext_u
            bNext_e = model.R*aNext_e

            
            ## determine income process depending on emp/uemp| work/retirement
            if k <=model.T-1:
                ## work age 
                perm_shks_G_ef = perm_shks* G[k+1]
                tran_shks_ef_u = np.ones_like(tran_shks)*(model.transfer
                                                        +(1-λ)*unemp_insurance)
                tran_shks_ef_e = np.ones_like(tran_shks)*(model.transfer
                                                        +(1-λ)*(1-λ_SS)*tran_shks)

                #mNext_ij = bNext_u[i]/perm_shks_G +model.transfer+ (1-λ)*unemp_insurance # Compute next period's market resources given todays bank balances bnext[i]
            else:
                ## retirement
                perm_shks_G_ef = np.ones_like(perm_shks)*G[k+1]
                tran_shks_ef_u = np.ones_like(tran_shks)*(model.transfer
                                                        + model.pension)
                tran_shks_ef_e = tran_shks_ef_u

            if fast==True:
                print('warning: the fast method is not fully developed yet!!!')
            
                # Generate Transition Matrix for u2u
                TranMatrix_uu = gen_tran_matrix_fast(this_dist_mGrid, 
                                                     bNext_u, 
                                                     shk_prbs_ntrl,
                                                     perm_shks_G_ef,
                                                     tran_shks_ef_u)
                
                # Generate Transition Matrix for u2e
                TranMatrix_ue = gen_tran_matrix_fast(this_dist_mGrid, 
                                                     bNext_e, 
                                                     shk_prbs_ntrl,
                                                     perm_shks_G_ef,
                                                     tran_shks_ef_e)
            

                # Generate Transition Matrix for e2e 
                
                TranMatrix_ee = gen_tran_matrix_fast(this_dist_mGrid, 
                                                     bNext_e, 
                                                     shk_prbs_ntrl,
                                                     perm_shks_G_ef,
                                                     tran_shks_ef_e)
            

                # Generate Transition Matrix for e2u 
                 
                TranMatrix_eu = gen_tran_matrix_fast(this_dist_mGrid, 
                                                     bNext_u, 
                                                     shk_prbs_ntrl,
                                                     perm_shks_G_ef,
                                                     tran_shks_ef_u)


            else:  ## slow method  (2-state Markov implemented)

                    
                # Generate Transition Matrix for u2u 
                TranMatrix_uu = gen_tran_matrix(this_dist_mGrid,
                                               this_dist_pGrid,
                                               bNext_u,
                                               shk_prbs,
                                               perm_shks_G_ef,
                                               tran_shks_ef_u)
                
    
                # Generate Transition Matrix for u2e
                
                TranMatrix_ue = gen_tran_matrix(this_dist_mGrid,
                                               this_dist_pGrid,
                                               bNext_u,
                                               shk_prbs,
                                               perm_shks_G_ef,
                                               tran_shks_ef_e)

                        
                # Generate Transition Matrix for e2u 
                
                TranMatrix_eu = gen_tran_matrix(this_dist_mGrid,
                                               this_dist_pGrid,
                                                bNext_e,
                                               shk_prbs,
                                               perm_shks_G_ef,
                                               tran_shks_ef_u)
                        
                # Generate Transition Matrix for e2e 
                TranMatrix_ee = gen_tran_matrix(this_dist_mGrid,
                                               this_dist_pGrid,
                                               bNext_e,
                                               shk_prbs,
                                               perm_shks_G_ef,
                                               tran_shks_ef_e)
                        
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
        ###.   the age/state dependent transition matrices necessary for computing aggregate consumption
        
        
        ## consumption policy and saving grid on each m, p, z and k grid 

        aPol_Grid_list = List([aPol_Grid_u_list,
                               aPol_Grid_e_list]) ## list of consumption 
        
        
        return tran_matrix_list, aPol_Grid_list #cPol_Grid_list

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
        len(model.eps_shk_true_draws)
    )*1/len(model.eps_shk_true_draws)
    
    λ = model.λ
    init_b = model.init_b
    ue_insurance = np.repeat(np.ones_like(model.eps_shk_true_draws),
                          len(init_p_plus_shk_probs))*model.unemp_insurance  
    init_p_draws = np.exp(np.repeat(init_p_plus_shk_draws,
                          len(model.eps_shk_true_draws)))
    
    ## this function starts with a state-specific initial distribution over m and p as a vector sized of (n_m x n_p) 
    NewBornDist = jump_to_grid(np.ones_like(init_p_draws)*((1-λ)*ue_insurance+init_b/init_p_draws+model.transfer), ## initial unemployment insurance and accidental bequest transfer
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
             for psi_shk in model.psi_shk_true_draws
            ]
        )
    )
    init_p_plus_shk_probs = np.ones(len(init_p_plus_shk_draws))/len(init_p_plus_shk_draws)
    shk_prbs = np.repeat(
        init_p_plus_shk_probs,
        len(model.eps_shk_true_draws)
    )*1/len(model.eps_shk_true_draws)
    
    λ = model.λ
    λ_SS = model.λ_SS
    init_b = model.init_b
    
    tran_shks = np.exp(np.repeat(model.eps_shk_true_draws,
                          len(init_p_plus_shk_probs)))
    init_p_draws = np.exp(np.repeat(init_p_plus_shk_draws,
                          len(model.eps_shk_true_draws)))
    ## this function starts with a state-specific initial distribution over m and p as a vector sized of (n_m x n_p) 
    NewBornDist = jump_to_grid((1-λ)*(1-λ_SS)*tran_shks+init_b/init_p_draws+model.transfer,
                               ## initial transitory risks and accidental bequest transfer
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

def flatten_list(grid_lists,      ## nb.z x T x nb x nm x np 
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



# + code_folding=[0, 5, 17, 111, 243, 265, 294, 325]
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
                                 num_pointsM = 40,  
                                 num_pointsP = 50, 
                                 max_m_fac = 100.0,
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
                                                maxg = model.a_grid[-1]*max_m_fac, 
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
                    max_p = max_p_fac*std_p*(1/(1-model.LivPrb[i]))**0.5 # Consider probability of staying alive this period
                    right_sided_grid = make_grid_exp_mult(1.05+1e-3, np.exp(max_p), num_pointsP, 2)
                    left_sided_gird = np.append(1.0/np.fliplr([right_sided_grid])[0],np.ones(1))
                    left_sided_gird = 1.0/np.fliplr([right_sided_grid])[0]
                    this_dist_pGrid = np.append(left_sided_gird,
                                                right_sided_grid) # Compute permanent income grid this period. Grid of permanent income may differ dependent on PermShkStd
                    dist_pGrid.append(this_dist_pGrid)

            else:
                dist_pGrid = [dist_pGrid] #If grid of permanent income prespecified then use as pgrid
                
            self.m_dist_grid_list = List(dist_mGrid)
            self.p_dist_grid_list = List(dist_pGrid)        
            
    ## get the distributions of each age by iterating forward over life cycle 

    def ComputeSSDist(self,
              ms_star = None,
              σs_star = None):
     
        model = self.model
        m_dist_grid_list = self.m_dist_grid_list
        p_dist_grid_list = self.p_dist_grid_list
        ss_dstn = self.ss_dstn
        age_dist = self.age_dist
        
        time_start = time()

        ## get the embedded list sized n_z x T x n_m x n_p
        tran_matrix_lists, a_PolGrid_list = calc_transition_matrix(model, 
                                                                   ms_star, ## 
                                                                 σs_star,
                                                                 m_dist_grid_list,
                                                                 p_dist_grid_list,
                                                                 fast = False)
        
        # storing as sparse matrix
        tran_matrix_lists = [[sparse.csr_matrix(tran_matrix_lists[0][i]) 
                              for i in range(len(tran_matrix_lists[0]))],
                             [sparse.csr_matrix(tran_matrix_lists[1][i]) 
                              for i in range(len(tran_matrix_lists[1]))]
                            ]
        
        ## save the output into the model 
        self.tran_matrix_lists = tran_matrix_lists
        
        ## the initial distribution in the first period of life 
        initial_dist_u = initial_distribution_u(model,
                                              m_dist_grid_list[0],
                                              p_dist_grid_list[0])

        initial_dist_e = initial_distribution_e(model,
                                                m_dist_grid_list[0],
                                                p_dist_grid_list[0])
        
        self.initial_dist_u = initial_dist_u
        self.initial_dist_e = initial_dist_e


        ## iterate forward 

        dist_u_lists = []
        dist_e_lists = []
        dist_u_lists.append(initial_dist_u)
        dist_e_lists.append(initial_dist_e)



        ## policy grid lists 
        ap_u_PolGrid_list = []
        ap_e_PolGrid_list = []


        ## m/p distribution in the first period in life (newborns)
        this_dist_u = initial_dist_u
        this_dist_e = initial_dist_e


        ## iterate forward for all periods in life 
       
        
        for k in range(model.L-1): ## no transition matrix in the last period !
            
            ## uemp 
            this_dist_u = tran_matrix_lists[0][k]@this_dist_u
            dist_u_lists.append(this_dist_u)
            

            ##emp
            this_dist_e = tran_matrix_lists[1][k]@this_dist_e
            dist_e_lists.append(this_dist_e)
                    
        ## get level of a over life cycle 
        ap_u_PolGrid_list = [np.multiply.outer(a_PolGrid_list[0][k],
                                           p_dist_grid_list[k]).flatten() for k in range(model.L)]
        ap_e_PolGrid_list = [np.multiply.outer(a_PolGrid_list[1][k],
                                           p_dist_grid_list[k]).flatten() for k in range(model.L)]
   
        
        ## stack the distribution lists 
        dist_lists = [dist_u_lists,
                     dist_e_lists]
       

        # a policy grid 
        ap_PolGrid_list = [ap_u_PolGrid_list,
                          ap_e_PolGrid_list]
        
        
        ## get the normalized a over life cycle 
        ap_ratio_u_PolGrid_list = [np.repeat(a_PolGrid_list[0][k],
                                           len(p_dist_grid_list[k])) for k in range(model.L)]
        ap_eratio_e_PolGrid_list = [np.repeat(a_PolGrid_list[1][k],
                                           len(p_dist_grid_list[k])) for k in range(model.L)]
        
        # a policy grid 
        ap_ratio_PolGrid_list = [ap_ratio_u_PolGrid_list,
                                 ap_eratio_e_PolGrid_list]

        ## get the permanent income grid over life cycle
        p_u_list = [np.multiply.outer(np.ones_like(a_PolGrid_list[0][k]),
                                      p_dist_grid_list[k]).flatten() for k in range(model.L)]
        p_e_list = [np.multiply.outer(np.ones_like(a_PolGrid_list[1][k]),
                                      p_dist_grid_list[k]).flatten() for k in range(model.L)]
        p_list = [p_u_list,
                 p_e_list]
        
        time_end = time()
        print('time taken to get SS dist:'+str(time_end-time_start))
        
        memory = Process().memory_info().rss/1024/1024/1024
        print('memory usage: '+str(memory))
        
        self.dist_lists = dist_lists
        self.p_list = p_list
        self.ap_PolGrid_list = ap_PolGrid_list
        ## also store flatten list of level of a and c
        self.ap_grid_dist, self.ap_pdfs_dist = flatten_list(ap_PolGrid_list,
                                                            dist_lists,
                                                            ss_dstn,
                                                            age_dist)
        
        self.a_grid_dist, self.a_pdfs_dist = flatten_list(ap_ratio_PolGrid_list,
                                                            dist_lists,
                                                            ss_dstn,
                                                            age_dist)
    ### Aggregate C or A

    def Aggregate(self):
        ## compute aggregate A 
        ap_PolGrid_list = self.ap_PolGrid_list
        p_list = self.p_list
        dist_lists = self.dist_lists
        ss_dstn = self.ss_dstn
        age_dist = self.age_dist

        self.A = AggregateDist(ap_PolGrid_list,
                              dist_lists,
                              ss_dstn,
                              age_dist)
        
        self.P = AggregateDist(p_list,
                              dist_lists,
                              ss_dstn,
                              age_dist)
        
        self.A_norm = self.A/self.P 
        
    ### Aggregate within age 
    
    def AggregatebyAge(self):
        
        model = self.model 
        
        ap_PolGrid_list = self.ap_PolGrid_list
        dist_lists = self.dist_lists
        ss_dstn = self.ss_dstn

        ### Aggregate distributions within age

        A_life = []


        for t in range(model.L):
            age_dist_sparse = np.zeros(model.L)
            age_dist_sparse[t] = 1.0 ## a fake age distribution that gives the age t the total weight

            ## age-specific wealth

            A_this_age = AggregateDist(ap_PolGrid_list,
                                  dist_lists,
                                  ss_dstn,
                                  age_dist_sparse)
            A_life.append(A_this_age)
            
        self.A_life = A_life
        
    ### Wealth distribution over life cycle 

    def get_lifecycle_dist(self):

        model = self.model 
        ap_PolGrid_list = self.ap_PolGrid_list
        dist_lists = self.dist_lists
        ss_dstn = self.ss_dstn

        ## Flatten distribution by age
        ap_grid_dist_life = []
        ap_pdfs_dist_life = []
 

        for t in range(model.L):

            age_dist_sparse = np.zeros(model.L)
            age_dist_sparse[t] = 1.0

            ap_grid_dist_this_age, ap_pdfs_dist_this_age = flatten_list(ap_PolGrid_list,
                                                                        dist_lists,
                                                                        ss_dstn,
                                                                        age_dist_sparse)

            ap_grid_dist_life.append(ap_grid_dist_this_age)
            ap_pdfs_dist_life.append(ap_pdfs_dist_this_age)


        self.ap_grid_dist_life = ap_grid_dist_life
        self.ap_pdfs_dist_life = ap_pdfs_dist_life
        
    ### Get lorenz weights  
    
    def Lorenz(self,
              variable='a'):
        """
        returns the lorenz weights and value 
        """
        ap_grid_dist = self.ap_grid_dist
        ap_pdfs_dist = self.ap_pdfs_dist
        
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


# + code_folding=[1]
### Get lorenz weights  
def Lorenz(model_results):
    """
    input
    ======
    list of household types with equal fractions
    
    output
    =======
    returns the lorenz weights and value 
    """
    probs = 1/len(model_results) ## equal probable types 
    ap_grid_dist = np.array([model_result['ap_grid_dist'] for model_result in model_results]).flatten()
    ap_pdfs_dist = np.array([[model_result['ap_pdfs_dist'] for model_result in model_results]]).flatten()*probs
    
    ## sort 
    ap_grid_dist_sort = ap_grid_dist.argsort()
    ap_grid_dist = ap_grid_dist[ap_grid_dist_sort]
    ap_pdfs_dist = ap_pdfs_dist[ap_grid_dist_sort]
    
    ## compute things needed for lorenz curve plot of asset accumulation 

    share_agents_ap, share_ap = lorenz_curve(ap_grid_dist,
                                             ap_pdfs_dist,
                                             nb_share_grid = 200)

    return share_agents_ap,share_ap


# + code_folding=[2, 61, 79]
## new functions that are needed for heterogenous-types 

def solve_1type(model,
                m_star,
                σ_star,
                n_m=40,
                n_p=40):
    """
    this function obtains the stationary distribution a particular life cycle type, 
      and produces all results regarding the stationary distribution.
      
    input
    =====
    model: life cycle class instance
    m_star, \sigma_star: optimal consumption policy of this agent 
    
    output
    ======
    a dictionary storing results regarding the stationary dist for the household block with only this type
    the household block instance to be appended with other household blocks. 
    """
    HH_this = HH_OLG_Markov(model = model)
    HH_this.define_distribution_grid(num_pointsM = n_m, 
                                    num_pointsP = n_p)
    HH_this.ComputeSSDist(ms_star = m_star,
                          σs_star = σ_star)
    HH_this.Aggregate()
    print('aggregate savings under stationary distribution:', str(HH_this.A))
    print('aggregate savings under stationary distribution:', str(HH_this.A_norm))


    ## lorenz 
    share_agents_ap_this,share_ap_this = HH_this.Lorenz()

    ## gini in pe
    gini_this_pe = gini(share_agents_ap_this,
                     share_ap_this)

    ## life cycle 

    HH_this.AggregatebyAge()

    A_life_this = HH_this.A_life
    
        
    model_dct =  {'A':HH_this.A,
                  'A_norm':HH_this.A_norm,
                'A_life': HH_this.A_life,
                'share_agents_ap':share_agents_ap_this,
                'share_ap':share_ap_this,
               'ap_grid_dist':HH_this.ap_grid_dist,
                'ap_pdfs_dist':HH_this.ap_pdfs_dist,
                'a_grid_dist':HH_this.a_grid_dist,
                'a_pdfs_dist':HH_this.a_pdfs_dist,
                'gini':gini_this_pe
                 }
    
    ## save it as a pkl

    return model_dct,HH_this

def solve_types(model_list,
                ms_star_list,
                σs_star_list):
    
    sols_all_types = []
    HH_all_types = []
    ## loop over different models 
    for k, model in enumerate(model_list):
        print('solving model {}'.format(str(k)))
        print('\n')
        sol_this_type,HH_this = solve_1type(model,
                                            ms_star_list[k],
                                            σs_star_list[k])
        sols_all_types.append(sol_this_type)
        HH_all_types.append(HH_this)
        print('\n')
    return sols_all_types,HH_all_types

def combine_results(results_by_type):
    """
    input
    =====
    results_by_type: a list of dictioary storing all results by type
    
    output
    ======
    a dictionary storing results combining all types 
    
    """
    
    nb_types = len(results_by_type)
    probs = 1/nb_types 
    
    ## aggregate wealth 
    A_pe = np.sum(
        np.array([result['A'] 
                  for result in results_by_type])
    )*probs
    
    ## aggregate wealth 
    A_norm_pe = np.sum(
        np.array([result['A_norm'] 
                  for result in results_by_type])
    )*probs
    
    ## life cycle wealth
    A_life_pe = np.sum(np.array([result['A_life'] for result in results_by_type]),axis=0)*probs
    
    ## distribution 
    ap_grid_dist_pe = np.array([result['ap_grid_dist'] for result in results_by_type]).flatten()
    ap_pdfs_dist_pe = np.array([result['ap_pdfs_dist'] for result in results_by_type]).flatten()*probs
    ap_grid_dist_pe_sort = ap_grid_dist_pe.argsort()
    ap_grid_dist_pe = ap_grid_dist_pe[ap_grid_dist_pe_sort]
    ap_pdfs_dist_pe = ap_pdfs_dist_pe[ap_grid_dist_pe_sort]
    
    a_grid_dist_pe = np.array([result['a_grid_dist'] for result in results_by_type]).flatten()
    a_pdfs_dist_pe = np.array([result['a_pdfs_dist'] for result in results_by_type]).flatten()*probs
    a_grid_dist_pe_sort = a_grid_dist_pe.argsort()
    a_grid_dist_pe = a_grid_dist_pe[a_grid_dist_pe_sort]
    a_pdfs_dist_pe = a_pdfs_dist_pe[a_grid_dist_pe_sort]
    

    ## lorenz share
    share_agents_ap_pe,share_ap_pe = Lorenz(results_by_type)
    
    ## gini
    gini_this_pe = gini(share_agents_ap_pe,
                        share_ap_pe)


    model_dct_pe =  {'A':A_pe,
                     'A_norm':A_norm_pe,
                  'A_life': A_life_pe,
                  'share_agents_ap':share_agents_ap_pe,
                  'share_ap':share_ap_pe,
                  'ap_grid_dist':ap_grid_dist_pe,
                  'ap_pdfs_dist':ap_pdfs_dist_pe,
                   'a_grid_dist':a_grid_dist_pe,
                   'a_pdfs_dist':a_pdfs_dist_pe,
                  'gini':gini_this_pe,
                   }
    
    return model_dct_pe


# + code_folding=[0, 7, 32]
## market class

class Market_OLG_mkv_hetero_types:
    """
    A class of the market
    """

    def __init__(self,
                 households_list=None, ## a list of different ``types'' of households block
                 production=None):  
        
        n_types = len(households_list)
        self.households_list = households_list   ## HH block 
        self.model_list =  [self.households_list[i].model for i in range(n_types)]  ## life-cycle model 
        
        ### normalize A based on the model parameters first
        ss_dstn_all = np.array([self.households_list[i].ss_dstn for i in range(n_types)])
        ss_dstn = ss_dstn_all.mean(axis=0) ## the average emp uemp fraction over types 
        self.ss_dstn = ss_dstn ## storing it in the market class for future use 
        age_dist = self.households_list[0].age_dist 
        T =  self.model_list[0].T
        
        L_ss_all = np.sum(age_dist[:T-1])*ss_dstn[1] ## employment fraction for working age population
        self.emp_ss = L_ss_all ## average across equal-probable agents 
        self.uemp_ss = np.sum(age_dist[:T-1])*ss_dstn[0]
        
        ## the producation side 
        production.normlize_Z(N_ss = self.emp_ss)
        self.production = production   ## Production function

    ## stationary asset demand for a given capital stock/factor price

    def StE_K_d(self,
                K_s,   ## given asset supply 
                dstn):  ## distribution between emp and unemp 
        """
        Given a proposed capital stock/asset supply ,
        this function generates the stationary asset demands 

        """
        model_list = self.model_list
        households_list = self.households_list
        production = self.production 
        age_dist = self.households_list[0].age_dist
        T = self.model_list[0].T ## retirement age 
        
        ################################
        ## Step 0. Parameterize the model 
        ################################
        ## get the L based on current employed fraction
        uemp_now,emp_now = dstn[0]*np.sum(age_dist[:T-1]),dstn[1]*np.sum(age_dist[:T-1])
        print('Labor force',str(emp_now))

        ## Obtain factor prices from FOC of firms
        
        production.K = K_s
        production.L = emp_now
        
        print('Capital stock',str(K_s))
        W,R = production.YL(),production.R()
        print('Wage rate',str(W))
        print('Real interest rate',str(R))
        
        ## stable age distribution 
        age_dist = households_list[0].age_dist
        λ = unemp_insurance2tax(model_list[0].unemp_insurance,
                                uemp_now)
        print('Tax rate',str(λ))
                     
        ## obtain social security rate balancing the SS replacement ratio 
        G_av = np.mean([model_list[i].G for i in range(len(model_list))],axis=0)
        #λ_SS = SS2tax(model_list[0].pension, ## social security /pension replacement ratio 
        #             model_list[0].T,  ## retirement years
        #            age_dist,  ## age distribution in the economy 
        #            G_av,         ## permanent growth factor lists over cycle
        #            emp_now)
        #print('Social security tax rate',str(λ_SS))
    
                                      
        ## solutions 
        ms_stars = []
        σs_stars = []
        
        ## start the loop calculating K_d for each type/model of consumers 
        K_d = 0.0
        for model_i,model in enumerate(model_list):
                                      
            ##################################
            model.W, model.R = W,R
            ##################################

            ## obtain tax rate from the government budget balance 
            model.λ = λ

            ## obtain social security rate balancing the SS replacement ratio 
            #model.λ_SS = λ_SS
                                      
            ################################
            ## Step 1. Solve the model 
            ################################

            ## terminal period solution
            m_init,σ_init = model.terminal_solution()

            ## solve the model 
            ms_star, σs_star = solve_model_backward_iter(model,
                                                         m_init,
                                                         σ_init)
            ms_stars.append(ms_star)
            σs_stars.append(σs_star)                    

            ################################
            ## Step 2. StE distribution
            ################################

            ## accidental transfers 

            ## Get the StE K_d
            ## get the transition matrix and policy grid 

            n_m = 40
            n_p = 50
            
            households_list[model_i].define_distribution_grid(num_pointsM = n_m,
                                                        num_pointsP = n_p)
            households_list[model_i].ComputeSSDist(ms_star = ms_star,
                                             σs_star = σs_star)

            households_list[model_i].Aggregate()

            K_d_this = households_list[model_i].A*model.W  ## no population growth otherwise diluted by (1+n)
        
            K_d += K_d_this
            
        K_d = K_d/len(model_list)
        
        print('Induced capital stock',str(K_d))
        
        return K_d
    
    def get_equilibrium_k(self):
        ss_dstn = self.ss_dstn
        
        ## function to solve the equilibrium 
        eq_func = lambda K: self.StE_K_d(K_s = K,
                                         dstn = ss_dstn)
        ## solve the fixed point 
        K_eq = op.fixed_point(eq_func,
                              x0 = 7.3)
        
        self.K_eq = K_eq
    
    def get_equilibrium_dist(self):
        
        households_list = self.households_list 
        model_list = self.model_list 
        
        ### get equilibrium values 
        K_eq = self.K_eq 
        L_ss = self.emp_ss
        
        ## compute factor prices in StE
        production.K = K_eq
        production.L = L_ss
        
        print('SS Capital stock',str(K_eq))
        W_eq,R_eq = production.YL(),production.R()
        print('SS Wage Rate',str(W_eq))
        print('SS Real interest rate',str(R_eq))
        
        λ = unemp_insurance2tax(model_list[0].unemp_insurance,
                                self.uemp_ss)
        print('Tax rate',str(λ))
                     
        ## obtain social security rate balancing the SS replacement ratio 
        G_av = np.mean([model_list[i].G for i in range(len(model_list))],axis=0)
        age_dist = households_list[0].age_dist
        #λ_SS = SS2tax(model_list[0].pension, ## social security /pension replacement ratio 
        #             model_list[0].T,  ## retirement years
        #            age_dist,  ## age distribution in the economy 
        #            G_av,         ## permanent growth factor lists over cycle
         #           self.emp_ss)
        
        #print('Social security tax rate',str(λ_SS))
        
        ms_stars = []
        σs_stars = []
        model_updated_list = []
        
        for model_i, model in enumerate(model_list):
            ## get the distribution under SS
            model.W,model.R = W_eq,R_eq
            model.λ = λ
            #model.λ_SS = λ_SS
            
            ## solve the model again 

            ## terminal period solution
            m_init,σ_init = model.terminal_solution()

            ms_star, σs_star = solve_model_backward_iter(model,
                                                         m_init,
                                                         σ_init)
            
            ms_stars.append(ms_star)
            σs_stars.append(σs_star)
            model_updated_list.append(model)
                    
        results_by_type,_ = solve_types(model_updated_list, ## use updated list because it has updated factor price 
                                          ms_stars,
                                          σs_stars)
        
        results_combined = combine_results(results_by_type) 
        
        return results_combined

# + code_folding=[0]
## initializations 
production = CDProduction(α = production_paras['α'],
                          δ = production_paras['δ'],
                          target_KY = production_paras['K2Y ratio'],
                          target_W = production_paras['W']) 


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

# -

# ## Solve the heterogenous-type model 

# + code_folding=[2]
## solve a list of models and save all solutions as pickles 

results_by_type,households_by_type = solve_types(types,
                                                ms_stars,
                                                σs_stars)

# + code_folding=[]
## obtain the combined results 
results_combined_pe = combine_results(results_by_type)

# + code_folding=[0]
## aggregate lorenz curve 

print('gini from model:'+str(results_combined_pe['gini']))

gini_SCF = gini(SCF_share_agents_ap,
                 SCF_share_ap)
print('gini from SCE:'+str(gini_SCF))

# + code_folding=[0, 3]
## Lorenz curve of steady state wealth distribution

fig, ax = plt.subplots(figsize=(8,8))
ax.plot(results_combined_pe['share_agents_ap'],
        results_combined_pe['share_ap'], 
        'r--',
        label='Model, Gini={:.2f}'.format(results_combined_pe['gini']))
ax.plot(SCF_share_agents_ap,
        SCF_share_ap, 
        'b-.',
        label='SCF, Gini={:.2f}'.format(gini_SCF))
ax.plot(results_combined_pe['share_agents_ap'],
        results_combined_pe['share_agents_ap'], 
        'k-',
        label='equality curve')
ax.legend()
plt.xlim([0,1])
plt.ylim([0,1])
#plt.savefig('../Graphs/model/lorenz_a_hetero_type.png')

# + code_folding=[0]
## Wealth distribution 
h2m_cut_off = round(1/24,2)

fig, ax = plt.subplots(figsize=(8,6))
ax.set_title('Wealth distribution')
## h2m share 
h2m_share = h2m_ratio(results_combined_pe['a_grid_dist'],
                      results_combined_pe['a_pdfs_dist'],
                      h2m_cut_off)
plt.hist(np.log(results_combined_pe['ap_grid_dist']+1e-4),
         weights = results_combined_pe['ap_pdfs_dist'],
                                 density=True,
                                bins=300,
                                alpha=0.5,
                                #color='red',
                                label='H2M={:.2f}'.format(h2m_share))

ax.legend(loc=0)
ax.set_xlabel(r'$a$')
ax.set_ylabel(r'$prob(a)$')
#fig.savefig('../Graphs/model/distribution_a_hetero_type.png')
# -

pickle.dump(results_combined_pe,open('./model_solutions/SHPRUR_PE.pkl','wb'))

# ## GE with multiple types 

# + code_folding=[0]
## using a higher beta for GE results 
households_by_type_ge =[]
for this_type in households_by_type:
    this_type.β = lc_paras_y['β_h']
    households_by_type_ge.append(this_type)
    
market_OLG_mkv_this = Market_OLG_mkv_hetero_types(households_list = households_by_type_ge,
                                                  production = production)
# -

market_OLG_mkv_this.get_equilibrium_k()

results_combined_ge = market_OLG_mkv_this.get_equilibrium_dist()

pickle.dump(results_combined_ge,open('./model_solutions/'+'SHPRUR_GE.pkl','wb'))
# + code_folding=[0]
## Lorenz curve of steady state wealth distribution

fig, ax = plt.subplots(figsize=(8,8))
ax.plot(results_combined_ge['share_agents_ap'],
        results_combined_ge['share_ap'], 
        'r--',
        label='Model, Gini={:.2f}'.format(results_combined_ge['gini']))
ax.plot(SCF_share_agents_ap,
        SCF_share_ap, 
        'b-.',
        label='SCF, Gini={:.2f}'.format(gini_SCF))
ax.plot(results_combined_pe['share_agents_ap'],
        results_combined_pe['share_agents_ap'], 
        'k-',
        label='equality curve')
ax.legend()
plt.xlim([0,1])
plt.ylim([0,1])
#plt.savefig('../Graphs/model/lorenz_a_hetero_type.png')


# + code_folding=[0]
## Wealth distribution 

fig, ax = plt.subplots(figsize=(8,6))
ax.set_title('Wealth distribution')
## h2m share 
h2m_share = h2m_ratio(results_combined_ge['a_grid_dist'],
                      results_combined_ge['a_pdfs_dist'],
                      h2m_cut_off)
plt.hist(np.log(results_combined_ge['ap_grid_dist']+1e-4),
         weights = results_combined_ge['ap_pdfs_dist'],
                                 density=True,
                                bins=300,
                                alpha=0.5,
                                #color='red',
                                label='H2M={:.2f}'.format(h2m_share))
ax.legend(loc=0)
ax.set_xlabel(r'$a$')
ax.set_ylabel(r'$prob(a)$')
#fig.savefig('../Graphs/model/distribution_a_hetero_type.png')
# -



