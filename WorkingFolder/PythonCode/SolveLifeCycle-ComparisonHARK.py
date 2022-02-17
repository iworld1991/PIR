# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Comparing the solution of this model with that in HARK 
#
# - author: Tao Wang
# - date: Feb 2022
# - this is a companion notebook to the paper "Perceived income risks"

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
#from HARK.utilities import make_grid_exp_mult
from scipy import sparse as sp
import scipy.sparse.linalg
from scipy import linalg as lg 
from numba.typed import List
from Utility import cal_ss_2markov

# + code_folding=[]
## figure plotting configurations

mp.rc('xtick', labelsize=11) 
mp.rc('ytick', labelsize=11) 

mp.rc('legend',fontsize=11)
plt.rc('font',size=11) 
# -

from SolveLifeCycle import LifeCycle, EGM, solve_model_iter
from PrepareParameters import life_cycle_paras_y as inf_paras

# ### parameters 

inf_paras['G'] =  np.ones_like(inf_paras['G'])
inf_paras['unemp_insurance'] = 0.0
inf_paras['P'] = np.array([[0.9,0.1],[0.2,0.8]])

# + code_folding=[0]
inf_mkv = LifeCycle(U = inf_paras['U'], ## transitory ue risk
                    unemp_insurance = inf_paras['unemp_insurance'],
                    pension = inf_paras['pension'], ## pension
                    sigma_n = inf_paras['σ_ψ'], # permanent 
                    sigma_eps = inf_paras['σ_θ'], # transitory 
                    P = inf_paras['P'],   ## transitory probability of markov state z
                    z_val = inf_paras['z_val'], ## markov state from low to high  
                    x = 0.0,           ## MA(1) coefficient of non-permanent inocme shocks
                    ue_markov = True,   
                    adjust_prob = 1.0,
                    sigma_p_init = inf_paras['σ_ψ_init'],
                    init_b = inf_paras['init_b'],
                    ## subjective risk prifile 
                    sigma_n_2mkv = inf_paras['σ_ψ_2mkv'],  ## permanent risks in 2 markov states
                    sigma_eps_2mkv = inf_paras['σ_θ_2mkv'],  ## transitory risks in 2 markov states
                    λ = inf_paras['λ'],  ## tax rate
                    λ_SS = inf_paras['λ_SS'], ## social tax rate
                    transfer = inf_paras['transfer'],  ## transfer 
                    bequest_ratio = inf_paras['bequest_ratio'],
                    LivPrb = inf_paras['LivPrb'],       ## living probability 
                    ## life cycle 
                    T = inf_paras['T'],
                    L = inf_paras['L'],
                    #TGPos = int(L/3) ## one third of life sees income growth 
                    #GPos = 1.01*np.ones(TGPos)
                    #GNeg= 0.99*np.ones(L-TGPos)
                    #G = np.concatenate([GPos,GNeg])
                    #G = np.ones(L)
                    G = inf_paras['G'],
                    #YPath = np.cumprod(G),
                    ## other parameters 
                    ρ = inf_paras['ρ'],     ## relative risk aversion  
                    β = inf_paras['β'],    ## discount factor
                    R = inf_paras['R'],           ## interest factor 
                    W = inf_paras['W'],            ## Wage rate
                    ## subjective models 
                    theta = 0.0, ## extrapolation parameter 

                    ## no persistent state
                    b_y = 0.0,
                    ## wether to have zero borrowing constraint 
                    borrowing_cstr = True
    )

# + code_folding=[9]
## initial consumption functions 

k = len(inf_mkv.s_grid)
k2 =len(inf_mkv.eps_grid)

n = len(inf_mkv.P)
σ_init = np.empty((k,k2,n))
a_init = np.empty((k,k2,n))

for z in range(n):
    for j in range(k2):
        a_init[:,j,z] = inf_mkv.s_grid
        σ_init[:,j,z] = 0.1*a_init[:,j,z]

t_start = time()

a_inf_star, σ_inf_star = solve_model_iter(inf_mkv,
                                          a_init,
                                          σ_init)
t_finish = time()

print("Time taken, in seconds: "+ str(t_finish - t_start))   


## plot c func 

eps = 10 ## a random number 
m_plt_u, c_plt_u = a_inf_star[:,eps,0],σ_inf_star[:,eps,0] 
m_plt_e, c_plt_e = a_inf_star[:,eps,1], σ_inf_star[:,eps,1]
plt.plot(m_plt_u,
         c_plt_u,
         label = 'unemployed',
         lw=3
        )
plt.plot(m_plt_e,
         c_plt_e,
         label = 'employed',
         lw=3
        )
plt.legend()
plt.xlabel(r'$m$')
plt.ylabel(r'$c$')
plt.title('Inifite horizon solution')
# -

# ## Solving the same model with HARK
#

from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType
from HARK.distribution import DiscreteDistribution
from HARK.ConsumptionSaving.ConsIndShockModel import init_lifecycle
from copy import copy
from HARK.ConsumptionSaving.ConsIndShockModel import init_idiosyncratic_shocks
from HARK.utilities import plot_funcs

## for infinite horizon 
init_serial_unemployment = copy(init_idiosyncratic_shocks)
init_serial_unemployment["MrkvArray"] = [inf_paras['P']]
init_serial_unemployment["UnempPrb"] = inf_paras['U']  # to make income distribution when employed
init_serial_unemployment["global_markov"] = False
init_serial_unemployment['CRRA'] = inf_paras['ρ']
init_serial_unemployment['Rfree'] = inf_paras['R']
init_serial_unemployment['LivPrb'] = [inf_paras['LivPrb']]
init_serial_unemployment['PermGroFac'] = [1.0]
init_serial_unemployment['PermShkStd'] = [inf_paras['σ_ψ']]
init_serial_unemployment['TranShkStd'] = [inf_paras['σ_θ']]
init_serial_unemployment['DiscFac'] = inf_paras['β']

print('HARK parameterization',str(init_serial_unemployment))
print('PIR parameterization',str(inf_paras))

SerialUnemploymentExample = MarkovConsumerType(**init_serial_unemployment)
SerialUnemploymentExample.cycles = 0
SerialUnemploymentExample.vFuncBool = False  # for easy toggling here

# Interest factor, permanent growth rates, and survival probabilities are constant arrays
SerialUnemploymentExample.assign_parameters(Rfree = np.array(2 * [SerialUnemploymentExample.Rfree]))
SerialUnemploymentExample.PermGroFac = [
    np.array(2 * SerialUnemploymentExample.PermGroFac)
]
SerialUnemploymentExample.LivPrb = [SerialUnemploymentExample.LivPrb * np.ones(2)]

# Replace the default (lognormal) income distribution with a custom one
employed_income_dist = DiscreteDistribution(np.ones(1), [np.ones(1), np.ones(1)])  # Definitely get income
unemployed_income_dist = DiscreteDistribution(np.ones(1), [np.ones(1), np.zeros(1)]) # Definitely don't
SerialUnemploymentExample.IncShkDstn = [
    [
        unemployed_income_dist,
        employed_income_dist
    ]
]

start_time = time()
SerialUnemploymentExample.solve()
end_time = time()
print(
    "Solving a Markov consumer with serially correlated unemployment took "
    + str(end_time - start_time)
    + " seconds."
)
print("Consumption functions for each discrete state:")
plot_funcs(SerialUnemploymentExample.solution[0].cFunc, 0, 50)
if SerialUnemploymentExample.vFuncBool:
    print("Value functions for each discrete state:")
    plot_funcs(SerialUnemploymentExample.solution[0].vFunc, 5, 50)

## for life cycle 
"""
init_life_cycle_new = copy(init_lifecycle)
init_life_cycle_new['T_cycle'] = lc_paras['L']-1   ## minus 1 because T_cycle is nb periods in a life cycle - 1 in HARK 
init_serial_unemployment['CRRA'] = lc_paras['ρ']
init_serial_unemployment['Rfree'] = lc_paras['R']
init_serial_unemployment['LivPrb'] = [lc_paras['LivPrb']]
init_serial_unemployment['PermGroFac'] = [1.0]
init_serial_unemployment['PermShkStd'] = [lc_paras['σ_ψ']]*init_life_cycle_new['T_cycle']
init_serial_unemployment['TranShkStd'] = [lc_paras['σ_θ']]*init_life_cycle_new['T_cycle']
init_serial_unemployment['DiscFac'] = [lc_paras['β']]
#init_life_cycle_new['PermGroFacAgg'] = list(G[1:])
#init_life_cycle_new['aXtraMin'] = a_min+0.00001
#init_life_cycle_new['aXtraMax'] = a_max
#init_life_cycle_new['aXtraCount'] = 800
#init_life_cycle_new['ShareCount'] = 100

init_life_cycle_new["MrkvArray"] = [lc_paras['P']]
init_life_cycle_new["UnempPrb"] = lc_paras['U']  # to make income distribution when employed
init_life_cycle_new["global_markov"] = False

LifeCycleType = MarkovConsumerType(**init_life_cycle_new)

LifeCycleType.cycles = 1 ## life cycle problem instead of infinite horizon
LifeCycleType.vFuncBool = False  ## no need to calculate the value for the purpose here 
"""

# +
## compare the solutions 

c_u_HARK = SerialUnemploymentExample.solution[0].cFunc[0](m_plt_u)
c_e_HARK = SerialUnemploymentExample.solution[0].cFunc[1](m_plt_e)

plt.title('Comparing consumption policies')
plt.plot(m_plt_u,c_u_HARK,'k-',label='uemp:HARK')
plt.plot(m_plt_u,c_plt_u,'r*',label='uemp:PIR')
plt.plot(m_plt_e,c_e_HARK,'b--',label='emp:HARK')
plt.plot(m_plt_e,c_plt_e,'go',label='emp:PIR')
plt.legend(loc=0)
# -

