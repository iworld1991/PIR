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

# ## Comparing the PIR and [HARK](https://econ-ark.org) solution
#
# - this notebook compare my code and HARK solutions 
# - author: Tao Wang
# - created in Feb 2022
# - modified in Jul 2024
# - this is a companion notebook to the paper "Perceived income risks"

import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from time import time
from copy import copy


# + code_folding=[]
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

from SolveLifeCycleBelief import LifeCycle, solve_model_iter
from PrepareParameters import life_cycle_paras_y 

# ### parameters 

inf_paras = copy(life_cycle_paras_y)
inf_paras['G'] =  np.ones_like(inf_paras['G'])
inf_paras['unemp_insurance'] = 0.0
inf_paras['P'] = np.array([[0.9,0.1],[0.2,0.8]])
inf_paras['λ_SS'] = 0.0 


# + code_folding=[]
inf_mkv_paras_dict = { 'U':inf_paras['U'], ## transitory ue risk
                    'unemp_insurance':inf_paras['unemp_insurance'],
                    'pension':inf_paras['pension'], ## pension
                    'sigma_psi':inf_paras['σ_ψ'], # permanent 
                    'sigma_eps':inf_paras['σ_θ'], # transitory 
                    'P':inf_paras['P'],   ## transitory probability of markov state z
                    'z_val':inf_paras['z_val'], ## markov state from low to high  
                    'x': 0.0,           ## MA(1) coefficient of non-permanent inocme shocks
                    'ue_markov':True,   
                    'adjust_prob':1.0,
                    'sigma_p_init':inf_paras['σ_ψ_init'],
                    'init_b':inf_paras['init_b'],
                    ## subjective risk prifile 
                    'sigma_psi_2mkv':inf_paras['σ_ψ_2mkv'],  ## permanent risks in 2 markov states
                    'sigma_eps_2mkv':inf_paras['σ_θ_2mkv'],  ## transitory risks in 2 markov states
                    'λ':inf_paras['λ'],  ## tax rate
                    'λ_SS':inf_paras['λ_SS'], ## social tax rate
                    'transfer':inf_paras['transfer'],  ## transfer 
                    'bequest_ratio':inf_paras['bequest_ratio'],
                    'LivPrb':inf_paras['LivPrb'],       ## living probability 
                    ## life cycle 
                    'T': inf_paras['T'],
                    'L': inf_paras['L'],
                    'G':inf_paras['G'],
                    ## other parameters 
                    'ρ':inf_paras['ρ'],     ## relative risk aversion  
                    'β': inf_paras['β'],    ## discount factor
                    'R':inf_paras['R'],           ## interest factor 
                    'W':inf_paras['W'],            ## Wage rate
                    ## subjective models 
                    'theta':0.0, ## extrapolation parameter 
                    ## no persistent state
                    'b_y': 0.0,
                    ## wether to have zero borrowing constraint 
                    'borrowing_cstr':True,
                    ## a grids 
                    'grid_max': 20.0,
                    'grid_size': 500}


inf_mkv = LifeCycle(**inf_mkv_paras_dict)

# + code_folding=[0]
## initial consumption functions 


t_start = time()

a_init,σ_init = inf_mkv.terminal_solution()

a_inf_star, σ_inf_star = solve_model_iter(inf_mkv,
                                          a_init,
                                          σ_init)
t_finish = time()

print("Time taken, in seconds: "+ str(t_finish - t_start))   

## plot c func 
z_l = 0
z_h = 1

m_plt_u, c_plt_u = a_inf_star[:-1,0,z_l,0],σ_inf_star[:-1,0,z_l,0] 
m_plt_e, c_plt_e = a_inf_star[:-1,0,z_h,0], σ_inf_star[:-1,0,z_h,0]
plt.plot(m_plt_u,
         c_plt_u,
         label = 'unemployed',
        )
plt.plot(m_plt_e,
         c_plt_e,
         label = 'employed',
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
from HARK.ConsumptionSaving.ConsIndShockModel import init_idiosyncratic_shocks
from HARK.utilities import plot_funcs

## for infinite horizon 
hark_mkv_para = copy(init_idiosyncratic_shocks)
hark_mkv_para["MrkvArray"] = [inf_paras['P']]
hark_mkv_para["UnempPrb"] = inf_paras['U']  # to make income distribution when employed
hark_mkv_para['IncUnemp'] = inf_paras['unemp_insurance']
hark_mkv_para["global_markov"] = False
hark_mkv_para['CRRA'] = inf_paras['ρ']
hark_mkv_para['Rfree'] = inf_paras['R']
hark_mkv_para['LivPrb'] = [inf_paras['LivPrb'][0]] ## constant liv prob 
hark_mkv_para['PermGroFac'] = [1.0]
hark_mkv_para['PermShkStd'] = [inf_paras['σ_ψ']]
hark_mkv_para['TranShkStd'] = [inf_paras['σ_θ']]
hark_mkv_para['DiscFac'] = inf_paras['β']
hark_mkv_para['aXtraMax'] = inf_mkv_paras_dict['grid_max']
hark_mkv_para['aXtraCount'] = inf_mkv_paras_dict['grid_size']-1

print('HARK parameterization',str(hark_mkv_para))
print('PIR parameterization',str(inf_mkv_paras_dict))

hark_mkv = MarkovConsumerType(**hark_mkv_para)
hark_mkv.cycles = 0 ## infinite horizon
hark_mkv.vFuncBool = False  # for easy toggling here

hark_mkv.LivPrb

# + code_folding=[]
# Interest factor, permanent growth rates, and survival probabilities are constant arrays
hark_mkv.assign_parameters(Rfree = np.array(2 * [hark_mkv.Rfree]))
hark_mkv.PermGroFac = [
    np.array(2 * hark_mkv.PermGroFac)
]
hark_mkv.LivPrb = [
    np.array(2 * hark_mkv.LivPrb)
]


# + code_folding=[0]
#Replace the default (lognormal) income distribution with a custom one
employed_income_dist = DiscreteDistribution(np.ones(1), [np.ones(1), np.ones(1)])  # Definitely get income
unemployed_income_dist = DiscreteDistribution(np.ones(1), [np.ones(1), np.zeros(1)]) # Definitely don't
hark_mkv.IncShkDstn = [
    [
        unemployed_income_dist,
        employed_income_dist
    ]
]

# + code_folding=[0]
## solve the model 

start_time = time()
hark_mkv.solve()
end_time = time()
print(
    "Solving a Markov consumer with serially correlated unemployment took "
    + str(end_time - start_time)
    + " seconds."
)
print("Consumption functions for each discrete state:")
plot_funcs(hark_mkv.solution[0].cFunc, 0, 20)
if hark_mkv.vFuncBool:
    print("Value functions for each discrete state:")
    plot_funcs(hark_mkv.solution[0].vFunc, 5, 20)

# + code_folding=[0]
## compare the solutions 

## get the HARK c  
c_u_HARK = hark_mkv.solution[0].cFunc[0](m_plt_u)
c_e_HARK = hark_mkv.solution[0].cFunc[1](m_plt_e)

plt.title('Comparing consumption policies')
plt.plot(m_plt_u,c_u_HARK,'k-',label='uemp: HARK')
plt.plot(m_plt_u,c_plt_u,'r*',label='uemp: PIR')
plt.plot(m_plt_e,c_e_HARK,'b--',label='emp: HARK')
plt.plot(m_plt_e,c_plt_e,'go',label='emp: PIR')
plt.legend(loc=0)
# -
# ### Simulation 

# +
hark_mkv.unpack('cFunc')                      # Expose the consumption rules

# Which variables do we want to track
hark_mkv.track_vars = ['aNrm','pLvl','mNrm','cNrm']

hark_mkv.T_sim =120     
hark_mkv.MrkvPrbsInit = [0.5, 0.5]
hark_mkv.make_shock_history()                 # This is optional
hark_mkv.initialize_sim()                     # Construct the age-25 distribution of income and assets
hark_mkv.simulate()
# -

hark_mkv.shock_history['Mrkv'].shape

# ### Comparing consumption policies under different risks 
#   - back to the infinite horizon for simplicity

# + code_folding=[4]
sigma_eps_ls = [0.01,0.05,0.1,0.2]
sigma_psi_ls = [0.01,0.05,0.1,0.2]

cFunc_list = []
for i,sigma_eps in enumerate(sigma_eps_ls):
    hark_mkv_para['TranShkStd'] = [sigma_eps]
    print(hark_mkv_para['TranShkStd'])
    hark_mkv_para['PermShkStd'] = [sigma_psi_ls[i]]
    print(hark_mkv_para['PermShkStd'])
    hark_mkv_new = MarkovConsumerType(**hark_mkv_para)
    hark_mkv_new.cycles = 0
    hark_mkv_new.vFuncBool = False  # for easy toggling here
    # Interest factor, permanent growth rates, and survival probabilities are constant arrays
    hark_mkv_new.assign_parameters(Rfree = np.array(2 * [hark_mkv_new.Rfree]))
    hark_mkv_new.PermGroFac = [
        np.array(2 * hark_mkv_new.PermGroFac)
    ]
    hark_mkv_new.LivPrb = [hark_mkv_new.LivPrb * np.ones(2)]
    # Replace the default (lognormal) income distribution with a custom one
    employed_income_dist = DiscreteDistribution(np.ones(1), [np.ones(1), np.ones(1)])  # Definitely get income
    unemployed_income_dist = DiscreteDistribution(np.ones(1), [np.ones(1), np.zeros(1)]) # Definitely don't
    hark_mkv_new.IncShkDstn = [
        [
            unemployed_income_dist,
            employed_income_dist
        ]
    ]
    ## solve the model 

    start_time = time()
    hark_mkv_new.solve()
    end_time = time()
    print(
        "Solving a Markov consumer with serially correlated unemployment took "
        + str(end_time - start_time)
        + " seconds."
    )
    
    #print("Consumption functions for each discrete state:")
    #plot_funcs(hark_mkv.solution[0].cFunc, 0, 20)
    cFunc_list.append(hark_mkv_new.solution[0].cFunc)
    
m_values = np.linspace(0.0,20.0,200)

for i,sigma_eps in enumerate(sigma_eps_ls):
    plt.plot(m_values,
             cFunc_list[i][0](m_values),
            label=r'$\sigma^\theta.{}$'.format(sigma_eps))
plt.legend(loc=1)

# + code_folding=[0, 8]
## for life cycle models 
import HARK.ConsumptionSaving.ConsIndShockModel as HARK_model         # The consumption-saving micro model

sigma_eps_ls = [0.01,0.05,0.1,0.2]
sigma_psi_ls = [0.01,0.05,0.1,0.2]

cFunc_list = []

for i,sigma_eps in enumerate(sigma_eps_ls):
    init_life_cycle_new = copy(init_lifecycle)
    lc_paras = copy(life_cycle_paras_y)
    #years_retire = lc_paras['L']- lc_paras['T']
    #init_life_cycle_new['T_cycle'] = lc_paras['L']-1   ## minus 1 because T_cycle is nb periods in a life cycle - 1 in HARK
    #init_life_cycle_new['T_retire'] = lc_paras['T']-1
    #init_life_cycle_new['LivPrb'] = [lc_paras['LivPrb']]*init_life_cycle_new['T_cycle']
    #init_life_cycle_new['PermGroFac'] = lc_paras['G']
    init_life_cycle_new['PermShkStd'] = [sigma_psi_ls[i]]*init_life_cycle_new['T_cycle']
    init_life_cycle_new['TranShkStd'] = [sigma_eps]*init_life_cycle_new['T_cycle']

    LifeCyclePop = HARK_model.IndShockConsumerType(**init_life_cycle_new)
    LifeCyclePop.cycles = 1
    LifeCyclePop.vFuncBool = False  # for easy toggling here
    
    ## solve the model 

    start_time = time()
    LifeCyclePop.solve()                            # Obtain consumption rules by age 
    LifeCyclePop.unpack('cFunc')                      # Expose the consumption rules
    end_time = time()
    print(
        "Solving a Markov consumer with serially correlated unemployment took "
        + str(end_time - start_time)
        + " seconds."
    )
    
    cFunc_list.append(LifeCyclePop.solution[50].cFunc)
    
m_values = np.linspace(0.0,3.0,200)

for i,sigma_eps in enumerate(sigma_eps_ls):
    plt.plot(m_values,
             cFunc_list[i](m_values),
            label=r'$\sigma^\theta=.{}$'.format(sigma_eps))
plt.legend(loc=1)
# -

# ## Life Cycle Model

# + code_folding=[]
## for life cycle 
init_life_cycle_new = copy(init_lifecycle)
lc_paras = copy(life_cycle_paras_y)

years_retire = lc_paras['L']- lc_paras['T']

init_life_cycle_new['T_cycle'] = lc_paras['L']-1   ## minus 1 because T_cycle is nb periods in a life cycle - 1 in HARK 
init_life_cycle_new['CRRA'] = lc_paras['ρ']
init_life_cycle_new['T_retire'] = lc_paras['T']-1
init_life_cycle_new['Rfree'] = lc_paras['R']
init_life_cycle_new['LivPrb'] = [lc_paras['LivPrb'][0]]*init_life_cycle_new['T_cycle']
init_life_cycle_new['PermGroFac'] = lc_paras['G']
init_life_cycle_new['PermShkStd'] = [lc_paras['σ_ψ']]*init_life_cycle_new['T_retire']+[0.0]*years_retire
init_life_cycle_new['TranShkStd'] = [lc_paras['σ_θ']]*init_life_cycle_new['T_retire']+[0.0]*years_retire
init_life_cycle_new['DiscFac'] = lc_paras['β']
init_life_cycle_new['PermGroFacAgg'] = 1.0
init_life_cycle_new['aNrmInitMean']= np.log(lc_paras['init_b'])
init_life_cycle_new['aNrmInitStd']= 0.0
init_life_cycle_new['pLvlInitMean']= np.log(1.0)
init_life_cycle_new['pLvlInitStd']= lc_paras['σ_ψ_init']
init_life_cycle_new["UnempPrb"] = lc_paras['U']  # to make income distribution when employed
init_life_cycle_new['UnempPrbRet'] = 0.0
init_life_cycle_new['IncUnemp'] = 0.0
init_life_cycle_new['aXtraMax'] = 5.0

"""
LifeCycleType = MarkovConsumerType(**init_life_cycle_new)

LifeCycleType.cycles = 1 ## life cycle problem instead of infinite horizon
LifeCycleType.vFuncBool = False  ## no need to calculate the value for the purpose here 
"""
# -

print(init_life_cycle_new)

from HARK.utilities import plot_funcs_der, plot_funcs    
import HARK.ConsumptionSaving.ConsIndShockModel as HARK_model         # The consumption-saving micro model
        # Some tools

LifeCyclePop = HARK_model.IndShockConsumerType(**init_life_cycle_new)
LifeCyclePop.cycles = 1
LifeCyclePop.vFuncBool = False  # for easy toggling here

# +
LifeCyclePop.solve()                            # Obtain consumption rules by age 
LifeCyclePop.unpack('cFunc')                      # Expose the consumption rules

# Which variables do we want to track
LifeCyclePop.track_vars = ['aNrm','pLvl','mNrm','cNrm']

LifeCyclePop.T_sim = lc_paras['L']                
LifeCyclePop.MrkvPrbsInit = [0.5, 0.5]
LifeCyclePop.make_shock_history()                 # This is optional
LifeCyclePop.initialize_sim()                     # Construct the age-25 distribution of income and assets
LifeCyclePop.simulate()
# -

LifeCyclePop.history['aLvl'] = LifeCyclePop.history['aNrm']*LifeCyclePop.history['pLvl']
aGro41=LifeCyclePop.history['aLvl'][41]/LifeCyclePop.history['aLvl'][40]
aGro41NoU=aGro41[aGro41[:]>0.2] # Throw out extreme outliers
aGro41NoU = aGro41NoU[aGro41NoU[:]<2]

LifeCyclePop.shocks

## wealth distribution  
wealth_dist=plt.hist(np.log(LifeCyclePop.history['aLvl'].flatten()+1e-5),bins=100)

# Plot the distribution of growth rates of wealth between age 65 and 66 (=25 + 41)
n, bins, patches = plt.hist(aGro41NoU,50,density=True)

# ## Wealthy over life cycle  

A_life = LifeCyclePop.history['aLvl'].mean(axis=1)

# +
import pandas as pd
SCF_profile = pd.read_pickle('data/SCF_age_profile.pkl')

#SCF_profile['mv_wealth'] = SCF_profile['av_wealth'].rolling(3).mean()
## plot life cycle profile

age_lc = SCF_profile.index

fig, ax = plt.subplots(figsize=(15,8))
plt.title('Life cycle profile of wealth and consumption')
ax.plot(age_lc[1:],
        np.log(A_life),
       'r-',
       label='model from HARK')

ax2 = ax.twinx()
ax2.set_ylim([10.5,15])
ax2.bar(age_lc[1:],
        np.log(SCF_profile['av_wealth'][1:]),
       #'k--',
       label='SCF (RHS)')

ax.set_xlabel('Age')
ax.set_ylabel('Log wealth')
ax2.set_ylabel('Log wealth SCF')
ax.legend(loc=1)
ax2.legend(loc=2)
