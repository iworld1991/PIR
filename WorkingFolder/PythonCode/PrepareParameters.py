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

# + code_folding=[]
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numba as nb
from copy import copy
# -

# ### Age profile of income 

# +
## some life cycle paras 
T = 40
L = 60

T_q = T*4
L_q = L*4


# + code_folding=[0]
def y2q_interpolate(xs_y):
    
    """
    this function turns an array of yealry rates into quarterly rates via linear interpolation 
    """

    n_y = len(xs_y)
    y_id = np.arange(n_y)
    q_id = y_id*4

    n_q = 4*n_y
    xs_q = np.empty(n_q)
    
    for i_y in range(n_y):
        xs_q[4*i_y] = xs_y[i_y]
        for i_q in np.arange(1,4):
            q_id_this = 4*i_y+i_q
            xs_q[q_id_this] = np.interp(q_id_this,q_id,xs_y)
            
    return xs_q


# + code_folding=[2]
age_profile_data ='SCF'

if age_profile_data=='SIPP':
    ## import age income profile 

    age_profile = pd.read_stata('../OtherData/age_profile.dta')   

    ## select age range for the model and turn it into an array 
    lc_wages = np.array(age_profile[(age_profile['age']>=24) &(age_profile['age']<=64)]['wage_age'])
    #print(str(len(lc_wages)),'years since age 25')

    ## growth rates since initial age over life cicle before retirement
    lc_G = lc_wages[1:]/lc_wages[:-1]

    ## growth rates after retirement

    lc_G_rt = np.ones(L-T)
    lc_G_rt[0] = 0.6


    lc_G_full = np.concatenate([lc_G,lc_G_rt])
    assert len(lc_G_full) == L,'length of G needs to be equal to L'
    #lc_G_full = np.ones_like(lc_G_full)

    
    ## turn yearly number to quarterly number with interpolation 

    ## get the quarterly income profile over life cycle before retirement 
    lc_G_q = y2q_interpolate(lc_G)

    lc_G_q_rt = y2q_interpolate(lc_G_rt)

    lc_G_q_full = np.concatenate([lc_G_q,lc_G_q_rt])
    
    
elif age_profile_data=='SCF':
    ## import age income profile 
    SCF_profile = pd.read_pickle('data/SCF_age_profile.pkl')
    SCF_profile = SCF_profile[(SCF_profile.index>=24) & (SCF_profile.index<=85)]
    lc_p_incom = np.exp(np.array(SCF_profile['av_lnorminc']))
    lc_G_full = lc_p_incom[1:]/lc_p_incom[:-1] 
    assert len(lc_G_full) == L,'length of G needs to be equal to L'
    lc_G_q_full = y2q_interpolate(lc_G_full)

# -

# ### Income risk estimates 

# +
risks_est = pd.read_stata('../OtherData/sipp/sipp_history_vol_decomposed.dta')
## risks of permanent and transitory component 

σ_ψ_q_sipp = np.sqrt(risks_est['permanent']**2*3)
σ_θ_q_sipp = np.sqrt(risks_est['permanent']**2/3)

## p/t ratio 
kappas_sipp  = risks_est['permanent']/risks_est['transitory']
kappa_sipp = np.median(kappas_sipp.dropna())

# +
## risks of permanent and transitory component 

σ_ψ_q_sipp = np.sqrt(risks_est['permanent']**2*3)
σ_θ_q_sipp = np.sqrt(risks_est['permanent']**2/3)

## p/t ratio 
kappas_sipp  = risks_est['permanent']/risks_est['transitory']
kappa_sipp = np.median(kappas_sipp.dropna())
# -

# ### Initial conditions

# +
from statsmodels.stats.weightstats import DescrStatsW

## SCF data 
SCF2016 = pd.read_stata('rscfp2016.dta')
SCF2016 = SCF2016[SCF2016['age']==25]
SCF2016 = SCF2016.drop_duplicates(subset=['yy1'])
SCF2016 = SCF2016[['norminc','networth','wgt']]
SCF2016 = SCF2016[SCF2016['norminc']>0]

## permanent income at age 25 
pinc_SCF = SCF2016[['norminc','wgt']]
lpinc_SCF_pos, lpinc_SCF_wgt = np.log(pinc_SCF['norminc']),np.array(pinc_SCF['wgt'])

## wealth 
b_SCF = SCF2016[['networth','wgt']]
lb_SCF_pos, lb_SCF_wgt = b_SCF['networth'],np.array(b_SCF['wgt'])

## wealth to permanent income
SCF2016['networth2pinc'] = SCF2016['networth']/SCF2016['norminc']
b_SCF, b_wgt = SCF2016['networth2pinc'], SCF2016['wgt']

## compute data moments 
σ_ψ_init_SCF = DescrStatsW(lpinc_SCF_pos, weights=lpinc_SCF_wgt, ddof=1).std
σ_b_init_SCF = DescrStatsW(lb_SCF_pos, weights=lb_SCF_wgt, ddof=1).std
b_SCF = DescrStatsW(b_SCF,weights=b_wgt, ddof=1).mean ## annual 
b_q_SCF = b_SCF*4  ## quarterly  
# -

# ### subjective profile estiamtion 

## import subjective profile estimation results 
SCE_est_q = pd.read_pickle('data/subjective_profile_est_q.pkl')
SCE_est_y = pd.read_pickle('data/subjective_profile_est_y.pkl')
SCE_est_q = SCE_est_q['baseline']
SCE_est_y = SCE_est_y['baseline']

# + code_folding=[1]
## create a dictionary of parameters 
life_cycle_paras_q = {'ρ': 1.0, 
                    'β': 0.98**(1/4), 
                    'P': np.array([[0.18, 0.82],
                                   [0.04, 0.96]]), 
                    'z_val': np.array([0., 1.]), 
                    'σ_ψ': np.sqrt(0.15**2*4/11), 
                    'σ_θ': np.sqrt(0.1**2*4), 
                    'U': 0.0, 
                    'LivPrb': 1.0-0.00625, 
                    'R': 1.01**(1/4), 
                    'W': 1.0, 
                    'T': T_q, 
                    'L': L_q, 
                    'G':lc_G_q_full, 
                    'unemp_insurance': 0.15, 
                    'pension': 1.0, 
                    'σ_ψ_init': σ_ψ_init_SCF, 
                    'init_b': b_q_SCF, 
                    'λ': 0.0, 
                    'λ_SS': 0.0, 
                    'transfer': 0.0, 
                    'bequest_ratio': 0.0,
                    'κ':kappa_sipp,
                    
                    ## subjective profile
                    'q':SCE_est_q.loc['$q$'],
                    'p':SCE_est_q.loc['$p$'],
                    'σ_ψ_2mkv':np.array([SCE_est_q.loc['$\tilde\sigma^l_\psi$'],
                                       SCE_est_q.loc['$\tilde\sigma^h_\psi$']]),
                    'σ_θ_2mkv':np.array([SCE_est_q.loc['$\tilde\sigma^l_\theta$'],
                                       SCE_est_q.loc['$\tilde\sigma^h_\theta$']]),
                    'mho_2mkv':np.array([SCE_est_q.loc['$\tilde \mho^l$'],
                                         SCE_est_q.loc['$\tilde \mho^h$']]),
                    'E_2mkv':np.array([SCE_est_q.loc['$\tilde E^l$'],
                                      SCE_est_q.loc['$\tilde E^h$']])
                }
# -

life_cycle_paras_q

# + code_folding=[1]
## create a dictionary of parameters 
life_cycle_paras_y = {'ρ': 1.5, 
                    'β': 0.98, 
                    'P': np.array([[0.18, 0.82],
                                   [0.04, 0.96]]), 
                    'z_val': np.array([0., 1.]), 
                    'σ_ψ': np.sqrt(0.15**2), 
                    'σ_θ': np.sqrt(0.1**2), 
                    'U': 0.0, 
                    'LivPrb': 1.0-0.00625, 
                    'R': 1.01, 
                    'W': 1.0, 
                    'T': T, 
                    'L': L, 
                    'G':lc_G_full, 
                    'unemp_insurance': 0.15, 
                    'pension': 1.0, 
                    'σ_ψ_init': σ_ψ_init_SCF, 
                    'init_b': b_SCF, 
                    'λ': 0.0, 
                    'λ_SS': 0.0, 
                    'transfer': 0.0, 
                    'bequest_ratio': 0.0,
                    'κ':kappa_sipp,
                    
                    ## subjective profile
                    'q':SCE_est_y.loc['$q$'],
                    'p':SCE_est_y.loc['$p$'],
                    'σ_ψ_2mkv':np.array([SCE_est_y.loc['$\tilde\sigma^l_\psi$'],
                                       SCE_est_y.loc['$\tilde\sigma^h_\psi$']]),
                    'σ_θ_2mkv':np.array([SCE_est_y.loc['$\tilde\sigma^l_\theta$'],
                                       SCE_est_y.loc['$\tilde\sigma^h_\theta$']]),
                    'mho_2mkv':np.array([SCE_est_y.loc['$\tilde \mho^l$'],
                                         SCE_est_y.loc['$\tilde \mho^h$']]),
                    'E_2mkv':np.array([SCE_est_y.loc['$\tilde E^l$'],
                                      SCE_est_y.loc['$\tilde E^h$']])
                }
# -

life_cycle_paras_y

# ### Production function parameters 

production_paras_y={}
production_paras_y['K2Y ratio'] = 3.0
production_paras_y['W'] = 1.0
production_paras_y['α'] = 0.33
production_paras_y['δ'] = 0.025

## quarterly paras 
production_paras_q = copy(production_paras_y)
production_paras_q['K2Y ratio'] = production_paras_q['K2Y ratio']*4

# ### Export the parameters into a table used in the draft


life_cycle_paras_y_copy = copy(life_cycle_paras_y)

# +
del life_cycle_paras_y_copy['G']  
del life_cycle_paras_y_copy['σ_ψ_2mkv']  
del life_cycle_paras_y_copy['σ_θ_2mkv']  
del life_cycle_paras_y_copy['mho_2mkv']  
del life_cycle_paras_y_copy['E_2mkv']  
del life_cycle_paras_y_copy['P']  
del life_cycle_paras_y_copy['z_val']  
del life_cycle_paras_y_copy['U']  
del life_cycle_paras_y_copy['κ']  
del life_cycle_paras_y_copy['q']  
del life_cycle_paras_y_copy['p']  
del life_cycle_paras_y_copy['transfer']  


## rewrite some parameters' names

life_cycle_paras_y_copy['U2U'] = life_cycle_paras_y['P'][0,0]
life_cycle_paras_y_copy['E2E'] = life_cycle_paras_y['P'][1,1]

## rename some 
life_cycle_paras_y_copy['1-D'] =  life_cycle_paras_y_copy.pop('LivPrb')
life_cycle_paras_y_copy['μ'] =  life_cycle_paras_y_copy.pop('unemp_insurance')
life_cycle_paras_y_copy['b_init'] =  life_cycle_paras_y_copy.pop('init_b')

## rounding 
life_cycle_paras_y_copy['1-D'] = round(life_cycle_paras_y_copy['1-D'],3)
life_cycle_paras_y_copy['σ_ψ_init'] = round(life_cycle_paras_y_copy['σ_ψ_init'],3)
life_cycle_paras_y_copy['b_init'] = round(life_cycle_paras_y_copy['b_init'],3)

# +
## merge life-cycle and production paras into model paras

model_paras = copy(life_cycle_paras_y_copy)
model_paras.update(production_paras_y)
# -

model_paras

# + code_folding=[2, 16, 27]
# making blocks 

blocknames =['risk',
            'initial condition',
            'life cycle',
            'preference',
            'policy',
            'production']

prefernece = ['ρ','β']
lifecycle = ['T','L','1-D']
risk  = ['σ_ψ','σ_θ','U2U','E2E']
initial = ['σ_ψ_init','init_b','bequest_ratio']
policy = ['μ','λ','λ_SS']
production=['K2Y ratio','W','α','δ']

block_all= [risk,
            initial,
            lifecycle,
            prefernece,
            policy,
           production]

## create multiple layer dictionary 

model_paras_by_block = {}

for i,block in enumerate(block_all):
    model_paras_by_block[blocknames[i]] =  {k: v for k, v 
                                               in model_paras.items() 
                                               if k in block}
# -

model_paras_by_block

# +
model_paras_by_block_df = pd.DataFrame.from_dict({(i,j): model_paras_by_block[i][j] 
                           for i in model_paras_by_block.keys() 
                           for j in model_paras_by_block[i].keys()},
                       orient='index')

model_paras_by_block_df.columns =['values']
Mindex = pd.MultiIndex.from_tuples(list(model_paras_by_block_df.index), names=["block", "parameter"])
model_paras_by_block_df.index = Mindex
# -

model_paras_by_block_df['source']=''

model_paras_by_block_df

# +
## add source of the parameters 

model_paras_by_block_df.loc['risk','source']='Median estimates from the literature'
model_paras_by_block_df.loc['initial condition','source']='Estimated for age 25 in the 2016 SCF'
model_paras_by_block_df.loc[('initial condition','bequest_ratio'),'source']='assumption'
model_paras_by_block_df.loc['life cycle','source']='standard assumption'
model_paras_by_block_df.loc['preference','source']='standard assumption'
model_paras_by_block_df.loc['policy','source']='standard assumption'
model_paras_by_block_df.loc[('policy','λ'),'source']='endogenously determined'
model_paras_by_block_df.loc[('policy','λ_SS'),'source']='endogenously determined'

model_paras_by_block_df.loc['production','source']='standard assumption'
model_paras_by_block_df.loc[('production','W'),'source']='target values in steady state'
model_paras_by_block_df.loc[('production','K2Y ratio'),'source']='target values in steady state'


# +
## for latex symbols 
parameter_list = [para[1] for para in list(model_paras_by_block_df.index)]

para_latex = ['$\\sigma_\\psi$',
              '$\\sigma_\\theta$',
              '$U2U$',
              '$E2E$',
              '$\\sigma_\\psi^{\\text{init}}$',
              'bequest ratio',
              '$T$',
              '$L$',
              '$1-D$',
              '$\\rho$',
              '$\\beta$',
             '$\\lambda$',
             '$\\lambda_{SS}$',
             '$\\mu$',
             '$W$',
             'K2Y ratio',
             '$\\alpha$',
             '$\\delta$']

model_paras_by_block_df['parameter name']= para_latex

model_paras_by_block_df = model_paras_by_block_df[['parameter name','values','source']]

model_paras_by_block_df=model_paras_by_block_df.reset_index(level=1, drop=True)
# -

## export to excel 
model_paras_by_block_df.to_excel('../Tables/calibration.xlsx')

model_paras_by_block_df


