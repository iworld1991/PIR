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

# + code_folding=[]
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numba as nb
from copy import copy

# +
## figure plotting configurations


plt.style.use('seaborn')
plt.rcParams["font.family"] = "Times New Roman" #'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['axes.labelweight'] = 'bold'

## Set the plotting style
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


# -

# ### Survival probability

#https://www.ssa.gov/oact/STATS/table4c6.html
life_table = pd.read_excel("../OtherData/lifetable.xlsx",sheet_name='data')
life_table['av_survial'] = 1-(life_table['male_D']+life_table['female_D'])/2
LivProbs_y = np.array(life_table[(life_table['age']>25)&(life_table['age']<=25+L)]['av_survial'])
LivProbs_q = y2q_interpolate(LivProbs_y)

# ### Population growth rate 
#

pop_n = 0.005

# ### Age profile of income 

# + code_folding=[6, 39]
#############################################
### Choose the data source of age profile here
#############################################

age_profile_data ='SIPP'

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
    lc_G_rt[0] = 1/np.cumprod(lc_G)[-1]


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
    SCF_profile = SCF_profile[(SCF_profile.index>=24) & (SCF_profile.index<=65)]
    ## make age profile using age polynomial regressions 
    
    import statsmodels.api as sm
    # Extract age and lwage_income from the data Series
    age = SCF_profile.index+25
    lwage_income = SCF_profile['av_lwage_income']
        
    # Create a design matrix with polynomial features
    X = np.column_stack((age, age**2,age**3,age**4))  # Third-order polynomial
    # Add a constant term for the intercept
    X = sm.add_constant(X)
    # Fit the OLG-regression model
    ols_results = sm.OLS(lwage_income, X).fit()
    # Predict values using the fitted model
    predicted_values = ols_results.predict(X)
    SCF_profile['av_lwage_income_pr'] = predicted_values

    lc_p_incom = np.exp(np.array(SCF_profile['av_lwage_income_pr']))
    lc_G = lc_p_incom[1:]/lc_p_incom[:-1]
    
    lc_G_rt = np.ones(L-T)
    lc_G_rt[0] = 1/np.cumprod(lc_G)[-1]

    lc_G_full = np.concatenate([lc_G,lc_G_rt])
    assert len(lc_G_full) == L,'length of G needs to be equal to L'
    
    lc_G_q_full = y2q_interpolate(lc_G_full)


# + code_folding=[0, 2]
if __name__ == "__main__":
    plt.title('Determinstic life-cycle wage profile')
    plt.plot(np.cumprod(lc_G_full),
            'k-v')
    plt.savefig('../Graphs/sipp/age_wage_profile.pdf')

# + code_folding=[0]
## subjective growth expectations 

SCE = pd.read_stata('../SurveyData/SCE/IncExpSCEProbIndM.dta')   
SCE = SCE.rename(columns={'Q24_mean': 'incexp',
                       'Q24_var': 'incvar',
                       'Q24_iqr': 'inciqr',
                       'Q24_rmean':'rincexp',
                       'Q24_rvar': 'rincvar',
                       'Q13new':'UE_s',
                       'Q22new':'UE_f'
                       })
SCE = SCE.rename(columns = {'D6':'HHinc',
                          'Q10_1':'fulltime',
                          'Q10_2':'parttime',
                          'Q12new':'selfemp',
                          'Q32':'age',
                          'Q33':'gender',
                          'Q36':'educ'})

lc_G_sub = np.array(1+SCE[(SCE['age']>=26) &(SCE['age']<64)].groupby('age')['rincexp'].mean())
## growth rates after retirement
lc_G_rt_sub = np.ones(L-T)
#lc_G_rt[0] = 1/np.cumprod(lc_G)[-1]
#lc_G_rt = lc_G_rt*lc_G[-1]
lc_G_full_sub = np.concatenate([lc_G_sub,lc_G_rt_sub])
assert len(lc_G_full_sub) == L,'length of G needs to be equal to L'
lc_G_q_full_sub = y2q_interpolate(lc_G_full_sub)
# -

if __name__ == "__main__":
    plt.title('subjective life-cycle profile')
    plt.plot(np.cumprod(lc_G_full_sub))

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

# +
## subjective constant profile 

sigma_eps_sub = np.sqrt(1/(1+kappa_sipp**2)*0.04**2) ## 0.04 is an upper bound for the average PR in standard deviation from SCE 
sigma_psi_sub = sigma_eps_sub*kappa_sipp

if __name__ == "__main__":    
    print('perceived transitory PR:',sigma_eps_sub)
    print('perceived permanent PR:',sigma_psi_sub)

# +
## subjective markov profile 


## import subjective profile estimation results 
SCE_est_q = pd.read_pickle('data/subjective_profile_est_q.pkl')
SCE_est_y = pd.read_pickle('data/subjective_profile_est_y.pkl')
SCE_est_q = SCE_est_q['baseline']
SCE_est_y = SCE_est_y['baseline']
P_sub_q = np.array([[SCE_est_q.loc['$q$'],1-SCE_est_q.loc['$q$']],
                    [1-SCE_est_q.loc['$p$'],SCE_est_q.loc['$p$']]])
P_sub_y = np.array([[SCE_est_y.loc['$q$'],1-SCE_est_y.loc['$q$']],
                    [1-SCE_est_y.loc['$p$'],SCE_est_y.loc['$p$']]])
# -

SCE_est_q

# + code_folding=[]
## create a dictionary of parameters 
life_cycle_paras_q = {'ρ': 2.0, 
                    'β': 0.97**(1/4), 
                    'P': np.array([[0.18, 0.82],
                                   [0.04, 0.96]]), 
                    'z_val': np.array([0., 1.]), 
                    'σ_ψ': np.sqrt(0.15**2*4/11), 
                    'σ_θ': np.sqrt(0.15**2*4), 
                    'U': 0.0, 
                    'LivPrb': LivProbs_q, 
                    'n': (1+pop_n)**(1/4)-1,
                    'R': 1.01**(1/4), 
                    'W': 1.0, 
                    'T': T_q, 
                    'L': L_q, 
                    'G':lc_G_q_full, 
                    'G_sub':lc_G_q_full_sub, 
                    'unemp_insurance': 0.15, 
                    'pension': 0.65, 
                    'σ_ψ_init': σ_ψ_init_SCF, 
                    'init_b': b_q_SCF, 
                    'λ': 0.0, 
                    'λ_SS': 0.065, 
                    'transfer': 0.0, 
                    'bequest_ratio': 0.0,
                    'κ':kappa_sipp,
                     ## subjective profile
                    'σ_ψ_sub':sigma_psi_sub**2*4/11,
                    'σ_θ_sub':sigma_eps_sub**2*4,
                    'P_sub': P_sub_q,
                    'σ_ψ_2mkv':np.array([SCE_est_q.loc['$\tilde\sigma^l_\psi$'],
                                       SCE_est_q.loc['$\tilde\sigma^h_\psi$']]),
                    'σ_θ_2mkv':np.array([SCE_est_q.loc['$\tilde\sigma^l_\theta$'],
                                       SCE_est_q.loc['$\tilde\sigma^h_\theta$']])
                }


if '$\tilde \mho^l$' in SCE_est_q.index:

    life_cycle_paras_q['mho_2mkv'] = np.array([SCE_est_q.loc['$\tilde \mho^l$'],
                                             SCE_est_q.loc['$\tilde \mho^h$']])
    life_cycle_paras_q['E_2mkv'] = np.array([SCE_est_q.loc['$\tilde E^l$'],
                                          SCE_est_q.loc['$\tilde E^h$']])
    
else:
    life_cycle_paras_q['mho_2mkv'] = np.array([life_cycle_paras_q['P'][0,0],
                                               life_cycle_paras_q['P'][0,0]])
    life_cycle_paras_q['E_2mkv'] = np.array([life_cycle_paras_q['P'][1,1],
                                           life_cycle_paras_q['P'][1,1]])
# -

life_cycle_paras_q

# + code_folding=[]
## create a dictionary of parameters 
life_cycle_paras_y = {'ρ': 2.0, 
                    'β': 0.97, 
                    'P': np.array([[0.18, 0.82],
                                   [0.04, 0.96]]), 
                    'z_val': np.array([0., 1.]), 
                    'σ_ψ': np.sqrt(0.15**2), 
                    'σ_θ': np.sqrt(0.15**2), 
                    'U': 0.0, 
                    'LivPrb': LivProbs_y,
                    'n': pop_n,
                    'R': 1.01, 
                    'W': 1.0, 
                    'T': T, 
                    'L': L, 
                    'G':lc_G_full, 
                    'G_sub':lc_G_full_sub, 
                    'unemp_insurance': 0.15, 
                    'pension': 0.65, 
                    'σ_ψ_init': σ_ψ_init_SCF, 
                    'init_b': b_SCF, 
                    'λ': 0.0, 
                    'λ_SS': 0.065, 
                    'transfer': 0.0, 
                    'bequest_ratio': 0.0,
                    'κ':kappa_sipp,
                    ## subjective profile
                    'σ_ψ_sub':sigma_psi_sub,
                    'σ_θ_sub':sigma_eps_sub,
                    'P_sub': P_sub_y,
                    'σ_ψ_2mkv':np.array([SCE_est_y.loc['$\tilde\sigma^l_\psi$'],
                                       SCE_est_y.loc['$\tilde\sigma^h_\psi$']]),
                    'σ_θ_2mkv':np.array([SCE_est_y.loc['$\tilde\sigma^l_\theta$'],
                                       SCE_est_y.loc['$\tilde\sigma^h_\theta$']])
                }


if '$\tilde \mho^l$' in SCE_est_y.index:

    life_cycle_paras_y['mho_2mkv'] = np.array([SCE_est_y.loc['$\tilde \mho^l$'],
                                             SCE_est_y.loc['$\tilde \mho^h$']])
    life_cycle_paras_y['E_2mkv'] = np.array([SCE_est_y.loc['$\tilde E^l$'],
                                          SCE_est_y.loc['$\tilde E^h$']])
    
else:
    life_cycle_paras_y['mho_2mkv'] = np.array([life_cycle_paras_y['P'][0,0],
                                               life_cycle_paras_y['P'][0,0]])
    life_cycle_paras_y['E_2mkv'] = np.array([life_cycle_paras_y['P'][1,1],
                                           life_cycle_paras_y['P'][1,1]])
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
del life_cycle_paras_y_copy['G_sub']  
del life_cycle_paras_y_copy['σ_ψ_2mkv']
del life_cycle_paras_y_copy['σ_θ_2mkv']

if 'mho_2mkv' in life_cycle_paras_y_copy.keys():
    del life_cycle_paras_y_copy['mho_2mkv']  
if 'E_2mkv' in life_cycle_paras_y_copy.keys():
    del life_cycle_paras_y_copy['E_2mkv']  
del life_cycle_paras_y_copy['P']  
del life_cycle_paras_y_copy['z_val']  
del life_cycle_paras_y_copy['U']  
del life_cycle_paras_y_copy['κ']  
del life_cycle_paras_y_copy['P_sub']  
#del life_cycle_paras_y_copy['q']  
#del life_cycle_paras_y_copy['p']  
del life_cycle_paras_y_copy['transfer']  


## rewrite some parameters' names

life_cycle_paras_y_copy['U2U'] = life_cycle_paras_y['P'][0,0]
life_cycle_paras_y_copy['E2E'] = life_cycle_paras_y['P'][1,1]

## rename some 
#life_cycle_paras_y_copy['1-D'] =  life_cycle_paras_y_copy.pop('LivPrb')
life_cycle_paras_y_copy['μ'] =  life_cycle_paras_y_copy.pop('unemp_insurance')
life_cycle_paras_y_copy['b_init'] =  life_cycle_paras_y_copy.pop('init_b')

## rounding 
#life_cycle_paras_y_copy['1-D'] = round(life_cycle_paras_y_copy['1-D'],3)
life_cycle_paras_y_copy['σ_ψ_init'] = round(life_cycle_paras_y_copy['σ_ψ_init'],3)
life_cycle_paras_y_copy['b_init'] = round(life_cycle_paras_y_copy['b_init'],3)

# +
## merge life-cycle and production paras into model paras

model_paras = copy(life_cycle_paras_y_copy)
model_paras.update(production_paras_y)
# -

model_paras

# + code_folding=[]
# making blocks 

blocknames =['risk',
            'initial condition',
            'life cycle',
            'preference',
            'policy',
            'production',
            'subjective']

prefernece = ['ρ','β']
lifecycle = ['T','L','n',
             #'1-D'
            ]
risk  = ['σ_ψ','σ_θ','U2U','E2E']
initial = ['σ_ψ_init','init_b','bequest_ratio']
policy = ['μ','pension','λ','λ_SS']
production=['K2Y ratio','W','α','δ']
subjective= ['σ_ψ_sub','σ_θ_sub']

block_all= [risk,
            initial,
            lifecycle,
            prefernece,
            policy,
           production,
           subjective]

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

model_paras_by_block_df.loc['risk','source']='median estimates from the literature'
model_paras_by_block_df.loc['initial condition','source']='estimated for age 25 in the 2016 SCF'
model_paras_by_block_df.loc[('initial condition','bequest_ratio'),'source']='assumption'
model_paras_by_block_df.loc[('life cycle','T'),'source']='standard calibration'
model_paras_by_block_df.loc[('life cycle','L'),'source']='standard calibration'
model_paras_by_block_df.loc[('life cycle','n'),'source']='U.S. Census'
model_paras_by_block_df.loc[('preference','β'),'source']='calibrated to match average wealth/income ratio'
model_paras_by_block_df.loc[('preference','ρ'),'source']='standard calibration'
model_paras_by_block_df.loc['policy','source']='U.S. average'
model_paras_by_block_df.loc[('policy','λ'),'values']=np.nan
model_paras_by_block_df.loc[('policy','λ'),'source']='endogenously determined'
model_paras_by_block_df.loc[('policy','λ_SS'),'source']='U.S. average'
model_paras_by_block_df.loc[('policy','λ_SS'),'values']=np.nan

model_paras_by_block_df.loc['production','source']='standard calibration'
model_paras_by_block_df.loc[('production','W'),'source']='target values in steady state'
model_paras_by_block_df.loc[('production','K2Y ratio'),'source']='target values in steady state'
model_paras_by_block_df.loc['subjective','source']='estimated from SCE'


# +
## for latex symbols 
parameter_list = [para[1] for para in list(model_paras_by_block_df.index)]

para_latex = ['$\\sigma_\\psi$',
              '$\\sigma_\\theta$',
              '$U2U$',
              '$E2E$',
              '$\\sigma_\\psi^{\\text{init}}$',
              'bequest ratio',
               '$n$',
              '$T$',
              '$L$',        
              #'$1-D$',
              '$\\rho$',
              '$\\beta$',
              '$\\mathbb{S}$',
             '$\\lambda$',
             '$\\lambda_{SS}$',
             '$\\mu$',
             '$W$',
             'K2Y ratio',
             '$\\alpha$',
             '$\\delta$',
              '$\\sigma_\\psi^{\\text{sub}}$',
             '$\\sigma_\\theta^{\\text{sub}}$']

model_paras_by_block_df['parameter name']= para_latex

model_paras_by_block_df = model_paras_by_block_df[['parameter name','values','source']]

model_paras_by_block_df=model_paras_by_block_df.reset_index(level=1, drop=True)
# -

## export to excel 
model_paras_by_block_df.to_excel('../Tables/calibration.xlsx')

model_paras_by_block_df


