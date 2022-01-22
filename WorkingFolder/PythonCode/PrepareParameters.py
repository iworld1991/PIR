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

# +
## import age income profile 

age_profile = pd.read_stata('../OtherData/age_profile.dta')   

# +
## select age range for the model and turn it into an array 
lc_wages = np.array(age_profile[(age_profile['age']>=25) &(age_profile['age']<=64)]['wage_age'])
print(str(len(lc_wages)),'years since age 25')

## growth rates since initial age 
lc_G = lc_wages[1:]/lc_wages[:-1]
lc_G = np.insert(lc_G,0,1.0)

T = 40
L = 60

T_q = T*4
L_q = L*4


# + code_folding=[2]
## turn yearly number to quarterly number with interpolation 

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


## get the quarterly income profile in life cycle 
lc_G_q = y2q_interpolate(lc_G)
# -

## create a dictionary of parameters 
life_cycle_paras = {'ρ': 1, 'β': 0.98**(1/4), 'P': np.array([[0.9, 0.1],
       [0.1, 0.9]]), 'z_val': np.array([0., 1.]), 'σ_ψ': np.sqrt(0.01*4/11), 'σ_θ': np.sqrt(0.01*4), 'U': 0.0, 'LivPrb': 1.0-0.00625, 'σ_ψ_2mkv': np.array([0.01, 0.02]), 'σ_θ_2mkv': np.array([0.02, 0.04]), 'R': 1.01**(1/4), 'W': 1.0, 'T': T_q, 'L': L_q, 'G':lc_G_q, 'unemp_insurance': 0.15, 'pension': 1.0, 'σ_ψ_init': 0.01, 'init_b': 0.0, 'λ': 0.0, 'λ_SS': 0.0, 'transfer': 0.0, 'bequest_ratio': 0.0,'kappa':1.7}

with open("parameters.txt", "wb") as fp:
    pickle.dump(life_cycle_paras, fp)


