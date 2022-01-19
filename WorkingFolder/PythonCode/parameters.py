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


life_cycle_paras = {'ρ': 1, 'β': 0.98**(1/4), 'P': np.array([[0.9, 0.1],
       [0.1, 0.9]]), 'z_val': np.array([0., 1.]), 'σ_ψ': np.sqrt(0.01*4/11), 'σ_θ': np.sqrt(0.01*4), 'U': 0.0, 'LivPrb': 1.0-0.00625, 'σ_ψ_2mkv': np.array([0.01, 0.02]), 'σ_θ_2mkv': np.array([0.02, 0.04]), 'R': 1.01**(1/4), 'W': 1.0, 'T': 40, 'L': 60, 'G': np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1.]), 'unemp_insurance': 0.15, 'pension': 1.0, 'σ_ψ_init': 0.01, 'init_b': 0.0, 'λ': 0.0, 'λ_SS': 0.0, 'transfer': 0.0, 'bequest_ratio': 0.0,'kappa':1.7}
# -

with open("parameters.txt", "wb") as fp:
    pickle.dump(life_cycle_paras, fp)


