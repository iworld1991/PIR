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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## A life-cycle model of consumption/saving with experience-based income expectations and risk perceptions 
#
# ### Outline
#
# - This life-cycle model is an extension of Gourinchas and Parker (2002) in two following dimensions
#
#   - Instead of taking the size of permanent and transitory income risks as given as assumed by the economists, agents endogenously learn about the income risks and form income expectations based on their past experiences. Therefore, income expectations/variances are age-cohort specific 
#     - Depending on if there is subjective attribution, the income expectations and risk perceptions will be state-dependent. 
#     - From the point view of solving the dynamic problem, this means that the mean/variance of the perceived income at any given time $t$ become other two state variables, in addition to asset and saving. 
#       - They are exogeous state in the sense that agents' decision will not affect the law of the motion. Also, there will not be feed-back loop from the market outcome.
#     
#   - Instead of partial equilibrium, I explore the general equilibrium and its distributional features in the presence of heterogeneous agents in terms of age, experiences, and beliefs. 
#   
#   
# - In terms of the implementation, since the heterogenous perceptions are exogenous states that depend on the cross-sectional distributions of income realizations in the past, we can separate the perception formation problem from the standard life-cycle decision problem. 
#
#   - We need to solve the life-cycle problem for a list of possible grids of "perceived" parameters. For each of this problem, the policies functions are still only dependent on the age and assets. 
#   - Then we simulate the income realizations for a cross-sectional sample of N agents over a life cycle.
#      - There can be either idiosyncratic or aggregate risks.  
#   - The simulated histories will produce 
#       - life-cycle perception profiles of each individual based on the past income realizations. 
#       - then, the life-cycle profiles of consumption and assets of each individuals and its aggregate distributions using the perception-specific policy function solved from the begining.   
#       
#       - this gives us the aggregate consumption/savings, that may be potentially used to solve general equilibrium models. 


