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

# ### Estimating Subjective Income Distribution

# + code_folding=[]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta
# -

from DensityEst import SynDensityStat,GeneralizedBetaEst

# +
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

# + code_folding=[]
### loading probabilistic data on monthly income growth  
IndSCE = pd.read_stata('../SurveyData/SCE/IncExpSCEProbIndM.dta')   


# +
#IndSCE.tail()
# -

## how many observations?
len(IndSCE)

## how many observations have density forecasts
bin_name_list = ['Q24_bin'+str(n) for n in range(1,11)]
IndSCE_sub = IndSCE.dropna(subset = bin_name_list)
len(IndSCE_sub)

# +
#IndSCE_sub[bin_name_list].head()

# + code_folding=[]
## survey-specific parameters 
nobs = len(IndSCE)
SCE_bins = np.array([-20,-12,-8,-4,-2,0,2,4,8,12,20]) ## -20 and +20 are a fake lower and upper bound 
print("There are "+str(len(SCE_bins)-1)+" bins in SCE")

# + code_folding=[]
## make an illustration plot 

SCE_bins_all = ['<']+[str(bin) for bin in SCE_bins[1:-1]]+['>']

SCE_bins_all_names = [str(SCE_bins_all[i]+' '+str(SCE_bins_all[i+1])) for i in range(len(SCE_bins_all)-1)]
SCE_bins_id = np.arange(len(SCE_bins_all_names))
probs_example = np.flip(np.array([IndSCE.iloc[81,:]['Q24_bin'+str(n)]/100 for n in range(1,11)]))


## plot 
fig, ax = plt.subplots(figsize=(10,5))

pl1 = ax.bar(SCE_bins_id, 
            probs_example,
             color='orange',
           alpha=0.5)
ax.set_xlabel('expected wage growth (%)')

ax.set_ylabel('probs')
ax.set_title('An example of density forecast of wage growth from SCE')
ax.set_xticks(SCE_bins_id, 
              labels=SCE_bins_all_names)

plt.savefig('../Graphs/sce/density_bin_example.pdf')

# +
## example estimate 

sim_est = GeneralizedBetaEst(SCE_bins,
                           probs_example)
print(sim_est)

sim_x = np.linspace(SCE_bins[0],SCE_bins[-1],200)
sim_pdf= beta.pdf(sim_x,sim_est[0],sim_est[1],loc=sim_est[2],scale=sim_est[3]-sim_est[2])

fig, ax = plt.subplots(figsize=(10,5))

ax.set_title('An example of density distribution estimation')

ax.set_xlim(-13,13)
ax.plot(sim_x,
         sim_pdf,
        label='Estimated pdf')

ax.bar(SCE_bins[1:],
        probs_example,
       color='orange',
        width = 3,
       alpha = 0.5,
      label='Survey answer')
ax.legend(loc=0)
plt.savefig('../Graphs/sce/density_bin_est_example.pdf')

# + code_folding=[]
##############################################
### attention: the estimation happens here!!!!!
###################################################


## creating positions 
index  = IndSCE.index
columns = ['IncMean','IncVar','IncSkew','IncKurt']
IndSCE_moment_est = pd.DataFrame(index = index,
                                 columns = columns)
ct = 0
## invoking the estimation
for i in range(nobs):
    print(i)
    ## take the probabilities (flip to the right order, normalized to 0-1)
    Inc = np.flip(np.array([IndSCE.iloc[i,:]['Q24_bin'+str(n)]/100 for n in range(1,11)]))
    print(Inc)
    try:
    #if not np.isnan(Inc).any():
        stats_est = SynDensityStat(SCE_bins,
                                   Inc)     
        if not np.isnan(stats_est['mean']).any():
            ct = ct+1
            IndSCE_moment_est['IncMean'][i] = stats_est['mean']
            print(stats_est['mean'])
            IndSCE_moment_est['IncVar'][i] = stats_est['variance']
            print(stats_est['variance'])
            IndSCE_moment_est['IncSkew'][i] = stats_est['skewness']
            print(stats_est['skewness'])
            IndSCE_moment_est['IncKurt'][i] = stats_est['kurtosis']
            print(stats_est['kurtosis'])
    except:
        pass
# -

print(str(ct) + ' observations are estimated.')

# + code_folding=[]
## redo the estimation for those failed the first time 

ct_nan = 0
for i in range(nobs):
    #Inc = np.flip(np.array([IndSCE.iloc[i,:]['Q24_bin'+str(n)]/100 for n in range(1,11)]))
    #if IndSCE_moment_est['IncMean'][i]== None and np.isnan(IndSCE.iloc[i,:]['Q24_bin1']) == False:
    #    ct_nan = ct_nan+1
    #    print(i)
    #    print(Inc)
    #    print(IndSCE_moment_est['IncMean'][i])
    #    try:
    #        stats_est = SynDensityStat(SCE_bins,Inc)
    #        if len(stats_est)>0:
    #            IndSCE_moment_est['IncMean'][i] = stats_est['mean']
    #            print(stats_est['mean'])
    #            IndSCE_moment_est['IncVar'][i] = stats_est['variance']
    #            print(stats_est['variance'])
    #            IndSCE_moment_est['IncSkew'][i] = stats_est['skewness']
    #            print(stats_est['skewness'])
    #            IndSCE_moment_est['IncKurt'][i] = stats_est['kurtosis']
    #            print(stats_est['kurtosis'])
    #    except:
    #        pass
    if IndSCE_moment_est['IncMean'][i]!= None and np.isnan(IndSCE_moment_est['IncMean'][i]) and np.isnan(IndSCE.iloc[i,:]['Q24_bin1']) == False:
        ct_nan = ct_nan+1
        print(i)
        print(Inc)
        print(IndSCE_moment_est['IncMean'][i])
        try:
            stats_est = SynDensityStat(SCE_bins,Inc)
            if len(stats_est)>0:
                IndSCE_moment_est['IncMean'][i] = stats_est['mean']
                print(stats_est['mean'])
                IndSCE_moment_est['IncVar'][i] = stats_est['variance']
                print(stats_est['variance'])
                IndSCE_moment_est['IncSkew'][i] = stats_est['skewness']
                print(stats_est['skewness'])
                IndSCE_moment_est['IncKurt'][i] = stats_est['kurtosis']
                print(stats_est['kurtosis'])
        except:
            pass
# -

print(ct_nan)

### exporting moments estimates to pkl
IndSCE_est = pd.concat([IndSCE,IndSCE_moment_est], join='inner', axis=1)
IndSCE_est.to_pickle("../../../IncExpProjectShadow/WorkingFolder/PythonCode/IndSCEDstIndM.pkl")
IndSCE_pk = pd.read_pickle('../../../IncExpProjectShadow/WorkingFolder/PythonCode/IndSCEDstIndM.pkl')

columns_keep = ['date','year','month','userid','tenure','IncMean','IncVar','IncSkew','IncKurt']
IndSCE_pk_new = IndSCE_pk[columns_keep]

IndSCE_pk_new.tail()

IndSCE_pk_new =IndSCE_pk_new.astype({'IncMean': 'float',
                                     'IncVar': 'float',
                                     'IncSkew':'float',
                                     'IncKurt':'float'})

## export to stata
IndSCE_pk_new.to_stata('../SurveyData/SCE/IncExpSCEDstIndM.dta')

# + code_folding=[]
### Robustness checks: focus on big negative mean estimates 
sim_bins_data = SCE_bins
print(str(sum(IndSCE_pk['IncMean']<=0))+' abnormals')
ct=0
figure=plt.plot()
for id in IndSCE_pk.index[IndSCE_pk['IncMean']<-8]:
    print(id)
    print(IndSCE_pk['IncMean'][id])
    sim_probs_data= np.flip(np.array([IndSCE['Q24_bin'+str(n)][id]/100 for n in range(1,11)]))
    plt.bar(sim_bins_data[1:],sim_probs_data)
    print(sim_probs_data)
    stats_est=SynDensityStat(SCE_bins,sim_probs_data)
    print(stats_est['mean'])
# -


