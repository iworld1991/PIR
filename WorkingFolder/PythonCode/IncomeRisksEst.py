# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
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

# ## Income Risks Estimation 
#
# This notebook draws from the integrated moving average (IMA) process class to estimate the income risks using PSID data (annual/biennial) and SIPP panel (monthly).

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import pandas as pd
import copy as cp


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
# -

from IncomeProcess import IMAProcess as ima

# + {"code_folding": []}
## creating an instance used for data estimation 

t = 10
ma_nosa = np.array([1])  ## ma coefficient without serial correlation
p_sigmas = np.random.uniform(0,1,t) ## allowing for time-variant shocks 
pt_ratio = 0.33
t_sigmas = pt_ratio * p_sigmas # sizes of the time-varying permanent volatility
sigmas = np.array([p_sigmas,
                   t_sigmas])

dt = ima(t = t,
         ma_coeffs = ma_nosa)
dt.sigmas = sigmas


# + {"code_folding": [2]}
### define the general function

def estimate_sample(sample):
    """
    this function take a sample of the first differences of income in different periods (column) of all individuals (row)
    and returns the estimates of the potentially time-varying permanent and transitory sigmas. 
    """
    data = np.array(sample)
    data_mean = np.nanmean(data,axis=0)
    data_var = ma.cov(ma.masked_invalid(data), rowvar=False)
    moms_data = {'Mean':data_mean,
                 'Var':data_var}
    ## initialize 
    dt_data_est = cp.deepcopy(dt)
    t_data = len(data_var)+1
    dt_data_est.t = t_data
    dt_data_est.GetDataMoments(moms_data)
    para_guess_this = np.ones(2*t_data + dt_data_est.ma_q)
    
    ## estimation
    data_para_est = dt_data_est.EstimatePara(method='BFGS',
                               para_guess = para_guess_this)
    
    return data_para_est


# -

# ### Estimation using SIPP data (yearly)

## SIPP data 
SIPP_Y = pd.read_stata('../../../SIPP/sipp_matrix_Y2Y.dta',
                    convert_categoricals=False)   
SIPP_Y.index = SIPP_Y['uniqueid']
SIPP_Y = SIPP_Y.drop(['uniqueid'], axis=1)
SIPP_Y = SIPP_Y.dropna(axis=0,how='all')

# +
## different samples 

education_groups = [1, #'HS dropout',
                   2, # 'HS graduate',
                   3] #'college graduates/above'
gender_groups = [1, #'male',
                2] #'female'

#byear_groups = list(np.array(SIPP.byear_5yr.unique(),dtype='int32'))

age_groups = list(np.array(SIPP_Y.age_5yr.unique(),dtype='int32'))

group_by = ['educ','gender','age_5yr']
all_drop = group_by #+['age_h','byear_5yr']

## full sample 
sample_full =  SIPP_Y.drop(all_drop,axis=1)

## sub sample 
sub_samples = []
para_est_list = []
sub_group_names = []

for edu in education_groups:
    for gender in gender_groups:
        for age5 in age_groups:
            belong = (SIPP_Y['educ']==edu) & (SIPP_Y['gender']==gender) & (SIPP_Y['age_5yr']==age5)
            obs = np.sum(belong)
            #print(obs)
            if obs > 1:
                sample = SIPP_Y.loc[belong].drop(all_drop,axis=1)
                sub_samples.append(sample)
                sub_group_names.append((edu,gender,age5))

# +
## estimation for full sample 

data_para_est_full = estimate_sample(sample_full)
# -

sample_full.columns

## time stamp 
year_str = [string.replace('lwage_id_shk_y2y_gr','') for string in sample_full.columns if 'lwage_id_shk_y2y_gr' in string]
years = np.array(year_str) 

years

# + {"code_folding": [3]}
## plot estimate 

lw = 3
for i,paras_est in enumerate([data_para_est_full]):
    print('whole sample')
    fig = plt.figure(figsize=([13,12]))
    this_est = paras_est
    p_risk = abs(this_est[1][0][1:])
    p_risk[p_risk<1e-5]= np.nan ## replace non-identified with nan
    print('Average permanent risk (std): ',str(np.nanmean(np.sqrt(p_risk))))
    t_risk = abs(this_est[1][1][1:])
    t_risk[t_risk<1e-5]= np.nan ## replace non-identified with nan
    print('Average transitory risk (std): ',str(np.nanmean(np.sqrt(t_risk))))
    plt.subplot(2,1,1)
    plt.title('Permanent Risk')
    plt.plot(years,
             p_risk,
             'bv',
             markersize=20,
             label='Estimation')
    plt.ylim(0.0,0.5)
    plt.ylabel('std of permanent risk')
    plt.xticks(years,
               rotation='vertical')
    plt.grid(True)

    plt.subplot(2,1,2)
    plt.title('Transitory Risk')
    plt.plot(years,
             t_risk,
             'bv',
             markersize=20,
             label='Estimation')
    plt.xticks(years,
               rotation='vertical')
    plt.ylim(0.0,0.35)
    plt.ylabel('std of transitory risk')
    #plt.legend(loc=0)
    plt.grid(True)
    #plt.savefig('../Graphs/sipp/permanent-transitory-risk-yearly.jpg')

# +
sipp_ests = {'YearlyPermanent':np.nanmean(np.sqrt(p_risk)),
                   'YearlyTransitory':np.nanmean(np.sqrt(t_risk))}

print(sipp_ests)
# -

# ### Estimation using SIPP data (quarterly)
#
#

## SIPP data 
SIPP_Q = pd.read_stata('../../../SIPP/sipp_matrix_Q.dta',
                    convert_categoricals=False)   
SIPP_Q.index = SIPP_Q['uniqueid']
SIPP_Q = SIPP_Q.drop(['uniqueid'], axis=1)
SIPP_Q = SIPP_Q.dropna(axis=0,how='all')

# + {"code_folding": [0, 2, 5]}
## different samples 

education_groups = [1, #'HS dropout',
                   2, # 'HS graduate',
                   3] #'college graduates/above'
gender_groups = [1, #'male',
                2] #'female'

#byear_groups = list(np.array(SIPP.byear_5yr.unique(),dtype='int32'))

age_groups = list(np.array(SIPP_Q.age_5yr.unique(),dtype='int32'))

group_by = ['educ','gender','age_5yr']
all_drop = group_by #+['age_h','byear_5yr']

## full sample 
sample_full =  SIPP_Q.drop(all_drop,axis=1)

## sub sample 
sub_samples = []
para_est_list = []
sub_group_names = []

for edu in education_groups:
    for gender in gender_groups:
        for age5 in age_groups:
            belong = (SIPP_Q['educ']==edu) & (SIPP_Q['gender']==gender) & (SIPP_Q['age_5yr']==age5)
            obs = np.sum(belong)
            #print(obs)
            if obs > 1:
                sample = SIPP_Q.loc[belong].drop(all_drop,axis=1)
                sub_samples.append(sample)
                sub_group_names.append((edu,gender,age5))

# +
## estimation for full sample 

data_para_est_full = estimate_sample(sample_full)

## time stamp 
quarter_str = [string.replace('lwage_Q_id_shk_gr','') for string in sample_full.columns if 'lwage_Q_id_shk_gr' in string]
quarter_str = [qst[:4]+'Q'+qst[4:] for qst in quarter_str]
quarters = np.array(quarter_str) 

# +
## plot estimate 

lw = 3
for i,paras_est in enumerate([data_para_est_full]):
    print('whole sample')
    fig = plt.figure(figsize=([13,12]))
    this_est = paras_est
    p_risk = abs(this_est[1][0][1:])
    t_risk = abs(this_est[1][1][1:])
    p_risk[p_risk<1e-6]= np.nan
    t_risk[t_risk<1e-6]= np.nan
    plt.subplot(2,1,1)
    plt.title('Permanent Risk')
    plt.plot(quarters,
             p_risk,
             'c--o',
             markersize=10,
             #markerfacecolor='none',
             lw=lw,
             label='Estimation')
    plt.ylabel('std of permanent risk')
    plt.xticks(quarters,
               rotation='vertical')
    plt.grid(True)

    plt.subplot(2,1,2)
    plt.title('Transitory Risk')
    plt.plot(quarters,
             t_risk, 
             'c--o',
             markersize=10,
             #markerfacecolor='none',
             lw=lw,
             label='Estimation')
    plt.xticks(quarters,
               rotation='vertical')
    plt.ylabel('std of transitory risk')
    plt.legend(loc=0)
    plt.grid(True)
    plt.subplots_adjust(
                    #left=0.1,
                    #bottom=0.1, 
                    #right=0.9, 
                   # top=0.9, 
                    #wspace=0.4, 
                    hspace=0.4)
    #plt.savefig('../Graphs/sipp/permanent-transitory-risk-quarterly.jpg')

# +
sipp_ests['QuarterlyPermanent']=np.nanmean(np.sqrt(p_risk))
sipp_ests['QuarterlyTransitory']=np.nanmean(np.sqrt(t_risk))

print(sipp_ests)

# +
sipp_est_df = pd.DataFrame.from_dict(sipp_ests, orient='index')

# Export the DataFrame as LaTeX
latex_table = sipp_est_df.to_latex(index=True)

# Print or save the LaTeX table
print(latex_table)
# -

# ### Estimation using SIPP data (monthly)

# + {"code_folding": []}
## SIPP data 
SIPP = pd.read_stata('../../../SIPP/sipp_matrix.dta',
                    convert_categoricals=False)   
SIPP.index = SIPP['uniqueid']
SIPP = SIPP.drop(['uniqueid'], axis=1)
SIPP = SIPP.dropna(axis=0,how='all')
#SIPP = SIPP.dropna(axis=1,how='all')

#SIPP=SIPP.dropna(subset=['byear_5yr'])
#SIPP['byear_5yr'] = SIPP['byear_5yr'].astype('int32')
# -

SIPP.columns

SIPP.dtypes

# + {"code_folding": [0]}
## different samples 

education_groups = [1, #'HS dropout',
                   2, # 'HS graduate',
                   3] #'college graduates/above'
gender_groups = [1, #'male',
                2] #'female'

#byear_groups = list(np.array(SIPP.byear_5yr.unique(),dtype='int32'))

age_groups = list(np.array(SIPP.age_5yr.unique(),dtype='int32'))

group_by = ['educ','gender','age_5yr']
all_drop = group_by #+['age_h','byear_5yr']

## full sample 
sample_full =  SIPP.drop(all_drop,axis=1)

## sub sample 
sub_samples = []
para_est_list = []
sub_group_names = []

for edu in education_groups:
    for gender in gender_groups:
        for age5 in age_groups:
            belong = (SIPP['educ']==edu) & (SIPP['gender']==gender) & (SIPP['age_5yr']==age5)
            obs = np.sum(belong)
            #print(obs)
            if obs > 1:
                sample = SIPP.loc[belong].drop(all_drop,axis=1)
                sub_samples.append(sample)
                sub_group_names.append((edu,gender,age5))

# + {"code_folding": []}
## estimation for full sample 

data_para_est_full = estimate_sample(sample_full)

# + {"code_folding": []}
## time stamp 
months_str = [string.replace('lwage_id_shk_gr','') for string in sample_full.columns if 'lwage_id_shk_gr' in string]
months = np.array(months_str)
#months_str2 = [mst[:4]+'M'+mst[4:] for mst in months_str]
#months2 = np.array(months_str2) # notice I dropped the first month 


# + {"code_folding": [0]}
## plot estimate 

lw = 3
for i,paras_est in enumerate([data_para_est_full]):
    print('whole sample')
    fig = plt.figure(figsize=([13,12]))
    this_est = paras_est
    p_risk = abs(this_est[1][0][1:])
    t_risk = abs(this_est[1][1][1:])
    p_risk[p_risk<1e-4]= np.nan
    t_risk[t_risk<1e-4]= np.nan
    p_risk_mv = (p_risk[0:-2]+p_risk[1:-1]+p_risk[2:])/3
    t_risk_mv = (t_risk[0:-2]+t_risk[1:-1]+t_risk[2:])/3
    plt.subplot(2,1,1)
    plt.title('Permanent Risk')
    plt.plot(months[2:],
             p_risk_mv,
             'r--o',
             markersize=8,
             lw=lw,
             label='Estimation')
    plt.ylabel('std of permanent risk')
    plt.xticks(months[2::4],
               rotation='vertical')
    plt.grid(True)

    plt.subplot(2,1,2)
    plt.title('Transitory Risk')
    plt.plot(months[2:],
             t_risk_mv,
             'r--o',
             markersize=8,
             lw=lw,
             label='Estimation')
    plt.xticks(months[2::4],
               rotation='vertical')
    plt.ylabel('std of transitory risk')
    plt.legend(loc=0)
    plt.grid(True)
    plt.subplots_adjust(
                    #left=0.1,
                    #bottom=0.1, 
                    #right=0.9, 
                   # top=0.9, 
                    #wspace=0.4, 
                    hspace=0.4)
    plt.savefig('../Graphs/sipp/permanent-transitory-risk.jpg')

# + {"code_folding": [5]}
## generate a dataset of date, permanent and transitory 

est_df = pd.DataFrame()

#vols_est_sub_group = pd.DataFrame([])
for i,para_est in enumerate([data_para_est_full]):
    #print(i)
    #print(group_var)
    times = len(months)
    this_est = pd.DataFrame([list(months),
                             np.abs(para_est[1][0]), 
                             np.abs(para_est[1][1])] 
                           ).transpose()
    est_df = pd.concat([est_df,
                        this_est])
    
    
## post-processing
est_df.columns = ['YM','permanent','transitory']
est_df=est_df.dropna(how='any')
for var in ['YM']:
    est_df[var] = est_df[var].astype('int32')

est_df['permanent']= est_df['permanent'].astype('float')
est_df['transitory']=est_df['transitory'].astype('float')

# +
## replace extreme values 
for date in [201303,201401,201501,201601,201701]:
    est_df.loc[est_df['YM']==date,'permanent']=np.nan
    
est_df
# -

est_df[['permanent','transitory']].plot(figsize=(13,5))

## export to stata
est_df.to_stata('../OtherData/sipp/sipp_history_vol_decomposed.dta')

# + {"code_folding": [0]}
## estimation for sub-group

for sample in sub_samples:
    ## estimation
    data_para_est = estimate_sample(sample)
    para_est_list.append(data_para_est)

# + {"code_folding": []}
## generate a dataset of year, edu, gender, byear_5yr permanent and transitory 

est_df = pd.DataFrame()

#vols_est_sub_group = pd.DataFrame([])
for i,para_est in enumerate(para_est_list):
    #print(i)
    group_var = sub_group_names[i]
    #print(group_var)
    times = len(months)
    this_est = pd.DataFrame([[group_var[0]]*times,   ##educ
                             [group_var[1]]*times,   ## gender 
                             [group_var[2]]*times,   ## age_5yr
                             list(months),
                             np.abs(para_est[1][0]), 
                             np.abs(para_est[1][1])] 
                           ).transpose()
    est_df = pd.concat([est_df,
                         this_est])

# +
## post-processing
est_df.columns = group_by+['YM','permanent','transitory']
est_df=est_df.dropna(how='any')
for var in group_by+['YM']:
    est_df[var] = est_df[var].astype('int32')
    

est_df['permanent']=est_df['permanent'].astype('float')
est_df['transitory']=est_df['transitory'].astype('float')
# -

for date in [201303,201401,201501,201601,201701]:
    est_df.loc[est_df['YM']==date,'permanent']=np.nan

## export to stata
est_df.to_stata('../OtherData/sipp/sipp_history_vol_decomposed_edu_gender_age5.dta')

est_df.head()


