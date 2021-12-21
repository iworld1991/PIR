# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Income Risks Estimation 
#
# This noteobok contains the following
#
#  - Estimation functions of time-varying income risks for an integrated moving average(IMA) process of income/earnings
#  - It uses the function to estimate the realized risks using PSID data(annual/biennial) and SIPP panel(monthly) 

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
#from scipy.optimize import minimize
import pandas as pd
import copy as cp

from IncomeProcess import IMAProcess as ima

# + {"code_folding": [0]}
## debugging test of the data 

t = 100
ma_nosa = np.array([1])  ## ma coefficient without serial correlation
p_sigmas = np.arange(t)  # sizes of the time-varying permanent volatility 
p_sigmas_rw = np.ones(t) # a special case of time-invariant permanent volatility, random walk 
p_sigmas_draw = np.random.uniform(0,1,t) ## allowing for time-variant shocks 

pt_ratio = 0.33
t_sigmas = pt_ratio * p_sigmas_draw # sizes of the time-varyingpermanent volatility
sigmas = np.array([p_sigmas_draw,
                   t_sigmas])

dt = ima(t = t,
         ma_coeffs = ma_nosa)
dt.sigmas = sigmas
sim_data = dt.SimulateSeries(n_sim = 2000)
sim_moms = dt.SimulatedMoments()

# + {"code_folding": [0]}
## get the computed moments 

comp_moms = dt.ComputeGenMoments()

av_comp = comp_moms['Mean']
cov_var_comp = comp_moms['Var']
var_comp = dt.AutocovarComp(step=0) #np.diagonal(cov_var_comp)
autovarb1_comp = dt.AutocovarComp(step=-1)  #np.array([cov_var_comp[i,i+1] for i in range(len(cov_var_comp)-1)]) 

# + {"code_folding": [0]}
## get the simulated moments 
av = sim_moms['Mean']
cov_var = sim_moms['Var']
var = dt.Autocovar(step = 0)   #= np.diagonal(cov_var)
autovarb1 = dt.Autocovar(step = -1) #np.array([cov_var[i,i+1] for i in range(len(cov_var)-1)]) 

# + {"code_folding": [0]}
## plot simulated moments of first diff 

plt.figure(figsize=((20,4)))

plt.subplot(1,4,1)
plt.title(r'$\sigma_{\theta,t},\sigma_{\epsilon,t}$')
plt.plot(p_sigmas_draw,label='sigma_p')
plt.plot(t_sigmas,label='sigma_t')
plt.legend(loc=0)

plt.subplot(1,4,2)
plt.title(r'$\Delta(y_t)$')
plt.plot(av,label='simulated')
plt.plot(av_comp,label='computed')
plt.legend(loc=0)

plt.subplot(1,4,3)
plt.title(r'$Var(\Delta y_t)$')
plt.plot(var,label='simulated')
plt.plot(var_comp,label='computed')
plt.legend(loc=0)

plt.subplot(1,4,4)
plt.title(r'$Cov(\Delta y_t,\Delta y_{t+1})$')
plt.plot(autovarb1,label='simulated')
plt.plot(autovarb1_comp,label='computed')
plt.legend(loc = 0)

# + {"code_folding": [0]}
## robustness check if the transitory risks is approximately equal to the assigned level

sigma_t_est = np.array(np.sqrt(abs(autovarb1)))
plt.plot(sigma_t_est,'r-',label=r'$\widehat \sigma_{\theta,t}$')
plt.plot(t_sigmas[1:-1],'b-.',label=r'$\sigma_{\theta,t}$')  # the head and tail trimmed
plt.legend(loc=1)
# -

# ### Estimation

# + {"code_folding": [0]}
## some fake data moments with alternative parameters

## fix ratio of p and t risks
pt_ratio_fake = 0.66
t_sigmas = pt_ratio_fake * p_sigmas_draw # sizes of the time-varyingpermanent volatility

## both p and t risks are draws
p_sigmas_draw = np.random.uniform(0,1,t)
t_sigmas_draw = np.random.uniform(0,1,t)

sigmas = np.array([p_sigmas_draw,
                   t_sigmas_draw])

dt_fake = ima(t = t,
              ma_coeffs = ma_nosa)
dt_fake.sigmas = sigmas
data_fake = dt_fake.SimulateSeries(n_sim = 5000)
moms_fake = dt_fake.SimulatedMoments()
# -

# ### Estimation using fake data

# + {"code_folding": [0]}
## estimation of income risks 

dt_est = cp.deepcopy(dt)
dt_est.GetDataMoments(moms_fake)

para_guess_this = np.ones(2*t  + dt_est.ma_q)  # make sure the length of the parameters are right 

# + {"code_folding": [0]}
para_est = dt_est.EstimatePara(method='BFGS',
                               para_guess = para_guess_this)


# + {"code_folding": []}
## check the estimation and true parameters 

fig = plt.figure(figsize=([10,4]))

plt.subplot(1,2,1)
plt.title('Permanent Risk')
plt.plot(dt_est.para_est[1][0][1:].T**2,'r-',label='Estimation')
plt.plot(dt_fake.sigmas[0][1:]**2,'-*',label='Truth')

plt.subplot(1,2,2)
plt.title('Transitory Risk')
plt.plot(dt_est.para_est[1][1][1:].T**2,'r-',label='Estimation')
plt.plot(dt_fake.sigmas[1][1:]**2,'-*',label='Truth')
plt.legend(loc=0)


# + {"code_folding": [2]}
### define the general function

def estimate_sample(sample):
    """
    this function take a sample of the first differences of income in different time(column) of all individuals(row)
    and returns the estimates of the permanent and transitory sigmas. 
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

# ### Estimation using SIPP data

# +
## SIPP data 
SIPP = pd.read_stata('../../../SIPP/sipp_matrix.dta',
                    convert_categoricals=False)   
SIPP.index = SIPP['uniqueid']
SIPP = SIPP.drop(['uniqueid'], axis=1)
SIPP = SIPP.dropna(axis=0,how='all')
SIPP = SIPP.dropna(axis=1,how='all')

#SIPP=SIPP.dropna(subset=['byear_5yr'])
#SIPP['byear_5yr'] = SIPP['byear_5yr'].astype('int32')
# -

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

# +
## estimation for full sample 

data_para_est_full = estimate_sample(sample_full)
# -

## time stamp 
months_str = [string.replace('lwage_id_shk_gr','') for string in sample_full.columns if 'lwage_id_shk_gr' in string]
months = np.array(months_str)

# + {"code_folding": [3]}
## plot estimate 

lw = 3
for i,paras_est in enumerate([data_para_est_full]):
    print('whole sample')
    fig = plt.figure(figsize=([13,12]))
    this_est = paras_est
    plt.subplot(2,1,1)
    plt.title('Permanent Risk')
    plt.plot(months,
             this_est[1][0][1:].T**2,
             'r-o',
             lw=lw,
             label='Estimation')
    plt.xticks(rotation='vertical')
    plt.grid(True)

    plt.subplot(2,1,2)
    plt.title('Transitory Risk')
    plt.plot(months,
             this_est[1][1][1:].T**2,
             'r-o',
             lw=lw,
             label='Estimation')
    plt.xticks(rotation='vertical')
    plt.legend(loc=0)
    plt.grid(True)

# +
## estimation for sub-group

for sample in sub_samples:
    ## estimation
    data_para_est = estimate_sample(sample)
    para_est_list.append(data_para_est)

# +
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
                             [group_var[2]]*times,   ## 
                             list(months),
                             para_est[1][0],
                             para_est[1][1]]).transpose()
    est_df = est_df.append(this_est)

# +
## post processing
est_df.columns = group_by+['YM','permanent','transitory']
est_df=est_df.dropna(how='any')
for var in group_by+['YM']:
    est_df[var] = est_df[var].astype('int32')
    

est_df['permanent']=est_df['permanent'].astype('float')
est_df['transitory']=est_df['transitory'].astype('float')
# -

est_df.to_stata('../OtherData/sipp/sipp_history_vol_decomposed_edu_gender_age5.dta')

est_df.tail()

# ### Estimation using PSID data
#
#

# + {"code_folding": []}
## PSID data 
PSID = pd.read_stata('../../../PSID/J276289/psid_matrix.dta',
                    convert_categoricals=False)   
PSID.index = PSID['uniqueid']
PSID = PSID.drop(['uniqueid'], axis=1)
PSID = PSID.dropna(axis=0,how='all')
PSID = PSID.dropna(axis=1,how='all')

#PSID = PSID.rename(columns={'edu_i_g':'edu_i_g', ##
#                           'sex_h':'sex_h',   # 1 male 2 female 
#                           'byear_5yr':'byear_5yr'})

#PSID=PSID.dropna(subset=['byear_5yr'])

#PSID['byear_5yr'] = PSID['byear_5yr'].astype('int32')

# -

PSID.dtypes

# + {"code_folding": [0]}
## different samples 

education_groups = [1, #'HS dropout',
                   2, # 'HS graduate',
                   3] #'college graduates/above'
gender_groups = [1, #'male',
                2] #'male'

#byear_groups = list(np.array(PSID.byear_5yr.unique(),dtype='int32'))

age_groups = list(np.array(PSID.age_5yr.unique(),dtype='int32'))


group_by = ['edu_i_g','sex_h','age_5yr']
all_drop = group_by #+['age_h','byear_5yr']

## full sample 
sample_full =  PSID.drop(all_drop,axis=1)


## sub sample 
sub_samples = []
para_est_list = []
sub_group_names = []

for edu in education_groups:
    for gender in gender_groups:
        for age5 in age_groups:
            belong = (PSID['edu_i_g']==edu) & (PSID['sex_h']==gender) & (PSID['age_5yr']==age5)
            obs = np.sum(belong)
            #print(obs)
            if obs > 1:
                sample = PSID.loc[belong].drop(all_drop,axis=1)
                sub_samples.append(sample)
                sub_group_names.append((edu,gender,age5))

# + {"code_folding": []}
## estimation for full sample 

data_para_est_full = estimate_sample(sample_full)

# + {"code_folding": []}
## estimation for sub-group

for sample in sub_samples:
    ## estimation
    data_para_est = estimate_sample(sample)
    para_est_list.append(data_para_est)

# +
## time stamp 
#years_str = [string.replace('lwage_id_shk_gr','') for string in sample_full.columns if 'lwage_id_shk_gr' in string]
#years = np.array(years_str)

# + {"code_folding": []}
## time stamp 
t0 = 1971
tT = 2016
t_break = 1998 #the year when no annual data was released i.e. no 1998 data 
years = np.arange(t0+1,tT+2)
years=years.astype(int)

years_sub = np.concatenate((np.arange(t0+1,t_break),np.arange(t_break+1,tT+2,2)))
# -

years_sub

years

# +
## plot estimate 

lw = 3
for i,paras_est in enumerate([data_para_est_full]):
    print('whole sample')
    fig = plt.figure(figsize=([12,4]))
    this_est = paras_est
    plt.subplot(1,2,1)
    plt.title('Permanent Risk')
    plt.plot(years_sub,
             this_est[1][0][1:].T**2,
             'r-o',
             lw=lw,
             label='Estimation')
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.title('Transitory Risk')
    plt.plot(years_sub,
             this_est[1][1][1:].T**2,
             'r-o',
             lw=lw,
             label='Estimation')
    plt.legend(loc=0)
    plt.grid(True)

# + {"code_folding": [5]}
## generate a dataset of year, edu, gender, byear_5yr permanent and transitory 

est_df = pd.DataFrame()

#vols_est_sub_group = pd.DataFrame([])
for i,para_est in enumerate(para_est_list):
    #print(i)
    group_var = sub_group_names[i]
    #print(group_var)
    times = len(years_sub)
    this_est = pd.DataFrame([[group_var[0]]*times,   ##educ
                             [group_var[1]]*times,   ## gender 
                             [group_var[2]]*times,   ## 
                             list(years_sub),
                             para_est[1][0],
                             para_est[1][1]]).transpose()
    est_df = est_df.append(this_est)
# -

## post processing
est_df.columns = grouping_by+['year','permanent','transitory']
est_df=est_df.dropna(how='any')
for var in grouping_by+['year']:
    est_df[var] = est_df[var].astype('int32')

est_df.to_stata('../OtherData/psid/psid_history_vol_decomposed_edu_gender_age5.dta')

# ### Experienced volatility specific to cohort 

# + {"code_folding": []}
## full sample 
history_vols_whole = pd.DataFrame([list(years_sub),para_est_full[1][0],para_est_full[1][1]]).transpose()


## sub group 
history_vols_hsd = pd.DataFrame([list(years_sub),para_est_list[0][1][0],para_est_list[0][1][1]]).transpose()
history_vols_hsg = pd.DataFrame([list(years_sub),para_est_list[1][1][0],para_est_list[1][1][1]]).transpose()
history_vols_cg = pd.DataFrame([list(years_sub),para_est_list[2][1][0],para_est_list[2][1][1]]).transpose()

for dt in [history_vols_whole,
          history_vols_hsd,
          history_vols_hsg,
          history_vols_cg]:
    dt.columns = ['year','permanent','transitory']
# -

## whole data 
dataset_psid = pd.read_excel('../OtherData/psid/psid_history_vol.xls')
## education data
dataset_psid_edu = pd.read_excel('../OtherData/psid/psid_history_vol_edu.xls')

# + {"code_folding": []}
## for whole sample
names = ['whole'] ## whole sample/ high school dropout / high school graduate / college graduate above
for sample_id,sample in enumerate([history_vols_whole]):
    # prepare data 
    history_vols = dataset_psid
    history_vols['permanent'] = np.nan
    history_vols['transitory'] = np.nan
    
    
    for i in history_vols.index:
        #print(i)
        year = history_vols['year'].iloc[i]
        #print(year)
        born = history_vols['cohort'].iloc[i]-20
        #print(born)
        av_per_vol = np.mean(sample['permanent'].loc[(sample['year']>born) & 
                                                                (sample['year']<=year)] )
        #print(av_per_vol)
        av_tran_vol = np.mean(sample['transitory'].loc[(sample['year']>born) & 
                                                                (sample['year']<=year)])
        #print(av_tran_vol)
        history_vols['permanent'].iloc[i] = av_per_vol
        history_vols['transitory'].iloc[i] = av_tran_vol
        
    ## save to excel for further analysis 
    history_vols.to_excel('../OtherData/psid/psid_history_vol_decomposed_'+str(names[sample_id])+'.xlsx')

# + {"code_folding": []}
## for sub-education group 

# prepare data 
history_vols = dataset_psid_edu
history_vols['permanent'] = np.nan
history_vols['transitory'] = np.nan

samples = [history_vols_hsd,
           history_vols_hsg,
           history_vols_cg]
        

for i in history_vols.index:
    #print(i)
    year = history_vols['year'].iloc[i]
    #print(year)
    born = history_vols['cohort'].iloc[i]-20
    #print(born)
    edu = history_vols['edu'].iloc[i]
    #print(edu)
    
    sample = samples[edu-1] ## 1 hsd, 2 hsg, 3 cg 
    
    av_per_vol = np.mean(sample['permanent'].loc[(sample['year']>born) &
                                                 (sample['year']<=year)] )
    #print(av_per_vol)
    av_tran_vol = np.mean(sample['transitory'].loc[(sample['year']>born) &
                                                   (sample['year']<=year)])
    #print(av_tran_vol)
    history_vols['permanent'].iloc[i] = av_per_vol
    history_vols['transitory'].iloc[i] = av_tran_vol
        
## save to excel for further analysis 
history_vols.to_excel('../OtherData/psid/psid_history_vol_decomposed_edu.xlsx')

# + [markdown] {"code_folding": []}
# ### Estimation using simulated moments 

# + {"code_folding": []}
"""
para_guess_this2 = para_guess_this*0.3

bounds_this = ((0,1),) + ((0,0.5),)*(2*t)

para_est_sim = dt_est.EstimateParabySim(method='TNC',
                                        para_guess = para_guess_this2,
                                        options={'disp':True}
                                       )
                                       
"""

# + {"code_folding": []}
"""
## check the estimation and true parameters

fig = plt.figure(figsize=([10,4]))

plt.subplot(1,2,1)
plt.title('Permanent Risk')
plt.plot(dt_est.para_est_sim[1][0][1:].T**2,'r-',label='Estimation(sim)')
plt.plot(dt_fake.sigmas[0][1:]**2,'-*',label='Truth')


plt.subplot(1,2,2)
plt.title('Transitory Risk')
plt.plot(dt_est.para_est_sim[1][1][1:].T**2,'r-',label='Estimation(sim)')
plt.plot(dt_fake.sigmas[1][1:]**2,'-*',label='Truth')
plt.legend(loc=0)

"""

# + {"code_folding": [0]}
### reapeating the estimation for many times

"""
n_loop = 5

para_est_sum_sim = (np.array([0]),np.zeros([2,50]))
for i in range(n_loop):
    para_est_this_time = dt_est.EstimateParabySim(method='CG',
                                                  para_guess = para_guess_this2,
                                                  options = {'disp': True})
    para_est_sum_sim  = para_est_sum_sim + para_est_this_time
    
    
"""

# +
#para_est_av = sum([abs(para_est_sum_sim[2*i+1]) for i in range(1,n_loop+1)] )/n_loop

# + {"code_folding": [0]}
## check the estimation and true parameters

"""
fig = plt.figure(figsize=([14,4]))

plt.subplot(1,2,1)
plt.title('Permanent Risk')
plt.plot(para_est_av[0][1:].T**2,'r-',label='Estimation(sim)')
plt.plot(dt_fake.sigmas[0][1:]**2,'-*',label='Truth')


plt.subplot(1,2,2)
plt.title('Transitory Risk')
plt.plot(para_est_av[1][1:].T**2,'r-',label='Estimation(sim)')
plt.plot(dt_fake.sigmas[1][1:]**2,'-*',label='Truth')
plt.legend(loc=0)

"""
# -

# ### Time Aggregation

# +
## time aggregation 

sim_data = dt.SimulateSeries(n_sim = 1000)
dt.n_agg = 3
agg_series = dt.TimeAggregate()
agg_series_moms = dt.SimulateMomentsAgg()

# + {"code_folding": [0, 2]}
## difference times degree of time aggregation leads to different autocorrelation

for ns in np.array([2,8]):
    an_instance = cp.deepcopy(dt)
    an_instance.n_agg = ns
    series = an_instance.SimulateSeries(n_sim =500)
    agg_series = an_instance.TimeAggregate()
    agg_series_moms = an_instance.SimulateMomentsAgg()
    var_sim = an_instance.AutocovarAgg(step=0)
    var_b1 = an_instance.AutocovarAgg(step=-1)
    plt.plot(var_b1,label=r'={}'.format(ns))
plt.legend(loc=1)
plt.title('1-degree autocovariance of different \n level of time aggregation')

# -

# #### Estimation using time aggregated data

# + {"code_folding": []}
## get some fake aggregated data moments
"""
moms_agg_fake = dt_fake.TimeAggregate()
moms_agg_dct_fake = dt_fake.SimulateMomentsAgg()
"""

# + {"code_folding": []}
## estimation 
"""
para_guess_this3 = para_guess_this*0.5
dt_est.GetDataMomentsAgg(moms_agg_dct_fake)
dt_est.n_periods = 12
para_est_agg = dt_est.EstimateParaAgg(method ='Powell',
                                      para_guess = para_guess_this3,
                                      options={'disp':True,
                                              'ftol': 0.000000001}
                                     )
                                     
"""

# + {"code_folding": []}
## check the estimation and true parameters

"""
fig = plt.figure(figsize=([10,4]))

plt.subplot(1,2,1)
plt.title('Permanent Risk')
plt.plot(dt_est.para_est_agg[1][0][11:-1].T**2,'r-',label='Estimation(agg)')
plt.plot(dt_fake.sigmas[0][11:-1]**2,'-*',label='Truth')

plt.subplot(1,2,2)
plt.title('Transitory Risk')
plt.plot(dt_est.para_est_agg[1][1][11:-1].T**2,'r-',label='Estimation(agg)')
plt.plot(dt_fake.sigmas[1][11:-1]**2,'-*',label='Truth')
plt.legend(loc=0)

"""

# + {"code_folding": []}
### reapeating the estimation for many times

"""
n_loop = 5

para_est_sum_agg = (np.array([0]),np.zeros([2,50]))
for i in range(n_loop):
    para_est_this_time = dt_est.EstimateParaAgg(method ='Powell',
                                      para_guess = para_guess_this3,
                                      options={'disp':True,
                                              'ftol': 0.000000001})
    para_est_sum_agg = para_est_sum_agg + para_est_this_time
    
"""
# -

para_est_av_agg = sum([abs(para_est_sum_agg[2*i+1]) for i in range(1,n_loop+1)] )/n_loop

# + {"code_folding": []}
## check the estimation and true parameters


"""
fig = plt.figure(figsize=([14,4]))


plt.subplot(1,2,1)
plt.title('Permanent Risk')
plt.plot(para_est_av_agg[0][11:].T**2,'r-',label='Estimation(agg)')
plt.plot(dt_fake.sigmas[0][11:]**2,'-*',label='Truth')

plt.subplot(1,2,2)
plt.title('Transitory Risk')
plt.plot(para_est_av_agg[1][11:].T**2,'r-',label='Estimation(agg)')
plt.plot(dt_fake.sigmas[1][11:]**2,'-*',label='Truth')
plt.legend(loc=0)

"""
# -


