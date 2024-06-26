# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
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

# ### Basic facts about heterogeneity of perceived income risks 
#
#
# - this notebook first plots the cross-sectional distribution of perceived income risks 
#
# - and, then it runs regressions to inspects the covariants of individual perceived income moments
#   - individual demogrpahics, level of household income, education, etc.
#   - job-types, part-time vs full-time, selfemployment, etc. 
#   - other expectations: probability of unemployment, other job-related expectations 
#   - **experienced volatility** estimated from PSID 
#   - numeracy from SCE 
#

# ###  1. Loading and cleaning data

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

# +
## figure plotting configurations


plt.style.use('seaborn')
plt.rcParams["font.family"] = "Times New Roman" #'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['axes.labelweight'] = 'bold'

## Set the 
plt.rc('font', size=10)
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

## precision of showing float  
pd.options.display.float_format = '{:,.2f}'.format

dataset = pd.read_stata('../SurveyData/SCE/IncExpSCEProbIndM.dta')   
dataset_est = pd.read_stata('../SurveyData/SCE/IncExpSCEDstIndM.dta')
sipp_individual = pd.read_stata('../OtherData/sipp/sipp_individual_risk.dta')

# + {"code_folding": [0, 14]}
## variable list by catogrories 

vars_id = ['userid','date']

moms_nom = ['Q24_mean','Q24_iqr','Q24_var']

moms_real = ['Q24_rmean','Q24_rvar']

vars_demog = ['D6']   ## level of income, 11 groups 

vars_job = ['Q10_1',  # full-time 
            'Q10_2',  # part-time
            'Q12new'] ## =1 worked for others; = 2 self-employment 

vars_demog_sub = ['Q32',  ## age 
                  'Q33',  ## gender 
                  'Q36',  ## education (1-8 low to high, 9 other)
                  'educ_gr',##education group (1-3)
                  'byear',
                 'nlit'] ## year of birth

vars_decision = ['Q26v2',  ## spending growth or decrease next year 
                 'Q26v2part2']  # household spending growth 

## these variables are only available for a sub sample 

vars_empexp = ['Q13new']  ## probability of unemployment 

vars_macroexp = ['Q6new',  ## stock market going up 
                 'Q4new']  ## UE goes up 

# + {"code_folding": [0, 2, 15]}
## subselect variables 

vars_all_reg_long = (vars_id+moms_nom + moms_real + vars_job + 
                     vars_demog + vars_demog_sub + 
                     vars_empexp + vars_macroexp + vars_decision)

vars_est_all= vars_id + ['IncMean','IncVar','IncSkew','IncKurt']


## select dataset 

SCEM_base = dataset[vars_all_reg_long]
SCEM_est = dataset_est[vars_est_all]

## merge 
SCEM = pd.merge(SCEM_base, 
                SCEM_est,  
                how='left', 
                left_on = vars_id, 
                right_on = vars_id)

# + {"code_folding": [0]}
## renaming 

SCEM = SCEM.rename(columns={'Q24_mean': 'incexp',
                           'Q24_var': 'incvar',
                           'Q24_iqr': 'inciqr',
                           'Q24_rmean':'rincexp',
                           'Q24_rvar': 'rincvar',
                           'IncMean':'incmeanest',
                            'IncVar':'incvarest',
                           'IncSkew':'incskew',
                           'IncKurt':'inckurt'})

SCEM = SCEM.rename(columns = {'D6':'HHinc',
                              'Q13new':'UEprobInd',
                              'Q6new':'Stkprob',
                              'Q4new':'UEprobAgg',
                              'Q10_1':'fulltime',
                              'Q10_2':'parttime',
                              'Q12new':'selfemp',
                              'Q32':'age',
                              'Q33':'gender',
                              'Q36':'educ',
                              'Q26v2': 'spending_dum',
                              'Q26v2part2':'spending'})
#dataset_psid_edu = dataset_psid_edu.rename(columns={'edu':'educ_gr'})

SCEM.columns

# +
## generate age square

SCEM['age2']=SCEM['age']**2
SCEM['age3']=SCEM['age']**3
SCEM['age4']=SCEM['age']**4
# -

## convert categorical educ_gr to int to merge 
code_educ_gr = {'educ_gr': {"HS dropout": 1, "HS graduate": 2, "College/above": 3}}
SCEM = SCEM.replace(code_educ_gr)
SCEM = SCEM.astype({'educ_gr': 'int32'})
SCEM['educ_gr'].value_counts()

## index 
SCEM['year'] = pd.DatetimeIndex(SCEM['date']).year

# + {"code_folding": []}
SCEM['age'] = SCEM['age'].astype('int',
                                 errors='ignore')


# +
## creat some less fine groups 

SCEM['HHinc_gr'] = SCEM['HHinc']>= 6
SCEM['nlit_gr'] = SCEM['nlit']>= 4
# -

len(SCEM)

# +
## filtering non-working group 

SCEM = SCEM[(SCEM.age < 66) & (SCEM.age > 20)]
# -

len(SCEM)

SCEM.columns

# +
SCEM = SCEM.dropna(subset=['date'])

## add year and month variable 
SCEM['year'] = SCEM.date.dt.year
SCEM['month'] = SCEM.date.dt.month 
# -

len(SCEM)

# +
## filter the data sample 
import datetime as dt 

date_before = dt.datetime(2020, 3, 1)

SCEM = SCEM[SCEM['date']<=date_before]
# -

len(SCEM)

# ### 2. Correlation pattern 

# +
## data types 

SCEM.dtypes
for col in ['HHinc','age','educ','HHinc_gr','educ_gr','nlit_gr']:
    SCEM[col] = SCEM[col].astype('int',
                                 errors='ignore')

# + {"code_folding": [0]}
inc_grp = {1:"10k",
           2:'20k',
           3:'30k',
           4:'40k',
           5:'50k',
           6:'60k',
           7:'75k',
           8:'100k',
           9:'150k',
           10:'200k',
           11:'200k+'}

cleanup_nums = {'parttime': {0: 'no', 1: 'yes'},
                'fulltime': {0: 'no', 1: 'yes'},
                'selfemp':{1: 'no', 2: 'yes'},
                'gender':{1:'male',2:'female'},
               'HHinc_gr':{0:'low income',1:'high income'},
               'educ_gr':{1:'hs dropout',2:'high school', 3:'college'},
               'nlit_gr':{0:'low nlit',1:'high nlit'}}
SCEM.replace(cleanup_nums,
             inplace = True)

# + {"code_folding": []}
## create age group 

SCEM['age_gr'] = pd.cut(SCEM['age'],
                        5,
                        labels= ["20-30","30-39","40-48","49-57",">57"])


## create cohort group


SCEM['byear_gr'] = pd.cut(SCEM['byear'],
                          6,
                         labels = ['40s','50s','60s','70s','80s','90s'])



# + {"code_folding": []}
## categorical variables 

vars_cat = ['HHinc','fulltime','parttime','selfemp',
            'gender','educ','userid','date','byear',
            'year','HHinc_gr','educ_gr','nlit_gr'] 

for var in vars_cat:
    SCEM[var] = pd.Categorical(SCEM[var],ordered = False)

# + {"code_folding": []}
# correlation heatmap 

non_categorical = [var for var in SCEM.columns if var not in vars_cat+ ['age_gr','byear_gr']]

fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(SCEM[non_categorical].corr(), annot = True)
# -

# ###  3. Histograms

# + {"code_folding": []}
moms = ['incexp','rincexp','incvar','rincvar','incskew']

## by age 
fig,axes = plt.subplots(len(moms),figsize=(6,14))

for i,mom in enumerate(moms):
    #plt.style.use('ggplot')
    SCEM.groupby('age_gr')[mom].mean().plot(kind='bar', ax=axes[i],title=mom)
    #axes[i].set_ylabel(mom,size = 15)
    
    if i == len(moms)-1:
        axes[i].set_xlabel('group by the year of birth \n (from young to old)',
              size = 15)
plt.tight_layout()
#plt.savefig('../Graphs/ind/bar_by_age')

# + {"code_folding": []}
## by cohort 

fig,axes = plt.subplots(len(moms),figsize=(6,14))

for i,mom in enumerate(moms):
    #plt.style.use('ggplot')
    SCEM.groupby('byear_gr')[mom].mean().plot(kind='bar', ax=axes[i],title=mom)
    #axes[i].set_ylabel(mom,size = 15)
    
    if i == len(moms)-1:
        axes[i].set_xlabel('group by the year of birth \n (from older generation to the young)',
              size = 15)
plt.tight_layout()

#plt.savefig('../Graphs/ind/bar_by_cohort')


# +
## by hh income 

fig,axes = plt.subplots(len(moms),figsize=(6,14))

for i,mom in enumerate(moms):
    #plt.style.use('ggplot')
    SCEM.groupby('HHinc_gr')[mom].mean().plot(kind='bar', ax=axes[i],title=mom)
    #axes[i].set_ylabel(mom,size = 15)
    
    if i == len(moms)-1:
        axes[i].set_xlabel('group by household income (from low to high)',
              size = 15)
plt.tight_layout()

#plt.savefig('../Graphs/ind/bar_by_inc')


# + {"code_folding": []}
## by education

fig,axes = plt.subplots(len(moms),figsize=(6,14))

for i,mom in enumerate(moms):
    #plt.style.use('ggplot')
    SCEM.groupby('educ_gr')[mom].mean().plot(kind='bar', ax=axes[i],title=mom)
    #axes[i].set_ylabel(mom,size = 15)
    
    if i == len(moms)-1:
        axes[i].set_xlabel('group by education (from low to high)',
                          size = 15)
plt.tight_layout()
#plt.savefig('../Graphs/ind/bar_by_educ')


# + {"code_folding": []}
## by gender 

fig,axes = plt.subplots(len(moms),
                        figsize=(6,14))

for i,mom in enumerate(moms):
    #plt.style.use('ggplot')
    SCEM.groupby('gender')[mom].mean().plot(kind='bar', ax=axes[i],title=mom)
    #axes[i].set_ylabel(mom,size = 15)
    
    if i == len(moms)-1:
        axes[i].set_xlabel('group by gender',
              size = 15)
plt.tight_layout()
#plt.savefig('../Graphs/ind/bar_by_gender')


# +
## by numeracy literacy  

fig,axes = plt.subplots(len(moms),figsize=(6,14))

for i,mom in enumerate(moms):
    #plt.style.use('ggplot')
    SCEM.groupby('nlit_gr')[mom].mean().plot(kind='bar', ax=axes[i],title=mom)
    #axes[i].set_ylabel(mom,size = 15)
    
    if i == len(moms)-1:
        axes[i].set_xlabel('group by numeracy',
              size = 15)
plt.tight_layout()

#plt.savefig('../Graphs/ind/bar_by_nlit')
# -

# ### 4.1. Cross-sectional heterogeneity 

# +
mom_list = ['incexp',
            'incvar',
            'inciqr',
            'rincexp',
            'incstd',
            'rincstd',
            'rincvar',
            'incskew'
           ]

labels_list = ['expected nominal wage growth',
            'perceived nominal wage risks',
            'perceived nominal wage IQR',
            'expected real wage growth',
            'perceived nominal wage risks (std)',
            'perceived real wage risks (std)',
            'perceived nominal wage risks',
            'perceived skewness of wage growth' ]

SCEM['incstd'] = np.sqrt(SCEM['incvar'])
SCEM['rincstd'] = np.sqrt(SCEM['rincvar'])

# +
### histograms

for mom_id,mom in enumerate(mom_list):
    if mom !='incskew':
        to_plot = SCEM[mom]
    else:
        mom_nonan = SCEM[mom].dropna()
        mom_lb, mom_ub = np.percentile(mom_nonan,2),np.percentile(mom_nonan,98) ## exclude top and bottom 3% observations
        to_keep = (mom_nonan < mom_ub) & (mom_nonan > mom_lb) & (mom_nonan!=0)
        to_plot = mom_nonan[to_keep]
    
    fig,ax = plt.subplots(figsize=(8,6))
    sns.histplot(data = to_plot,
                 kde = True,
                 stat="density", 
                 color = 'red',
                 bins = 40,
                alpha = 0.3
                )
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.xlabel(labels_list[mom_id], fontsize = 15)
    #plt.ylabel("Frequency",fontsize = 15)
    plt.savefig('../Graphs/ind/hist_'+str(mom)+'_uc.jpg')
# -

sipp_individual.describe()

# + {"code_folding": [0, 3, 19]}
## plot only perceived risks 

fig,ax = plt.subplots(figsize=(8,6))
sns.histplot(data = SCEM['rincstd'],
             kde = True,
             color = 'red',
             bins = 60,
            alpha = 0.3,
             stat="density", 
             fill = True,
            label='Dist of PRs in SCE')


## filter extreme values 
individual_risk = sipp_individual['lwage_Y_id_shk_gr_sq'].dropna()
lb, ub = np.percentile(individual_risk,10),np.percentile(individual_risk,70) ## exclude top and bottom 3% observations
to_keep = (individual_risk < ub) & (individual_risk!=0)
individual_risk_keep = individual_risk[to_keep]

sns.histplot(data = individual_risk_keep,
             kde = True,
             color = 'blue',
             bins = 60,
             stat="density", 
            alpha = 0.5,
             fill = False,
            label='Dist of wage volatility (SIPP)')

plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.xlabel('PRs and Volatility in std', fontsize = 15)
plt.legend(loc=0)
plt.savefig('../Graphs/ind/hist_compare_PRs.jpg')

# + {"code_folding": [0]}
## plot only perceived risks 

fig,ax = plt.subplots(figsize=(8,6))
sns.histplot(data = SCEM['rincstd'],
             kde = True,
             color = 'red',
             bins = 100,
            alpha = 0.3,
             stat="density", 
             fill = True,
            label='Dist of PRs in SCE')


## filter extreme values 
individual_risk = sipp_individual['lwage_Y_id_shk_gr_sq_pr'].dropna()
lb, ub = np.percentile(individual_risk,10),np.percentile(individual_risk,90) ## exclude top and bottom 3% observations
to_keep = (individual_risk < ub) & (individual_risk!=0)
individual_risk_keep = individual_risk[to_keep]

sns.histplot(data = individual_risk_keep,
             kde = True,
             color = 'blue',
             bins = 100,
             stat="density", 
            alpha = 0.5,
             fill = False,
            label='Dist of fitted wage volatility (SIPP)')

plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.xlabel('PRs and Predicted Volatility in std', 
           fontsize = 15)
plt.legend(loc=0)
plt.savefig('../Graphs/ind/hist_compare_fitted_PRs.jpg')
# -

# ### 4.2. Within-group heterogeneity 

# +
mean_std = np.sqrt(SCEM['incvar'].mean())
print("Mean of nominal risk perception is " 
      +str(round(mean_std,3))
     +' in stv')

med_std = np.sqrt(SCEM['incvar'].median())
print("Median of nominal risk perception is " 
      +str(round(med_std,3))
     +' in stv')


med_std = np.sqrt(SCEM['incvar'].median())
print("Median of nominal risk perception is " 
      +str(round(med_std,3))
     +' in stv')

# +
mean_std = np.sqrt(SCEM['rincvar'].mean())
print("Mean of risk perception is " 
      +str(round(mean_std,3))
     +' in stv')

med_std = np.sqrt(SCEM['rincvar'].median())
print("Median of risk perception is " 
      +str(round(med_std,3))
     +' in stv')


med_std = np.sqrt(SCEM['rincvar'].median())
print("Median of risk perception is " 
      +str(round(med_std,3))
     +' in stv')

# + {"code_folding": []}
### first step regression 


for i,mom in enumerate(mom_list):
    model = smf.ols(formula = str(mom)
                    +'~ age+age2+C(parttime) + C(selfemp) + C(gender)+ C(HHinc_gr) + C(nlit_gr)+C(educ_gr)+C(year)', #
                    data = SCEM)
    result = model.fit()
    SCEM[mom+'_rd']=result.resid
    
# -

SCEM.columns

print('First-step R2 is '+str(round(result.rsquared,2)))

# +
riqr = SCEM['rincstd_rd'].quantile(q=0.9)-SCEM['rincstd_rd'].quantile(q=0.1)

print("10/90 IQR of risk perception is " 
      +str(round(riqr,3))
     +' in stv')

iqr = SCEM['incstd_rd'].quantile(q=0.9)-SCEM['incstd_rd'].quantile(q=0.1)
print("10/90 IQR of nominal risk perception is " 
      +str(round(iqr,3))
     +' in stv')

# + {"code_folding": []}
### histograms

for mom_id,mom in enumerate(mom_list):
    if mom !='incskew':
        to_plot = SCEM[mom+str('_rd')]
    else:
        mom_nonan = SCEM[mom].dropna()
        mom_lb, mom_ub = np.percentile(mom_nonan,2),np.percentile(mom_nonan,98) ## exclude top and bottom 3% observations
        #print(mom_lb)
        #print(mom_ub)
        to_keep = (mom_nonan < mom_ub) & (mom_nonan > mom_lb) & (mom_nonan!=0)
        #print(to_keep.shape)
        to_plot = mom_nonan[to_keep]
    #print(mom_nonan_truc.shape)
    
    fig,ax = plt.subplots(figsize=(8,6))
    sns.histplot(data = to_plot,
                 kde = True,
                 stat="density", 
                 color = 'gray',
                 bins = 60,
                alpha = 0.3)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.xlabel('Residuals of '+labels_list[mom_id], fontsize = 15)
    plt.savefig('../Graphs/ind/hist_'+str(mom)+'.jpg')

# +
import joypy
from matplotlib import cm

labels=[y for y in list(SCEM.year.unique())]




for mom_id,mom in enumerate(mom_list):
    
    fig, axes = joypy.joyplot(SCEM, 
                              by="year", 
                              column= mom+'_rd', 
                              labels=labels, 
                              kind="kde", 
                              range_style='own', 
                              grid="y", 
                              linewidth=1, 
                              legend=False, 
                              figsize=(6,7),
                              title=labels_list[mom_id],
                              colormap=cm.Wistia
                             )
    #plt.savefig('../Graphs/ind/joy_'+str(mom)+'.jpg')
# -

# ### 5. Experienced volatility and risks (not used)

# ###  6. Main regression

# +
## preps 

dep_list =  ['incvar'] 
dep_list2 =['incexp','rincexp']
indep_list_ct = ['UEprobInd','UEprobAgg']
indep_list_dc = ['HHinc','selfemp','fulltime','nlit_gr']


# + {"code_folding": [5, 47]}
## full-table for risks  

rs_list = {}  ## list to store results 
nb_spc = 5  ## number of specifications 

for i,mom in enumerate(dep_list):
   
    ## model 1  age 
    
    model1 = smf.ols(formula = str(mom)
                    +'~ age+age2+age3',
                    data = SCEM)
    rs_list[nb_spc*i+0] = model1.fit()
    
    ## model 2 experienced vol, age, income, education 
    
    model2 = smf.ols(formula = str(mom)
                    +'~ age+age2+age3 + C(gender)+ C(nlit_gr)',
                    data = SCEM)
    rs_list[nb_spc*i+1] = model2.fit()
    
    ## model 3 + job characteristics 
    
    model3 = smf.ols(formula = str(mom)
                    +'~ age+age2+age3+C(gender)+ C(nlit_gr)+C(educ_gr)',
                    data = SCEM)
    rs_list[nb_spc*i+2] = model3.fit()
    
    
    ## model 4 + job characteristics 
    model4 = smf.ols(formula = str(mom)
                    +'~ age+age2+age3+C(parttime) + C(selfemp) + C(gender)+ C(HHinc_gr) + C(nlit_gr)+C(educ_gr)',
                    data = SCEM)
    rs_list[nb_spc*i+3] = model4.fit()
    
    
    ## model 5 + job characteristics 
    ct_str = '+'.join([var for var in indep_list_ct])
    model5 = smf.ols(formula = str(mom)
                    +'~age+age2+age3+C(parttime) + C(selfemp) + C(gender)+ C(HHinc_gr) + C(nlit_gr) +C(educ_gr)+'
                     + ct_str,
                    data = SCEM)
    rs_list[nb_spc*i+4] = model5.fit()
    
    
rs_names = [rs_list[i] for i in range(len(rs_list))]

dfoutput = summary_col(rs_names,
                        float_format='%0.2f',
                        stars = True,
                        regressor_order = ['age',
                                           'age2',
                                           'age3',
                                           'C(HHinc_gr)[T.low inc]',
                                           'C(gender)[T.male]',
                                           'C(nlit_gr)[T.low nlit]',
                                           'C(parttime)[T.yes]',
                                           'C(selfemp)[T.yes]',
                                           'UEprobAgg',
                                           'UEprobInd'],
                        info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),
                                  'R2':lambda x: "{:.2f}".format(x.rsquared)
                                  })
dfoutput.title = 'Perceived Income Risks'
print(dfoutput)


# -

def droptables(table,
    to_drop):
    table = table.reset_index()
    to_drop_ids = []
    for var in to_drop:
        to_drop_idx = table[table['index']==var].index[0]
        to_drop_ids.append(to_drop_idx)
        to_drop_ids.append(to_drop_idx+1)
    table = table.drop(index = to_drop_ids)
    table = table.set_index('index')
    table.index.name = ''
    return table


# + {"code_folding": []}
## output tables 

beginningtex = """
\\begin{table}[p]
\\centering
\\begin{adjustbox}{width=\\textwidth}
\\begin{threeparttable}
\\caption{Perceived Income Risks, Experienced Volatility and Individual Characteristics}
\\label{micro_reg}"""

endtex = """\\begin{tablenotes}\item Standard errors are clustered by household. *** p$<$0.001, ** p$<$0.01 and * p$<$0.05. 
\item This table reports results from a regression of looged perceived income risks (incvar) on logged indiosyncratic($\\text{IdExpVol}$), aggregate experienced volatility($\\text{AgExpVol}$), experienced unemployment rate (AgExpUE), and a list of household specific variables such as age, income, education, gender, job type and other economic expectations.
\\end{tablenotes}
\\end{threeparttable}
\\end{adjustbox}
\\end{table}"""

## relabel rows 

def CatRename(table):
    relabels = {}
    rows = [idx for idx in table.index if ')[T.' in idx]
    for i in range(len(rows)):
        string = rows[i]
        var = string.split('C(')[1].split(')[T')[0]
        val = string.split('[T.')[1].split(']')[0]
        if '.0' in val:
            val = val.split('.0')[0]
        else:
            val = val 
        relabels[rows[i]] = var + '=' + str(val)
    table = table.rename(index = relabels)
    return table 
table = CatRename(dfoutput.tables[0])

## excluding rows that are not to be exported 

to_drop = ['Intercept','R-squared']
 
tb = droptables(table,
                to_drop)

## write to latex 
f = open('../Tables/latex/micro_reg.tex', 'w')
f.write(beginningtex)
tb_ltx = tb.to_latex().replace('lllllllll','ccccccccc')   # hard coded here 
#print(tb)
f.write(tb_ltx)
f.write(endtex)
f.close()

## save
tb.to_excel('../Tables/micro_reg.xlsx')

# + {"code_folding": [5, 7, 37]}
## full-table for expected growth, appendix 

rs_list = {}  ## list to store results 
nb_spc = 4  ## number of specifications 

for i,mom in enumerate(dep_list2):
    ## model 1 
    model = smf.ols(formula = str(mom)
                    +'~ C(parttime)+C(selfemp)',
                    data = SCEM)
    rs_list[nb_spc*i] = model.fit()
    
    ## model 2
    ct_str = '+'.join([var for var in indep_list_ct])
    model2 = smf.ols(formula = str(mom)
                    +'~ age+age2+age3+C(parttime)+C(selfemp) + '
                     + ct_str,
                    data = SCEM)
    rs_list[nb_spc*i+1] = model2.fit()
    
    ## model 3 
    model3 = smf.ols(formula = str(mom)
                    +'~age+age2+age3+ C(parttime) + C(selfemp) + C(HHinc_gr) + C(educ_gr) +'
                     + ct_str,
                    data = SCEM)
    rs_list[nb_spc*i+2] = model3.fit()
    
    ## model 4 
    model4 = smf.ols(formula = str(mom)
                    +'~age+age2+age3+ C(parttime) + C(selfemp) + C(gender)+ C(educ_gr) + + C(HHinc_gr) +'
                     + ct_str,
                    data = SCEM)
    rs_list[nb_spc*i+3] = model4.fit()
    
    
rs_names = [rs_list[i] for i in range(len(rs_list))]

dfoutput2 = summary_col(rs_names,
                        float_format='%0.2f',
                        stars = True,
                        regressor_order = ['age',
                                           'age2',
                                           'age3',
                                           'C(parttime)[T.yes]',
                                           'C(selfemp)[T.yes]',
                                           'UEprobAgg','UEprobInd',
                                           'C(HHinc_gr)[T.low inc]',
                                           'C(educ_gr)[T.low educ]',
                                           'C(gender)[T.male]'
                                           ],
                        info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),
                                  'R2':lambda x: "{:.2f}".format(x.rsquared)
                                  })

dfoutput2.title = 'Perceived Income Growth'
print(dfoutput2)

## relabel 
table = CatRename(dfoutput2.tables[0])

## excluding rows that are not to be exported 

to_drop = ['Intercept','R-squared']
 
tb = droptables(table,
                to_drop)

## latex setting 

beginningtex = """
\\begin{table}[p]
\\centering
\\begin{adjustbox}{width={\\textwidth}}
\\begin{threeparttable}
\\caption{Perceived Income Growth and Individual Characteristics}
\\label{micro_reg_exp}"""

endtex = """\\begin{tablenotes}\item Standard errors are clustered by household. *** p$<$0.001, ** p$<$0.01 and * p$<$0.05. 
\item This table reports regression results of perceived labor income(incexp for nominal, rincexp for real) growth on household specific variables. HHinc: household income group ranges from lowests (=1, less than \$10k/year) to the heightst (=11, greater than \$200k/year). Education, educ ranges from the lowest (=1, less than high school) to the highest (=9).
\\end{tablenotes}
\\end{threeparttable}
\\end{adjustbox}
\\end{table}"""

"""
## write to latex 
f = open('../Tables/latex/micro_reg_exp.tex', 'w')
f.write(beginningtex)
tb_ltx = tb.to_latex().replace('lllllllll','ccccccccc')   # hard coded here 
#print(tb)
f.write(tb_ltx)
f.write(endtex)
f.close()
"""
# -

# ### 6. Perceived risks and decisions

# + {"code_folding": [0]}
## full-table for risks  

rs_list = {}  ## list to store results 
nb_spc = 1  ## number of specifications 

dep_list3 = ['incexp','incvar','rincvar','incskew','UEprobAgg']


for i,mom in enumerate(dep_list3):
    ## model 1 
    model = smf.ols(formula = 'spending~'+ '+'+ mom,
                    data = SCEM)
    rs_list[nb_spc*i] = model.fit()
    
    
rs_names = [rs_list[i] for i in range(len(rs_list))]

dfoutput = summary_col(rs_names,
                        float_format='%0.2f',
                        stars = True,
                        regressor_order = ['incexp',
                                           'incvar',
                                           'rincvar',
                                           'incskew',
                                          'UEprobAgg'],
                        info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),
                                  'R2':lambda x: "{:.2f}".format(x.rsquared)})
dfoutput.title = 'Perceived Income Risks and Household Spending'
print(dfoutput)
# + {"code_folding": [20]}
## output tables 

beginningtex = """
\\begin{table}[p]
\\centering
\\begin{adjustbox}{width={0.9\\textwidth}}
\\begin{threeparttable}
\\caption{Perceived Income Risks and Household Spending}
\\label{spending_reg}"""

endtex = """\\begin{tablenotes}\item Standard errors are clustered by household. *** p$<$0.001, ** p$<$0.01 and * p$<$0.05. 
\item This table reports regression results of expected spending growth on perceived income risks (incvar for nominal, rincvar for real).
\\end{tablenotes}
\\end{threeparttable}
\\end{adjustbox}
\\end{table}"""


## relabel rows 

def CatRename(table):
    relabels = {}
    rows = [idx for idx in table.index if ')[T.' in idx]
    for i in range(len(rows)):
        string = rows[i]
        var = string.split('C(')[1].split(')[T')[0]
        val = string.split('[T.')[1].split(']')[0]
        if '.0' in val:
            val = val.split('.0')[0]
        else:
            val = val 
        relabels[rows[i]] = var + '=' + str(val)
    table = table.rename(index = relabels)
    return table 
table = CatRename(dfoutput.tables[0])

## excluding rows that are not to be exported 

to_drop = ['Intercept','R-squared']
 
tb = droptables(table,
                to_drop)

## excel version 
tb.to_excel('../Tables/spending_reg.xlsx')

## write to latex 
f = open('../Tables/latex/spending_reg.tex', 'w')
f.write(beginningtex)
tb_ltx = tb.to_latex().replace('llllll','cccccc')   # hard coded here 
#print(tb)
f.write(tb_ltx)
f.write(endtex)
f.close()

