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

# ## Perceived Labor Income Risks and Macroeconomic conditions
#
#
# - This notebook first downloads asset return indicators
# - Then we examine the correlation of higher moments of labor income risks and asset returns
# - It also inspects the cross-sectional pattern of the subjective moments

# +
import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as st
from statsmodels.graphics import tsaplots as tsplt

# +
plt.style.use('ggplot')
plt.rcParams.update({'figure.max_open_warning': 0})
#from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('pdf','png','jpg')
#plt.rcParams['savefig.dpi'] = 75

#plt.rcParams['figure.autolayout'] = False
#plt.rcParams['figure.figsize'] = 10, 6
#plt.rcParams['axes.labelsize'] = 18
#plt.rcParams['axes.titlesize'] = 20
#plt.rcParams['font.size'] = 16
#plt.rcParams['lines.linewidth'] = 2.0
#plt.rcParams['lines.markersize'] = 8
#plt.rcParams['legend.fontsize'] = 14

#plt.rcParams['text.usetex'] = True
#plt.rcParams['font.family'] = "serif"
#plt.rcParams['font.serif'] = "cm"
# -

pd.options.display.float_format = '{:,.2f}'.format

# ###  1. Download stock return and wage rate series 

# + {"code_folding": []}
## s&p 500 series

start = datetime.datetime(2000, 1, 30)
end = datetime.datetime(2020, 3, 30)

# + {"code_folding": []}
## downloading the data from Fred
sp500D= web.DataReader('sp500', 'fred', start, end)
vixD = web.DataReader('VIXCLS','fred',start,end)
he = web.DataReader('CES0500000003','fred',start,end) #hourly earning private
# -

vixD.plot(lw = 2)
vixplt = plt.title('vix')

# + {"code_folding": []}
#plotting
#sp500D.plot(lw=2)
#sp500plt = plt.title('S&P 500')

# +
## collapse to monthly data
sp500D.index = pd.to_datetime(sp500D.index)
sp500M = sp500D.resample('M').last()

vixD.index = pd.to_datetime(vixD.index)
vixM = vixD.resample('M').mean()
# -

sp500M.plot(lw = 3)
#sp500Mplt = plt.title('S&P 500 (end of month)')

sp500MR = np.log(sp500M).diff(periods = 3)
he = he.diff(periods = 3)
he.columns = ['he']

sp500MR.plot(lw = 3 )
sp500MRplt = plt.title('Monthly return of S&P 500')

he.plot(lw = 2)

# ###  2. Loading and cleaning perceived income series

# + {"code_folding": []}
## loading the stata file
SCEProbIndM = pd.read_stata('../SurveyData/SCE/IncExpSCEProbIndM.dta')
SCEDstIndM = pd.read_stata('../SurveyData/SCE/IncExpSCEDstIndM.dta')

# +
# merge the two 

SCEIndM = pd.merge(SCEProbIndM,
                   SCEDstIndM,
                   on = ['userid','date'])

# + {"code_folding": []}
## subselect the dataframe
sub_var = ['date',
           'userid',
           'Q24_var',
           'Q24_mean',
           'Q24_iqr',
           'Q24_rmean',
           'Q24_rvar']
dem_var = ['Q32',  ## age
           'Q33',  ## gender
           'Q36',  ## education (1-8 low to high, 9 other)
           'byear', ## cohort
           'D6',
           'Q6new']

sub_var2 = ['IncSkew']

IncSCEIndMoms = SCEIndM[sub_var+dem_var+sub_var2]

#IncSCEIndMomsEst = SCEDstIndM[sub_var2]

## drop nan observations
IncSCEIndMoms = IncSCEIndMoms.dropna(how='any')
#IncSCEIndMomsEst = IncSCEIndMomsEst.dropna(how='any')

# + {"code_folding": []}
## rename

IncSCEIndMoms = IncSCEIndMoms.rename(columns={'Q24_mean': 'incexp',
                                               'Q24_var': 'incvar',
                                               'Q24_iqr': 'inciqr',
                                               'Q24_rmean':'rincexp',
                                               'Q24_rvar': 'rincvar',
                                              'IncSkew':'incskew',
                                              'D6':'HHinc',
                                              'Q32':'age',
                                              'Q33':'gender',
                                              'Q36':'educ',
                                              'Q6new': 'Stkprob'
                                             })


# + {"code_folding": []}
## deal with clusterring skewness value around zero first

#IncSCEIndMoms['incskew'] = IncSCEIndMoms['incskew'].copy().replace(0,np.nan)

# +
## create new groups

IncSCEIndMoms['educ_gr'] = pd.cut(IncSCEIndMoms['educ'],
                                   2,
                                   labels= ['low','high'])


IncSCEIndMoms['HHinc_gr'] = pd.cut(IncSCEIndMoms['HHinc'],
                                   2,
                                   labels= ['low','high'])

IncSCEIndMoms['age_gr'] = pd.cut(IncSCEIndMoms['age'],
                                 3,
                                 labels= ['young','middle-age','old'])


IncSCEIndMoms['byear_gr'] = pd.cut(IncSCEIndMoms['byear'],
                                   4,
                                   labels = ['50s','60s','70s','80s'])


## categorical variables

vars_cat = ['HHinc','gender','age_gr','byear_gr','HHinc_gr','educ_gr']

for var in vars_cat:
    IncSCEIndMoms[var] = pd.Categorical(IncSCEIndMoms[var],ordered = False)

# + {"code_folding": [14]}
moms = ['incexp','incvar','inciqr','rincexp','rincvar','incskew']
#moms_est = ['IncSkew','IncKurt']

## compute population summary stats for these ind moms
IncSCEPopMomsMed = pd.pivot_table(data = IncSCEIndMoms,
                                  index=['date'],
                                  values = moms,
                                  aggfunc= 'median').reset_index().rename(columns={'incexp': 'expMed',
                                                                                   'rincexp':'rexpMed',
                                                                                   'incvar': 'varMed',
                                                                                   'inciqr': 'iqrMed',
                                                                                   'rincvar':'rvarMed',
                                                                                  'incskew':'skewMed'})

IncSCEPopMomsMean = pd.pivot_table(data = IncSCEIndMoms,
                                   index = ['date'],
                                   values = moms,
                                   aggfunc = 'mean').reset_index().rename(columns={'incexp': 'expMean',
                                                                                   'rincexp':'rexpMean',
                                                                                   'incvar': 'varMean',
                                                                                   'inciqr': 'iqrMean',
                                                                                   'rincvar':'rvarMean',
                                                                                  'incskew':'skewMean'})
# -

T = len(IncSCEPopMomsMed)

# ### 3. Combinine the two series

# + {"code_folding": []}
## streamline the dates for merging

# adjusting the end-of-month dates to the begining-of-month for combining
sp500MR.index = sp500MR.index.shift(1,freq='D')
vixM.index = vixM.index.shift(1,freq='D')


IncSCEPopMomsMed.index = pd.DatetimeIndex(IncSCEPopMomsMed['date'] ,freq='infer')
IncSCEPopMomsMean.index = pd.DatetimeIndex(IncSCEPopMomsMean['date'] ,freq='infer')


# + {"code_folding": []}
dt_combM = pd.concat([sp500MR,
                      vixM,
                      he,
                      IncSCEPopMomsMed,
                      IncSCEPopMomsMean],
                     join="inner",
                     axis=1).drop(columns=['date'])
# -

dt_combM.tail()

dt_combM.to_stata('../SurveyData/SCE/IncExpSCEPopMacroM.dta')

# +
## save sp500 as stata for other analysis

sp500MR.to_stata('../OtherData/sp500.dta')

# + {"code_folding": []}
## date index for panel

IncSCEIndMoms.index = IncSCEIndMoms['date']
IncSCEIndMoms.index.name = None

# + {"code_folding": []}
## merge individual moments and macro series

IncSCEIndMoms.index = IncSCEIndMoms['date']

dt_combMacroM = pd.merge(sp500MR,
                         vixM,
                         left_index = True,
                         right_index = True)

dt_combMacroM = pd.merge(dt_combMacroM,
                         he,
                         left_index = True,
                         right_index = True)

dt_combIndM = pd.merge(dt_combMacroM,
                       IncSCEIndMoms,
                       left_index = True,
                       right_index = True)


dt_combIndM.to_stata('../SurveyData/SCE/IncExpSCEIndMacroM.dta')
# -

# ### 4. Seasonal adjustment (not successful yet)

# +
#to_sa_test = ['meanMed']
#to_sa_list = list(dt_combM.columns.drop('sp500'))

## inspect for seasonal pattern by looking into the autocovariance
"""
for sr in to_sa_list:
    tsplt.plot_acf(dt_combM[sr],
                   lags = np.arange(T-20),
                   alpha = 0.03)
    plt.title('Autocorrelation of '+ sr)
    plt.savefig('../Graphs/pop/acf_'+str(sr)+'.jpg')

"""
# -

# - Judging from the acf plots, it seems that only the population mean and median of expected earning growth has seasonal patterns at 12 month frequency, higher moments such as variance, skewness does not have this pattern.

# + {"code_folding": [0, 2]}
"""
for sr in to_sa_test:
    series = dt_combM[sr]
    samodel = sm.tsa.UnobservedComponents(series,
                                          level='fixed intercept',
                                          seasonal = 12)
    res = samodel.fit(disp = True)
    print(res.summary())
    #res_plt = res.plot_components(figsize=(4,11))
    #plt.plot(res.level.filtered)

"""
# -

# ### 5. Correlation with labor market outcomes 

corr_table = dt_combM.corr()
corr_table.to_excel('../Tables/corrM.xlsx')
corr_table

# + {"code_folding": [2, 16]}
lag_loop = 7

def pval_str(pval):
    if pval < 0.01:
        star = '***'
    elif pval >= 0.01 and pval<0.05:
        star = '**'
    elif pval >= 0.05 and pval <= 0.1:
        star = '*'
    else:
        star = ''
    return star

def corrtostr(corr):
    return str(round(corr[0],2)) + str(pval_str(corr[1]))

def corrprint(corr,
              var):
    print('correlation coefficient betwen ue and median'+
              str(var) +
              ' is ' +
              str(round(corr[0],2)) +
              ', and p-value is ' +
              str(round(corr[1],2))
             )

corr_list = []
col_list = []


## mean
for moms in ['var','iqr','rvar','skew']:
    col_list.append('mean:'+str(moms))
    for lag in range(lag_loop):
        corr = st.pearsonr(np.array(dt_combM['he'][:-(lag+1)]),
                           np.array(dt_combM[str(moms)+'Mean'])[(lag+1):]
                          )
        corr_str = corrtostr(corr)
        corr_list.append(corr_str)
        #corrprint(corr,moms)
"""
## median
for moms in ['var','iqr','rvar']:
    col_list.append('median:'+str(moms))
    for lag in range(lag_loop):
        corr = st.pearsonr(np.array(dt_combM['he'][:-(lag+1)]),
                           np.array(dt_combM[str(moms)+'Med'])[(lag+1):]
                          )
        corr_str = corrtostr(corr)
        corr_list.append(corr_str)
        #corrprint(corr, moms)
"""


corr_array = np.array(corr_list).reshape([int(len(corr_list)/lag_loop),
                                          lag_loop])
corr_df = pd.DataFrame(corr_array,
                       index = col_list)

corr_df.T

# +
## output tables to latex

beginningtex = """
\\begin{table}[ht]
\\centering
\\begin{adjustbox}{width={\\textwidth}}
\\begin{threeparttable}
\\caption{Current Labor Market Conditions and Perceived Income Risks}
\\label{macro_corr_he}
"""

endtex = """\\begin{tablenotes}
\item *** p$<$0.001, ** p$<$0.01 and * p$<$0.05.
\item This table reports correlation coefficients between different perceived income moments(inc for nominal
and rinc for real) at time
$t$ and the quarterly growth rate in hourly earning at $t,t-1,...,t-k$.
\\end{tablenotes}
\\end{threeparttable}
\\end{adjustbox}
\\end{table}"""

## write to latex

f = open('../Tables/latex/macro_corr_he.tex', 'w')
f.write(beginningtex)
tb_ltx = corr_df.T.to_latex().replace('llllll','cccccc')
f.write(tb_ltx)
f.write(endtex)
f.close()

## output table to excel
corr_df.T.to_excel('../Tables/macro_corr_he.xlsx')
# -

mom_dict = {'exp':'expected nominal growth',
          'rexp':'expected real growth',
          'var':'nominal income risk',
          'rvar':'real income risk',
          'iqr':'nomial 75/25 IQR',
           'skew':'skewness'}

# + {"code_folding": []}
## plots of correlation for Mean population stats

figsize = (80,40)
lw = 20
fontsize = 80

for i,moms in enumerate( ['exp','var','iqr','rexp','rvar','skew']):
    fig, ax = plt.subplots(figsize = figsize)
    ax2 = ax.twinx()
    ax.plot(dt_combM['he'],
           color='black',
           lw= lw,
           label= 'wage YoY ')
    ax2.plot(dt_combM[str(moms)+'Mean'],
             'r--',
             lw = lw,
             label = str(mom_dict[moms])+' (RHS)')
    #ax.set_xticklabels(dt_combM.index)
    ax.legend(loc = 2,
             fontsize = fontsize)
    ax.set_xlabel("month",fontsize = fontsize)
    ax.grid()
    ax.set_ylabel('% growth',fontsize = fontsize)
    ax.tick_params(axis='both', 
                   which='major', 
                   labelsize = fontsize)
    ax2.tick_params(axis='both', 
                   which='major', 
                   labelsize = fontsize)
    ax2.legend(loc = 1,
              fontsize = fontsize)
    plt.savefig('../Graphs/pop/tsMean'+str(moms)+'_he.jpg')

# +
## moving average

dt_combM3mv = dt_combM.rolling(3).mean()

# + {"code_folding": [2]}
## plots of correlation for 3-month moving mean average

for i,moms in enumerate( ['exp','var','iqr','rexp','rvar','skew']):
    fig, ax = plt.subplots(figsize = figsize)
    ax2 = ax.twinx()
    ax.plot(dt_combM3mv['he'],
           color='black',
           lw = lw,
           label=' wage YoY')
    ax2.plot(dt_combM3mv[str(moms)+'Mean'],
             'r--',
             lw = lw,
             label=str(mom_dict[moms])+' (RHS)')
    ax.legend(loc= 1,
             fontsize = fontsize)
    ax.set_xlabel("month",fontsize = fontsize)
    ax.set_ylabel('% growth',fontsize = fontsize)
    ax2.legend(loc = 2,
             fontsize = fontsize)
    ax.tick_params(axis='both', 
                   which='major', 
                   labelsize = fontsize)
    ax2.tick_params(axis='both',
                    which='major',
                    labelsize = fontsize)
    plt.savefig('../Graphs/pop/tsMean3mv'+str(moms)+'_he.jpg')
# -

# ### 5b. Individual regressions

"""
for i,moms in enumerate( ['incexp','incvar','inciqr','rincvar','incskew']):
    print(moms)
    Y = np.array(dt_combIndM[moms])[forward:]
    X = np.array(dt_combIndM['he'])[:-forward]
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    rs = model.fit()
    print(rs.summary())
"""

# ### 6. Correlation with stock market returns and times series patterns

# + {"code_folding": [4, 18]}
## try different lags or leads

lead_loop = 13

def pval_str(pval):
    if pval < 0.01:
        star = '***'
    elif pval >= 0.01 and pval<0.05:
        star = '**'
    elif pval >= 0.05 and pval <= 0.1:
        star = '*'
    else:
        star = ''
    return star

def corrtostr(corr):
    return str(round(corr[0],2)) + str(pval_str(corr[1]))

def corrprint(corr,
              var):
    print('correlation coefficient betwen sp500 and median'+
              str(var) +
              ' is ' +
              str(round(corr[0],2)) +
              ', and p-value is ' +
              str(round(corr[1],2))
             )

corr_list = []
col_list = []

#print('median')
for moms in ['var','iqr','rvar']:
    col_list.append('median:'+str(moms))
    for lead in range(lead_loop):
        corr = st.pearsonr(np.array(dt_combM['sp500'][lead+1:]),
                           np.array(dt_combM[str(moms)+'Med'])[:-(lead+1)]
                          )
        corr_str = corrtostr(corr)
        corr_list.append(corr_str)
        #corrprint(corr, moms)

for moms in ['var','iqr','rvar']:
    col_list.append('mean:'+str(moms))
    for lead in range(lead_loop):
        corr = st.pearsonr(np.array(dt_combM['sp500'][lead+1:]),
                           np.array(dt_combM[str(moms)+'Mean'])[:-(lead+1)]
                          )
        corr_str = corrtostr(corr)
        corr_list.append(corr_str)
        #corrprint(corr,moms)


corr_array = np.array(corr_list).reshape([int(len(corr_list)/lead_loop),
                                          lead_loop])
corr_df = pd.DataFrame(corr_array,
                       index = col_list)
# -

corr_df.T

# + {"code_folding": [0]}
## output tables to latex

beginningtex = """
\\begin{table}[ht]
\\centering
\\begin{adjustbox}{width={\\textwidth}}
\\begin{threeparttable}
\\caption{Correlation between Perceived Income Risks and Stock Market Return}
\\label{macro_corr}
"""

endtex = """\\begin{tablenotes}
\item *** p$<$0.001, ** p$<$0.01 and * p$<$0.05.
\item This table reports correlation coefficients between different perceived income moments(inc for nominal
and rinc for real) at time
$t$ and the monthly s\&p500 return by the end of $t+k$ for $k=0,1,..,11$.
\\end{tablenotes}
\\end{threeparttable}
\\end{adjustbox}
\\end{table}"""

## write to latex

f = open('../Tables/latex/macro_corr.tex', 'w')
f.write(beginningtex)
tb_ltx = corr_df.T.to_latex().replace('llllll','cccccc')
f.write(tb_ltx)
f.write(endtex)
f.close()

## output table to excel
corr_df.to_excel('../Tables/macro_corr.xlsx')
# -

IncSCEIndMoms = IncSCEIndMoms.drop(columns='date')

# + {"code_folding": [0, 12]}
## correlation coefficients by sub group generation

### subgroup population summary statistics

gr_vars = ['byear_gr']
lead_loop = 12

moms = ['incvar','rincvar','incskew']


## mean est 

for gr in gr_vars:
    
    corr_list = []
    col_list = []
    
    sub_pd = pd.pivot_table(data = IncSCEIndMoms,
                            index=['date',gr],
                            values = moms,
                            aggfunc= 'mean').unstack()  ## index being time only 
    sub_pd = sub_pd.dropna(how='any')
    ## mean moments 
    for mom in moms:
        # group 1 
        col_list.append('mean:'+str(mom)+str(' for 50s'))
        sp500 = np.array(sp500MR.loc[sub_pd.index]).flatten()
        
        for lead in range(lead_loop):    
            corr = st.pearsonr(sp500[lead+1:],
                               np.array(sub_pd[mom,'50s'])[:-(lead+1)]
                              )
            corr_str = corrtostr(corr)
            corr_list.append(corr_str)
            
        # group 2 
        col_list.append('mean:'+str(mom)+str(' for 60s'))
        sp500 = np.array(sp500MR.loc[sub_pd.index]).flatten()
        
        for lead in range(lead_loop):    
            corr = st.pearsonr(sp500[lead+1:],
                               np.array(sub_pd[mom,'60s'])[:-(lead+1)]
                              )
            corr_str = corrtostr(corr)
            corr_list.append(corr_str)
            
            
        # group 3
        col_list.append('mean:'+str(mom)+str(' for 70s'))
        sp500 = np.array(sp500MR.loc[sub_pd.index]).flatten()
        
        for lead in range(lead_loop):    
            corr = st.pearsonr(sp500[lead+1:],
                               np.array(sub_pd[mom,'70s'])[:-(lead+1)]
                              )
            corr_str = corrtostr(corr)
            corr_list.append(corr_str)
            
           # group 3
        col_list.append('mean:'+str(mom)+str(' for 80s'))
        sp500 = np.array(sp500MR.loc[sub_pd.index]).flatten()
        
        for lead in range(lead_loop):    
            corr = st.pearsonr(sp500[lead+1:],
                               np.array(sub_pd[mom,'80s'])[:-(lead+1)]
                              )
            corr_str = corrtostr(corr)
            corr_list.append(corr_str)
            

    corr_array = np.array(corr_list).reshape([int(len(corr_list)/lead_loop),
                                              lead_loop])
    corr_df = pd.DataFrame(corr_array,
                           index = col_list)
    
## save 
corr_df.T.to_excel('../Tables/macro_corr_byear_gr.xlsx')

# + {"code_folding": [23]}
## correlation coefficients by sub group age

### subgroup population summary statistics

gr_vars = ['age_gr']
lead_loop = 12

moms = ['incvar','rincvar','incskew']


## mean est 

for gr in gr_vars:
    
    corr_list = []
    col_list = []
    
    sub_pd = pd.pivot_table(data = IncSCEIndMoms,
                            index=['date',gr],
                            values = moms,
                            aggfunc= 'mean').unstack()  ## index being time only 
    sub_pd = sub_pd.dropna(how='any')
    ## mean moments 
    for mom in moms:
        # group 1 
        col_list.append('mean:'+str(mom)+str(' for young'))
        sp500 = np.array(sp500MR.loc[sub_pd.index]).flatten()
        
        for lead in range(lead_loop):    
            corr = st.pearsonr(sp500[lead+1:],
                               np.array(sub_pd[mom,'young'])[:-(lead+1)]
                              )
            corr_str = corrtostr(corr)
            corr_list.append(corr_str)
            
        # group 2 
        col_list.append('mean:'+str(mom)+str(' for middle-age'))
        sp500 = np.array(sp500MR.loc[sub_pd.index]).flatten()
        
        for lead in range(lead_loop):    
            corr = st.pearsonr(sp500[lead+1:],
                               np.array(sub_pd[mom,'middle-age'])[:-(lead+1)]
                              )
            corr_str = corrtostr(corr)
            corr_list.append(corr_str)
            
            
        # group 3
        col_list.append('mean:'+str(mom)+str(' for old'))
        sp500 = np.array(sp500MR.loc[sub_pd.index]).flatten()
        
        for lead in range(lead_loop):    
            corr = st.pearsonr(sp500[lead+1:],
                               np.array(sub_pd[mom,'old'])[:-(lead+1)]
                              )
            corr_str = corrtostr(corr)
            corr_list.append(corr_str)
            
            

    corr_array = np.array(corr_list).reshape([int(len(corr_list)/lead_loop),
                                              lead_loop])
    corr_df = pd.DataFrame(corr_array,
                           index = col_list)
    
## save 
corr_df.T.to_excel('../Tables/macro_corr_age_gr.xlsx')

# + {"code_folding": [12, 23]}
## correlation coefficients by sub group HHinc

### subgroup population summary statistics

gr_vars = ['HHinc_gr']
lead_loop = 12

moms = ['incvar','rincvar','incskew']


## mean est 

for gr in gr_vars:
    
    corr_list = []
    col_list = []
    
    sub_pd = pd.pivot_table(data = IncSCEIndMoms,
                            index=['date',gr],
                            values = moms,
                            aggfunc= 'mean').unstack()  ## index being time only 
    sub_pd = sub_pd.dropna(how='any')
    ## mean moments 
    for mom in moms:
        # group 1 
        col_list.append('mean:'+str(mom)+str(' for low'))
        sp500 = np.array(sp500MR.loc[sub_pd.index]).flatten()
        
        for lead in range(lead_loop):    
            corr = st.pearsonr(sp500[lead+1:],
                               np.array(sub_pd[mom,'low'])[:-(lead+1)]
                              )
            corr_str = corrtostr(corr)
            corr_list.append(corr_str)
            
        # group 2 
        col_list.append('mean:'+str(mom)+str(' for high'))
        sp500 = np.array(sp500MR.loc[sub_pd.index]).flatten()
        
        for lead in range(lead_loop):    
            corr = st.pearsonr(sp500[lead+1:],
                               np.array(sub_pd[mom,'high'])[:-(lead+1)]
                              )
            corr_str = corrtostr(corr)
            corr_list.append(corr_str)
     

    corr_array = np.array(corr_list).reshape([int(len(corr_list)/lead_loop),
                                              lead_loop])
    corr_df = pd.DataFrame(corr_array,
                           index = col_list)
    
## save 
corr_df.T.to_excel('../Tables/macro_corr_HHinc_gr.xlsx')

# + {"code_folding": []}
## correlation coefficients by sub group educ 

### subgroup population summary statistics

gr_vars = ['educ_gr']
lead_loop = 12

moms = ['incvar','rincvar','incskew']


## mean est 

for gr in gr_vars:
    
    corr_list = []
    col_list = []
    
    sub_pd = pd.pivot_table(data = IncSCEIndMoms,
                            index=['date',gr],
                            values = moms,
                            aggfunc= 'mean').unstack()  ## index being time only 
    sub_pd = sub_pd.dropna(how='any')
    ## mean moments 
    for mom in moms:
        # group 1 
        col_list.append('mean:'+str(mom)+str(' for low'))
        sp500 = np.array(sp500MR.loc[sub_pd.index]).flatten()
        
        for lead in range(lead_loop):    
            corr = st.pearsonr(sp500[lead+1:],
                               np.array(sub_pd[mom,'low'])[:-(lead+1)]
                              )
            corr_str = corrtostr(corr)
            corr_list.append(corr_str)
            
        # group 2 
        col_list.append('mean:'+str(mom)+str(' for high'))
        sp500 = np.array(sp500MR.loc[sub_pd.index]).flatten()
        
        for lead in range(lead_loop):    
            corr = st.pearsonr(sp500[lead+1:],
                               np.array(sub_pd[mom,'high'])[:-(lead+1)]
                              )
            corr_str = corrtostr(corr)
            corr_list.append(corr_str)
            
            

    corr_array = np.array(corr_list).reshape([int(len(corr_list)/lead_loop),
                                              lead_loop])
    corr_df = pd.DataFrame(corr_array,
                           index = col_list)
    
## save 
corr_df.T.to_excel('../Tables/macro_corr_educ_gr.xlsx')
# -

corr_df

# + {"code_folding": []}
## plot sp500 return forward months later and realized income 

forward = 12

# + {"code_folding": []}
## plots of correlation for MEDIAN population stats

figsize = (80,40)
lw = 20
fontsize = 80

for i,moms in enumerate( ['exp','var','iqr','rexp','rvar']):
    fig, ax = plt.subplots(figsize = figsize)
    ax2 = ax.twinx()
    ax.bar(dt_combM.index[:-forward], 
           dt_combM['sp500'][forward:],
           color = 'gray', 
           width = 25,
           label = 'sp500 YoY '+str(forward)+'m later')
    ax2.plot(dt_combM.index,
             dt_combM[str(moms)+'Med'],
             'r--',
             lw = lw,
             label=str(mom_dict[moms])+' (RHS)')
    #ax.set_xticklabels(dt_combM.index)
    ax.legend(loc = 2,
              fontsize = fontsize)
    ax.set_xlabel("month",fontsize = fontsize)
    ax.set_ylabel('% return',fontsize = fontsize)
    ax.tick_params(axis='both', 
                   which='major', 
                   labelsize = fontsize)
    ax2.tick_params(axis='both', 
                   which='major', 
                   labelsize = fontsize)
    ax2.legend(loc = 1,
               fontsize = fontsize)
    ax2.set_ylabel(moms,fontsize = fontsize)
    plt.savefig('../Graphs/pop/tsMed'+str(moms)+'.jpg')
    #cor,pval =st.pearsonr(np.array(dt_combM['sp500']),
    #                      np.array(dt_combM[str(moms)+'Med']))
    #print('Correlation coefficient is '+str(round(cor,3)) + ', p-value is '+ str(round(pval,3)))



# + {"code_folding": [0]}
## plots of correlation for Mean population stats

for i,moms in enumerate( ['exp','var','iqr','rexp','rvar']):
    fig, ax = plt.subplots(figsize = figsize)
    ax2 = ax.twinx()
    ax.bar(dt_combM.index[:-forward],
           dt_combM['sp500'][forward:],
           color='gray',
           width= 25,
           label= 'sp500 YoY '+str(forward)+'m later')
    ax2.plot(dt_combM[str(moms)+'Mean'],
             'r--',
             lw = lw,
             label = str(mom_dict[moms])+' (RHS)')
    #ax.set_xticklabels(dt_combM.index)
    ax.legend(loc = 2,
             fontsize = fontsize)
    ax.set_xlabel("month",fontsize = fontsize)
    ax.set_ylabel('% return',fontsize = fontsize)
    ax.tick_params(axis='both', 
                   which='major', 
                   labelsize = fontsize)
    ax2.tick_params(axis='both', 
                   which='major', 
                   labelsize = fontsize)
    ax2.legend(loc = 1,
              fontsize = fontsize)
    plt.savefig('../Graphs/pop/tsMean'+str(moms)+'.jpg')

    #cor,pval = st.pearsonr(np.array(dt_combM['sp500']),
    #                      np.array(dt_combM[str(moms)+'Mean']))
    #print('Correlation coefficient is '+str(round(cor,3)) + ', p-value is '+ str(round(pval,3)))
# -

crr3mv_table = dt_combM3mv.corr()
crr3mv_table.to_excel('../Tables/corr3mvM.xlsx')
crr3mv_table

# + {"code_folding": [0]}
## plots of correlation for 3-month moving MEDIAN average

for i,moms in enumerate( ['exp','var','iqr','rexp','rvar']):
    fig, ax = plt.subplots(figsize = figsize)
    ax2 = ax.twinx()
    ax.bar(dt_combM3mv.index[:-forward],
           dt_combM3mv['sp500'][forward:],
           color='gray',
           width=25,
           label = 'sp500 YoY '+str(forward)+'m later')
    ax2.plot(dt_combM3mv[str(moms)+'Med'],
             'r--',
             lw = lw,
             label = str(mom_dict[moms])+' (RHS)')
    #ax.set_xticklabels(dt_combM.index)
    ax.legend(loc=0,
              fontsize = fontsize)
    ax.set_xlabel("month",fontsize = fontsize)
    ax.set_ylabel('% return',fontsize = fontsize)
    ax.tick_params(axis='both', 
                   which='major', 
                   labelsize = fontsize)
    ax2.tick_params(axis='both', 
                   which='major', 
                   labelsize = fontsize)
    ax2.legend(loc = 2,
              fontsize = fontsize)
    plt.savefig('../Graphs/pop/tsMed3mv'+str(moms)+'.jpg')
    #cor,pval = st.pearsonr(np.array(dt_combM3mv['sp500']),
    #                      np.array(dt_combM3mv[str(moms)+'Med']))
    #print('Correlation coefficient is '+str(cor) + ', p-value is '+ str(pval))


# + {"code_folding": []}
## plots of correlation for 3-month moving mean average

for i,moms in enumerate( ['exp','var','iqr','rexp','rvar']):
    fig, ax = plt.subplots(figsize = figsize)
    ax2 = ax.twinx()
    ax.bar(dt_combM3mv.index[:-forward],
           dt_combM3mv['sp500'][forward:],
           color='gray',
           width=25,
           label='sp500 YoY '+str(forward)+'m later')
    ax2.plot(dt_combM3mv[str(moms)+'Mean'],
             'r--',
             lw = lw,
             label=str(mom_dict[moms])+' (RHS)')
    #ax.set_xticklabels(dt_combM.index)
    ax.legend(loc=0,
             fontsize = fontsize)
    ax.set_xlabel("month",fontsize = fontsize)
    ax.set_ylabel('% return',fontsize = fontsize)
    ax2.legend(loc=2,
             fontsize = fontsize)
    ax.tick_params(axis='both', 
                   which='major', 
                   labelsize = fontsize)
    ax2.tick_params(axis='both',
                    which='major',
                    labelsize = fontsize)
    plt.savefig('../Graphs/pop/tsMean3mv'+str(moms)+'.jpg')
    #cor,pval =st.pearsonr(np.array(dt_combM['sp500']),
    #                      np.array(dt_combM3mv[str(moms)+'Mean']))
    #print('Correlation coefficient is '+str(cor) + ', p-value is '+ str(pval))


# -

# ### 7b. Individual regressions 

# + {"code_folding": []}
for i,moms in enumerate( ['incexp','incvar','inciqr','rincvar','incskew']):
    print(moms)
    Y = np.array(dt_combIndM[moms])[forward:]
    X = np.array(dt_combIndM['sp500'])[:-forward]
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    rs = model.fit()
    print(rs.summary())

# -


