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

# # Perceived Income Risks and Macroeconomic Conditions
#
#
# - This notebook first downloads macroeconomic series 
# - Then we examine the correlation of higher moments of labor income risks and these macro series 

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
## some configurations on plotting 
plt.style.use('ggplot')
plt.rcParams.update({'figure.max_open_warning': 0})
#from IPython.display import set_matplotlib_formats
plt.rcParams["font.family"] = "Times New Roman" #'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 15
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.labelweight'] = 'bold'

plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 15
plt.rcParams['figure.titlesize'] = 15

#set_matplotlib_formats('pdf','png','jpg')
#plt.rcParams['savefig.dpi'] = 75
#plt.rcParams['figure.autolayout'] = False
#plt.rcParams['figure.figsize'] = 10, 6
#plt.rcParams['axes.labelsize'] = 18
#plt.rcParams['axes.titlesize'] = 20
#plt.rcParams['lines.linewidth'] = 2.0
#plt.rcParams['lines.markersize'] = 8
#plt.rcParams['legend.fontsize'] = 14

#plt.rcParams['text.usetex'] = True

pd.options.display.float_format = '{:,.2f}'.format
# -

# ###  1. Download stock return/wage rate/unemployment rate

# + {"code_folding": []}
## time span 

start = datetime.datetime(2000, 1, 30)
end = datetime.datetime(2020, 3, 30)

# + {"code_folding": []}
## downloading the data from Fred

sp500D= web.DataReader('sp500', 'fred', start, end)
vixD = web.DataReader('VIXCLS','fred',start,end)
he = web.DataReader('CES0500000003','fred',start,end) #hourly earning private
ue = web.DataReader('UNRATE','fred',start,end)
cpi = web.DataReader('CPIAUCSL','fred',start,end)

## rename 

vixD = vixD.rename(columns={'VIXCLS':'vix'})
he = he.rename(columns={'CES0500000003':'he'})
ue = ue.rename(columns={'UNRATE':'ue'})
cpi = cpi.rename(columns={'CPIAUCSL':'cpi'})
# -

## ue rate 
ue.plot(lw = 2)
ueplt = plt.title('unemployment rate')

# +
## collapse to monthly data
sp500D.index = pd.to_datetime(sp500D.index)
sp500M = sp500D.resample('M').last()

vixD.index = pd.to_datetime(vixD.index)
vixM = vixD.resample('M').mean()

sp500M.plot(lw = 3)
#sp500Mplt = plt.title('S&P 500 (end of month)')

# +
## compute change/growths 
sp500MR = np.log(sp500M).diff(periods = 3)

## quarterly wage growth  
he = np.log(he).diff(periods = 3)
he.columns = ['he']
 
# -

he.plot(lw = 2)

# adjusting the end-of-month dates to the begining-of-month for combining
sp500MR.index = sp500MR.index.shift(1,freq='D')
vixM.index = vixM.index.shift(1,freq='D')

# +
## merge all monthly variables 

macroM = pd.concat([sp500MR,
                    vixM,
                    he,
                   ue,
                   cpi],
                   join="inner",
                   axis=1)
# -

macroM

## save macroM for further analysis 
macroM.to_stata('../OtherData/macroM_raw.dta')

# ###  2. Loading and cleaning perceived income series

# + {"code_folding": []}
## loading the stata file
SCEProbIndM = pd.read_stata('../SurveyData/SCE/IncExpSCEProbIndM.dta')
SCEDstIndM = pd.read_stata('../SurveyData/SCE/IncExpSCEDstIndM.dta')

# +
# merge the two 

SCEIndM = pd.merge(SCEProbIndM,
                   SCEDstIndM,
                   on = ['userid','date'],
                  how='inner')

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

# + {"code_folding": []}
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

# ### 3. Combinine the macro series and SCE data 

# + {"code_folding": []}
## streamline the dates for merging

# adjusting the end-of-month dates to the begining-of-month for combining
#sp500MR.index = sp500MR.index.shift(1,freq='D')
#vixM.index = vixM.index.shift(1,freq='D')


IncSCEPopMomsMed.index = pd.DatetimeIndex(IncSCEPopMomsMed['date'] ,freq='infer')
IncSCEPopMomsMean.index = pd.DatetimeIndex(IncSCEPopMomsMean['date'] ,freq='infer')


# + {"code_folding": []}
dt_combM = pd.concat([macroM,
                      IncSCEPopMomsMed,
                      IncSCEPopMomsMean],
                     join="inner",
                     axis=1).drop(columns=['date'])
# -

dt_combM.tail()

## export to stata
dt_combM.to_stata('../SurveyData/SCE/IncExpSCEPopMacroM.dta')

# + {"code_folding": []}
## date index for panel

IncSCEIndMoms.index = IncSCEIndMoms['date']
IncSCEIndMoms.index.name = None

# +
## merge individual moments and macro series

dt_combIndM = pd.merge(macroM,
                       IncSCEIndMoms,
                       left_index = True,
                       right_on ='date')
# -

dt_combIndM

## export to stata
dt_combIndM.to_stata('../SurveyData/SCE/IncExpSCEIndMacroM.dta')

# ### 4. Correlation with labor market outcomes 

corr_table = dt_combM.corr()
corr_table.to_excel('../Tables/corrM.xlsx')
corr_table

# + {"code_folding": [2, 13, 16, 31]}
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
    ax.set_ylabel('growth',fontsize = fontsize)
    ax.tick_params(axis='both', 
                   which='major', 
                   labelsize = fontsize)
    ax2.tick_params(axis='both', 
                   which='major', 
                   labelsize = fontsize)
    ax2.legend(loc = 1,
              fontsize = fontsize)
    #plt.savefig('../Graphs/pop/tsMean'+str(moms)+'_he.jpg')

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
    ax.set_ylabel('log growth',fontsize = fontsize)
    ax2.legend(loc = 2,
             fontsize = fontsize)
    ax.tick_params(axis='both', 
                   which='major', 
                   labelsize = fontsize)
    ax2.tick_params(axis='both',
                    which='major',
                    labelsize = fontsize)
    plt.savefig('../Graphs/pop/tsMean3mv'+str(moms)+'_he.jpg')
